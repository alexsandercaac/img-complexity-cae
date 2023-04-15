"""
    This stage loads the model trained in the previous stage of hyperparameter
    tuning and trains it on a general purpose dataset. The goal is that
    the model will learn generalizable low level features that can be used
    for the task of interest.
"""
import os
import logging

import tensorflow as tf
from tqdm.keras import TqdmCallback
import pandas as pd

from utils.data.tfdatasets import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params
from utils.models.kerasaux import CustomLearningRateScheduler, \
    reset_model_weights

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters


params = get_params()

# Data parameters
DATASET = params['dataset']
INPUT_SIZE = tuple(params['input_size'])
SCALE = params['scale']
GRAYSCALE = params['grayscale']

# Model parameters
BATCH_SIZE = params['batch_size']
BOTTLENECK_FILTERS = params['bottleneck_filters']
ALPHA = params['alpha']
BETA = params['beta']
PATIENCE = params['patience']
EARLY_STOPPING = params['early_stopping']
REVIVE_BEST = params['revive_best']
MIN_LR = params['min_lr']
MAX_TRIALS = params['max_trials']
SEED = params['seed']
EPOCHS = params['epochs']

# Augmentation parameters
RANDOM_CROP = tuple(params['random_crop'])
RANDOM_FLIP = params['random_flip']
RANDOM_ROTATION = params['random_rotation']
RANDOM_ZOOM = tuple(params['random_zoom'])
RANDOM_BRIGHTNESS = params['random_brightness']
RANDOM_CONTRAST = params['random_contrast']
RANDOM_TRANSLATION_HEIGHT = tuple(params['random_translation_height'])
RANDOM_TRANSLATION_WIDTH = tuple(params['random_translation_width'])

# * Dataset loading
augmentation = augmentation_model(
    random_crop=RANDOM_CROP,
    random_flip=RANDOM_FLIP,
    random_rotation=RANDOM_ROTATION,
    random_zoom=RANDOM_ZOOM,
    random_brightness=RANDOM_BRIGHTNESS,
    random_contrast=RANDOM_CONTRAST,
    random_translation_height=RANDOM_TRANSLATION_HEIGHT,
    random_translation_width=RANDOM_TRANSLATION_WIDTH
)

data_dir = os.path.join('data', 'raw', 'tiny-imagenet-200')

train_dataset = load_tf_img_dataset(
    dir='train',
    dir_path=data_dir,
    input_size=INPUT_SIZE[:2],
    mode='autoencoder',
    scale=SCALE,
    shuffle=True,
    augmentation=augmentation,
    batch_size=BATCH_SIZE,
    color_mode='grayscale' if GRAYSCALE else 'rgb'
)

val_dataset = load_tf_img_dataset(
    dir='val',
    dir_path=data_dir,
    input_size=RANDOM_CROP,
    mode='autoencoder',
    scale=SCALE,
    shuffle=False,
    batch_size=BATCH_SIZE,
    color_mode='grayscale' if GRAYSCALE else 'rgb'
)

model_dir = os.path.join('models', DATASET, 'bin')
model = tf.keras.models.load_model(
    filepath=os.path.join(model_dir, 'hp_search_best.hdf5')
)

# Randomize model weights
model.set_weights(
    [tf.random.normal(shape=weight.shape) for weight in model.weights]
)

model.compile(loss=['mse'],
              optimizer=tf.keras.optimizers.Adam(
    learning_rate=params['learning_rate']),
    metrics=['mae', 'mse']
)
lr_schedule = CustomLearningRateScheduler(
    metric='val_mse',
    secondary_metric='val_mae',
    alpha=params['alpha'],
    beta=params['beta'],
    verbose=2,
    patience=params['patience'],
    early_stopping=params['early_stopping'],
    revive_best=params['revive_best'],
    min_lr=params['min_lr']
)

train_folder = os.path.join(DATASET_PATH, 'train')

train_size = sum(
    [len(files) for _, _, files in os.walk(train_folder)])

print(f'Training samples: {train_size}')

# * Train model

history = model.fit(
    train_dataset,
    epochs=params['epochs'],
    validation_data=val_dataset,
    callbacks=[lr_schedule,
               TqdmCallback(verbose=2,
                            data_size=train_size,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'])],
    verbose=0
)

# * Save model and history

print('Saving model...')
model.save(filepath='models/casting/bin/pretrained_cae.hdf5')
history_df = pd.DataFrame(history.history)
history_df.to_csv(
    'models/casting/logs/pretraining_history.csv', index=False)
