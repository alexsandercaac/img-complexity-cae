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
from utils.models.kerascallbacks import CustomLearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

stage_params = get_params()
all_params = get_params('all')

params = {**stage_params, **all_params}

DATASET_PATH = 'data/raw/tiny-imagenet-200'

# * Augmentation

augmentation = augmentation_model(
    random_crop=tuple(params['random_crop']),
    random_flip=params['random_flip'],
    random_rotation=params['random_rotation'],
    random_zoom=tuple(params['random_zoom']),
    random_brightness=params['random_brightness'],
    random_contrast=params['random_contrast'],
    random_translation_height=tuple(params['random_translation_height']),
    random_translation_width=tuple(params['random_translation_width'])
)

# * Load dataset
train_dataset = load_tf_img_dataset(
    dir='train',
    dir_path=DATASET_PATH,
    input_size=tuple(params['input_size'])[:2],
    mode='autoencoder',
    scale=255,
    shuffle=True,
    augmentation=augmentation,
    batch_size=params['batch_size'],
    color_mode='grayscale'
)

val_dataset = load_tf_img_dataset(
    dir='val',
    dir_path=DATASET_PATH,
    input_size=tuple(params['input_size'])[:2],
    mode='autoencoder',
    scale=255,
    shuffle=True,
    batch_size=params['batch_size'],
    color_mode='grayscale'
)

model = tf.keras.models.load_model(
    filepath='models/casting/bin/hp_search_best.hdf5'
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
