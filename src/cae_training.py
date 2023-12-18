"""
    This stage loads the model pretrained in the previous stage and fits it to
    the training dataset. The model is trained for a number of epochs and the
    best performing model in validation is saved.

    The rained model binary will be saved in the models/$DATASET/bin folder and
    the training logs will be saved in the models/$DATASET/logs folder.
"""
import os
import logging
os.chdir('..')
import tensorflow as tf
from tqdm.keras import TqdmCallback
import pandas as pd

from utils.data.tfdatasets import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params
from utils.models.kerasaux import CustomLearningRateScheduler

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

params = get_params('cae_training')

# Data parameters
DATASET = params['dataset']
INPUT_SIZE = tuple(params['input_size'])
SCALE = params['scale']
GRAYSCALE = params['grayscale']

# Augmentation parameters
RANDOM_CROP = tuple(params['random_crop'])
RANDOM_FLIP = params['random_flip']
RANDOM_ROTATION = params['random_rotation']
RANDOM_ZOOM = tuple(params['random_zoom'])
RANDOM_BRIGHTNESS = params['random_brightness']
RANDOM_CONTRAST = params['random_contrast']
RANDOM_TRANSLATION_HEIGHT = tuple(params['random_translation_height'])
RANDOM_TRANSLATION_WIDTH = tuple(params['random_translation_width'])

# Model parameters
BATCH_SIZE = params['batch_size']
LEARNING_RATE = params['learning_rate']
ALPHA = params['alpha']
BETA = params['beta']
PATIENCE = params['patience']
EARLY_STOPPING = params['early_stopping']
REVIVE_BEST = params['revive_best']
MIN_LR = params['min_lr']
SEED = params['seed']
EPOCHS = params['epochs']

# * Directories
DATA_DIR = os.path.join('data', 'processed', DATASET)
MODEL_DIR = os.path.join('models', DATASET, 'bin')


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

train_dataset = load_tf_img_dataset(
    dir_name='train/negative',
    dir_path=DATA_DIR,
    input_size=INPUT_SIZE[:2],
    mode='autoencoder',
    scale=SCALE,
    shuffle=True,
    augmentation=augmentation,
    batch_size=BATCH_SIZE,
    color_mode='grayscale' if GRAYSCALE else 'rgb'
)

val_dataset = load_tf_img_dataset(
    dir_name='val/negative',
    dir_path=DATA_DIR,
    input_size=RANDOM_CROP,
    mode='autoencoder',
    scale=SCALE,
    shuffle=False,
    batch_size=BATCH_SIZE,
    color_mode='grayscale' if GRAYSCALE else 'rgb'
)


model = tf.keras.models.load_model(
    filepath=os.path.join(MODEL_DIR, 'pretrained_cae.hdf5')
)

model.compile(loss=['mse'],
              optimizer=tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE),
    metrics=['mae', 'mse']
)

lr_schedule = CustomLearningRateScheduler(
    metric='val_mse',
    secondary_metric='val_mae',
    alpha=ALPHA,
    beta=BETA,
    verbose=2,
    patience=PATIENCE,
    early_stopping=EARLY_STOPPING,
    revive_best=REVIVE_BEST,
    min_lr=MIN_LR
)

train_size = sum(
    [len(files) for _, _, files in os.walk(os.path.join(
        DATA_DIR, 'train/negative'))])

print(f'Training samples: {train_size}')


print('Evaluating model on validation dataset before training...')
val_loss_pre, val_mae_pre, val_mse_pre = model.evaluate(val_dataset, verbose=2)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[lr_schedule,
               TqdmCallback(verbose=2,
                            data_size=train_size,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS)],
    verbose=0
)

print('Evaluating model on validation dataset after training...')
val_loss_post, val_mae_post, val_mse_post = model.evaluate(
    val_dataset, verbose=2)


print('Saving model...')
model.save(filepath=os.path.join(MODEL_DIR, 'trained_cae.hdf5'))
history_df = pd.DataFrame(history.history)
history_df.to_csv(
    os.path.join('models', DATASET, 'logs', 'training_history.csv'),
    index=False)
