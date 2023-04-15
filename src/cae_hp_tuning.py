"""
    DVC stage for hyperparameter optimzation of the CAE model using the
    Keras Tuner library.
"""
import os
import logging

import keras_tuner as kt
import tensorflow as tf

from utils.data.tfdatasets import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params
from utils.models.ktmodels import CAE
from utils.models.kerasaux import CustomLearningRateScheduler
from utils.misc import catch_stdout

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

data_dir = os.path.join('data', 'processed', DATASET)

train_dataset = load_tf_img_dataset(
    dir='train/negative',
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
    dir='val/negative',
    dir_path=data_dir,
    input_size=RANDOM_CROP,
    mode='autoencoder',
    scale=SCALE,
    shuffle=False,
    batch_size=BATCH_SIZE,
    color_mode='grayscale' if GRAYSCALE else 'rgb'
)

# * Hyperparameter tuning
model_input_dim = ((RANDOM_CROP[0], RANDOM_CROP[1], 1) if GRAYSCALE
                   else (RANDOM_CROP[0], RANDOM_CROP[1], 3))
search_model = CAE(model_input_dim, BOTTLENECK_FILTERS)

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

log_path = os.path.join('models', DATASET, 'logs', 'hp_search')

tbcallback = tf.keras.callbacks.TensorBoard(
    os.path.join(log_path, 'tb')
)

tuner = kt.BayesianOptimization(
    search_model,
    'val_mse',
    max_trials=MAX_TRIALS,
    overwrite=False,
    project_name='CAE',
    directory=log_path,
    seed=SEED,
    distribution_strategy=tf.distribute.MirroredStrategy()
)
tuner.search_space_summary()

history = tuner.search(train_dataset,
                       validation_data=val_dataset,
                       callbacks=[lr_schedule,
                                  tbcallback],
                       epochs=EPOCHS,
                       shuffle=True,
                       batch_size=BATCH_SIZE,
                       verbose=1
                       )

results_summary = catch_stdout(tuner.results_summary)
results = results_summary()

with open(os.path.join(log_path, 'hp_search_results.txt'), 'w') as file:
    file.write(results)

best_model = tuner.get_best_models()[0]
models_bin_dir = os.path.join('models', DATASET, 'bin')
best_model.save(os.path.join(models_bin_dir, 'hp_search_best.hdf5'))
