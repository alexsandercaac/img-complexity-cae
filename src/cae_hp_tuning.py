"""
    DVC stage for hyperparameter optimzation of the CAE model using the
    Keras Tuner library.

    The script will save the best model found in the
    models/$DATASET/bin folder and a description of the top models in the
    models/$DATASET/logs/hp_search folder.

    The tensorboard logs will be saved in the models/$DATASET/logs/hp_search
    folder.
"""
import os
import logging

import keras_tuner as kt
import tensorflow as tf

from utils.data.tfdatasets import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params
from utils.models.ktmodels import CAE
from utils.models.kerasaux import CustomLearningRateScheduler
from utils.misc import catch_stdout, create_dir

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

params = get_params()

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
MODEL_INPUT_DIM = ((RANDOM_CROP[0], RANDOM_CROP[1], 1) if GRAYSCALE
                   else (RANDOM_CROP[0], RANDOM_CROP[1], 3))

# * Directories
DATA_DIR = os.path.join('data', 'processed', DATASET)
LOG_PATH = os.path.join('models', DATASET, 'logs', 'hp_search')
create_dir(LOG_PATH)
MODELS_BIN_DIR = os.path.join('models', DATASET, 'bin')
create_dir(MODELS_BIN_DIR)


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


search_model = CAE(MODEL_INPUT_DIM, BOTTLENECK_FILTERS)

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

tbcallback = tf.keras.callbacks.TensorBoard(
    os.path.join(LOG_PATH, 'tb')
)

tuner = kt.BayesianOptimization(
    search_model,
    'val_mse',
    max_trials=MAX_TRIALS,
    overwrite=False,
    project_name='CAE',
    directory=LOG_PATH,
    seed=SEED,
    distribution_strategy=tf.distribute.MirroredStrategy()
)
tuner.search_space_summary()

tuner.search(train_dataset,
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

with open(os.path.join(LOG_PATH, 'hp_search_results.txt'), 'w') as file:
    file.write(results)

best_model = tuner.get_best_models()[0]
best_model.save(os.path.join(MODELS_BIN_DIR, 'hp_search_best.hdf5'))
