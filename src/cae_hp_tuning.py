"""
    DVC stage for hyperparameter optimzation of the CAE model using the
    Keras Tuner library.
"""
import os
import sys
import io
import logging

import keras_tuner as kt
import tensorflow as tf

from utils.data.tfdatasets import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params
from utils.models.ktmodels import CAE
from utils.models.kerascallbacks import CustomLearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

stage_params = get_params()
all_params = get_params('all')

params = {**stage_params, **all_params}

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
    dir='train/ok_front',
    dir_path='data/processed',
    input_size=tuple(params['input_size'])[:2],
    mode='autoencoder',
    scale=255,
    shuffle=True,
    augmentation=augmentation,
    batch_size=params['batch_size'],
    color_mode='grayscale'
)

val_dataset = load_tf_img_dataset(
    dir='val/ok_front',
    dir_path='data/processed',
    input_size=tuple(params['input_size'])[:2],
    mode='autoencoder',
    scale=255,
    shuffle=True,
    batch_size=params['batch_size'],
    color_mode='grayscale'
)

search_model = CAE(tuple(params['input_size']), params['bottleneck_filters'])

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
tbcallback = tf.keras.callbacks.TensorBoard(params['log_path'] + '/tb')

strategy = tf.distribute.MirroredStrategy()
tuner = kt.BayesianOptimization(
    search_model, 'val_mse', max_trials=params['max_trials'], overwrite=False,
    project_name='CAE', directory=params['log_path'], seed=params['seed'],
    distribution_strategy=strategy
)
tuner.search_space_summary()

history = tuner.search(train_dataset,
                       validation_data=val_dataset,
                       callbacks=[lr_schedule,
                                  tbcallback],
                       epochs=params['epochs'],
                       shuffle=True,
                       batch_size=params['batch_size'],
                       verbose=1
                       )

old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout
tuner.results_summary(1)
results = new_stdout.getvalue()
with open(params['log_path'] + '/hp_search_results.txt', 'w') as file:
    file.write(results)
sys.stdout = old_stdout

best_model = tuner.get_best_models()[0]
best_model.save('models/casting/bin/hp_search_best.hdf5')
