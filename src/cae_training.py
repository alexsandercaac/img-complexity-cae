"""
    This stage loads the model pretrained in the previous stage and fits it to
    the training dataset. The model is trained for a number of epochs and the
    best performing model in validation is saved. The training history is
    saved as a csv file and the model is saved as an hdf5 file.
"""
import os
import logging

import tensorflow as tf
from tqdm.keras import TqdmCallback
import pandas as pd

from utils.data.tfdatasets import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params
from utils.models.kerasaux import CustomLearningRateScheduler

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

model = tf.keras.models.load_model(
    filepath='models/casting/bin/pretrained_cae.hdf5'
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

train_size = sum(
    [len(files) for _, _, files in os.walk('data/processed/train')])

print(f'Training samples: {train_size}')

# Evaluate model on validation dataset before training
print('Evaluating model on validation dataset before training...')
val_loss_pre, val_mae_pre, val_mse_pre = model.evaluate(val_dataset, verbose=2)

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

# Evaluate model on validation dataset after training
print('Evaluating model on validation dataset after training...')
val_loss_post, val_mae_post, val_mse_post = model.evaluate(
    val_dataset, verbose=2)

# * Save model and history

if val_mse_post > val_mse_pre:
    print('Validation loss increased after training.')
    print('Reverting to best model...')
    model = tf.keras.models.load_model(
        filepath='models/casting/bin/best_cae.hdf5')
    model.save(filepath='models/casting/bin/best_cae.hdf5')
else:
    print('Validation loss decreased after training.')
    print('Saving model...')
    model.save(filepath='models/casting/bin/best_cae.hdf5')
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/casting/logs/training_history.csv', index=False)
