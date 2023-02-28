"""
    This script will generate exaples of augmented images with parameters
    specified by the user. The images will be saved in the folder
    data/interim/augmentation_visualisation.
"""
import matplotlib.pyplot as plt
import os
import logging
import tensorflow as tf

from utils.data.tfdatasets import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

params = get_params('all')

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

dataset = load_tf_img_dataset(
    dir='raw',
    dir_path='data',
    mode='image_only',
    scale=255,
    shuffle=False,
    augmentation=augmentation,
    batch_size=4
)

plt.figure(figsize=(10, 10))
for images in dataset:
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(images[i].numpy())
        plt.axis("off")
    break

plt.savefig('visualisation/augmentation/augmented_images.png')

# Get reconstruction of images and save them beside the original

dataset = load_tf_img_dataset(
    dir='val',
    dir_path='data/processed',
    input_size=tuple(params['input_size'])[:2],
    mode='autoencoder',
    scale=255,
    shuffle=True,
    augmentation=augmentation,
    batch_size=2,
    color_mode='grayscale'
)

model = tf.keras.models.load_model('models/best_cae.hdf5')

plt.figure(figsize=(10, 10))
for images, _ in dataset:
    for i in range(2):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(model(images)[i].numpy(), cmap='gray')
        plt.title('Reconstructed')
        plt.axis("off")
        ax = plt.subplot(2, 2, i + 3)
        plt.imshow(images[i].numpy(), cmap='gray')
        plt.title('Original')
        plt.axis("off")
    break

plt.savefig('visualisation/reconstruction/reconstructed_images.png')
