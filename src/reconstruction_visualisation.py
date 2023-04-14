"""
    This script will generate exaples of augmented images with parameters
    specified by the user. The images will be saved in the folder
    data/interim/augmentation_visualisation.
"""
import matplotlib.pyplot as plt
import os
import logging
import tensorflow as tf

from utils.data.tfdatasets import load_tf_img_dataset
from utils.dvc.params import get_params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

params = get_params('all')

# * Augmentation

dataset = load_tf_img_dataset(
    dir='val',
    dir_path='data/processed',
    input_size=tuple(params['input_size'])[:2],
    mode='autoencoder',
    scale=255,
    shuffle=True,
    batch_size=2,
    color_mode='grayscale'
)

model = tf.keras.models.load_model('models/casting/bin/best_cae.hdf5')

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

plt.savefig('visualisation/reconstruction/casting/reconstructed_images.png')
