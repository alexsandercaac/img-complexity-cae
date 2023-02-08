"""
    This script will generate exaples of augmented images with parameters
    specified by the user. The images will be saved in the folder
    data/interim/augmentation_visualisation.
"""
from utils.data.tensorflow_based import load_tf_img_dataset, augmentation_model
from utils.dvc.params import get_params
import matplotlib.pyplot as plt
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

params = get_params()

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

plt.savefig('data/interim/augmentation_visualisation/augmented_images.png')
