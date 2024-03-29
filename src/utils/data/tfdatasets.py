"""
    Set of functions that use tensorflow to load and manipulate image datasets.
"""
import os
from typing import Tuple

import tensorflow as tf


def load_tf_img_dataset(dir_name: str, dir_path: str = '', batch_size: int = 1,
                        input_size: Tuple[int, int] = (224, 224),
                        mode: str = 'autoencoder', labels: list = None,
                        augmentation: tf.keras.Sequential = None,
                        shuffle: bool = True, scale: float = None,
                        color_mode: str = 'rgb'
                        ) -> tf.data.Dataset:
    '''

    Loads image dataset using tensorflow and returns a tf.data.Dataset object.

    Args:
        dir_name (str): Directory containing the images.
        dir_path (str): Path to the directory containing the images.
            Defaults to ''.
        batch_size (int): Batch size set on the tf.data.Dataset object.
            Defaults to 1.
        mode (str): Type of dataset to load. Can be 'autoencoder', 'classifier'
             or 'image_only'. If 'autoencoder', returns a batch of tuples of
            nature (input, input); if 'classifier', returns tuples
             of (input, label); if 'image_only', returns only a batch of
             inputs. Defaults to 'autoencoder'.
        labels (list): List of labels to use for the dataset, to be used when
        mode is 'classifier'. Defaults to None.
        input_size ((int, int)): Size of the input images (height, width).
            Defaults to (224, 224).
        augmentation (bool): Optional augmentation function to apply to the
            images. Defaults to None.
        shuffle (bool): Whether to shuffle the dataset or not.
            Defaults to True.
        scale (float): Factor used to scale the images. If None, no scaling is
            applied. Defaults to None.
        color_mode (str): Color mode to use. Can be 'rgb' or 'grayscale'.
            Defaults to 'rgb'.

    Returns:
        tf.data.Dataset: Dataset object containing the images.
    '''
    AUTOTUNE = tf.data.AUTOTUNE
    if mode in ('autoencoder', 'image_only'):
        dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dir_path, dir_name),
            labels=None,
            batch_size=batch_size,
            image_size=input_size,
            shuffle=shuffle,
            color_mode=color_mode)
    elif mode == 'classifier':
        dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dir_path, dir_name),
            labels=labels if labels else 'inferred',
            batch_size=batch_size,
            image_size=input_size,
            shuffle=shuffle,
            color_mode=color_mode)

    if augmentation:
        dataset = dataset.map(lambda x: augmentation(x, training=True),
                              num_parallel_calls=AUTOTUNE)

    if scale:
        rescale = tf.keras.Sequential(
            [tf.keras.layers.Rescaling(1 / scale)]
        )
        if mode == 'autoencoder':
            dataset = dataset.map(lambda x: (
                rescale(x), rescale(x)),
                num_parallel_calls=AUTOTUNE)
        elif mode == 'classifier':
            dataset = dataset.map(lambda x, y: (
                rescale(x), y),
                num_parallel_calls=AUTOTUNE)
        elif mode == 'image_only':
            dataset = dataset.map(rescale, num_parallel_calls=AUTOTUNE)
    else:
        if mode == 'autoencoder':
            dataset = dataset.map(lambda x: (x, x),
                                  num_parallel_calls=AUTOTUNE)
        elif mode == 'classifier':
            dataset = dataset.map(lambda x, y: (x, y),
                                  num_parallel_calls=AUTOTUNE)
        elif mode == 'image_only':
            dataset = dataset.map(lambda x: x,
                                  num_parallel_calls=AUTOTUNE)

    return dataset


def augmentation_model(
        random_crop: Tuple[int, int] = (None, None),
        random_flip: str = None, random_rotation: float = None,
        random_zoom: Tuple[float, float] = (None, None),
        random_brightness: float = None,
        random_contrast: float = None,
        random_translation_height: Tuple[float, float] = (None, None),
        random_translation_width: Tuple[float, float] = (None, None),
) -> tf.keras.Sequential:
    '''
    Creates a tf.keras.Sequential object containing the augmentation
    operations specified by the user.

    Args:
        random_crop (Tuple[int, int]): Size of the random crop to apply.
            Defaults to None.
        random_flip (str): . Can be "horizontal", "vertical",
            or "horizontal_and_vertical". Defaults to None.
        random_rotation (float): Maximum rotation angle to apply.
            Defaults to None.
        random_zoom (Tuple[float, float]): Range of the random zoom to apply.
            Defaults to None.
        random_brightness (float): Maximum brightness to apply.
            Defaults to None.
        random_contrast (float): Maximum contrast to apply.
            Defaults to None.
        random_translation_height (Tuple[float, float]): Range of the random
            translation to apply on the height. Defaults to None.
        random_translation_width (Tuple[float, float]): Range of the random
            translation to apply on the width. Defaults to None.

    Returns:
        tf.keras.Sequential: Sequential object containing the augmentation
            operations.

    '''
    layers = []
    if random_crop[0] and random_crop[1]:
        layers.append(tf.keras.layers.RandomCrop(random_crop[0],
                                                 random_crop[1]))
    if random_flip:
        layers.append(tf.keras.layers.RandomFlip(random_flip))
    if random_rotation:
        layers.append(tf.keras.layers.RandomRotation(random_rotation))
    if random_zoom[0] and random_zoom[1]:
        layers.append(tf.keras.layers.RandomZoom(random_zoom))
    if random_brightness:
        layers.append(tf.keras.layers.RandomBrightness(random_brightness))
    if random_contrast:
        layers.append(tf.keras.layers.RandomContrast(random_contrast))
    if ((random_translation_height[0] and random_translation_height[1]) and
            random_translation_width[0] and random_translation_width[1]):
        layers.append(tf.keras.layers.RandomTranslation(
            random_translation_height, random_translation_width))
    if not layers:
        raise ValueError('No augmentation operation specified.')

    return tf.keras.Sequential(layers)
