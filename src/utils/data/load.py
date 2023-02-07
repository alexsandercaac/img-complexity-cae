"""
    Set of functions used to load image datasets.
"""
import tensorflow as tf
import os
from typing import Tuple


def load_tf_img_dataset(dir: str, dir_path: str = '', batch_size: int = 1,
                        input_size: Tuple[int, int] = (224, 224),
                        mode: str = 'autoencoder', labels: list = None,
                        augmentation:  tf.keras.Sequential = None,
                        shuffle: bool = True, scale: float = None
                        ) -> tf.data.Dataset:
    '''

    Loads image dataset using tensorflow and returns a tf.data.Dataset object.

    Args:
        dir (str): Directory containing the images.
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

    Returns:
        tf.data.Dataset: Dataset object containing the images.
    '''
    AUTOTUNE = tf.data.AUTOTUNE
    if mode == 'autoencoder' or mode == 'image_only':
        dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dir_path, dir),
            labels=None,
            batch_size=batch_size,
            image_size=input_size,
            shuffle=shuffle)
    elif mode == 'classifier':
        dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dir_path, dir),
            labels=labels if labels else 'inferred',
            batch_size=batch_size,
            image_size=input_size,
            shuffle=shuffle)

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
            dataset = dataset.map(lambda x: rescale(x),
                                  num_parallel_calls=AUTOTUNE)
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
