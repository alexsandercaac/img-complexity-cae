"""
    Module with simple functions that are used by all different image
    complexity measures.
"""
import numpy as np
import tensorflow as tf
from typing import Union
from PIL import Image

RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def image_mse(image1: Union[np.ndarray, tf.Tensor],
              image2: Union[np.ndarray, tf.Tensor]) -> float:
    '''
    Calculates the mean squared error between two images.

    Args:
        image1 (ndarray or tf.Tensor): First image.
        image2 (ndarray, tf.Tensor): Second image.

    Returns:
        float: Mean squared error between the two images.
    '''
    return np.mean(tf.keras.metrics.mean_squared_error(image1, image2))


def image_rgb_to_gray_to_numpy(
        image: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
    '''
    Converts a tensor image to grayscale and then to numpy array.

    Args:
        image (ndarray or tf.Tensor): Image to be converted.

    Returns:
        ndarray: Grayscale image as numpy array.
    '''
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if np.max(image) > 1:
        img_gs = image.astype('uint8') @ RGB2GRAY
    else:
        img_gs = (image * 255).astype('uint8') @ RGB2GRAY
        img_gs = img_gs.astype('float32') / 255

    return img_gs


def load_imgs_gen(imgs_path: list,
                  target_size: tuple = (224, 224),
                  scale: float = None,
                  return_pil: bool = False,
                  grayscale: bool = False,
                  shuffle: bool = False) -> np.ndarray:
    '''
    This function returns a generator that loads an images from list of paths
    and resize them.

    Args:
        imgs_path (list): List of paths to images.
        target_size (tuple): Size to which the images will be resized.
        scale (float): Factor used to scale the images. If None, no scaling is
            applied. Defaults to None.
        return_pil (bool): If True, the images are returned as PIL images.
            Defaults to False.
        grayscale (bool): If True, the images are converted to grayscale.
            Defaults to False.
        shuffle (bool): If True, the images are shuffled before being returned.
    '''
    if not isinstance(imgs_path, list):
        imgs_path = [imgs_path]
    if shuffle:
        np.random.shuffle(imgs_path)
    if grayscale:
        mode = 'L'
    else:
        mode = 'RGB'
    for img_path in imgs_path:
        img = np.asarray(
            Image.open(img_path).convert(mode).resize(target_size))
        if scale:
            img = img / scale
        if return_pil:
            yield Image.fromarray(img)
        else:
            yield np.expand_dims(img, axis=0)


def save_img_as_jpg(path: str, quality: int = 75) -> None:
    '''
    Saves an image with JPEG using a given quality factor.

    Args:
        path (str): Path to the image.
        quality (int): Quality factor to be used when saving the image.
            Defaults to 75.

    Returns:
        None
    '''
    img = Image.open(path)
    img.save(path, quality=quality)
