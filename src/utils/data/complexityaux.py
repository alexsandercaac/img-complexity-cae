"""
    Module with simple functions that are used by all different image
    complexity measures.
"""
import numpy as np
import tensorflow as tf

from typing import Union
from PIL import Image
from scipy import signal

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


def image_rgb_to_grayscale(
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
    # Check if pixels in image are scaled or not. If the maximum value is
    # greater than 1, then the pixels are not scaled. Otherwise, the pixels
    # are assumed to be scaled in the range [0, 1].
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


def image_gradients(image: Union[np.ndarray, tf.Tensor]
                    ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Calculates the x and y gradients of an image.

    The sobel filter is used to calculate the gradients, with the following
    kernels:

        g_x = [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]]

        g_y = [[+1, +2, +1],
                [0,  0,  0],
                [-1,  -2,  -1]]

    Args:
        x_gradient, y_gradient

    Returns:
        ndarray: Gradients of the image.
    '''

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    image = np.squeeze(image)

    # Calculate gradients
    g_x = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    g_y = np.array([[1, 2, 1],
                   [0,  0,  0],
                   [-1,  -2,  -1]])

    # Calculate gradients. Remember that the convolution operation is
    # commutative, so the order of the kernels does not matter. We keep the
    # image first to use the 'same' mode and get an output of the same shape as
    # the input image.
    x_gradient = signal.convolve2d(image, g_x, mode='same')

    y_gradient = signal.convolve2d(image, g_y, mode='same')

    return x_gradient, y_gradient
