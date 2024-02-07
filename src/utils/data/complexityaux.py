"""
    Module with simple functions that are used by all different image
    complexity measures.
"""
from typing import Union

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
rng = np.random.default_rng(2403)


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
        rng.shuffle(imgs_path)
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
    # Check if image is grayscale and convert to grayscale if not
    if len(image.shape) == 3:
        image = image_rgb_to_grayscale(image)
    if np.max(image) <= 1:
        image = image * 255
    # Calculate gradients
    g_x = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    g_y = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    # Calculate gradients. Remember that the convolution operation is
    # commutative, so the order of the kernels does not matter. We keep the
    # image first to use the 'same' mode and get an output of the same shape as
    # the input image.
    x_gradient = signal.convolve2d(image, g_x, mode='same')

    y_gradient = signal.convolve2d(image, g_y, mode='same')

    return x_gradient, y_gradient


def deldensity(x_gradient: np.ndarray,
               y_gradient: np.ndarray,
               visualise: bool = False) -> np.ndarray:
    '''

    Calculates the deldensity of an image, as presented in:

    Reflections on Shannon Information: In search of a natural
    information-entropy for images, https://arxiv.org/pdf/1609.01117.pdf

    Args:
        x_gradient (ndarray): X gradient of the image.
        y_gradient (ndarray): Y gradient of the image.
        visualise (bool): If True, the deldensity is visualised. Defaults to
            False.

    Returns:
        ndarray: Deldensity of the image.

    '''

    deldensity_value, xedges, yedges = np.histogram2d(
        x_gradient.flatten(), y_gradient.flatten(), bins=256,
        density=True, range=[[-255, 255], [-255, 255]])
    deldensity_value = deldensity_value.T

    if visualise:
        # gamma enhancements and inversion for better viewing pleasure
        deldensity_value = np.max(deldensity_value) - deldensity_value
        gamma = 1.8
        deldensity_value = (
            deldensity_value / np.max(deldensity_value)
        )**gamma * np.max(deldensity_value)
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(132, title='Image delentropy',
                             aspect='equal')
        x, y = np.meshgrid(xedges, yedges)
        # Set cmap to black and white
        cmap = plt.get_cmap('binary')
        ax.pcolormesh(x, y, deldensity_value, cmap=cmap)
        ax.set_xlabel('X gradient')
        ax.set_ylabel('Y gradient')

    return deldensity_value


def deldensity_entropy(deldensity_value: np.ndarray) -> float:
    '''
    Calculates the entropy of the deldensity of an image.

    Args:
        deldensity (ndarray): Deldensity of an image.

    Returns:
        float: Entropy of the deldensity of an image.
    '''
    nonzero_deldesnity = deldensity_value[deldensity_value.nonzero()]

    entropy = -0.5 * np.sum(nonzero_deldesnity * np.log2(nonzero_deldesnity))

    return entropy
