"""
    Module that implements entropy based complexity measures.
"""
from typing import Union

import numpy as np
import tensorflow as tf

import utils.data.complexityaux as caux


def delentropy_complexity(image: Union[np.ndarray, tf.Tensor]) -> float:
    '''
    Calculates the complexity of an image using the Delentropy method.

    Args:
        image (ndarray or tf.Tensor): Image to be analysed.

    Returns:
        float: Complexity of the image.
    '''

    x_gradient, y_gradient = caux.image_gradients(image)

    deldensity_value = caux.deldensity(x_gradient, y_gradient)

    entropy = caux.deldensity_entropy(deldensity_value)

    return entropy
