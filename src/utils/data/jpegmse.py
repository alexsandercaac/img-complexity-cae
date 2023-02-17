"""
    Module that implements the JPEG MSE image complexity measure.
"""
import numpy as np
import cv2
import src.utils.data.complexityaux as caux


def calculate_jpeg_mse(
        image: np.ndarray, quality: int = 75) -> float:
    """Calculates the JPEG MSE image complexity measure.

    Args:
        image (numpy.ndarray): Image to be processed.
        quality (int, optional): JPEG quality. Defaults to 75.

    Returns:
        float: JPEG MSE image complexity measure.
    """
    image = np.squeeze(image)

    # Check if pixels in image are scaled or not. If the maximum value is
    # greater than 1, then the pixels are not scaled. Otherwise, the pixels
    # are assumed to be scaled in the range [0, 1]. If the image is scaled,
    # it is converted to the range 0-255 before being encoded, because of the
    # constraints of the cv2.imencode function.
    if np.max(image) > 1:
        image = image.astype(np.uint8)
        scaled = False
    else:
        image = (image * 255).astype(np.uint8)
        scaled = True
    # Check if image is grayscale or not
    if len(image.shape) == 2:
        decode_flag = cv2.IMREAD_GRAYSCALE
    else:
        decode_flag = cv2.IMREAD_COLOR

    _, jpeg_image = cv2.imencode(
        ".jpg", image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    jpeg_image = cv2.imdecode(jpeg_image, decode_flag)
    # The MSE is calculated using the original range of the input image.
    if scaled:
        jpeg_image = jpeg_image.astype(np.float32) / 255
        image = image.astype(np.float32) / 255
        mse = caux.image_mse(image, jpeg_image)
    else:
        mse = caux.image_mse(
            image.astype(np.float16), jpeg_image.astype(np.float16)
        )
    return mse
