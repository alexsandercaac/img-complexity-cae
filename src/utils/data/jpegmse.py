"""
    Module that implements the JPEG MSE image complexity measure.
"""
import numpy as np
import PIL
import tempfile
import cv2
import utils.data.complexityaux as caux


def jpeg_mse(
        image: np.ndarray, quality: int = 75) -> float:
    """
    Calculates the MSE  between an image and its JPEG compressed version.

    This function uses cv2.imencode and cv2.imdecode to compress and
    decompress the image, respectively, without writing the compressed image
    to disk.

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


def jpeg_mse_complexity(
        image: np.ndarray, quality: int = 75) -> float:
    """
    Calculates the JPEG RMSE image complexity measure. This measure is obtained
    by dividing the MSE between the image and its compressed version by the
    compression ratio, which is the size of the image before compression
    divided by its size after compression.

    Args:
        image (numpy.ndarray): Image to be processed.
        quality (int, optional): JPEG quality. Defaults to 75.

    Returns:
        float: JPEG MSE image complexity measure.
    """
    # Check if image is not in PIL format
    if not isinstance(image, PIL.Image.Image):
        # If greyscale, remove the channel dimension
        image = np.squeeze(image)
        # Check if image is scaled
        if np.max(image) > 1:
            image = image.astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)
        image = PIL.Image.fromarray(image)

    # Save image to a temporary jpeg file
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
        image.save(tmp_file.name, quality=quality)
        # Get the size of the file
        tmp_file.seek(0, 2)
        jpeg_size = tmp_file.tell()
        # Reload the compressed image
        jpeg_image = PIL.Image.open(tmp_file.name)

    # Save image to a temporary png file
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp_file:
        image.save(tmp_file.name)
        # Get the size of the file
        tmp_file.seek(0, 2)
        png_size = tmp_file.tell()

    compression_ratio = png_size / jpeg_size

    # Calculate the MSE between the original image and the compressed image
    mse = caux.image_mse(image, jpeg_image)

    return mse / compression_ratio
