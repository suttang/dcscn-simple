import math

import numpy as np
from PIL import Image
import tensorflow as tf


def he_initializer(shape):
    n = shape[0] * shape[1] * shape[2]
    stddev = math.sqrt(2.0 / n)
    return tf.truncated_normal(shape=shape, stddev=stddev)


def load_image(filename):
    image = np.array(Image.open(filename))
    return image


def save_image(image_array, filepath, is_rgb=True):
    if is_rgb:
        image = Image.fromarray(np.uint8(image_array))
    else:
        image = Image.fromarray(np.uint8(image_array[:, :, 0]), "L")
    image.save(filepath)


def resize_image(image_array, scale, resampling_method="bicubic"):
    resampling_methods = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
    }

    height, width = image_array.shape[0:2]
    new_width = int(width * scale)
    new_height = int(width * scale)

    if resampling_method not in resampling_methods:
        raise ValueError(
            "The resampling method: {} dame, bicubic or bilinear or nearest or lanczos."
        )

    method = resampling_methods[resampling_method]

    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # RGB
        image_array = Image.fromarray(image_array, "RGB")
        image_array = image_array.resize([new_width, new_height], resample=method)
        image_array = np.asarray(image_array)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        # RGBA
        image_array = Image.fromarray(image_array, "RGB")
        image_array = image_array.resize([new_width, new_height], resample=method)
        image_array = np.asarray(image_array)
    else:
        image_array = Image.fromarray(image_array.reshape(height, width))
        image_array = image_array.resize([new_width, new_height], resample=method)
        image_array = np.asarray(image_array)
        image_array = image_array.reshape(new_height, new_width, 1)

    return image_array
