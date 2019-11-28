import math
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

# from skimage.measure import compare_psnr, compare_ssim


def he_initializer(shape):
    n = shape[0] * shape[1] * shape[2]
    stddev = math.sqrt(2.0 / n)
    return tf.truncated_normal(shape=shape, stddev=stddev)


def load_image(filename):
    image = np.array(Image.open(filename))
    return image


def save_image(image_array, filepath, is_rgb=True):
    if len(image_array.shape) >= 3 and image_array.shape[2] == 1:
        image_array = image_array.reshape(image_array.shape[0], image_array.shape[1])
    image_array = image_array.round().clip(0, 200).astype(np.uint8)

    image = Image.fromarray(np.uint8(image_array))
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
    new_height = int(height * scale)

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


def add_summaries(
    scope,
    name,
    value,
    save_stddev=False,
    save_mean=False,
    save_max=False,
    save_min=False,
):
    with tf.name_scope(scope):
        mean = tf.reduce_mean(value)

        if save_mean:
            tf.summary.scalar("mean/{}".format(name), mean)
        if save_stddev:
            stddev = tf.sqrt(tf.reduce_mean(tf.square(value - mean)))
            tf.summary.scalar("stddev/{}".format(name), stddev)
        if save_max:
            tf.summary.scalar("max/{}".format(name), tf.reduce_max(value))
        if save_max:
            tf.summary.scalar("min/{}".format(name), tf.reduce_max(value))

        # Too slow
        # tf.summary.histogram(name, value)


def calc_psnr(mse, max_value=255.0):
    if mse is None or mse == float("Inf") or mse == 0:
        psnr = 0
    else:
        psnr = 20 * math.log(max_value / math.sqrt(mse), 10)
    return psnr


def convert_rgb_to_y(image):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image


def convert_ycbcr_to_rgb(ycbcr_image):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])

    rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
    rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
    xform = np.array(
        [
            [298.082 / 256.0, 0, 408.583 / 256.0],
            [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
            [298.082 / 256.0, 516.412 / 256.0, 0],
        ]
    )
    rgb_image = rgb_image.dot(xform.T)

    return rgb_image


def convert_rgb_to_ycbcr(image):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    xform = np.array(
        [
            [65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
            [-37.945 / 256.0, -74.494 / 256.0, 112.439 / 256.0],
            [112.439 / 256.0, -94.154 / 256.0, -18.285 / 256.0],
        ]
    )

    ycbcr_image = image.dot(xform.T)
    ycbcr_image[:, :, 0] += 16.0
    ycbcr_image[:, :, [1, 2]] += 128.0

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image):
    if len(y_image.shape) <= 2:
        y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image[:, :, 0]
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image)


def image_alignment(image, alignment):
    alignment = int(alignment)
    import pdb

    pdb.set_trace()
    # width, height = image.shape


def align_image(image, alignment):
    alignment = int(alignment)
    width, height = image.shape[1], image.shape[0]
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment

    if image.shape[1] != width or image.shape[0] != height:
        image = image[:height, :width, :]

    if len(image.shape) >= 3 and image.shape[2] >= 4:
        image = image[:, :, 0:3]

    return image


def trim_image_as_file(image):
    trimmed = np.clip(np.rint(image), 0, 255)

    return trimmed


def calc_psnr_and_ssim(image1, image2, border=0):
    image1 = trim_image_as_file(image1)
    image2 = trim_image_as_file(image2)

    if border > 0:
        image1 = image1[border:-border, border:-border, :]
        image2 = image2[border:-border, border:-border, :]

    psnr = compare_psnr(image1, image2, data_range=255)
    ssim = compare_ssim(
        image1,
        image2,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        K1=0.01,
        K2=0.03,
        sigma=1.5,
        data_range=255,
    )

    return psnr, ssim


def get_validation_files(dataset_name):
    dataset_dir = os.path.join(os.getcwd(), "data", dataset_name)
    return [
        os.path.join(dataset_dir, file)
        for file in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, file)) and not file.startswith(".")
    ]
