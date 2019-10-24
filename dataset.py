import functools
from glob import glob
import os
import random

import numpy as np

from utils import load_image, resize_image


def convert_rgb_to_y(image):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image


class Loader:
    def __init__(
        self,
        name,
        scale,
        image_size,
        batch_size,
        channels=1,
        resampling_method="bicubic",
    ):
        self.name = name
        self.scale = scale
        self.image_size = image_size
        self.batch_size = batch_size
        self.channels = channels
        self.resampling_method = resampling_method

        self.dataset_dir = os.path.join(os.getcwd(), "data", name)

        self.index = 0

    @property
    @functools.lru_cache(maxsize=None)
    def files(self):
        return [filepath for filepath in glob(os.path.join(self.dataset_dir, "*.png"))]

    def get_random_patched_image(self, filename):
        image = load_image(filename)
        height, width = image.shape[0:2]

        size = self.image_size * self.scale

        if height < size or width < size:
            print(
                "Error: {} should have more than {} x {} size.".format(
                    filename, size, size
                )
            )
            return None

        x = random.randrange(height - size) if height != size else 0
        y = random.randrange(width - size) if width != size else 0

        image = image[x : x + size, y : y + size, :]

        # Convert 1 channel (y)
        image = convert_rgb_to_y(image)

        return image

    def get_images(self, index):
        image = self.get_random_patched_image(self.files[index])

        if random.randrange(2) == 0:
            image = np.fliplr(image)

        input_image = resize_image(image, 1 / self.scale)
        input_scaled_image = resize_image(input_image, self.scale)

        return input_image, input_scaled_image, image

    def feed(self):
        return self.__next__()

    def __next__(self):
        input_images = []
        upscaled_images = []
        original_images = []
        for i in range(self.batch_size):
            input_image, upscaled_image, original_image = self.get_images(self.index)
            input_images.append(input_image)
            upscaled_images.append(upscaled_image)
            original_images.append(original_image)

            self.index += 1
            if self.index >= len(self.files):
                self.index = 0

        return input_images, upscaled_images, original_images

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)
