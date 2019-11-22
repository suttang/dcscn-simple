import os

import click

from model import Dcscn
from utils import load_image
from utils import save_image
from utils import resize_image


@click.command()
@click.argument("input")
@click.argument("output")
def main(input, output):
    input_image = load_image(input)
    model = Dcscn()

    # Save original image
    os.makedirs(output, exist_ok=True)
    save_image(input_image, "{}/original.jpg".format(output))

    # Save bicubic resize image
    scaled_image = resize_image(input_image, 2)
    save_image(scaled_image, "{}/bicubic.jpg".format(output))

    model.inference(input_image, output)


if __name__ == "__main__":
    main()
