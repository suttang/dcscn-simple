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

    os.makedirs(output, exist_ok=True)
    model.inference(input_image, output, save_images=True)


if __name__ == "__main__":
    main()
