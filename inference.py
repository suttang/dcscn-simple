import os

import click

from model import Dcscn
from utils import load_image
from utils import save_image
from utils import resize_image


@click.command()
@click.argument("input")
@click.argument("output")
@click.option("-m", "--model", "model_name")
def main(input, output, model_name):

    input_image = load_image(input)
    model = Dcscn()
    model.load(model)

    os.makedirs(output, exist_ok=True)
    model.inference(input_image, output, save_images=True)


if __name__ == "__main__":
    main()
