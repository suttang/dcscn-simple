import os

import click

from model import Dcscn


@click.command()
@click.argument("output")
def main(output):
    model = Dcscn()
    model.train(output_path=output, validation_dataset="set5")

    # test_files = get_test_files("set5")
    # psnr, ssim = calc_metrics(model, test_files)

    # pdb.set_trace()

    # psnr, ssim = model.evaluate()


if __name__ == "__main__":
    main()
