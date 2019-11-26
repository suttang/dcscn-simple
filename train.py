import os

import numpy as np

from model import Dcscn


def main():
    model = Dcscn()

    test_files = get_test_files("set5")
    psnr, ssim = get_metrics(model.test_files)

    model.train()

    import pdb

    pdb.set_trace()

    test_files = "set5"
    # psnr, ssim = model.evaluate()


def get_metrics(model, files):
    psnrs = 0
    ssims = 0

    for file in files:
        psnr, ssim = model.evaluate(file)
        psnrs += psnr
        ssims += ssim
    psnr /= len(files)
    ssim = ssims / len(files)

    return psnr, ssim


def get_test_files(dataset_name):
    dataset_dir = os.path.join(os.getcwd(), "data", dataset_name)
    return [
        dataset_dir + file
        for file in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, file)) and not file.startswith(".")
    ]


if __name__ == "__main__":
    main()
