import numpy as np

from model import Dcscn


def main():
    inputs = np.random.rand(100).astype(np.float32)
    labels = inputs * 0.1 + 0.3

    model = Dcscn()

    model.train(inputs, labels)


if __name__ == "__main__":
    main()
