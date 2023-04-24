import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import sys
from time import time

from tensorflow import keras

from data import get_data


def main(context):
    if not os.path.exists(context.dir):
        print("Error: no such dir: {}".format(context.dir))
        sys.exit(1)

    model = keras.models.load_model(context.dir)
    _, _, x_test, y_test = get_data()

    t = time()
    loss, metric = model.evaluate(x_test, y_test)
    t = time() - t
    t = round(t, 2)

    speed = t / x_test.shape[0]

    print("Loss: {}".format(loss))
    print("Metric: {}".format(metric))
    print("Test vector size: {}".format(x_test.shape[0]))
    print("Model time prediction: {} sec".format(t))
    print("Speed: {}".format(speed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)

    main(parser.parse_args())

