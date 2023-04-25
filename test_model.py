import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
import sys
from time import time

from tensorflow import keras

from data import get_data
from config import setup_logger


def main(context):
    if not os.path.exists(context.dir):
        logging.error("Error: no such dir: {}".format(context.dir))
        sys.exit(1)

    model = keras.models.load_model(context.dir)
    _, _, x_test, y_test = get_data()

    t = time()
    loss, metric = model.evaluate(x_test, y_test)
    t = time() - t
    t = round(t, 2)

    speed = t / x_test.shape[0]

    logging.info("Loss: {}".format(loss))
    logging.info("Metric: {}".format(metric))
    logging.info("Test vector size: {}".format(x_test.shape[0]))
    logging.info("Model time prediction: {} sec".format(t))
    logging.info("Speed: {}".format(speed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)

    setup_logger()
    main(parser.parse_args())
