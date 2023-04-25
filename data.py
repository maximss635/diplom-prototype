import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from protection import AugmentationMechanism
from utils import setup_logger


def get_data(path_csv):
    setup_logger()
    path = path_csv

    df = pd.read_csv(path)
    df = df.drop(["proto", "state"], axis=1)
    df = df.sample(frac=1)

    logging.info("Preprocessing data")

    X = StandardScaler().fit_transform(df.drop(["attack"], axis=1).values)
    y = df.attack.values

    logging.debug(
        "After fit-transform: X.shape={}, y.shape={}".format(X.shape, y.shape)
    )

    # Hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

    logging.debug(
        "After hold-out: X_train.shape={}, y_train.shape={}, X_test.shape={}, y_test.shape={}".format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape
        )
    )

    return X_train, y_train, X_test, y_test
