import logging
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from config import setup_logger


def augmentation(X, y):
    X_copy = np.array(X, copy=True)
    y_copy = np.array(y, copy=True)

    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        # Случайное изменение порядка признаков
        np.random.shuffle(X_copy[i])

        X_augmented.append(X_copy[i])
        y_augmented.append(y[i])

        # Прибавление случайного шума
        noise = np.random.normal(0, 0.1, len(X_copy[0]))
        X_noisy = X_copy[i] + noise
        X_augmented.append(np.array(X_noisy))
        y_augmented.append(y[i])

    return np.array(X_augmented), np.array(y_augmented)


NEED_AUGMENTATION = True


def get_data():
    setup_logger()

    path = "data/Balanced.csv"
    logging.debug("Reading dataframe from %s", path)

    df = pd.read_csv(path)
    df = df.drop(["proto", "state"], axis=1)
    df = df.sample(frac=1)

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

    if NEED_AUGMENTATION:
        X_augmented, y_augmented = augmentation(X_train, y_train)
        logging.debug(
            "After augmentation: X_augmented.shape={}, y_augmented={}".format(
                X_augmented.shape, y_augmented.shape
            )
        )

        X_train = np.concatenate((X_train, X_augmented), axis=0)
        y_train = np.concatenate((y_train, y_augmented), axis=0)

        logging.debug(
            "After augmentation: X_train.shape={}, y_train.shape={}, X_test.shape={}, y_test.shape={}".format(
                X_train.shape, y_train.shape, X_test.shape, y_test.shape
            )
        )

    return X_train, y_train, X_test, y_test
