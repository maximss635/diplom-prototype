import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import setup_logger
from protection import AugmentationMechanism

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
        # Выполняем аугментацию
        augmentation_mechanism = AugmentationMechanism(X_train, y_train, X_test, y_test)
        augmentation_mechanism.execute()
        X_train, y_train, X_test, y_test = augmentation_mechanism.get_data()

        logging.debug(
            "After augmentation: X_train.shape={}, y_train.shape={}, X_test.shape={}, y_test.shape={}".format(
                X_train.shape, y_train.shape, X_test.shape, y_test.shape
            )
        )

    return X_train, y_train, X_test, y_test
