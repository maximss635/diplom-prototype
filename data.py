import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def get_data():
    df = pd.read_csv("data/Balanced.csv")
    df = df.drop(["proto", "state"], axis=1)
    df = df.sample(frac=1)
    X = StandardScaler().fit_transform(df.drop(["attack"], axis=1).values)
    y = df.attack.values

    # Hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

    return X_train, y_train, X_test, y_test
