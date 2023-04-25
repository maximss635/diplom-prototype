import numpy as np
from utils import read_config


class AugmentationMechanism:
    """
    Защитный механизм аугментации.
    Применяется на этапе предобработки данных для обучения

    Входные данные: предобработанные данные для обучения модели
    Выходные данные: аугментированные данные
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self._config = read_config()["protection_methods"]["augmentation"]

    def execute(self):
        X_copy = np.array(self.X_train, copy=True)
        y_copy = np.array(self.y_train, copy=True)

        X_augmented = []
        y_augmented = []

        l = len(self.X_train) * self._config["fraction"]
        l = int(l)

        for i in range(l // 2):
            # Случайное изменение порядка признаков
            np.random.shuffle(X_copy[i])

            X_augmented.append(X_copy[i])
            y_augmented.append(self.y_train[i])

        for i in range(l // 2, l):
            # Прибавление случайного шума
            noise = np.random.normal(0, 0.1, len(X_copy[0]))
            X_noisy = X_copy[i] + noise
            X_augmented.append(np.array(X_noisy))
            y_augmented.append(self.y_train[i])

        self.X_train = np.concatenate((self.X_train, np.array(X_augmented)), axis=0)
        self.y_train = np.concatenate((self.y_train, np.array(y_augmented)), axis=0)

    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test
