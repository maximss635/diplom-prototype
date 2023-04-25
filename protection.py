import logging
import os

import numpy as np
from tensorflow import keras

from models import Distiller, StudentModel
from student_model_utils import (
    compile_student_model,
    save_plots_student_model,
    train_student_model,
)
from utils import read_config, save_model_schema


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


class DistillationMechanism:
    """
    Защитный механизм дистилляции

    Входные параметры: Обученная модель МО
    Выходные параметры: Дистиллированная модель МО
    """

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.teacher_model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.student_model = None

        self._config = read_config()["protection_methods"]["distillation"]

    def execute(self):
        logging.debug("Create student model")
        student_model = StudentModel()
        student_scratch = keras.models.clone_model(student_model)

        logging.debug("Create destiller")
        distiller = Distiller(student=student_model, teacher=self.teacher_model)
        distiller.compile()

        logging.debug("Training distiller")
        distiller.fit(self.X_train, self.y_train)

        logging.debug("Testing distiller")
        test_results = distiller.evaluate(self.X_test, self.y_test)
        logging.debug("Test results = {}".format(test_results))

        # Train student as doen usually
        student_scratch = compile_student_model(student_scratch, self._config)
        history = train_student_model(
            student_scratch, self._config, self.X_train, self.y_train
        )

        logging.info(
            "Saving student model to '{}'".format(self._config["student_model"]["dir"])
        )
        student_scratch.save(self._config["student_model"]["dir"])

        save_model_schema(
            student_scratch,
            os.path.join(self._config["student_model"]["dir"], "debug"),
        )

        save_plots_student_model(history, self._config)

        self.student_model = student_scratch

    def get_distillated_model(self):
        return self.student_model
