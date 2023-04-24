import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import logging
import shutil
from contextlib import suppress

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _read_config(path):
    with open(path, "r") as fd:
        return json.load(fd)


CONFIG = _read_config("config.json")


class ModelBase(keras.Sequential):
    def __init__(self, model_name):
        keras.Sequential.__init__(self, name=model_name)

        self.model_name = model_name

        self.__config = _read_config("config.json")

    def fit(self, X, y):
        logging.debug(
            "Start training model '{}'. epochs={}, training_vector_size={}".format(
                self.name, self.__config[self.model_name]["train"]["epochs"], X.shape[0]
            )
        )

        keras.Sequential.fit(
            self,
            X,
            y,
            epochs=self.__config[self.model_name]["train"]["epochs"],
            validation_split=self.__config[self.model_name]["train"]["validation_split"],
        )

    def save(self):
        with suppress(FileNotFoundError):
            shutil.rmtree("teacher_model")

        keras.Sequential.save(self, self.__config[self.model_name]["dir"])


class TeacherModel(ModelBase):
    def __init__(self):
        ModelBase.__init__(self, "teacher")

        self.add(keras.Input(shape=(12,)))
        self.add(layers.Dense(64, activation="relu"))
        self.add(layers.Dropout(0.15))
        self.add(layers.Dense(32, activation="relu"))
        self.add(layers.Dropout(0.25))
        self.add(layers.Dense(16, activation="relu"))
        self.add(layers.Dropout(0.25))
        self.add(layers.Dense(1, activation="sigmoid"))

    def compile(self):
        keras.Sequential.compile(
            self,
            optimizer=keras.optimizers.Adam(0.002, 0.5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )


class StudentModel(ModelBase):
    def __init__(self):
        ModelBase.__init__(self, "student")

        self.add(keras.Input(shape=(12,)))
        self.add(layers.Dense(16, activation="relu"))
        self.add(layers.Dropout(0.25))
        self.add(layers.Dense(1, activation="sigmoid"))


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        keras.Model.__init__(self)

        self.teacher = teacher
        self.student = student

    def compile(self):
        optimizer = keras.optimizers.Adam()
        metrics = ['accuracy']

        keras.Model.compile(self, optimizer=optimizer, metrics=metrics)

        self.student_loss_fn = keras.losses.BinaryCrossentropy()
        self.distillation_loss_fn = keras.losses.KLDivergence()
        self.alpha = 0.1
        self.temperature = 10

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    def fit(self, X, y):
        keras.Model.fit(self, X, y, epochs=CONFIG["distiller"]['train']["epochs"])
