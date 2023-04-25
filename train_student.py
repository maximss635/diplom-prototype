import json
import logging

from tensorflow import keras

from config import read_config, setup_logger
from data import get_data
from models import Distiller, StudentModel


def compile_student_model(student_model, config):
    config = config["protection_methods"]["distillation"]["student_model"]

    logging.debug("Compiling student model")

    optimizer = keras.optimizers.Adam(
        config["train"]["learning_rate"],
        config["train"]["beta_1"],
        config["train"]["beta_2"],
    )

    loss = config["train"]["loss"]
    metrics = config["train"]["metrics"]

    student_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return student_model


def train_student_model(student_model, config, X, y):
    logging.info("Training student model")

    config = config["protection_methods"]["distillation"]["student_model"]["train"]
    student_model.fit(X, y, epochs=config["epochs"])


def main():
    config = read_config()
    student_model_config = config["protection_methods"]["distillation"]["student_model"]

    x_train, y_train, x_test, y_test = get_data()

    teacher_model_dir = config["model"]["dir"]
    logging.info("Loading teacher model - {}".format(teacher_model_dir))
    teacher_model = keras.models.load_model(teacher_model_dir)

    logging.info("Create student model")
    student_model = StudentModel()
    student_scratch = keras.models.clone_model(student_model)

    logging.info("Create destiller")
    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile()

    logging.info("Training distiller {} {}".format(x_train.shape, y_train.shape))
    distiller.fit(x_train, y_train)

    logging.info("Testing distiller")
    test_results = distiller.evaluate(x_test, y_test)
    logging.info("Test results = {}".format(test_results))

    # Train student as doen usually
    student_scratch = compile_student_model(student_scratch, config)
    train_student_model(student_scratch, config, x_train, y_train)

    logging.info("Saving student model to '{}'".format(student_model_config["dir"]))
    student_scratch.save(student_model_config["dir"])


if __name__ == "__main__":
    setup_logger()
    main()
