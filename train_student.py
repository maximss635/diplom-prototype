import json
import logging
import os

from matplotlib import pyplot as plt
from tensorflow import keras

from data import get_data
from models import Distiller, StudentModel
from utils import read_config, save_model_schema, setup_logger


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


def save_plots(history, config):
    debug_path = os.path.join(config["dir"], "debug")
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)

    path_accuracy = os.path.join(debug_path, "accuracy.png")
    path_loss = os.path.join(debug_path, "loss.png")

    logging.debug("Draw plots")

    plt.figure(figsize=(16, 10))

    # accuracy
    plt.plot(history.history["accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(path_accuracy)

    # "Loss"
    plt.figure(figsize=(16, 10))
    plt.plot(history.history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(path_loss)


def train_student_model(student_model, config, X, y):
    logging.info("Training student model")

    config = config["protection_methods"]["distillation"]["student_model"]["train"]
    return student_model.fit(X, y, epochs=config["epochs"])


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
    history = train_student_model(student_scratch, config, x_train, y_train)

    logging.info("Saving student model to '{}'".format(student_model_config["dir"]))
    student_scratch.save(student_model_config["dir"])

    save_model_schema(
        student_scratch, os.path.join(student_model_config["dir"], "debug")
    )

    save_plots(history, student_model_config)


if __name__ == "__main__":
    setup_logger()
    main()
