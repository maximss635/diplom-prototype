import logging
import os

from matplotlib import pyplot as plt
from tensorflow import keras


def compile_student_model(student_model, config):
    config = config["student_model"]

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


def save_plots_student_model(history, config):
    config = config["student_model"]

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

    config = config["student_model"]["train"]
    return student_model.fit(X, y, epochs=config["epochs"])
