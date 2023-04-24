import logging

from tensorflow import keras

from data import get_data
from logger import setup_logger
from models import Distiller, StudentModel


def main():
    x_train, y_train, x_test, y_test = get_data()

    logging.info("Loading teacher model")
    teacher_model_dir = "models/model_1"
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
    student_scratch.compile(
        optimizer=keras.optimizers.Adam(0.002, 0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    logging.info("Training student model")
    student_scratch.fit(x_train, y_train, epochs=15)

    student_scratch.save("models/student_model")


if __name__ == "__main__":
    setup_logger()
    main()

