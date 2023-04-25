import logging
import os

from data import get_data
from models import TeacherModel
from protection import AugmentationMechanism, DistillationMechanism
from utils import read_config, save_model_schema, setup_logger

CONFIG = read_config()


def train_main_model(x_train, y_train):
    logging.info("Training main model")
    teacher_model = TeacherModel()
    teacher_model.compile()
    teacher_model.fit(x_train, y_train)
    teacher_model.save()

    save_model_schema(teacher_model, os.path.join(teacher_model.path(), "debug"))

    return teacher_model


def main():
    setup_logger()

    path_csv = CONFIG["data"]["path_csv"]
    logging.info("Reading data from %s", path_csv)
    X_train, y_train, X_test, y_test = get_data(path_csv)

    # Механизм защиты 1 - аугментация данных
    if CONFIG["protection_methods"]["augmentation"]["state"]:
        logging.info("(PROTECTION-MECHANISM-1) Doing data-augmentation")

        augmentation_mechanism = AugmentationMechanism(X_train, y_train, X_test, y_test)
        augmentation_mechanism.execute()
        X_train, y_train, X_test, y_test = augmentation_mechanism.get_data()

        logging.debug(
            "After augmentation: X_train.shape={}, y_train.shape={}, X_test.shape={}, y_test.shape={}".format(
                X_train.shape, y_train.shape, X_test.shape, y_test.shape
            )
        )

    main_model = train_main_model(X_train, y_train)

    # Механизм защиты 2 - дистилляция модели
    if CONFIG["protection_methods"]["distillation"]["state"]:
        logging.info("(PROTECTION-MECHANISM-2) Doing model-distillation")

        distillation_mechanism = DistillationMechanism(
            main_model, X_train, y_train, X_test, y_test
        )
        distillation_mechanism.execute()

        # Подменяем модель
        model = distillation_mechanism.get_distillated_model()
    else:
        model = main_model


if __name__ == "__main__":
    main()
