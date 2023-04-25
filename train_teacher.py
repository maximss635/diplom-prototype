import os

from data import get_data
from models import TeacherModel
from utils import save_model_schema


def main():
    x_train, y_train, _, _ = get_data()

    teacher_model = TeacherModel()
    teacher_model.compile()
    teacher_model.fit(x_train, y_train)
    teacher_model.save()

    save_model_schema(teacher_model, os.path.join(teacher_model.path(), "debug"))


if __name__ == "__main__":
    main()
