from models import TeacherModel
from data import get_data


def main():
    x_train, y_train, _, _ = get_data()

    teacher_model = TeacherModel()
    teacher_model.compile()
    teacher_model.fit(x_train, y_train)
    teacher_model.save()


if __name__ == "__main__":
    main()

