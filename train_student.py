from models import StudentModel, Distiller
from tensorflow import keras
from data import get_data


def main():
    x_train, y_train, x_test, y_test = get_data()

    print("Loading teacher model")
    teacher_model_dir = "teacher_model"
    teacher_model = keras.models.load_model(teacher_model_dir)

    print("Create student model")
    student_model = StudentModel()
    student_scratch = keras.models.clone_model(student_model)

    print("Create destiller")
    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile()

    print("Training distiller")
    distiller.fit(x_train, y_train)

    print("Testing distiller")
    print(distiller.evaluate(x_test, y_test))

    # Train student as doen usually
    student_scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("Training student model")
    student_scratch.fit(x_train, y_train, epochs=3)

    student_scratch.save("student_model")


if __name__ == "__main__":
    main()

