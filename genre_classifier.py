import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])

    return inputs, labels


def plot(history):
    fig, axis = plt.subplots(2)

    # accuracy subplot
    axis[0].plot(history.history["accuracy"], label="train accuracy")
    axis[0].plot(history.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy Evaluation")

    # error subplot
    axis[1].plot(history.history["loss"], label="train error")
    axis[1].plot(history.history["val_loss"], label="test error")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error Evaluation")

    plt.show()


if __name__ == "__main__":
    # load data
    inputs, labels = load_data("data.json")

    # split data
    train_x, test_x, train_y, test_y = train_test_split(inputs, labels, test_size=0.2)

    # create model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation="relu"),  # 1st layer
        keras.layers.Dense(256, activation="relu"),  # 2st layer
        keras.layers.Dense(64, activation="relu"),   # 3st layer
        keras.layers.Dense(10, activation="softmax") # output layer
    ])

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # train model
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=50, batch_size=32)
    plot(history)
