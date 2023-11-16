#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
import random
import requests
import cv2
from PIL import Image

np.random.seed(0)
num_samples = []
num_classes = 10
num_pixels = 784


def main():
    X_train, y_train, X_test, y_test = pre_process_data()
    model = create_leNet_model()
    print(model.summary())
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10,
                        batch_size=200, verbose=1, shuffle=1)
    plot_performance(history)
    evaluate_model(model, X_test, y_test)
    test_model_with_input(model)


def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])


def test_model_with_input(model):
    url = 'https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png'
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    plt.imshow(img)
    plt.show()
    img_array = np.asarray(img)
    resized = cv2.resize(img_array, (28, 28))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(grayscale)
    image = image / 255
    image = image.reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(image), axis=-1)
    print("Predicted digit: ", str(prediction))


def plot_performance(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['loss', 'validation_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()


def create_leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(
        28, 28, 1), activation='relu'))  # 24*24
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 12*12
    model.add(Conv2D(15, (3, 3), activation='relu'))  # 10*10
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 5*5
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def one_hot_encode_labels(y_train, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(y_train[0])
    return y_train, y_test


def pre_process_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    label_shape_match(X_train, y_train, X_test, y_test)
    print_sample_images(X_train, y_train, num_classes)
    print(num_samples)
    plot_num_samples(num_samples)
    y_train, y_test = one_hot_encode_labels(y_train, y_test)
    X_train, X_test = convert_zero_one(X_train, X_test)
    X_train, X_test = reshape_data(X_train, X_test, num_pixels)
    return X_train, y_train, X_test, y_test


def reshape_data(X_train, X_test, num_pixels):
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    return X_train, X_test


def convert_zero_one(X_train, X_test):
    X_train = X_train / 255
    X_test = X_test / 255
    return X_train, X_test


def plot_num_samples(num_samples):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()


def print_sample_images(X_train, y_train, num_classes):
    cols = 5
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
    fig.tight_layout()
    for i in range(cols):
        for j in range(num_classes):
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(
                0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                num_samples.append(len(x_selected))
    plt.show()


def label_shape_match(X_train, y_train, X_test, y_test):
    assert (X_train.shape[0] == y_train.shape[0]
            ), "The number of images is not equal to the number of labels."
    assert (X_test.shape[0] == y_test.shape[0]
            ), "The number of images is not equal to the number of labels."
    assert (X_train.shape[1:] == (28, 28)
            ), "The dimensions of the images are not 28x28."
    assert (X_test.shape[1:] == (28, 28)
            ), "The dimensions of the images are not 28x28."


if __name__ == '__main__':
    main()
