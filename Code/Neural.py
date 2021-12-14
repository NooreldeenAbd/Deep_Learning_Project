import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


model = models.Sequential()


def train(train_images, train_targets, test_images, test_targets, num_classes):
    model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))
    model.summary()

    model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
    model.fit(train_images, train_targets, epochs=11, validation_data=(test_images, test_targets))
    return model.evaluate(test_images, test_targets, verbose=2)


def test(data):
    return from_one_hot(model.predict(data))


def from_one_hot(hot_data):
    result = np.zeros(len(hot_data))
    for i in range(len(hot_data)):
        result[i] = np.argmax(hot_data[i])
    return result


def compute_confusion(true, pred, num_classes):
    k = num_classes
    result = np.zeros((k, k))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result
