import numpy as np
from keras.utils import to_categorical

def cifar10(dataset):
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    inputs = np.concatenate((x_train, x_test))
    x_train, x_test = None, None
    inputs = inputs.astype('float32')
    inputs /= 255
    labels = np.concatenate((y_train, y_test))
    y_train, y_test = None, None
    targets = to_categorical(labels, 10)
    labels = np.array([str(label[0]) for label in labels], dtype=str)
    dataset.name = "CIFAR-10"
    dataset.description = """
Original source: https://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10
classes, with 6000 images per class.

The classes are completely mutually exclusive. There is no overlap
between automobiles and trucks. "Automobile" includes sedans, SUVs,
things of that sort. "Truck" includes only big trucks. Neither
includes pickup trucks.
"""
    dataset.load_direct([inputs], [targets], [labels])
