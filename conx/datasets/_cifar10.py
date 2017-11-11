import numpy as np
from keras.utils import to_categorical

def cifar10(dataset):
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    inputs = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))
    targets = to_categorical(labels, 10)
    inputs = inputs.astype('float32')
    inputs /= 255
    dataset.load_direct([inputs], [targets], labels)
