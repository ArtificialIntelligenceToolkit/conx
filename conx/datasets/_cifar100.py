import numpy as np
from keras.utils import to_categorical

def cifar100(dataset):
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    inputs = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))
    targets = to_categorical(labels, 100)
    inputs = inputs.astype('float32')
    inputs /= 255
    dataset.load_direct([inputs], [targets], labels)
