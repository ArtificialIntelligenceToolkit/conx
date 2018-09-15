import conx as cx
import numpy as np
from keras.datasets import mnist
from keras.utils import (to_categorical, get_file)

description = """
Original source: http://yann.lecun.com/exdb/mnist/

The MNIST dataset contains 70,000 images of handwritten digits (zero
to nine) that have been size-normalized and centered in a square grid
of pixels.  Each image is a 28 × 28 × 1 array of floating-point numbers
representing grayscale intensities ranging from 0 (black) to 1
(white).  The target data consists of one-hot binary vectors of size
10, corresponding to the digit classification categories zero through
nine.  Some example MNIST images are shown below:

![MNIST Images](https://github.com/Calysto/conx-data/raw/master/mnist/mnist_images.png)
"""

def mnist_h5(*args, **kwargs):
    """
    Load the Keras MNIST dataset from an H5 file.
    """
    import h5py

    path = "mnist.h5"
    url = "https://raw.githubusercontent.com/Calysto/conx-data/master/mnist/mnist.h5"
    path = get_file(path, origin=url)
    h5 = h5py.File(path, "r")
    dataset = cx.Dataset()
    dataset._inputs = h5["inputs"]
    dataset._targets = h5["targets"]
    dataset._labels = h5["labels"]
    dataset.h5 = h5
    dataset.name = "MNIST-H5"
    dataset.description = description
    dataset._cache_values()
    return dataset

def mnist(*args, **kwargs):
    from keras.datasets import mnist
    import keras.backend as K

    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float16')
    x_test = x_test.astype('float16')
    inputs = np.concatenate((x_train,x_test)) / 255
    labels = np.concatenate((y_train,y_test)) # ints, 0 to 10
    ###########################################
    # fix mis-labeled image(s) in Keras dataset
    labels[10994] = 9
    ###########################################
    targets = to_categorical(labels).astype("uint8")
    labels = np.array([str(label) for label in labels], dtype=str)
    dataset = cx.Dataset()
    dataset.load_direct([inputs], [targets], [labels])
    return dataset
