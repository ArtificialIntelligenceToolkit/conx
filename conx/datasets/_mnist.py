import numpy as np
from keras.utils import to_categorical

def mnist(dataset):
    """
    Load the Keras MNIST dataset and format it as images.
    """
    from keras.datasets import mnist
    import keras.backend as K
    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    ## We need to convert the data to images, but which format?
    ## We ask this Keras instance what it wants, and convert:
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    inputs = np.concatenate((x_train,x_test))
    labels = np.concatenate((y_train,y_test))
    targets = to_categorical(labels)
    labels = np.array([str(label) for label in labels], dtype=str)
    dataset.name = "MNIST"
    dataset.description = """
Original source: http://yann.lecun.com/exdb/mnist/

The MNIST database of handwritten digits, available from this page,
has 70,000 examples. It is a subset of a larger set available from
NIST. The digits have been size-normalized and centered in a
fixed-size image.  It is a good database for people who want to try
learning techniques and pattern recognition methods on real-world data
while spending minimal efforts on preprocessing and formatting.
"""
    dataset.load_direct([inputs], [targets], [labels])
