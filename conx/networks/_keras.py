import os

import keras.applications
from keras.utils import get_file

from ..utils import import_keras_model

def vgg16(*args, **kwargs):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1'
    WEIGHTS_NAME = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    if "weights" not in kwargs:
        kwargs["weights"] = None
    model = keras.applications.VGG16(**kwargs)
    network = import_keras_model(model, "VGG16")
    weights_path = get_file(
        WEIGHTS_NAME,
        os.path.join(WEIGHTS_PATH, WEIGHTS_NAME),
        cache_subdir='models',
        file_hash='64373286793e3c8b2b4e3219cbf3544b')
    network.load_weights(*weights_path.rsplit("/", 1))
    return network

def vgg16(*args, **kwargs):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1'
    WEIGHTS_NAME = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    if "weights" not in kwargs:
        kwargs["weights"] = None
    model = keras.applications.VGG16(**kwargs)
    network = import_keras_model(model, "VGG16")
    weights_path = get_file(
        WEIGHTS_NAME,
        os.path.join(WEIGHTS_PATH, WEIGHTS_NAME),
        cache_subdir='models',
        file_hash='64373286793e3c8b2b4e3219cbf3544b')
    network.load_weights(*weights_path.rsplit("/", 1))
    return network

def vgg19(*args, **kwargs):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1'
    WEIGHTS_NAME = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    if "weights" not in kwargs:
        kwargs["weights"] = None
    model = keras.applications.VGG19(**kwargs)
    network = import_keras_model(model, "VGG19")
    weights_path = get_file(
        WEIGHTS_NAME,
        os.path.join(WEIGHTS_PATH, WEIGHTS_NAME),
        cache_subdir='models',
        file_hash='253f8cb515780f3b799900260a226db6')
    network.load_weights(*weights_path.rsplit("/", 1))
    return network
