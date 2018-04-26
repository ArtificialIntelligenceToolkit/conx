import os

import keras.applications
from keras.utils import get_file

from ..utils import import_keras_model

def vgg16(*args, **kwargs):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1'
    WEIGHTS_NAME = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model = keras.applications.VGG16()
    network = import_keras_model(model, "vgg16")
    weights_path = get_file(
        WEIGHTS_NAME,
        os.path.join(WEIGHTS_PATH, WEIGHTS_NAME),
        cache_subdir='models',
        file_hash='64373286793e3c8b2b4e3219cbf3544b')
    network.load_weights(*weights_path.rsplit("/", 1))
    return network
