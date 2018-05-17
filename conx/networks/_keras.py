import os
import numpy as np

import keras.applications
from keras.utils import get_file
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from ..utils import import_keras_model

def vgg_preprocess(input):
    batch = np.array(input).reshape((1, 224, 224, 3))
    assert np.min(batch) >= 0 and np.max(batch) <= 1
    batch *= 255
    b =  preprocess_input(batch)
    return b[0].tolist()

def inceptionv3_preprocess(input):
    batch = np.array(input).reshape((1, 299, 299, 3))
    assert np.min(batch) >= 0 and np.max(batch) <= 1
    batch *= 255
    b = preprocess_input(batch, mode='tf')
    return b[0].tolist()

def vgg_decode(probabilities, top=5):
    return decode_predictions(np.array(probabilities).reshape((1,1000)), top=top)[0]

def vgg16(*args, **kwargs):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1'
    WEIGHTS_NAME = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    if "weights" not in kwargs:
        kwargs["weights"] = None
    model = keras.applications.VGG16(**kwargs)
    network = import_keras_model(model, "VGG16", build_propagate_from_models=False)
    weights_path = get_file(
        WEIGHTS_NAME,
        os.path.join(WEIGHTS_PATH, WEIGHTS_NAME),
        cache_subdir='models',
        file_hash='64373286793e3c8b2b4e3219cbf3544b')
    network.load_weights(*weights_path.rsplit("/", 1))
    network.config["hspace"] = 200
    network.preprocess = vgg_preprocess
    network.postprocess = vgg_decode
    network.information = """
This network architecture comes from the paper:

Very Deep Convolutional Networks for Large-Scale Image Recognition
by Karen Simonyan and Andrew Zisserman.

Their network was trained on the ImageNet challenge dataset.
The dataset contains 32,326 images broken down into 1,000 categories.

The network was trained for 74 epochs on the training data. This typically
took 3 to 4 weeks time on a computer with 4 GPUs. This network's weights were
converted from the original Caffe model into Keras.

Sources:
   * https://arxiv.org/pdf/1409.1556.pdf 
   * http://www.robots.ox.ac.uk/~vgg/research/very_deep/ 
   * http://www.image-net.org/challenges/LSVRC/ 
      * http://image-net.org/challenges/LSVRC/2014/ 
      * http://image-net.org/challenges/LSVRC/2014/browse-synsets 
"""
    return network

def vgg19(*args, **kwargs):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1'
    WEIGHTS_NAME = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    if "weights" not in kwargs:
        kwargs["weights"] = None
    model = keras.applications.VGG19(**kwargs)
    network = import_keras_model(model, "VGG19", build_propagate_from_models=False)
    weights_path = get_file(
        WEIGHTS_NAME,
        os.path.join(WEIGHTS_PATH, WEIGHTS_NAME),
        cache_subdir='models',
        file_hash='253f8cb515780f3b799900260a226db6')
    network.load_weights(*weights_path.rsplit("/", 1))
    network.config["hspace"] = 200
    network.preprocess = vgg_preprocess
    network.postprocess = vgg_decode
    network.information = """
This network architecture comes from the paper:

Very Deep Convolutional Networks for Large-Scale Image Recognition
by Karen Simonyan and Andrew Zisserman.

Their network was trained on the ImageNet challenge dataset.
The dataset contains 32,326 images broken down into 1,000 categories.

The network was trained for 74 epochs on the training data. This typically
took 3 to 4 weeks time on a computer with 4 GPUs. This network's weights were
converted from the original Caffe model into Keras.

Sources:
   * https://arxiv.org/pdf/1409.1556.pdf
   * http://www.robots.ox.ac.uk/~vgg/research/very_deep/
   * http://www.image-net.org/challenges/LSVRC/
      * http://image-net.org/challenges/LSVRC/2014/
      * http://image-net.org/challenges/LSVRC/2014/browse-synsets
"""
    return network

def inceptionv3(*args, **kwargs):
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5'
    WEIGHTS_NAME = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    if "weights" not in kwargs:
        kwargs["weights"] = None
    model = keras.applications.InceptionV3(**kwargs)
    network = import_keras_model(model, "InceptionV3", build_propagate_from_models=False)
    weights_path = get_file(
        WEIGHTS_NAME,
        os.path.join(WEIGHTS_PATH, WEIGHTS_NAME),
        cache_subdir='models',
        file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
    network.load_weights(*weights_path.rsplit("/", 1))
    network.config["hspace"] = 200
    network.preprocess = inceptionv3_preprocess
    network.postprocess = vgg_decode
    network.information = """
This network architecture comes from the paper:

Rethinking the Inception Architecture for Computer Vision

The default input size for this model is 299 x 299.

These weights are released under the [Apache License](https://github.com/tensorflow/models/blob/master/LICENSE).

Sources:

   * http://arxiv.org/abs/1512.00567
"""
    return network
