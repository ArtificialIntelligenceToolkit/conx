
import keras.backend as K
from keras.activations import (softmax as k_softmax,
                               selu as k_selu)

def softmax(tensor, axis=-1):
    """
    Softmax.
    """
    return K.eval(k_softmax(K.variable([tensor]), axis))[0].tolist()

def elu(x, alpha=1.0):
    """
    Exponential Linear Unit.
    """
    return K.eval(K.elu(K.variable(x), alpha)).tolist()

def selu(x):
    """
    Scaled Exponential Linear Unit.
    """
    return K.eval(k_selu(K.variable(x))).tolist()

def softplus(x):
    return K.eval(K.softplus(K.variable(x))).tolist()

def softsign(x):
    return K.eval(K.softsign(K.variable(x))).tolist()

def relu(x, alpha=0., max_value=None):
    return K.eval(K.relu(K.variable(x), alpha, max_value)).tolist()

def tanh(x):
    return K.eval(K.tanh(K.variable(x))).tolist()

def sigmoid(x):
    return K.eval(K.sigmoid(K.variable(x))).tolist()

def hard_sigmoid(x):
    return K.eval(K.hard_sigmoid(K.variable(x))).tolist()

def linear(x):
    return x
