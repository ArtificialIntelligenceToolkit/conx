
import keras.backend as K
from keras.activations import (softmax as k_softmax,
                               selu as k_selu)

def softmax(tensor, axis=-1):
    """
    Softmax activation function.

    >>> len(softmax([0.1, 0.1, 0.7, 0.0]))
    4
    """
    return K.eval(k_softmax(K.variable([tensor]), axis))[0].tolist()

def elu(x, alpha=1.0):
    """
    Exponential Linear Unit activation function.

    See: https://arxiv.org/abs/1511.07289v1

    def elu(x):
        if x >= 0:
            return x
        else:
            return alpha * (math.exp(x) - 1.0)

    >>> elu(0.0)
    0.0
    >>> elu(1.0)
    1.0
    >>> elu(0.5, alpha=0.3)
    0.5
    >>> round(elu(-1), 1)
    -0.6
    """
    return K.eval(K.elu(K.variable(x), alpha)).tolist()

def selu(x):
    """
    Scaled Exponential Linear Unit activation function.

    >>> selu(0)
    0.0
    """
    return K.eval(k_selu(K.variable(x))).tolist()

def softplus(x):
    """
    Softplus activation function.

    >>> round(softplus(0), 1)
    0.7
    """
    return K.eval(K.softplus(K.variable(x))).tolist()

def softsign(x):
    """
    Softsign activation function.

    >>> softsign(1)
    0.5
    >>> softsign(-1)
    -0.5
    """
    return K.eval(K.softsign(K.variable(x))).tolist()

def relu(x, alpha=0., max_value=None):
    """
    Rectified Linear Unit activation function.

    >>> relu(1)
    1.0
    >>> relu(-1)
    0.0
    """
    return K.eval(K.relu(K.variable(x), alpha, max_value)).tolist()

def tanh(x):
    """
    Tanh activation function.

    >>> tanh(0)
    0.0
    """
    return K.eval(K.tanh(K.variable(x))).tolist()

def sigmoid(x):
    """
    Sigmoid activation function.

    >>> sigmoid(0)
    0.5
    """
    return K.eval(K.sigmoid(K.variable(x))).tolist()

def hard_sigmoid(x):
    """
    Hard Sigmoid activation function.

    >>> round(hard_sigmoid(-1), 1)
    0.3
    """
    return K.eval(K.hard_sigmoid(K.variable(x))).tolist()

def linear(x):
    """
    Linear activation function.

    >>> linear(1) == 1
    True
    >>> linear(-1) == -1
    True
    """
    return x
