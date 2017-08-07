# conx

## Deep Learning for Simple Folk

Built in Python on Keras.

[![CircleCI](https://circleci.com/gh/Calysto/conx/tree/master.svg?style=svg)](https://circleci.com/gh/Calysto/conx/tree/master) [![codecov](https://codecov.io/gh/Calysto/conx/branch/master/graph/badge.svg)](https://codecov.io/gh/Calysto/conx) [![Documentation Status](https://readthedocs.org/projects/conx/badge/?version=latest)](http://conx.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/conx.svg)](https://badge.fury.io/py/conx)

Read the documentation at [conx.readthedocs.io](http://conx.readthedocs.io/)

Implements Deep Learning neural network algorithms using a simple interface. Built on top of Keras, which can use either TensorFlow or Theano.

The network is specified to the constructor by providing sizes. For example, Network("XOR", 2, 5, 1) specifies a network named "XOR" with a 2-node input layer, 5-unit hidden layer, and a 1-unit output layer.

## Example

Computing XOR via a target function:

```python
from conx import Network, SGD

dataset = [[[0, 0], [0]],
          [[0, 1], [1]],
          [[1, 0], [1]],
          [[1, 1], [0]]]

net = Network("XOR", 2, 5, 1, activation="sigmoid")
net.set_dataset(dataset)
net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))
net.train(2000, report_rate=10, accuracy=1)
net.test()
```

## Install

```shell
pip install conx -U
```

You will need to decide whether to use Theano or Tensorflow. Pick one:

```shell
pip install theano
```

or

```shell
pip install tensorflow
```

To use Theano as the Keras backend rather than TensorFlow, edit (or create) `~/.keras/kerson.json` to:

```json
{
    "backend": "theano",
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

## Examples

See the examples and notebooks folders for additional examples.
