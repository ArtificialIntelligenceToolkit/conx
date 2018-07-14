# Conx Neural Networks

## The On-Ramp to Deep Learning

Built in Python 3 on Keras 2.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Calysto/conx/master?filepath=binder%2Findex.ipynb) [![CircleCI](https://circleci.com/gh/Calysto/conx/tree/master.svg?style=svg)](https://circleci.com/gh/Calysto/conx/tree/master) [![codecov](https://codecov.io/gh/Calysto/conx/branch/master/graph/badge.svg)](https://codecov.io/gh/Calysto/conx) [![Documentation Status](https://readthedocs.org/projects/conx/badge/?version=latest)](http://conx.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/conx.svg)](https://badge.fury.io/py/conx)

Read the documentation at [conx.readthedocs.io](http://conx.readthedocs.io/)

Ask questions on the mailing list: [conx-users](https://groups.google.com/forum/#!forum/conx-users)

Implements Deep Learning neural network algorithms using a simple interface with easy visualizations and useful analytical. Built on top of Keras, which can use either [TensorFlow](https://www.tensorflow.org/), [Theano](http://www.deeplearning.net/software/theano/), or [CNTK](https://www.cntk.ai/pythondocs/).

The network is specified to the constructor by providing sizes. For example, Network("XOR", 2, 5, 1) specifies a network named "XOR" with a 2-node input layer, 5-unit hidden layer, and a 1-unit output layer.

Computing XOR via a target function:

```python
import conx as cx

dataset = [[[0, 0], [0]],
           [[0, 1], [1]],
           [[1, 0], [1]],
           [[1, 1], [0]]]

net = cx.Network("XOR", 2, 5, 1, activation="sigmoid")
net.dataset.load(dataset)
net.compile(error='mean_squared_error',
            optimizer="sgd", lr=0.3, momentum=0.9)
net.train(2000, report_rate=10, accuracy=1.0)
net.test(show=True)
```

Creates dynamic, rendered visualizations like this:

<img src="https://raw.githubusercontent.com/Calysto/conx-notebooks/master/network.png" width="500"></img>

## Examples

See [conx-notebooks](https://github.com/Calysto/conx-notebooks/blob/master/00_Index.ipynb) and the [documentation](http://conx.readthedocs.io/en/latest/) for additional examples.

## Installation

See [How To Run Conx](https://github.com/Calysto/conx-notebooks/tree/master/HowToRun#how-to-run-conx)
to see options on running virtual machines, in the cloud, and personal
installation.
