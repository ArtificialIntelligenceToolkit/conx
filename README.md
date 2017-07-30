# conx

Neural Network Library for Cognitive Scientists

Built in Python on Keras

[![CircleCI](https://circleci.com/gh/Calysto/conx/tree/master.svg?style=svg)](https://circleci.com/gh/Calysto/conx/tree/master)

Networks implement neural network algorithms. Networks can have as many hidden layers as you desire.

The network is specified to the constructor by providing sizes. For example, Network(2, 5, 1) specifies a 2-node input layer, 5-unit hidden layer, and a 1-unit output layer.

## Example

Computing XOR via a target function:

```
from conx import Network, SGD

dataset = [[[0, 0], [0]],
          [[0, 1], [1]],
          [[1, 0], [1]],
          [[1, 1], [0]]]

net = Network(2, 2, 1)
net.set_dataset(dataset)
net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))
net.train(2000, report_rate=10, accuracy=1)
net.test()
```

## Install

```python
pip install conx -U
```

To use Theano as the keras backend rather than TensorFlow, edit (or create) `~/.keras/kerson.json` to:

```
{
    "backend": "theano",
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

## Examples

See the examples and notebooks folders for additional examples.
