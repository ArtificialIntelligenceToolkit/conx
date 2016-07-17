# conx

Neural network library in Python built on Theano

Networks implement backpropagation of error algorithm. Networks can have as many hidden layers as you desire.

The network is specified to the constructor by providing sizes. For example, Network(2, 5, 1) specifies a 2-node input layer, 5-unit hidden layer, and a 1-unit output layer.

## Example

Computing XOR via a target function:

```
from conx import Network

inputs = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

def xor(inputs):
    a = inputs[0]
    b = inputs[1]
    return [int((a or b) and not(a and b))]

net = Network(2, 2, 1)
net.set_inputs(inputs)
net.set_target_function(xor)
net.train()
net.test()
```

Given a specified XOR target:

```
from conx import Network
inputs = [[[0, 0], [0]],
          [[0, 1], [1]],
          [[1, 0], [1]],
          [[1, 1], [0]]]
net = Network(2, 2, 1)
net.set_inputs(inputs)
net.train()
net.test()
```

## Install

```python
pip install conx -U
```

## Examples

See the examples folder for additional examples, including handwritten letter recognition of MNIST data.
