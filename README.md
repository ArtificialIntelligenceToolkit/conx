# conx

## Deep Learning for Simple Folk

Built in Python 3 on Keras 2.

[![CircleCI](https://circleci.com/gh/Calysto/conx/tree/master.svg?style=svg)](https://circleci.com/gh/Calysto/conx/tree/master) [![codecov](https://codecov.io/gh/Calysto/conx/branch/master/graph/badge.svg)](https://codecov.io/gh/Calysto/conx) [![Documentation Status](https://readthedocs.org/projects/conx/badge/?version=latest)](http://conx.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/conx.svg)](https://badge.fury.io/py/conx)

Read the documentation at [conx.readthedocs.io](http://conx.readthedocs.io/)

Ask questions on the mailing list: [conx-users](https://groups.google.com/forum/#!forum/conx-users)

Implements Deep Learning neural network algorithms using a simple interface with easy visualizations and useful analytical. Built on top of Keras, which can use either [TensorFlow](https://www.tensorflow.org/), [Theano](http://www.deeplearning.net/software/theano/), or [CNTK](https://www.cntk.ai/pythondocs/).

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
net.compile(error='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))
net.train(2000, report_rate=10, accuracy=1)
net.test()
```

Creates dynamic, rendered visualizations like this:

<img src="https://raw.githubusercontent.com/Calysto/conx/master/notebooks/network.png" width="500"></img>

## Install

`conx` requires Python3, Keras version 2.0.8 or greater, and some other Python modules that are installed automatically with pip.

**Note**: you may need to use pip3, or admin privileges (eg, sudo), or a user environment.

```bash
pip install conx -U
```

You will need to decide whether to use Theano, TensorFlow, or CNTK. Pick one. See [docs.microsoft.com](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) for installing CNTK on Windows or Linux. All platforms can also install either of the others using pip:

```bash
pip install theano
```

or

```bash
pip install tensorflow
```

On MacOS, you may also need to render the SVG visualizations:

```bash
brew install cairo
```

### Use with Jupyter Notebooks

To use the Network.dashboard() and camera functions, you will need to install and enable `ipywidgets`:

With pip:

``` bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

With conda

``` bash
conda install -c conda-forge ipywidgets
```

Installing **ipywidgets** with conda will also enable the extension for you.

### Changing Keras Backends

To use a Keras backend other than TensorFlow, edit (or create) `~/.keras/kerson.json`, like:

```json
{
    "backend": "theano",
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

## Examples

See the [notebooks folder](https://github.com/Calysto/conx/tree/master/notebooks) and the [documentation](http://conx.readthedocs.io/en/latest/) for additional examples.

## Differences with Keras

1. Conx does not allow targets to be a single value. Keras will
automatically turn single values into a onehot encoded vectors. In
conx, you should just convert such "labels" into their encodings
before training.

