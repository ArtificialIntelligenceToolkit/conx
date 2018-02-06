# Conx Neural Networks

## The On-Ramp to Deep Learning

Built in Python 3 on Keras 2.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Calysto/conx/master?filepath=binder%2Findex.ipynb) [![CircleCI](https://circleci.com/gh/Calysto/conx/tree/master.svg?style=svg)](https://circleci.com/gh/Calysto/conx/tree/master) [![codecov](https://codecov.io/gh/Calysto/conx/branch/master/graph/badge.svg)](https://codecov.io/gh/Calysto/conx) [![Documentation Status](https://readthedocs.org/projects/conx/badge/?version=latest)](http://conx.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/conx.svg)](https://badge.fury.io/py/conx)

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

<img src="https://raw.githubusercontent.com/Calysto/conx-notebooks/master/network.png" width="500"></img>

## Install

Rather than installing conx, consider using our [mybinder](https://mybinder.org/v2/gh/Calysto/conx/master?filepath=binder%2Findex.ipynb) in-the-cloud version. Availability may be limited due to demand.

`conx` requires Python3, Keras version 2.0.8 or greater, and some other Python modules that are installed automatically with pip.

On Linux, you may need to install `libffi` and `libffi-dev` in order to render layers for the network display. If you attempt to display a network and it appears empty, or if you attempt to network.propagate_to_image() and it gives a PIL error, you need these libraries.

On Ubuntu or other Debian-based system:

```bash
sudo apt install libffi-dev libffi6
```
Next, we use `pip` to install the Python packages. 

**Note**: you may need to use `pip3`, or admin privileges (eg, sudo), or install into a user environment.

```bash
pip install conx -U
```

You will need to decide whether to use Theano, TensorFlow, or CNTK. Pick one. See [docs.microsoft.com](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine) for installing CNTK on Windows or Linux. All platforms can also install either of the others using pip:

```bash
pip install theano
```

**or**

```bash
pip install tensorflow
```

On MacOS, you may also need to render the SVG visualizations:

```bash
brew install cairo
```

To make MP4 movies, you will need the `ffmpeg` executable installed and available on your default path.

On MacOS, you could use:

```bash
brew install ffmpeg
```

On Windows:

See, for example, https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg

On Linux:

```bash
sudo apt install ffmpeg
# or perhaps:
sudo yum install ffmpeg
```

## Use with Jupyter Notebooks

To use the Network.dashboard() and camera functions, you will need to enable `ipywidgets`:

``` bash
jupyter nbextension enable --py widgetsnbextension
```

If you install via conda, then it will already be enabled:

``` bash
conda install -c conda-forge ipywidgets
```

### Setting the Keras Backend

To use a Keras backend other than TensorFlow, edit (or create) `~/.keras/kerson.json`, like:

```json
{
    "backend": "theano",
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

## Troubleshooting

1. If you have a problem after installing matplotlib with pip, and you already have matplotlib installed (say, with apt) you may want to remove the apt-installed version of matplotlib.

## Examples

See the [notebooks folder](https://github.com/Calysto/conx/tree/master/notebooks) and the [documentation](http://conx.readthedocs.io/en/latest/) for additional examples.

