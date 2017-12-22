# conx - a neural network library
#
# Copyright (c) Douglas S. Blank <doug.blank@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301  USA

import numbers
import PIL
import base64
import io
import numpy as np
from keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt

#------------------------------------------------------------------------
# configuration constants

AVAILABLE_COLORMAPS = sorted(list(plt.cm.cmap_d.keys()))
CURRENT_COLORMAP = "seismic_r"
ERROR_COLORMAP = "seismic_r"

def set_colormap(s):
    global CURRENT_COLORMAP
    assert s in AVAILABLE_COLORMAPS, "Unknown colormap: %s" % s
    CURRENT_COLORMAP = s

def set_error_colormap(s):
    global ERROR_COLORMAP
    assert s in AVAILABLE_COLORMAPS, "Unknown colormap: %s" % s
    ERROR_COLORMAP = s

def get_error_colormap():
    return ERROR_COLORMAP

def get_colormap():
    return CURRENT_COLORMAP

#------------------------------------------------------------------------
# utility functions

def choice(seq=None, p=None):
    """
    Get a random choice from sequence, optionally given a probability
    distribution.

    >>> choice(1)
    0

    >>> choice([42])
    42

    >>> choice("abcde", p=[0, 1, 0, 0, 0])
    'b'

    >>> choice(p=[0, 0, 1, 0, 0])
    2
    """
    if seq is None and p is None:
        raise Exception("seq and p can't both be None")
    elif seq is None:
        seq = len(p)
    ## Allow seq to be a number:
    if isinstance(seq, numbers.Real):
        seq = range(int(seq))
    if p is not None:
        # Make sure that it sums to 1.0
        # or else np.random.choice will crash
        total = sum(p)
        p = [item/total for item in p]
    pick = np.random.choice(len(seq), 1, p=p)[0]
    return seq[pick]

def frange(start, stop=None, step=1.0):
    """
    Like range(), but with floats.

    May not be exactly correct due to rounding issues.
    """
    if stop is None:
        stop = start
        start = 0.0
    return np.arange(start, stop, step)

def argmax(seq):
    """
    Find the index of the maximum value in seq.
    """
    return np.argmax(seq)

def image2array(image):
    """
    Convert an image filename or PIL.Image into a numpy array.
    """
    if isinstance(image, str):
        image = PIL.image.open(image)
    return np.array(image, "float32") / 255.0

def array2image(array, scale=1.0):
    """
    Convert an array (with shape) to a PIL. Image.
    """
    array = np.array(array) # let's make sure
    array =  (array * 255).astype("uint8")
    image = PIL.Image.fromarray(array)
    if scale != 1.0:
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
    return image

def onehot(i, width):
    """
    >>> onehot(0, 5)
    [1, 0, 0, 0, 0]

    >>> onehot(3, 5)
    [0, 0, 0, 1, 0]
    """
    v = [0] * width
    v[i] = 1
    return v

def binary(i, width):
    """
    >>> binary(0, 5)
    [0, 0, 0, 0, 0]

    >>> binary(15, 4)
    [1, 1, 1, 1]

    >>> binary(14, 4)
    [1, 1, 1, 0]
    """
    bs = bin(i)[2:]
    bs = ("0" * width + bs)[-width:]
    b = [int(c) for c in bs]
    return b

def find_path(net, start_layer, end_layer):
    """
    Given a conx network, a start layer, and an ending
    layer, find the path between them.
    """
    ## FIXME: doesn't work with merges.
    queue = [start_layer]
    get_to = {} # get to key from value
    found = False
    while queue:
        current = queue.pop(0) # BFS
        if current == end_layer:
            found = True
            break
        else:
            expand = [layer.name for layer in net[current].outgoing_connections]
            for layer_name in expand:
                if layer_name not in get_to:
                    get_to[layer_name] = current
                    queue.append(layer_name)
    if found:
        retval = []
        while current != start_layer:
            retval.append(net[current])
            current = get_to[current]
        return reversed(retval)

def topological_sort(net, layers):
    """
    Given a conx network and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    ## Initilize all:
    for layer in net.layers:
        layer.visited = False
    stack = []
    # Only track and sort these:
    for layer in layers:
        if not layer.visited:
            visit(layer, stack)
    stack.reverse()
    return stack

def visit(layer, stack):
    """
    Utility function for topological_sort.
    """
    layer.visited = True
    for outgoing_layer in layer.outgoing_connections:
        if not outgoing_layer.visited:
            visit(outgoing_layer, stack)
    stack.append(layer)

def autoname(index, sizes):
    """
    Given an index and list of sizes, return a
    name for the layer.
    """
    if index == 0:
        n = "input"
    elif index == sizes - 1:
        n = "output"
    elif sizes == 3:
        n = "hidden"
    else:
        n = "hidden%d" % index
    return n

def valid_shape(x):
    """
    Is this a valid shape for Keras layers?

    >>> valid_shape(1)
    True
    >>> valid_shape(None)
    True
    >>> valid_shape((1,))
    True
    >>> valid_shape((None, ))
    True
    """
    return ((isinstance(x, numbers.Integral) and (x > 0)) or
            (x is None) or
            (isinstance(x, (tuple, list)) and
             (len(x) > 0) and
             all([((isinstance(n, numbers.Integral) and (n > 0)) or
                   (n is None)) for n in x])))

def valid_vshape(x):
    """
    Is this a valid shape (i.e., size) to display vectors using PIL?
    """
    # vshape must be a single int or a 2-dimensional tuple
    return valid_shape(x) and (isinstance(x, numbers.Integral) or len(x) == 2)

def rescale_numpy_array(a, old_range, new_range, new_dtype, truncate=False):
    """
    Given a vector, old min/max, a new min/max and a numpy type,
    create a new vector scaling the old values.
    """
    assert isinstance(old_range, (tuple, list)) and isinstance(new_range, (tuple, list))
    old_min, old_max = old_range
    if a.min() < old_min or a.max() > old_max:
        if truncate:
            a = np.clip(a, old_min, old_max)
        else:
            raise Exception('array values are outside range %s' % (old_range,))
    new_min, new_max = new_range
    old_delta = old_max - old_min
    new_delta = new_max - new_min
    if old_delta == 0:
        return ((a - old_min) + (new_min + new_max)/2).astype(new_dtype)
    else:
        return (new_min + (a - old_min)*new_delta/old_delta).astype(new_dtype)

def uri_to_image(image_str, width=320, height=240):
    header, image_b64 = image_str.split(",")
    image_binary = base64.b64decode(image_b64)
    image = PIL.Image.open(io.BytesIO(image_binary)).resize((width, height))
    return image

def get_device():
    """
    Returns 'cpu' or 'gpu' indicating which device
    the system will use.
    """
    import keras.backend as K
    if K._BACKEND == "theano":
        from theano import function, tensor as T, shared, config
        x = shared(np.array([1.0], config.floatX))
        f = function([], T.exp(x))
        if np.any([isinstance(x.op, T.Elemwise) and
                   ('Gpu' not in type(x.op).__name__)
                   for x in f.maker.fgraph.toposort()]):
            return "cpu"
        else:
            return "gpu"
    elif K._BACKEND == "tensorflow":
        from tensorflow.python.client import device_lib
        devices = [dev.name for dev in device_lib.list_local_devices()]
        ## ['/device:CPU:0', '/device:GPU:0']
        return "gpu" if any(["GPU" in dev for dev in devices]) else "cpu"
    else:
        return "unknown"

def import_keras_model(model, network_name):
    """
    """
    from .network import Network
    import inspect
    import conx
    network = Network(network_name)
    network.model = model
    conx_layers = {name: layer for (name,layer)
                   in inspect.getmembers(conx.layers, inspect.isclass)}
    # First, make all of the conx layers:
    for layer in model.layers:
        clayer_class = conx_layers[layer.__class__.__name__ + "Layer"]
        if clayer_class.__name__ == "InputLayerLayer":
            clayer = conx.layers.InputLayer(layer.name, None)
            #clayer.make_input_layer_k = lambda layer=layer: layer
            clayer.shape = None
            clayer.params["batch_shape"] = layer.get_config()["batch_input_shape"]
            #clayer.params = layer.get_config()
            clayer.k = clayer.make_input_layer_k()
            clayer.keras_layer = clayer.k
        else:
            clayer = clayer_class(**layer.get_config())
            clayer.k = layer
            clayer.keras_layer = layer
        network.add(clayer)
    # Next, connect them up:
    for layer_from in model.layers:
        for node in layer.outbound_nodes:
            network.connect(layer_from, node.outbound_layer.name)
            print("connecting:", layer_from, node.outbound_layer.name)
    # Connect them all up, and set input banks:
    network.connect()
    for clayer in network.layers:
        clayer.input_names = network.input_bank_order
    # Finally, make the internal models:
    for clayer in network.layers:
        ## FIXME: the appropriate inputs:
        if clayer.kind() != "input":
            clayer.model = keras.models.Model(inputs=model.layers[0].input,
                                              outputs=clayer.keras_layer.output)
    return network

def plot_f(f, frange=(-1, 1, .1), symbol="o-"):
    """
    Plot a function.
    """
    xs = np.arange(*frange)
    ys = [f(x) for x in xs]
    plt.plot(xs, ys, symbol)
    plt.show()
    #bytes = io.BytesIO()
    #plt.savefig(bytes, format='svg')
    #svg = bytes.getvalue()
    #plt.close(fig)
    #return SVG(svg.decode())

def plot(lines, width=8.0, height=4.0, xlabel="", ylabel="",
         ymin=None, xmin=None, ymax=None, xmax=None, interactive=True):
    """
    SVG(plot([["Error", "+", [1, 2, 4, 6, 1, 2, 3]]],
             ylabel="error",
             xlabel="hello"))
    """
    if plt is None:
        raise Exception("matplotlib was not loaded")
    plt.rcParams['figure.figsize'] = (width, height)
    fig, ax = plt.subplots()
    for (label, symbol, data) in lines:
        kwargs = {}
        args = [data]
        if label:
            kwargs["label"] = label
        if symbol:
            args.append(symbol)
        plt.plot(*args, **kwargs)
    if any([line[0] for line in lines]):
        plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if ymin is not None:
        plt.ylim(ymin=ymin)
    if ymax is not None:
        plt.ylim(ymax=ymax)
    if xmin is not None:
        plt.xlim(xmin=xmin)
    if xmax is not None:
        plt.xlim(xmax=xmax)
    if interactive:
        plt.show(block=False)
    else:
        from IPython.display import SVG
        bytes = io.BytesIO()
        plt.savefig(bytes, format='svg')
        img_bytes = bytes.getvalue()
        plt.close(fig)
        return SVG(img_bytes.decode())

def scatter(lines, width=8.0, height=4.0, xlabel="", ylabel="", title="",
            ymin=None, xmin=None, ymax=None, xmax=None, interactive=True):
    if plt is None:
        raise Exception("matplotlib was not loaded")
    plt.rcParams['figure.figsize'] = (width, height)
    fig, ax = plt.subplots()
    for (label, symbol, pairs) in lines:
        kwargs = {}
        args = []
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        if label:
            kwargs["label"] = label
        if symbol:
            args.append(symbol)
        plt.plot(xs, ys, *args, **kwargs)
    if any([line[0] for line in lines]):
        plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ymin is not None:
        plt.ylim(ymin=ymin)
    if ymax is not None:
        plt.ylim(ymax=ymax)
    if xmin is not None:
        plt.xlim(xmin=xmin)
    if xmax is not None:
        plt.xlim(xmax=xmax)
    if title:
        plt.title(title)
    if interactive:
        plt.show(block=False)
    else:
        from IPython.display import SVG
        bytes = io.BytesIO()
        plt.savefig(bytes, format='svg')
        img_bytes = bytes.getvalue()
        plt.close(fig)
        return SVG(img_bytes.decode())
