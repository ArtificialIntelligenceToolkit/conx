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
import itertools
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
    """
    Set the global colormap for displaying all network activations.
    """
    global CURRENT_COLORMAP
    assert s in AVAILABLE_COLORMAPS, "Unknown colormap: %s" % s
    CURRENT_COLORMAP = s

def set_error_colormap(s):
    """
    Set the error color map for display error values.
    """
    global ERROR_COLORMAP
    assert s in AVAILABLE_COLORMAPS, "Unknown colormap: %s" % s
    ERROR_COLORMAP = s

def get_error_colormap():
    """
    Get the global error colormap.
    """
    return ERROR_COLORMAP

def get_colormap():
    """
    Get the global colormap.
    """
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

    >>> len(frange(-1, 1, .1))
    20
    """
    if stop is None:
        stop = start
        start = 0.0
    return np.arange(start, stop, step)

def argmax(seq):
    """
    Find the index of the maximum value in seq.

    >>> argmax([0.1, 0.2, 0.3, 0.1])
    2
    """
    return np.argmax(seq)

def image2array(image):
    """
    Convert an image filename or PIL.Image into a numpy array.
    """
    if isinstance(image, str):
        image = PIL.Image.open(image)
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

    >>> autoname(0, sizes=4)
    'input'
    >>> autoname(1, sizes=4)
    'hidden1'
    >>> autoname(2, sizes=4)
    'hidden2'
    >>> autoname(3, sizes=4)
    'output'
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
    Given a numpy array, old min/max, a new min/max and a numpy type,
    create a new numpy array that scales the old values into the new_range.

    >>> import numpy as np
    >>> new_array = rescale_numpy_array(np.array([0.1, 0.2, 0.3]), (0, 1), (0.5, 1.), float)
    >>> ", ".join(["%.2f" % v for v in new_array])
    '0.55, 0.60, 0.65'
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
    """
    Given an URI, return an image.
    """
    header, image_b64 = image_str.split(",")
    image_binary = base64.b64decode(image_b64)
    image = PIL.Image.open(io.BytesIO(image_binary)).resize((width, height))
    return image

def get_device():
    """
    Returns 'cpu' or 'gpu' indicating which device
    the system will use.

    >>> get_device() in ["gpu", "cpu"]
    True
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
    Import a keras model into conx.

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

def movie(function, movie_name="movie.gif", play_range=None,
          loop=0, optimize=True, duration=100, return_it=True):
    """
    Make a movie from a function.

    function has signature: function(index) and should return
    a PIL.Image.
    """
    from IPython.display import HTML
    frames = []
    for index in range(*play_range):
        frames.append(function(index))
    if frames:
        frames[0].save(movie_name, save_all=True, append_images=frames[1:],
                       optimize=optimize, loop=loop, duration=duration)
        if return_it:
            data = open(movie_name, "rb").read()
            data = base64.b64encode(data)
            if not isinstance(data, str):
                data = data.decode("latin1")
            return HTML("""<img src="data:image/gif;base64,{0}" alt="{1}">""".format(html.escape(data), movie_name))

def plot_f(f, frange=(-1, 1, .1), symbol="o-", xlabel="", ylabel="", title="",
           interactive=True, format='svg'):
    """
    Plot a function.

    >>> plot_f(lambda x: x, frange=(-1, 1, .1), interactive=False)
    <IPython.core.display.SVG object>
    """
    xs = np.arange(*frange)
    ys = [f(x) for x in xs]
    fig, ax = plt.subplots()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.plot(xs, ys, symbol)
    if interactive:
        plt.show(block=False)
    else:
        from IPython.display import SVG
        bytes = io.BytesIO()
        if format == "svg":
            plt.savefig(bytes, format="svg")
            plt.close(fig)
            img_bytes = bytes.getvalue()
            return SVG(img_bytes.decode())
        elif format == "pil":
            plt.savefig(bytes, format="png")
            plt.close(fig)
            bytes.seek(0)
            pil_image = PIL.Image.open(bytes)
            return pil_image
        else:
            raise Exception("format must be 'svg' or 'pil'")

def plot(data=[], width=8.0, height=4.0, xlabel="", ylabel="", title="",
         label="", symbols=None, default_symbol=None, ymin=None, xmin=None, ymax=None, xmax=None,
         interactive=True, format='svg', xs=None):
    """
    >>> p = plot(["Error", [1, 2, 4, 6, 1, 2, 3]],
    ...           ylabel="error",
    ...           xlabel="hello", interactive=False)
    >>> p
    <IPython.core.display.SVG object>
    >>> p = plot([["Error", [1, 2, 4, 6, 1, 2, 3]]],
    ...           ylabel="error",
    ...           xlabel="hello", interactive=False)
    >>> p
    <IPython.core.display.SVG object>
    """
    if plt is None:
        raise Exception("matplotlib was not loaded")
    set_plt_param('figure.figsize', (width, height))
    fig, ax = plt.subplots()
    if len(data) == 2 and isinstance(data[0], str):
        data = [data]
    for (data_label, vectors) in data:
        kwargs = {}
        if xs is not None:
            args = [xs, vectors]
        else:
            args = [vectors]
        if label: ## override
            kwargs["label"] = label
        elif data_label:
            kwargs["label"] = data_label
        symbol = get_symbol(kwargs.get("label", None), symbols, default_symbol)
        if symbol:
            args.append(symbol)
        plt.plot(*args, **kwargs)
    if any([line[0] for line in data]):
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
        result = None
    else:
        from IPython.display import SVG
        bytes = io.BytesIO()
        if format == "svg":
            plt.savefig(bytes, format="svg")
            plt.close(fig)
            img_bytes = bytes.getvalue()
            result = SVG(img_bytes.decode())
        elif format == "pil":
            plt.savefig(bytes, format="png")
            plt.close(fig)
            bytes.seek(0)
            pil_image = PIL.Image.open(bytes)
            result = pil_image
        else:
            raise Exception("format must be 'svg' or 'pil'")
    reset_plt_param('figure.figsize')
    return result

def heatmap(function_or_matrix, in_range=(0,1), width=8.0, height=4.0, xlabel="", ylabel="", title="",
            resolution=None, out_min=None, out_max=None, colormap=None, interactive=True, format='svg'):
    """
    >>> import math
    >>> def function(x, y):
    ...     return math.sqrt(x ** 2 + y ** 2)
    >>> hm = heatmap(function,
    ...              interactive=False)
    >>> hm
    <IPython.core.display.SVG object>
    """
    in_min, in_max = in_range
    if plt is None:
        raise Exception("matplotlib was not loaded")
    set_plt_param('figure.figsize', (width, height))
    fig, ax = plt.subplots()
    if callable(function_or_matrix):
        function = function_or_matrix
        if resolution is None:
            resolution = (in_max - in_min) / 50  # 50x50 pixels by default
        xmin, xmax, xstep = in_min, in_max, resolution
        ymin, ymax, ystep = in_min, in_max, resolution
        xspan = xmax - xmin
        yspan = ymax - ymin
        xpixels = int(xspan/xstep)+1
        ypixels = int(yspan/ystep)+1
        mat = np.zeros((ypixels, xpixels))
        for row in range(ypixels):
            for col in range(xpixels):
                # (x,y) corresponds to lower left corner point of pixel
                x = xmin + xstep * col
                y = ymin + ystep * row
                mat[row,col] = function(x, y)
    else:
        mat = np.array(function_or_matrix)
    if out_min is None:
        out_min = mat.min()
    if out_max is None:
        out_max = mat.max()
    if colormap is None:
        colormap = get_colormap()
    axim = ax.imshow(mat, origin='lower', cmap=colormap, vmin=out_min, vmax=out_max)
    fig.colorbar(axim)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if interactive:
        plt.show(block=False)
        result = None
    else:
        from IPython.display import SVG
        bytes = io.BytesIO()
        if format == "svg":
            plt.savefig(bytes, format="svg")
            plt.close(fig)
            img_bytes = bytes.getvalue()
            result = SVG(img_bytes.decode())
        elif format == "pil":
            plt.savefig(bytes, format="png")
            plt.close(fig)
            bytes.seek(0)
            pil_image = PIL.Image.open(bytes)
            result = pil_image
        else:
            raise Exception("format must be 'svg' or 'pil'")
    reset_plt_param('figure.figsize')
    return result

CACHE_PARAMS = {}

def set_plt_param(setting, value):
    """
    Set the matplotlib setting to a new value.

    >>> original = plt.rcParams["font.size"]
    >>> set_plt_param("font.size", 8.0)
    >>> original == plt.rcParams["font.size"]
    False
    >>> 8.0 == plt.rcParams["font.size"]
    True
    >>> reset_plt_param("font.size")
    >>> original == plt.rcParams["font.size"]
    True
    """
    CACHE_PARAMS[setting] = plt.rcParams[setting]
    plt.rcParams[setting] = value

def reset_plt_param(setting):
    """
    Reset the matplotlib setting to its default.

    >>> original = plt.rcParams["figure.figsize"]
    >>> set_plt_param("figure.figsize", [5, 5])
    >>> original == plt.rcParams["figure.figsize"]
    False
    >>> [5, 5] == plt.rcParams["figure.figsize"]
    True
    >>> reset_plt_param("figure.figsize")
    >>> original == plt.rcParams["figure.figsize"]
    True
    """
    plt.rcParams[setting] = CACHE_PARAMS[setting]

def scatter(data=[], width=6.0, height=6.0, xlabel="", ylabel="", title="", label="",
            symbols=None, default_symbol=None, ymin=None, xmin=None, ymax=None, xmax=None,
            interactive=True, format='svg'):
    """
    Create a scatter plot with series of (x,y) data.

    >>> scatter(["Test 1", [(0,4), (2,3), (1,2)]], interactive=False)
    <IPython.core.display.SVG object>
    """
    if plt is None:
        raise Exception("matplotlib was not loaded")
    set_plt_param('figure.figsize', (width, height))
    fig, ax = plt.subplots()
    if len(data) == 2 and isinstance(data[0], str):
        data = [data]
    for (data_label, vectors) in data:
        kwargs = {}
        args = []
        xs = [vector[0] for vector in vectors]
        ys = [vector[1] for vector in vectors]
        if label: ## override
            kwargs["label"] = label
        elif data_label:
            kwargs["label"] = data_label
        symbol = get_symbol(kwargs.get("label", None), symbols, default_symbol)
        if symbol:
            args.append(symbol)
        plt.plot(xs, ys, *args, **kwargs)
    if any([line[0] for line in data]):
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
        result = None
    else:
        from IPython.display import SVG
        bytes = io.BytesIO()
        if format == "svg":
            plt.savefig(bytes, format="svg")
            plt.close(fig)
            img_bytes = bytes.getvalue()
            result = SVG(img_bytes.decode())
        elif format == "pil":
            plt.savefig(bytes, format="png")
            plt.close(fig)
            bytes.seek(0)
            pil_image = PIL.Image.open(bytes)
            result = pil_image
        else:
            raise Exception("format must be 'svg' or 'pil'")
    reset_plt_param('figure.figsize')
    return result

class PCA():
    """
    Compute the Prinicpal Component Analysis for the points
    in a multi-dimensional space.

    Example:
        >>> data = [
        ...         [0.00, 0.00, 0.00],
        ...         [0.25, 0.25, 0.25],
        ...         [0.50, 0.50, 0.50],
        ...         [0.75, 0.75, 0.75],
        ...         [1.00, 1.00, 1.00],
        ... ]
        >>> pca = PCA(data)
        >>> new_data = pca.transform(data)
        >>> len(new_data)
        5
    """
    def __init__(self, states, dim=2, solver="randomized"):
        from sklearn.decomposition import PCA
        self.dim = dim
        self.solver = solver
        self.pca = PCA(n_components=self.dim, svd_solver=self.solver)
        self.pca.fit(states)
        ## Now, compute and cache stats about this space:
        self.mins = {}
        self.maxs = {}
        states_pca = self.pca.transform(states)
        for i in range(self.dim):
            self.mins[i] = min([state[i] for state in states_pca])
            self.maxs[i] = max([state[i] for state in states_pca])

    def transform_one(self, vector, scale=False):
        """
        Transform a vector into the PCA of the trained states.

        >>> from conx import Network
        >>> net = Network("Example", 2, 2, 1)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.load([
        ...        [[0, 0], [0], "0"],
        ...        [[0, 1], [1], "1"],
        ...        [[1, 0], [1], "1"],
        ...        [[1, 1], [0], "0"],
        ... ])
        >>> states = [net.propagate_to("hidden", input) for input in net.dataset.inputs]
        >>> pca = PCA(states)
        >>> new_state = pca.transform_one(states[0])
        >>> len(new_state)
        2
        """
        vector_prime = self.pca.transform([vector])[0]
        if scale:
            return self.scale(vector_prime)
        else:
            return vector_prime

    def transform(self, vectors, scale=False):
        """
        >>> from conx import Network
        >>> net = Network("Example", 2, 2, 1)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.load([
        ...        [[0, 0], [0], "0"],
        ...        [[0, 1], [1], "1"],
        ...        [[1, 0], [1], "1"],
        ...        [[1, 1], [0], "0"],
        ... ])
        >>> states = [net.propagate_to("hidden", input) for input in net.dataset.inputs]
        >>> pca = PCA(states)
        >>> new_states = pca.transform(states)
        >>> len(new_states)
        4
        """
        vectors_prime = self.pca.transform(vectors)
        if scale:
            return np.array([self.scale(v) for v in vectors])
        else:
            return vectors_prime

    def transform_network_bank(self, network, bank, label_index=0, tolerance=None, test=True,
                               scale=False):
        """
        >>> from conx import Network
        >>> net = Network("Example", 2, 2, 1)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.load([
        ...        [[0, 0], [0], "0"],
        ...        [[0, 1], [1], "1"],
        ...        [[1, 0], [1], "1"],
        ...        [[1, 1], [0], "0"],
        ... ])
        >>> states = [net.propagate_to("hidden", input) for input in net.dataset.inputs]
        >>> pca = PCA(states)
        >>> results = pca.transform_network_bank(net, "hidden")
        >>> sum([len(vectors) for (label, vectors) in results["data"]])
        4
        >>> "xmin" in results
        True
        >>> "xmax" in results
        True
        >>> "ymin" in results
        True
        >>> "ymax" in results
        True
        """
        categories = {}
        if test:
            tolerance = tolerance if tolerance is not None else network.tolerance
            if len(network.dataset.inputs) == 0:
                raise Exception("nothing to test")
            inputs = network.dataset._inputs
            targets = network.dataset._targets
            results = network._test(inputs, targets, "train dataset", tolerance=tolerance,
                                    show_inputs=False, show_outputs=False, filter="all",
                                    interactive=False)
        for i in range(len(network.dataset.inputs)):
            label = network.dataset._labels[label_index][i]
            input_vector = network.dataset.inputs[i]
            if test:
                category = "%s (%s)" % (label, "correct" if results[i] else "wrong")
            else:
                category = label
            hid = network.propagate_to(bank, input_vector, visualize=False)
            hid_prime = self.transform_one(hid, scale)
            if category not in categories:
                categories[category] = []
            categories[category].append(hid_prime)
        return {
            "data": sorted(categories.items()),
            "xmin": self.mins[0],
            "xmax": self.maxs[0],
            "ymin": self.mins[1],
            "ymax": self.maxs[1],
        }

    def scale(self, ovector):
        """
        Scale a transformed vector to (0, 1).
        """
        vector = np.array(ovector)
        for i in range(len(vector)):
            span = (self.maxs[i] - self.mins[i])
            vector[i] = (vector[i] - self.mins[i]) / span
        return vector

def get_symbol(label: str, symbols: dict=None, default='o') -> str:
    """
    Get a matplotlib symbol from a label.

    Possible shape symbols:

        * '-'	solid line style
        * '--'	dashed line style
        * '-.'	dash-dot line style
        * ':'	dotted line style
        * '.'	point marker
        * ','	pixel marker
        * 'o'	circle marker
        * 'v'	triangle_down marker
        * '^'	triangle_up marker
        * '<'	triangle_left marker
        * '>'	triangle_right marker
        * '1'	tri_down marker
        * '2'	tri_up marker
        * '3'	tri_left marker
        * '4'	tri_right marker
        * 's'	square marker
        * 'p'	pentagon marker
        * '*'	star marker
        * 'h'	hexagon1 marker
        * 'H'	hexagon2 marker
        * '+'	plus marker
        * 'x'	x marker
        * 'D'	diamond marker
        * 'd'	thin_diamond marker
        * '|'	vline marker
        * '_'	hline marker

    In addition, the shape symbol can be preceded by the following color abbreviations:

        * ‘b’	blue
        * ‘g’	green
        * ‘r’	red
        * ‘c’	cyan
        * ‘m’	magenta
        * ‘y’	yellow
        * ‘k’	black
        * ‘w’	white

    Examples:
        >>> get_symbol("Apple")
        'o'
        >>> get_symbol("Apple", {'Apple': 'x'})
        'x'
        >>> get_symbol("Banana", {'Apple': 'x'})
        'o'
    """
    if symbols is None:
        return default
    else:
        return symbols.get(label, default)

def atype(dtype):
    """
    Given a numpy dtype, return the associated Python type.
    If unable to determine, just return the dtype.kind code.

    >>> atype(np.float64(23).dtype)
    <class 'numbers.Number'>
    """
    if dtype.kind in ["i", "f", "u"]:
        return numbers.Number
    elif dtype.kind in ["U", "S"]:
        return str
    else:
        return dtype.kind

def format_collapse(ttype, dims):
    """
    Given a type and a tuple of dimensions, return a struct of
    [[[ttype, dims[-1]], dims[-2]], ...]

    >>> format_collapse(int, (1, 2, 3))
    [[[<class 'int'>, 3], 2], 1]
    """
    if len(dims) == 1:
        return [ttype, dims[0]]
    else:
        return format_collapse([ttype, dims[-1]], dims[:-1])

def types(item):
    """
    Get the types of (possibly) nested list(s), and collapse
    if possible.

    >>> types(0)
    <class 'numbers.Number'>

    >>> types([0, 1, 2])
    [<class 'numbers.Number'>, 3]
    """
    try:
        length = len(item)
    except:
        return (numbers.Number
                if isinstance(item, numbers.Number)
                else type(item))
    if isinstance(item, str):
        return str
    elif length == 0:
        return [None, 0]
    array = None
    try:
        array = np.asarray(item)
    except:
        pass
    if array is None or array.dtype == object:
        return [types(x) for x in item]
    else:
        dtype = array.dtype ## can be many things!
        return format_collapse(atype(dtype), array.shape)

def all_same(iterator):
    """
    Are there more than one item, and all the same?

    >>> all_same([int, int, int])
    True

    >>> all_same([int, float, int])
    False
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return False
    return all([first == rest for rest in iterator])

def is_collapsed(item):
    """
    Is this a collapsed item?

    >>> is_collapsed([int, 3])
    True

    >>> is_collapsed([int, int, int])
    False
    """
    try:
        return (len(item) == 2 and
                isinstance(item[0], (type, np.dtype)) and
                isinstance(item[1], numbers.Number))
    except:
        return False

def collapse(item):
    """
    For any repeated structure, return [struct, count].

    >>> collapse([[int, int, int], [float, float]])
    [[<class 'int'>, 3], [<class 'float'>, 2]]
    """
    if is_collapsed(item):
        return item
    try:
        length = len(item)
    except:
        return item
    items = [collapse(i) for i in item]
    if all_same(items):
        return [items[0], length]
    else:
        return items

def get_form(item):
    """
    First, get the types of all items, and then collapse
    repeated structures.

    >>> get_form([1, [2, 5, 6], 3])
    [<class 'numbers.Number'>, [<class 'numbers.Number'>, 3], <class 'numbers.Number'>]
    """
    return collapse(types(item))

def get_shape(form):
    """
    Given a form, format it in [type, dimension] format.

    >>> get_shape(get_form([[0.00], [0.00]]))
    (<class 'numbers.Number'>, [2, 1])
    """
    if (isinstance(form, list) and
        len(form) == 2 and
        isinstance(form[1], numbers.Number)):
        ## Is it [type, count]
        if isinstance(form[0], (np.dtype, numbers.Number, type)):
            return (form[0], [form[1]])
        else:
            f = get_shape(form[0])
            if isinstance(f, tuple): ## FIXME: and same
                return (f[0], [form[1]] + f[1])
            else:
                return ([get_shape(f) for f in form], [len(form)])
    elif isinstance(form, list):
        return ([get_shape(x) for x in form], [len(form)])
    else:
        return (form, [0]) # scalar

def shape(item):
    """
    Shortcut for get_shape(get_form(item)).

    >>> shape([1])
    (1,)
    >>> shape([1, 2])
    (2,)
    >>> shape([[1, 2, 3], [4, 5, 6]])
    (2, 3)
    """
    s = get_shape(get_form(item))
    if (isinstance(s, tuple) and
        len(s) == 2 and
        all([isinstance(v, (np.dtype, numbers.Number, type)) for v in s[1]])):
        return tuple(s[1])
    else:
        return [tuple(v[1]) for v in s]

class Experiment():
    """
    Run a series of experiments.

    function() should take any options, and return a network.

    >>> from conx import Network
    >>> def function(optimizer, activation, **options):
    ...     net = Network("XOR", 2, 2, 1, activation=activation)
    ...     net.compile(error="mse", optimizer=optimizer)
    ...     net.dataset.add_by_function(2, (0, 4), "binary", lambda i,v: [int(sum(v) == len(v))])
    ...     net.train(report_rate=100, verbose=0, **options)
    ...     category = "%s-%s" % (optimizer, activation)
    ...     return category, net
    >>> exp = Experiment("XOR")
    >>> exp.run(function,
    ...         epochs=[5],
    ...         accuracy=[0.8],
    ...         tolerance=[0.2],
    ...         optimizer=["adam", "sgd"],
    ...         activation=["sigmoid", "relu"],
    ...         dir="/tmp/")
    >>> len(exp.results)
    4
    >>> exp.plot("loss", interactive=False)
    <IPython.core.display.SVG object>
    >>> exp.apply(lambda category, exp_name: (category, exp_name))
    [('adam-sigmoid', '/tmp/XOR-00001-00001'), ('sgd-sigmoid', '/tmp/XOR-00001-00002'), ('adam-relu', '/tmp/XOR-00001-00003'), ('sgd-relu', '/tmp/XOR-00001-00004')]
    """
    def __init__(self, name):
        self.name = name
        self.results = []
        self.cache = {}

    def run(self, function, trials=1, dir="./", save=True, cache=False, **options):
        """
        Run a set of experiments, varying parameters.

        If save == True, then the networks will be saved to disk.

        If cache == True, then the networks will be saved in memory.
        """
        options = sorted(options.items())
        keys = [option[0] for option in options]
        values = [option[1] for option in options]
        for trial in range(1, trials + 1):
            count = 1
            for combination in itertools.product(*values):
                opts = dict(zip(keys, combination))
                category, net = function(**opts)
                exp_name = "%s%s-%05d-%05d" % (dir, self.name, trial, count)
                if save:
                    net.save(exp_name)
                if cache:
                    self.cache[exp_name] = net
                self.results.append((category, exp_name))
                count += 1

    def apply(self, function, *args, **kwargs):
        """
        Apply a function to experimental runs.

        function() takes either:

            category, network-name, *args, and **kwargs

        or

            category, network, *args, **kwargs

        depending on cache, and returns some results.
        """
        from conx import Network
        results = []
        for (category, exp_name) in self.results:
            if exp_name in self.cache:
                results.append(function(category, self.cache[exp_name], *args, **kwargs))
            else:
                results.append(function(category, exp_name, *args, **kwargs))
        return results

    def plot(self, metrics='loss', symbols=None, interactive=True, format='svg'):
        """
        Plot all of the results of the experiment on a single plot.
        """
        from conx import Network
        colors = list('bgrcmyk')
        symbols = {}
        count = 0
        for (category, exp_name) in self.results:
            if category not in symbols:
                symbols[category] = colors[count % len(colors)] + '-'
                count += 1
        fig_ax = None
        for (category, exp_name) in self.results:
            if exp_name in self.cache:
                net = self.cache[exp_name]
            else:
                net = Network.load(exp_name)
            fig_ax = net.plot(metrics, return_fig_ax=True, fig_ax=fig_ax, label=category,
                              symbols=symbols, title=self.name)
        fig, ax = fig_ax
        if interactive:
            plt.show(block=False)
        else:
            from IPython.display import SVG
            bytes = io.BytesIO()
            if format == "svg":
                plt.savefig(bytes, format="svg")
                plt.close(fig)
                img_bytes = bytes.getvalue()
                return SVG(img_bytes.decode())
            elif format == "pil":
                plt.savefig(bytes, format="png")
                plt.close(fig)
                bytes.seek(0)
                pil_image = PIL.Image.open(bytes)
                return pil_image
            else:
                raise Exception("format must be 'svg' or 'pil'")
