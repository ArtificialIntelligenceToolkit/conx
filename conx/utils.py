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

def choice(seq, p=None):
    """
    Get a random choice from sequence, optinally given a probability
    distribution.

    >>> choice(1)
    0

    >>> choice([42])
    42

    >>> choice("abcde", p=[0, 1, 0, 0, 0])
    'b'
    """
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

def argmax(seq):
    """
    Find the index of the maximum value in seq.
    """
    return np.argmax(seq)

def image2array(image):
    """
    Convert a PIL.Image into a numpy array.
    """
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
        return "gpu" if any(["gpu" in dev for dev in devices]) else "cpu"
    else:
        return "unknown"

'''

def evaluate(model, test_inputs, test_targets, threshold=0.50, indices=None, show=False):
    assert len(test_targets) == len(test_inputs), "number of inputs and targets must be the same"
    if type(indices) not in (list, tuple) or len(indices) == 0:
        indices = range(len(test_inputs))
    # outputs = [np.argmax(t) for t in model.predict(test_inputs[indices]).round()]
    # targets = list(test_labels[indices])
    wrong = 0
    for i in indices:
        target_vector = test_targets[i]
        target_class = np.argmax(target_vector)
        output_vector = model.predict(test_inputs[i:i+1])[0]
        output_class = np.argmax(output_vector)  # index of highest probability in output_vector
        probability = output_vector[output_class]
        if probability < threshold or output_class != target_class:
            if probability < threshold:
                output_class = '???'
            print('image #%d (%s) misclassified as %s' % (i, target_class, output_class))
            wrong += 1
            if show:
                plt.imshow(test_images[i], cmap='binary', interpolation='nearest')
                plt.draw()
                answer = raw_input('RETURN to continue, q to quit...')
                if answer in ('q', 'Q'):
                    return
    total = len(indices)
    correct = total - wrong
    correct_percent = 100.0*correct/total
    wrong_percent = 100.0*wrong/total
    print('%d test images: %d correct (%.1f%%), %d wrong (%.1f%%)' %
          (total, correct, correct_percent, wrong, wrong_percent))

'''
