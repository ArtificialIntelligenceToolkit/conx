# conx - a neural network library
#
# Copyright (c) 2016-2017 Douglas S. Blank <dblank@cs.brynmawr.edu>
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

"""
The Dataset class is useful for loading standard datasets, or for
manipulating a set of inputs/targets.
"""

from functools import reduce
import operator
import numpy as np
import copy
import numbers

import keras
from keras.utils import to_categorical

from .utils import valid_shape, onehot, binary

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
    [<class 'numbers.Number'>, [2, 1]]
    """
    if (isinstance(form, list) and
        len(form) == 2 and
        isinstance(form[1], numbers.Number)):
        ## Is it [type, count]
        if form[0] in (np.dtype, numbers.Number):
            return form[0], [form[1]]
        else:
            ## or [[...], [...]]
            f = get_shape(form[0])
            return [f[0], [form[1]] + f[1]]
    else:
        return [get_shape(x) for x in form]

def shape(item):
    """
    Shortcut for get_shape(get_form(item)).
    """
    return tuple(get_shape(get_form(item))[1])

class _DataVector():
    """
    Class to make internal Keras numpy arrays look like
    lists in the [bank, bank, ...] format.
    """
    def __init__(self, dataset, item):
        self.dataset = dataset
        self.item = item
        self._iter_index = 0

    def __getitem__(self, pos):
        """
        >>> from conx import Network, Dataset
        >>> net = Network("Test 0", 3, 2)
        >>> net.compile(error="mse", optimizer="adam")
        >>> ds = net.dataset
        >>> ds.add([1, 2, 3], [4, 5])
        >>> ds.inputs[0]
        [1.0, 2.0, 3.0]
        >>> ds.inputs[0][1]
        2.0
        >>> ds.targets[0]
        [4.0, 5.0]
        >>> ds.targets[0][1]
        5.0
        """
        if self.item == "targets":
            return self.dataset._get_target(pos)
        elif self.item == "inputs":
            return self.dataset._get_input(pos)
        elif self.item == "test_inputs":
            return self.dataset._get_test_input(pos)
        elif self.item == "train_inputs":
            return self.dataset._get_train_input(pos)
        elif self.item == "test_targets":
            return self.dataset._get_test_target(pos)
        elif self.item == "train_targets":
            return self.dataset._get_train_target(pos)
        else:
            raise Exception("unknown vector: %s" % (item,))

    def __setitem__(self, pos, value):
        """
        Assigning a value is not permitted.
        """
        raise Exception("setting value in a dataset is not permitted;" +
                        " you'll have to recreate the dataset and re-load")

    def shape(self, bank_index=None):
        """
        Get the shape of the tensor at bank_index.
        """
        if self.item in ["targets", "test_targets", "train_targets"]:
            if bank_index is None:
                if self.dataset._num_target_banks > 1:
                    return [self.shape(i) for i in range(self.dataset._num_target_banks)]
                else:
                    bank_index = 0
            if bank_index >= self.dataset._num_target_banks:
                raise Exception("targets bank_index is out of range")
            if self.dataset._num_target_banks == 1:
                return self.dataset._targets.shape[1:]
            else:
                return self.dataset._targets[bank_index].shape[1:]
        elif self.item in ["inputs", "test_inputs", "train_inputs"]:
            if bank_index is None:
                if self.dataset._num_input_banks > 1:
                    return [self.shape(i) for i in range(self.dataset._num_input_banks)]
                else:
                    bank_index = 0
            if bank_index >= self.dataset._num_input_banks:
                raise Exception("inputs bank_index is out of range")
            if self.dataset._num_input_banks == 1:
                return self.dataset._inputs.shape[1:]
            else:
                return self.dataset._inputs[bank_index].shape[1:]
        else:
            raise Exception("unknown vector: %s" % (item,))

    def reshape(self, bank_index, new_shape=None):
        """
        Reshape the tensor at bank_index.

        >>> from conx import Network
        >>> net = Network("Test 1", 10, 2, 3, 28 * 28)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.add([0] * 10, [0] * 28 * 28)
        >>> net.dataset.targets.reshape(0, (28, 28, 1))
        >>> net.dataset._targets.shape
        (1, 28, 28, 1)
        >>> net.dataset.inputs.reshape(0, (2, 5))
        >>> net.dataset._inputs.shape
        (1, 2, 5)
        """
        if new_shape is None:
            new_shape = bank_index
            bank_index = 0
        if not isinstance(new_shape, (list, tuple)):
            new_shape = tuple([new_shape])
        else:
            new_shape = tuple(new_shape)
        if self.item == "targets":
            if bank_index >= self.dataset._num_target_banks:
                raise Exception("targets bank_index is out of range")
            if self.dataset._num_target_banks == 1:
                shape = self.dataset._targets.shape
                self.dataset._targets = self.dataset._targets.reshape((shape[0],) + new_shape)
            else:
                shape = self.dataset._targets[0].shape
                self.dataset_targets[0] = self.dataset._targets[0].reshape((shape[0],) + new_shape)
        elif self.item == "inputs":
            if bank_index >= self.dataset._num_target_banks:
                raise Exception("inputs bank_index is out of range")
            if self.dataset._num_target_banks == 1:
                shape = self.dataset._inputs.shape
                self.dataset._inputs = self.dataset._inputs.reshape((shape[0],) + new_shape)
            else:
                shape = self.dataset._inputs[0].shape
                self.dataset._inputs[0] = self.dataset._inputs[0].reshape((shape[0],) + new_shape)
        elif self.item in ["test_targets", "train_targets"]:
            raise Exception("unable to reshape vector '%s';  call dataset.targets.reshape(), and re-split" % (item,))
        elif self.item in ["test_inputs", "train_inputs"]:
            raise Exception("unable to reshape vector '%s'; call dataset.inputs.rehsape(), and re-split" % (item,))
        else:
            raise Exception("unknown vector: %s" % (item,))
        self.dataset._cache_values()

    def flatten(self):
        """
        >>> from conx import Network
        >>> net = Network("Test 2", 10, 2, 3, 28 * 28)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.add([0] * 10, [0] * 28 * 28)
        >>> net.dataset.targets.reshape(0, (28, 28, 1))
        >>> net.dataset._targets.shape
        (1, 28, 28, 1)
        >>> net.dataset.inputs.reshape(0, (2, 5))
        >>> net.dataset._inputs.shape
        (1, 2, 5)
        """
        if self.item == "targets":
            if self.dataset._num_target_banks == 1:
                self.dataset._targets = self.dataset._targets.reshape((shape[0],) + new_shape)
            else:
                self.dataset._targets = [[c.flatten() for c in row] for row in self.dataset_targets]
        elif self.item == "inputs":
            if self.dataset._num_target_banks == 1:
                self.dataset._inputs = np.array([row.flatten() for row in self.dataset._inputs])
            else:
                self.dataset._inputs = [[c.flatten() for c in row] for row in self.dataset_inputs]
        elif self.item in ["test_targets", "train_targets"]:
            raise Exception("unable to flatten vector '%s';  call dataset.targets.flatten(), and re-split" % (item,))
        elif self.item in ["test_inputs", "train_inputs"]:
            raise Exception("unable to flatten vector '%s'; call dataset.inputs.flatten(), and re-split" % (item,))
        else:
            raise Exception("unknown vector: %s" % (item,))
        self.dataset._cache_values()

    def __len__(self):
        """
        >>> from conx import Network
        >>> net = Network("Test 3", 10, 2, 3, 28)
        >>> net.compile(error="mse", optimizer="adam")
        >>> for i in range(20):
        ...     net.dataset.add([i] * 10, [i] * 28)
        >>> len(net.dataset.targets)
        20
        >>> len(net.dataset.inputs)
        20
        >>> len(net.dataset.test_targets)
        0
        >>> len(net.dataset.train_targets)
        20
        """
        if self.item == "targets":
            return self.dataset._get_targets_length()
        elif self.item == "inputs":
            return self.dataset._get_inputs_length()
        elif self.item == "train_targets":
            return int((1 - self.dataset._split) * len(self.dataset.targets))
        elif self.item == "train_inputs":
            return int((1 - self.dataset._split) * len(self.dataset.inputs))
        elif self.item == "test_targets":
            return int(self.dataset._split * len(self.dataset.targets))
        elif self.item == "test_inputs":
            return int(self.dataset._split * len(self.dataset.inputs))
        else:
            raise Exception("unknown vector type: %s" % (item,))

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self):
            result = self[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

    def __repr__(self):
        length = len(self)
        if length:
            ## type and shape:
            shape = get_shape(get_form(self[0]))
            return "<Dataset '%s', length: %s, shape: %s>" % (
                self.item, length, tuple(shape[1]))
        else:
            return "<Dataset '%s', length: %s, shape: None>" % (
                self.item, length)

class Dataset():
    DATASETS = ['mnist', 'cifar10', 'cifar100']

    """
    Contains the dataset, and metadata about it.

    input_shapes = [shape, ...]
    target_shapes = [shape, ...]
    """
    def __init__(self, network):
        """
        Dataset constructor requires a network.
        """
        self._num_input_banks = 0
        self._num_target_banks = 0
        self.clear()
        self.network = network

    def set_bank_counts(self):
        """
        Called when network is compiled.
        """
        self._num_input_banks = len(self.network.input_bank_order)
        self._num_target_banks = len(self.network.output_bank_order)

    def __getattr__(self, item):
        """
        Construct a virtual Vector for easy access to internal
        format.
        """
        if item in [
                "inputs", "targets",
                "test_inputs", "test_targets",
                "train_inputs", "train_targets",
        ]:
            return _DataVector(self, item)
        else:
            raise AttributeError("type object 'Dataset' has no attribute '%s'" % (item,))

    def random(self, length, frange=(-1, 1)):
        """
        Append a number of random values in the range frange
        to inputs and targets.

        >>> from conx import *
        >>> net = Network("Random", 5, 2, 3, 4)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.random(100)
        >>> len(net.dataset.inputs)
        100
        >>> len(net.dataset.targets)
        100
        """
        diff = abs(frange[1] - frange[0])
        inputs = []
        ## inputs:
        if self._num_input_banks == 1:
            for i in range(length):
                shape = self.network[self.network.input_bank_order[0]].keras_layer.input_shape[1:]
                inputs.append(np.random.rand(*shape) * diff + frange[0])
        else:
            for i in range(length):
                row = []
                for layer_name in self.network.input_bank_order:
                    shape = self.network[layer_name].keras_layer.input_shape[1:]
                    row.append(np.random.rand(*shape) * diff + frange[0])
                inputs.append(row)
        targets = []
        if self._num_target_banks == 1:
            for i in range(length):
                shape = self.network[self.network.output_bank_order[0]].keras_layer.output_shape[1:]
                targets.append(np.random.rand(*shape) * diff + frange[0])
        else:
            for i in range(length):
                row = []
                for layer_name in self.network.output_bank_order:
                    shape = self.network[layer_name].keras_layer.output_shape[1:]
                    row.append(np.random.rand(*shape) * diff + frange[0])
                targets.append(row)
        self.load(list(zip(inputs, targets)))

    def clear(self):
        """
        Remove all of the inputs/targets.
        """
        self._inputs = []
        self._targets = []
        self._labels = []
        self._targets_range = (0,0)
        self._target_shapes = []
        self._input_shapes = []
        self._split = 0

    def add(self, inputs, targets):
        """
        Add a single (input, target) pair to the dataset.
        """
        self.load(list(zip([inputs], [targets])))

    def add_by_function(self, width, frange, ifunction, tfunction):
        """
        width - length of an input vector
        frange - (start, stop) or (start, stop, step)
        ifunction - "onehot" or "binary" or callable(i, width)
        tfunction - a function given (i, input vector), return target vector

        To add an AND problem:

        >>> from conx import Network
        >>> net = Network("Test 1", 2, 2, 3, 1)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.add_by_function(2, (0, 4), "binary", lambda i,v: [int(sum(v) == len(v))])
        >>> len(net.dataset.inputs)
        4

        Adds the following for inputs/targets:
        [0, 0], [0]
        [0, 1], [0]
        [1, 0], [0]
        [1, 1], [1]

        >>> net = Network("Test 2", 10, 2, 3, 10)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.add_by_function(10, (0, 10), "onehot", lambda i,v: v)
        >>> len(net.dataset.inputs)
        10

        >>> import numpy as np
        >>> net = Network("Test 3", 10, 2, 3, 10)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.add_by_function(10, (0, 10), lambda i, width: np.random.rand(width), lambda i,v: v)
        >>> len(net.dataset.inputs)
        10
        """
        if len(frange) == 2:
            frange = frange + (1, )
        if ifunction == "onehot":
            ifunction = onehot
        elif ifunction == "binary":
            ifunction = binary
        elif callable(ifunction):
            pass # ok
        else:
            raise Exception("unknown vector construction function: " +
                            "use 'onehot', or 'binary' or callable")
        inputs = []
        targets = []
        current = frange[0] # start
        while current < frange[1]: # stop, inclusive
            v = ifunction(current, width)
            inputs.append(v)
            targets.append(tfunction(current, v))
            current += frange[2] # increment
        self.load(list(zip(inputs, targets)))

    def load_direct(self, inputs=None, targets=None, labels=None):
        """
        Set the inputs/targets in the specific internal format:

        [input-vector, input-vector, ...] if single input layer

        [[input-layer-1-vectors ...], [input-layer-2-vectors ...], ...] if input target layers

        [target-vector, target-vector, ...] if single output layer

        [[target-layer-1-vectors], [target-layer-2-vectors], ...] if multi target layers

        """
        ## need to set: _num_input_banks, _input_shapes, _inputs
        ## need to set: _num_target_banks, _target_shapes, _targets
        ## inputs is a list [multiple] or np.array() [single]
        if inputs is not None:
            self._inputs = inputs
        if targets is not None:
            self._targets = targets
        if labels is not None:
            self._labels = labels # should be a np.array/list of single values
        self._cache_values()

    def load(self, pairs=None, inputs=None, targets=None):
        """
        Set the human-specified dataset to a proper keras dataset.

        Multi-inputs or multi-targets must be: [vector, vector, ...] for each layer input/target pairing.

        Note:
            If you have images in your dataset, they must match K.image_data_format().

        See also :any:`matrix_to_channels_last` and :any:`matrix_to_channels_first`.
        """
        if self.network.model is None:
            raise Exception("compile network before setting dataset")
        if inputs is not None:
            if targets is not None:
                if pairs is not None:
                    raise Exception("Use pairs or inputs/targets but not both")
                pairs = zip(inputs, targets)
            else:
                raise Exception("you cannot set inputs without targets")
        elif targets is not None:
            raise Exception("you cannot set targets without inputs")
        if pairs is None:
            raise Exception("you need to call with pairs or with input/targets")
        ## first we check the form of the inputs and targets:
        if len(pairs) == 0:
            raise Exception("need more than zero pairs of inputs/targets")
        for pair in pairs:
            if len(pair) != 2:
                raise Exception("need a pair of inputs/targets for each pattern")
        inputs = [pair[0] for pair in pairs]
        targets = [pair[1] for pair in pairs]
        if len(inputs) > 1:
            form = get_form(inputs[0])
            for i in range(1, len(inputs)):
                if form != get_form(inputs[i]):
                    raise Exception("Malformed input at number %d" % (i + 1))
        if len(targets) > 1:
            form = get_form(targets[0])
            for i in range(1, len(targets)):
                if form != get_form(targets[i]):
                    raise Exception("Malformed target at number %d" % (i + 1))
        # Test the inputs, see if outputs match:
        if self._num_input_banks > 1:
            inputs = [np.array(bank, "float32") for bank in inputs[0]]
        else:
            inputs = np.array([inputs[0]], "float32")
        ## Predict:
        ## try:
        prediction = self.network.model.predict(inputs, batch_size=1)
        ## raise Exception("Invalid input form; got %s" % (inputs,))
        if self._num_target_banks > 1:
            targets = [np.array(bank, "float32") for bank in targets[0]]
            for i in range(len(targets[0])):
                shape = tuple(get_shape(get_form(targets[i]))[1])
                if prediction[0][i].shape != shape:
                    raise Exception("Invalid output shape on bank #%d; got %s, expecting %s" % (i, shape, prediction[0][i].shape))
        else:
            targets = np.array(targets[0], "float32")
            shape = tuple(get_shape(get_form(targets))[1])
            if prediction[0].shape != shape:
                raise Exception("Invalid target shape; got %s, expecting %s" % (shape, prediction[0].shape))
        self.compile(pairs)

    def compile(self, pairs):
        if self._num_input_banks > 1:
            inputs = []
            for i in range(len(pairs[0][0])):
                inputs.append(np.array([x[i] for (x,y) in pairs], "float32"))
        else:
            inputs = np.array([x for (x, y) in pairs], "float32")
        if self._num_target_banks > 1:
            targets = []
            for i in range(len(pairs[0][1])):
                targets.append(np.array([y[i] for (x,y) in pairs], "float32"))
        else:
            targets = np.array([y for (x, y) in pairs], "float32")
        ## inputs:
        if self._num_input_banks == 1: ## np.array
            if len(self._inputs) == 0:
                self._inputs = inputs
            else:
                self._inputs = np.append(self._inputs, inputs, 0)
        else: ## list
            self._inputs.extend(inputs)
        ## targets:
        if self._num_target_banks == 1: ## np.array
            if len(self._targets) == 0:
                self._targets = targets
            else:
                self._targets = np.append(self._targets, targets, 0)
        else: ## list
            self._targets.extend(targets)
        self._cache_values()

    def _get_cifar10(self):
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        inputs = np.concatenate((x_train, x_test))
        labels = np.concatenate((y_train, y_test))
        targets = to_categorical(labels, 10)
        inputs = inputs.astype('float32')
        inputs /= 255
        self.load_direct(inputs, targets, labels)

    def get(self, dataset_name=None):
        """
        Get a known dataset by name.
        """
        if dataset_name == "mnist":
            self._get_mnist()
        elif dataset_name == "cifar10":
            self._get_cifar10()
        elif dataset_name == "cifar100":
            self._get_cifar100()
        else:
            raise Exception(
                ("unknown dataset name '%s': should be one of %s" %
                 (dataset_name, Dataset.DATASETS)))

    def copy(self, dataset):
        """
        Copy the inputs/targets from one dataset into
        this one.
        """
        self.load_direct(inputs=dataset._inputs,
                         targets=dataset._targets,
                         labels=dataset._labels)

    def _get_cifar100(self):
        from keras.datasets import cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        inputs = np.concatenate((x_train, x_test))
        labels = np.concatenate((y_train, y_test))
        targets = to_categorical(labels, 100)
        inputs = inputs.astype('float32')
        inputs /= 255
        self.load_direct(inputs, targets, labels)

    def _get_mnist(self):
        """
        Load the Keras MNIST dataset and format it as images.
        """
        from keras.datasets import mnist
        import keras.backend as K
        # input image dimensions
        img_rows, img_cols = 28, 28
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        ## We need to convert the data to images, but which format?
        ## We ask this Keras instance what it wants, and convert:
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        inputs = np.concatenate((x_train,x_test))
        labels = np.concatenate((y_train,y_test))
        targets = to_categorical(labels)
        self.load_direct(inputs, targets, labels)

    def slice(self, start=None, stop=None):
        """
        Cut out some input/targets.

        net.slice(100) - reduce to first 100 inputs/targets
        net.slice(100, 200) - reduce to second 100 inputs/targets
        """
        if start is not None:
            if stop is None: # (#, None)
                stop = start
                start = 0
            else: #  (#, #)
                pass # ok
        else:
            if stop is None: # (None, None)
                start = 0
                stop = len(self._inputs)
            else: # (None, #)
                start = 0
        if self._num_input_banks > 1:
            inputs = [np.array(row[start:stop]) for row in self._inputs]
        else:
            inputs = self._inputs[start:stop] # ok
        self._inputs = inputs
        if self._num_target_banks > 1:
            targets = [np.array(row[start:stop]) for row in self._targets]
        else:
            targets = self._targets[start:stop]
        self._targets = targets
        if len(self._labels) > 0:
            self._labels = self._labels[start:stop]
        self._cache_values()

    def _cache_values(self):
        if len(self.inputs) > 0:
            if self._num_input_banks > 1:
                self._inputs_range = (min([x.min() for x in self._inputs]),
                                      max([x.max() for x in self._inputs]))
            else:
                self._inputs_range = (self._inputs.min(), self._inputs.max())
        else:
            self._inputs_range = (0,0)
        if len(self.targets) > 0:
            if self._num_target_banks > 1:
                self._targets_range = (min([x.min() for x in self._targets]),
                                       max([x.max() for x in self._targets]))
            else:
                self._targets_range = (self._targets.min(), self._targets.max())
        else:
            self._targets_range = (0,0)
        ## Set shape cache:
        if self._num_input_banks > 1:
            self._input_shapes = [x[0].shape for x in self._inputs]
        else:
            self._input_shapes = [self._inputs[0].shape]
        if self._num_target_banks > 1:
            self._target_shapes = [x[0].shape for x in self._targets]
        else:
            self._target_shapes = [self._targets[0].shape]
        # Final checks:
        if len(self.inputs) != len(self.targets):
            raise Exception("WARNING: inputs/targets lengths do not match")

    ## FIXME: add these for users' convenience:
    #def matrix_to_channels_last(self, matrix): ## vecteor
    # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #def matrix_to_channels_first(self, matrix):
    # x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    ## FIXME: Define when we have a specific file to test on:
    # def load_npz_dataset(self, filename, verbose=True):
    #     """loads a dataset from an .npz file and returns data, labels"""
    #     if filename[-4:] != '.npz':
    #         raise Exception("filename must end in .npz")
    #     if verbose:
    #         print('Loading %s dataset...' % filename)
    #     try:
    #         f = np.load(filename)
    #         self._inputs = f['data']
    #         self._labels = f['labels']
    #         self._targets = []
    #         if self._get_inputs_length() != len(self._labels):
    #             raise Exception("Dataset contains different numbers of inputs and labels")
    #         if self._get_inputs_length() == 0:
    #             raise Exception("Dataset is empty")
    #         self._cache_dataset_values()
    #         self._split_dataset(self._num_inputs, verbose=False)
    #         if verbose:
    #             self.summary()
    #     except:
    #         raise Exception("couldn't load .npz dataset %s" % filename)

    def set_targets_from_inputs(self, f=None):
        """
        Copy the inputs to targets. Optionally, apply a function f to
        input copy.

        >>> from conx import Network
        >>> net = Network("Sample", 2, 2, 1)
        >>> ds = [[[0, 0], [0]],
        ...       [[0, 1], [1]],
        ...       [[1, 0], [1]],
        ...       [[1, 1], [0]]]
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.load(ds)
        >>> net.dataset.set_targets_from_inputs(lambda iv: [iv[0]])
        >>> net.dataset.targets[1]
        [0.0]
        """
        if f:
            ## First, apply the function to human form:
            ts = []
            for i in range(len(self.inputs)):
                ts.append(f(self.inputs[i]))
            ## Next, we convert as normal:
            if self._num_target_banks > 1:
                targets = []
                for i in range(len(ts)):
                    targets.append(np.array(ts[i], "float32"))
            else:
                targets = np.array(ts, "float32")
            self._targets = targets
        else: ## no function: just copy the inputs directly
            self._targets = copy.copy(self._inputs)
        self._cache_values()

    def set_inputs_from_targets(self, f=None):
        """
        Copy the targets to inputs. Optionally, apply a function f to
        target copy.

        >>> from conx import Network
        >>> net = Network("Sample", 2, 2, 1)
        >>> ds = [[[0, 0], [0]],
        ...       [[0, 1], [1]],
        ...       [[1, 0], [1]],
        ...       [[1, 1], [0]]]
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.load(ds)
        >>> net.dataset.set_inputs_from_targets(lambda tv: [tv[0], tv[0]])
        >>> net.dataset.inputs[1]
        [1.0, 1.0]
        """
        if f:
            ## First, apply the function to human form:
            ins = []
            for i in range(len(self.targets)):
                ins.append(f(self.targets[i]))
            ## Next, we convert as normal:
            if self._num_target_banks > 1:
                inputs = []
                for i in range(len(ins)):
                    inputs.append(np.array(ins[i], "float32"))
            else:
                inputs = np.array(ins, "float32")
            self._inputs = inputs
        else: ## no function: just copy the targets directly
            self._inputs = copy.copy(self._targets)
        self._cache_values()

    def set_targets_from_labels(self, num_classes):
        """
        Given net.labels are integers, set the net.targets to onehot() categories.
        """
        ## FIXME: allow working on multi-targets
        if self._num_target_banks > 1:
            raise Exception("set_targets_from_labels does not yet work on multi-target patterns")
        if len(self.inputs) == 0:
            raise Exception("no dataset loaded")
        if not isinstance(num_classes, numbers.Integral) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self._targets = to_categorical(self._labels, num_classes).astype("uint8")
        self._cache_values()
        print('Generated %d target vectors from %d labels' % (len(self.targets), num_classes))

    def summary(self):
        """
        Print out a summary of the dataset.
        """
        print('Input Summary:')
        if self._split == 1:
            num_validation = len(self.inputs)
            num_training = len(self.inputs)
        else:
            num_validation = int(self._split * len(self.inputs))
            num_training = len(self.inputs) - num_validation
        print('   count  : %d (%d for training, %s for testing)' %
              (len(self.inputs), num_training, num_validation))
        if len(self.inputs) != 0:
            if self._num_target_banks > 1:
                print('   shape  : %s' % ([x[0].shape for x in self._inputs],))
            else:
                print('   shape  : %s' % (self._inputs[0].shape,))
            print('   range  : %s' % (self._inputs_range,))
        print('Target Summary:')
        print('   count  : %d (%d for training, %s for testing)' %
              (len(self.targets), num_training, num_validation))
        if len(self.targets) != 0:
            if self._num_target_banks > 1:
                print('   shape  : %s' % ([x[0].shape for x in self._targets],))
            else:
                print('   shape  : %s' % (self._targets[0].shape,))
            print('   range  : %s' % (self._targets_range,))

    def rescale_inputs(self, old_range, new_range, new_dtype):
        """
        Rescale the inputs. WIP.
        """
        ## FIXME: allow working on multi-inputs
        if self._num_input_banks > 1:
            raise Exception("rescale_inputs does not yet work on multi-input patterns")
        old_min, old_max = old_range
        new_min, new_max = new_range
        if self._inputs.min() < old_min or self._inputs.max() > old_max:
            raise Exception('range %s is incompatible with inputs' % (old_range,))
        if old_min > old_max:
            raise Exception('range %s is out of order' % (old_range,))
        if new_min > new_max:
            raise Exception('range %s is out of order' % (new_range,))
        self._inputs = rescale_numpy_array(self._inputs, old_range, new_range, new_dtype)
        self._inputs_range = (self._inputs.min(), self._inputs.max())
        print('Inputs rescaled to %s values in the range %s - %s' %
              (self._inputs.dtype, new_min, new_max))

    def shuffle(self):
        """
        Shuffle the inputs/targets. WIP.
        """
        ## FIXME: allow working on multi-inputs/-targets
        if self._num_target_banks > 1:
            raise Exception("shuffle does not yet work on multi-input/-target patterns")
        if len(self.inputs) == 0:
            raise Exception("no dataset loaded")
        indices = np.random.permutation(len(self.inputs))
        self._inputs = self._inputs[indices]
        if len(self._labels) != 0:
            self._labels = self._labels[indices]
        if len(self.targets) != 0:
            self._targets = self._targets[indices]

    def split(self, split=0.50):
        """Splits the inputs/targets into training and validation sets. The
        split parameter specifies what portion of the dataset to use
        for validation. It can be a fraction in the range [0,1), or an
        integer number of patterns, or 'all'. For example, a split of
        0.25 reserves the last 1/4 of the dataset for validation.

        """
        if len(self.inputs) == 0:
            raise Exception("no dataset loaded")
        if split == 'all':
            self._split = 1.0
        elif isinstance(split, numbers.Integral):
            if not 0 <= split <= len(self.inputs):
                raise Exception("split out of range: %d" % split)
            self._split = split/len(self.inputs)
        elif isinstance(split, numbers.Real):
            if not 0 <= split < 1:
                raise Exception("split is not in the range [0,1): %s" % split)
            self._split = split
        else:
            raise Exception("invalid split: %s" % split)

    def chop(self, retain=0.5):
        """Chop off the inputs/targets and reset split, test data. Retains
        only the specified portion of the original dataset patterns,
        starting from the beginning. retain can be a fraction in the
        range 0-1, or an integer number of patterns to retain.
        """
        if len(self.inputs) == 0:
            raise Exception("no dataset loaded")
        if isinstance(retain, numbers.Integral):
            if not 0 <= retain <= len(self.inputs):
                raise Exception("out of range: %d" % retain)
        elif isinstance(retain, numbers.Real):
            if not 0.0 <= retain <= 1.0:
                raise Exception("not in the range [0,1]: %s" % retain)
            retain = int(len(self.inputs) * retain)
        else:
            raise Exception("invalid input: %s" % retain)
        if self._num_input_banks > 1:
            self._inputs = [col[:retain] for col in self._inputs]
        else:
            self._inputs = self._inputs[:retain]
        if len(self._labels) != 0:
            self._labels = self._labels[:retain]
        if len(self.targets) != 0:
            if self._num_target_banks > 1:
                self._targets = [col[:retain] for col in self._targets]
            else:
                self._targets = self._targets[:retain]

    def _get_input(self, i):
        """
        Get an input from the internal dataset and
        format it in the human API.
        """
        if self._num_input_banks == 1:
            return self._inputs[i].tolist()
        else:
            inputs = []
            for c in range(self._num_input_banks):
                inputs.append(self._inputs[c][i].tolist())
            return inputs

    def _get_target(self, i):
        """
        Get a target from the internal dataset and
        format it in the human API.
        """
        if self._num_target_banks == 1:
            return self._targets[i].tolist()
        else:
            targets = []
            for c in range(self._num_target_banks):
                targets.append(self._targets[c][i].tolist())
            return targets

    def _get_train_input(self, i):
        """
        Get a training input from the internal dataset and
        format it in the human API.
        """
        if self._num_input_banks == 1:
            return self._inputs[i].tolist()
        else:
            inputs = []
            for c in range(self._num_input_banks):
                inputs.append(self._inputs[c][i].tolist())
            return inputs

    def _get_train_target(self, i):
        """
        Get a training target from the internal dataset and
        format it in the human API.
        """
        if self._num_target_banks == 1:
            return self._targets[i].tolist()
        else:
            targets = []
            for c in range(self._num_target_banks):
                targets.append(self._targets[c][i].tolist())
            return targets

    def _get_test_input(self, i):
        """
        Get a test input from the internal dataset and
        format it in the human API.
        """
        pos = int((1 - self._split) * len(self.targets))
        if self._num_input_banks == 1:
            return self._inputs[pos + i].tolist()
        else:
            inputs = []
            for c in range(self._num_input_banks):
                inputs.append(self._inputs[c][pos + i].tolist())
            return inputs

    def _get_test_target(self, i):
        """
        Get a test target from the internal dataset and
        format it in the human API.
        """
        pos = int((1 - self._split) * len(self.targets))
        if self._num_target_banks == 1:
            return self._targets[pos + i].tolist()
        else:
            targets = []
            for c in range(self._num_target_banks):
                targets.append(self._targets[c][pos + i].tolist())
            return targets

    def _get_inputs_length(self):
        """
        Get the number of input patterns.
        """
        if len(self._inputs) == 0:
            return 0
        if self._num_input_banks > 1:
            return self._inputs[0].shape[0]
        else:
            return self._inputs.shape[0]

    def _get_targets_length(self):
        """
        Get the number of target patterns.
        """
        if len(self._targets) == 0:
            return 0
        if self._num_target_banks > 1:
            return self._targets[0].shape[0]
        else:
            return self._targets.shape[0]

    def _get_test_inputs_length(self):
        """
        Get the number of test input patterns.
        """
        if len(self._inputs) == 0:
            return 0
        if self._num_input_banks > 1:
            return self._inputs[0].shape[0]
        else:
            return self._inputs.shape[0]

    def _get_test_targets_length(self):
        """
        Get the number of test target patterns.
        """
        if len(self._targets) == 0:
            return 0
        if self._num_target_banks > 1:
            return self._targets[0].shape[0]
        else:
            return self._targets.shape[0]

    def _get_train_inputs_length(self):
        """
        Get the number of training input patterns.
        """
        if len(self._inputs) == 0:
            return 0
        if self._num_input_banks > 1:
            return self._inputs[0].shape[0]
        else:
            return self._inputs.shape[0]

    def _get_train_targets_length(self):
        """
        Get the number of training target patterns.
        """
        if len(self._targets) == 0:
            return 0
        if self._num_target_banks > 1:
            return self._targets[0].shape[0]
        else:
            return self._targets.shape[0]
