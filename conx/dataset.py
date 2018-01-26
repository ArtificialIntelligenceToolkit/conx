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

import numpy as np
import copy
import numbers
import inspect

from .utils import *
import conx.datasets

class DataVector():
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
        >>> ds.add([1, 2, 3], [4, 5])
        >>> ds.add([1, 2, 3], [4, 5])
        >>> ds.add([1, 2, 3], [4, 5])
        >>> ds.split(1)
        >>> ds.inputs[0]
        [1.0, 2.0, 3.0]
        >>> ds.inputs[0][1]
        2.0
        >>> ds.targets[0]
        [4.0, 5.0]
        >>> ds.targets[0][1]
        5.0
        >>> ds.inputs[:] == [x for x in ds.inputs]
        True
        >>> ds.targets[:] == [x for x in ds.targets]
        True
        >>> ds.test_inputs[:] == [x for x in ds.test_inputs]
        True
        >>> ds.train_inputs[:] == [x for x in ds.train_inputs]
        True
        >>> ds.test_targets[:] == [x for x in ds.test_targets]
        True
        >>> ds.train_targets[:] == [x for x in ds.train_targets]
        True

        >>> ds = Dataset()
        >>> ds.add([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
        >>> ds.add([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
        >>> ds.add([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
        >>> ds.add([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
        >>> ds.split(1)
        >>> ds.inputs[0]
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        >>> ds.inputs[0][1]
        [1.0, 2.0, 3.0]
        >>> ds.inputs[0][1][1]
        2.0
        >>> ds.targets[0]
        [[4.0, 5.0], [4.0, 5.0]]
        >>> ds.targets[0][1]
        [4.0, 5.0]
        >>> ds.targets[0][1][1]
        5.0
        >>> ds.inputs[:] == [x for x in ds.inputs]
        True
        >>> ds.targets[:] == [x for x in ds.targets]
        True
        >>> ds.test_inputs[:] == [x for x in ds.test_inputs]
        True
        >>> ds.train_inputs[:] == [x for x in ds.train_inputs]
        True
        >>> ds.test_targets[:] == [x for x in ds.test_targets]
        True
        >>> ds.train_targets[:] == [x for x in ds.train_targets]
        True
        """
        if self.item == "targets":
            if isinstance(pos, slice):
                return [self.dataset._get_target(i) for i in
                        range(len(self.dataset.targets))[pos]]
            else:
                return self.dataset._get_target(pos)
        elif self.item == "labels":
            if isinstance(pos, slice):
                return [self.dataset._get_label(i) for i in
                        range(len(self.dataset.labels))[pos]]
            else:
                return self.dataset._get_label(pos)
        elif self.item == "test_labels":
            if isinstance(pos, slice):
                return [self.dataset._get_test_label(i) for i in
                        range(len(self.dataset.test_labels))[pos]]
            else:
                return self.dataset._get_test_label(pos)
        elif self.item == "train_labels":
            if isinstance(pos, slice):
                return [self.dataset._get_train_label(i) for i in
                        range(len(self.dataset.train_labels))[pos]]
            else:
                return self.dataset._get_train_label(pos)
        elif self.item == "inputs":
            if isinstance(pos, slice):
                return [self.dataset._get_input(i) for i in
                        range(len(self.dataset.inputs))[pos]]
            else:
                return self.dataset._get_input(pos)
        elif self.item == "test_inputs":
            if isinstance(pos, slice):
                return [self.dataset._get_test_input(i) for i in
                        range(len(self.dataset.test_inputs))[pos]]
            else:
                return self.dataset._get_test_input(pos)
        elif self.item == "train_inputs":
            if isinstance(pos, slice):
                return [self.dataset._get_train_input(i) for i in
                        range(len(self.dataset.train_inputs))[pos]]
            else:
                return self.dataset._get_train_input(pos)
        elif self.item == "test_targets":
            if isinstance(pos, slice):
                return [self.dataset._get_test_target(i) for i in
                        range(len(self.dataset.test_targets))[pos]]
            else:
                return self.dataset._get_test_target(pos)
        elif self.item == "train_targets":
            if isinstance(pos, slice):
                return [self.dataset._get_train_target(i) for i in
                        range(len(self.dataset.train_targets))[pos]]
            else:
                return self.dataset._get_train_target(pos)
        else:
            raise Exception("unknown vector: %s" % (self.item,))

    def __setitem__(self, pos, value):
        """
        Assigning a value is not permitted.
        """
        raise Exception("setting value in a dataset is not permitted;" +
                        " you'll have to recreate the dataset and re-load")

    def get_shape(self, bank_index=None):
        """
        Get the shape of the tensor at bank_index.

        >>> from conx import Network, Layer
        >>> net = Network("Get Shape")
        >>> net.add(Layer("input1", 5))
        >>> net.add(Layer("input2", 6))
        >>> net.add(Layer("output", 3))
        >>> net.connect("input1", "output")
        >>> net.connect("input2", "output")
        >>> net.compile(optimizer="adam", error="mse")
        >>> net.dataset.load([
        ...   (
        ...     [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]],
        ...     [0.5, 0.5, 0.5]
        ...   ),
        ... ])
        >>> net.dataset.inputs.get_shape()
        [(5,), (6,)]
        >>> net.dataset.inputs.get_shape(0)
        (5,)
        >>> net.dataset.inputs.get_shape(1)
        (6,)
        >>> net.dataset.targets.get_shape()
        [(3,)]
        >>> net.dataset.targets.get_shape(0)
        (3,)
        >>> net.dataset.inputs.shape
        [(5,), (6,)]
        >>> net.dataset.targets.shape
        [(3,)]
        """
        if self.item in ["targets", "test_targets", "train_targets"]:
            if bank_index is None:
                return [self.get_shape(i) for i in range(self.dataset._num_target_banks())]
            if bank_index >= self.dataset._num_target_banks():
                raise Exception("targets bank_index is out of range")
            if len(self.dataset.targets) > 0:
                return self.dataset._targets[bank_index].shape[1:]
            else:
                return self.dataset._target_shapes[bank_index]
        elif self.item in ["inputs", "test_inputs", "train_inputs"]:
            if bank_index is None:
                return [self.get_shape(i) for i in range(self.dataset._num_input_banks())]
            if bank_index >= self.dataset._num_input_banks():
                raise Exception("inputs bank_index is out of range")
            if len(self.dataset.inputs) > 0:
                return self.dataset._inputs[bank_index].shape[1:]
            else:
                return self.dataset._target_shapes[bank_index]
        else:
            raise Exception("unknown vector: %s" % (self.item,))

    def reshape(self, bank_index, new_shape=None):
        """
        Reshape the tensor at bank_index.

        >>> from conx import Network
        >>> net = Network("Test 1", 10, 2, 3, 28 * 28)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.add([0] * 10, [0] * 28 * 28)
        >>> net.dataset.inputs.shape
        [(10,)]
        >>> net.dataset.inputs.reshape(0, (2, 5))
        >>> net.dataset.inputs.shape
        [(2, 5)]
        >>> net.dataset.targets.shape
        [(784,)]
        >>> net.dataset.targets.shape = (28 * 28,)
        >>> net.dataset.targets.shape
        [(784,)]
        """
        if new_shape is None:
            new_shape = bank_index
            bank_index = 0
        if not isinstance(new_shape, (list, tuple)):
            new_shape = tuple([new_shape])
        else:
            new_shape = tuple(new_shape)
        if self.item == "targets":
            if bank_index >= self.dataset._num_target_banks():
                raise Exception("targets bank_index is out of range")
            shape = self.dataset._targets[bank_index].shape
            self.dataset._targets[bank_index] = self.dataset._targets[bank_index].reshape((shape[0],) + new_shape)
        elif self.item == "inputs":
            if bank_index >= self.dataset._num_target_banks():
                raise Exception("inputs bank_index is out of range")
            shape = self.dataset._inputs[bank_index].shape
            self.dataset._inputs[bank_index] = self.dataset._inputs[0].reshape((shape[0],) + new_shape)
        elif self.item in ["test_targets", "train_targets"]:
            raise Exception("unable to reshape vector '%s';  call dataset.targets.reshape(), and re-split" % (self.item,))
        elif self.item in ["test_inputs", "train_inputs"]:
            raise Exception("unable to reshape vector '%s'; call dataset.inputs.rehsape(), and re-split" % (self.item,))
        else:
            raise Exception("unknown vector: %s" % (self.item,))
        self.dataset._cache_values()

    def flatten(self, bank_index=None):
        """
        >>> from conx import Network
        >>> net = Network("Test 2", 10, 2, 3, 28 * 28)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.add([0] * 10, [0] * 28 * 28)
        >>> net.dataset.targets.shape
        [(784,)]
        >>> net.dataset.inputs.shape
        [(10,)]
        >>> net.dataset.inputs.reshape(0, (2, 5))
        >>> net.dataset.inputs.shape
        [(2, 5)]
        """
        if self.item == "targets":
            if bank_index is None: # flatten all
                self.dataset._targets = [[c.flatten() for c in row] for row in self.dataset._targets]
            else:
                self.dataset._targets[bank_index] = self.dataset._targets[bank_index].flatten()
        elif self.item == "inputs":
            if bank_index is None:
                self.dataset._inputs = [[c.flatten() for c in row] for row in self.dataset._inputs]
            else:
                self.dataset._inputs[bank_index] = self.dataset._inputs[bank_index].flatten()
        elif self.item in ["test_targets", "train_targets"]:
            raise Exception("unable to flatten vector '%s';  call dataset.targets.flatten(), and re-split" % (self.item,))
        elif self.item in ["test_inputs", "train_inputs"]:
            raise Exception("unable to flatten vector '%s'; call dataset.inputs.flatten(), and re-split" % (self.item,))
        else:
            raise Exception("unknown vector: %s" % (self.item,))
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
        size, num_train, num_test = self.dataset._get_split_sizes()
        if self.item == "targets":
            return size
        elif self.item == "labels":
            return size
        elif self.item == "inputs":
            return size
        elif self.item == "train_targets":
            return num_train
        elif self.item == "train_labels":
            return num_train
        elif self.item == "train_inputs":
            return num_train
        elif self.item == "test_targets":
            return num_test
        elif self.item == "test_labels":
            return num_test
        elif self.item == "test_inputs":
            return num_test
        else:
            raise Exception("unknown vector type: %s" % (self.item,))

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
        if "label" in self.item:
            return "<Dataset '%s', length=%s>" % (self.item, length)
        if length > 0:
            ## type and shape:
            shape = get_shape(get_form(self[0]))
            return "<Dataset '%s', length: %s, shape: %s>" % (
                self.item, length, tuple(shape[1]))
        else:
            return "<Dataset '%s', length: %s, shape: None>" % (
                self.item, length)

    shape = property(get_shape, reshape)

class Dataset():
    """
    Contains the dataset, and metadata about it.

    input_shapes = [shape, ...]
    target_shapes = [shape, ...]
    """
    def __init__(self,
                 network=None,
                 name=None,
                 description=None,
                 input_shapes=None,
                 target_shapes=None):
        """
        Dataset constructor.

        You either:

        * give a network
        * give input_shapes and target_shapes as list of shapes
        * or assume that there are one input bank and one
          target bank.

        Defaults inputs and targets are given as a list of tuple shapes,
        one shape per bank.
        """
        self.network = network
        self.name = name
        self.description = description
        self.DATASETS = {name: function for (name, function) in
                         inspect.getmembers(conx.datasets, inspect.isfunction)}
        self.clear()
        if input_shapes is not None:
            self._input_shapes = input_shapes
        if target_shapes is not None:
            self._target_shapes = target_shapes

    def __getattr__(self, item):
        """
        Construct a virtual Vector for easy access to internal
        format.
        """
        if item in [
                "inputs", "targets",
                "test_inputs", "test_targets",
                "train_inputs", "train_targets",
                "labels", "test_labels", "train_labels",
        ]:
            return DataVector(self, item)
        else:
            raise AttributeError("type object 'Dataset' has no attribute '%s'" % (item,))

    def __len__(self):
        """
        Return the size of the dataset (number of inputs/targets).
        """
        return self._get_size()

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
        ## inputs:
        inputs = []
        for i in range(length):
            if self.network:
                for layer_name in self.network.input_bank_order:
                    shape = self.network[layer_name].shape
                    inputs.append(np.random.rand(*shape) * diff + frange[0])
            else:
                for shape in self._input_shapes:
                    inputs.append(np.random.rand(*shape) * diff + frange[0])
        ## targets:
        targets = []
        for i in range(length):
            if self.network:
                for layer_name in self.network.output_bank_order:
                    shape = self.network[layer_name].shape
                    targets.append(np.random.rand(*shape) * diff + frange[0])
            else:
                for shape in self._target_shapes:
                    targets.append(np.random.rand(*shape) * diff + frange[0])
        self.load(list(zip(inputs, targets)))

    def clear(self):
        """
        Remove all of the inputs/targets.
        """
        self._inputs = []
        self._targets = []
        self._labels = []
        self._targets_range = []
        self._split = 0
        self._input_shapes = [(None,)]
        self._target_shapes = [(None,)]

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

        [[input-layer-1-vectors, ...], [input-layer-2-vectors, ...], ...]

        [[target-layer-1-vectors, ...], [target-layer-2-vectors, ...], ...]

        """
        ## inputs/targets are each [np.array(), ...], one np.array()
        ## per bank
        if inputs is not None:
            self._inputs = inputs
        if targets is not None:
            self._targets = targets
        if labels is not None:
            self._labels = labels # should be a list of np.arrays(dtype=str), one per bank
        self._cache_values()

    def load(self, pairs=None, inputs=None, targets=None, labels=None):
        """
        Set the human-specified dataset to a proper keras dataset.

        Multi-inputs or multi-targets must be: [vector, vector, ...] for each layer input/target pairing.

        Note:
            If you have images in your dataset, they must match K.image_data_format().

        See also :any:`matrix_to_channels_last` and :any:`matrix_to_channels_first`.
        """
        if inputs is not None:
            if targets is not None:
                if pairs is not None:
                    raise Exception("Use pairs or inputs/targets but not both")
                if labels is not None:
                    pairs = list(zip(inputs, targets, labels))
                else:
                    pairs = list(zip(inputs, targets))
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
            if len(pair) not in [2, 3]:
                raise Exception("need a pair of inputs/targets for each pattern")
        inputs = [pair[0] for pair in pairs] ## all inputs, human format
        if self._num_input_banks() == 1:
            inputs = [[input] for input in inputs] ## standard format
        targets = [pair[1] for pair in pairs] ## all targets, human format
        if self._num_target_banks() == 1:
            targets = [[target] for target in targets] ## standard format
        labels = []
        if len(pairs[0]) == 3:
            if self._num_target_banks() == 1:
                labels = [[label] for label in labels] ## now standard format
            else:
                labels = [pair[2] for pair in pairs] ## now standard format
        ### standard format from here down:
        if len(inputs) > 1:
            form = get_form(inputs[0]) # get the first form
            for i in range(1, len(inputs)):
                if form != get_form(inputs[i]):
                    raise Exception("Malformed input at number %d" % (i + 1))
        if len(targets) > 1:
            form = get_form(targets[0])
            for i in range(1, len(targets)):
                if form != get_form(targets[i]):
                    raise Exception("Malformed target at number %d" % (i + 1))
        # Test the inputs, see if outputs match:
        if self.network and self.network.model:
            #### Get one to test output: list of np.array() per banks
            inputs = [np.array([bank], "float32") for bank in inputs[0]]
            ## Predict:
            try:
                prediction = self.network.model.predict(inputs, batch_size=1)
            except:
                raise Exception("Invalid input form: %s did not propagate through network" % (inputs,))
            ## NOTE: output of targets varies by number of target banks!!!
            if self._num_target_banks() > 1:
                targets = [np.array([bank], "float32") for bank in targets[0]]
                for i in range(len(targets[0])):
                    shape = targets[0][i].shape
                    if prediction[0][i].shape != shape:
                        raise Exception("Invalid output shape on bank #%d; got %s, expecting %s" % (i, shape, prediction[0][i].shape))
            else:
                targets = [np.array(bank, "float32") for bank in targets[0]]
                shape = targets[0].shape
                if prediction[0].shape != shape:
                    raise Exception("Invalid output shape on bank #%d; got %s, expecting %s" % (0, shape, prediction[0].shape))
        self.compile(pairs)

    def compile(self, pairs):
        if self._num_input_banks() > 1: ## for incoming format
            inputs = []
            for i in range(len(pairs[0][0])):
                inputs.append(np.array([x[0][i] for x in pairs], "float32"))
        else:
            inputs = [np.array([x[0] for x in pairs], "float32")]
        if self._num_target_banks() > 1: ## for incoming format
            targets = []
            for i in range(len(pairs[0][1])):
                targets.append(np.array([y[1][i] for y in pairs], "float32"))
        else:
            targets = [np.array([y[1] for y in pairs], "float32")]
        labels = []
        if len(pairs[0]) == 3:
            if self._num_target_banks() > 1: ## for incoming format
                for i in range(len(pairs[0][2])):
                    labels.append(np.array([y[2][i] for y in pairs], str))
            else:
                labels = [np.array([y[2] for y in pairs], str)]
        ## inputs:
        if len(self._inputs) == 0:
            self._inputs = inputs
        else:
            for i in range(len(self._inputs)):
                self._inputs[i] = np.append(self._inputs[i], inputs[i], 0)
        ## targets:
        if len(self._targets) == 0:
            self._targets = targets
        else:
            for i in range(len(self._targets)):
                self._targets[i] = np.append(self._targets[i], targets[i], 0)
        ## labels:
        if len(self._labels) == 0:
            self._labels = labels
        else:
            for i in range(len(self._labels)):
                self._labels[i] = np.append(self._labels[i], labels[i], 0)
        self._cache_values()

    def datasets(self=None):
        """
        Returns the list of available datasets.

        Can be called on the Dataset class.

        >>> len(Dataset.datasets())
        8

        >>> ds = Dataset()
        >>> len(ds.datasets())
        8
        """
        if self is None:
            self = Dataset()
        return sorted(self.DATASETS.keys())

    def get(self, dataset_name=None, *args, **kwargs):
        """
        Get a known dataset by name.

        Can be called on the Dataset class. If it is, returns a new
        Dataset instance.

        >>> print("Downloading..."); ds = Dataset.get("mnist") # doctest: +ELLIPSIS
        Downloading...
        >>> len(ds.inputs)
        70000

        >>> ds = Dataset()
        >>> ds.get("mnist")
        >>> len(ds.targets)
        70000
        >>> ds.targets[0]
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        """
        return_it = False
        if isinstance(self, str):
            dataset_name, self = self, Dataset()
            return_it = True
        if dataset_name in self.DATASETS:
            self.DATASETS[dataset_name](self, *args, **kwargs)
            if return_it:
                return self
        else:
            raise Exception(
                ("unknown dataset name '%s': should be one of %s" %
                 (dataset_name, list(self.DATASETS.keys()))))

    def copy(self, dataset):
        """
        Copy the inputs/targets from one dataset into
        this one.
        """
        self.load_direct(inputs=dataset._inputs,
                         targets=dataset._targets,
                         labels=dataset._labels)

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
                stop = len(self._inputs[0])
            else: # (None, #)
                start = 0
        self._inputs = [np.array(row[start:stop]) for row in self._inputs]

        self._targets = [np.array(row[start:stop]) for row in self._targets]
        if len(self._labels) > 0:
            self._labels = [np.array(row[start:stop]) for row in self._labels]
        self._cache_values()

    def _cache_values(self):
        if len(self.inputs) > 0:
            self._inputs_range = list(zip([x.min() for x in self._inputs],
                                          [x.max() for x in self._inputs]))
        else:
            self._inputs_range = []
        if len(self.targets) > 0:
            self._targets_range = list(zip([x.min() for x in self._targets],
                                           [x.max() for x in self._targets]))
        else:
            self._targets_range = []
        ## Set shape cache:
        if len(self._inputs) > 0:
            self._input_shapes = [x[0].shape for x in self._inputs]
        if len(self._targets) > 0:
            self._target_shapes = [x[0].shape for x in self._targets]
        # Final checks:
        if len(self.inputs) != len(self.targets):
            raise Exception("WARNING: inputs/targets lengths do not match")
        if self.network:
            self.network.test_dataset_ranges()

    def set_targets_from_inputs(self, f=None, input_bank=0, target_bank=0):
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
                if self._num_input_banks() == 1:
                    if input_bank != 0:
                        raise Exception("invalid input_bank: %d" % input_bank)
                    ts.append(f(self.inputs[i]))
                else:
                    ts.append(f(self.inputs[i][0]))
            self._targets[target_bank] = np.array(ts)
        else: ## no function: just copy the inputs directly
            self._targets = copy.copy(self._inputs)
        self._cache_values()

    def set_inputs_from_targets(self, f=None, input_bank=0, target_bank=0):
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
                if self._num_target_banks() == 1:
                    if target_bank != 0:
                        raise Exception("invalid target_bank: %d" % target_bank)
                    ins.append(f(self.targets[i]))
                else:
                    ins.append(f(self.targets[i][target_bank]))
            self._inputs[input_bank] = np.array(ins)
        else: ## no function: just copy the targets directly
            self._inputs = copy.copy(self._targets)
        self._cache_values()

    def set_targets_from_labels(self, num_classes=None, bank_index=0):
        """
        Given net.labels are integers, set the net.targets to onehot() categories.
        """
        if len(self.inputs) == 0:
            raise Exception("no dataset loaded")
        if num_classes is None:
            num_classes = len(set(self._labels[bank_index]))
        if not isinstance(num_classes, numbers.Integral) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self._targets[bank_index] = to_categorical([int(v) for v in self._labels[bank_index]], num_classes).astype("uint8")
        self._cache_values()
        print('Generated %d target vectors from %d labels' % (len(self.targets), num_classes))

    def _repr_markdown_(self):
        return self.make_summary()

    def __repr__(self):
        return self.make_summary()

    def make_summary(self):
        retval = ""
        if self.name is not None:
            retval += "**Dataset name**: %s\n\n" % self.name
        if self.description is not None:
            retval += self.description
            retval += "\n"
        size, num_train, num_test = self._get_split_sizes()
        retval += '**Dataset Split**:\n'
        retval += '   * training  : %d\n' % (num_train,)
        retval += '   * testing   : %d\n' % (num_test,)
        retval += '   * total     : %d\n\n' % (size,)
        retval += '**Input Summary**:\n'
        if size != 0:
            retval += '   * shape  : %s\n' % self.inputs.shape
            retval += '   * range  : %s\n\n' % (self._inputs_range,)
        retval += '**Target Summary**:\n'
        if size != 0:
            retval += '   * shape  : %s\n' % self.targets.shape
            retval += '   * range  : %s\n\n' % (self._targets_range,)
        if self.network:
            self.network.test_dataset_ranges()
        return retval

    def summary(self):
        """
        Print out a summary of the dataset.
        """
        return display(self)

    def rescale_inputs(self, bank_index, old_range, new_range, new_dtype):
        """
        Rescale the inputs.
        """
        old_min, old_max = old_range
        new_min, new_max = new_range
        if self._inputs[bank_index].min() < old_min or self._inputs[bank_index].max() > old_max:
            raise Exception('range %s is incompatible with inputs' % (old_range,))
        if old_min > old_max:
            raise Exception('range %s is out of order' % (old_range,))
        if new_min > new_max:
            raise Exception('range %s is out of order' % (new_range,))
        self._inputs[bank_index] = rescale_numpy_array(self._inputs[bank_index], old_range, new_range, new_dtype)
        self._cache_values()

    def shuffle(self):
        """
        Shuffle the inputs/targets.
        """
        if len(self.inputs) == 0:
            raise Exception("no dataset loaded")
        permutation = np.random.permutation(len(self.inputs))
        self._inputs = [self._inputs[b][permutation] for b in range(self._num_input_banks())]
        self._targets = [self._targets[b][permutation] for b in range(self._num_target_banks())]
        if len(self._labels) != 0:
            self._labels = [self._labels[b][permutation] for b in range(self._num_target_banks())]

    def split(self, split=0.50):
        """Splits the inputs/targets into training and validation sets. The
        split keyword parameter specifies what portion of the dataset
        to use for validation. It can be a fraction in the range
        [0,1), or an integer number of patterns from 0 to the dataset
        size, or 'all'. For example, a split of 0.25 reserves the last
        1/4 of the dataset for validation.  A split of 1.0 (specified
        as 'all' or an int equal to the dataset size) is a special
        case in which the entire dataset is used for both training and
        validation.
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

    def _get_split_sizes(self):
        # need a more elegant name for this method
        """returns a tuple (dataset_size, train_set_size, test_set_size),
        based on the current split value
        """
        dataset_size = self._get_size()
        if self._split == 1:
            train_set_size, test_set_size = dataset_size, dataset_size
        else:
            test_set_size = int(self._split * dataset_size)
            train_set_size = dataset_size - test_set_size
        return (dataset_size, train_set_size, test_set_size)

    def _split_data(self):
        size, num_train, num_test = self._get_split_sizes()
        # self._inputs and self._targets are lists of numpy arrays
        train_inputs, train_targets, test_inputs, test_targets = [], [], [], []
        for inputs, targets in zip(self._inputs, self._targets):
            train_inputs.append(inputs[:num_train])
            train_targets.append(targets[:num_train])
            test_inputs.append(inputs[size - num_test:])
            test_targets.append(targets[size - num_test:])
        return (train_inputs, train_targets), (test_inputs, test_targets)

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
                raise Exception("not in the interval [0,1]: %s" % retain)
            retain = int(len(self.inputs) * retain)
        else:
            raise Exception("invalid value: %s" % retain)
        self._inputs = [self._inputs[b][:retain] for b in range(self._num_input_banks())]
        self._targets = [self._targets[b][:retain] for b in range(self._num_target_banks())]
        if len(self._labels) != 0:
            self._labels = [self._labels[b][:retain] for b in range(self._num_target_banks())]

    def _get_input(self, i):
        """
        Get an input from the internal dataset and
        format it in the human API.
        """
        size = self._get_size()
        if not 0 <= i < size:
            raise Exception("input index %d is out of bounds" % (i,))
        else:
            data = [self._inputs[b][i].tolist() for b in range(self._num_input_banks())]
        if self._num_input_banks() == 1:
            return data[0]
        else:
            return data

    def _get_target(self, i):
        """
        Get a target from the internal dataset and
        format it in the human API.
        """
        size = self._get_size()
        if not 0 <= i < size:
            raise Exception("target index %d is out of bounds" % (i,))
        data = [self._targets[b][i].tolist() for b in range(self._num_target_banks())]
        if self._num_target_banks() == 1:
            return data[0]
        else:
            return data

    def _get_label(self, i):
        """
        Get a label from the internal dataset and
        format it in the human API.
        """
        size = self._get_size()
        if not 0 <= i < size:
            raise Exception("label index %d is out of bounds" % (i,))
        data = [self._labels[b][i] for b in range(self._num_target_banks())]
        if self._num_target_banks() == 1:
            return data[0]
        else:
            return data

    def _get_train_input(self, i):
        """
        Get a training input from the internal dataset and
        format it in the human API.
        """
        size, num_train, num_test = self._get_split_sizes()
        if not 0 <= i < num_train:
            raise Exception("training input index %d is out of bounds" % (i,))
        data = [self._inputs[b][i].tolist() for b in range(self._num_input_banks())]
        if self._num_input_banks() == 1:
            return data[0]
        else:
            return data

    def _get_train_target(self, i):
        """
        Get a training target from the internal dataset and
        format it in the human API.
        """
        size, num_train, num_test = self._get_split_sizes()
        if not 0 <= i < num_train:
            raise Exception("training target index %d is out of bounds" % (i,))
        data = [self._targets[b][i].tolist() for b in range(self._num_target_banks())]
        if self._num_target_banks() == 1:
            return data[0]
        else:
            return data

    def _get_train_label(self, i):
        """
        Get a training label from the internal dataset and
        format it in the human API.
        """
        size, num_train, num_test = self._get_split_sizes()
        if not 0 <= i < num_train:
            raise Exception("training label index %d is out of bounds" % (i,))
        data = [self._labels[b][i] for b in range(self._num_target_banks())]
        if self._num_target_banks() == 1:
            return data[0]
        else:
            return data

    def _get_test_input(self, i):
        """
        Get a test input from the internal dataset and
        format it in the human API.
        """
        size, num_train, num_test = self._get_split_sizes()
        if not 0 <= i < num_test:
            raise Exception("test input index %d is out of bounds" % (i,))
        j = size - num_test + i
        data = [self._inputs[b][j].tolist() for b in range(self._num_input_banks())]
        if self._num_input_banks() == 1:
            return data[0]
        else:
            return data

    def _get_test_target(self, i):
        """
        Get a test target from the internal dataset and
        format it in the human API.
        """
        size, num_train, num_test = self._get_split_sizes()
        if not 0 <= i < num_test:
            raise Exception("test target index %d is out of bounds" % (i,))
        j = size - num_test + i
        data = [self._targets[b][j].tolist() for b in range(self._num_target_banks())]
        if self._num_target_banks() == 1:
            return data[0]
        else:
            return data

    def _get_test_label(self, i):
        """
        Get a test label from the internal dataset and
        format it in the human API.
        """
        size, num_train, num_test = self._get_split_sizes()
        if not 0 <= i < num_test:
            raise Exception("test label index %d is out of bounds" % (i,))
        j = size - num_test + i
        data = [self._labels[b][j] for b in range(self._num_target_banks())]
        if self._num_target_banks() == 1:
            return data[0]
        else:
            return data

    def _num_input_banks(self):
        """
        How many input banks?

        1. we ask network, if one
        2. if not, we check previous inputs
        3. else we fall back on defaults
        """
        if self.network and self.network.num_input_layers != 0 :
            return self.network.num_input_layers
        else:
            return len(self._input_shapes)

    def _num_target_banks(self):
        """
        How many target banks?

        1. we ask network, if one
        2. else we fall back on defaults
        """
        if self.network and self.network.num_target_layers:
            return self.network.num_target_layers
        else:
            return len(self._target_shapes)

    def _get_size(self):
        """
        Returns the total number of patterns/targets in the dataset

        >>> ds = Dataset()
        >>> ds._get_size()
        0
        """
        if len(self._inputs) > 0:
            return self._inputs[0].shape[0]
        else:
            return 0
