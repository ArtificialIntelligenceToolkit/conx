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
import sys
from IPython.display import display
from functools import reduce
import types
import keras
import operator
import os

from .utils import *
import conx.datasets

py_slice = slice

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
        Get a pos (positive or negative) or a slice from the
        vector.


        >>> from conx import Network, Dataset
        >>> net = Network("Test 0", 3, 2)
        >>> net.compile(error="mse", optimizer="adam")
        >>> ds = net.dataset
        >>> ds.append([1, 2, 3], [4, 5])
        >>> ds.append([1, 2, 3], [4, 5])
        >>> ds.append([1, 2, 3], [4, 5])
        >>> ds.append([1, 2, 3], [4, 5])
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
        >>> ds.append([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
        >>> ds.append([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
        >>> ds.append([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
        >>> ds.append([[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]])
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
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_target(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_target(len(self.dataset.targets) + pos)
            else:
                return self.dataset._get_target(pos)
        elif self.item == "labels":
            if isinstance(pos, slice):
                return [self.dataset._get_label(i) for i in
                        range(len(self.dataset.labels))[pos]]
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_label(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_label(len(self.dataset.labels) + pos)
            else:
                return self.dataset._get_label(pos)
        elif self.item == "test_labels":
            if isinstance(pos, slice):
                return [self.dataset._get_test_label(i) for i in
                        range(len(self.dataset.test_labels))[pos]]
            elif pos < 0:
                return self.dataset._get_test_label(len(self.dataset.test_labels) + pos)
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_test_label(i) for i in pos]
            else:
                return self.dataset._get_test_label(pos)
        elif self.item == "train_labels":
            if isinstance(pos, slice):
                return [self.dataset._get_train_label(i) for i in
                        range(len(self.dataset.train_labels))[pos]]
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_train_label(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_train_label(len(self.dataset.train_labels) + pos)
            else:
                return self.dataset._get_train_label(pos)
        elif self.item == "inputs":
            if isinstance(pos, slice):
                return [self.dataset._get_input(i) for i in
                        range(len(self.dataset.inputs))[pos]]
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_input(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_input(len(self.dataset.inputs) + pos)
            else:
                return self.dataset._get_input(pos)
        elif self.item == "test_inputs":
            if isinstance(pos, slice):
                return [self.dataset._get_test_input(i) for i in
                        range(len(self.dataset.test_inputs))[pos]]
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_test_input(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_test_input(len(self.dataset.test_inputs) + pos)
            else:
                return self.dataset._get_test_input(pos)
        elif self.item == "train_inputs":
            if isinstance(pos, slice):
                return [self.dataset._get_train_input(i) for i in
                        range(len(self.dataset.train_inputs))[pos]]
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_train_input(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_train_input(len(self.dataset.train_inputs) + pos)
            else:
                return self.dataset._get_train_input(pos)
        elif self.item == "test_targets":
            if isinstance(pos, slice):
                return [self.dataset._get_test_target(i) for i in
                        range(len(self.dataset.test_targets))[pos]]
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_test_target(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_test_target(len(self.dataset.test_targets) + pos)
            else:
                return self.dataset._get_test_target(pos)
        elif self.item == "train_targets":
            if isinstance(pos, slice):
                return [self.dataset._get_train_target(i) for i in
                        range(len(self.dataset.train_targets))[pos]]
            elif isinstance(pos, (list, tuple)):
                return [self.dataset._get_train_target(i) for i in pos]
            elif pos < 0:
                return self.dataset._get_train_target(len(self.dataset.train_targets) + pos)
            else:
                return self.dataset._get_train_target(pos)
        else:
            raise Exception("unknown vector: %s" % (self.item,))

    def delete_bank(self, position):
        """
        Delete a bank of inputs, targets, or labels.


        >>> ds = Dataset()
        >>> ds.load(inputs=[[0, 0]], targets=[[0, 0, 0]], labels=["zero"])
        >>> ds.inputs.append_bank(4)
        >>> ds.targets.append_bank(5)
        >>> ds.labels.append_bank(10)

        >>> ds.inputs.delete_bank(0)
        >>> ds.targets.delete_bank(0)
        >>> ds.labels.delete_bank(0)
        """
        if self.item == "targets":
            del self.dataset._targets[position]
        elif self.item == "inputs":
            del self.dataset._inputs[position]
        elif self.item == "labels":
            del self.dataset._labels[position]
        else:
            raise Exception("remove_bank can only be applied to targets, inputs, and labels")
        self.dataset._cache_values()

    def append_bank(self, shape=None, dtype=None):
        """
        Add a new bank of inputs, targets, or labels with given shape.

        For labels, shape is max length of any string in bank. dtype is
        str for labels.

        For inputs and targets, shape is shape of tensor. dtype is
        'float32' by default.

        Note:
            labels and targets should have the same number of banks.

        >>> ds = Dataset()
        >>> ds.load(inputs=[[0,0], [1,1]], targets=[[0,0,0], [1,1,1]], labels=["zero", "one"])
        >>> ds.inputs.append_bank(4)
        >>> ds.inputs.shape
        [(2,), (4,)]
        >>> ds.targets.append_bank(5)
        >>> ds.targets.shape
        [(3,), (5,)]
        >>> ds.labels.append_bank(1)
        >>> ds.labels[0]
        ['zero', ' ']
        >>> ds.labels[1]
        ['one', ' ']
        """
        if self.item == "targets":
            if shape is not None:
                if isinstance(shape, int):
                    shape = [shape]
            size = reduce(operator.mul, shape) * len(self.dataset)
            new_shape = [len(self.dataset)] + list(shape)
            dtype = 'float32' if dtype is None else dtype
            array = np.array([0 for i in range(size)], dtype=dtype)
            array = array.reshape(new_shape)
            self.dataset._targets.append(array)
        elif self.item == "inputs":
            if shape is not None:
                if isinstance(shape, int):
                    shape = [shape]
            size = reduce(operator.mul, shape) * len(self.dataset)
            new_shape = [len(self.dataset)] + list(shape)
            dtype = 'float32' if dtype is None else dtype
            array = np.array([0 for i in range(size)], dtype=dtype)
            array = array.reshape(new_shape)
            self.dataset._inputs.append(array)
        elif self.item == "labels":
            if shape is None:
                raise Exception("labels must have a maximum length of string for bank given as 'shape'")
            size = len(self.dataset)
            array = np.array([(" " * shape) for i in range(size)], dtype='str')
            self.dataset._labels.append(array)
        else:
            raise Exception("append_bank can only be applied to targets, inputs, and labels")
        self.dataset._cache_values()

    def __setitem__(self, pos, value):
        """
        Set the position in the inputs, targets, or label to a new value
        (tensor, or string).
        """
        if self.item == "targets":
            if isinstance(pos, slice):
                for i in range(len(self.dataset.targets))[pos]:
                    self.targets[i] = value
            elif isinstance(pos, (list, tuple)):
                for i in pos:
                    self.targets[i] = value
            else:
                if self.dataset._num_target_banks() == 1:
                    self.dataset._targets[0][pos] = np.array(value)
                else:
                    for bank in range(self.dataset._num_target_banks()):
                        self.dataset._targets[bank][pos] = np.array(value[bank])
        elif self.item == "inputs":
            if isinstance(pos, slice):
                for i in range(len(self.dataset.inputs))[pos]:
                    self.inputs[i] = value
            elif isinstance(pos, (list, tuple)):
                for i in pos:
                    self.inputs[i] = value
            else:
                if self.dataset._num_input_banks() == 1:
                    self.dataset._inputs[0][pos] = np.array(value)
                else:
                    for bank in range(self.dataset._num_input_banks()):
                        self.dataset._inputs[bank][pos] = np.array(value[bank])
        elif self.item == "labels":
            if isinstance(pos, slice):
                for i in range(len(self.dataset.targets))[pos]:
                    self.labels[i] = value
            elif isinstance(pos, (list, tuple)):
                for i in pos:
                    self.labels[i] = value
            else:
                if self.dataset._num_target_banks() == 1:
                    self.dataset._labels[0][pos] = value
                else:
                    for bank in range(self.dataset._num_target_banks()):
                        self.dataset._labels[bank][pos] = value[bank]
        else:
             raise Exception("unknown vector: %s, should be inputs, targets, or labels" % (self.item,))

    def get_shape(self, bank_index=None):
        """
        Get the shape of the tensor at bank_index.

        >>> from conx import Network, Layer
        >>> net = Network("Get Shape")
        >>> net.add(Layer("input1", 5))
        'input1'
        >>> net.add(Layer("input2", 6))
        'input2'
        >>> net.add(Layer("output", 3))
        'output'
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
            elif self.dataset.network and self.dataset.network.model: ## compiled and connected
                layer_name = self.dataset.network.output_bank_order[bank_index]
                return self.dataset.network[layer_name].shape
            else:
                return self.dataset._target_shapes[bank_index]
        elif self.item in ["inputs", "test_inputs", "train_inputs"]:
            if bank_index is None:
                return [self.get_shape(i) for i in range(self.dataset._num_input_banks())]
            if bank_index >= self.dataset._num_input_banks():
                raise Exception("inputs bank_index is out of range")
            if len(self.dataset.inputs) > 0:
                return self.dataset._inputs[bank_index].shape[1:]
            elif self.dataset.network and self.dataset.network.model: ## compiled and connected
                layer_name = self.dataset.network.input_bank_order[bank_index]
                return self.dataset.network[layer_name].shape
            else:
                return self.dataset._input_shapes[bank_index]
        else:
            raise Exception("unknown vector: %s" % (self.item,))

    def test(self, tolerance=None, slice=None, index=False, batch_size=32):
        """
        Test the inputs, train_inputs, or test_inputs to see which
        ones are correct (within tolerance).

        Examples:
            >>> from conx import Network, Layer, Dataset
            >>> net = Network("XOR", 2, 2, 1, activation="sigmoid")
            >>> net.compile(error='mean_squared_error',
            ...             optimizer="SGD", lr=0.3, momentum=0.9)
            >>> ds = [[[0, 0], [0]],
            ...       [[0, 1], [1]],
            ...       [[1, 0], [1]],
            ...       [[1, 1], [0]]]
            >>> net.dataset.load(ds)
            >>> net.dataset.inputs.test(tolerance=0.0, index=True)
            []
            >>> net.dataset.inputs.test(tolerance=1.0, index=True)
            [0, 1, 2, 3]
        """
        tolerance = tolerance if tolerance else self.dataset.network.tolerance

        def test(i, tolerance):
            length = len(self.dataset)
            if self.item == "inputs":
                inputs = [np.array([column[i]]) for column in self.dataset._inputs]
                targets = [np.array([column[i]]) for column in self.dataset._targets]
            elif self.item == "train_inputs":
                inputs = [np.array([column[i]]) for column in self.dataset._inputs]
                targets = [np.array([column[i]]) for column in self.dataset._targets]
            elif self.item == "test_inputs":
                inputs = [np.array([column[i + int((1.0 - self.dataset._split) * length)]])
                          for column in self.dataset._inputs]
                targets = [np.array([column[i + int((1.0 - self.dataset._split) * length)]])
                           for column in self.dataset._targets]
            outputs = self.dataset.network.model.predict(inputs, batch_size=batch_size)
            if self.dataset.network.num_target_layers > 1:
                correct = self.dataset.network.compute_correct(outputs, targets, tolerance)
            else:
                ## Warning:
                ## keras returns outputs as a single column
                ## conx targets are always multi-column
                correct = self.dataset.network.compute_correct([outputs], targets, tolerance)
            return correct[0]

        if self.item == "inputs":
            retval = (i if index else self.dataset.inputs[i]
                      for i in range(len(self.dataset.inputs)) if test(i, tolerance))
        elif self.item == "train_inputs":
            retval = (i if index else self.dataset.train_inputs[i]
                      for i in range(len(self.dataset.train_inputs)) if test(i, tolerance))
        elif self.item == "test_inputs":
            retval = (i if index else self.dataset.test_inputs[i]
                      for i in range(len(self.dataset.test_inputs)) if test(i, tolerance))
        if slice is None:
            return list(retval)
        else:
            if not isinstance(slice, (list, tuple)):
                slice = (slice,)
            args = py_slice(*slice)
            return list(itertools.islice(retval, args.start, args.stop, args.step))

    def select(self, function, slice=None, index=False):
        """
        select selects items or indices from a dataset pattern.

        function() takes (i, dataset) and returns True or False
        filter will return all items that match the filter.

        Examples:
            >>> import conx as cx
            >>> print("Downloading...");ds = cx.Dataset.get("mnist") # doctest: +ELLIPSIS
            Downloading...
            >>> ds.inputs.select(lambda i,dataset: True, slice=10, index=True)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> s = ds.inputs.select(lambda i,dataset: ds.inputs[i], slice=(10, 20, 2))
            >>> shape(s)
            (5, 28, 28, 1)

            >>> ds.clear()

        Arguments:
            function - callable that takes (i, dataset) and returns True/False
            slice - range of items/indices to return
            index - if index is True, then return indices, else return the items.
        """
        import itertools
        if self.item == "targets":
            retval = (i if index else self.dataset.targets[i]
                      for i in range(len(self.dataset)) if function(i, self.dataset))
        elif self.item == "inputs":
            retval = (i if index else self.dataset.inputs[i]
                      for i in range(len(self.dataset)) if function(i, self.dataset))
        elif self.item == "labels":
            retval = (i if index else self.dataset.labels[i]
                      for i in range(len(self.dataset)) if function(i, self.dataset))
        if self.item == "test_targets":
            retval = (i if index else self.dataset.test_targets[i]
                      for i in range(len(self.dataset.test_inputs)) if function(i, self.dataset))
        elif self.item == "test_inputs":
            retval = (i if index else self.dataset.test_inputs[i]
                      for i in range(len(self.dataset.test_inputs)) if function(i, self.dataset))
        if self.item == "train_targets":
            retval = (i if index else self.dataset.train_targets[i]
                      for i in range(len(self.dataset.train_inputs)) if function(i, self.dataset))
        elif self.item == "train_inputs":
            retval = (i if index else self.dataset.train_inputs[i]
                      for i in range(len(self.dataset.train_inputs)) if function(i, self.dataset))
        if slice is None:
            return list(retval)
        else:
            if not isinstance(slice, (list, tuple)):
                slice = (slice,)
            args = py_slice(*slice)
            return list(itertools.islice(retval, args.start, args.stop, args.step))

    def reshape(self, bank_index, new_shape=None):
        """
        Reshape the tensor at bank_index.

        >>> from conx import Network
        >>> net = Network("Test 1", 10, 2, 3, 28 * 28)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.append([0] * 10, [0] * 28 * 28)
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

    def __len__(self):
        """
        >>> from conx import Network
        >>> net = Network("Test 2", 10, 2, 3, 28)
        >>> net.compile(error="mse", optimizer="adam")
        >>> for i in range(20):
        ...     net.dataset.append([i] * 10, [i] * 28)
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
            return 0 if len(self.dataset._targets) == 0 else len(self.dataset._targets[0])
        elif self.item == "labels":
            return 0 if len(self.dataset._labels) == 0 else len(self.dataset._labels[0])
        elif self.item == "inputs":
            return 0 if len(self.dataset._inputs) == 0 else len(self.dataset._inputs[0])
        else:
            size, num_train, num_test = self.dataset._get_split_sizes()
            if self.item == "train_targets":
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
    Vector = DataVector

    def __init__(self,
                 network=None,
                 name=None,
                 description=None,
                 input_shapes=None,
                 target_shapes=None,
                 version="1.0.0"):
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
        self.warnings = {}
        self.warn_categories = {}
        self.network = network
        self.name = name
        self.version = version
        self.description = description
        self.DATASETS = {name: function for (name, function) in
                         inspect.getmembers(conx.datasets, inspect.isfunction)}
        self.clear()
        if input_shapes is not None:
            self._input_shapes = input_shapes
        if target_shapes is not None:
            self._target_shapes = target_shapes

    def get_id(self):
        """
        Get the version number of the dataset.

        >>> ds = Dataset(name="Test", version="1.0.0")
        >>> ds.get_id()
        'Test-1.0.0'

        >>> ds.version = "2.0.1"
        >>> ds.get_id()
        'Test-2.0.1'

        """
        return "%s-%s" % (self.name, self.version)

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
            return self.Vector(self, item)
        else:
            raise AttributeError("type object 'Dataset' has no attribute '%s'" % (item,))

    def __len__(self):
        """
        Return the size of the dataset (number of inputs/targets).
        """
        return self._get_size()

    def append_random(self, count, frange=(-1, 1)):
        """
        Append a number of random values in the range `frange`
        to inputs and targets.

        Requires that dataset belongs to a network with
        input layers.

        >>> from conx import *
        >>> net = Network("Random", 5, 2, 3, 4)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.append_random(100)
        >>> len(net.dataset.inputs)
        100
        >>> shape(net.dataset.inputs)
        (100, 5)
        >>> len(net.dataset.targets)
        100
        >>> shape(net.dataset.targets)
        (100, 4)
        """
        if self.network is None:
            raise Exception("please call network.set_dataset() on this dataset")
        if (len(self.network.input_bank_order) == 0 or
            len(self.network.output_bank_order) == 0):
            raise Exception("please connect network layers")
        diff = abs(frange[1] - frange[0])
        ## inputs:
        inputs = []
        for i in range(count):
            if self.network:
                for layer_name in self.network.input_bank_order:
                    shape = self.network[layer_name].shape
                    inputs.append(np.random.rand(*shape) * diff + frange[0])
            else:
                for shape in self._input_shapes:
                    inputs.append(np.random.rand(*shape) * diff + frange[0])
        ## targets:
        targets = []
        for i in range(count):
            if self.network:
                for layer_name in self.network.output_bank_order:
                    shape = self.network[layer_name].shape
                    targets.append(np.random.rand(*shape) * diff + frange[0])
            else:
                for shape in self._target_shapes:
                    targets.append(np.random.rand(*shape) * diff + frange[0])
        self._load(list(zip(inputs, targets)), mode="append")

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

    def _add(self, inputs, targets):
        """
        Add a single (input, target) pair to the dataset.
        """
        self._load(list(zip([inputs], [targets])), mode="append")

    def append_by_function(self, width, frange, ifunction, tfunction):
        """
        width - length of an input vector
        frange - (start, stop) or (start, stop, step)
        ifunction - "onehot" or "binary" or callable(i, width)
        tfunction - a function given (i, input vector), return target vector

        To add an AND problem:

        >>> from conx import Network
        >>> net = Network("Test 3", 2, 2, 3, 1)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.append_by_function(2, (0, 4), "binary", lambda i,v: [int(sum(v) == len(v))])
        >>> len(net.dataset.inputs)
        4

        Adds the following for inputs/targets:
        [0, 0], [0]
        [0, 1], [0]
        [1, 0], [0]
        [1, 1], [1]

        >>> net = Network("Test 4", 10, 2, 3, 10)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.append_by_function(10, (0, 10), "onehot", lambda i,v: v)
        >>> len(net.dataset.inputs)
        10

        >>> import numpy as np
        >>> net = Network("Test 5", 10, 2, 3, 10)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.dataset.append_by_function(10, (0, 10), lambda i, width: np.random.rand(width), lambda i,v: v)
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
        self._load(list(zip(inputs, targets)), mode="append")

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

    def save_to_disk(self, dir, *args, **kwargs):
        """
        Save the dataset into the given directory.
        """
        import h5py
        if not os.path.isdir(dir):
            os.makedirs(dir)
        with h5py.File(os.path.join(dir, "dataset.h5"), "w") as h5:
            h5.create_dataset('inputs', data=self._inputs, compression='gzip', compression_opts=9)
            h5.create_dataset('targets', data=self._targets, compression='gzip', compression_opts=9)
            if len(self._labels) > 0:
                string = h5py.special_dtype(vlen=str)
                if isinstance(self._labels, np.ndarray) and self._labels.dtype != string:
                    labels = self._labels.astype(string)
                else:
                    labels = self._labels
            h5.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)

    def load_from_disk(self, dir, *args, **kwargs):
        """
        Load the dataset from the given directory.
        """
        import h5py
        loaded = False
        path = os.path.join(dir, "dataset.h5")
        if os.path.exists(path):
            h5 = h5py.File(path, "r")
            self._inputs = h5["inputs"]
            self._targets = h5["targets"]
            self._labels = h5["labels"]
            self.h5 = h5
            self.name = dir
            #self.description = description
            self._cache_values()
            loaded = True
        return loaded

    def load(self, pairs=None, inputs=None, targets=None, labels=None):
        """
        Dataset.load() will clear and load a new dataset.

        You can load a dataset through a number of variations:

        * dataset.load([[input, target], ...])
        * dataset.load(inputs=[input, ...], targets=[target, ...])
        * dataset.load(generator, count)

        >>> ds = Dataset()
        >>> ds.load([[[0, 0], [0]],
        ...          [[0, 1], [1]],
        ...          [[1, 0], [1]],
        ...          [[1, 1], [0]]])
        >>> len(ds)
        4
        >>> ds.load(inputs=[[0, 0], [0, 1], [1, 0], [1, 1]], # inputs
        ...         targets=[[0], [1], [1], [0]]) # targets
        >>> len(ds)
        4
        >>> def generator(self):
        ...     for data in [[[0, 0], [0]],
        ...                  [[0, 1], [1]],
        ...                  [[1, 0], [1]],
        ...                  [[1, 1], [0]]]:
        ...         yield data
        >>> ds.load(generator(ds), 4)
        >>> len(ds)
        4
        """
        self._load(pairs, inputs, targets, labels, mode="load")

    def append(self, pairs=None, inputs=None):
        """
        Append a input, and a target or a list of [[input, target], ...].

        >>> ds = Dataset()
        >>> ds.append([0, 0], [0])
        >>> ds.append([0, 1], [1])
        >>> ds.append([1, 0], [1])
        >>> ds.append([1, 1], [0])
        >>> len(ds)
        4
        >>> ds.clear()
        >>> len(ds)
        0
        >>> ds.append([[[0, 0], [0]],
        ...            [[0, 1], [1]],
        ...            [[1, 0], [1]],
        ...            [[1, 1], [0]]])
        >>> len(ds)
        4
        >>> ds.append([[[0, 0], [0]],
        ...            [[0, 1], [1]],
        ...            [[1, 0], [1]],
        ...            [[1, 1], [0]]])
        >>> len(ds)
        8
        """
        if inputs is None:
            self._load(pairs, mode="append")
        else:
            self._add(pairs, inputs) ## really inputs and targets

    def _load(self, pairs=None, inputs=None, targets=None, labels=None, mode=None):
        """
        Set the human-specified dataset to a proper keras dataset.

        Multi-inputs or multi-targets must be: [vector, vector, ...] for each layer input/target pairing.

        Note:
            If you have images in your dataset, they must match K.image_data_format().

        See also :any:`matrix_to_channels_last` and :any:`matrix_to_channels_first`.
        """
        if isinstance(pairs, types.GeneratorType) and not isinstance(inputs, numbers.Integral):
            raise Exception("load with a generator also requires integer number to load")
        elif isinstance(pairs, types.GeneratorType) and isinstance(inputs, numbers.Integral):
            ## do it all here:
            ## create space
            self._inputs = [np.zeros([inputs] + list(self.inputs.get_shape(i)), "float32")
                            for i in range(self._num_input_banks())]
            self._targets = [np.zeros([inputs] + list(self.targets.get_shape(i)), "float32")
                             for i in range(self._num_target_banks())]
            self._labels = ["" for i in range(inputs)] ## convert to numpy at end
            count = 0
            for line in pairs:
                ## at least inputs, targets:
                if self._num_input_banks() == 1:
                    self._inputs[0][count] = line[0]
                else:
                    for i in range(self._num_input_banks()):
                        self._inputs[i][count] = line[0][i]
                if self._num_target_banks() == 1:
                    self._targets[0][count] = line[1]
                else:
                    for i in range(self._num_target_banks()):
                        self._targets[i][count] = line[1][i]
                if len(line) == 3: ## inputs, targets, labels
                    self._labels[count] = line[2]
                count += 1
                if count == inputs:
                    break
            self._labels = [np.array(self._labels, dtype=str)]
            self._cache_values()
            return
        ## else, either pairs=[[[inputs...], [targets...]]...] or pairs=inputs, inputs=targets
        elif inputs is not None:
            if targets is not None:
                if pairs is not None:
                    raise Exception("Use pairs or inputs/targets but not both")
                if labels is not None:
                    pairs = list(zip(inputs, targets, labels))
                else:
                    pairs = list(zip(inputs, targets))
            else:
                raise Exception("you cannot set inputs without targets")
                ## FIXME: make more general, and set targets to [None, None, ...]
                # if self._num_input_banks() > 1: ## for incoming format
                #     ins = []
                #     for i in range(len(ins[0])):
                #         ins.append(np.array([row[i] for row in inputs], "float32"))
                # else:
                #     ins = [np.array(inputs, "float32")]
                # if len(self._inputs) == 0:
                #     self._inputs = ins
                # else:
                #     for i in range(len(self._inputs)):
                #         self._inputs[i] = np.append(self._inputs[i], ins[i], 0)
                # self._cache_values()
                # return
        elif targets is not None:
            raise Exception("you cannot set targets without inputs")
        if pairs is None:
            raise Exception("you need to call with pairs or with input/targets")
        ## first we check the form of the inputs and targets:
        try:
            length = len(pairs)
        except:
            try:
                pairs = [_ for _ in pairs]
                length = len(pairs)
            except:
                raise Exception("pairs should be a sequence, zip, or generator")
        if length == 0:
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
            dtype = get_dtype(inputs[0]) or "float32"
            inputs = [np.array([bank], dtype) for bank in inputs[0]]
            ## Predict:
            try:
                prediction = self.network.model.predict(inputs, batch_size=1)
            except:
                raise Exception("Invalid input form: %s did not propagate through network" % (inputs,))
            ## NOTE: output of targets varies by number of target banks!!!
            if self._num_target_banks() > 1:
                dtype = get_dtype(targets[0]) or "float32"
                targets = [np.array([bank], dtype) for bank in targets[0]]
                for i in range(len(targets[0])):
                    shape = targets[0][i].shape
                    if prediction[0][i].shape != shape:
                        raise Exception("Invalid output shape on bank #%d; got %s, expecting %s" % (i, shape, prediction[0][i].shape))
            else:
                dtype = get_dtype(targets[0]) or "float32"
                targets = [np.array(bank, dtype) for bank in targets[0]]
                shape = targets[0].shape
                if prediction[0].shape != shape:
                    raise Exception("Invalid output shape on bank #%d; got %s, expecting %s" % (0, shape, prediction[0].shape))
        if len(self._inputs) > 0 and mode == "load":
            self.clear()
        self.compile(pairs)

    def compile(self, pairs):
        if self._num_input_banks() > 1: ## for incoming format
            inputs = []
            for i in range(len(pairs[0][0])):
                dtype = get_dtype(pairs[0][0][0]) or "float32"
                inputs.append(np.array([x[0][i] for x in pairs], dtype))
        else:
            dtype = get_dtype(pairs[0][0]) or "float32"
            inputs = [np.array([x[0] for x in pairs], dtype)]
        if self._num_target_banks() > 1: ## for incoming format
            targets = []
            dtype = get_dtype(pairs[0][1][0]) or "float32"
            for i in range(len(pairs[0][1])):
                targets.append(np.array([y[1][i] for y in pairs], dtype))
        else:
            dtype = get_dtype(pairs[0][1]) or "float32"
            targets = [np.array([y[1] for y in pairs], dtype)]
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
        11

        >>> ds = Dataset()
        >>> len(ds.datasets())
        11
        """
        if self is None:
            self = Dataset()
        return sorted(self.DATASETS.keys())

    def get(dataset_name=None, *args, **kwargs):
        """
        Get a known dataset by name.

        Must be called from the Dataset class.

        >>> import conx as cx
        >>> print("Downloading..."); ds = cx.Dataset.get("mnist") # doctest: +ELLIPSIS
        Downloading...
        >>> len(ds.inputs)
        70000
        >>> ds.targets[0]
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        >>> ds.clear()
        """
        if not isinstance(dataset_name, str):
            raise Exception("Please use network.get_dataset() or Dataset.get()")
        self = Dataset() ## a dummy to load the datasets
        if dataset_name.lower() in self.DATASETS:
            return self.DATASETS[dataset_name.lower()](*args, **kwargs)
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
        if self._split > 0:
            print("WARNING: dataset split reset to 0", file=sys.stderr)
        self._split = 0
        self._cache_values()

    def _cache_values(self):
        if isinstance(self, VirtualDataset):
            if self.network:
                self.network.test_dataset_ranges()
                self._verify_network_dataset_match()
            return
        self._cache_values_actual()

    def _cache_values_actual(self):
        ## Regular dataset:
        if len(self.inputs) > 0:
            if isinstance(self._inputs[0], (np.ndarray,)):
                self._inputs_range = list(zip([x.min() for x in self._inputs],
                                              [x.max() for x in self._inputs]))
            elif isinstance(self._inputs[0], (list, tuple)):
                self._inputs_range = list(zip([min(x) for x in self._inputs],
                                              [max(x) for x in self._inputs]))
            else: ## assume it is a list of numbers
                self._inputs_range = list(min(self._inputs), max(self._inputs))
        else:
            self._inputs_range = []
        if len(self.targets) > 0:
            if isinstance(self._targets[0], (np.ndarray,)):
                self._targets_range = list(zip([x.min() for x in self._targets],
                                               [x.max() for x in self._targets]))
            elif isinstance(self._targets[0], (list, tuple)):
                self._targets_range = list(zip([min(x) for x in self._targets],
                                               [max(x) for x in self._targets]))
            else: ## assume it is a list of numbers
                self._targets_range = list(min(self._targets), max(self._targets))
        else:
            self._targets_range = []
        ## Set shape cache:
        if len(self._inputs) > 0:
            self._input_shapes = [shape(x[0]) for x in self._inputs]
        if len(self._targets) > 0:
            self._target_shapes = [shape(x[0]) for x in self._targets]
        # Final checks:
        if len(self.inputs) != len(self.targets):
            self.warn_once("WARNING: inputs/targets lengths do not match")
        if self.network:
            self.network.test_dataset_ranges()
            self._verify_network_dataset_match()

    def _verify_network_dataset_match(self):
        """
        """
        warning = False
        if (self.network is None) or (self.network.model is None or len(self) == 0):
            return ## Nothing to test
        ## check to see if number of input banks match
        if len(self.network.input_bank_order) != self._num_input_banks():
            warning = True
            self.warn_once("WARNING: number of dataset input banks != network input banks in network '%s'" % self.network.name, "VERIFY")
        if len(self.inputs) > 0 and not isinstance(self, VirtualDataset):
            try:
                self.network.propagate(self.inputs[0])
            except:
                warning = True
                self.warn_once("WARNING: dataset does not yet work with network '%s'" % self.network.name, "VERIFY")
        ## check to see if number of output banks match
        if len(self.network.output_bank_order) != self._num_target_banks():
            warning = True
            self.warn_once("WARNING: number of dataset target banks != network output banks in network '%s'" % self.network.name, "VERIFY")
        if not warning and self.warned("VERIFY"):
            self.warn_once("INFO: dataset now works with network '%s'" % self.network.name)

    def warned(self, category):
        """
        Has the user been warned about this category of error before?
        """
        return category in self.warn_categories

    def warn_once(self, message, category=None):
        """
        Warning the user just once about this particular message.
        """
        if category:
            self.warn_categories[category] = True
        if message not in self.warnings:
            print(message, file=sys.stderr)
            self.warnings[message] = True

    def set_labels_from_function(self, function, length=None):
        """
        The function should take an index from 0 to length - 1,
        where length is given, or, if not, is the length of the
        inputs/targets. The function should return a string
        that represents the category of the input/target
        pair.

        >>> ds = Dataset()
        >>> ds.load([[[0, 0], [0]],
        ...          [[0, 1], [1]],
        ...          [[1, 0], [1]],
        ...          [[1, 1], [0]]])

        >>> ds.set_labels_from_function(lambda i: "0" if ds.targets[i] == [0] else "1")
        >>> ds.labels[:]
        ['0', '1', '1', '0']

        """
        length = length if length is not None else len(self)
        labels = [function(i) for i in range(length)]
        self._labels = [np.array(labels, dtype=str)]

    def set_targets_from_function(self, function, length=None):
        """
        The function should take an index from 0 to length - 1,
        where length is given, or, if not, is the length of the
        inputs/targets. The function should return a list or matrix
        that is the target pattern.

        >>> import conx as cx
        >>> ds = cx.Dataset()
        >>> ds.set_inputs_from_function(lambda i: cx.binary(i, 2), length=4)
        >>> def xor(i1, i2):
        ...     return 1 if ((i1 or i2) and not (i1 and i2)) else 0
        >>> ds.set_targets_from_function(lambda i: [xor(*ds.inputs[i])])
        >>> ds.set_labels_from_function(lambda i: "0" if ds.targets[i] == [0] else "1")
        >>> ds.labels[:]
        ['0', '1', '1', '0']
        """
        length = length if length is not None else len(self)
        targets = [function(i) for i in range(length)]
        self._targets = [np.array(targets)]
        self._cache_values()

    def set_inputs_from_function(self, function, length=None):
        """
        The function should take an index from 0 to length - 1,
        where length is given, or, if not, is the length of the
        inputs/targets. The function should return a list or matrix
        that is the input pattern.

        >>> import conx as cx

        >>> ds = cx.Dataset()
        >>> ds.set_inputs_from_function(lambda i: cx.binary(i, 2), length=4)
        >>> def xor(i1, i2):
        ...     return 1 if ((i1 or i2) and not (i1 and i2)) else 0
        >>> ds.set_targets_from_function(lambda i: [xor(*ds.inputs[i])])
        >>> ds.set_labels_from_function(lambda i: "0" if ds.targets[i] == [0] else "1")
        >>> ds.labels[:]
        ['0', '1', '1', '0']
        """
        length = length if length is not None else len(self)
        inputs = [function(i) for i in range(length)]
        self._inputs = [np.array(inputs)]
        self._cache_values()

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
        return self.make_info()

    def __repr__(self):
        return self.make_info()

    def make_info(self):
        retval = ""
        if self.name:
            name = self.name
        elif self.network:
            name = "Dataset for %s" % self.network.name
        else:
            name = "Unnamed Dataset"
        retval += "**Dataset**: %s\n\n" % name
        if self.description is not None:
            retval += self.description
            retval += "\n"
        size, num_train, num_test = self._get_split_sizes()
        retval += '**Information**:\n'
        retval += '   * name    : %s\n' % (self.name,)
        retval += '   * length  : %s\n' % (size,)
        retval += '\n'
        retval += '**Input Summary**:\n'
        if size != 0:
            if len(self.inputs.shape) == 1:
                retval += '   * shape  : %s\n' % (self.inputs.shape[0],)
                retval += '   * range  : %s\n\n' % (self._inputs_range[0],)
            else:
                retval += '   * shape  : %s\n' % (self.inputs.shape,)
                retval += '   * range  : %s\n\n' % (self._inputs_range,)
        retval += '**Target Summary**:\n'
        if size != 0:
            if len(self.targets.shape) == 1:
                retval += '   * shape  : %s\n' % (self.targets.shape[0],)
                retval += '   * range  : %s\n\n' % (self._targets_range[0],)
            else:
                retval += '   * shape  : %s\n' % (self.targets.shape,)
                retval += '   * range  : %s\n\n' % (self._targets_range,)
        if self.network:
            self.network.test_dataset_ranges()
        return retval

    def info(self):
        """
        Print out high-level information about the dataset.
        """
        return display(self)

    def summary(self):
        size, num_train, num_test = self._get_split_sizes()
        retval = ''
        retval += ('_' * 65) + "\n"
        if self.name:
            name = self.name
        elif self.network:
            name = "%s Dataset" % self.network.name
        else:
            name = "Unnamed Dataset"
        template = '%-10s  %-20s  %-30s\n'
        retval +=  "%s:\n" % (name,)
        if size != 0:
            retval +=  template % ("Patterns", "Shape", "Range",)
            retval += ('=' * 65) + "\n"
            if len(self.inputs.shape) == 1:
                retval += template % ("inputs", self.inputs.shape[0], self._inputs_range[0],)
            else:
                retval += template % ("inputs", self.inputs.shape, self._inputs_range,)
            if len(self.targets.shape) == 1:
                retval += template % ("targets", self.targets.shape[0], self._targets_range[0],)
            else:
                retval += template % ("targets", self.targets.shape, self._targets_range,)
        retval += ('=' * 65) + "\n"
        retval += 'Total patterns: %d\n' % (size,)
        retval += '   Training patterns: %d\n' % (num_train,)
        retval += '   Testing patterns: %d\n' % (num_test,)
        retval += ('_' * 65)
        print(retval)
        if self.network:
            self.network.test_dataset_ranges()

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
        if 0 < self._split < 1:
            print("WARNING: reshuffling all data; test data has changed", file=sys.stderr)

    def split(self, split=None):
        """Splits the inputs/targets into training and validation sets.
        The split keyword parameter specifies what portion of the dataset
        to use for validation. It can be a fraction in the range
        [0,1), or an integer number of patterns from 0 to the dataset
        size, or 'all'. For example, a split of 0.25 reserves the last
        1/4 of the dataset for validation.  A split of 1.0 (specified
        as 'all' or an int equal to the dataset size) is a special
        case in which the entire dataset is used for both training and
        validation.
        """
        if split is None:
            size, num_train, num_test = self._get_split_sizes()
            return (num_train, num_test)
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

    def chop(self, amount):
        """Chop off the specified amount of input and target patterns from the
        dataset, starting from the end. Amount can be a fraction in the range
        0-1, or an integer number of patterns to drop.

        >>> import conx as cx
        >>> print("Downloading..."); dataset = cx.Dataset.get("mnist") # doctest: +ELLIPSIS
        Downloading...
        >>> len(dataset)
        70000
        >>> dataset.chop(10000)
        >>> len(dataset)
        60000
        >>> dataset.split(0.25)
        >>> dataset.split()
        (45000, 15000)
        >>> dataset.chop(0.10)
        >>> dataset.split()
        (54000, 0)

        >>> dataset.clear()
        """
        if len(self.inputs) == 0:
            raise Exception("no dataset loaded")
        if isinstance(amount, numbers.Integral):
            if not 0 <= amount < len(self.inputs):
                raise Exception("out of range: %d" % amount)
        elif isinstance(amount, numbers.Real):
            if not 0 <= amount < 1:
                raise Exception("not in the interval [0,1): %s" % amount)
            amount = int(len(self.inputs) * amount)
        else:
            raise Exception("invalid value: %s" % (amount,))
        new_size = self._get_size() - amount
        self._inputs = [self._inputs[b][:new_size] for b in range(self._num_input_banks())]
        self._targets = [self._targets[b][:new_size] for b in range(self._num_target_banks())]
        if len(self._labels) != 0:
            self._labels = [self._labels[b][:new_size] for b in range(self._num_target_banks())]
        if self._split > 0:
            print("WARNING: dataset split reset to 0", file=sys.stderr)
        self._split = 0

    def _get_input(self, i):
        """
        Get an input from the internal dataset and
        format it in the human API.
        """
        class DataVectorTopList(list):
            def __setitem__(self, pos, value):
                raise Exception("cannot assign to DataVector[item][position]; assign instead to DataVector[item]")
        size = self._get_size()
        if not 0 <= i < size:
            raise Exception("input index %d is out of bounds" % (i,))
        else:
            data = DataVectorTopList([self._tolist(self._inputs[b][i], "inputs", b) for b in range(self._num_input_banks())])
        if self._num_input_banks() == 1:
            return data[0]
        else:
            return data

    def _tolist(self, nparray, item, bank):
        class DataVectorList(list):
            """
            """
            def __init__(self, lyst, network, item, bank):
                super().__init__(lyst)
                self.network = network
                self.item = item
                self.bank = bank

            def __setitem__(self, pos, value):
                raise Exception("cannot assign to DataVector[item][position]; assign instead to DataVector[item]")

            def _repr_image_(self):
                if self.network:
                    config = {"pixels_per_unit": self.network.config["pixels_per_unit"],
                              "svg_rotate": False}
                    if self.item.endswith("inputs"):
                        if self.bank < len(self.network.input_bank_order):
                            layer_name = self.network.input_bank_order[self.bank]
                        else:
                            return array_to_image(self)
                    elif self.item.endswith("targets"):
                        if self.bank < len(self.network.output_bank_order):
                            layer_name = self.network.output_bank_order[self.bank]
                        else:
                            return array_to_image(self)
                    else:
                        raise Exception("DataVectorList display error: I don't know how to display %s" % self.item)
                    return self.network[layer_name].make_image(np.array(self), config=config)
                else:
                    return array_to_image(self)

        return DataVectorList(nparray.tolist(), self.network, item, bank)

    def _get_target(self, i):
        """
        Get a target from the internal dataset and
        format it in the human API.
        """
        size = self._get_size()
        if not 0 <= i < size:
            raise Exception("target index %d is out of bounds" % (i,))
        data = [self._tolist(self._targets[b][i], "targets", b) for b in range(self._num_target_banks())]
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
        data = [self._tolist(self._inputs[b][i], "train_inputs", b) for b in range(self._num_input_banks())]
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
        data = [self._tolist(self._targets[b][i], "train_targets", b) for b in range(self._num_target_banks())]
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
        data = [self._tolist(self._inputs[b][j], "test_inputs", b) for b in range(self._num_input_banks())]
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
        data = [self._tolist(self._targets[b][j], "test_targets", b) for b in range(self._num_target_banks())]
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

class VirtualDataVector(DataVector):
    def __getitem__(self, pos):
        if isinstance(pos, slice):
            return [self[i] for i in range(len(self))[pos]]
        ## Handle negative positions:
        if isinstance(pos, int) and pos < 0:
            pos = len(self) + pos
        original_pos = None
        if self.item.startswith("test_"):
            ## dataset.test_inputs[pos]
            if self.dataset._split == 0: ## no test_ patterns
                raise Exception("no test data; use dataset.split(VALUE)")
            elif self.dataset._split == 1.0: ## same as regular, nothing to do
                pass
            else: ## self._split is between 0 and 1
                size, num_train, num_test = self.dataset._get_split_sizes()
                original_pos = pos
                pos = num_train + pos
        ## if pos is not in cached values:
        cache = int(np.floor(pos / self.dataset._cache_size))
        if not(self.dataset._cached[0] <= pos < self.dataset._cached[1]):
            ## find the min/max cache_size around pos, and fill:
            self.dataset._cached = (cache * self.dataset._cache_size,
                                    min((cache + 1)*self.dataset._cache_size, len(self.dataset)))
            if cache != (self.dataset._current_cache + 1) and self.dataset._generator_ordered:
                self.dataset._generator = self.dataset._function(self.dataset)
                ## first, find the set to skip over, if needed:
                if self.dataset._load_cache_direct:
                    ## skip entire caches:
                    for i in range(0, cache):
                        ## next cache
                        next(self.dataset._generator)
                else:
                    ## skip one pos at a time:
                    for i in range(0, self.dataset._cached[0]):
                        ## next pos
                        next(self.dataset._generator)
            ## now fill the cached values with these:
            if self.dataset._load_cache_direct:
                if self.dataset._generator:
                    ## generator
                    all_data = next(self.dataset._generator)
                else:
                    ## function:
                    all_data = self.dataset._generate(self.dataset, cache)
                if all_data is not None:
                    self.dataset._inputs = all_data[0]
                    self.dataset._targets = all_data[1]
                # else, assumes that you did it through self._inputs = ... in function/generator
            else:
                all_inputs = []
                all_targets = []
                for i in range(self.dataset._cached[0],
                               self.dataset._cached[1]):
                    if self.dataset._generator:
                        inputs, targets = next(self.dataset._generator)
                    else:
                        inputs, targets = self.dataset._generate(self.dataset, i)
                    all_inputs.append(inputs)
                    all_targets.append(targets)
                self.dataset._inputs = []
                self.dataset._targets = []
                self.dataset.compile(list(zip(all_inputs, all_targets)))
        ## Do math to get the actual position:
        self.dataset._current_cache = cache
        if original_pos is None:
            return super().__getitem__(pos - (cache * self.dataset._cache_size))
        else: ## test_ data:
            size, num_train, num_test = self.dataset._get_split_sizes()
            pos = (original_pos + num_train) % self.dataset._cache_size
            self.item = self.item[5:] # "test_" ...
            retval = super().__getitem__(pos)
            self.item = "test_" + self.item
            return retval

    def __len__(self):
        size, num_train, num_test = self.dataset._get_split_sizes()
        if self.item.startswith("test_"):
            return num_test
        elif self.item.startswith("train_"):
            return num_train
        else:
            return size

    def reshape(self, *args, **kwargs):
        print("WARNING: applying to virtual dataset, this cache only!",
              file=sys.stderr)
        super().reshape(*args, **kwargs)

    ## FIXME: any other virtual functions that should get a datavector warning?

class VirtualDataset(Dataset):
    """
    Create a virtual dataset. VirtualData set takes:

    * a function that is either:
      * a generator function (no arguments) that returns an
        input and output if load_cache_direct is False, otherwise
        should return a low-level numpy input and target
        cache, one np.array() for each input/target bank
      * a generate(pos OR cache) function that takes a position
        (if load_cache_direct) or cache (if not load_cache_direct). If
        load_cache_direct, the function(cache) returns a low-level
        numpy input and target cache, one np.array() for each
        input/target bank. If not load_cache_direct, return an
        input/target pair.
    * shape of inputs
    * shape of targets
    * ranges of inputs
    * ranges of targets

    Optional:

    * cache_size - the size of the cache
    * load_cache_direct - if True, must generator function must return or set
      the low-level list of np.arrays for inputs and targets
    * generator_ordered - if True, the data is not ordered (e.g., random)
    * network - if None, use network.set_dataset() later
    * name - name of dataset
    * description - descriptive text, in Markdown

    Examples:

    >>> import conx as cx
    >>> import numpy as np
    >>> import random

    >>> def f(self, cache):
    ...     pos1 = cache * self._cache_size
    ...     pos2 = (cache + 1) * self._cache_size
    ...     inputs = np.array([[pos, pos] for pos in range(pos1, pos2)])
    ...     targets = np.array([[pos] for pos in range(pos1, pos2)])
    ...     return [inputs], [targets]

    >>> ds = cx.VirtualDataset(f, 1000, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                        load_cache_direct=True, cache_size=20)

    >>> ds.split(.25)

    >>> ds.inputs[0][0] == 0.0, ds.inputs[-1][0] == 999
    (True, True)

    >>> ds.train_inputs[0][0] == 0, ds.train_inputs[-1][0] == 749
    (True, True)

    >>> ds.test_inputs[0][0] == 750, ds.test_inputs[-1][0] == 999
    (True, True)

    >>> def f(self, pos):
    ...     return [pos, pos], [pos]

    >>> ds = cx.VirtualDataset(f, 1000, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                        load_cache_direct=False, cache_size=20)

    >>> ds.split(.25)

    >>> ds.inputs[0][0] == 0.0, ds.inputs[-1][0] == 999
    (True, True)

    >>> ds.train_inputs[0][0] == 0, ds.train_inputs[-1][0] == 749
    (True, True)

    >>> ds.test_inputs[0][0] == 750, ds.test_inputs[-1][0] == 999
    (True, True)

    >>> def test_dataset(net):
    ...     net.dataset.split(.1)
    ...     net.train(1, accuracy=1.0, report_rate=10, plot=False, batch_size=8)
    ...     ds = net.dataset
    ...     i0t0 = ds.inputs[0]
    ...     i50t0 = ds.inputs[50]
    ...     i0t1 = ds.inputs[0]
    ...     if ds._generator:
    ...         if ds._generator_ordered:
    ...             assert i0t0 == i0t1, "item 0 should be the same when generator_ordered"
    ...         else:
    ...             assert i0t0 != i0t1, "item 0 should be diff when not generator_ordered"
    ...     else: # function
    ...             assert i0t0 == i0t1, "item 0 should be the same with this function"
    ...     assert i0t0 != i50t0, "items 0 and 50 should be different"


    generator_ordered = False, load_cache_direct = False, with function(position):

    >>> def f(self, pos):
    ...    return ([pos/100, pos/100], [pos/100])
    >>> dataset = cx.VirtualDataset(f, 100, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                             load_cache_direct=False,
    ...                             cache_size=16)
    >>> net = cx.Network("GEN", 2, 3, 1, activation="sigmoid")
    >>> net.compile(error="mse", optimizer="adam")
    >>> net.set_dataset(dataset)
    >>> test_dataset(net) # doctest: +ELLIPSIS
    Evaluating initial ...

    generator_ordered = False, load_cache_direct = True, with function(cache):

    >>> def f(self, cache):
    ...     i = cache * 16
    ...     while True:
    ...         all_inputs = [[]]
    ...         all_targets = [[]]
    ...         while i < (cache + 1) * 16:
    ...             all_inputs[0].append([i/100, i/100])
    ...             all_targets[0].append([i/100])
    ...             i += 1
    ...         return ([np.array(inputs) for inputs in all_inputs],
    ...                 [np.array(targets) for targets in all_targets])
    >>> dataset = cx.VirtualDataset(f, 100, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                             cache_size=16)
    >>> net = cx.Network("GEN", 2, 3, 1, activation="sigmoid")
    >>> net.compile(error="mse", optimizer="adam")
    >>> net.set_dataset(dataset)
    >>> test_dataset(net) # doctest: +ELLIPSIS
    Evaluating initial ...

    generator_ordered = True, load_cache_direct = False, with generator function():

    >>> def f(self):
    ...     i = 0
    ...     while True:
    ...         yield ([i/100, i/100], [i/100])
    ...         i += 1
    >>> dataset = cx.VirtualDataset(f, 100, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                             generator_ordered=True,
    ...                             load_cache_direct=False,
    ...                             cache_size=16)
    >>> net = cx.Network("GEN", 2, 3, 1, activation="sigmoid")
    >>> net.compile(error="mse", optimizer="adam")
    >>> net.set_dataset(dataset)
    >>> test_dataset(net) # doctest: +ELLIPSIS
    Evaluating initial ...

    generator_ordered = True, load_cache_direct = True, with generator function():

    >>> def f(self):
    ...     i = 0
    ...     while True:
    ...         i_16 = i + 16
    ...         all_inputs = [[]]
    ...         all_targets = [[]]
    ...         while i < i_16:
    ...             all_inputs[0].append([i/100, i/100])
    ...             all_targets[0].append([i/100])
    ...             i += 1
    ...         yield ([np.array(inputs) for inputs in all_inputs],
    ...                [np.array(targets) for targets in all_targets])
    >>> dataset = cx.VirtualDataset(f, 100, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                             generator_ordered=True,
    ...                             cache_size=16)
    >>> net = cx.Network("GEN", 2, 3, 1, activation="sigmoid")
    >>> net.compile(error="mse", optimizer="adam")
    >>> net.set_dataset(dataset)
    >>> test_dataset(net) # doctest: +ELLIPSIS
    Evaluating initial ...

    generator_ordered = True, load_cache_direct = False, with generator function() (showing another
    function style):

    >>> def f(self):
    ...     for i in range(100):
    ...         yield ([i/100, i/100], [i/100])
    >>> dataset = cx.VirtualDataset(f, 100, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                             generator_ordered=True,
    ...                             load_cache_direct=False,
    ...                             cache_size=16)
    >>> net = cx.Network("GEN", 2, 3, 1, activation="sigmoid")
    >>> net.compile(error="mse", optimizer="adam")
    >>> net.set_dataset(dataset)
    >>> test_dataset(net) # doctest: +ELLIPSIS
    Evaluating initial ...

    generator_ordered = False, load_cache_direct = False, with generator function():

    >>> def f(self):
    ...     while True:
    ...         r = random.random()
    ...         yield ([r, r], [r])
    >>> dataset = cx.VirtualDataset(f, 100, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                             load_cache_direct=False,
    ...                             cache_size=16)
    >>> net = cx.Network("GEN", 2, 3, 1, activation="sigmoid")
    >>> net.compile(error="mse", optimizer="adam")
    >>> net.set_dataset(dataset)
    >>> test_dataset(net) # doctest: +ELLIPSIS
    Evaluating initial ...

    generator_ordered = False, load_cache_direct = True, with generator function():

    >>> def f(self):
    ...     while True:
    ...         all_inputs = [[]]
    ...         all_targets = [[]]
    ...         for i in range(16):
    ...             r = random.random()
    ...             all_inputs[0].append([r, r])
    ...             all_targets[0].append([r])
    ...         yield ([np.array(inputs) for inputs in all_inputs],
    ...                [np.array(targets) for targets in all_targets])
    >>> dataset = cx.VirtualDataset(f, 100, [(2,)], [(1,)], [(0,1)], [(0,1)],
    ...                             cache_size=16)
    >>> net = cx.Network("GEN", 2, 3, 1, activation="sigmoid")
    >>> net.compile(error="mse", optimizer="adam")
    >>> net.set_dataset(dataset)
    >>> test_dataset(net) # doctest: +ELLIPSIS
    Evaluating initial ...
    """
    Vector = VirtualDataVector

    def __init__(self, function, length, input_shapes, target_shapes,
                 inputs_range, targets_range, generator_ordered=False, cache_size=3200,
                 load_cache_direct=True, network=None, name=None, description=None):
        super().__init__(network, name, description, input_shapes, target_shapes)
        self._function = function
        self._length = length
        self._generator_ordered = generator_ordered
        self._cache_size = cache_size
        self._inputs_range = inputs_range
        self._targets_range = targets_range
        self._load_cache_direct = load_cache_direct
        self.reset()

    def chop(self, amount):
        if isinstance(amount, numbers.Integral):
            if not 0 <= amount < len(self.inputs):
                raise Exception("out of range: %d" % amount)
        elif isinstance(amount, numbers.Real):
            if not 0 <= amount < 1:
                raise Exception("not in the interval [0,1): %s" % amount)
            amount = int(len(self.inputs) * amount)
        self._length = self._length - amount
        if self._split > 0:
            print("WARNING: dataset split reset to 0", file=sys.stderr)
        self._split = 0

    def _get_size(self):
        return self._length

    def reset(self):
        try:
            self._generator = self._function(self)
            self._generate = False
        except:
            self._generator = None
            self._generate = self._function
        if self._generator_ordered and self._generator is None:
            raise Exception("generator_order should only be True when function is a generator")
        self._cached = [-1, -1]
        self._current_cache = -1
        ## Force a load of first cache:
        self.inputs[0]

    def get_generator(self, batch_size):
        if self._cache_size % batch_size != 0:
            raise Exception("cache size (%s) is not a multiple of batch_size (%s)" % (self._cache_size, batch_size))
        return DatasetGenerator(self)

    def get_validation_generator(self, batch_size):
        if self._cache_size % batch_size != 0:
            raise Exception("cache size (%s) is not a multiple of batch_size (%s)" % (self._cache_size, batch_size))
        if self._split == 0:
            return None ## there is no validation partition
        else:
            ## split can be 1.0 (use all), or > 0 and < 1
            return DatasetGenerator(self, validation_set=True)

    def set_targets_from_labels(self, *args, **kwargs):
        print("WARNING: applying to virtual dataset, this cache only!",
              file=sys.stderr)
        super().set_targets_from_labels(*args, **kwargs)

    def set_targets_from_inputs(self, *args, **kwargs):
        print("WARNING: applying to virtual dataset, this cache only!",
              file=sys.stderr)
        super().set_targets_from_inputs(*args, **kwargs)

    def set_inputs_from_targets(self, *args, **kwargs):
        print("WARNING: applying to virtual dataset, this cache only!",
              file=sys.stderr)
        super().set_inputs_from_targets(*args, **kwargs)

    ## FIXME: any other virtual functions that should get a dataset warning?

class DatasetGenerator(keras.utils.Sequence):
    """
    DatasetGenerator takes a VirtualDataset and can be trained
    without using a lot of memory.

    See `VirtualDataset` for more information.

    """
    def __init__(self, dataset, validation_set=False):
        """
        dataset - a VirtualDataset
        """
        self.dataset = dataset
        self.validation_set = validation_set

    def __len__(self):
        if self.validation_set == False or self.dataset._split == 1:
            return int(np.ceil(len(self.dataset) / float(self.dataset._cache_size)))
        else:
            return int(np.ceil(len(self.dataset.test_inputs) / float(self.dataset._cache_size)))

    def __getitem__(self, cache):
        if self.validation_set == False or self.dataset._split == 1:
            i = cache * self.dataset._cache_size
            self.dataset.inputs[i] ## force to load in _inputs/_targets
            return (self.dataset._inputs, self.dataset._targets)
        else:
            i = cache * self.dataset._cache_size
            self.dataset.test_inputs[i] ## force to load in _inputs/_targets
            return (self.dataset._inputs, self.dataset._targets)

class H5Dataset(VirtualDataset):
    """
    A h5py dataset wrapper.
    """

    def __init__(self, f, filename, key, input_banks, target_banks, cache_size=3200,
                 name=None, description=None, network=None, load_cache_direct=True,
                 length=None, endpoint=None, username=None, password=None, api_key=None,
                 use_session=True, use_cache=False):
        """

        Fonts from:
        https://erikbern.com/2016/01/21/analyzing-50k-fonts-using-deep-neural-networks.html

        >>> def f(self, pos):
        ...     return self.h5[self.key][pos][1], self.h5[self.key][pos][1] # where 1 is B

        >>> def get_cache(self, cache):
        ...     pos = cache * self._cache_size
        ...     b = self.h5[self.key][pos:pos + self._cache_size,1,:] # where 1 is the B's
        ...     self._inputs = [b]
        ...     self._targets = [b]

        >>> import os
        >>> font_file = os.path.expanduser("~/.keras/datasets/fonts.hdf5")

        >>> if os.path.exists(font_file):
        ...     ds1 = H5Dataset(f, font_file, "fonts", 1, 1, name="Fonts", load_cache_direct=False)
        ...     ds1.close()
        ...     ds2 = H5Dataset(get_cache, font_file, "fonts", 1, 1, name="Fonts", cache_size=32)
        ...     ds2.close()

        >>> try:
        ...     import h5pyd
        ... except:
        ...     h5pyd = None

        >>> if h5pyd:
        ...     ds3 = H5Dataset(get_cache, "fonts.public.hdfgroup.org", "fonts", 1, 1, name="Fonts", cache_size=32,
        ...                     endpoint="http://192.168.1.105:5000")
        ...     assert len(ds3.inputs) == 56443
        ...     ds3.close()
        """
        import h5py
        self.key = key
        if endpoint is not None:
            import h5pyd
            self.h5 = h5pyd.File(filename, 'r', endpoint=endpoint, username=username,
                                 password=password, api_key=api_key, use_session=use_session,
                                 use_cache=use_cache)
        else:
            self.h5 = h5py.File(filename)
        super().__init__(f, length if length is not None else len(self.h5[self.key]),
                         generator_ordered=False,
                         load_cache_direct=load_cache_direct,
                         cache_size=cache_size,
                         name=name, description=description, network=network,
                         ## dummy values, for now:
                         input_shapes=[(None,) for i in range(input_banks)],
                         target_shapes=[(None,) for i in range(target_banks)],
                         inputs_range=[(0,1) for i in range(input_banks)],
                         targets_range=[(0,1) for i in range(target_banks)])
        ## Now fill in dummy values:
        ## Normally, don't call this with a virtual dataset, but we want shapes and ranges
        ## so we use first cache as sample:
        self._cache_values_actual()
        ## FIXME: this good, sample more, or get stats from all?

    def close(self):
        """
        Close the virtual file connection.
        """
        self.h5.close()
