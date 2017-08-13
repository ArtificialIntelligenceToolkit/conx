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

from .utils import valid_shape

class _DataVector():
    def __init__(self, dataset, item):
        self.dataset = dataset
        self.item = item

    def __getitem__(self, pos):
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

    def __len__(self):
        if self.item == "targets":
            return self.dataset._get_targets_length()
        elif self.item == "inputs":
            return self.dataset._get_inputs_length()
        elif self.item == "test_inputs":
            return self.dataset._get_test_inputs_length()
        elif self.item == "train_inputs":
            return self.dataset._get_train_inputs_length()
        elif self.item == "test_targets":
            return self.dataset._get_test_targets_length()
        elif self.item == "train_targets":
            return self.dataset._get_train_targets_length()
        else:
            raise Exception("unknown vector type: %s" % (item,))

    def __repr__(self):
        return "<Dataset %s vector>" % self.item

class Dataset():
    """
    Contains the dataset, and metadata about it.

    input_shapes = [shape, ...]
    target_shapes = [shape, ...]
    """
    def __init__(self, pairs=None, inputs=None, targets=None):
        self._num_input_banks = 0
        self._num_target_banks = 0
        self._inputs = []
        self._targets = []
        self._labels = []
        self._train_inputs = []
        self._train_targets = []
        self._test_inputs = []
        self._test_targets = []
        self._test_labels = []
        self._train_labels = []
        self._inputs_range = (0,0)
        self._targets_range = (0,0)
        self._num_inputs = 0
        self._num_targets = 0
        self._target_shapes = []
        self._input_shapes = []
        self._split = 0
        if inputs is not None:
            if targets is not None:
                self.load(zip(inputs, targets))
            else:
                raise Exception("you cannot set inputs without targets")
        elif targets is not None:
            raise Exception("you cannot set targets without inputs")
        if pairs:
            self.load(pairs)

    def __getattr__(self, item):
        if item in [
                "inputs", "targets",
                "test_inputs", "test_targets",
                "train_inputs", "train_targets",
        ]:
            return _DataVector(self, item)
        else:
            raise AttributeError()

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
            self._set_input_info(inputs)
        if targets is not None:
            self._set_target_info(targets)
        if labels is not None:
            self._labels = labels # should be a np.array/list of single values
        self._cache_values()
        self.split(self._num_inputs)

    def _set_input_info(self, inputs):
        self._inputs = inputs
        if isinstance(inputs, (list, tuple)): ## multiple inputs
            self._num_input_banks = len(inputs)
            self._input_shapes = [x.shape for x in inputs]
            self._num_inputs = len(inputs[0])
        else:
            self._num_input_banks = 1
            self._input_shapes = [inputs[0].shape]
            self._num_inputs = len(inputs)

    def _set_target_info(self, targets):
        self._targets = targets
        if isinstance(targets, (list, tuple)): ## multiple inputs
            self._num_target_banks = len(targets)
            self._target_shapes = [x.shape for x in targets]
            self._num_targets = len(targets[0])
        else:
            self._num_target_banks = 1
            self._target_shapes = [targets[0].shape]
            self._num_targets = len(targets)

    def load(self, pairs):
        """
        Set the human-specified dataset to a proper keras dataset.

        Multi-inputs or multi-outputs must be: [vector, vector, ...] for each layer input/target pairing.

        Note:
            If you have images in your dataset, they must match K.image_data_format().

        See also :any:`matrix_to_channels_last` and :any:`matrix_to_channels_first`.
        """
        ## Either the inputs/targets are a list of a list -> np.array(...) (np.array() of vectors)
        ## or are a list of list of list -> [np.array(), np.array()]  (list of np.array cols of vectors)
        self._num_input_banks = len(np.array(pairs[0][0]).shape)
        self._num_target_banks = len(np.array(pairs[0][1]).shape)
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
        self._labels = []
        self._set_input_info(inputs)
        self._set_target_info(targets)
        self._cache_values()
        self.split(self._num_inputs)

    @classmethod
    def get_cifar10(cls):
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        inputs = np.concatenate((x_train, x_test))
        targets = np.concatenate((y_train, y_test))
        inputs = inputs.astype('float32')
        targets = targets.astype('float32')
        inputs /= 255
        targets /= 255
        ds = Dataset()
        ds.load_direct(inputs, targets)
        return ds

    @classmethod
    def get_cifar100(cls):
        from keras.datasets import cifar100
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        inputs = np.concatenate((x_train, x_test))
        targets = np.concatenate((y_train, y_test))
        inputs = inputs.astype('float32')
        targets = targets.astype('float32')
        inputs /= 255
        targets /= 255
        ds = Dataset()
        ds.load_direct(inputs, targets)
        return ds

    @classmethod
    def get_mnist(cls):
        """
        Load the Keras MNIST dataset and format it as images.
        """
        from keras.datasets import mnist
        import keras.backend as K
        dataset = Dataset()
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
        dataset.load_direct(inputs, targets, labels)
        return dataset

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
            inputs = [np.array([vector for vector in row[start:stop]]) for row in self._inputs]
        else:
            inputs = self._inputs[start:stop] # ok
        if self._num_target_banks > 1:
            targets = [np.array([vector for vector in row[start:stop]]) for row in self._targets]
        else:
            targets = self._targets[start:stop]
        if len(self._labels) > 0:
            self._labels = self._labels[start:stop]
        self._set_input_info(inputs)
        self._set_target_info(targets)
        self._cache_values()
        self.split(self._num_inputs)

    def _cache_values(self):
        ## ASSUMES all _nums are set!
        if self._num_inputs > 0:
            if self._num_input_banks > 1:
                self._inputs_range = (min([x.min() for x in self._inputs]),
                                     max([x.max() for x in self._inputs]))
            else:
                self._inputs_range = (self._inputs.min(), self._inputs.max())
        else:
            self._inputs_range = (0,0)
        if self._num_inputs > 0:
            if self._num_target_banks > 1:
                self._targets_range = (min([x.min() for x in self._targets]),
                                      max([x.max() for x in self._targets]))
            else:
                self._targets_range = (self._targets.min(), self._targets.max())
        else:
            self._targets_range = (0, 0)
        # Clear any previous settings:
        self._train_inputs = []
        self._train_targets = []
        self._test_inputs = []
        self._test_targets = []
        # Final checks:
        assert len(self.test_inputs) == len(self.test_targets), "test inputs/targets lengths do not match"
        assert len(self.train_inputs) == len(self.train_targets), "train inputs/targets lengths do not match"
        assert len(self.inputs) == len(self.targets), "inputs/targets lengths do not match"

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

    def set_targets_from_inputs(self):
        """
        Copy the inputs to targets
        """
        self._set_target_info(copy.copy(self._inputs))

    def set_inputs_from_targets(self):
        """
        Copy the targets to inputs
        """
        self._set_input_info(copy.copy(self._targets))

    def reshape_inputs(self, new_shape):
        """
        Reshape the input vectors. WIP.
        """
        ## FIXME: allow working on multi inputs
        if self._num_input_banks > 1:
            raise Exception("reshape_inputs does not yet work on multi-input patterns")
        if self._num_inputs == 0:
            raise Exception("no dataset loaded")
        if isinstance(new_shape, numbers.Integral):
            pass ## ok
        elif not valid_shape(new_shape):
            raise Exception("bad shape: %s" % (new_shape,))
        if isinstance(new_shape, numbers.Integral):
            new_size = self._num_inputs * new_shape
        else:
            new_size = self._num_inputs * reduce(operator.mul, new_shape)
        if new_size != self._inputs.size:
            raise Exception("shape %s is incompatible with inputs" % (new_shape,))
        if isinstance(new_shape, numbers.Integral):
            new_shape = (new_shape,)
        self._set_input_info(self._inputs.reshape((self._num_inputs,) + new_shape))
        self.split(self._split)

    def set_targets_from_labels(self, num_classes):
        """
        Given net.labels are integers, set the net.targets to one_hot() categories.
        """
        ## FIXME: allow working on multi-targets
        if self._num_target_banks > 1:
            raise Exception("set_targets_from_labels does not yet work on multi-target patterns")
        if self._num_inputs == 0:
            raise Exception("no dataset loaded")
        if not isinstance(num_classes, numbers.Integral) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self._targets = keras.utils.to_categorical(self._labels, num_classes).astype("uint8")
        self._train_targets = self._targets[:self._split]
        self._test_targets = self._targets[self._split:]
        print('Generated %d target vectors from %d labels' % (self._num_inputs, num_classes))

    def summary(self):
        """
        Print out a summary of the dataset.
        """
        print('Input Summary:')
        print('   count  : %d (%d for training, %s for testing)' % (
            len(self.inputs), self._split, len(self.inputs) - self._split))
        if len(self.inputs) != 0:
            if self._num_target_banks > 1:
                print('   shape  : %s' % ([x[0].shape for x in self._inputs],))
            else:
                print('   shape  : %s' % (self._inputs[0].shape,))
            print('   range  : %s' % (self._inputs_range,))
        print('Target Summary:')
        print('   count  : %d (%d for training, %s for testing)' % (
            len(self.targets), self._split, len(self.inputs) - self._split))
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
        if self._num_inputs == 0:
            raise Exception("no dataset loaded")
        indices = np.random.permutation(self._num_inputs)
        self._inputs = self._inputs[indices]
        if len(self._labels) != 0:
            self._labels = self._labels[indices]
        if len(self.targets) != 0:
            self._targets = self._targets[indices]
        self.split(self._split)

    def split(self, split=0.50):
        """
        Split the inputs/targets into training/test datasets.
        """
        if self._num_inputs == 0:
            raise Exception("no dataset loaded")
        if isinstance(split, numbers.Integral):
            if not 0 <= split <= self._num_inputs:
                raise Exception("split out of range: %d" % split)
            self._split = split
        elif isinstance(split, numbers.Real):
            if not 0 <= split <= 1:
                raise Exception("split is not in the range 0-1: %s" % split)
            self._split = int(self._num_inputs * split)
        else:
            raise Exception("invalid split: %s" % split)
        if self._num_input_banks > 1:
            self._train_inputs = [col[:self._split] for col in self._inputs]
            self._test_inputs = [col[self._split:] for col in self._inputs]
        else:
            self._train_inputs = self._inputs[:self._split]
            self._test_inputs = self._inputs[self._split:]
        if len(self._labels) != 0:
            self._train_labels = self._labels[:self._split]
            self._test_labels = self._labels[self._split:]
        if len(self.targets) != 0:
            if self._num_target_banks > 1:
                self._train_targets = [col[:self._split] for col in self._targets]
                self._test_targets = [col[self._split:] for col in self._targets]
            else:
                self._train_targets = self._targets[:self._split]
                self._test_targets = self._targets[self._split:]

    def _get_input(self, i):
        """
        Get an input from the internal dataset and
        format it in the human API.
        """
        if self._num_input_banks == 1:
            return list(self._inputs[i])
        else:
            inputs = []
            for c in range(self._num_input_banks):
                inputs.append(list(self._inputs[c][i]))
            return inputs

    def _get_target(self, i):
        """
        Get a target from the internal dataset and
        format it in the human API.
        """
        if self._num_target_banks == 1:
            return list(self._targets[i])
        else:
            targets = []
            for c in range(self._num_target_banks):
                targets.append(list(self._targets[c][i]))
            return targets

    def _get_train_input(self, i):
        """
        Get a training input from the internal dataset and
        format it in the human API.
        """
        if self._num_input_banks == 1:
            return list(self._train_inputs[i])
        else:
            inputs = []
            for c in range(self._num_input_banks):
                inputs.append(list(self._train_inputs[c][i]))
            return inputs

    def _get_train_target(self, i):
        """
        Get a training target from the internal dataset and
        format it in the human API.
        """
        if self._num_target_banks == 1:
            return list(self._train_targets[i])
        else:
            targets = []
            for c in range(self._num_target_banks):
                targets.append(list(self._train_targets[c][i]))
            return targets

    def _get_test_input(self, i):
        """
        Get a test input from the internal dataset and
        format it in the human API.
        """
        if self._num_input_banks == 1:
            return list(self._test_inputs[i])
        else:
            inputs = []
            for c in range(self._num_input_banks):
                inputs.append(list(self._test_inputs[c][i]))
            return inputs

    def _get_test_target(self, i):
        """
        Get a test target from the internal dataset and
        format it in the human API.
        """
        if self._num_target_banks == 1:
            return list(self._test_targets[i])
        else:
            targets = []
            for c in range(self._num_target_banks):
                targets.append(list(self._test_targets[c][i]))
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
        if len(self._test_inputs) == 0:
            return 0
        if self._num_input_banks > 1:
            return self._test_inputs[0].shape[0]
        else:
            return self._test_inputs.shape[0]

    def _get_test_targets_length(self):
        """
        Get the number of test target patterns.
        """
        if len(self._test_targets) == 0:
            return 0
        if self._num_target_banks > 1:
            return self._test_targets[0].shape[0]
        else:
            return self._test_targets.shape[0]

    def _get_train_inputs_length(self):
        """
        Get the number of training input patterns.
        """
        if len(self._train_inputs) == 0:
            return 0
        if self._num_input_banks > 1:
            return self._train_inputs[0].shape[0]
        else:
            return self._train_inputs.shape[0]

    def _get_train_targets_length(self):
        """
        Get the number of training target patterns.
        """
        if len(self._train_targets) == 0:
            return 0
        if self._num_target_banks > 1:
            return self._train_targets[0].shape[0]
        else:
            return self._train_targets.shape[0]
