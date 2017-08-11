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
import numbers

class Dataset():
    """
    Defines input and target's names and shapes.

    inputs = [("name", shape), ...]
    targets = [("name", shape), ...]
    """
    def __init__(self, input_desc=None, target_desc=None):
        self.input_desc = input_desc
        self.target_desc = target_desc
        self.num_input_banks = 0
        self.num_target_banks = 0
        self.inputs = []
        self.targets = []
        self.labels = []
        self.train_inputs = []
        self.train_targets = []
        self.test_inputs = []
        self.test_targets = []
        self.test_labels = []
        self.train_labels = []
        self.inputs_range = (0,0)
        self.targets_range = (0,0)
        self.num_inputs = 0
        self.num_targets = 0
        self._split = 0

## FIXME: add interface for humans: dataset.inputs[0], etc.

    def load_direct(self, inputs, targets):
        """
        Set the inputs/targets in the specific internal format:

        [input-vector, input-vector, ...] if single input layer

        [[input-layer-1-vectors ...], [input-layer-2-vectors ...], ...] if input target layers

        [target-vector, target-vector, ...] if single output layer

        [[target-layer-1-vectors], [target-layer-2-vectors], ...] if multi target layers

        """
        ## Better be in correct format!
        ## each is either: list of np.arrays() [multi], or np.array() [single]
        self.inputs = inputs
        self.targets = targets
        self.labels = []
        self._cache_values()
        self.split(self.num_inputs, verbose=False)

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
        self.num_input_banks = len(np.array(pairs[0][0]).shape)
        self.num_target_banks = len(np.array(pairs[0][1]).shape)
        if self.num_input_banks > 1:
            self.inputs = []
            for i in range(len(pairs[0][0])):
                self.inputs.append(np.array([x[i] for (x,y) in pairs], "float32"))
        else:
            self.inputs = np.array([x for (x, y) in pairs], "float32")
        if self.num_target_banks > 1:
            self.targets = []
            for i in range(len(pairs[0][1])):
                self.targets.append(np.array([y[i] for (x,y) in pairs], "float32"))
        else:
            self.targets = np.array([y for (x, y) in pairs], "float32")
        self.labels = []
        self._cache_values()
        self.split(self.num_inputs, verbose=False)

    @classmethod
    def get_mnist(cls, verbose=True):
        """
        Load the Keras MNIST dataset and format it as images.
        """
        from keras.datasets import mnist
        from keras.utils import to_categorical
        import keras.backend as K
        dataset = Dataset([["input", (28, 28, 1)]])
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
        dataset.inputs = np.concatenate((x_train,x_test))
        dataset.labels = np.concatenate((y_train,y_test))
        dataset.targets = to_categorical(dataset.labels)
        dataset.num_input_banks = 1
        dataset.num_target_banks = 1
        dataset._cache_values()
        dataset.split(dataset.num_inputs, verbose=False)
        return dataset

    def slice(self, start=None, stop=None, verbose=True):
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
                stop = len(self.inputs)
            else: # (None, #)
                start = 0
        if verbose:
            print("Slicing dataset %d:%d..." % (start, stop))
        if self.num_input_banks > 1:
            self.inputs = [np.array([vector for vector in row[start:stop]]) for row in self.inputs]
        else:
            self.inputs = self.inputs[start:stop] # ok
        if self.num_target_banks > 1:
            self.targets = [np.array([vector for vector in row[start:stop]]) for row in self.targets]
        else:
            self.targets = self.targets[start:stop]
        if len(self.labels) > 0:
            self.labels = self.labels[start:stop]
        self._cache_values()
        self.split(self.num_inputs, verbose=False)
        if verbose:
            self.summary()

    def _cache_values(self):
        self.num_inputs = self.get_inputs_length()
        if self.num_inputs > 0:
            if self.num_input_banks > 1:
                self.inputs_range = (min([x.min() for x in self.inputs]),
                                     max([x.max() for x in self.inputs]))
            else:
                self.inputs_range = (self.inputs.min(), self.inputs.max())
        else:
            self.inputs_range = (0,0)
        self.num_targets = self.get_targets_length()
        if self.num_inputs > 0:
            if self.num_target_banks > 1:
                self.targets_range = (min([x.min() for x in self.targets]),
                                      max([x.max() for x in self.targets]))
            else:
                self.targets_range = (self.targets.min(), self.targets.max())
        else:
            self.targets_range = (0, 0)
        # Clear any previous settings:
        self.train_inputs = []
        self.train_targets = []
        self.test_inputs = []
        self.test_targets = []
        # Final checks:
        assert self.get_test_inputs_length() == self.get_test_targets_length(), "test inputs/targets lengths do not match"
        assert self.get_train_inputs_length() == self.get_train_targets_length(), "train inputs/targets lengths do not match"
        assert self.get_inputs_length() == self.get_targets_length(), "inputs/targets lengths do not match"

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
    #         self.inputs = f['data']
    #         self.labels = f['labels']
    #         self.targets = []
    #         if self.get_inputs_length() != len(self.labels):
    #             raise Exception("Dataset contains different numbers of inputs and labels")
    #         if self.get_inputs_length() == 0:
    #             raise Exception("Dataset is empty")
    #         self._cache_dataset_values()
    #         self.split_dataset(self.num_inputs, verbose=False)
    #         if verbose:
    #             self.summary_dataset()
    #     except:
    #         raise Exception("couldn't load .npz dataset %s" % filename)

    def reshape_inputs(self, new_shape, verbose=True):
        """
        Reshape the input vectors. WIP.
        """
        ## FIXME: allow working on multi inputs
        if self.num_input_banks > 1:
            raise Exception("reshape_inputs does not yet work on multi-input patterns")
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if not valid_shape(new_shape):
            raise Exception("bad shape: %s" % (new_shape,))
        if isinstance(new_shape, numbers.Integral):
            new_size = self.num_inputs * new_shape
        else:
            new_size = self.num_inputs * reduce(operator.mul, new_shape)
        if new_size != self.inputs.size:
            raise Exception("shape %s is incompatible with inputs" % (new_shape,))
        if isinstance(new_shape, numbers.Integral):
            new_shape = (new_shape,)
        self.inputs = self.inputs.reshape((self.num_inputs,) + new_shape)
        self.split(self._split, verbose=False)
        if verbose:
            self.summary()

    def set_targets_to_categories(self, num_classes):
        """
        Given net.labels are integers, set the net.targets to one_hot() categories.
        """
        ## FIXME: allow working on multi-targets
        if self.num_target_banks > 1:
            raise Exception("set_targets_to_categories does not yet work on multi-target patterns")
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if not isinstance(num_classes, numbers.Integral) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self.targets = keras.utils.to_categorical(self.labels, num_classes).astype("uint8")
        self.train_targets = self.targets[:self._split]
        self.test_targets = self.targets[self._split:]
        print('Generated %d target vectors from labels' % self.num_inputs)

    def summary(self):
        """
        Print out a summary of the dataset.
        """
        print('Input Summary:')
        print('   count  : %d' % (self.get_inputs_length(),))
        if self.get_inputs_length() != 0:
            if self.num_target_banks > 1:
                print('   shape  : %s' % ([x[0].shape for x in self.inputs],))
            else:
                print('   shape  : %s' % (self.inputs[0].shape,))
            print('   range  : %s' % (self.inputs_range,))
        print('Target Summary:')
        print('   count  : %d' % (self.get_targets_length(),))
        if self.get_targets_length() != 0:
            if self.num_target_banks > 1:
                print('   shape  : %s' % ([x[0].shape for x in self.targets],))
            else:
                print('   shape  : %s' % (self.targets[0].shape,))
            print('   range  : %s' % (self.targets_range,))

    def rescale_inputs(self, old_range, new_range, new_dtype):
        """
        Rescale the inputs. WIP.
        """
        ## FIXME: allow working on multi-inputs
        if self.num_input_banks > 1:
            raise Exception("rescale_inputs does not yet work on multi-input patterns")
        old_min, old_max = old_range
        new_min, new_max = new_range
        if self.inputs.min() < old_min or self.inputs.max() > old_max:
            raise Exception('range %s is incompatible with inputs' % (old_range,))
        if old_min > old_max:
            raise Exception('range %s is out of order' % (old_range,))
        if new_min > new_max:
            raise Exception('range %s is out of order' % (new_range,))
        self.inputs = rescale_numpy_array(self.inputs, old_range, new_range, new_dtype)
        self.inputs_range = (self.inputs.min(), self.inputs.max())
        print('Inputs rescaled to %s values in the range %s - %s' %
              (self.inputs.dtype, new_min, new_max))

    def shuffle(self, verbose=True):
        """
        Shuffle the inputs/targets. WIP.
        """
        ## FIXME: allow working on multi-inputs/-targets
        if self.num_target_banks > 1:
            raise Exception("shuffle does not yet work on multi-input/-target patterns")
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        indices = np.random.permutation(self.num_inputs)
        self.inputs = self.inputs[indices]
        if len(self.labels) != 0:
            self.labels = self.labels[indices]
        if self.get_targets_length() != 0:
            self.targets = self.targets[indices]
        self.split(self._split, verbose=False)
        if verbose:
            print('Shuffled all %d inputs' % self.num_inputs)

    def split(self, split=0.50, verbose=True):
        """
        Split the inputs/targets into training/test datasets.
        """
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if isinstance(split, numbers.Integral):
            if not 0 <= split <= self.num_inputs:
                raise Exception("split out of range: %d" % split)
            self._split = split
        elif isinstance(split, numbers.Real):
            if not 0 <= split <= 1:
                raise Exception("split is not in the range 0-1: %s" % split)
            self._split = int(self.num_inputs * split)
        else:
            raise Exception("invalid split: %s" % split)
        if self.num_input_banks > 1:
            self.train_inputs = [col[:self._split] for col in self.inputs]
            self.test_inputs = [col[self._split:] for col in self.inputs]
        else:
            self.train_inputs = self.inputs[:self._split]
            self.test_inputs = self.inputs[self._split:]
        if len(self.labels) != 0:
            self.train_labels = self.labels[:self._split]
            self.test_labels = self.labels[self._split:]
        if self.get_targets_length() != 0:
            if self.num_target_banks > 1:
                self.train_targets = [col[:self._split] for col in self.targets]
                self.test_targets = [col[self._split:] for col in self.targets]
            else:
                self.train_targets = self.targets[:self._split]
                self.test_targets = self.targets[self._split:]
        if verbose:
            print('Split dataset into:')
            print('   train set count: %d' % self.get_train_inputs_length())
            print('   test set count : %d' % self.get_test_inputs_length())

    def get_input(self, i):
        """
        Get an input from the internal dataset and
        format it in the human API.
        """
        if self.num_input_banks == 1:
            return list(self.inputs[i])
        else:
            inputs = []
            for c in range(self.num_input_banks):
                inputs.append(list(self.inputs[c][i]))
            return inputs

    def get_target(self, i):
        """
        Get a target from the internal dataset and
        format it in the human API.
        """
        if self.num_target_banks == 1:
            return list(self.targets[i])
        else:
            targets = []
            for c in range(self.num_target_banks):
                targets.append(list(self.targets[c][i]))
            return targets

    def get_train_input(self, i):
        """
        Get a training input from the internal dataset and
        format it in the human API.
        """
        if self.num_input_banks == 1:
            return list(self.train_inputs[i])
        else:
            inputs = []
            for c in range(self.num_input_banks):
                inputs.append(list(self.train_inputs[c][i]))
            return inputs

    def get_train_target(self, i):
        """
        Get a training target from the internal dataset and
        format it in the human API.
        """
        if self.num_target_banks == 1:
            return list(self.train_targets[i])
        else:
            targets = []
            for c in range(self.num_target_banks):
                targets.append(list(self.train_targets[c][i]))
            return targets

    def get_test_input(self, i):
        """
        Get a test input from the internal dataset and
        format it in the human API.
        """
        if self.num_input_banks == 1:
            return list(self.test_inputs[i])
        else:
            inputs = []
            for c in range(self.num_input_banks):
                inputs.append(list(self.test_inputs[c][i]))
            return inputs

    def get_test_target(self, i):
        """
        Get a test target from the internal dataset and
        format it in the human API.
        """
        if self.num_target_banks == 1:
            return list(self.test_targets[i])
        else:
            targets = []
            for c in range(self.num_target_banks):
                targets.append(list(self.test_targets[c][i]))
            return targets

    def get_inputs_length(self):
        """
        Get the number of input patterns.
        """
        if len(self.inputs) == 0:
            return 0
        if self.num_input_banks > 1:
            return self.inputs[0].shape[0]
        else:
            return self.inputs.shape[0]

    def get_targets_length(self):
        """
        Get the number of target patterns.
        """
        if len(self.targets) == 0:
            return 0
        if self.num_target_banks > 1:
            return self.targets[0].shape[0]
        else:
            return self.targets.shape[0]

    def get_test_inputs_length(self):
        """
        Get the number of test input patterns.
        """
        if len(self.test_inputs) == 0:
            return 0
        if self.num_input_banks > 1:
            return self.test_inputs[0].shape[0]
        else:
            return self.test_inputs.shape[0]

    def get_test_targets_length(self):
        """
        Get the number of test target patterns.
        """
        if len(self.test_targets) == 0:
            return 0
        if self.num_target_banks > 1:
            return self.test_targets[0].shape[0]
        else:
            return self.test_targets.shape[0]

    def get_train_inputs_length(self):
        """
        Get the number of training input patterns.
        """
        if len(self.train_inputs) == 0:
            return 0
        if self.num_input_banks > 1:
            return self.train_inputs[0].shape[0]
        else:
            return self.train_inputs.shape[0]

    def get_train_targets_length(self):
        """
        Get the number of training target patterns.
        """
        if len(self.train_targets) == 0:
            return 0
        if self.num_target_banks > 1:
            return self.train_targets[0].shape[0]
        else:
            return self.train_targets.shape[0]
