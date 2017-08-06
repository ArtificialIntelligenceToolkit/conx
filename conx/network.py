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

"""
The network module contains the code for the Network class.
"""

import operator
import importlib
from functools import reduce
import signal
import numbers
import base64
import copy
import io

import numpy as np
import keras

from .utils import *
from .layers import Layer

try:
    from IPython import get_ipython
except:
    get_ipython = lambda: None

#------------------------------------------------------------------------

class Network():
    """
    The main class for the conx neural network package.
    """
    OPTIMIZERS = ("sgd", "rmsprop", "adagrad", "adadelta", "adam",
                  "adamax", "nadam", "tfoptimizer")
    def __init__(self, name, *sizes, **config):
        """
        Create a neural network.
        if sizes is given, create a full network.
        Optional keywork: activation
        """
        if not isinstance(name, str):
            raise Exception("first argument should be a name for the network")
        self.config = {
            "font_size": 12, # for svg
            "font_family": "monospace", # for svg
            "border_top": 25, # for svg
            "border_bottom": 25, # for svg
            "hspace": 150, # for svg
            "vspace": 30, # for svg, arrows
            "image_maxdim": 200, # for svg
            "activation": "linear", # Dense default, if none specified
            "arrow_color": "blue",
            "arrow_width": "2",
            "border_width": "2",
            "border_color": "blue",
            "show_targets": True,
            "minmax": None,
            "colormap": None,
            "show_errors": True,
            "pixels_per_unit": 1,
            "pp_max_length": 20,
            "pp_precision": 1,
        }
        if not isinstance(name, str):
            raise Exception("conx layers need a name as a first parameter")
        self.config.update(config)
        self.compile_options = {}
        self.train_options = {}
        self.name = name
        self.layers = []
        self.layer_dict = {}
        self.inputs = []
        self.train_inputs = []
        self.train_targets = []
        self.test_inputs = []
        self.test_targets = []
        self.labels = []
        self.targets = []
        # If simple feed-forward network:
        for i in range(len(sizes)):
            if i > 0:
                self.add(Layer(autoname(i, len(sizes)), shape=sizes[i],
                               activation=self.config["activation"]))
            else:
                self.add(Layer(autoname(i, len(sizes)), shape=sizes[i]))
        self.num_input_layers = 0
        self.num_target_layers = 0
        # Connect them together:
        for i in range(len(sizes) - 1):
            self.connect(autoname(i, len(sizes)), autoname(i+1, len(sizes)))
        self.epoch_count = 0
        self.acc_history = []
        self.loss_history = []
        self.val_percent_history = []
        self.input_layer_order = []
        self.output_layer_order = []
        self.num_inputs = 0
        self.num_targets = 0
        self.multi_inputs = False
        self.multi_targets = False
        self.visualize = False
        self._comm = None
        self.inputs_range = (0,0)
        self.targets_range = (0,0)
        self.test_labels = []
        self.train_labels = []
        self.model = None
        self.split = 0
        self.prop_from_dict = {}
        self._svg_counter = 1

    def __getitem__(self, layer_name):
        if layer_name not in self.layer_dict:
            return None
        else:
            return self.layer_dict[layer_name]

    def _repr_svg_(self):
        if all([layer.model for layer in self.layers]):
            return self.build_svg()
        else:
            return None

    def __repr__(self):
        return "<Network name='%s'>" % self.name

    def add(self, layer):
        """
        Add a layer to the network layer connections. Order is not
        important, unless using the default net.connect() form.
        """
        if layer.name in self.layer_dict:
            raise Exception("duplicate layer name '%s'" % layer.name)
        self.layers.append(layer)
        self.layer_dict[layer.name] = layer

    def connect(self, from_layer_name=None, to_layer_name=None):
        """
        Connect two layers together if called with arguments. If
        called with no arguments, then it will make a sequential
        run through the layers in order added.
        """
        if from_layer_name is None and to_layer_name is None:
            for i in range(len(self.layers) - 1):
                self.connect(self.layers[i].name, self.layers[i+1].name)
        else:
            if from_layer_name not in self.layer_dict:
                raise Exception('unknown layer: %s' % from_layer_name)
            if to_layer_name not in self.layer_dict:
                raise Exception('unknown layer: %s' % to_layer_name)
            from_layer = self.layer_dict[from_layer_name]
            to_layer = self.layer_dict[to_layer_name]
            from_layer.outgoing_connections.append(to_layer)
            to_layer.incoming_connections.append(from_layer)
            input_layers = [layer for layer in self.layers if layer.kind() == "input"]
            self.num_input_layers = len(input_layers)
            target_layers = [layer for layer in self.layers if layer.kind() == "output"]
            self.num_target_layers = len(target_layers)

    def summary(self):
        """
        Print out a summary of the network.
        """
        for layer in self.layers:
            layer.summary()

    def set_dataset_direct(self, inputs, targets, verbose=True):
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
        self.multi_inputs = isinstance(self.inputs, (list, tuple))
        self.multi_targets = isinstance(self.targets, (list, tuple))
        self._cache_dataset_values()
        self.split_dataset(self.num_inputs, verbose=False)
        if verbose:
            self.summary_dataset()

    def slice_dataset(self, start=None, stop=None, verbose=True):
        """
        Cut out some input/targets.
            
        net.slice_dataset(100) - reduce to first 100 inputs/targets
        net.slice_dataset(100, 200) - reduce to second 100 inputs/targets
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
        if self.multi_inputs:
            self.inputs = [np.array([vector for vector in row[start:stop]]) for row in self.inputs]
        else:
            self.inputs = self.inputs[start:stop] # ok
        if self.multi_targets:
            self.targets = [np.array([vector for vector in row[start:stop]]) for row in self.targets]
        else:
            self.targets = self.targets[start:stop]
        if len(self.labels) > 0:
            self.labels = self.labels[start:stop]
        self._cache_dataset_values()
        self.split_dataset(self.num_inputs, verbose=False)
        if verbose:
            self.summary_dataset()

    def _cache_dataset_values(self):
        self.num_inputs = self.get_inputs_length()
        if self.num_inputs > 0:
            if self.multi_inputs:
                self.inputs_range = (min([x.min() for x in self.inputs]),
                                     max([x.max() for x in self.inputs]))
            else:
                self.inputs_range = (self.inputs.min(), self.inputs.max())
        else:
            self.inputs_range = (0,0)
        self.num_targets = self.get_targets_length()
        if self.num_inputs > 0:
            if self.multi_targets:
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

    def set_dataset(self, pairs, verbose=True):
        """
        Set the human-specified dataset to a proper keras dataset.

        Multi-inputs or multi-outputs must be: [vector, vector, ...] for each layer input/target pairing.
        """
        ## Either the inputs/targets are a list of a list -> np.array(...) (np.array() of vectors)
        ## or are a list of list of list -> [np.array(), np.array()]  (list of np.array cols of vectors)
        self.multi_inputs = len(np.array(pairs[0][0]).shape) > 1
        self.multi_targets = len(np.array(pairs[0][1]).shape) > 1
        if self.multi_inputs:
            self.inputs = []
            for i in range(len(pairs[0][0])):
                self.inputs.append(np.array([x[i] for (x,y) in pairs], "float32"))
        else:
            self.inputs = np.array([x for (x, y) in pairs], "float32")
        if self.multi_targets:
            self.targets = []
            for i in range(len(pairs[0][1])):
                self.targets.append(np.array([y[i] for (x,y) in pairs], "float32"))
        else:
            self.targets = np.array([y for (x, y) in pairs], "float32")
        self.labels = []
        self._cache_dataset_values()
        self.split_dataset(self.num_inputs, verbose=verbose)
        if verbose:
            self.summary_dataset()

    def load_mnist_dataset(self, verbose=True):
        """
        Load the Keras MNIST dataset and format it as images.
        """
        from keras.datasets import mnist
        from keras.utils import to_categorical
        import keras.backend as K
        # input image dimensions
        img_rows, img_cols = 28, 28
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
        self.inputs = np.concatenate((x_train,x_test))
        self.labels = np.concatenate((y_train,y_test))
        self.targets = to_categorical(self.labels)
        self.multi_targets = False
        self.multi_inputs = False
        self._cache_dataset_values()
        self.split_dataset(self.num_inputs, verbose=False)
        if verbose:
            self.summary_dataset()

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
        if self.multi_inputs:
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
        self.split_dataset(self.split, verbose=False)
        if verbose:
            self.summary_dataset()

    def set_input_layer_order(self, *layer_names):
        """
        When multiple input banks, you must set this.
        """
        if len(layer_names) == 1:
            raise Exception("set_input_layer_order cannot be a single layer")
        self.input_layer_order = []
        for layer_name in layer_names:
            if layer_name not in self.input_layer_order:
                self.input_layer_order.append(layer_name)
            else:
                raise Exception("duplicate name in set_input_layer_order: '%s'" % layer_name)

    def set_output_layer_order(self, *layer_names):
        """
        When multiple output banks, you must set this.
        """
        if len(layer_names) == 1:
            raise Exception("set_output_layer_order cannot be a single layer")
        self.output_layer_order = []
        for layer_name in layer_names:
            if layer_name not in self.output_layer_order:
                self.output_layer_order.append(layer_name)
            else:
                raise Exception("duplicate name in set_output_layer_order: '%s'" % layer_name)

    def set_targets_to_categories(self, num_classes):
        """
        Given net.labels are integers, set the net.targets to one_hot() categories.
        """
        ## FIXME: allow working on multi-targets
        if self.multi_targets:
            raise Exception("set_targets_to_categories does not yet work on multi-target patterns")
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if not isinstance(num_classes, numbers.Integral) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self.targets = keras.utils.to_categorical(self.labels, num_classes).astype("uint8")
        self.train_targets = self.targets[:self.split]
        self.test_targets = self.targets[self.split:]
        print('Generated %d target vectors from labels' % self.num_inputs)

    def summary_dataset(self):
        """
        Print out a summary of the dataset.
        """
        print('Input Summary:')
        print('   length  : %d' % (self.get_inputs_length(),))
        print('   training: %d' % (self.get_train_inputs_length(),))
        print('   testing : %d' % (self.get_test_inputs_length(),))
        if self.get_inputs_length() != 0:
            if self.multi_targets:
                print('   shape  : %s' % ([x[0].shape for x in self.inputs],))
            else:
                print('   shape  : %s' % (self.inputs[0].shape,))
            print('   range  : %s' % (self.inputs_range,))
        print('Target Summary:')
        print('   length  : %d' % (self.get_targets_length(),))
        print('   training: %d' % (self.get_train_targets_length(),))
        print('   testing : %d' % (self.get_test_targets_length(),))
        if self.get_targets_length() != 0:
            if self.multi_targets:
                print('   shape  : %s' % ([x[0].shape for x in self.targets],))
            else:
                print('   shape  : %s' % (self.targets[0].shape,))
            print('   range  : %s' % (self.targets_range,))

    def rescale_inputs(self, old_range, new_range, new_dtype):
        """
        Rescale the inputs. WIP.
        """
        ## FIXME: allow working on multi-inputs
        if self.multi_inputs:
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

    def _make_weights(self, shape):
        """
        Makes a vector/matrix of random weights centered around 0.0.
        """
        size = reduce(operator.mul, shape) # (in, out)
        magnitude = max(min(1/shape[0] * 50, 1.16), 0.06)
        rmin, rmax = -magnitude, magnitude
        span = (rmax - rmin)
        return np.array(span * np.random.rand(size) - span/2.0,
                        dtype='float32').reshape(shape)

    def reset(self):
        """
        Reset all of the weights/biases in a network.
        The magnitude is based on the size of the network.
        """
        self.epoch_count = 0
        self.acc_history = []
        self.loss_history = []
        self.val_percent_history = []
        if self.model:
            for layer in self.model.layers:
                weights = layer.get_weights()
                new_weights = []
                for weight in weights:
                    new_weights.append(self._make_weights(weight.shape))
                layer.set_weights(new_weights)

    def shuffle_dataset(self, verbose=True):
        """
        Shuffle the inputs/targets. WIP.
        """
        ## FIXME: allow working on multi-inputs/-targets
        if self.multi_inputs or self.multi_targets:
            raise Exception("shuffle_dataset does not yet work on multi-input/-target patterns")
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        indices = np.random.permutation(self.num_inputs)
        self.inputs = self.inputs[indices]
        if len(self.labels) != 0:
            self.labels = self.labels[indices]
        if self.get_targets_length() != 0:
            self.targets = self.targets[indices]
        self.split_dataset(self.split, verbose=False)
        if verbose:
            print('Shuffled all %d inputs' % self.num_inputs)

    def split_dataset(self, split=0.50, verbose=True):
        """
        Split the inputs/targets into training/test datasets.
        """
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if isinstance(split, numbers.Integral):
            if not 0 <= split <= self.num_inputs:
                raise Exception("split out of range: %d" % split)
            self.split = split
        elif isinstance(split, numbers.Real):
            if not 0 <= split <= 1:
                raise Exception("split is not in the range 0-1: %s" % split)
            self.split = int(self.num_inputs * split)
        else:
            raise Exception("invalid split: %s" % split)
        if self.multi_inputs:
            self.train_inputs = [col[:self.split] for col in self.inputs]
            self.test_inputs = [col[self.split:] for col in self.inputs]
        else:
            self.train_inputs = self.inputs[:self.split]
            self.test_inputs = self.inputs[self.split:]
        if len(self.labels) != 0:
            self.train_labels = self.labels[:self.split]
            self.test_labels = self.labels[self.split:]
        if self.get_targets_length() != 0:
            if self.multi_targets:
                self.train_targets = [col[:self.split] for col in self.targets]
                self.test_targets = [col[self.split:] for col in self.targets]
            else:
                self.train_targets = self.targets[:self.split]
                self.test_targets = self.targets[self.split:]
        if verbose:
            print('Split dataset into:')
            print('   %d train inputs' % self.get_train_inputs_length())
            print('   %d test inputs' % self.get_test_inputs_length())

    def test(self, inputs=None, targets=None, batch_size=32, tolerance=0.1):
        """
        Requires items in proper internal format, if given (for now).
        """
        ## FIXME: allow human format of inputs, if given
        dataset_name = "provided"
        if inputs is None:
            if self.split == self.num_inputs:

                inputs = self.train_inputs
                dataset_name = "training"
            else:
                inputs = self.test_inputs
                dataset_name = "testing"
        if targets is None:
            if self.split == self.num_targets:
                targets = self.train_targets
            else:
                targets = self.test_targets
        print("Testing on %s dataset..." % dataset_name)
        outputs = self.model.predict(inputs, batch_size=batch_size)
        if self.num_input_layers == 1:
            ins = [self.ppf(x) for x in inputs.tolist()]
        else:
            ins = [("[" + ", ".join([self.ppf(vector) for vector in row]) + "]") for row in np.array(list(zip(*inputs))).tolist()]
        ## targets:
        if self.num_target_layers == 1:
            targs = [self.ppf(x) for x in targets.tolist()]
        else:
            targs = [("[" + ", ".join([self.ppf(vector) for vector in row]) + "]") for row in np.array(list(zip(*targets))).tolist()]
        ## outputs:
        if self.num_target_layers == 1:
            outs = [self.ppf(x) for x in outputs.tolist()]
        else:
            outs = [("[" + ", ".join([self.ppf(vector) for vector in row]) + "]") for row in np.array(list(zip(*outputs))).tolist()]
        ## correct?
        if self.num_target_layers == 1:
            correct = [all(x) for x in map(lambda v: v <= tolerance,
                                           np.abs(outputs - targets))]
        else:
            outs = np.array(list(zip(*[out.flatten().tolist() for out in outputs])))
            targs = np.array(list(zip(*[out.flatten().tolist() for out in targets])))
            correct = [all(row) for row in (np.abs(outs - targs) < tolerance)]
        print("# | inputs | targets | outputs | result")
        for i in range(len(outs)):
            print(i, "|", ins[i], "|", targs[i], "|", outs[i], "|", "correct" if correct[i] else "X")
        print("Total count:", len(correct))
        print("Total percentage correct:", list(correct).count(True)/len(correct))

    def train_one(self, inputs, targets, batch_size=32):
        """
        Train on one input/target pair. Requires internal format.
        """
        pairs = [(inputs, targets)]
        if self.num_input_layers == 1:
            ins = np.array([x for (x, y) in pairs], "float32")
        else:
            ins = []
            for i in range(len(pairs[0][0])):
                ins.append(np.array([x[i] for (x,y) in pairs], "float32"))
        if self.num_target_layers == 1:
            targs = np.array([y for (x, y) in pairs], "float32")
        else:
            targs = []
            for i in range(len(pairs[0][1])):
                targs.append(np.array([y[i] for (x,y) in pairs], "float32"))
        history = self.model.fit(ins, targs, epochs=1, verbose=0, batch_size=batch_size)
        ## may need to update history?
        outputs = self.propagate(inputs, batch_size=batch_size)
        return outputs

    def retrain(self, **overrides):
        """
        Call network.train() again with same options as last call, unless overrides.
        """
        self.train_options.update(overrides)
        self.train(**self.train_options)

    def train(self, epochs=1, accuracy=None, batch_size=None,
              report_rate=1, tolerance=0.1, verbose=1, shuffle=True,
              class_weight=None, sample_weight=None):
        """
        Train the network.
        """
        ## IDEA: train_options could be a history of dicts
        ## to keep track of a schedule of learning over time
        self.train_options = {
            "epochs": epochs,
            "accuracy": accuracy,
            "batch_size": batch_size,
            "report_rate": report_rate,
            "tolerance": tolerance,
            "verbose": verbose,
            "shuffle": shuffle,
            "class_weight": class_weight,
            "sample_weight": sample_weight,
            }
        if batch_size is None:
            if self.num_input_layers == 1:
                batch_size = self.train_inputs.shape[0]
            else:
                batch_size = self.train_inputs[0].shape[0]
        if not (isinstance(batch_size, numbers.Integral) or batch_size is None):
            raise Exception("bad batch size: %s" % (batch_size,))
        if accuracy is None and epochs > 1 and report_rate > 1:
            print("Warning: report_rate is ignored when in epoch mode")
        if self.split == self.num_inputs:
            validation_inputs = self.train_inputs
            validation_targets = self.train_targets
        else:
            validation_inputs = self.test_inputs
            validation_targets = self.test_targets
        if verbose: print("Training...")
        with InterruptHandler() as handler:
            if accuracy is None: # train them all using fit
                result = self.model.fit(self.train_inputs, self.train_targets,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=verbose,
                                        shuffle=shuffle,
                                        class_weight=class_weight,
                                        sample_weight=sample_weight)
                outputs = self.model.predict(validation_inputs, batch_size=batch_size)
                if self.num_target_layers == 1:
                    correct = [all(x) for x in map(lambda v: v <= tolerance,
                                                   np.abs(outputs - validation_targets))].count(True)
                else:
                    correct = [all(x) for x in map(lambda v: v <= tolerance,
                                                   np.abs(np.array(outputs) - np.array(validation_targets)))].count(True)
                self.epoch_count += epochs
                acc = 0
                # In multi-outputs, acc is given by output layer name + "_acc"
                for key in result.history:
                    if key.endswith("acc"):
                        acc += result.history[key][0]
                #acc = result.history['acc'][0]
                self.acc_history.append(acc)
                loss = result.history['loss'][0]
                self.loss_history.append(loss)
                val_percent = correct/len(validation_targets)
                self.val_percent_history.append(val_percent)
            else:
                for e in range(1, epochs+1):
                    result = self.model.fit(self.train_inputs, self.train_targets,
                                            batch_size=batch_size,
                                            epochs=1,
                                            verbose=0,
                                            shuffle=shuffle,
                                            class_weight=class_weight,
                                            sample_weight=sample_weight)
                    outputs = self.model.predict(validation_inputs, batch_size=batch_size)
                    if self.num_target_layers == 1:
                        correct = [all(x) for x in map(lambda v: v <= tolerance,
                                                       np.abs(outputs - validation_targets))].count(True)
                    else:
                        correct = [all(x) for x in map(lambda v: v <= tolerance,
                                                       np.abs(np.array(outputs) - np.array(validation_targets)))].count(True)
                    self.epoch_count += 1
                    acc = 0
                    # In multi-outputs, acc is given by output layer name + "_acc"
                    for key in result.history:
                        if key.endswith("acc"):
                            acc += result.history[key][0]
                    #acc = result.history['acc'][0]
                    self.acc_history.append(acc)
                    loss = result.history['loss'][0]
                    self.loss_history.append(loss)
                    val_percent = correct/len(validation_targets)
                    self.val_percent_history.append(val_percent)
                    if self.epoch_count % report_rate == 0:
                        if verbose: print("Epoch #%5d | train error %7.5f | train accuracy %7.5f | validate%% %7.5f" %
                                          (self.epoch_count, loss, acc, val_percent))
                    if val_percent >= accuracy or handler.interrupted:
                        break
            if handler.interrupted:
                print("=" * 72)
                print("Epoch #%5d | train error %7.5f | train accuracy %7.5f | validate%% %7.5f" %
                      (self.epoch_count, loss, acc, val_percent))
                raise KeyboardInterrupt
        if verbose:
            print("=" * 72)
            print("Epoch #%5d | train error %7.5f | train accuracy %7.5f | validate%% %7.5f" %
                  (self.epoch_count, loss, acc, val_percent))
        else:
            return (self.epoch_count, loss, acc, val_percent)

        # # evaluate the model
        # print('Evaluating performance...')
        # loss, accuracy = self.model.evaluate(self.test_inputs, self.test_targets, verbose=0)
        # print('Test loss:', loss)
        # print('Test accuracy:', accuracy)
        # #print('Most recent weights saved in model.weights')
        # #self.model.save_weights('model.weights')

    def get_input(self, i):
        """
        Get an input from the internal dataset and
        format it in the human API.
        """
        if self.num_input_layers == 1:
            return list(self.inputs[i])
        else:
            inputs = []
            for c in range(self.num_input_layers):
                inputs.append(list(self.inputs[c][i]))
            return inputs

    def get_target(self, i):
        """
        Get a target from the internal dataset and
        format it in the human API.
        """
        if self.num_target_layers == 1:
            return list(self.targets[i])
        else:
            targets = []
            for c in range(self.num_target_layers):
                targets.append(list(self.targets[c][i]))
            return targets

    def get_train_input(self, i):
        """
        Get a training input from the internal dataset and
        format it in the human API.
        """
        if self.num_input_layers == 1:
            return list(self.train_inputs[i])
        else:
            inputs = []
            for c in range(self.num_input_layers):
                inputs.append(list(self.train_inputs[c][i]))
            return inputs

    def get_train_target(self, i):
        """
        Get a training target from the internal dataset and
        format it in the human API.
        """
        if self.num_target_layers == 1:
            return list(self.train_targets[i])
        else:
            targets = []
            for c in range(self.num_target_layers):
                targets.append(list(self.train_targets[c][i]))
            return targets

    def get_test_input(self, i):
        """
        Get a test input from the internal dataset and
        format it in the human API.
        """
        if self.num_input_layers == 1:
            return list(self.test_inputs[i])
        else:
            inputs = []
            for c in range(self.num_input_layers):
                inputs.append(list(self.test_inputs[c][i]))
            return inputs

    def get_test_target(self, i):
        """
        Get a test target from the internal dataset and
        format it in the human API.
        """
        if self.num_target_layers == 1:
            return list(self.test_targets[i])
        else:
            targets = []
            for c in range(self.num_target_layers):
                targets.append(list(self.test_targets[c][i]))
            return targets

    def propagate(self, input, batch_size=32):
        """
        Propagate an input (in human API) through the network.
        If visualizing, the network image will be updated.
        """
        if self.num_input_layers == 1:
            outputs = list(self.model.predict(np.array([input]), batch_size=batch_size)[0])
        else:
            inputs = [np.array(x, "float32") for x in input]
            outputs = [[list(y) for y in x][0] for x in self.model.predict(inputs, batch_size=batch_size)]
        if self.visualize and get_ipython():
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            for layer in self.layers:
                image = self.propagate_to_image(layer.name, input, batch_size)
                data_uri = self._image_to_uri(image)
                self._comm.send({'class': "%s_%s" % (self.name, layer.name), "href": data_uri})
        return outputs

    def propagate_from(self, layer_name, input, output_layer_names=None, batch_size=32):
        """
        Propagate activations from the given layer name to the output layers.
        """
        if layer_name not in self.layer_dict:
            raise Exception("No such layer '%s'" % layer_name)
        if output_layer_names is None:
            if self.num_target_layers == 1:
                output_layer_names = [layer.name for layer in self.layers if layer.kind() == "output"]
            else:
                output_layer_names = self.output_layer_order
        else:
            if isinstance(output_layer_names, str):
                output_layer_names = [output_layer_names]
        outputs = []
        for output_layer_name in output_layer_names:
            prop_model = self.prop_from_dict.get((layer_name, output_layer_name), None)
            if prop_model is None:
                path = topological_sort(self, self[layer_name].outgoing_connections)
                # Make a new Input to start here:
                k = input_k = keras.layers.Input(self[layer_name].shape, name=self[layer_name].name)
                # So that we can display activations here:
                self.prop_from_dict[(layer_name, layer_name)] = keras.models.Model(inputs=input_k,
                                                                                   outputs=input_k)
                for layer in path:
                    k = self.prop_from_dict.get((layer_name, layer.name), None)
                    if k is None:
                        k = input_k
                        fs = layer.make_keras_functions()
                        for f in fs:
                            k = f(k)
                    self.prop_from_dict[(layer_name, layer.name)] = keras.models.Model(inputs=input_k,
                                                                                       outputs=k)
                # Now we should be able to get the prop_from model:
                prop_model = self.prop_from_dict.get((layer_name, output_layer_name), None)
            inputs = np.array([input])
            outputs.append([list(x) for x in prop_model.predict(inputs)][0])
        if self.visualize and get_ipython():
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            ## Update from start to rest of graph
            for layer in topological_sort(self, [self[layer_name]]):
                model = self.prop_from_dict[(layer_name, layer.name)]
                vector = model.predict(inputs)[0]
                image = layer.make_image(vector, self.config)
                data_uri = self._image_to_uri(image)
                self._comm.send({'class': "%s_%s" % (self.name, layer.name), "href": data_uri})
        if len(output_layer_names) == 1:
            return outputs[0]
        else:
            return outputs

    def display_component(self, vector, component, **opts): #minmax=None, colormap=None):
        """
        vector is a list, one each per output layer. component is "errors" or "targets"
        """
        config = copy.copy(self.config)
        config.update(opts)
        ## FIXME: this doesn't work on multi-targets/outputs
        if self.output_layer_order:
            output_names = self.output_layer_order
        else:
            output_names = [layer.name for layer in self.layers if layer.kind() == "output"]
        for (target, layer_name) in zip(vector, output_names):
            array = np.array(target)
            image = self[layer_name].make_image(array, config) # minmax=minmax, colormap=colormap)
            data_uri = self._image_to_uri(image)
            self._comm.send({'class': "%s_%s_%s" % (self.name, layer_name, component), "href": data_uri})

    def propagate_to(self, layer_name, inputs, batch_size=32, visualize=True):
        """
        Computes activation at a layer. Side-effect: updates visualized SVG.
        """
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        if self.num_input_layers == 1:
            outputs = self[layer_name].model.predict(np.array([inputs]), batch_size=batch_size)
        else:
            # get just inputs for this layer, in order:
            vector = [np.array(inputs[self.input_layer_order.index(name)]) for name in self[layer_name].input_names]
            outputs = self[layer_name].model.predict(vector, batch_size=batch_size)
        if self.visualize and visualize and get_ipython():
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            # Update path from input to output
            for layer in self.layers: # FIXME??: update all layers for now
                out = self.propagate_to(layer.name, inputs, visualize=False)
                image = self[layer.name].make_image(np.array(out), self.config) # single vector, as an np.array
                data_uri = self._image_to_uri(image)
                self._comm.send({'class': "%s_%s" % (self.name, layer.name), "href": data_uri})
        outputs = outputs[0].tolist()
        return outputs

    def propagate_to_image(self, layer_name, input, batch_size=32):
        """
        Gets an image of activations at a layer.
        """
        outputs = self.propagate_to(layer_name, input, batch_size)
        array = np.array(outputs)
        image = self[layer_name].make_image(array, self.config)
        return image

    def compile(self, **kwargs):
        """
        Check and compile the network.
        """
        ## Error checking:
        if len(self.layers) == 0:
            raise Exception("network has no layers")
        for layer in self.layers:
            if layer.kind() == 'unconnected':
                raise Exception("'%s' layer is unconnected" % layer.name)
        if "error" in kwargs: # synonym
            kwargs["loss"] = kwargs["error"]
            del kwargs["error"]
        if "optimizer" in kwargs:
            optimizer = kwargs["optimizer"]
            if (not ((isinstance(optimizer, str) and optimizer in self.OPTIMIZERS) or
                     (isinstance(optimizer, object) and issubclass(optimizer.__class__, keras.optimizers.Optimizer)))):
                raise Exception("invalid optimizer '%s'; use valid function or one of %s" %
                                (optimizer, Network.OPTIMIZERS,))
        input_layers = [layer for layer in self.layers if layer.kind() == "input"]
        if len(input_layers) == 1 and len(self.input_layer_order) == 0:
            pass # ok!
        elif len(input_layers) == len(self.input_layer_order):
            # check to make names all match
            for layer in input_layers:
                if layer.name not in self.input_layer_order:
                    raise Exception("layer '%s' is not listed in set_input_layer_order()" % layer.name)
        else:
            raise Exception("improper set_input_layer_order() names")
        output_layers = [layer for layer in self.layers if layer.kind() == "output"]
        if len(output_layers) == 1 and len(self.output_layer_order) == 0:
            pass # ok!
        elif len(output_layers) == len(self.output_layer_order):
            # check to make names all match
            for layer in output_layers:
                if layer.name not in self.output_layer_order:
                    raise Exception("layer '%s' is not listed in set_output_layer_order()" % layer.name)
        else:
            raise Exception("improper set_output_layer_order() names")
        sequence = topological_sort(self, self.layers)
        for layer in sequence:
            if layer.kind() == 'input':
                layer.k = layer.make_input_layer_k()
                layer.input_names = [layer.name]
                layer.model = keras.models.Model(inputs=layer.k, outputs=layer.k) # identity
            else:
                if len(layer.incoming_connections) == 0:
                    raise Exception("non-input layer '%s' with no incoming connections" % layer.name)
                kfuncs = layer.make_keras_functions()
                if len(layer.incoming_connections) == 1:
                    k = layer.incoming_connections[0].k
                    layer.input_names = layer.incoming_connections[0].input_names
                else: # multiple inputs, need to merge
                    k = keras.layers.Concatenate()([incoming.k for incoming in layer.incoming_connections])
                    # flatten:
                    layer.input_names = [item for sublist in
                                         [incoming.input_names for incoming in layer.incoming_connections]
                                         for item in sublist]
                for f in kfuncs:
                    k = f(k)
                layer.k = k
                ## get the inputs to this branch, in order:
                input_ks = self._get_input_ks_in_order(layer.input_names)
                layer.model = keras.models.Model(inputs=input_ks, outputs=layer.k)
        output_k_layers = self._get_ordered_output_layers()
        input_k_layers = self._get_ordered_input_layers()
        self.model = keras.models.Model(inputs=input_k_layers, outputs=output_k_layers)
        kwargs['metrics'] = ['accuracy']
        self.compile_options = kwargs
        self.model.compile(**kwargs)

    def _get_input_ks_in_order(self, layer_names):
        """
        Get the Keras function for each of a set of layer names.
        """
        if self.input_layer_order:
            result = []
            for name in self.input_layer_order:
                if name in layer_names:
                    result.append(self[name].k)
            return result
        else:
            # the one input name:
            return [[layer for layer in self.layers if layer.kind() == "input"][0].k]

    def _get_output_ks_in_order(self):
        """
        Get the Keras function for each output layer, in order.
        """
        if self.output_layer_order:
            result = []
            for name in self.output_layer_order:
                if name in [layer.name for layer in self.layers if layer.kind() == "output"]:
                    result.append(self[name].k)
            return result
        else:
            # the one output name:
            return [[layer for layer in self.layers if layer.kind() == "output"][0].k]

    def _get_ordered_output_layers(self):
        """
        Return the ordered output layers' Keras functions.
        """
        if self.output_layer_order:
            layers = []
            for layer_name in self.output_layer_order:
                layers.append(self[layer_name].k)
        else:
            layers = [layer.k for layer in self.layers if layer.kind() == "output"][0]
        return layers

    def _get_ordered_input_layers(self):
        """
        Get the Keras functions for all layers, in order.
        """
        if self.input_layer_order:
            layers = []
            for layer_name in self.input_layer_order:
                layers.append(self[layer_name].k)
        else:
            layers = [layer.k for layer in self.layers if layer.kind() == "input"][0]
        return layers

    def _image_to_uri(self, img_src):
        # Convert to binary data:
        b = io.BytesIO()
        img_src.save(b, format='gif')
        data = b.getvalue()
        data = base64.b64encode(data)
        if not isinstance(data, str):
            data = data.decode("latin1")
        return "data:image/gif;base64,%s" % data

    def build_svg(self, opts={}):
        """
        opts - temporary override of config

        includes:
            "font_size": 12,
            "border_top": 25,
            "border_bottom": 25,
            "hspace": 100,
            "vspace": 50,
            "image_maxdim": 200

        See .config for all options.
        """
        # defaults:
        config = copy.copy(self.config)
        config.update(opts)
        self.visualize = False # so we don't try to update previously drawn images
        ordering = list(reversed(self._get_level_ordering())) # list of names per level, input to output
        image_svg = """<rect x="{{rx}}" y="{{ry}}" width="{{rw}}" height="{{rh}}" style="fill:none;stroke:{border_color};stroke-width:{border_width}"/><image id="{netname}_{{name}}_{{svg_counter}}" class="{netname}_{{name}}" x="{{x}}" y="{{y}}" height="{{height}}" width="{{width}}" href="{{image}}"><title>{{tooltip}}</title></image>""".format(
            **{
                "netname": self.name,
                "border_color": config["border_color"],
                "border_width": config["border_width"],
            })
        arrow_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{arrow_color}" stroke-width="{arrow_width}" marker-end="url(#arrow)"><title>{{tooltip}}</title></line>""".format(**self.config)
        arrow_rect = """<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" style="fill:white;stroke:none"><title>{tooltip}</title></rect>"""
        label_svg = """<text x="{x}" y="{y}" font-family="{font_family}" font-size="{font_size}">{label}</text>"""
        max_width = 0
        images = {}
        image_dims = {}
        row_height = []
        # Go through and build images, compute max_width:
        for level_names in ordering:
            # first make all images at this level
            total_width = 0 # for this row
            max_height = 0
            for layer_name in level_names:
                if not self[layer_name].visible:
                    continue
                if self.model: # thus, we can propagate
                    if self.get_inputs_length() != 0:
                        v = self.get_input(0)
                    else:
                        if self.input_layer_order:
                            v = []
                            for in_name in self.input_layer_order:
                                v.append(self[in_name].make_dummy_vector())
                        else:
                            in_layer = [layer for layer in self.layers if layer.kind() == "input"][0]
                            v = in_layer.make_dummy_vector()
                    image = self.propagate_to_image(layer_name, v)
                else: # no propagate
                    # get image based on ontputs
                    raise Exception("compile model before building svg")
                (width, height) = image.size
                images[layer_name] = image ## little image
                max_dim = max(width, height)
                if self[layer_name].image_maxdim:
                    image_maxdim = self[layer_name].image_maxdim
                else:
                    image_maxdim = config["image_maxdim"]
                width, height = (int(width/max_dim * image_maxdim),
                                 int(height/max_dim * image_maxdim))
                if min(width, height) < 25:
                    width, height = (image_maxdim, 25)
                image_dims[layer_name] = (width, height)
                total_width += width + config["hspace"] # space between
                max_height = max(max_height, height)
            row_height.append(max_height)
            max_width = max(max_width, total_width)
        svg = ""
        cheight = config["border_top"] # top border
        ## Display target?
        if config["show_targets"]:
            # Find the spacing for row:
            row_layer_width = 0
            for layer_name in ordering[0]:
                if not self[layer_name].visible:
                    continue
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                row_layer_width += width
            spacing = (max_width - row_layer_width) / (len(ordering[0]) + 1)
            # draw the row of targets:
            cwidth = spacing
            for layer_name in ordering[0]:
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                svg += image_svg.format(**{"name": layer_name + "_targets",
                                           "svg_counter": self._svg_counter,
                                           "x": cwidth,
                                           "y": cheight,
                                           "image": self._image_to_uri(image),
                                           "width": width,
                                           "height": height,
                                           "tooltip": self[layer_name].tooltip(),
                                           "rx": cwidth - 1, # based on arrow width
                                           "ry": cheight - 1,
                                           "rh": height + 2,
                                           "rw": width + 2})
                ## show a label
                svg += label_svg.format(
                    **{"x": cwidth + width + 5,
                       "y": cheight + height/2 + 2,
                       "label": "targets",
                       "font_size": config["font_size"],
                       "font_family": config["font_family"],
                    })
                cwidth += width + spacing
            ## Then we need to add height for output layer again, plus a little bit
            cheight += row_height[0] + 10 # max height of row, plus some
        ## Display error?
        if config["show_errors"]:
            # Find the spacing for row:
            row_layer_width = 0
            for layer_name in ordering[0]:
                if not self[layer_name].visible:
                    continue
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                row_layer_width += width
            spacing = (max_width - row_layer_width) / (len(ordering[0]) + 1)
            # draw the row of errors:
            cwidth = spacing
            for layer_name in ordering[0]:
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                svg += image_svg.format(**{"name": layer_name + "_errors",
                                           "svg_counter": self._svg_counter,
                                           "x": cwidth,
                                           "y": cheight,
                                           "image": self._image_to_uri(image),
                                           "width": width,
                                           "height": height,
                                           "tooltip": self[layer_name].tooltip(),
                                           "rx": cwidth - 1, # based on arrow width
                                           "ry": cheight - 1,
                                           "rh": height + 2,
                                           "rw": width + 2})
                ## show a label
                svg += label_svg.format(
                    **{"x": cwidth + width + 5,
                       "y": cheight + height/2 + 2,
                       "label": "errors",
                       "font_size": config["font_size"],
                       "font_family": config["font_family"],
                    })
                cwidth += width + spacing
            ## Then we need to add height for output layer again, plus a little bit
            cheight += row_height[0] + 10 # max height of row, plus some
        # Now we go through again and build SVG:
        positioning = {}
        for level_names in ordering:
            # compute width of just pictures for this row:
            row_layer_width = 0
            for layer_name in level_names:
                if not self[layer_name].visible:
                    continue
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                row_layer_width += width
            spacing = (max_width - row_layer_width) / (len(level_names) + 1)
            cwidth = spacing
            # See if there are any connections up:
            any_connections_up = False
            last_connections_up = False
            for layer_name in level_names:
                if not self[layer_name].visible:
                    continue
                for out in self[layer_name].outgoing_connections:
                    if out.name not in positioning:
                        continue
                    any_connections_up = True
            if any_connections_up:
                cheight += config["vspace"] # for arrows
            else: # give a bit of room:
                if not last_connections_up:
                    cheight += 5
            last_connections_up = any_connections_up
            max_height = 0 # for row of images
            for layer_name in level_names:
                if not self[layer_name].visible:
                    continue
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                positioning[layer_name] = {"name": layer_name,
                                           "svg_counter": self._svg_counter,
                                           "x": cwidth,
                                           "y": cheight,
                                           "image": self._image_to_uri(image),
                                           "width": width,
                                           "height": height,
                                           "tooltip": self[layer_name].tooltip(),
                                           "rx": cwidth - 1, # based on arrow width
                                           "ry": cheight - 1,
                                           "rh": height + 2,
                                           "rw": width + 2}
                x1 = cwidth + width/2
                y1 = cheight - 1
                for out in self[layer_name].outgoing_connections:
                    if out.name not in positioning:
                        continue
                    # draw background to arrows to allow mouseover tooltips:
                    x2 = positioning[out.name]["x"] + positioning[out.name]["width"]/2
                    y2 = positioning[out.name]["y"] + positioning[out.name]["height"]
                    rect_width = abs(x1 - x2)
                    rect_extra = 0
                    if rect_width < 20:
                        rect_extra = 10
                    tooltip = self.describe_connection_to(self[layer_name], out)
                    svg += arrow_rect.format(**{"tooltip": tooltip,
                                                "rx": min(x2, x1) - rect_extra,
                                                "ry": min(y2, y1) + 2, # bring down
                                                "rw": rect_width + rect_extra * 2,
                                                "rh": abs(y1 - y2) - 2})
                for out in self[layer_name].outgoing_connections:
                    if out.name not in positioning:
                        continue
                    # draw an arrow between layers:
                    tooltip = self.describe_connection_to(self[layer_name], out)
                    x2 = positioning[out.name]["x"] + positioning[out.name]["width"]/2
                    y2 = positioning[out.name]["y"] + positioning[out.name]["height"]
                    svg += arrow_svg.format(
                        **{"x1":x1,
                           "y1":y1,
                           "x2":x2,
                           "y2":y2 + 2,
                           "tooltip": tooltip
                        })
                svg += image_svg.format(**positioning[layer_name])
                svg += label_svg.format(
                    **{"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] + 5,
                       "y": positioning[layer_name]["y"] + positioning[layer_name]["height"]/2 + 2,
                       "label": layer_name,
                       "font_size": config["font_size"],
                       "font_family": config["font_family"],
                    })
                cwidth += width + spacing # spacing between
                max_height = max(max_height, height)
                self._svg_counter += 1
            cheight += max_height
        cheight += config["border_bottom"]
        self.visualize = True
        if get_ipython():
            self._initialize_javascript()
        return ("""
        <svg id='{netname}' xmlns='http://www.w3.org/2000/svg' width="{width}" height="{height}" image-rendering="pixelated">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="{arrow_color}" />
        </marker>
    </defs>
""".format(
    **{
        "width": max_width,
        "height": cheight,
        "netname": self.name,
        "arrow_color": config["arrow_color"],
        "arrow_width": config["arrow_width"],
    }) + svg + """</svg>""")

    def _initialize_javascript(self):
        from IPython.display import Javascript, display
        js = """
require(['base/js/namespace'], function(Jupyter) {
    Jupyter.notebook.kernel.comm_manager.register_target('conx_svg_control', function(comm, msg) {
        comm.on_msg(function(msg) {
            var data = msg["content"]["data"];
            var images = document.getElementsByClassName(data["class"]);
            for (var i = 0; i < images.length; i++) {
                images[i].setAttributeNS(null, "href", data["href"]);
            }
        });
    });
});
"""
        display(Javascript(js))

    def _get_level_ordering(self):
        ## First, get a level for all layers:
        levels = {}
        for layer in topological_sort(self, self.layers):
            if not hasattr(layer, "model"):
                continue
            level = max([levels[lay.name] for lay in layer.incoming_connections] + [-1])
            levels[layer.name] = level + 1
        max_level = max(levels.values())
        # Now, sort by input layer indices:
        ordering = []
        for i in range(max_level + 1):
            layer_names = [layer.name for layer in self.layers if levels[layer.name] == i]
            if self.input_layer_order:
                inputs = [([self.input_layer_order.index(name)
                            for name in self[layer_name].input_names], layer_name)
                          for layer_name in layer_names]
            else:
                inputs = [([0 for name in self[layer_name].input_names], layer_name)
                          for layer_name in layer_names]
            level = [row[1] for row in sorted(inputs)]
            ordering.append(level)
        return ordering

    def describe_connection_to(self, layer1, layer2):
        """
        Returns a textual description of the weights for the SVG tooltip.
        """
        retval = "Weights from %s to %s" % (layer1.name, layer2.name)
        for klayer in self.model.layers:
            if klayer.name == layer2.name:
                weights = klayer.get_weights()
                for w in range(len(klayer.weights)):
                    retval += "\n %s has shape %s" % (klayer.weights[w], weights[w].shape)
        ## FIXME: how to show merged layer weights?
        return retval

    def save(self, filename=None):
        """
        Save the weights to a file.
        """
        if filename is None:
            filename = "%s.wts" % self.name
        with open(filename, "wb") as fp:
            for layer in self.model.layers:
                for weight in layer.get_weights():
                    np.save(fp, weight)

    def load(self, filename=None):
        """
        Load the weights from a file.
        """
        if filename is None:
            filename = "%s.wts" % self.name
        with open(filename, "rb") as fp:
            for layer in self.model.layers:
                weights = layer.get_weights()
                new_weights = []
                for w in range(len(weights)):
                    new_weights.append(np.load(fp))
                layer.set_weights(new_weights)

    def get_inputs_length(self):
        """
        Get the number of input patterns.
        """
        if len(self.inputs) == 0:
            return 0
        if self.multi_inputs:
            return self.inputs[0].shape[0]
        else:
            return self.inputs.shape[0]

    def get_targets_length(self):
        """
        Get the number of target patterns.
        """
        if len(self.targets) == 0:
            return 0
        if self.multi_targets:
            return self.targets[0].shape[0]
        else:
            return self.targets.shape[0]

    def get_test_inputs_length(self):
        """
        Get the number of test input patterns.
        """
        if len(self.test_inputs) == 0:
            return 0
        if self.multi_inputs:
            return self.test_inputs[0].shape[0]
        else:
            return self.test_inputs.shape[0]

    def get_test_targets_length(self):
        """
        Get the number of test target patterns.
        """
        if len(self.test_targets) == 0:
            return 0
        if self.multi_targets:
            return self.test_targets[0].shape[0]
        else:
            return self.test_targets.shape[0]

    def get_train_inputs_length(self):
        """
        Get the number of training input patterns.
        """
        if len(self.train_inputs) == 0:
            return 0
        if self.multi_inputs:
            return self.train_inputs[0].shape[0]
        else:
            return self.train_inputs.shape[0]

    def get_train_targets_length(self):
        """
        Get the number of training target patterns.
        """
        if len(self.train_targets) == 0:
            return 0
        if self.multi_targets:
            return self.train_targets[0].shape[0]
        else:
            return self.train_targets.shape[0]

    def build_widget(self, width="100%", height="550px"):
        """
        Build the control-panel for Jupyter widgets. Requires running
        in a notebook/jupyterlab.
        """
        from ipywidgets import HTML, Button, VBox, HBox, IntSlider, Select, Layout

        def dataset_move(position):
            if control_select.value == "Train":
                length = self.get_train_inputs_length()
            elif control_select.value == "Test":
                length = self.get_test_inputs_length()
            #### Position it:
            if position == "begin":
                control_slider.value = 0
            elif position == "end":
                control_slider.value = length - 1
            elif position == "prev":
                control_slider.value = max(control_slider.value - 1, 0)
            elif position == "next":
                control_slider.value = min(control_slider.value + 1, length - 1)

        def update_control_slider(change):
            if control_select.value == "Test":
                control_slider.value = 0
                control_slider.min = 0
                control_slider.max = max(self.get_test_inputs_length() - 1, 0)
                if self.get_test_inputs_length() == 0:
                    disabled = True
                else:
                    disabled = False
            elif control_select.value == "Train":
                control_slider.value = 0
                control_slider.min = 0
                control_slider.max = max(self.get_train_inputs_length() - 1, 0)
                if self.get_train_inputs_length() == 0:
                    disabled = True
                else:
                    disabled = False
            control_slider.disabled = disabled
            for child in control_buttons.children:
                child.disabled = disabled

        def update_slider_control(change):
            if change["name"] == "value":
                if control_select.value == "Train" and self.get_train_targets_length() > 0:
                    output = self.propagate(self.get_train_input(control_slider.value))
                    if self.config["show_targets"]:
                        self.display_component([self.get_train_target(control_slider.value)], "targets", minmax=(0, 1))
                    if self.config["show_errors"]:
                        errors = np.array(self.get_train_target(control_slider.value)) - np.array(output)
                        self.display_component([errors.tolist()], "errors", minmax=(-1, 1), colormap="hot")
                elif control_select.value == "Test" and self.get_test_targets_length() > 0:
                    output = self.propagate(self.get_test_input(control_slider.value))
                    if self.config["show_targets"]:
                        self.display_component([self.get_test_target(control_slider.value)], "targets", minmax=(0, 1))
                    if self.config["show_errors"]:
                        errors = np.array(self.get_test_target(control_slider.value)) - np.array(output)
                        self.display_component([errors.tolist()], "errors", minmax=(-1, 1), colormap="hot")

        def train_one(button):
            if control_select.value == "Train" and self.get_train_targets_length() > 0:
                outputs = self.train_one(self.get_train_input(control_slider.value),
                                       self.get_train_target(control_slider.value))
            elif control_select.value == "Test" and self.get_test_targets_length() > 0:
                outputs = self.train_one(self.get_test_input(control_slider.value),
                                       self.get_test_target(control_slider.value))

        net_svg = HTML(value=self.build_svg(), layout=Layout(width=width, height=height, overflow_x='auto'))
        button_begin = Button(icon="fast-backward", layout=Layout(width='100%'))
        button_prev = Button(icon="backward", layout=Layout(width='100%'))
        button_next = Button(icon="forward", layout=Layout(width='100%'))
        button_end = Button(icon="fast-forward", layout=Layout(width='100%'))
        button_train = Button(description="Train", layout=Layout(width='100%'))
        control_buttons = HBox([
            button_begin,
            button_prev,
            button_train,
            button_next,
            button_end,
               ], layout=Layout(width='100%'))
        control_select = Select(
            options=['Test', 'Train'],
            value='Train',
            description='Dataset:',
               )
        control_slider = IntSlider(description="Dataset position",
                                   continuous_update=False,
                                   min=0,
                                   max=max(self.get_train_inputs_length() - 1, 0),
                                   value=0,
                                   layout=Layout(width='100%'))

        ## Hook them up:
        button_begin.on_click(lambda button: dataset_move("begin"))
        button_end.on_click(lambda button: dataset_move("end"))
        button_next.on_click(lambda button: dataset_move("next"))
        button_prev.on_click(lambda button: dataset_move("prev"))
        button_train.on_click(train_one)
        control_select.observe(update_control_slider)
        control_slider.observe(update_slider_control)

        # Put them together:
        control = VBox([control_select, control_slider, control_buttons], layout=Layout(width='100%'))
        widget = VBox([net_svg, control], layout=Layout(width='100%'))
        widget.on_displayed(lambda widget: update_slider_control({"name": "value"}))
        return widget

    def pp(self, *args, **opts):
        """
        Pretty-print a vector.
        """
        if isinstance(args[0], str):
            label = args[0]
            vector = args[1]
        else:
            label = ""
            vector = args[0]
        print(label + self.ppf(vector[:20], **opts))

    def ppf(self, vector, **opts):
        """
        Pretty-format a vector.
        """
        config = copy.copy(self.config)
        config.update(opts)
        max_length = config["pp_max_length"]
        precision = config["pp_precision"]
        truncated = len(vector) > max_length
        return "[" + ", ".join([("%." + str(precision) + "f") % v for v in vector[:max_length]]) + ("..." if truncated else "") + "]"

    ## FIXME: add these:
    #def to_array(self):
    #def from_array(self):

class InterruptHandler():
    """
    Class for handling interrupts so that state is not left
    in inconsistant situation.
    """
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self._release()
            if self.interrupted:
                raise KeyboardInterrupt
            print("\nStopping at end of epoch... (^C again to quit now)...")
            self.interrupted = True

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self._release()

    def _release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
