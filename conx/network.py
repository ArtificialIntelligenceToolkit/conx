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

import operator
import importlib
from functools import reduce
import signal
import numbers
import base64
import io

import numpy as np
import keras
from keras.models import Model

from .utils import *
from .layers import Layer

#------------------------------------------------------------------------

class Network():
    def __init__(self, name, *sizes, **kwargs):
        """
        Create a neural network.
        if sizes is given, create a full network.
        Optional keywork: activation
        """
        if not isinstance(name, str):
            raise Exception("first argument should be a name for the network")
        self.name = name
        self.layers = []
        self.layer_dict = {}
        self.inputs = None
        self.train_inputs = []
        self.train_targets = []
        self.test_inputs = []
        self.test_targets = []
        self.labels = None
        self.targets = None
        # If simple feed-forward network:
        for i in range(len(sizes)):
            if i > 0:
                self.add(Layer(autoname(i, len(sizes)), shape=sizes[i],
                               activation=kwargs.get("activation", "linear")))
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
        self.visualize = False
        self._comm = None
        self.inputs_range = (0,0)
        self.targets_range = (0,0)
        self.test_labels = []
        self.train_labels = []
        self.model = None
        self.split = 0

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
        if layer.name in self.layer_dict:
            raise Exception("duplicate layer name '%s'" % layer.name)
        self.layers.append(layer)
        self.layer_dict[layer.name] = layer

    def connect(self, from_layer_name=None, to_layer_name=None):
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
        for layer in self.layers:
            layer.summary()

    def set_dataset_direct(self, inputs, targets, verbose=True):
        self.inputs = inputs
        self.targets = targets
        self.labels = None
        self._cache_dataset_values()
        self.split_dataset(self.num_inputs, verbose=False)
        if verbose:
            self.summary_dataset()

    def _cache_dataset_values(self):
        if self.num_input_layers == 1:
            self.inputs_range = (self.inputs.min(), self.inputs.max())
            self.num_inputs = self.inputs.shape[0]
        else:
            self.inputs_range = (min([x.min() for x in self.inputs]),
                                 max([x.max() for x in self.inputs]))
            self.num_inputs = self.inputs[0].shape[0]
        if self.targets is not None:
            if self.num_target_layers == 1:
                self.targets_range = (self.targets.min(), self.targets.max())
            else:
                self.targets_range = (min([x.min() for x in self.targets]),
                                      max([x.max() for x in self.targets]))
        else:
            self.targets_range = (0, 0)

    def set_dataset(self, pairs, verbose=True):
        if self.num_input_layers == 1:
            self.inputs = np.array([x for (x, y) in pairs], "float32")
        else:
            self.inputs = []
            for i in range(len(pairs[0][0])):
                self.inputs.append(np.array([x[i] for (x,y) in pairs], "float32"))
        if self.num_target_layers == 1:
            self.targets = np.array([y for (x, y) in pairs], "float32")
        else:
            self.targets = []
            for i in range(len(pairs[0][1])):
                self.targets.append(np.array([y[i] for (x,y) in pairs], "float32"))
        self.labels = None
        self._cache_dataset_values()
        self.split_dataset(self.num_inputs, verbose=False)
        if verbose:
            self.summary_dataset()

    def load_keras_dataset(self, name, verbose=True):
        available_datasets = [x for x in dir(keras.datasets) if '__' not in x and x != 'absolute_import']
        if name not in available_datasets:
            s = "unknown keras dataset: %s" % name
            s += "\navailable datasets: %s" % ','.join(available_datasets)
            raise Exception(s)
        if verbose:
            print('Loading %s dataset...' % name)
        load_data = importlib.import_module('keras.datasets.' + name).load_data
        (x_train,y_train), (x_test,y_test) = load_data()
        self.inputs = np.concatenate((x_train,x_test))
        self.labels = np.concatenate((y_train,y_test))
        self.targets = None
        self._cache_dataset_values()
        self.split_dataset(self.num_inputs, verbose=False)
        if verbose:
            self.summary_dataset()

    def load_npz_dataset(self, filename, verbose=True):
        """loads a dataset from an .npz file and returns data, labels"""
        if filename[-4:] != '.npz':
            raise Exception("filename must end in .npz")
        if verbose:
            print('Loading %s dataset...' % filename)
        try:
            f = np.load(filename)
            self.inputs = f['data']
            self.labels = f['labels']
            self.targets = None
            if len(self.inputs) != len(self.labels):
                raise Exception("Dataset contains different numbers of inputs and labels")
            if len(self.inputs) == 0:
                raise Exception("Dataset is empty")
            self._cache_dataset_values()
            self.split_dataset(self.num_inputs, verbose=False)
            if verbose:
                self.summary_dataset()
        except:
            raise Exception("couldn't load .npz dataset %s" % filename)

    def reshape_inputs(self, new_shape, verbose=True):
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if not valid_shape(new_shape):
            raise Exception("bad shape: %s" % (new_shape,))
        if isinstance(new_shape, numbers.Integral):
            new_size = self.num_inputs * new_shape
        else:
            new_size = self.num_inputs * reduce(operator.mul, new_shape)
        ## FIXME: work on multi-inputs?
        if new_size != self.inputs.size:
            raise Exception("shape %s is incompatible with inputs" % (new_shape,))
        if isinstance(new_shape, numbers.Integral):
            new_shape = (new_shape,)
        self.inputs = self.inputs.reshape((self.num_inputs,) + new_shape)
        self.split_dataset(self.split, verbose=False)
        if verbose:
            print('Input data shape: %s, range: %s, type: %s' %
                  (self.inputs.shape[1:], self.inputs_range, self.inputs.dtype))

    def set_input_layer_order(self, *layer_names):
        if len(layer_names) == 1:
            raise Exception("set_input_layer_order cannot be a single layer")
        self.input_layer_order = []
        for layer_name in layer_names:
            if layer_name not in self.input_layer_order:
                self.input_layer_order.append(layer_name)
            else:
                raise Exception("duplicate name in set_input_layer_order: '%s'" % layer_name)

    def set_output_layer_order(self, *layer_names):
        if len(layer_names) == 1:
            raise Exception("set_output_layer_order cannot be a single layer")
        self.output_layer_order = []
        for layer_name in layer_names:
            if layer_name not in self.output_layer_order:
                self.output_layer_order.append(layer_name)
            else:
                raise Exception("duplicate name in set_output_layer_order: '%s'" % layer_name)

    def set_targets_to_categories(self, num_classes):
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if not isinstance(num_classes, numbers.Integral) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self.targets = keras.utils.to_categorical(self.labels, num_classes).astype("uint8")
        self.train_targets = self.targets[:self.split]
        self.test_targets = self.targets[self.split:]
        print('Generated %d target vectors from labels' % self.num_inputs)

    def summary_dataset(self):
        if self.num_inputs == 0:
            print("no dataset loaded")
            return
        print('%d train inputs, %d test inputs' %
              (len(self.train_inputs), len(self.test_inputs)))
        if self.inputs is not None:
            if self.num_input_layers == 1:
                print('Set %d inputs and targets' % (self.num_inputs,))
                print('Input data shape: %s, range: %s, type: %s' %
                      (self.inputs.shape[1:], self.inputs_range, self.inputs.dtype))
            else:
                print('Set %d inputs and targets' % (self.num_inputs,))
                print('Input data shapes: %s, range: %s, types: %s' %
                      ([x[0].shape for x in self.inputs],
                       self.inputs_range,
                       [x[0].dtype for x in self.inputs]))
        else:
            print("No inputs")
        if self.targets is not None:
            if self.num_target_layers == 1:
                print('Target data shape: %s, range: %s, type: %s' %
                      (self.targets.shape[1:], self.targets_range, self.targets.dtype))
            else:
                print('Target data shapes: %s, range: %s, types: %s' %
                      ([x[0].shape for x in self.targets],
                       self.targets_range,
                       [x[0].dtype for x in self.targets]))
        else:
            print("No targets")

    def rescale_inputs(self, old_range, new_range, new_dtype):
        old_min, old_max = old_range
        new_min, new_max = new_range
        ## FIXME: work on multi-inputs?
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
        return np.array(span * np.random.rand(size) - range/2.0,
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
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        indices = np.random.permutation(self.num_inputs)
        self.inputs = self.inputs[indices]
        if self.labels is not None:
            self.labels = self.labels[indices]
        if self.targets is not None:
            self.targets = self.targets[indices]
        self.split_dataset(self.split, verbose=False)
        if verbose:
            print('Shuffled all %d inputs' % self.num_inputs)

    def split_dataset(self, split=0.50, verbose=True):
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
        if self.num_input_layers == 1:
            self.train_inputs = self.inputs[:self.split]
            self.test_inputs = self.inputs[self.split:]
        else:
            self.train_inputs = [col[:self.split] for col in self.inputs]
            self.test_inputs = [col[self.split:] for col in self.inputs]
        if self.labels is not None:
            self.train_labels = self.labels[:self.split]
            self.test_labels = self.labels[self.split:]
        if self.targets is not None:
            if self.num_target_layers == 1:
                self.train_targets = self.targets[:self.split]
                self.test_targets = self.targets[self.split:]
            else:
                self.train_targets = [col[:self.split] for col in self.targets]
                self.test_targets = [col[self.split:] for col in self.targets]
        if verbose:
            print('Split dataset into:')
            if self.num_input_layers == 1:
                print('   %d train inputs' % len(self.train_inputs))
            else:
                print('   %d train inputs' % len(self.train_inputs[0]))
            if self.num_input_layers == 1:
                print('   %d test inputs' % len(self.test_inputs))
            else:
                print('   %d test inputs' % len(self.test_inputs[0]))

    def test(self, dataset=None, batch_size=None):
        if dataset is None:
            if self.split == self.num_inputs:
                dataset = self.train_inputs
            else:
                dataset = self.test_inputs
        print("Testing...")
        if batch_size is not None:
            outputs = self.model.predict(dataset, batch_size=batch_size)
        else:
            outputs = self.model.predict(dataset)
        if self.num_target_layers > 1:
            outputs = [[list(y) for y in x] for x in zip(*outputs)]
        for output in outputs:
            print(output)

    def train(self, epochs=1, accuracy=None, batch_size=None,
              report_rate=1, tolerance=0.1, verbose=1, shuffle=True,
              class_weight=None, sample_weight=None):
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
                if batch_size is not None:
                    outputs = self.model.predict(validation_inputs, batch_size=batch_size)
                else:
                    outputs = self.model.predict(validation_inputs)
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
                    if batch_size is not None:
                        outputs = self.model.predict(validation_inputs, batch_size=batch_size)
                    else:
                        outputs = self.model.predict(validation_inputs)
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
                        if verbose: print("Epoch #%5d | train loss %7.5f | train acc %7.5f | validate%% %7.5f" %
                                          (self.epoch_count, loss, acc, val_percent))
                    if val_percent >= accuracy or handler.interrupted:
                        break
            if handler.interrupted:
                print("=" * 72)
                print("Epoch #%5d | train loss %7.5f | train acc %7.5f | validate%% %7.5f" %
                      (self.epoch_count, loss, acc, val_percent))
                raise KeyboardInterrupt
        if verbose:
            print("=" * 72)
            print("Epoch #%5d | train loss %7.5f | train acc %7.5f | validate%% %7.5f" %
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
        Get an input from the internal dataset and
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
        Get a target from the internal dataset and
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
        Get an input from the internal dataset and
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
        Get a target from the internal dataset and
        format it in the human API.
        """
        if self.num_target_layers == 1:
            return list(self.test_targets[i])
        else:
            targets = []
            for c in range(self.num_target_layers):
                targets.append(list(self.test_targets[c][i]))
            return targets

    def propagate(self, input, batch_size=None):
        if self.num_input_layers == 1:
            if batch_size is not None:
                outputs = list(self.model.predict(np.array([input]), batch_size=batch_size)[0])
            else:
                outputs = list(self.model.predict(np.array([input]))[0])
        else:
            inputs = [np.array(x, "float32") for x in input]
            if batch_size is not None:
                outputs = [[list(y) for y in x][0] for x in self.model.predict(inputs, batch_size=batch_size)]
            else:
                outputs = [[list(y) for y in x][0] for x in self.model.predict(inputs)]
        if self.visualize:
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            for layer in self.layers:
                image = self.propagate_to_image(layer.name, input, batch_size)
                data_uri = self._image_to_uri(image)
                self._comm.send({'id': "%s_%s" % (self.name, layer.name), "href": data_uri})
        return outputs

    def propagate_from(self, layer_name, input, batch_size=None):
        ## FIXME:
        ## This needs to be build brand new from here to output(s)
        # if self[layer_name].decode_model is None:
        #     outks = self._get_output_ks_in_order()
        #     if len(outks) == 1:
        #         self[layer_name].decode_model = Model(inputs=Input(self[layer_name].shape), outputs=outks[0])
        #     else:
        #         self[layer_name].decode_model = Model(inputs=Input(self[layer_name].shape), outputs=outks)
        # if batch_size is None:
        #     outputs = [[list(y) for y in x][0] for x in self.decode_model.predict(inputs)]
        # else:
        #     outputs = [[list(y) for y in x][0] for x in self.decode_model.predict(inputs, batch_size=batch_size)]
        # if self.visualize:
        #     ## FIXME: should visualize this all the way through
        #     ## For now, we just update the outputs
        #     if not self._comm:
        #         from ipykernel.comm import Comm
        #         self._comm = Comm(target_name='conx_svg_control')
        #     if self.output_layer_order:
        #         layers = self.output_layer_order # layer_names
        #         outpairs = zip(layers, outputs)
        #     else:
        #         layers = [layer.name for layer in self.layers if layer.kind() == "output"]
        #         outpairs = [(layers[0], outputs)]
        #     for (layer_name, vector) in vectors:
        #         image = self._make_image(layer_name, vector, colormap=self[layer_name].colormap)
        #         data_uri = self._image_to_uri(image)
        #         self._comm.send({'id': "%s_%s" % (self.name, layer.name), "href": data_uri})
        # return outputs
        return

    def propagate_to(self, layer_name, input, batch_size=None):
        """
        Computes activation at a layer. Side-effect: updates visualized SVG.
        """
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        if self.num_input_layers == 1:
            if batch_size is not None:
                outputs = list(self[layer_name].propagate_to(np.array([input]), batch_size=batch_size)[0])
            else:
                outputs = list(self[layer_name].propagate_to(np.array([input]))[0])
        else:
            inputs = [np.array(x, "float32") for x in input]
            # get just inputs for this layer, in order:
            inputs = [inputs[self.input_layer_order.index(name)] for name in self[layer_name].input_names]
            if len(inputs) == 1:
                inputs = inputs[0]
            if batch_size is not None:
                outputs = list(self[layer_name].propagate_to(inputs, batch_size=batch_size)[0])
            else:
                outputs = list(self[layer_name].propagate_to(inputs)[0])
        if self.visualize:
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            array = np.array(outputs)
            image = self._make_image(layer_name, array, colormap=self[layer_name].colormap)
            data_uri = self._image_to_uri(image)
            self._comm.send({'id': "%s_%s" % (self.name, layer_name), "href": data_uri})
        return outputs

    def propagate_to_image(self, layer_name, input, batch_size=None):
        """
        Gets an image of activations at a layer.
        """
        outputs = self.propagate_to(layer_name, input, batch_size)
        array = np.array(outputs)
        image = self._make_image(layer_name, array, colormap=self[layer_name].colormap)
        return image

    def compile(self, **kwargs):
        ## Error checking:
        if len(self.layers) == 0:
            raise Exception("network has no layers")
        for layer in self.layers:
            if layer.kind() == 'unconnected':
                raise Exception("'%s' layer is unconnected" % layer.name)
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
        sequence = topological_sort(self)
        for layer in sequence:
            if layer.kind() == 'input':
                layer.k = layer.make_input_layer_k()
                layer.input_names = [layer.name]
                layer.model = Model(inputs=layer.k, outputs=layer.k) # identity
            else:
                if len(layer.incoming_connections) == 0:
                    raise Exception("non-input layer '%s' with no incoming connections" % layer.name)
                kfuncs = layer.make_keras_functions()
                if len(layer.incoming_connections) == 1:
                    f = layer.incoming_connections[0].k
                    layer.input_names = layer.incoming_connections[0].input_names
                else: # multiple inputs, need to merge
                    f = keras.layers.Concatenate()([incoming.k for incoming in layer.incoming_connections])
                    # flatten:
                    layer.input_names = [item for sublist in
                                         [incoming.input_names for incoming in layer.incoming_connections]
                                         for item in sublist]
                for k in kfuncs:
                    f = k(f)
                layer.k = f
                ## get the inputs to this branch, in order:
                input_ks = self._get_input_ks_in_order(layer.input_names)
                layer.model = Model(inputs=input_ks, outputs=layer.k)
        output_k_layers = self._get_ordered_output_layers()
        input_k_layers = self._get_ordered_input_layers()
        self.model = Model(inputs=input_k_layers, outputs=output_k_layers)
        kwargs['metrics'] = ['accuracy']
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

    def _get_minmax(self, layer, vector):
        if layer.minmax:
            return layer.minmax
        # ('relu', 'sigmoid', 'linear', 'softmax', 'tanh')
        if layer.activation in ["tanh"]:
            return (-1,+1)
        elif layer.activation in ["sigmoid", "softmax"]:
            return (0,+1)
        elif layer.activation in ["relu"]:
            return (0,vector.max())
        else: # activation in ["linear"] or otherwise
            return (-1,+1)

    def _scale_output_for_image(self, vector, minmax):
        """
        Given an activation name (or something else) and an output
        vector, scale the vector.
        """
        return rescale_numpy_array(vector, minmax, (0,255), 'uint8')

    def _make_image(self, layer_name, vector, size=25, transpose=False, colormap=None):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget.
        """
        from matplotlib import cm
        import PIL
        layer = self[layer_name]
        if layer.vshape != layer.shape:
            vector = vector.reshape(layer.vshape)
        minmax = self._get_minmax(layer, vector)
        vector = self._scale_output_for_image(vector, minmax)
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        new_width = vector.shape[0] * size # in, pixels
        new_height = vector.shape[1] * size # in, pixels
        if colormap:
            cm_hot = cm.get_cmap(colormap)
            vector = cm_hot(vector)
            vector = np.uint8(vector * 255)
            image = PIL.Image.fromarray(vector)
        else:
            image = PIL.Image.fromarray(vector, 'P')
        image = image.resize((new_height, new_width))
        return image

    def _image_to_uri(self, img_src):
        # Convert to binary data:
        b = io.BytesIO()
        img_src.save(b, format='gif')
        data = b.getvalue()
        data = base64.b64encode(data)
        if not isinstance(data, str):
            data = data.decode("latin1")
        return "data:image/gif;base64,%s" % data

    def _make_dummy_vector(self, layer):
        v = np.ones(layer.shape)
        lo, hi = self._get_minmax(layer, v)
        v *= (lo + hi) / 2.0
        return v

    def build_svg(self):
        self.visualize = False # so we don't try to update previously drawn images
        ordering = list(reversed(self._get_level_ordering())) # list of names per level, input to output
        image_svg = """<rect x="{{rx}}" y="{{ry}}" width="{{rw}}" height="{{rh}}" style="fill:none;stroke:blue;stroke-width:2"/><image id="{netname}_{{name}}" x="{{x}}" y="{{y}}" height="{{height}}" width="{{width}}" href="{{image}}"><title>{{tooltip}}</title></image>""".format(**{"netname": self.name})
        arrow_svg = """<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="blue" stroke-width="2" marker-end="url(#arrow)"><title>{tooltip}</title></line>"""
        arrow_rect = """<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" style="fill:white;stroke:none"><title>{tooltip}</title></rect>"""
        label_svg = """<text x="{x}" y="{y}" font-family="Verdana" font-size="{size}">{label}</text>"""
        total_height = 25 # top border
        max_width = 0
        images = {}
        # Go through and build images, compute size:
        for level_names in ordering:
            # first make all images at this level
            max_height = 0 # per row
            total_width = 0
            for layer_name in level_names:
                if self.inputs is not None:
                    v = self.get_input(0)
                else:
                    v = self._make_dummy_vector(self[layer_name])
                image = self.propagate_to_image(layer_name, v)
                (width, height) = image.size
                max_dim = max(width, height)
                if max_dim > 200:
                    image = image.resize((int(width/max_dim * 200), int(height/max_dim * 200)))
                    (width, height) = image.size
                images[layer_name] = image
                total_width += width + 100 # space between
                max_height = max(max_height, height)
            total_height += max_height + 50 # 50 for arrows
            max_width = max(max_width, total_width)
        # Now we go through again and build SVG:
        svg = ""
        cheight = 25 # top border
        positioning = {}
        for level_names in ordering:
            row_layer_width = 0
            for layer_name in level_names:
                image = images[layer_name]
                (width, height) = image.size
                row_layer_width += width
            spacing = (max_width - row_layer_width) / (len(level_names) + 1)
            cwidth = spacing
            max_height = 0
            for layer_name in level_names:
                image = images[layer_name]
                (width, height) = image.size
                positioning[layer_name] = {"name": layer_name,
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
                    # draw an arrow between layers:
                    tooltip = self.describe_connection_to(self[layer_name], out)
                    x2 = positioning[out.name]["x"] + positioning[out.name]["width"]/2
                    y2 = positioning[out.name]["y"] + positioning[out.name]["height"]
                    svg += arrow_svg.format(**{"x1":x1,
                                               "y1":y1,
                                               "x2":x2,
                                               "y2":y2 + 2,
                                               "tooltip": tooltip})
                svg += image_svg.format(**positioning[layer_name])
                svg += label_svg.format(**{"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] + 5,
                                           "y": positioning[layer_name]["y"] + positioning[layer_name]["height"]/2 + 2,
                                           "label": layer_name,
                                           "size": 12})
                cwidth += width + spacing # spacing between
                max_height = max(max_height, height)
            cheight += max_height + 50 # 50 for arrows
        self.visualize = True
        self._initialize_javascript()
        return ("""
        <svg id='{netname}' xmlns='http://www.w3.org/2000/svg' width="{width}" height="{height}">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="blue" />
        </marker>
    </defs>
""".format(**{"width": max_width, "height": total_height, "netname": self.name}) + svg + """</svg>""")

    def _initialize_javascript(self):
        from IPython.display import Javascript, display
        js = """
require(['base/js/namespace'], function(Jupyter) {
    Jupyter.notebook.kernel.comm_manager.register_target('conx_svg_control', function(comm, msg) {
        comm.on_msg(function(msg) {
            var data = msg["content"]["data"];
            var image = document.getElementById(data["id"]);
            if (image) {
                image.setAttributeNS(null, "href", data["href"]);
            }
        });
    });
});
"""
        display(Javascript(js))

    def _get_level_ordering(self):
        ## First, get a level for all layers:
        levels = {}
        for layer in topological_sort(self):
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
        retval = "Weights from %s to %s" % (layer1.name, layer2.name)
        for klayer in self.model.layers:
            if klayer.name == layer2.name:
                weights = klayer.get_weights()
                for w in range(len(klayer.weights)):
                    retval += "\n %s has shape %s" % (klayer.weights[w], weights[w].shape)
        ## FIXME: how to show merged weights?
        return retval

    def save(self, filename=None):
        """
        Save the weights to a file.
        """
        if filename is None:
            filename = "%s.wts" % self.name
        with open(filename, "wb") as fp:
            for layer in self.model.layers:
                np.save(fp, layer.get_weights())

    def load(self, filename=None):
        """
        Load the weights from a file.
        """
        if filename is None:
            filename = "%s.wts" % self.name
        with open(filename, "rb") as fp:
            for layer in self.model.layers:
                new_weights = np.load(fp)
                layer.set_weights(new_weights)

    ## FIXME: add these:
    #def to_array(self):
    #def from_array(self):

class InterruptHandler():
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            if self.interrupted:
                raise KeyboardInterrupt
            print("\nStopping at end of epoch... (^C again to quit now)...")
            self.interrupted = True

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
