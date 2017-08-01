from __future__ import print_function, division, with_statement

#------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add, Concatenate
from keras.optimizers import RMSprop, SGD
from keras.utils import to_categorical

import glob
import random
import operator
import importlib
from functools import reduce
import signal
import base64
import io

#------------------------------------------------------------------------

def topological_sort(net):
    for layer in net.layers:
        layer.visited = False
    stack = []
    for layer in net.layers:
        if not layer.visited:
            visit(layer, stack)
    stack.reverse()
    return stack

def visit(layer, stack):
    layer.visited = True
    for outgoing_layer in layer.outgoing_connections:
        if not outgoing_layer.visited:
            visit(outgoing_layer, stack)
    stack.append(layer)

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
        self.labels = None
        self.targets = None
        # If simple feed-forward network:
        for i in range(len(sizes)):
            self.add(Layer(autoname(i, len(sizes)), shape=sizes[i],
                           activation=kwargs.get("activation", "sigmoid")))
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

    def __getitem__(self, layer_name):
        if layer_name not in self.layer_dict:
            return None
        else:
            return self.layer_dict[layer_name]

    def _repr_svg_(self):
        return self.build_svg()

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
                self.summary_database()
        except:
            raise Exception("couldn't load .npz dataset %s" % filename)

    def reshape_inputs(self, new_shape, verbose=True):
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if not valid_shape(new_shape):
            raise Exception("bad shape: %s" % (new_shape,))
        if isinstance(new_shape, int):
            new_size = self.num_inputs * new_shape
        else:
            new_size = self.num_inputs * reduce(operator.mul, new_shape)
        ## FIXME: work on multi-inputs?
        if new_size != self.inputs.size:
            raise Exception("shape %s is incompatible with inputs" % (new_shape,))
        if isinstance(new_shape, int):
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
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self.targets = to_categorical(self.labels, num_classes).astype("uint8")
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
        range = (rmax - rmin)
        return np.array(range * np.random.rand(size) - range/2.0,
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
        if isinstance(split, float):
            if not 0 <= split <= 1:
                raise Exception("split is not in the range 0-1: %s" % split)
            self.split = int(self.num_inputs * split)
        elif isinstance(split, int):
            if not 0 <= split <= self.num_inputs:
                raise Exception("split out of range: %d" % split)
            self.split = split
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
        if not (isinstance(batch_size, int) or batch_size is None):
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

    def _get_input(self, i):
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
                
    def _get_target(self, i):
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

    def _get_train_input(self, i):
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
                
    def _get_train_target(self, i):
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

    def _get_test_input(self, i):
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
                
    def _get_test_target(self, i):
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
                return list(self.model.predict(np.array([input]), batch_size=batch_size)[0])
            else:
                return list(self.model.predict(np.array([input]))[0])
        else:
            inputs = [np.array(x, "float32") for x in input]
            if batch_size is not None:
                return [[list(y) for y in x][0] for x in self.model.predict(inputs, batch_size=batch_size)]
            else:
                return [[list(y) for y in x][0] for x in self.model.predict(inputs)]

    def propagate_to(self, layer_name, input, batch_size=None):
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        if self.num_input_layers == 1:
            if batch_size is not None:
                return list(self[layer_name]._output(np.array([input]), batch_size=batch_size)[0])
            else:
                return list(self[layer_name]._output(np.array([input]))[0])
        else:
            inputs = [np.array(x, "float32") for x in input]
            # get just inputs for this layer, in order:
            inputs = [inputs[self.input_layer_order.index(name)] for name in self[layer_name].input_names]
            if len(inputs) == 1:
                inputs = inputs[0]
            if batch_size is not None:
                return list(self[layer_name]._output(inputs, batch_size=batch_size)[0])
            else:
                return list(self[layer_name]._output(inputs)[0])

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
                if "activation" in layer.params:
                    del layer.params["activation"]
                if layer.shape:
                    layer.k = Input(shape=layer.shape, **layer.params)
                else:
                    layer.k = Input(**layer.params)
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
                    f = Concatenate()([incoming.k for incoming in layer.incoming_connections])
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
            return self[layer_names[0]].k

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
        
    def _scale_output_for_image(self, activation, vector):
        """
        Given an activation name (or something else) and an output
        vector, scale the vector.
        """
        # ('relu', 'sigmoid', 'linear', 'softmax', 'tanh')
        if activation in ["tanh"]:
            return rescale_numpy_array(vector, (-1,+1), (0,255), 'uint8')
        elif activation in ["sigmoid", "softmax"]:
            return rescale_numpy_array(vector, (0,+1), (0,255), 'uint8')
        elif activation in ["relu"]:
            return rescale_numpy_array(vector, (0,vector.max()), (0,255), 'uint8')
        else: # activation in ["linear"] or otherwise
            return rescale_numpy_array(vector, (-1,+1), (0,255), 'uint8')
        
    def _make_image(self, layer_name, vector, size=25, transpose=False, colormap="hot"):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget.
        """
        import matplotlib as mpl
        import PIL
        layer = self[layer_name]
        activation = layer.activation
        if layer.vshape != layer.shape:
            vector = vector.reshape(layer.vshape)
        vector = self._scale_output_for_image(activation, vector)
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        image = PIL.Image.fromarray(vector, 'P')
        width = vector.shape[0] * size # in, pixels
        # Fixed size:
        hsize = vector.shape[1] * size # in, pixels
        #wpercent = (width/float(image.size[0]))
        #hsize = int((float(image.size[1])*float(wpercent)))
        #img_src = image.resize((width, hsize), PIL.Image.ANTIALIAS)
        img_src = image.resize((hsize, width)) # , PIL.Image.ANTIALIAS)
        # colorize:
        #cm_hot = mpl.cm.get_cmap(colormap)
        #im = np.array(img_src)
        #im = cm_hot(im)
        #im = np.uint8(im * 255)
        #im = PIL.Image.fromarray(im)
        return img_src

    def _image_to_uri(self, img_src):
        # Convert to binary data:
        b = io.BytesIO()
        img_src.save(b, format='gif')
        data = b.getvalue()
        data = base64.b64encode(data)
        if not isinstance(data, str):
            data = data.decode("latin1")
        return "data:image/gif;base64,%s" % data

    def propagate_input_to(self, index, layer_name):
        inputs = self._get_input(index)
        outputs = self.propagate_to(layer_name, inputs)
        outputs = np.array(outputs)
        image = self._make_image(layer_name, outputs)
        return image

    def update_svg(self, index):
        ## FIXME: need better method:
        from IPython.display import Javascript, display
        text = ""
        for name in [layer.name for layer in self.layers]:
            image = self.propagate_input_to(index, name)
            data_uri = self._image_to_uri(image)
            text +=  """
var image = document.getElementById("{netname}_{name}");
image.setAttributeNS(null, "href", "{data_uri}");
""".format(**{"netname": self.name, "name": name, "data_uri": data_uri})
        display(Javascript(text))
    
    def build_svg(self):
        ordering = list(reversed(self._get_level_ordering())) # list of names per level, input to output
        image_svg = """<image id="{netname}_{{name}}" x="{{x}}" y="{{y}}" height="{{height}}" width="{{width}}" xlink:href="{{image}}" />""".format(**{"netname": self.name})
        arrow_svg = """<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="blue" stroke-width="1" marker-end="url(#arrow)" />"""
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
                image = self.propagate_input_to(0, layer_name) # use first input
                (width, height) = image.size
                max_dim = max(width, height)
                if max_dim > 200:
                    image = image.resize((int(width/max_dim * 200), int(height/max_dim * 200)))
                    (width, height) = image.size
                images[layer_name] = image
                total_width += width + 75 # space between
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
                                           "height": height}
                for out in self[layer_name].outgoing_connections:
                    # draw an arrow to these
                    x = positioning[out.name]["x"] + positioning[out.name]["width"]/2
                    y = positioning[out.name]["y"] + positioning[out.name]["height"]
                    svg += arrow_svg.format(**{"x1":cwidth + width/2,
                                               "y1":cheight,
                                               "x2":x,
                                               "y2":y})
                svg += image_svg.format(**positioning[layer_name])
                svg += label_svg.format(**{"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] + 5,
                                           "y": positioning[layer_name]["y"] + positioning[layer_name]["height"]/2 + 2,
                                           "label": layer_name,
                                           "size": 10})
                cwidth += width + spacing # spacing between
                max_height = max(max_height, height)
            cheight += max_height + 50 # 50 for arrows
            top = False
        return ("""
        <svg id='{netname}' xmlns='http://www.w3.org/2000/svg' width="{width}" height="{height}">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="blue" />
        </marker>
    </defs>
""".format(**{"width": max_width, "height": total_height, "netname": self.name}) + svg + """</svg>""")
        
    def make_svg(self, data_uri):
        svg = """
<svg id='mysvg' xmlns='http://www.w3.org/2000/svg' width="400" height="400">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="black" />
        </marker>
    </defs>
    <image id="output1" x="0" y="0" height="25" width="25" 
        xlink:href="{output1}" />
    <rect x="0" y="0" width="25" height="25" stroke="purple" stroke-width="2px" fill="none" />
    <image id="output2" x="200" y="0" height="25" width="25" 
        xlink:href="{output2}" />
    <rect x="200" y="0" width="25" height="25" stroke="purple" stroke-width="2px" fill="none" />
    <image id="shared-hidden" x="100" y="100" height="25" width="50" 
        xlink:href="{shared-hidden}" />
    <rect x="100" y="100" width="50" height="25" stroke="purple" stroke-width="2px" fill="none" />
    <image id="hidden1" x="0" y="200" height="25" width="50" 
        xlink:href="{hidden1}" />
    <rect x="0" y="200" width="50" height="25" stroke="purple" stroke-width="2px" fill="none" />
    <image id="hidden2" x="200" y="200" height="25" width="50" 
        xlink:href="{hidden2}" />
    <rect x="200" y="200" width="50" height="25" stroke="purple" stroke-width="2px" fill="none" />
    <image id="input1" x="0" y="300" height="25" width="25" 
        xlink:href="{input1}" />
    <rect x="0" y="300" width="25" height="25" stroke="purple" stroke-width="2px" fill="none" />
    <image id="input2" x="200" y="300" height="25" width="25" 
        xlink:href="{input2}" />
    <rect x="200" y="300" width="25" height="25" stroke="purple" stroke-width="2px" fill="none" />
    <line x1="100" y1="100" x2="50" y2="75" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
    <text x="100" y="35" font-family="Verdana" font-size="20">
      input1
    </text>
</svg>
""".format(**data_uri)
        return svg

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

#------------------------------------------------------------------------
# utility functions

def autoname(index, sizes):
    if index == 0:
        n = "input"
    elif index == sizes - 1:
        n = "output"
    elif sizes == 3:
        n = "hidden"
    else:
        n = "hidden%d" % i
    return n

def valid_shape(x):
    return isinstance(x, int) and x > 0 \
        or isinstance(x, tuple) and len(x) > 1 and all([isinstance(n, int) and n > 0 for n in x])

def valid_vshape(x):
    # vshape must be a single int or a 2-dimensional tuple
    return valid_shape(x) and (isinstance(x, int) or len(x) == 2)

def rescale_numpy_array(a, old_range, new_range, new_dtype):
    assert isinstance(old_range, tuple) and isinstance(new_range, tuple)
    old_min, old_max = old_range
    if a.min() < old_min or a.max() > old_max:
        raise Exception('array values are outside range %s' % (old_range,))
    new_min, new_max = new_range
    old_delta = old_max - old_min
    new_delta = new_max - new_min
    if old_delta == 0:
        return ((a - old_min) + (new_min + new_max)/2).astype(new_dtype)
    else:
        return (new_min + (a - old_min)*new_delta/old_delta).astype(new_dtype)

#------------------------------------------------------------------------

class Layer():

    ACTIVATION_FUNCTIONS = ('relu', 'sigmoid', 'linear', 'softmax', 'tanh')
    CLASS = Dense
            
    def __repr__(self):
        return "<Layer name='%s', shape=%s, act='%s'>" % (
            self.name, self.shape, self.activation)

    def __init__(self, name, shape, **params):
        if not (isinstance(name, str) and len(name) > 0):
            raise Exception('bad layer name: %s' % (name,))
        self.name = name
        self.params = params
        if shape is None: # Special case: must be passed in
            self.shape = None
            self.size = None
        else:
            if not valid_shape(shape):
                raise Exception('bad shape: %s' % (shape,))
            # set layer topology (shape) and number of units (size)
            if isinstance(shape, int):
                # linear layer
                self.shape = (shape,)
                self.size = shape
            else:
                # multi-dimensional layer
                self.shape = shape
                self.size = reduce(operator.mul, shape)
        params["name"] = name

        # set visual shape for display purposes
        if 'vshape' in params:
            vs = params['vshape']
            del params["vshape"] # drop those that are not Keras parameters
            if not valid_vshape(vs):
                raise Exception('bad vshape: %s' % (vs,))
            elif isinstance(vs, int) and vs != self.size \
                 or isinstance(vs, tuple) and vs[0]*vs[1] != self.size:
                raise Exception('vshape incompatible with layer of size %d' % (self.size,))
            else:
                self.vshape = vs
        elif self.shape and len(self.shape) > 2:
            self.vshape = (self.size,)
        else:
            self.vshape = self.shape
        
        if 'activation' in params:
            act = params['activation']
            if act == None: act = 'linear'
            if not (callable(act) or act in Layer.ACTIVATION_FUNCTIONS):
                raise Exception('unknown activation function: %s' % (act,))
            self.activation = act
        else:
            self.activation = 'linear'
            params["activation"] = "linear"

        if 'dropout' in params:
            dropout = params['dropout']
            del params["dropout"] # we handle dropout layers
            if dropout == None: dropout = 0
            if not (isinstance(dropout, (int, float)) and 0 <= dropout <= 1):
                raise Exception('bad dropout rate: %s' % (dropout,))
            self.dropout = dropout
        else:
            self.dropout = 0

        self.incoming_connections = []
        self.outgoing_connections = []

    def _output(self, input, batch_size=None):
        if batch_size is not None:
            output = self.model.predict(input, batch_size=batch_size)
        else:
            output = self.model.predict(input)
        return output

    def summary(self):
        print("Name: %s (%s) Shape: %s Size: %d VShape: %s Activation function: %s Dropout: %s" %
              (self.name, self.kind(), self.shape, self.size, self.vshape, self.activation, self.dropout))
        if len(self.outgoing_connections) > 0:
            print("Connected to:", [layer.name for layer in self.outgoing_connections])

    def kind(self):
        if len(self.incoming_connections) == 0 and len(self.outgoing_connections) == 0:
            return 'unconnected'
        elif len(self.incoming_connections) > 0 and len(self.outgoing_connections) > 0:
            return 'hidden'
        elif len(self.incoming_connections) > 0:
            return 'output'
        else:
            return 'input'
        
    def make_keras_functions(self):
        if self.kind() == 'input':
            raise Exception("Input layers are made automatically")
        k = self.CLASS(self.size, **self.params)
        if self.dropout > 0:
            return [k, Dropout(self.dropout)]
        else:
            return [k]


class LSTMLayer(Layer):
    CLASS = keras.layers.LSTM
        
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
