from __future__ import print_function, division

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
            print("\nStoppping at end of epoch... (^C again to quit now)...")
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

class Network:

    def __init__(self, *layers):
        layer_names = [layer.name for layer in layers]
        for name in layer_names:
            if layer_names.count(name) > 1:
                raise Exception("duplicate layer name '%s'" % name)
        self.layers = list(layers)
        self.layer_dict = {layer.name:layer for layer in layers}
        self.inputs = None
        self.labels = None
        self.targets = None
        self.epoch_count = 0
        self.acc_history = []
        self.loss_history = []
        self.val_acc_history = []
        self.input_layer_order = []
        self.output_layer_order = []

    def __getitem__(self, layer_name):
        if layer_name not in self.layer_dict:
            return None
        else:
            return self.layer_dict[layer_name]

    def add(self, layer):
        if layer.name in self.layer_dict:
            raise Exception("duplicate layer name '%s'" % layer.name)
        self.layers.append(layer)
        self.layer_dict[layer.name] = layer

    def connect(self, from_layer_name, to_layer_name):
        if from_layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % from_layer_name)
        if to_layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % to_layer_name)
        from_layer = self.layer_dict[from_layer_name]
        to_layer = self.layer_dict[to_layer_name]
        from_layer.outgoing_connections.append(to_layer)
        to_layer.incoming_connections.append(from_layer)

    def show(self):
        for layer in self.layers:
            layer.show()

    def set_dataset(self, pairs, verbose=True):
        ## FIXME: use ordered_inputs, or check shape:
        input_layers = [layer for layer in self.layers if layer.kind() == "input"]
        if len(input_layers) == 1:
            self.inputs = np.array([x for (x, y) in pairs]).astype('float32')
            self.targets = np.array([y for (x, y) in pairs]).astype('float32')
            self.inputs_range = (self.inputs.min(), self.inputs.max())
            self.targets_range = (self.targets.min(), self.targets.max())
            self.num_inputs = self.inputs.shape[0]
            if verbose:
                print('Set %d inputs and targets' % (self.num_inputs,))
                print('Input data shape: %s, range: %s, type: %s' %
                      (self.inputs.shape[1:], self.inputs_range, self.inputs.dtype))
                print('Target data shape: %s, range: %s, type: %s' %
                      (self.targets.shape[1:], self.targets_range, self.targets.dtype))
        else:
            self.inputs = np.array([[np.array([l]).astype('float32') for l in x] for (x, y) in pairs])
            self.targets = np.array([[np.array([l]).astype('float32') for l in y] for (x, y) in pairs])
            self.inputs_range = (self.inputs.min(), self.inputs.max())
            self.targets_range = (self.targets.min(), self.targets.max())
            self.num_inputs = self.inputs.shape[0]
            if verbose:
                print('Set %d inputs and targets' % (self.num_inputs,))
                print('Input data shapes: %s, range: %s, types: %s' %
                      ([x.shape[1:] for x in self.inputs[0]],
                       self.inputs_range,
                       [x.dtype for x in self.inputs[0]]))
                print('Target data shapes: %s, range: %s, types: %s' %
                      ([x.shape[1:] for x in self.targets[0]],
                       self.targets_range,
                       [x.dtype for x in self.targets[0]]))
        self.labels = None
        self.split_dataset(self.num_inputs, verbose=False)

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
        self.num_inputs = self.inputs.shape[0]
        self.inputs_range = (self.inputs.min(), self.inputs.max())
        self.targets_range = (0, 0)
        self.split_dataset(self.num_inputs, verbose=False)
        if verbose:
            print('Loaded %d inputs and labels' % (self.num_inputs,))
            print('Input data shape: %s, range: %s, type: %s' %
                  (self.inputs.shape[1:], self.inputs_range, self.inputs.dtype))

    def load_npz_dataset(self, filename, verbose=True):
        """loads a dataset from an .npz file and returns data, labels"""
        if filename[-4:] != '.npz':
            raise Exception("filename must end in .npz")
        try:
            f = np.load(filename)
            self.inputs = f['data']
            self.labels = f['labels']
            self.targets = None
            if len(self.inputs) != len(self.labels):
                raise Exception("Dataset contains different numbers of inputs and labels")
            if len(self.inputs) == 0:
                raise Exception("Dataset is empty")
            self.num_inputs = self.inputs.shape[0]
            self.inputs_range = (self.inputs.min(), self.inputs.max())
            self.targets_range = (0, 0)
            self.split_dataset(self.num_inputs, verbose=False)
            if verbose:
                print('Loaded %d inputs and labels into network' % self.num_inputs)
                print('Input data shape: %s, range: %s, type: %s' %
                      (self.inputs[0].shape[1:], self.inputs_range, self.inputs.dtype))
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
            
    def set_targets(self, num_classes):
        if self.num_inputs == 0:
            raise Exception("no dataset loaded")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise Exception("number of classes must be a positive integer")
        self.targets = to_categorical(self.labels, num_classes).astype('uint8')
        self.train_targets = self.targets[:self.split]
        self.test_targets = self.targets[self.split:]
        print('Generated %d target vectors from labels' % self.num_inputs)

    def show_dataset(self):
        if self.num_inputs == 0:
            print("no dataset loaded")
            return
        print('%d train inputs, %d test inputs' %
              (len(self.train_inputs), len(self.test_inputs)))
        print('Input data shape: %s, range: %s, type: %s' %
              (self.inputs.shape[1:], self.inputs_range, self.inputs.dtype))

    def rescale_inputs(self, old_range, new_range, new_dtype):
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

    def make_weights(self, shape):
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
        self.val_acc_history = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            new_weights = []
            for weight in weights:
                new_weights.append(self.make_weights(weight.shape))
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
        self.train_inputs = self.inputs[:self.split]
        self.test_inputs = self.inputs[self.split:]
        if self.labels is not None:
            self.train_labels = self.labels[:self.split]
            self.test_labels = self.labels[self.split:]
        if self.targets is not None:
            self.train_targets = self.targets[:self.split]
            self.test_targets = self.targets[self.split:]
        if verbose:
            print('Split dataset into %d train inputs, %d test inputs' %
                  (len(self.train_inputs), len(self.test_inputs)))

    def test(self, dataset=None):
        if dataset is None:
            if self.split == self.num_inputs:
                dataset = self.train_inputs
            else:
                dataset = self.test_inputs
        print("Testing...")
        outputs = self.model.predict(dataset)
        for output in outputs:
            print(output)
    
    def train(self, epochs=1, accuracy=None, batch_size=None,
              report_rate=1, tolerance=0.1):
        if batch_size is None:
            batch_size = self.train_inputs.shape[0]
        if not isinstance(batch_size, int):
            raise Exception("bad batch size: %s" % (batch_size,))
        if self.split == self.num_inputs:
            validation_inputs = self.train_inputs
            validation_targets = self.train_targets
        else:
            validation_inputs = self.test_inputs
            validation_targets = self.test_targets
        with InterruptHandler() as handler:
            for e in range(1, epochs+1):
                result = self.model.fit(self.train_inputs, self.train_targets,
                                        validation_data=(validation_inputs, validation_targets),
                                        batch_size=batch_size,
                                        epochs=1,
                                        verbose=0)
                outputs = self.model.predict(validation_inputs)
                correct = [all(x) for x in map(lambda v: v <= tolerance,
                                               np.abs(outputs - validation_targets))].count(True)
                self.epoch_count += 1
                acc = result.history['acc'][0]
                self.acc_history.append(acc)
                loss = result.history['loss'][0]
                self.loss_history.append(loss)
                val_acc = correct/len(validation_targets)
                self.val_acc_history.append(val_acc)
                if self.epoch_count % report_rate == 0:
                    print("Epoch #%5d | loss %7.5f | acc %7.5f | vacc %7.5f" %
                          (self.epoch_count, loss, acc, val_acc))
                if accuracy is not None and val_acc >= accuracy or handler.interrupted:
                    break
            if handler.interrupted:
                print("=" * 72)
                print("Epoch #%5d | loss %7.5f | acc %7.5f | vacc %7.5f" %
                      (self.epoch_count, loss, acc, val_acc))
                raise KeyboardInterrupt
        print("=" * 72)
        print("Epoch #%5d | loss %7.5f | acc %7.5f | vacc %7.5f" %
              (self.epoch_count, loss, acc, val_acc))

        # # evaluate the model
        # print('Evaluating performance...')
        # loss, accuracy = self.model.evaluate(self.test_inputs, self.test_targets, verbose=0)
        # print('Test loss:', loss)
        # print('Test accuracy:', accuracy)
        # #print('Most recent weights saved in model.weights')
        # #self.model.save_weights('model.weights')

    def propagate(self, input):
        # FIXME: assumes one input: net.model.predict([np.array([[0]]), np.array([[0]])])
        # take all inputs, but only use those needed
        return list(self.model.predict(np.array([input]))[0])

    def propagate_to(self, layer_name, input):
        # FIXME: assumes one input: net.model.predict([np.array([[0]]), np.array([[0]])])
        # take all inputs, but only use those needed
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        else:
            return self[layer_name]._output(input)

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
                layer.k = Input(shape=layer.shape)
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
                input_ks = self.get_input_ks_in_order(layer.input_names)
                layer.model = Model(inputs=input_ks, outputs=layer.k)
        output_k_layers = self.get_ordered_output_layers()
        input_k_layers = self.get_ordered_input_layers()
        self.model = Model(inputs=input_k_layers, outputs=output_k_layers)
        kwargs['metrics'] = ['accuracy']
        self.model.compile(**kwargs)

    def get_input_ks_in_order(self, layer_names):
        if self.input_layer_order:
            result = []
            for name in self.input_layer_order:
                if name in layer_names:
                    result.append(self[name].k)
            return result
        else:
            # the one input name:
            return self[layer_names[0]].k

    def get_ordered_output_layers(self):
        if self.output_layer_order:
            layers = []
            for layer_name in self.output_layer_order:
                layers.append(self[layer_name].k)
        else:
            layers = [layer.k for layer in self.layers if layer.kind() == "output"][0]
        return layers

    def get_ordered_input_layers(self):
        if self.input_layer_order:
            layers = []
            for layer_name in self.input_layer_order:
                layers.append(self[layer_name].k)
        else:
            layers = [layer.k for layer in self.layers if layer.kind() == "input"][0]
        return layers
        
    def scale_output_for_image(self, activation, vector):
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
        
    def make_image_widget(self, layer, vector, size=25, transpose=False, colormap="hot"):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget.
        """
        import ipywidgets
        import matplotlib as mpl
        import PIL
        activation = layer.activation
        if layer.vshape != layer.shape:
            vector = vector.reshape(layer.vshape)
        vector = self.scale_output_for_image(activation, vector)
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        image = PIL.Image.fromarray(vector, 'P')
        width = vector.shape[0] * size # in, pixels
        # Fixed size:
        hsize = vector.shape[1] * size # in, pixels
        #wpercent = (width/float(image.size[0]))
        #hsize = int((float(image.size[1])*float(wpercent)))
        #img_src = image.resize((width, hsize), PIL.Image.ANTIALIAS)
        img_src = image.resize((hsize, width), PIL.Image.ANTIALIAS)
        # colorize:
        cm_hot = mpl.cm.get_cmap(colormap)
        im = np.array(img_src)
        im = cm_hot(im)
        im = np.uint8(im * 255)
        im = PIL.Image.fromarray(im)
        # Convert to png binary data:
        b = io.BytesIO()
        im.save(b, format='png')
        data = b.getvalue()
        layout = ipywidgets.Layout(border='2px solid blue')
        widget = ipywidgets.Image(value=data, format='png', layout=layout)
        return widget

    def visualize(self, inputs, colormap="hot"):
        import ipywidgets
        from IPython.display import display
        self.widgets = {}
        self.index = 0
        for layer in self.layers:
            if layer.kind() == 'input':
                for lay in reversed(layer.chain()):
                    if hasattr(lay, "model"):
                        output = np.array(lay._output(inputs))
                    else:
                        continue
                    accordion = ipywidgets.Accordion((self.make_image_widget(lay, output, colormap=colormap),))
                    accordion.set_title(0, lay.name)
                    display(accordion)
        display(ipywidgets.Button(description="Next"))
        display(ipywidgets.Button(description="Previous"))

#------------------------------------------------------------------------
# utility functions

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

class Layer:

    ACTIVATION_FUNCTIONS = ('relu', 'sigmoid', 'linear', 'softmax', 'tanh')
            
    def __repr__(self):
        return self.name

    def __init__(self, name, shape, **params):
        if not (isinstance(name, str) and len(name) > 0):
            raise Exception('bad layer name: %s' % (name,))
        self.name = name
        self.params = params
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

        # set visual shape for display purposes
        if 'vshape' in params:
            vs = params['vshape']
            if not valid_vshape(vs):
                raise Exception('bad vshape: %s' % (vs,))
            elif isinstance(vs, int) and vs != self.size \
                 or isinstance(vs, tuple) and vs[0]*vs[1] != self.size:
                raise Exception('vshape incompatible with layer of size %d' % (self.size,))
            else:
                self.vshape = vs
        elif len(self.shape) > 2:
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

        if 'dropout' in params:
            dropout = params['dropout']
            if dropout == None: dropout = 0
            if not (isinstance(dropout, (int, float)) and 0 <= dropout <= 1):
                raise Exception('bad dropout rate: %s' % (dropout,))
            self.dropout = dropout
        else:
            self.dropout = 0

        self.incoming_connections = []
        self.outgoing_connections = []

    def _output(self, input):
        output = list(self.model.predict(np.array([input]))[0])
        return output

    def show(self):
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
            raise Exception("")
        k = Dense(self.size, activation=self.activation, name=self.name)
        if self.dropout > 0:
            return [k, Dropout(self.dropout)]
        else:
            return [k]

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
