from __future__ import print_function, division

#------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop, SGD
from keras.utils import to_categorical

from scipy import misc
import glob, random, operator, importlib
from functools import reduce

#------------------------------------------------------------------------

class Network:

    def __init__(self, *layers):
        self.layers = layers
        self.layer_dict = {layer.name:layer for layer in self.layers}

    def __getitem__(self, layer_name):
        if layer_name not in self.layer_dict:
            return None
        else:
            return self.layer_dict[layer_name]

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

    def load_keras_dataset(self, name, verbose=True):
        available_datasets = [x for x in dir(keras.datasets) if '__' not in x]
        if name not in available_datasets:
            raise Exception("unknown keras dataset: %s" % name)
        if verbose:
            print('Loading %s dataset...' % name)
        load_data = importlib.import_module('keras.datasets.' + name).load_data
        (x_train,y_train), (x_test,y_test) = load_data()
        self.inputs = np.concatenate((x_train,x_test))
        self.labels = np.concatenate((y_train,y_test))
        self.dataset_size = self.inputs.shape[0]
        self.inputs_range = (self.inputs.min(), self.inputs.max())
        self.split_dataset(self.dataset_size, verbose=False)
        if verbose:
            print('Loaded %d inputs and labels' % (self.dataset_size,))
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
            if len(self.inputs) != len(self.labels):
                raise Exception("Dataset contains different numbers of inputs and labels")
            if len(self.inputs) == 0:
                raise Exception("Dataset is empty")
            self.dataset_size = self.inputs.shape[0]
            self.inputs_range = (self.inputs.min(), self.inputs.max())
            self.split_dataset(self.dataset_size, verbose=False)
            if verbose:
                print('Loaded %d inputs and labels into network' % self.dataset_size)
                print('Input data shape: %s, range: %s, type: %s' %
                      (self.inputs[0].shape[1:], self.inputs_range, self.inputs.dtype))
        except:
            raise Exception("couldn't load .npz dataset %s" % filename)

    def reshape_inputs(self, new_shape):
        # do some error checking here
        self.inputs = self.inputs.reshape((self.dataset_size,) + new_shape)
        # resplit
        self.split_dataset(self.split, verbose=False)
        print("ok")

    def show_dataset(self):
        if self.dataset_size == 0:
            print("No dataset loaded")
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

    def shuffle_dataset(self, verbose=True):
        if self.dataset_size == 0:
            raise Exception("no dataset loaded")
        indices = np.random.permutation(self.dataset_size)
        self.inputs = self.inputs[indices]
        self.labels = self.labels[indices]
        self.split_dataset(self.split, verbose=False)
        if verbose:
            print('Shuffled all %d inputs and labels' % self.dataset_size)

    def split_dataset(self, split=0.50, verbose=True):
        # possibly need to deal with self.targets
        if self.dataset_size == 0:
            raise Exception("no dataset loaded")
        if isinstance(split, float):
            if not 0 <= split <= 1:
                raise Exception("split is not in the range 0-1: %s" % split)
            self.split = int(self.dataset_size * split)
        elif isinstance(split, int):
            if not 0 <= split <= self.dataset_size:
                raise Exception("split out of range: %d" % split)
            self.split = split
        else:
            raise Exception("invalid split: %s" % split)
        self.train_inputs = self.inputs[:self.split]
        self.test_inputs = self.inputs[self.split:]
        self.train_labels = self.labels[:self.split]
        self.test_labels = self.labels[self.split:]
        if verbose:
            print('Split dataset into %d train inputs, %d test inputs' %
                  (len(self.train_inputs), len(self.test_inputs)))

    def propagate(self, input):
        return list(self.model.predict(np.array([input]))[0])

    def propagate_to(self, layer_name, input):
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        else:
            return self[layer_name]._output(input)

    def compile(self, **kwargs):
        if len(self.layers) == 0:
            raise Exception("network has no layers")
        for layer in self.layers:
            if layer.kind() == 'unconnected':
                raise Exception("'%s' layer is unconnected" % layer.name)
        input_layers = []
        output_layers = []
        for layer in self.layers:
            if layer.kind() == 'input':
                f = Input(shape=layer.shape)
                input_layers.append(f)
                for c in layer.chain():
                    for k in c.make_keras_layers():
                        f = k(f)
                    c.model = Model(inputs=input_layers[-1], outputs=f)
                output_layers.append(f)
        if len(input_layers) == 1:
            input_layers = input_layers[0]
        if len(output_layers) == 1:
            output_layers = output_layers[0]
        self.model = Model(inputs=input_layers, outputs=output_layers)
        self.model.compile(**kwargs)

    def scale_output_for_image(self, activation, vector):
        """
        Given an activation name (or something else) and an output
        vector, scale the vector.
        """
        # ('relu', 'sigmoid', 'linear', 'softmax', 'tanh')
        if activation in ["tanh"]:
            return rescale_numpy_array(vector, (-1,+1), (0,255)).astype('uint8')
        elif activation in ["sigmoid", "softmax"]:
            return rescale_numpy_array(vector, (0,+1), (0,255)).astype('uint8')
        elif activation in ["relu"]:
            return rescale_numpy_array(vector, (0,vector.max()), (0,255)).astype('uint8')
        else: # activation in ["linear"] or otherwise
            return rescale_numpy_array(vector, (-1,+1), (0,255)).astype('uint8')
        
    def make_image_widget(self, activation, vector, size=25, transpose=False):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget.
        """
        import ipywidgets
        import PIL
        import io
        vector = self.scale_output_for_image(activation, vector)
        # FIXME: only needed if missing 1D second dim:
        vector = vector.reshape((1, vector.shape[0]))
        image = PIL.Image.fromarray(vector, 'P')
        width = vector.shape[0] * size # in, pixels
        wpercent = (width/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        im = image.resize((width,hsize), PIL.Image.ANTIALIAS)
        b = io.BytesIO()
        im.save(b, format='png')
        data = b.getvalue()
        layout = ipywidgets.Layout(border='2px solid blue')
        widget = ipywidgets.Image(value=data, format='png', layout=layout)
        return widget

    def visualize(self, inputs):
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
                    accordion = ipywidgets.Accordion((self.make_image_widget(lay.activation, output),))
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
<<<<<<< Updated upstream
        return ((a - old_min) + (new_min + new_max)/2).astype(new_dtype)
    else:
        return (new_min + (a - old_min)*new_delta/old_delta).astype(new_dtype)
=======
        old_delta = 2 * new_delta
    return new_min + (a - old_min)*new_delta/old_delta
>>>>>>> Stashed changes

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
        
    def make_keras_layers(self):
        if self.kind() == 'input':
            klayer = []
        else:
            klayer = [Dense(self.size, activation=self.activation)]
        if self.dropout > 0:
            return klayer + [Dropout(self.dropout)]
        else:
            return klayer

    def chain(self):
        if len(self.outgoing_connections) == 0:
            return [self]
        else:
            return [self] + self.outgoing_connections[0].chain()


    # def chain(self):
    #     if len(self.outgoing_connections) == 0:
    #         return [self]
    #     else:
    #         results = [self]
    #         for layer in self.outgoing_connections:
    #             chain = layer.chain()
    #             print(chain)
    #             if len(chain) == 1:
    #                 results.extend(chain)
    #             else:
    #                 results.append(chain)
    #         return results

