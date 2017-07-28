from __future__ import print_function, division

#------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop, SGD
from keras.utils import to_categorical

from scipy import misc
import glob, random, operator

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
            
    def propagate_to(self, layer_name, input):
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        else:
            return self[layer_name]._output(input)

    def propagate(self, input):
        return list(self.model.predict(np.array([input]))[0])

    def compile(self, **kwargs):
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

#------------------------------------------------------------------------

def valid_shape(x):
    return isinstance(x, int) and x > 0 \
        or isinstance(x, tuple) and len(x) > 1 and all([isinstance(n, int) and n > 0 for n in x])

def valid_vshape(x):
    # vshape must be a single int or a 2-dimensional tuple
    return valid_shape(x) and (isinstance(x, int) or len(x) == 2)

#------------------------------------------------------------------------

class Layer:

    ACTIVATION_FUNCTIONS = ('relu', 'sigmoid', 'linear', 'softmax')
            
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

#------------------------------------------------------------------------

# net = Network(
#     Layer("input1", shape=64, vshape=(8,8)),
#     Layer("input2", shape=(2,2,2), activation="relu"),
#     Layer("hidden", shape=16, activation="relu", dropout=0.5),
#     Layer("output1", shape=1, activation="sigmoid"),
#     Layer("output2", shape=1, activation="sigmoid")
# )

net = Network(
    Layer("input1", shape=2),
    Layer("hidden", shape=2, activation="sigmoid"),
    Layer("output1", shape=1, activation="sigmoid")
)

net.connect("input1", "hidden")
#net.connect("input2", "hidden")
net.connect("hidden", "output1")
#net.connect("hidden", "output2")

net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9),
            metrics=['accuracy'])

XOR_inputs = np.array([[0,0], [0,1], [1,0], [1,1]], 'float32')
XOR_targets = np.array([[0], [1], [1], [0]], 'float32')


