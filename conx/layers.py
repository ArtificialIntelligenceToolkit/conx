# conx - a neural network library
#
# Copyright (c) Douglas S. Blank <dblank@cs.brynmawr.edu>
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
The conx.layers module contains the code for all of the layers.
In addition, it dynamically loads all of the Keras layers and
wraps them as a conx layer.
"""

#------------------------------------------------------------------------

import numbers
import operator
from functools import reduce
import sys
import inspect
import html
import re
import os

import numpy as np
import keras
import keras.backend as K
from keras.optimizers import (SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam,
                              TFOptimizer)

from .utils import *

#------------------------------------------------------------------------
ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'
#------------------------------------------------------------------------

pypandoc = None
if ON_RTD:  ## takes too long to load, unless really needed
    try:
        import pypandoc
    except:
        pass # won't turn Keras comments into rft for documentation


#------------------------------------------------------------------------

class _BaseLayer():
    """
    The base class for all conx layers.

    See :any:`Layer` for more details.
    """
    ACTIVATION_FUNCTIONS = ('relu', 'sigmoid', 'linear', 'softmax', 'tanh',
                            'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid')
    CLASS = None

    def __init__(self, name, *args, **params):
        if not (isinstance(name, str) and len(name) > 0):
            raise Exception('bad layer name: %s' % (name,))
        self.name = name
        self.params = params
        self.args = args
        params["name"] = name
        self.shape = None
        self.vshape = None
        self.image_maxdim = None
        self.image_pixels_per_unit = None
        self.visible = True
        self.colormap = None
        self.minmax = (-1, 1)
        self.model = None
        self.decode_model = None
        self.input_names = []
        self.feature = 0
        self.keras_layer = None
        # used to determine image ranges:
        self.activation = params.get("activation", None) # make a copy, if one, and str
        if not isinstance(self.activation, str):
            self.activation = None
        # set visual shape for display purposes
        if 'vshape' in params:
            vs = params['vshape']
            del params["vshape"] # drop those that are not Keras parameters
            if not valid_vshape(vs):
                raise Exception('bad vshape: %s' % (vs,))
            else:
                self.vshape = vs

        if 'image_maxdim' in params:
            imd = params['image_maxdim']
            del params["image_maxdim"] # drop those that are not Keras parameters
            if not isinstance(imd, numbers.Integral):
                raise Exception('bad image_maxdim: %s' % (imd,))
            else:
                self.image_maxdim = imd

        if 'image_pixels_per_unit' in params:
            imd = params['image_pixels_per_unit']
            del params["image_pixels_per_image"] # drop those that are not Keras parameters
            if not isinstance(imd, numbers.Integral):
                raise Exception('bad image_pixels_per_unit: %s' % (imd,))
            else:
                self.image_pixels_per_unit = imd

        if 'visible' in params:
            visible = params['visible']
            del params["visible"] # drop those that are not Keras parameters
            self.visible = visible

        if 'colormap' in params:
            self.colormap = params['colormap']
            del params["colormap"] # drop those that are not Keras parameters

        if 'minmax' in params:
            self.minmax = params['minmax']
            del params["minmax"] # drop those that are not Keras parameters

        if 'dropout' in params:
            dropout = params['dropout']
            del params["dropout"] # we handle dropout layers
            if dropout == None: dropout = 0
            if not (isinstance(dropout, numbers.Real) and 0 <= dropout <= 1):
                raise Exception('bad dropout rate: %s' % (dropout,))
            self.dropout = dropout
        else:
            self.dropout = 0

        if 'time_distributed' in params:
            time_distributed = params['time_distributed']
            del params["time_distributed"] # we handle time distributed wrappers
            self.time_distributed = time_distributed
        else:
            self.time_distributed = False

        if 'activation' in params: # let's keep a copy of it
            self.activation = params["activation"]
            if not isinstance(self.activation, str):
                self.activation = None

        self.incoming_connections = []
        self.outgoing_connections = []

    def on_connect(self, relation, other_layer):
        """
        relation is "to"/"from" indicating which layer self is.
        """
        pass

    def __repr__(self):
        return "<%s name='%s'>" % (self.CLASS.__name__, self.name)

    def summary(self):
        """
        Print out a representation of the layer.
        """
        print("    Layer name: '%s' (%s)" % (self.name, self.kind()))
        print("        VShape:", self.vshape)
        print("        Dropout:", self.dropout)
        if len(self.outgoing_connections) > 0:
            print("        Connected to:", [layer.name for layer in self.outgoing_connections])

    def kind(self):
        """
        Determines whether a layer is a "input", "hidden", or "output" layer based on
        its connections. If no connections, then it is "unconnected".
        """
        if len(self.incoming_connections) == 0 and len(self.outgoing_connections) == 0:
            return 'unconnected'
        elif len(self.incoming_connections) > 0 and len(self.outgoing_connections) > 0:
            return 'hidden'
        elif len(self.incoming_connections) > 0:
            return 'output'
        else:
            return 'input'

    def make_input_layer_k(self):
        """
        Make an input layer for this type of layer. This allows Layers to have
        special kinds of input layers. Would need to be overrided in subclass.
        """
        return keras.layers.Input(self.shape, *self.args, **self.params)

    def make_input_layer_k_text(self):
        """
        Make an input layer for this type of layer. This allows Layers to have
        special kinds of input layers. Would need to be overrided in subclass.
        """
        ## FIXME: WIP, don't include args, params if empty
        return "keras.layers.Input(%s, *%s, **%s)" % (self.shape, self.args, self.params)

    def make_keras_function(self):
        """
        This makes the Keras function for the functional interface.
        """
        ## This is for all Keras layers:
        return self.CLASS(*self.args, **self.params)

    def make_keras_function_text(self):
        """
        This makes the Keras function for the functional interface.
        """
        ## This is for all Keras layers:
        ## FIXME: WIP, don't include args, params if empty
        return "keras.layers.%s(*%s, **%s)" % (self.CLASS.__name__, self.args, self.params)

    def make_keras_functions(self):
        """
        Make all Keras functions for this layer, including its own,
        dropout, etc.
        """
        from keras.layers import TimeDistributed
        k = self.make_keras_function() # can override
        if self.time_distributed:
            k = TimeDistributed(k, name=self.name)
        if self.dropout > 0:
            return [k, keras.layers.Dropout(self.dropout)]
        else:
            return [k]

    def make_keras_functions_text(self):
        """
        Make all Keras functions for this layer, including its own,
        dropout, etc.
        """
        program = self.make_keras_function_text()
        if self.time_distributed:
            program = "keras.layers.TimeDistributed(%s, name='%s')" % (program, self.name)
        if self.dropout > 0:
            return "[%s, keras.layers.Dropout(self.dropout)]" % program
        else:
            return "[%s]" % program

    # class: _BaseLayer
    def make_image(self, vector, colormap=None, config={}):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget.
        """
        import keras.backend as K
        from matplotlib import cm
        import PIL
        if self.vshape and self.vshape != self.shape:
            vector = vector.reshape(self.vshape)
        if len(vector.shape) > 2:
            ## Drop dimensions of vector:
            s = slice(None, None)
            args = []
            # The data is in the same format as Keras
            # so we can ask Keras what that format is:
            # ASSUMES: that the network that loaded the
            # dataset has the same image_data_format as
            # now:
            if K.image_data_format() == 'channels_last':
                for d in range(len(vector.shape)):
                    if d in [0, 1]:
                        args.append(s) # keep the first two
                    else:
                        args.append(self.feature) # pick which to use
            else: # 'channels_first'
                count = 0
                for d in range(len(vector.shape)):
                    if d in [0]:
                        args.append(self.feature) # pick which to use
                    else:
                        if count < 2:
                            args.append(s)
                            count += 1
            vector = vector[args]
        minmax = config.get("minmax")
        # if minmax is None:
        #     minmax = self.get_minmax(vector)
        vector = self.scale_output_for_image(vector, self.minmax, truncate=True)
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        size = config["pixels_per_unit"]
        new_width = vector.shape[0] * size # in, pixels
        new_height = vector.shape[1] * size # in, pixels
        if colormap is None:
            colormap = get_colormap() if self.colormap is None else self.colormap
        try:
            cm_hot = cm.get_cmap(colormap)
        except:
            cm_hot = cm.get_cmap("RdGy")
        vector = cm_hot(vector)
        vector = np.uint8(vector * 255)
        image = PIL.Image.fromarray(vector)
        image = image.resize((new_height, new_width))
        return image

    def scale_output_for_image(self, vector, minmax, truncate=False):
        """
        Given an activation name (or something else) and an output
        vector, scale the vector.
        """
        return rescale_numpy_array(vector, minmax, (0,255), 'uint8', truncate=truncate)

    def make_dummy_vector(self, default_value=0.0):
        """
        This is in the easy to use human format (list of lists ...)
        """
        ## FIXME: for pictures give a vector
        v = np.ones(self.shape) * default_value
        lo, hi = self.get_minmax(v)
        v *= (lo + hi) / 2.0
        return v.tolist()

    def get_act_minmax(self):
        """
        Get the activation (output) min/max for a layer.
        """
        # ('relu', 'sigmoid', 'linear', 'softmax', 'tanh')
        if self.activation in ["tanh"]:
            return (-1,+1)
        elif self.activation in ["sigmoid", "softmax"]:
            return (0,+1)
        elif self.activation in ["relu"]:
            return (0,+2)
        elif self.activation in ["linear"]:
            return (-2,+2)
        else: # activation in ["linear"] or otherwise
            return (-1,+1)

    def get_minmax(self, vector):
        """
        Get the min/max for an input vector to this
        layer. Attempts to guess based on activation function.
        """
        if self.minmax:
            return self.minmax
        # ('relu', 'sigmoid', 'linear', 'softmax', 'tanh')
        if self.activation in ["tanh"]:
            return (-1,+1)
        elif self.activation in ["sigmoid", "softmax"]:
            return (0,+1)
        elif self.activation in ["relu"]:
            return (0,vector.max())
        else: # activation in ["linear"] or otherwise
            return (-1,+1)

    def tooltip(self):
        """
        String (with newlines) for describing layer."
        """
        kind = self.kind()
        retval = "Layer: %s (%s)" % (self.name, kind)
        if self.shape:
            retval += "\n shape = %s" % (self.shape, )
        if self.dropout:
            retval += "\n dropout = %s" % self.dropout
        if kind == "input":
            retval += "\n Keras class = Input"
        else:
            retval += "\n Keras class = %s" % self.CLASS.__name__
        for key in self.params:
            if key in ["name"]:
                continue
            retval += "\n %s = %s" % (key, self.params[key])
        return html.escape(retval)

class Layer(_BaseLayer):
    """
    The default layer type. Will create either an InputLayer, or DenseLayer,
    depending on its context after :any:`Network.connect`.

    Arguments:
        name: The name of the layer. Must be unique in this network. Should
           not contain special HTML characters.

    Examples:
        >>> layer = Layer("input", 10)
        >>> layer.name
        'input'

        >>> from conx import Network
        >>> net = Network("XOR2")
        >>> net.add(Layer("input", 2))
        >>> net.add(Layer("hidden", 5))
        >>> net.add(Layer("output", 2))
        >>> net.connect()
        >>> net["input"].kind()
        'input'
        >>> net["output"].kind()
        'output'

    Note:
        See also: :any:`Network`, :any:`Network.add`, and :any:`Network.connect`
        for more information. See https://keras.io/ for more information on
        Keras layers.
    """
    CLASS = keras.layers.Dense
    def __init__(self, name: str, shape, **params):
        super().__init__(name, **params)
        if not valid_shape(shape):
            raise Exception('bad shape: %s' % (shape,))
        # set layer topology (shape) and number of units (size)
        if isinstance(shape, numbers.Integral) or shape is None:
            self.shape = (shape,)
            self.size = shape
        else:
            # multi-dimensional layer
            self.shape = shape
            if all([isinstance(n, numbers.Integral) for n in shape]):
                self.size = reduce(operator.mul, shape)
            else:
                self.size = None # can't compute size because some dim are None

        if 'activation' in params:
            act = params['activation']
            if act == None:
                act = 'linear'
            if not (callable(act) or act in Layer.ACTIVATION_FUNCTIONS):
                raise Exception('unknown activation function: %s' % (act,))
            self.activation = act
            if not isinstance(self.activation, str):
                self.activation = None

    def __repr__(self):
        return "<Layer name='%s', shape=%s, act='%s'>" % (
            self.name, self.shape, self.activation)

    def summary(self):
        """
        Print a summary of the dense/input layer.
        """
        super().summary()
        print("        Activation function:", self.activation)
        print("        Dropout percent:", self.dropout)

    def make_keras_function(self):
        """
        For all Keras-based functions. Returns the Keras class.
        """
        return self.CLASS(self.size, **self.params)

    def make_keras_function_text(self):
        """
        For all Keras-based functions. Returns the Keras class.
        """
        return "keras.layers.%s(%s, **%s)" % (self.CLASS.__name__, self.size, self.params)

class ImageLayer(Layer):
    """
    A class for images. WIP.
    """
    def __init__(self, name, dimensions, depth, **params):
        super().__init__(name, dimensions, **params)
        if self.vshape is None:
            self.vshape = self.shape
        self.dimensions = dimensions
        self.depth = depth
        if K.image_data_format() == "channels_last":
            self.shape = tuple(list(self.shape) + [depth])
            self.image_indexes = (0, 1)
        else:
            self.shape = tuple([depth] + list(self.shape))
            self.image_indexes = (1, 2)

    # class: ImageLayer
    def make_image(self, vector, colormap=None, config={}):
        """
        Given an activation name (or function), and an output vector, display
        make and return an image widget. Colormap is ignored.
        """
        ## see K.image_data_format() == 'channels_last': above
        ## We keep the dataset data in the right format.
        import PIL
        v = (vector * 255).astype("uint8")
        if self.depth == 1:
            v = v.squeeze() # get rid of nested lists (len of 1)
        else:
            v = v.reshape(self.dimensions[0],
                          self.dimensions[1],
                          self.depth)
        return PIL.Image.fromarray(v)

class EmbeddingLayer(Layer):
    """
    A class for embeddings. WIP.
    """
    def __init__(self, name, in_size, out_size, **params):
        super().__init__(name, in_size, **params)
        if self.vshape is None:
            self.vshape = self.shape
        self.in_size = in_size
        self.out_size = out_size
        self.sequence_size = None # get filled in on_connect

    def make_keras_function(self):
        from keras.layers import Embedding as KerasEmbedding
        return KerasEmbedding(self.in_size, self.out_size, input_length=self.sequence_size, **self.params)

    def on_connect(self, relation, other_layer):
        """
        relation is "to"/"from" indicating which layer self is.
        """
        if relation == "to":
            ## other_layer must be an Input layer
            self.sequence_size = other_layer.size # get the input_length
            self.shape = (self.sequence_size, self.out_size)
            if self.sequence_size:
                self.size = self.sequence_size * self.out_size
            else:
                self.size = None
            self.vshape = (self.sequence_size, self.out_size)
            other_layer.size = (None,)  # poke in this otherwise invalid size
            other_layer.shape = (self.sequence_size,)  # poke in this shape
            other_layer.params["dtype"] = "int32" # assume ints
            other_layer.make_dummy_vector = lambda v=0.0: np.zeros(self.sequence_size) * v
            other_layer.minmax = (0, self.in_size)

def process_class_docstring(docstring):
    docstring = re.sub(r'\n    # (.*)\n',
                       r'\n    __\1__\n\n',
                       docstring)
    docstring = re.sub(r'    ([^\s\\\(]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)
    docstring = docstring.replace('    ' * 5, '\t\t')
    docstring = docstring.replace('    ' * 3, '\t')
    docstring = docstring.replace('    ', '')
    return docstring

## Dynamically load all of the keras layers, making a conx layer:
## Al of these will have _BaseLayer as their superclass:
keras_module = sys.modules["keras.layers"]
for (name, obj) in inspect.getmembers(keras_module):
    if name in ["Embedding", "Input", "Dense", "TimeDistributed"]: continue
    if type(obj) == type and issubclass(obj, (keras.engine.Layer, )):
        new_name = "%sLayer" % name
        docstring = obj.__doc__
        if pypandoc:
            try:
                docstring_md  = '    **%s**\n\n' % (new_name,)
                docstring_md += obj.__doc__
                docstring = pypandoc.convert(process_class_docstring(docstring_md), "rst", "markdown_github")
            except:
                pass
        locals()[new_name] = type(new_name, (_BaseLayer,),
                                  {"CLASS": obj,
                                   "__doc__": docstring})

DenseLayer = Layer # for consistency
InputLayer = Layer # for consistency
