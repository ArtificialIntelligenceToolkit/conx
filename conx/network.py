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

import collections
import operator
from functools import reduce
import signal
import string
import numbers
import random
import pickle
import base64
import json
import html
import copy
import sys
import io
import os
import re
from typing import Any

import PIL
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import Callback, History
import keras.backend as K

from .utils import *
from .layers import Layer
from .dataset import Dataset

try:
    from IPython import get_ipython
except:
    get_ipython = lambda: None

#------------------------------------------------------------------------

class ReportCallback(Callback):
    def __init__(self, network, verbose, report_rate, mpl_backend, record):
        # mpl_backend is matplotlib backend
        super().__init__()
        self.network = network
        self.verbose = verbose
        self.report_rate = report_rate
        self.mpl_backend = mpl_backend
        self.in_console = self.network.in_console(mpl_backend)
        self.record = record

    def on_epoch_end(self, epoch, logs=None):
        self.network.history.append(logs)
        self.network.epoch_count += 1
        if (self.verbose > 0 and
            self.in_console and
            (epoch+1) % self.report_rate == 0):
            self.network.report_epoch(self.network.epoch_count, logs)
        if self.record != 0 and (epoch+1) % self.record == 0:
            self.network.weight_history[self.network.epoch_count] = self.network.get_weights()

class PlotCallback(Callback):
    def __init__(self, network, report_rate, mpl_backend):
        # mpl_backend te matplotlib backend string code
        #
        super().__init__()
        self.network = network
        self.report_rate = report_rate
        self.mpl_backend = mpl_backend
        self.in_console = self.network.in_console(mpl_backend)
        self.figure = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch == -1:
            # training loop finished, so make a final update to plot
            # in case the number of loop cycles wasn't a multiple of
            # report_rate
            self.network.plot_results(self)
            if not self.in_console:
                plt.close(self.figure[0])
        elif (epoch+1) % self.report_rate == 0:
            self.network.plot_results(self)

class FunctionCallback(Callback):
    """
        'on_batch_begin',
        'on_batch_end',
        'on_epoch_begin',
        'on_epoch_end',
        'on_train_begin',
        'on_train_end',
    """
    def __init__(self, network, on_method, function):
        super().__init__()
        self.network = network
        self.on_method = on_method
        self.function = function

    def on_batch_begin(self, batch, logs=None):
        if self.on_method == "on_batch_begin":
            self.function(self.network, batch, logs)

    def on_batch_end(self, batch, logs=None):
        if self.on_method == "on_batch_end":
            self.function(self.network, batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        if self.on_method == "on_epoch_begin":
            self.function(self.network, epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.on_method == "on_epoch_end":
            self.function(self.network, epoch, logs)

    def on_train_begin(self, logs=None):
        if self.on_method == "on_train_begin":
            self.function(self.network, logs)

    def on_train_end(self, logs=None):
        if self.on_method == "on_train_end":
            self.function(self.network, logs)

class StoppingCriteria(Callback):
    def __init__(self, item, op, value, use_validation_to_stop):
        super().__init__()
        self.item = item
        self.op = op
        self.value = value
        self.use_validation_to_stop = use_validation_to_stop

    def on_epoch_end(self, epoch, logs=None):
        key = ("val_" + self.item) if self.use_validation_to_stop else self.item
        if key in logs: # we get what we need directly:
            if self.compare(logs[key], self.op, self.value):
                self.model.stop_training = True
        else:
            ## ok, then let's sum/average anything that matches
            total = 0
            count = 0
            for item in logs:
                if self.use_validation_to_stop:
                    if item.startswith("val_") and item.endswith("_" + self.item):
                        count += 1
                        total += logs[item]
                else:
                    if item.endswith("_" + self.item) and not item.startswith("val_"):
                        count += 1
                        total += logs[item]
            if count > 0 and self.compare(total/count, self.op, self.value):
                self.model.stop_training = True

    def compare(self, v1, op, v2):
        if v2 is None: return False
        if op == "<":
            return v1 < v2
        elif op == ">":
            return v1 > v2
        elif op == "==":
            return v1 == v2
        elif op == "<=":
            return v1 <= v2
        elif op == ">=":
            return v1 >= v2

class Network():
    """
    The main class for the conx neural network package.

    Arguments:
        name: Required. The name of the network. Should not contain special HTML
           characters.
        sizes: Optional numbers. Defines the sizes of layers of a sequential
           network. These will be created, added, and connected automatically.
        config: Configuration overrides for the network.

    Note:
        To create a complete, operating network, you must do the following items:

        1. create a network
        2. add layers
        3. connect the layers
        4. compile the network
        5. set the dataset
        6. train the network

        See also :any:`Layer`, :any:`Network.add`, :any:`Network.connect`,
        and :any:`Network.compile`.

    Examples:
        >>> net = Network("XOR1", 2, 5, 2)
        >>> len(net.layers)
        3

        >>> net = Network("XOR2")
        >>> net.add(Layer("input", 2))
        'input'
        >>> net.add(Layer("hidden", 5))
        'hidden'
        >>> net.add(Layer("output", 2))
        'output'
        >>> net.connect()
        >>> len(net.layers)
        3

        >>> net = Network("XOR3")
        >>> net.add(Layer("input", 2))
        'input'
        >>> net.add(Layer("hidden", 5))
        'hidden'
        >>> net.add(Layer("output", 2))
        'output'
        >>> net.connect("input", "hidden")
        >>> net.connect("hidden", "output")
        >>> len(net.layers)
        3

        >>> net = Network("NMIST")
        >>> net.name
        'NMIST'
        >>> len(net.layers)
        0

        >>> net = Network("NMIST", 10, 5, 1)
        >>> len(net.layers)
        3

        >>> net = Network("NMIST", 10, 5, 5, 1, activation="sigmoid")
        >>> net.config["activation"]
        'sigmoid'
        >>> net["output"].activation == "sigmoid"
        True
        >>> net["hidden1"].activation == "sigmoid"
        True
        >>> net["hidden2"].activation == "sigmoid"
        True
        >>> net["input"].activation is None
        True
        >>> net.layers[0].name == "input"
        True
    """
    OPTIMIZERS = ("sgd", "rmsprop", "adagrad", "adadelta", "adam",
                  "adamax", "nadam", "tfoptimizer")

    ERROR_FUNCTIONS = ['binary_crossentropy', 'categorical_crossentropy',
                       'categorical_hinge', 'cosine', 'cosine_proximity', 'hinge',
                       'kld', 'kullback_leibler_divergence', 'logcosh', 'mae', 'mape',
                       'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error',
                       'mean_squared_logarithmic_error', 'mse', 'msle', 'poisson',
                       'sparse_categorical_crossentropy', 'squared_hinge']

    def __init__(self, name: str, *sizes: int, load_config=True, debug=False,
                 build_propagate_from_models=True, **config: Any):
        if not isinstance(name, str):
            raise Exception("first argument should be a name for the network")
        self.debug = debug
        self.build_propagate_from_models = build_propagate_from_models
        ## Pick a place in the random stream, and remember it:
        ## (can override randomness with a particular seed):
        if not isinstance(name, str):
            raise Exception("conx layers need a name as a first parameter")
        self._check_network_name(name)
        self.name = name
        if "seed" in config:
            seed = config["seed"]
            del config["seed"]
        else:
            seed = np.random.randint(2 ** 31 - 1)
        self.seed = seed
        np.random.seed(self.seed)
        self.reset_config()
        ## Next, load a config if available, and override defaults:
        self.layers = []
        if load_config:
            self.load_config()
        ## Override those with args:
        self.config.update(config)
        ## Set initial values:
        self.num_input_layers = 0
        self.num_target_layers = 0
        self.input_bank_order = []
        self.output_bank_order = []
        self.dataset = Dataset(self)
        self.compile_options = {}
        self.train_options = {}
        self._tolerance = K.variable(0.1, dtype='float32', name='tolerance')
        self.layer_dict = {}
        self.epoch_count = 0
        self.history = []
        self.weight_history = {}
        self.update_pictures = get_ipython() is not None
        self._comm = None
        self.model = None
        self.prop_from_dict = {} ## FIXME: can be multiple paths
        self.keras_functions = {}
        self._svg_counter = 1
        self._need_to_show_headings = True
        self._initialized_javascript = False
        # If simple feed-forward network:
        for i in range(len(sizes)):
            if i > 0:
                self.add(Layer(autoname(i, len(sizes)), shape=sizes[i],
                               activation=self.config["activation"]))
            else:
                self.add(Layer(autoname(i, len(sizes)), shape=sizes[i]))
        # Connect them together:
        for i in range(len(sizes) - 1):
            self.connect(autoname(i, len(sizes)), autoname(i+1, len(sizes)))

    def reset_config(self):
        """
        Reset the config back to factor defaults.
        """
        self.config = {
            "font_size": 12, # for svg
            "font_family": "monospace", # for svg
            "border_top": 25, # for svg
            "border_bottom": 25, # for svg
            "hspace": 150, # for svg
            "vspace": 30, # for svg, arrows
            "image_maxdim": 200, # for svg
            "image_pixels_per_unit": 50, # for svg
            "activation": "linear", # Dense default, if none specified
            "arrow_color": "black",
            "arrow_width": "2",
            "border_width": "2",
            "border_color": "black",
            "show_targets": False,
            "show_errors": False,
            "pixels_per_unit": 1,
            "precision": 2,
            "svg_scale": None, # for svg, 0 - 1, or None for optimal
            "svg_rotate": False, # for rotating SVG
            "svg_preferred_size": 400, # in pixels
            "svg_max_width": 800, # in pixels
            "dashboard.dataset": "Train",
            "dashboard.features.bank": "",
            "dashboard.features.columns": 3,
            "dashboard.features.scale": 1.0,
            "config_layers": {},
        }

    def _check_network_name(self, name):
        """
        Check to see if a network name is appropriate.
        Raises exception if invalid name.
        """
        valid_chars = string.ascii_letters + string.digits + " _-"
        if len(name) == 0:
            raise Exception("network name must not be length 0: '%s'" % name)
        if not all(char in valid_chars for char in name):
            raise Exception("network name must only contain letters, numbers, '-', ' ', and '_': '%s'" % name)

    def __getstate__(self):
        return {
            "name": self.name,
            "layers": [layer.__getstate__() for layer in self.layers],
            "outgoing_connections": {layer.name: [layer2.name for layer2 in layer.outgoing_connections]
                                     for layer in self.layers},
            "config": self.config,
        }

    def __setstate__(self, state):
        from .layers import make_layer
        Network.__init__(self, state["name"])
        self.config = state["config"]
        for layer_state in state["layers"]:
            self.add(make_layer(layer_state))
        for layer_from in self.layers:
            for layer_to in state["outgoing_connections"][layer_from.name]:
                self.connect(layer_from.name, layer_to)

    def _get_tolerance(self):
        return K.get_value(self._tolerance)

    def _set_tolerance(self, value):
        K.set_value(self._tolerance, value)

    tolerance = property(_get_tolerance,
                         _set_tolerance)

    def __getitem__(self, layer_name):
        if layer_name not in self.layer_dict:
            return None
        else:
            return self.layer_dict[layer_name]

    def _repr_svg_(self):
        return self.to_svg(show_errors=False, show_targets=False, svg_rotate=False,
                           svg_scale=None)

    def __repr__(self):
        return "<Network name='%s' (%s)>" % (
            self.name, ("uncompiled" if not self.model else "compiled"))

    def set_weights_from_history(self, index, epochs=None):
        """
        Set the weights of the network from a particular point in the learning
        sequence.

        net.set_weights_from_history(0)  # restore initial weights
        net.set_weights_from_history(-1) # restore last weights

        See also:
            * `Network.get_weights_from_history`
        """
        epochs = epochs if epochs is not None else sorted(self.weight_history.keys())
        return self.set_weights(self.get_weights_from_history(index, epochs))

    def get_weights_from_history(self, index, epochs=None):
        """
        Get the weights of the network from a particular point in the learning
        sequence.

        wts = net.get_weights_from_history(0)  # get initial weights
        wts = net.get_weights_from_history(-1) # get last weights

        See also:
            * `Network.set_weights_from_history`
        """
        epochs = epochs if epochs is not None else sorted(self.weight_history.keys())
        return self.weight_history[epochs[index]]

    def playback(self, function):
        """
        Playback a function over the set of recorded weights.

        function has signature: function(network, epoch) and returns
           a displayable, or list of displayables.

        Example:
        >>> net = Network("Playback Test", 2, 2, 1, activation="sigmoid")
        >>> net.compile(error="mse", optimizer="sgd")
        >>> net.dataset.load([
        ...                   [[0, 0], [0]],
        ...                   [[0, 1], [1]],
        ...                   [[1, 0], [1]],
        ...                   [[1, 1], [0]]])
        >>> results = net.train(10, record=True, verbose=0, plot=False)
        >>> def function(network, epoch):
        ...     return None
        >>> sv = net.playback(function)
        >>> ## Testing:
        >>> class Dummy:
        ...     def update(self, result):
        ...         return result
        >>> sv.displayers = [Dummy()]
        >>> print("Testing"); sv.goto("end") # doctest: +ELLIPSIS
        Testing...
        """
        from .widgets import SequenceViewer
        if len(self.weight_history) == 0:
            raise Exception("network wasn't trained with record=True; please train again")
        epochs = sorted(self.weight_history.keys())
        def display_weight_history(index):
            self.set_weights_from_history(index, epochs)
            return function(self, epochs[index])
        sv = SequenceViewer("%s Playback:" % self.name, display_weight_history, len(epochs))
        return sv

    def movie(self, function, movie_name=None, start=0, stop=None, step=1,
              loop=0, optimize=True, duration=100, embed=False, mp4=True):
        """
        Make a movie from a playback function over the set of recorded weights.

        function has signature: function(network, epoch) and should return
        a PIL.Image.

        Example:
            >>> net = Network("Movie Test", 2, 2, 1, activation="sigmoid")
            >>> net.compile(error='mse', optimizer="adam")
            >>> ds = [[[0, 0], [0]],
            ...       [[0, 1], [1]],
            ...       [[1, 0], [1]],
            ...       [[1, 1], [0]]]
            >>> net.dataset.load(ds)
            >>> epochs, khistory = net.train(10, verbose=0, report_rate=1000, record=True, plot=False)
            >>> img = net.movie(lambda net, epoch: net.propagate_to_image("hidden", [1, 1],
            ...                                                           resize=(500, 100)),
            ...                 "/tmp/movie.gif", mp4=False)
            >>> img
            <IPython.core.display.Image object>
        """
        from IPython.display import Image
        if len(self.weight_history) == 0:
            raise Exception("network wasn't trained with record=True; please train again")
        epochs = sorted(self.weight_history.keys())
        if stop is None:
            stop = len(epochs)
        frames = []
        indices = []
        for index in range(start, stop, step):
            self.set_weights_from_history(index, epochs)
            frames.append(function(self, epochs[index]))
            indices.append(index)
        if stop - 1 not in indices:
            self.set_weights_from_history(stop - 1, epochs)
            frames.append(function(self, epochs[stop - 1]))
        if movie_name is None:
            movie_name = "%s-movie.gif" % self.name.replace(" ", "_")
        if frames:
            frames[0].save(movie_name, save_all=True, append_images=frames[1:],
                           optimize=optimize, loop=loop, duration=duration)
            if mp4 is False:
                return Image(url=movie_name, embed=embed)
            else:
                return gif2mp4(movie_name)

    def picture(self, inputs=None, dynamic=False, rotate=False, scale=None,
                show_errors=False, show_targets=False, format="html", class_id=None,
                **kwargs):
        """
        Create an SVG of the network given some inputs (optional).

        >>> net = Network("Picture", 2, 2, 1)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.picture([.5, .5])
        <IPython.core.display.HTML object>
        >>> net.picture([.5, .5], dynamic=True)
        <IPython.core.display.HTML object>
        """
        from IPython.display import HTML
        if any([(layer.kind() == "unconnected") for layer in self.layers]) or len(self.layers) == 0:
            print("Network error: please add layers and connect them")
            return
        if not dynamic:
            if class_id is not None:
                print("WARNING: class_id given but ignored", file=sys.stderr)
            r = random.randint(1, 1000000)
            class_id = "picture-static-%s-%s" % (self.name, r)
        orig_rotate = self.config["svg_rotate"]
        orig_show_errors = self.config["show_errors"]
        orig_show_targets = self.config["show_targets"]
        orig_svg_scale = self.config["svg_scale"]
        self.config["svg_rotate"] = rotate
        self.config["show_errors"] = show_errors
        self.config["show_targets"] = show_targets
        self.config["svg_scale"] = scale
        svg = self.to_svg(inputs=inputs, class_id=class_id, **kwargs)
        self.config["svg_rotate"] = orig_rotate
        self.config["show_errors"] = orig_show_errors
        self.config["show_targets"] = orig_show_targets
        self.config["svg_scale"] = orig_svg_scale
        if format == "html":
            return HTML(svg)
        elif format == "svg":
            return svg
        elif format == "image":
            return svg_to_image(svg)

    def in_console(self, mpl_backend: str) -> bool:
        """
        Return True if running connected to a console; False if connected
        to notebook, or other non-console system.

        Possible values:
            * 'TkAgg' - console with Tk
            * 'Qt5Agg' - console with Qt
            * 'MacOSX' - mac console
            * 'module://ipykernel.pylab.backend_inline' - default for notebook and
              non-console, and when using %matplotlib inline
            * 'NbAgg' - notebook, using %matplotlib notebook

        Here, None means not plotting, or just use text.

        Note:
            If you are running ipython without a DISPLAY with the QT
            background, you may wish to:

            export QT_QPA_PLATFORM='offscreen'
        """
        return mpl_backend not in [
            'module://ipykernel.pylab.backend_inline',
            'NbAgg',
        ]

    def add(self, *layers: Layer) -> None:
        """
        Add layers to the network layer connections. Order is not
        important, unless calling :any:`Network.connect` without any
        arguments.

        Arguments:
            layer: One or more layer instances.

        Returns:
            layer_name (str) - name of last layer added

        Examples:
            >>> net = Network("XOR2")
            >>> net.add(Layer("input", 2))
            'input'
            >>> len(net.layers)
            1

            >>> net = Network("XOR3")
            >>> net.add(Layer("input", 2))
            'input'
            >>> net.add(Layer("hidden", 5))
            'hidden'
            >>> net.add(Layer("hidden2", 5),
            ...         Layer("hidden3", 5),
            ...         Layer("hidden4", 5),
            ...         Layer("hidden5", 5))
            'hidden5'
            >>> net.add(Layer("output", 2))
            'output'
            >>> len(net.layers)
            7

        Note:
            See :any:`Network` for more information.
        """
        last_name = None
        for layer in layers:
            if not isinstance(layer.name, str):
                raise Exception("layer_name should be a string")
            if layer.name in self.layer_dict:
                raise Exception("duplicate layer name '%s'" % layer.name)
            ## Automatic layer naming by pattern:
            if "%d" in layer.name:
                layer_names = [layer.name for layer in self.layers]
                i = 1
                while (layer.name % i) in layer_names:
                    i += 1
                layer.name = layer.name % i
                if hasattr(layer, "params") and "name" in layer.params:
                    layer.params["name"] = layer.name
            self.layers.append(layer)
            self.layer_dict[layer.name] = layer
            ## Layers have link back to network
            layer.network = self
            ## Finally, override any config from network.config:
            self.update_layer_from_config(layer)
            ## Return name, for possible connections
            last_name = layer.name
        return last_name

    def update_layer_from_config(self, layer):
        if layer.name in self.config["config_layers"]:
            for item in self.config["config_layers"][layer.name]:
                setattr(layer, item, self.config["config_layers"][layer.name][item])

    def connect(self, from_layer_name : str=None, to_layer_name : str=None):
        """
        Connect two layers together if called with arguments. If
        called with no arguments, then it will make a sequential
        run through the layers in order added.

        Arguments:
            from_layer_name: Name of layer where connect begins.
            to_layer_name: Name of layer where connection ends.

            If both from_layer_name and to_layer_name are None, then
            all of the layers are connected sequentially in the order
            added.

        Examples:
            >>> net = Network("XOR2")
            >>> net.add(Layer("input", 2))
            'input'
            >>> net.add(Layer("hidden", 5))
            'hidden'
            >>> net.add(Layer("output", 2))
            'output'
            >>> net.connect()
            >>> [layer.name for layer in net["input"].outgoing_connections]
            ['hidden']
        """
        if len(self.layers) == 0:
            raise Exception("no layers have been added")
        if from_layer_name is not None and not isinstance(from_layer_name, str):
            raise Exception("from_layer_name should be a string or None")
        if to_layer_name is not None and not isinstance(to_layer_name, str):
            raise Exception("to_layer_name should be a string or None")
        if from_layer_name is None and to_layer_name is None:
            if (any([layer.outgoing_connections for layer in self.layers]) or
                any([layer.incoming_connections for layer in self.layers])):
                raise Exception("layers already have connections")
            for i in range(len(self.layers) - 1):
                self.connect(self.layers[i].name, self.layers[i+1].name)
        else:
            if from_layer_name == to_layer_name:
                raise Exception("self connections are not allowed")
            if not isinstance(from_layer_name, str):
                raise Exception("from_layer_name should be a string")
            if from_layer_name not in self.layer_dict:
                raise Exception('unknown layer: %s' % from_layer_name)
            if not isinstance(to_layer_name, str):
                raise Exception("to_layer_name should be a string")
            if to_layer_name not in self.layer_dict:
                raise Exception('unknown layer: %s' % to_layer_name)
            from_layer = self.layer_dict[from_layer_name]
            to_layer = self.layer_dict[to_layer_name]
            ## NOTE: these could be allowed, I guess:
            if to_layer in from_layer.outgoing_connections:
                raise Exception("attempting to duplicate connection: %s to %s" % (from_layer_name, to_layer_name))
            from_layer.outgoing_connections.append(to_layer)
            if from_layer in to_layer.incoming_connections:
                raise Exception("attempting to duplicate connection: %s to %s" % (to_layer_name, from_layer_name))
            ## Check for input going to a Dense to warn:
            if from_layer.shape and len(from_layer.shape) > 1 and to_layer.CLASS.__name__ == "Dense":
                print("WARNING: connected multi-dimensional input layer '%s' to layer '%s'; consider adding a FlattenLayer between them" % (
                    from_layer.name, to_layer.name), file=sys.stderr)
            to_layer.incoming_connections.append(from_layer)
            ## Post connection hooks:
            to_layer.on_connect("to", from_layer)
            from_layer.on_connect("from", to_layer)
            ## Compute input/target layers:
            input_layers = [layer for layer in self.layers if layer.kind() == "input"]
            self.num_input_layers = len(input_layers)
            self.input_bank_order = [layer.name for layer in input_layers]
            target_layers = [layer for layer in self.layers if layer.kind() == "output"]
            self.num_target_layers = len(target_layers)
            self.output_bank_order = [layer.name for layer in target_layers]
            ## Set up a layer's input names, as best possible:
            sequence = topological_sort(self, self.layers)
            for layer in sequence:
                if layer.kind() == 'input':
                    layer.input_names = set([layer.name])
                else:
                    if len(layer.incoming_connections) == 1:
                        layer.input_names = layer.incoming_connections[0].input_names
                    else:
                        layer.input_names = set([item for sublist in
                                                 [incoming.input_names for incoming in layer.incoming_connections]
                                                 for item in sublist])
    def depth(self):
        """
        Find the depth of the network graph of connections.
        """
        max_depth = 0
        for in_layer_name in self.input_bank_order:
            for out_layer_name in self.output_bank_order:
                path = find_path(self, in_layer_name, out_layer_name)
                if path:
                    max_depth = max(len(list(path)) + 1, max_depth)
        return max_depth

    def summary(self):
        """
        Print out a summary of the network.
        """
        if self.model:
            self.model.summary()
        else:
            print("Compile network in order to see summary.")

    def reset(self, clear=False, **overrides):
        """
        Reset all of the weights/biases in a network.
        The magnitude is based on the size of the network.
        """
        self.epoch_count = 0
        self.history = []
        self.weight_history.clear()
        self.prop_from_dict.clear()
        self.keras_functions.clear()
        if self.model:
            if "seed" in overrides:
                self.seed = overrides["seed"]
                np.random.seed(self.seed)
                del overrides["seed"]
            # Compile the whole model again:
            if clear:
                self.compile_options = {}
            self.compile_options.update(overrides)
            self.compile(**self.compile_options)

    def test(self, batch_size=32, show=False, tolerance=None, force=False,
             show_inputs=True, show_outputs=True,
             filter="all", interactive=True):
        """
        Test a dataset.
        """
        tolerance = tolerance if tolerance is not None else self.tolerance
        if len(self.dataset.inputs) == 0:
            raise Exception("nothing to test")
        length = len(self.dataset.train_targets)
        if self.dataset._split == 1.0: ## special case; use entire set
            inputs = self.dataset._inputs
            targets = self.dataset._targets
        else:
            ## need to split; check format based on output banks:
            targets = [column[:length] for column in self.dataset._targets]
            inputs = [column[:length] for column in self.dataset._inputs]
        if interactive:
            self._test(inputs, targets, "validation dataset", batch_size, show,
                       tolerance, force, show_inputs, show_outputs, filter, interactive)
        else:
            results = self._test(inputs, targets, "validation dataset", batch_size, show,
                                 tolerance, force, show_inputs, show_outputs, filter, interactive)
            categories = {}
            for i in range(length):
                label = "%s (%s)" % (self.dataset.labels[i],
                                     "correct" if results[i] else "wrong")
                if not label in categories:
                    categories[label] = []
                categories[label].append(self.dataset.inputs[i])
            return sorted(categories.items())

    def _test(self, inputs, targets, dataset, batch_size=32, show=False,
              tolerance=None, force=False,
              show_inputs=True, show_outputs=True,
              filter="all", interactive=True):
        """

        >>> net = Network("Playback Test", 2, 2, 1, activation="sigmoid")
        >>> net.compile(error="mse", optimizer="sgd")
        >>> net.dataset.load([
        ...                   [[0, 0], [0]],
        ...                   [[0, 1], [1]],
        ...                   [[1, 0], [1]],
        ...                   [[1, 1], [0]]])
        >>> array = net.to_array()
        >>> net.from_array(np.zeros(len(array)))  ## Zero-out weights
        >>> net._test(net.dataset._inputs, net.dataset._targets, "TEST")
        ========================================================
        Testing TEST with tolerance None...
        Total count: 4
              correct: 0
              incorrect: 4
        Total percentage correct: 0.0
        >>> net._test(net.dataset._inputs, net.dataset._targets, "TEST", show=True)
        ========================================================
        Testing TEST with tolerance None...
        # | inputs | targets | outputs | result
        ---------------------------------------
        0 | [[0.00,0.00]] | [[0.00]] | [0.50] | X
        1 | [[0.00,1.00]] | [[1.00]] | [0.50] | X
        2 | [[1.00,0.00]] | [[1.00]] | [0.50] | X
        3 | [[1.00,1.00]] | [[0.00]] | [0.50] | X
        Total count: 4
              correct: 0
              incorrect: 4
        Total percentage correct: 0.0
        """
        if interactive:
            print("=" * 56)
            print("Testing %s with tolerance %.6s..." % (dataset, tolerance))
        outputs = self.model.predict(inputs, batch_size=batch_size)
        ## FYI: outputs not shaped
        if self.num_target_layers > 1:
            correct = self.compute_correct(outputs, targets, tolerance)
        else:
            ## Warning:
            ## keras returns outputs as a single column
            ## conx targets are always multi-column
            correct = self.compute_correct([outputs], targets, tolerance)
        count = len(correct)
        if show:
            if show_inputs:
                in_formatted = self.pf_matrix(inputs, force)
                count = len(in_formatted)
            if show_outputs:
                targ_formatted = self.pf_matrix(targets, force)
                out_formatted = self.pf_matrix(outputs, force)
                count = len(out_formatted)
            header = "# | "
            if show_inputs:
                header += "inputs | "
            if show_outputs:
                header += "targets | outputs | "
            header += "result"
            print(header)
            print("---------------------------------------")
            for i in range(count):
                show_it = ((filter == "all") or
                           (filter == "correct" and correct[i]) or
                           (filter == "incorrect" and not correct[i]))
                if show_it:
                    line = "%d | " % i
                    if show_inputs:
                        line += "%s | " % in_formatted[i]
                    if show_outputs:
                        line += "%s | %s | " % (targ_formatted[i], out_formatted[i])
                    line += "correct" if correct[i] else "X"
                    print(line)
        if interactive:
            print("Total count:", len(correct))
            print("      correct:", len([c for c in correct if c]))
            print("      incorrect:", len([c for c in correct if not c]))
            print("Total percentage correct:", list(correct).count(True)/len(correct))
        else:
            return list(correct)

    def compute_correct(self, outputs, targets, tolerance=None):
        """
        Both are np.arrays. Return [True, ...].
        """
        tolerance = tolerance if tolerance is not None else self.tolerance
        correct = []
        for r in range(len(outputs[0])):
            row = []
            for c in range(len(outputs)):
                row.extend(list(map(lambda v: v <= tolerance, np.abs(outputs[c][r] - targets[c][r]))))
            correct.append(all(row))
        return correct

    def train_one(self, inputs, targets, batch_size=32, update_pictures=False):
        """
        Train on one input/target pair.

        Inputs should be a vector if one input bank, or a list of vectors
        if more than one input bank.

        Targets should be a vector if one output bank, or a list of vectors
        if more than one output bank.

        Alternatively, inputs and targets can each be a dictionary mapping
        bank to vector.

        Examples:

            >>> from conx import Network, Layer, SGD, Dataset
            >>> net = Network("XOR", 2, 2, 1, activation="sigmoid")
            >>> net.compile(error='mean_squared_error',
            ...             optimizer=SGD(lr=0.3, momentum=0.9))
            >>> ds = [[[0, 0], [0]],
            ...       [[0, 1], [1]],
            ...       [[1, 0], [1]],
            ...       [[1, 1], [0]]]
            >>> net.dataset.load(ds)
            >>> out, err = net.train_one({"input": [0, 0]},
            ...                          {"output": [0]})
            >>> len(out)
            1
            >>> len(err)
            1

            >>> from conx import Network, Layer, SGD, Dataset
            >>> net = Network("XOR2")
            >>> net.add(Layer("input%d", shape=1))
            'input1'
            >>> net.add(Layer("input%d", shape=1))
            'input2'
            >>> net.add(Layer("hidden%d", shape=2, activation="sigmoid"))
            'hidden1'
            >>> net.add(Layer("hidden%d", shape=2, activation="sigmoid"))
            'hidden2'
            >>> net.add(Layer("shared-hidden", shape=2, activation="sigmoid"))
            'shared-hidden'
            >>> net.add(Layer("output%d", shape=1, activation="sigmoid"))
            'output1'
            >>> net.add(Layer("output%d", shape=1, activation="sigmoid"))
            'output2'
            >>> net.connect("input1", "hidden1")
            >>> net.connect("input2", "hidden2")
            >>> net.connect("hidden1", "shared-hidden")
            >>> net.connect("hidden2", "shared-hidden")
            >>> net.connect("shared-hidden", "output1")
            >>> net.connect("shared-hidden", "output2")
            >>> net.compile(error='mean_squared_error',
            ...             optimizer=SGD(lr=0.3, momentum=0.9))
            >>> ds = [([[0],[0]], [[0],[0]]),
            ...       ([[0],[1]], [[1],[1]]),
            ...       ([[1],[0]], [[1],[1]]),
            ...       ([[1],[1]], [[0],[0]])]
            >>> net.dataset.load(ds)
            >>> net.compile(error='mean_squared_error',
            ...             optimizer=SGD(lr=0.3, momentum=0.9))
            >>> out, err = net.train_one({"input1": [0], "input2": [0]},
            ...                          {"output1": [0], "output2": [0]})
            >>> len(out)
            2
            >>> len(err)
            2
            >>> net.dataset._num_input_banks()
            2
            >>> net.dataset._num_target_banks()
            2
        """
        if isinstance(inputs, dict):
            inputs = [inputs[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                inputs = inputs[0]
        if isinstance(targets, dict):
            targets = [targets[name] for name in self.output_bank_order]
            if self.num_target_layers == 1:
                targets = targets[0]
        pairs = [(inputs, targets)]
        if self.num_input_layers == 1:
            ins = np.array([pair[0] for pair in pairs], "float32")
        else:
            ins = []
            for i in range(len(pairs[0][0])):
                ins.append(np.array([pair[0][i] for pair in pairs], "float32"))
        if self.num_target_layers == 1:
            targs = np.array([pair[1] for pair in pairs], "float32")
        else:
            targs = []
            for i in range(len(pairs[0][1])):
                targs.append(np.array([pair[1][i] for pair in pairs], "float32"))
        history = self.model.fit(ins, targs, epochs=1, verbose=0, batch_size=batch_size)
        ## may need to update history?
        outputs = self.propagate(inputs, batch_size=batch_size, update_pictures=update_pictures)
        if len(self.output_bank_order) == 1:
            errors = (np.array(outputs) - np.array(targets)).tolist()
        else:
            errors = []
            for bank in range(len(self.output_bank_order)):
                errors.append((np.array(outputs[bank]) - np.array(targets[bank])).tolist())
        if update_pictures:
            if self.config["show_targets"]:
                if len(self.output_bank_order) == 1:
                    self.display_component([targets], "targets")
                else:
                    self.display_component(targets, "targets")
            if self.config["show_errors"]: ## min max is error:
                if len(self.output_bank_order) == 1:
                    self.display_component([errors], "errors", minmax=(-1, 1))
                else:
                    errors = []
                    for bank in range(len(self.output_bank_order)):
                        errors.append( np.array(outputs[bank]) - np.array(targets[bank]))
                    self.display_component(errors, "errors", minmax=(-1, 1))
        return (outputs, errors)

    def retrain(self, **overrides):
        """
        Call network.train() again with same options as last call, unless overrides.
        """
        for key in overrides:
            if key not in self.train_options:
                raise Exception("Unknown train option: %s" % key)
        self.train_options.update(overrides)
        self.train(**self.train_options)

    def _compute_result_acc(self, results):
        """
        Compute accuracy from results. There are no val_ items here.
        """
        if "acc" in results: return results["acc"]
        values = [results[key] for key in results if key.endswith("_acc")]
        if len(values) > 0:
            return sum(values)/len(values)
        else:
            raise Exception("attempting to find accuracy in results, but there aren't any")

    def evaluate(self, batch_size=32):
        """
        Test the network on the train and test data, returning a dict of results.

        Example:
            >>> net = Network("Evaluate", 2, 2, 1, activation="sigmoid")
            >>> net.compile(error='mean_squared_error', optimizer="adam")
            >>> ds = [[[0, 0], [0]],
            ...       [[0, 1], [1]],
            ...       [[1, 0], [1]],
            ...       [[1, 1], [0]]]
            >>> net.dataset.load(ds)
            >>> net.evaluate()           # doctest: +ELLIPSIS
            {'loss': ..., 'acc': ...}
        """
        if len(self.dataset.inputs) == 0:
            raise Exception("no dataset loaded")
        if self.model is None:
            raise Exception("need to compile network")
        (train_inputs, train_targets), (test_inputs, test_targets) = self.dataset._split_data()
        train_metrics = self.model.evaluate(train_inputs, train_targets, batch_size=batch_size, verbose=0)
        results = {k:v for k, v in zip(self.model.metrics_names, train_metrics)}
        if len(test_inputs) > 0:
            test_metrics = self.model.evaluate(test_inputs, test_targets, batch_size=batch_size, verbose=0)
            results.update({"val_"+k: v for k, v in zip(self.model.metrics_names, test_metrics)})
        return results

    def test_dataset_ranges(self):
        """
        Test the dataset ranges to see if in range of activation functions.
        """
        if len(self.dataset.targets) == 0:
            return # nothing to test
        for index in range(len(self.dataset._targets)):
            if len(self.dataset._targets[index].shape) > 2:
                print("WARNING: network '%s' target bank #%s has a multi-dimensional shape, which is not allowed" %
                      (self.name, index), file=sys.stderr)
        for index in range(len(self.output_bank_order)):
            layer_name = self.output_bank_order[index]
            if self[layer_name].activation == "linear":
                continue
            lmin, lmax = self[layer_name].get_act_minmax()
            # test dataset min to see if in range of act output:
            if not (lmin <= self.dataset._targets_range[index][0] <= lmax):
                print("WARNING: output bank '%s' has activation function, '%s', that is not consistent with minimum value of targets" %
                      (layer_name, self[layer_name].activation), file=sys.stderr)
            # test dataset min to see if in range of act output:
            if not (lmin <= self.dataset._targets_range[index][1] <= lmax):
                print("WARNING: output bank '%s' has activation function, '%s', that is not consistent with maximum value of targets" %
                      (layer_name, self[layer_name].activation), file=sys.stderr)

    def train(self, epochs=1, accuracy=None, error=None, batch_size=32,
              report_rate=1, verbose=1, kverbose=0, shuffle=True, tolerance=None,
              class_weight=None, sample_weight=None, use_validation_to_stop=False,
              plot=True, record=0, callbacks=None, save=False):
        """
        Train the network.

        To stop before number of epochs, give either error=VALUE, or accuracy=VALUE.

        Normally, it will check training info to stop, unless you
        use_validation_to_stop = True.

        Arguments:
            epochs (int): Maximum number of epochs (sweeps) through
                training data.
            accuracy (float): Value of correctness (0.0 - 1.0) to attain in
                order to stop. Depends on tolerance to determine accuracy.
            error (float): Error to attain in order to stop. Depends on error
                function given in `Network.compile`.
            batch_size (int): Size of batch to train on.
            report_rate (int): Rate of feedback on learning, in epochs.
            verbose (int): Level of feedback on training. verbose=0 gives no
                feedback, but returns (epoch_count, result)
            kverbose (int): Level of feedback from Keras.
            shuffle (bool or str): Should the training data be shuffled?
                'batch' shuffles in batch-sized chunks.
            tolerance (float): The maximum difference between target and output
                that should be considered correct.
            class_weight (float):
            sample_weight (float):
            use_validation_to_stop (bool): If `True`, then accuracy and error will
                use the validation set rather than the training set.
            plot (bool): If `True`, then the feedback will be shown in graphical form.
            record (int): If 'record != 0', the weights will be saved every record
                number of epochs.
            callbacks (list): A list of (str, function) where str is 'on_batch_begin',
                'on_batch_end', 'on_epoch_begin', 'on_epoch_end', 'on_train_begin',
                or 'on_train_end', and function takes a network, and other
                parameters, depending on str.
            save (bool): If `True`, then the network is saved at end, whether
                interrupted or not.

        Returns:
            tuple: (epoch_count, result) if verbose == 0
            None: if verbose != 0

        Examples:
            >>> net = Network("Train Test", 1, 3, 1)
            >>> net.compile(error="mse", optimizer="rmsprop")
            >>> net.dataset.append([0.0], [1.0])
            >>> net.dataset.append([1.0], [0.0])
            >>> net.train(plot=False)  # doctest: +ELLIPSIS
            Evaluating initial training metrics...
            Training...
            ...
        """
        self.train_options = {
            "epochs": epochs,
            "accuracy": accuracy,
            "error": error,
            "batch_size": batch_size,
            "report_rate": report_rate,
            "verbose": verbose,
            "shuffle": shuffle,
            "class_weight": class_weight,
            "sample_weight": sample_weight,
            "tolerance": tolerance,
            "use_validation_to_stop": use_validation_to_stop,
            "plot": plot,
            "record": record,
            "callbacks": callbacks,
            "save": save,
            }
        if plot:
            import matplotlib
            mpl_backend = matplotlib.get_backend()
        else:
            mpl_backend = None
        if self.model is None:
            raise Exception("need to compile network")
        if not isinstance(report_rate, numbers.Integral) or report_rate < 1:
            raise Exception("bad report rate: %s" % (report_rate,))
        if not (isinstance(batch_size, numbers.Integral) or batch_size is None):
            raise Exception("bad batch size: %s" % (batch_size,))
        ## Test for targets in range of activation function:
        self.test_dataset_ranges()
        if epochs == 0: return
        if len(self.dataset.inputs) == 0:
            print("No training data available")
            return
        if use_validation_to_stop:
            if (self.dataset._split == 0):
                print("Attempting to use validation to stop, but Network.dataset.split() is 0")
                return
            elif ((accuracy is None) and (error is None)):
                print("Attempting to use validation to stop, but neither accuracy nor error was set")
                return
        self._need_to_show_headings = True
        if tolerance is not None:
            if accuracy is None:
                raise Exception("tolerance given but unknown accuracy")
            K.set_value(self._tolerance, tolerance)
        ## Going to need evaluation on training set in any event:
        if self.dataset._split == 1.0: ## special case; use entire set
            inputs = self.dataset._inputs
            targets = self.dataset._targets
        else:
            ## need to split; check format based on output banks:
            length = len(self.dataset.train_targets)
            targets = [column[:length] for column in self.dataset._targets]
            inputs = [column[:length] for column in self.dataset._inputs]
        if len(self.history) > 0:
            results = self.history[-1]
        else:
            if verbose > 0:
                print("Evaluating initial training metrics...")
            values = self.model.evaluate(inputs, targets, batch_size=batch_size, verbose=0)
            if not isinstance(values, list): # if metrics is just a single value
                values = [values]
            results = {metric: value for metric,value in zip(self.model.metrics_names, values)}
        results_acc = self._compute_result_acc(results)
        ## look at split, use validation subset:
        if self.dataset._split == 0.0: ## None
            val_results = {}
        elif self.dataset._split == 1.0: ## special case; use entire set; already done!
            val_results = {"val_%s" % key: results[key] for key in results}
        else: # split is greater than 0, less than 1
            if verbose > 0:
                print("Evaluating initial validation metrics...")
            ## need to split; check format based on output banks:
            length = len(self.dataset.test_targets)
            targets = [column[-length:] for column in self.dataset._targets]
            inputs = [column[-length:] for column in self.dataset._inputs]
            val_values = self.model.evaluate(inputs, targets, batch_size=batch_size, verbose=0)
            val_results = {"val_%s" % metric: value for metric,value in zip(self.model.metrics_names, val_values)}
        if val_results:
            val_results_acc = self._compute_result_acc(val_results)
        if use_validation_to_stop:
            if ((self.dataset._split > 0) and
                ((accuracy is not None) or (error is not None))):
                need_to_train = True
                if ((accuracy is not None) and (val_results_acc >= accuracy)):
                    print("No training required: validation accuracy already to desired value")
                    need_to_train = False
                elif ((error is not None) and (val_results["loss"] <= error)):
                    print("No training required: validation error already to desired value")
                    need_to_train = False
                if not need_to_train:
                    print("Training dataset status:")
                    self.report_epoch(self.epoch_count, results)
                    print("Validation dataset status:")
                    self.report_epoch(self.epoch_count, val_results)
                    return (self.epoch_count, results) if verbose == 0 else None
        else: ## regular training to stop, use_validation_to_stop is False
            if ((accuracy is not None) and (results_acc >= accuracy)):
                print("No training required: accuracy already to desired value")
                print("Training dataset status:")
                self.report_epoch(self.epoch_count, results)
                return (self.epoch_count, results) if verbose == 0 else None
            elif ((error is not None) and (results["loss"] <= error)):
                print("No training required: error already to desired value")
                print("Training dataset status:")
                self.report_epoch(self.epoch_count, results)
                return (self.epoch_count, results) if verbose == 0 else None
        ## Ok, now we know we need to train:
        results.update(val_results)
        if len(self.history) == 0:
            self.history = [results]
            if record:
                self.weight_history[0] = self.get_weights()
        if verbose > 0:
            print("Training...")
        if self.in_console(mpl_backend) and verbose > 0:
            self.report_epoch(self.epoch_count, self.history[-1])
        interrupted = False
        kcallbacks = [
            History(),
            ReportCallback(self, verbose, report_rate, mpl_backend, record),
        ]
        if accuracy is not None:
            kcallbacks.append(StoppingCriteria("acc", ">=", accuracy, use_validation_to_stop))
        if error is not None:
            kcallbacks.append(StoppingCriteria("loss", "<=", error, use_validation_to_stop))
        if plot:
            pc = PlotCallback(self, report_rate, mpl_backend)
            kcallbacks.append(pc)
        if callbacks is not None:
            for (on_method, function) in callbacks:
                kcallbacks.append(FunctionCallback(self, on_method, function))
        with _InterruptHandler(self) as handler:
            if self.dataset._split == 1:
                result = self.model.fit(self.dataset._inputs,
                                        self.dataset._targets,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_data=(self.dataset._inputs,
                                                         self.dataset._targets),
                                        callbacks=kcallbacks,
                                        shuffle=shuffle,
                                        class_weight=class_weight,
                                        sample_weight=sample_weight,
                                        verbose=kverbose)
            else:
                result = self.model.fit(self.dataset._inputs,
                                        self.dataset._targets,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_split=self.dataset._split,
                                        callbacks=kcallbacks,
                                        shuffle=shuffle,
                                        class_weight=class_weight,
                                        sample_weight=sample_weight,
                                        verbose=kverbose)
            if plot:
                pc.on_epoch_end(-1)
            if handler.interrupted:
                interrupted = True
        if interrupted:
            if verbose:
                print("Interrupted! Cleaning up...")
        last_epoch = self.history[-1]
        if record:
            self.weight_history[self.epoch_count] = self.get_weights()
        assert len(self.history) == self.epoch_count+1  # +1 is for epoch 0
        if verbose:
            print("=" * 56)
            self.report_epoch(self.epoch_count, last_epoch)
        if save:
            if verbose:
                print("Saving network... ", end="")
            self.save()
            if verbose:
                print("Saved!")
        if interrupted:
            raise KeyboardInterrupt
        if verbose == 0:
            return (self.epoch_count, self.history[-1])

    def report_epoch(self, epoch_count, results):
        """
        Print out stats for the epoch.
        """
        if self._need_to_show_headings:
            h1 = "       "
            h2 = "Epochs "
            h3 = "------ "
            if 'loss' in results:
                h1 += "|  Training "
                h2 += "|     Error "
                h3 += "| --------- "
            if 'acc' in results:
                h1 += "|  Training "
                h2 += "|  Accuracy "
                h3 += "| --------- "
            if 'val_loss' in results:
                h1 += "|  Validate "
                h2 += "|     Error "
                h3 += "| --------- "
            if 'val_acc' in results:
                h1 += "|  Validate "
                h2 += "|  Accuracy "
                h3 += "| --------- "
            for other in sorted(results):
                if other not in ["loss", "acc", "val_loss", "val_acc"]:
                    if not other.endswith("_loss"):
                        w1, w2 = other.replace("_", " ").split(" ", 1)
                        maxlen = max(len(w1), len(w2), 9)
                        h1 += "| " + (("%%%ds " % maxlen) % w1)
                        h2 += "| " + (("%%%ds " % maxlen) % w2)
                        h3 += "| %s " % ("-" * (maxlen))
            print(h1)
            print(h2)
            print(h3)
            self._need_to_show_headings = False
        s = "#%5d " % (epoch_count,)
        if 'loss' in results:
            s += "| %9.5f " % (results['loss'],)
        if 'acc' in results:
            s += "| %9.5f " % (results['acc'],)
        if 'val_loss' in results:
            s += "| %9.5f " % (results['val_loss'],)
        if 'val_acc' in results:
            s += "| %9.5f " % (results['val_acc'],)
        for other in sorted(results):
            if other not in ["loss", "acc", "val_loss", "val_acc"]:
                if not other.endswith("_loss"):
                    other_str = other
                    if other.endswith("_acc"):
                        other_str = other[:-4] + " accuracy"
                    s += "| %9.5f " % results[other]
        print(s)

    def set_dataset(self, dataset):
        """
        Set the dataset for the network.

        Examples:
            >>> from conx import Dataset
            >>> data = [[[0, 0], [0]],
            ...         [[0, 1], [1]],
            ...         [[1, 0], [1]],
            ...         [[1, 1], [0]]]
            >>> ds = Dataset()
            >>> ds.load(data)
            >>> net = Network("Set Dataset Test", 2, 2, 1)
            >>> net.compile(error="mse", optimizer="adam")
            >>> net.set_dataset(ds)
        """
        if not isinstance(dataset, Dataset):
            raise Exception("Network.set_dataset() takes a Dataset object")
        if dataset.network is not None:
            print("INFO: using dataset on a new network, replacing old network", file=sys.stderr)
        self.dataset = dataset
        self.dataset.network = self
        self.test_dataset_ranges()
        self.dataset._verify_network_dataset_match()

    def set_activation(self, layer_name, activation):
        """
        Swap activation function of a layer after compile.
        """
        from keras.models import load_model
        import keras.activations
        import tempfile
        if not isinstance(layer_name, str):
            raise Exception("layer_name should be a string")
        if not isinstance(activation, str):
            activation = activation.__name__
        acts = {
            'relu': keras.activations.relu,
            'sigmoid': keras.activations.sigmoid,
            'linear': keras.activations.linear,
            'softmax': keras.activations.softmax,
            'tanh': keras.activations.tanh,
            'elu': keras.activations.elu,
            'selu': keras.activations.selu,
            'softplus': keras.activations.softplus,
            'softsign': keras.activations.softsign,
            'hard_sigmoid': keras.activations.hard_sigmoid,
        }
        if self.model:
            self[layer_name].keras_layer.activation = acts[activation]
            self[layer_name].activation = activation
            with tempfile.NamedTemporaryFile() as tf:
                filename = tf.name
                self.model.save(filename)
                self.model = load_model(filename)
        else:
            raise Exception("can't change activation until after compile")

    def get_weights_as_image(self, layer_name, colormap=None):
        """
        Get the weights from the model.

        >>> net = Network("Weight as Image Test", 2, 2, 5)
        >>> net.compile(error="mse", optimizer="adam")
        >>> net.get_weights_as_image("hidden") # doctest: +ELLIPSIS
        <PIL.Image.Image image mode=RGBA size=2x2 at ...>
        """
        from matplotlib import cm
        if not isinstance(layer_name, str):
            raise Exception("layer_name should be a string")
        if self.model is None:
            raise Exception("need to compile network")
        weights = [layer.get_weights() for layer in self.model.layers
                   if layer_name == layer.name][0]
        weights = weights[0] # get the weight matrix, not the biases
        vector = scale_output_for_image(weights, (-5,5), truncate=True)
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        size = self.config["pixels_per_unit"]
        new_width = vector.shape[0] * size # in, pixels
        new_height = vector.shape[1] * size # in, pixels
        if colormap is None:
            colormap = get_colormap() if self[layer_name].colormap is None else self[layer_name].colormap
        try:
            cm_hot = cm.get_cmap(colormap)
        except:
            cm_hot = cm.get_cmap("RdGy")
        vector = cm_hot(vector)
        vector = np.uint8(vector * 255)
        image = PIL.Image.fromarray(vector)
        image = image.resize((new_height, new_width))
        return image

    def get_weights(self, layer_name=None):
        """
        Get the weights from a layer, or the entire model.

        Examples:
            >>> net = Network("Weight Test", 2, 2, 5)
            >>> net.compile(error="mse", optimizer="adam")
            >>> len(net.get_weights("input"))
            0
            >>> len(net.get_weights("hidden"))
            2
            >>> shape(net.get_weights("hidden")[0])  ## weights
            (2, 2)
            >>> shape(net.get_weights("hidden")[1])  ## biases
            (2,)
            >>> len(net.get_weights("output"))
            2
            >>> shape(net.get_weights("output")[0])  ## weights
            (2, 5)
            >>> shape(net.get_weights("output")[1])  ## biases
            (5,)

            >>> net = Network("Weight Get Test", 2, 2, 1, activation="sigmoid")
            >>> net.compile(error="mse", optimizer="sgd")
            >>> len(net.get_weights())
            4

        See also:
            * `Network.to_array`
            * `Network.from_array`
            * `Network.get_weights_as_image`
        """
        if self.model is None:
            raise Exception("need to compile network")
        if layer_name is not None:
            weights = [layer.get_weights() for layer in self.model.layers
                       if layer_name == layer.name][0]
            return [m.tolist() for m in weights]
        else:
            return self.model.get_weights()

    def propagate(self, input, batch_size=32, class_id=None,
                  update_pictures=False, raw=False):
        """
        Propagate an input (in human API) through the network.
        If visualizing, the network image will be updated.

        Inputs should be a vector if one input bank, or a list of vectors
        if more than one input bank.

        Alternatively, inputs can be a dictionary mapping
        bank to vector.

        >>> net = Network("Prop Test", 2, 2, 5)
        >>> net.compile(error="mse", optimizer="adam")
        >>> len(net.propagate([0.5, 0.5]))
        5
        >>> len(net.propagate({"input": [1, 1]}))
        5
        """
        if self.model is None:
            raise Exception("Need to compile network first")
        if isinstance(input, dict):
            input = [input[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                input = input[0]
        elif isinstance(input, PIL.Image.Image):
            input = image_to_array(input)
        ## End of input setup
        if not is_array_like(input):
            raise Exception("inputs should be an array")
        if raw:
            outputs = self.model.predict(np.array(input), batch_size=batch_size)
        elif self.num_input_layers == 1:
            outputs = self.model.predict(np.array([input]), batch_size=batch_size)
        else:
            inputs = [np.array([x], "float32") for x in input]
            outputs = self.model.predict(inputs, batch_size=batch_size)
        ## Shape the outputs:
        if raw:
            pass
        elif self.num_target_layers == 1:
            shape = self[self.output_bank_order[0]].shape
            try:
                outputs = outputs[0].reshape(shape).tolist()
            except:
                outputs = outputs[0].tolist()  # can't reshape; maybe a dynamically changing output
        else:
            shapes = [self[layer_name].shape for layer_name in self.output_bank_order]
            ## FIXME: may not be able to reshape; dynamically changing output
            outputs = [outputs[i].reshape(shapes[i]).tolist() for i in range(len(self.output_bank_order))]
        if update_pictures:
            for layer in self.layers:
                self.propagate_to(layer.name, input, batch_size, class_id=class_id,
                                  update_pictures=update_pictures, raw=raw, update_path=False)
        return outputs

    def propagate_from(self, layer_name, input, output_layer_names=None,
                       batch_size=32, update_pictures=False, raw=False):
        """
        Propagate activations from the given layer name to the output layers.
        """
        if not isinstance(layer_name, str):
            raise Exception("layer_name should be a string")
        if layer_name not in self.layer_dict:
            raise Exception("No such layer '%s'" % layer_name)
        if isinstance(input, dict):
            input = [input[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                input = input[0]
        elif isinstance(input, PIL.Image.Image):
            input = image_to_array(input)
        ## End of input setup
        if not is_array_like(input):
            raise Exception("inputs should be an array")
        if output_layer_names is None:
            if self.num_target_layers == 1:
                output_layer_names = [layer.name for layer in self.layers if layer.kind() == "output"]
            else:
                output_layer_names = self.output_bank_order
        else:
            if isinstance(output_layer_names, str):
                output_layer_names = [output_layer_names]
        outputs = []
        for output_layer_name in output_layer_names:
            # We should be able to get the prop_from model:
            ## FIXME: could be multiple paths
            prop_model = self.prop_from_dict.get((layer_name, output_layer_name), None)
            if raw:
                inputs = input
            else:
                inputs = np.array([input])
            if prop_model is not None:
                outputs.append([list(x) for x in prop_model.predict(inputs)][0])
            ## FYI: outputs not shaped
        if update_pictures:
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            ## Update from start to rest of graph
            if self._comm.kernel:
                ## viz this layer:
                if self[layer_name].visible:
                    image = self[layer_name].make_image(inputs, config=self.config)
                    data_uri = self._image_to_uri(image)
                    class_id_name = "%s_%s" % (self.name, layer_name)
                    if self.config["svg_rotate"]:
                        class_id_name += "-rotated"
                    if self.debug: print("propagate_from 1: class_id_name:", class_id_name)
                    self._comm.send({'class': class_id_name, "href": data_uri})
                for output_layer_name in output_layer_names:
                    path = find_path(self, layer_name, output_layer_name)
                    if path is not None:
                        for layer in path:
                            if not layer.visible:
                                continue
                            if (layer_name, layer.name) not in self.prop_from_dict:
                                continue
                            ## FIXME: could be multiple paths
                            model = self.prop_from_dict[(layer_name, layer.name)]
                            vector = model.predict(inputs)[0]
                            ## FYI: outputs not shaped
                            image = layer.make_image(vector, config=self.config)
                            data_uri = self._image_to_uri(image)
                            class_id_name = "%s_%s" % (self.name, layer.name)
                            if self.config["svg_rotate"]:
                                class_id_name += "-rotated"
                            if self.debug: print("propagate_from 2: class_id_name:", class_id_name)
                            self._comm.send({'class': class_id_name, "href": data_uri})
        if raw:
            return outputs
        elif len(output_layer_names) == 1 and len(outputs) > 0:
            return outputs[0]
        else:
            return outputs

    def display_component(self, vector, component, class_id=None, **opts):
        """
        vector is a list, one each per output layer. component is "errors" or "targets"
        """
        config = copy.copy(self.config)
        config.update(opts)
        output_names = self.output_bank_order
        if self._comm.kernel:
            for (target, layer_name) in zip(vector, output_names):
                array = np.array(target)
                if component == "targets":
                    colormap = self[layer_name].colormap
                else:
                    colormap = get_error_colormap()
                image = self[layer_name].make_image(array, colormap, config)
                data_uri = self._image_to_uri(image)
                if class_id is None:
                    class_id_name = "%s_%s" % (self.name, layer_name)
                else:
                    class_id_name = "%s_%s" % (class_id, layer_name)
                if self.debug: print("display_component: sending to class_id:", class_id_name + "_" + component)
                self._comm.send({'class': class_id_name + "_" + component,
                                 "href": data_uri})

    def propagate_to(self, layer_name, inputs, batch_size=32, class_id=None,
                     update_pictures=False, update_path=True, raw=False):
        """
        Computes activation at a layer. Side-effect: updates live SVG.

        Arguments:
            layer_name (str) - name of layer to propagate activations to
            inputs - list of numbers, vector to propagate
            batch_size (int) - size of batch
            update_pictures (bool) - send images to notebook SVG images
            raw (bool) - if True, don't process inputs or outputs
        """
        if not isinstance(layer_name, str):
            raise Exception("layer_name should be a string")
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        if isinstance(inputs, dict):
            inputs = [inputs[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                inputs = inputs[0]
        elif isinstance(inputs, PIL.Image.Image):
            inputs = image_to_array(inputs)
        ## End of input setup
        if not is_array_like(inputs):
            raise Exception("inputs should be an array")
        if raw:
            outputs = self[layer_name].model.predict(np.array(inputs), batch_size=batch_size)
        elif self.num_input_layers == 1:
            outputs = self[layer_name].model.predict(np.array([inputs]), batch_size=batch_size)
        else:
            # get just inputs for this layer, in order:
            vector = [np.array([inputs[self.input_bank_order.index(name)]]) for name in
                      self._get_sorted_input_names(self[layer_name].input_names)]
            outputs = self[layer_name].model.predict(vector, batch_size=batch_size)
        ## output shaped below:
        if update_pictures:
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            if self._comm.kernel:
                if update_path: ## update the whole path, from all inputs to the layer_name, if a path
                    ## don't repeat any updates, so keep track of what you have done:
                    updated = set([])
                    for input_layer_name in self.input_bank_order:
                        if input_layer_name not in updated:
                            image = self._propagate_to_image(input_layer_name, inputs, raw=raw)
                            data_uri = self._image_to_uri(image)
                            if class_id is None:
                                class_id_name = "%s_%s" % (self.name, input_layer_name)
                            else:
                                class_id_name = "%s_%s" % (class_id, input_layer_name)
                            if self.config["svg_rotate"]:
                                class_id_name += "-rotated"
                            if self.debug: print("propagate_to 1: sending to class_id_name:", class_id_name)
                            self._comm.send({'class': class_id_name, "href": data_uri})
                            updated.add(input_layer_name)
                        path = find_path(self, input_layer_name, layer_name)
                        if path is not None:
                            for layer in path:
                                if layer.visible and layer.model is not None:
                                    if layer.name not in updated:
                                        image = self._propagate_to_image(layer.name, inputs, raw=raw)
                                        data_uri = self._image_to_uri(image)
                                        if class_id is None:
                                            class_id_name = "%s_%s" % (self.name, layer.name)
                                        else:
                                            class_id_name = "%s_%s" % (class_id, layer.name)
                                        if self.config["svg_rotate"]:
                                            class_id_name += "-rotated"
                                        if self.debug: print("propagate_to 2: sending to class_id_name:", class_id_name)
                                        self._comm.send({'class': class_id_name, "href": data_uri})
                                        updated.add(layer.name)
                else: # not the whole path, just to the layer_name
                    image = self._propagate_to_image(layer_name, inputs, raw=raw)
                    data_uri = self._image_to_uri(image)
                    if class_id is None:
                        class_id_name = "%s_%s" % (self.name, layer_name)
                    else:
                        class_id_name = "%s_%s" % (class_id, layer_name)
                    if self.config["svg_rotate"]:
                        class_id_name += "-rotated"
                    if self.debug: print("propagate_to 3: sending to class_id_name:", class_id_name)
                    self._comm.send({'class': class_id_name, "href": data_uri})
        ## Shape the outputs:
        if raw:
            return outputs
        shape = self[layer_name].shape
        if shape and all([isinstance(v, numbers.Integral) for v in shape]):
            try:
                outputs = outputs[0].reshape(shape).tolist()
            except:
                outputs = outputs[0].tolist()
        else:
            outputs = outputs[0].tolist()
        return outputs

    def _layer_has_features(self, layer_name):
        output_shape = self[layer_name].get_output_shape()
        return (isinstance(output_shape, tuple) and len(output_shape) == 4)


    def propagate_to_features(self, layer_name, inputs, cols=5, resize=None, scale=1.0,
                              html=True, size=None, display=True, class_id=None,
                              update_pictures=False, raw=False):
        """
        if html is True, then generate HTML, otherwise send images.
        """
        from IPython.display import HTML
        if isinstance(inputs, dict):
            inputs = [inputs[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                inputs = inputs[0]
        elif isinstance(inputs, PIL.Image.Image):
            inputs = image_to_array(inputs)
        ## End of input setup
        if not is_array_like(inputs):
            raise Exception("inputs should be an array")
        if not isinstance(layer_name, str):
            raise Exception("layer_name should be a string")
        output_shape = self[layer_name].get_output_shape()
        retval = """<table><tr>"""
        if self._layer_has_features(layer_name):
            if html:
                orig_feature = self[layer_name].feature
                for i in range(output_shape[3]):
                    self[layer_name].feature = i
                    ## This should return in proper orientation, regardless of rotate setting:
                    image = self.propagate_to_image(layer_name, inputs, class_id=class_id,
                                                    update_pictures=update_pictures, raw=raw)
                    if resize is not None:
                        image = image.resize(resize)
                    if scale != 1.0:
                        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
                    data_uri = self._image_to_uri(image)
                    retval += """<td style="border: 1px solid black;"><img style="image-rendering: pixelated;" class="%s_%s_feature%s" src="%s"/><br/><center>Feature %s</center></td>""" % (
                        self.name, layer_name, i, data_uri, i)
                    if (i + 1) % cols == 0:
                        retval += """</tr><tr>"""
                retval += "</tr></table>"
                self[layer_name].feature = orig_feature
                if display:
                    return HTML(retval)
                else:
                    return retval
            else:
                orig_feature = self[layer_name].feature
                for i in range(output_shape[3]):
                    self[layer_name].feature = i
                    ## This should return in proper orientation, regardless of rotate setting:
                    image = self.propagate_to_image(layer_name, inputs, class_id=class_id,
                                                    update_pictures=update_pictures, raw=raw)
                    if resize is not None:
                        image = image.resize(resize)
                    if scale != 1.0:
                        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
                    data_uri = self._image_to_uri(image)
                    if not self._comm:
                        from ipykernel.comm import Comm
                        self._comm = Comm(target_name='conx_svg_control')
                    if self._comm.kernel:
                        self._comm.send({'class': "%s_%s_feature%s" % (self.name, layer_name, i), "src": data_uri})
                self[layer_name].feature = orig_feature
        else:
            raise Exception("layer '%s' has no features" % layer_name)

    def propagate_to_image(self, layer_name, input, batch_size=32, resize=None, scale=1.0,
                           class_id=None, update_pictures=False, raw=False, feature=None):
        """
        Gets an image of activations at a layer. Always returns image in
        proper orientation.
        """
        orig_rotate = self.config["svg_rotate"]
        self.config["svg_rotate"] = False
        if feature is not None:
            orig_feature = self[layer_name].feature
            self[layer_name].feature = feature
        image = self._propagate_to_image(layer_name, input, batch_size, resize, scale,
                                         class_id, update_pictures, raw)
        self.config["svg_rotate"] = orig_rotate
        if feature is not None:
            self[layer_name].feature = orig_feature
        return image

    def _propagate_to_image(self, layer_name, input, batch_size=32, resize=None, scale=1.0,
                            class_id=None, update_pictures=False, raw=False):
        """
        Internal version. Draws to whatever rotation is set.
        """
        if isinstance(input, dict):
            input = [input[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                input = input[0]
        elif isinstance(input, PIL.Image.Image):
            input = image_to_array(input)
        ## End of input setup
        if not is_array_like(input):
            raise Exception("inputs should be an array")
        if not isinstance(layer_name, str):
            raise Exception("layer_name should be a string")
        outputs = self.propagate_to(layer_name, input, batch_size, class_id=class_id,
                                    update_pictures=update_pictures, raw=raw)
        array = np.array(outputs)
        image = self[layer_name].make_image(array, config=self.config)
        if resize is not None:
            image = image.resize(resize)
        if scale != 1.0:
            image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
        return image

    def plot_activation_map(self, from_layer='input', from_units=(0,1), to_layer='output',
                            to_unit=0, colormap=None, default_from_layer_value=0,
                            resolution=None, act_range=(0,1), show_values=False, title=None,
                            scatter=None, symbols=None, default_symbol="o",
                            format=None, update_pictures=False):
        """
        Plot the activations at a bank/unit given two input units.
        """
        # first do some error checking
        assert self[from_layer] is not None, "unknown layer: %s" % (from_layer,)
        assert type(from_units) in (tuple, list) and len(from_units) == 2, \
            "expected a pair of ints for the %s units but got %s" % (from_layer, from_units)
        ix, iy = from_units
        assert 0 <= ix < self[from_layer].size, "no such %s layer unit: %d" % (from_layer, ix)
        assert 0 <= iy < self[from_layer].size, "no such %s layer unit: %d" % (from_layer, iy)
        assert self[to_layer] is not None, "unknown layer: %s" % (to_layer,)
        assert type(to_unit) is int, "expected an int for the %s unit but got %s" % (to_layer, to_unit)
        assert 0 <= to_unit < self[to_layer].size, "no such %s layer unit: %d" % (to_layer, to_unit)
        if colormap is None: colormap = get_colormap()
        if plt is None:
            raise Exception("matplotlib was not loaded")
        act_min, act_max = self[from_layer].get_act_minmax() if act_range is None else act_range
        out_min, out_max = self[to_layer].get_act_minmax()
        if resolution is None:
            resolution = (act_max - act_min) / 50  # 50x50 pixels by default
        xmin, xmax, xstep = act_min, act_max, resolution
        ymin, ymax, ystep = act_min, act_max, resolution
        xspan = xmax - xmin
        yspan = ymax - ymin
        xpixels = int(xspan/xstep)+1
        ypixels = int(yspan/ystep)+1
        mat = np.zeros((ypixels, xpixels))
        ovector = self[from_layer].make_dummy_vector(default_from_layer_value)
        for row in range(ypixels):
            for col in range(xpixels):
                # (x,y) corresponds to lower left corner point of pixel
                x = xmin + xstep*col
                y = ymin + ystep*row
                vector = copy.deepcopy(ovector)
                vector[ix] = x
                vector[iy] = y
                activations = self.propagate_from(from_layer, vector, to_layer, update_pictures=update_pictures)
                mat[row,col] = activations[to_unit]
        fig, ax = plt.subplots()
        axim = ax.imshow(mat, origin='lower', cmap=colormap, vmin=out_min, vmax=out_max)
        if scatter is not None:
            if isinstance(scatter, dict):
                scatter = scatter["data"]
            if len(scatter) == 2 and isinstance(scatter[0], str):
                scatter = [scatter]
            for (label, data) in scatter:
                kwargs = {}
                args = []
                xs = [min(vector[0], act_max - .01) * xpixels for vector in data]
                ys = [min(vector[1], act_max - .01) * ypixels for vector in data]
                if label:
                    kwargs["label"] = label
                symbol = get_symbol(label, symbols, default_symbol)
                if symbol:
                    args.append(symbol)
                ax.plot(xs, ys, *args, **kwargs)
            ax.legend()
        if title is not None:
            ax.set_title("Activation of %s[%s]: %s" % (to_layer, to_unit, title))
        else:
            ax.set_title("Activation of %s[%s]" % (to_layer, to_unit))
        ax.set_xlabel("%s[%s]" % (from_layer, ix))
        ax.set_ylabel("%s[%s]" % (from_layer, iy))
        ax.xaxis.tick_bottom()
        ax.set_xticks([i*(xpixels-1)/4 for i in range(5)])
        ax.set_xticklabels([xmin+i*xspan/4 for i in range(5)])
        ax.set_yticks([i*(ypixels-1)/4 for i in range(5)])
        ax.set_yticklabels([ymin+i*yspan/4 for i in range(5)])
        cbar = fig.colorbar(axim)
        if format is None:
            plt.show(block=False)
        else:
            from IPython.display import SVG
            bytes = io.BytesIO()
            if format == "svg":
                plt.savefig(bytes, format="svg")
                plt.close(fig)
                img_bytes = bytes.getvalue()
                return SVG(img_bytes.decode())
            elif format == "image":
                plt.savefig(bytes, format="png")
                plt.close(fig)
                bytes.seek(0)
                pil_image = PIL.Image.open(bytes)
                return pil_image
            else:
                raise Exception("format must be None, 'svg', or 'image'")
        # optionally print out a table of activation values
        if show_values:
            s = '\n'
            for y in np.linspace(act_max, act_min, 20):
                for x in np.linspace(act_min, act_max, 20):
                    vector = [default_from_layer_value] * self[from_layer].size
                    vector[ix] = x
                    vector[iy] = y
                    out = self.propagate_from(from_layer, vector, to_layer)[to_unit]
                    s += '%4.2f ' % out
                s += '\n'
            separator = 100 * '-'
            s += separator
            print("%s\nActivation of %s[%d] as a function of %s[%d] and %s[%d]" %
                  (separator, to_layer, to_unit, from_layer, ix, from_layer, iy))
            print("rows: %s[%d] decreasing from %.2f to %.2f" % (from_layer, iy, act_max, act_min))
            print("cols: %s[%d] increasing from %.2f to %.2f" % (from_layer, ix, act_min, act_max))
            print(s)

    def plot_layer_weights(self, layer_name, units='all', wrange=None, wmin=None, wmax=None,
                           colormap='gray', vshape=None, cbar=True, ticks=5, format=None,
                           layout=None, spacing=0.2, figsize=None, scale=None, title=None):
        """weight range displayed on the colorbar can be specified as wrange=(wmin, wmax),
        or individually via wmin/wmax keywords.  if wmin or wmax is None, the actual min/max
        value of the weight matrix is used. wrange overrides provided wmin/wmax values. ticks
        is the number of colorbar ticks displayed.  cbar=False turns off the colorbar.  units
        can be a single unit index number or a list/tuple/range of indices.
        """
        if self[layer_name] is None:
            raise Exception("unknown layer: %s" % (layer_name,))
        if units == 'all':
            units = list(range(self[layer_name].size))
        elif isinstance(units, numbers.Integral):
            units = [units]
        elif not isinstance(units, (list, tuple, range)) or len(units) == 0:
            raise Exception("units: expected an int or sequence of ints, but got %s" % (units,))
        for unit in units:
            if not 0 <= unit < self[layer_name].size:
                raise Exception("no such unit: %s" % (unit,))
        W, b = self[layer_name].keras_layer.get_weights()
        W = W.transpose()
        to_size, from_size = W.shape
        if vshape is None:
            rows, cols = 1, from_size
        elif not isinstance(vshape, (list, tuple)) or len(vshape) != 2 \
           or not isinstance(vshape[0], numbers.Integral) \
           or not isinstance(vshape[1], numbers.Integral):
            raise Exception("vshape: expected a pair of ints but got %s" % (vshape,))
        else:
            rows, cols = vshape
        if rows*cols != from_size:
            raise Exception("vshape %s is incompatible with the number of incoming weights to each %s unit (%d)"
                            % (vshape, layer_name, from_size))
        aspect_ratio = max(rows,cols)/min(rows,cols)
        #print("aspect_ratio is", aspect_ratio)
        if aspect_ratio > 50:   # threshold may need further refinement
            print("WARNING: using a visual display shape of (%d, %d), which may be hard to see."
                  % (rows, cols), file=sys.stderr)
            print("You can use vshape=(rows, cols) to specify a different display shape.")
        if not isinstance(wmin, (numbers.Number, type(None))):
            raise Exception("wmin: expected a number or None but got %s" % (wmin,))
        if not isinstance(wmax, (numbers.Number, type(None))):
            raise Exception("wmax: expected a number or None but got %s" % (wmax,))
        if wrange is None:
            if wmin is None:
                wmin = np.min(W)
                wmin_label = '0' if wmin == 0 else '%+.2f' % (wmin,)
            else:
                wmin_label = r'$\leq$ 0' if wmin == 0 else r'$\leq$ %+.2f' % (wmin,)
            if wmax is None:
                wmax = np.max(W)
                wmax_label = '0' if wmax == 0 else '%+.2f' % (wmax,)
            else:
                wmax_label = r'$\geq$ 0' if wmax == 0 else r'$\geq$ %+.2f' % (wmax,)
        elif not isinstance(wrange, (list, tuple)) or len(wrange) != 2 \
             or not isinstance(wrange[0], (numbers.Number, type(None))) \
             or not isinstance(wrange[1], (numbers.Number, type(None))):
            raise Exception("wrange: expected a pair of numbers but got %s" % (wrange,))
        else: # wrange overrides provided wmin/wmax values
            wmin, wmax = wrange
            return self.plot_layer_weights(layer_name, units=units, wrange=None, wmin=wmin,
                                           wmax=wmax, colormap=colormap, vshape=vshape,
                                           cbar=cbar, ticks=ticks, format=format, layout=layout,
                                           spacing=spacing, figsize=figsize, scale=scale,
                                           title=title)
        if wmin >= wmax:
            raise Exception("specified weight range is empty")
        if not isinstance(ticks, numbers.Integral) or ticks < 2:
            raise Exception("invalid number of colorbar ticks: %s" % (ticks,))
        # clip weights to the range [wmin, wmax] and normalize to [0, 1]:
        scaled_W = (np.clip(W, wmin, wmax) - wmin) / (wmax - wmin)

        if not 0 <= spacing <= 1:
            raise Exception("spacing must be between 0 and 1")
            return
        if scale is None:
            scale = 1
        elif scale <= 0:
            raise Exception("scale must be an int > 0")
        if layout is None:
            layout = (1, len(units))
        layout_rows, layout_cols = layout
        border = spacing / max(layout_rows, layout_cols)
        if figsize is None:
            size_factor = 2.5
            width = min(10, size_factor*(layout_cols+1)*scale)
            height = min(8, size_factor*layout_rows*scale)
            figsize = (width, height)
        fig, axes = plt.subplots(layout_rows, layout_cols, squeeze=False,
                                 figsize=figsize, num=title,
                                 gridspec_kw={'wspace': spacing,
                                              'hspace': spacing,
                                              'left': border,
                                              'right': 1-border,
                                              'bottom': border,
                                              'top': 1-border})
        if title is not None:
            fig.canvas.set_window_title(title)
        for ax in axes.reshape(axes.size):
            ax.axis('off')
        k = 0
        for r in range(layout_rows):
            for c in range(layout_cols):
                if k < len(units):
                    u = units[k]
                    axes[r][c].set_title('%s[%d]' % (layer_name, u))
                    axes[r][c].title.set_fontsize(8)
                    im = scaled_W[u,:].reshape((rows, cols))
                    axim = axes[r][c].imshow(im, cmap=colormap, vmin=0, vmax=1)
                    k += 1
        if k < len(units):
            print("WARNING: could not plot all requested weights with layout %s" % (layout,),
                  file=sys.stderr)
        if cbar:
            tick_locations = np.linspace(0, 1, ticks)
            tick_values = tick_locations * (wmax - wmin) + wmin
            s = 0.5 if layout_rows > 3 else 0.75 if 2 <= layout_rows <= 3 else 1
            colorbar = fig.colorbar(axim, ax=axes, ticks=tick_locations, shrink=s)
            cbar_labels = ['0' if t == 0 else '%+.2f' % (t,) for t in tick_values]
            cbar_labels[0] = wmin_label
            cbar_labels[-1] = wmax_label
            colorbar.ax.tick_params(labelsize=8)
            colorbar.ax.set_yticklabels(cbar_labels)
        if format is None:
            plt.show(block=False)
        else:
            from IPython.display import SVG
            bytes = io.BytesIO()
            if format == "svg":
                plt.savefig(bytes, format="svg")
                plt.close(fig)
                img_bytes = bytes.getvalue()
                return SVG(img_bytes.decode())
            elif format == "image":
                plt.savefig(bytes, format="png")
                plt.close(fig)
                bytes.seek(0)
                pil_image = PIL.Image.open(bytes)
                return pil_image
            else:
                raise Exception("format must be None, 'svg', or 'image'")

    def show_unit_weights(self, layer_name, unit, vshape=None, ascii=False):
        if self[layer_name] is None:
            raise Exception("unknown layer: %s" % (layer_name,))
        W, b = self[layer_name].keras_layer.get_weights()
        W = W.transpose()
        to_size, from_size = W.shape
        if vshape is None:
            rows, cols = 1, from_size
        elif not isinstance(vshape, (list, tuple)) or len(vshape) != 2 \
           or not isinstance(vshape[0], numbers.Integral) \
           or not isinstance(vshape[1], numbers.Integral):
            raise Exception("vshape: expected a pair of ints but got %s" % (vshape,))
        else:
            rows, cols = vshape
        if rows*cols != from_size:
            raise Exception("vshape %s is incompatible with the number of incoming weights to each %s unit (%d)"
                            % (vshape, layer_name, from_size))
        weights = W[unit].reshape((rows,cols))
        for r in range(rows):
            for c in range(cols):
                w = weights[r][c]
                if ascii:
                    ch = ' ' if w <= 0 else '.' if w < 0.50 else 'o' if w < 0.75 else '@'
                    print(ch, end=" ")
                else:
                    print('%5.2f' % (w,), end=" ")
            print()

    def get_metrics(self):
        """
        Returns a list of the metrics available in the Network's history.

        """
        metrics = set()
        for epoch in self.history:
            metrics = metrics.union(set(epoch.keys()))
        return sorted(metrics)

    def get_metric(self, metric):
        """
        Returns the metric data from the network's history.

        >>> net = Network("Test", 2, 2, 1)
        >>> net.get_metric("loss")
        []
        """
        return [epoch[metric] if metric in epoch else None for epoch in self.history]

    def plot(self, metrics=None, ymin=None, ymax=None, start=0, end=None, legend='best',
             label=None, symbols=None, default_symbol="-", title=None, return_fig_ax=False, fig_ax=None,
             format=None):
        """Plots the current network history for the specific epoch range and
        metrics. metrics is '?', 'all', a metric keyword, or a list of metric keywords.
        if metrics is None, loss and accuracy are plotted on separate graphs.

        >>> net = Network("Plot Test", 1, 3, 1)
        >>> net.compile(error="mse", optimizer="rmsprop")
        >>> net.dataset.append([0.0], [1.0])
        >>> net.dataset.append([1.0], [0.0])
        >>> net.train(plot=False)  # doctest: +ELLIPSIS
        Evaluating initial training metrics...
        Training...
        ...
        >>> net.plot('?')
        Available metrics: acc, loss
        """
        ## https://matplotlib.org/api/markers_api.html
        ## https://matplotlib.org/api/colors_api.html
        if isinstance(ymin, str):
            raise Exception("Network.plot() should be called with a metric, or list of metrics")
        if len(self.history) == 0:
            print("No history available")
            return
        available_metrics = self.get_metrics()
        if metrics is None:
            metrics = ['loss']
        elif metrics is '?':
            print("Available metrics:", ", ".join(available_metrics))
            return
        elif metrics == 'all':
            metrics = available_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif isinstance(metrics, (list, tuple)):
            pass
        else:
            print("metrics: expected a list or a string but got %s" % (metrics,))
            return
        ## Check metrics, and expand regular expressions:
        proposed_metrics = metrics
        metrics = []
        for metric in proposed_metrics:
            for available_metric in available_metrics:
                if re.match(metric, available_metric) is not None:
                    metrics.append(available_metric)
        if fig_ax:
            fig, ax = fig_ax
        else:
            fig, ax = plt.subplots(1)
        x_values = range(self.epoch_count+1)
        x_values = x_values[start:end]
        ax.set_xlabel('Epoch')
        data_found = False
        for metric in metrics:
            y_values = self.get_metric(metric)
            y_values = y_values[start:end]
            if y_values.count(None) == len(y_values):
                print("WARNING: No %s data available for the specified epochs (%s-%s)" % (metric, start, end), file=sys.stderr)
            else:
                next_label = label if label else metric
                symbol = get_symbol(label, symbols, default_symbol)
                ax.plot(x_values, y_values, symbol, label=next_label)
                data_found = True
        if not data_found:
            if return_fig_ax:
                return (fig, ax)
            else:
                plt.close(fig)
            return
        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)
        if legend is not None:
            plt.legend(loc=legend)
        if title is None:
            title = self.name
        plt.title(title)
        if return_fig_ax:
            return (fig, ax)
        elif format is None:
            plt.show(block=False)
        else:
            from IPython.display import SVG
            bytes = io.BytesIO()
            if format == "svg":
                plt.savefig(bytes, format="svg")
                plt.close(fig)
                img_bytes = bytes.getvalue()
                return SVG(img_bytes.decode())
            elif format == "image":
                plt.savefig(bytes, format="png")
                plt.close(fig)
                bytes.seek(0)
                pil_image = PIL.Image.open(bytes)
                return pil_image
            else:
                raise Exception("format must be None, 'svg', or 'image'")

    def show_results(self, report_rate=None):
        """
        Show the history of training results. If report_rate is given
        use that, else, try to use the last trained report_rate.
        """
        report_rate = (report_rate
                       if report_rate is not None
                       else self.train_options.get("report_rate", 1))
        self._need_to_show_headings = True
        for epoch_count in range(0, len(self.history), report_rate):
            results = self.history[epoch_count]
            self.report_epoch(epoch_count, results)
        if len(self.history) > 0:
            print("=" * 56)
            self.report_epoch(len(self.history) - 1, self.history[-1])

    def plot_results(self, callback=None, format=None):
        """plots loss and accuracy on separate graphs, ignoring any other metrics"""
        #print("called on_epoch_end with epoch =", epoch)
        metrics = self.get_metrics()
        if callback is not None and callback.figure is not None:
            # figure and axes objects have already been created
            fig, loss_ax, acc_ax = callback.figure
            loss_ax.clear()
            if acc_ax is not None:
                acc_ax.clear()
        else:
            # first time called, so create figure and axes objects
            if 'acc' in metrics or 'val_acc' in metrics:
                fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10,4))
            else:
                fig, loss_ax = plt.subplots(1)
                acc_ax = None
            if callback is not None:
                callback.figure = fig, loss_ax, acc_ax
        x_values = range(self.epoch_count+1)
        for metric in metrics:
            y_values = self.get_metric(metric)
            if metric == 'loss':
                loss_ax.plot(x_values, y_values, label='Training set')
            elif metric == 'val_loss':
                loss_ax.plot(x_values, y_values, label='Validation set')
            elif metric == 'acc' and acc_ax is not None:
                acc_ax.plot(x_values, y_values, label='Training set')
            elif metric == 'val_acc' and acc_ax is not None:
                acc_ax.plot(x_values, y_values, label='Validation set')
        loss_ax.set_ylim(bottom=0)
        loss_ax.set_title("%s: Error" % (self.name,))
        loss_ax.set_xlabel('Epoch')
        loss_ax.legend(loc='best')
        if acc_ax is not None:
            acc_ax.set_ylim([-0.1, 1.1])
            acc_ax.set_title("%s: Accuracy" % (self.name,))
            acc_ax.set_xlabel('Epoch')
            acc_ax.legend(loc='best')
        if (callback is not None and not callback.in_console) or format == "svg":
            from IPython.display import SVG, clear_output, display
            bytes = io.BytesIO()
            plt.savefig(bytes, format='svg')
            img_bytes = bytes.getvalue()
            clear_output(wait=True)
            display(SVG(img_bytes.decode()))
            #return SVG(img_bytes.decode())
        else: # format is None
            plt.pause(0.01)
            #plt.show(block=False)

    def compile(self, **kwargs):
        """
        Check and compile the network.

        You must provide error/loss and optimizer keywords.

        Possible error/loss functions are:
            * 'mse' - mean_squared_error
            * 'mae' - mean_absolute_error
            * 'mape' - mean_absolute_percentage_error
            * 'msle' - mean_squared_logarithmic_error
            * 'kld' - kullback_leibler_divergence
            * 'cosine' - cosine_proximity

        Possible optimizers are:
            * 'sgd'
            * 'rmsprop'
            * 'adagrad'
            * 'adadelta'
            * 'adam'
            * 'adamax'
            * 'nadam'

        See https://keras.io/ `Model.compile` method for more details.
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
        if kwargs["loss"] == 'sparse_categorical_crossentropy':
            raise Exception("'sparse_categorical_crossentropy' is not a valid error metric in conx; use 'categorical_crossentropy' with proper targets")
        if "optimizer" not in kwargs or "loss" not in kwargs:
            raise Exception("both optimizer and error/loss are required to compile a network")
        if isinstance(kwargs["optimizer"], str) and kwargs["optimizer"].lower() not in self.OPTIMIZERS:
            raise Exception("invalid optimizer '%s'; use valid function or one of %s" % (kwargs["optimizer"], Network.OPTIMIZERS,))
        ## Build an optimizer:
        config = kwargs.get("config", {})
        for kw in list(kwargs.keys()):
            if kw not in ["loss", "metrics", "optimizer",
                          "loss_weights", "sample_weight_mode",
                          "weighted_metrics", "target_tensors"]:
                if kw != "config":
                    config[kw] = kwargs[kw]
                del kwargs[kw]
        if config != {}:
            error = False
            try:
                kwargs["optimizer"] = keras.optimizers.get({"class_name": kwargs["optimizer"], "config": config})
            except:
                error = True
            if error:
                class_instance = keras.optimizers.get(kwargs["optimizer"])
                raise Exception("invalid optimizer arguments %s(**%s); for more information type: help(cx.%s)" % (
                    kwargs["optimizer"], config, class_instance.__class__.__name__))
        ### Optimizer is an instance, if given kwargs
        using_softmax = False
        for layer in self.layers:
            if layer.kind() == "output":
                if layer.activation is not None and layer.activation == "softmax":
                    using_softmax = True
                    if "crossentropy" not in kwargs["loss"]:
                        print("WARNING: you are using the 'softmax' activation function on layer '%s'" % layer.name, file=sys.stderr)
                        print("         but not using a 'crossentropy' error measure.", file=sys.stderr)
                if "crossentropy" in kwargs["loss"]:
                    if layer.activation is not None and layer.activation != "softmax":
                        print("WARNING: you are using a crossentropy error measure", file=sys.stderr)
                        print("         but not using the 'softmax' activation function on layer '%s'"
                              % layer.name, file=sys.stderr)
        self._build_intermediary_models()
        output_k_layers = self._get_output_ks_in_order()
        input_k_layers = self._get_input_ks_in_order(self.input_bank_order)
        self.model = keras.models.Model(inputs=input_k_layers, outputs=output_k_layers)
        if "metrics" in kwargs and kwargs["metrics"] is not None:
            pass ## ok allow override
        elif using_softmax: ## let's use Keras' default acc function
            kwargs['metrics'] = ["acc"] ## Keras' default
            if "tolerance" in kwargs:
                print("WARNING: using softmax activation function; tolerance is ignored", file=sys.stderr)
        else:
            kwargs['metrics'] = [self.acc] ## Conx's default
        self.compile_options = copy.copy(kwargs)
        self.model.compile(**kwargs)
        # set each conx layer to point to corresponding keras model layer
        for layer in self.layers:
            layer.keras_layer = self._find_keras_layer(layer.name)

    def acc(self, targets, outputs):
        # This is only used on non-multi-output-bank training:
        return K.mean(K.all(K.less_equal(K.abs(targets - outputs), self._tolerance), axis=-1), axis=-1)

    def _find_keras_layer(self, layer_name):
        """
        Find the associated keras layer.
        """
        return [x for x in self.model.layers if x.name == layer_name][0]

    def _delete_intermediary_models(self):
        """
        Remove these, as they don't pickle.
        """
        for layer in self.layers:
            layer.k = None
            layer.input_names = set([])
            layer.model = None

    def update_model(self):
        """
        Useful if you change, say, an activation function after training.
        """
        self._build_intermediary_models()

    def _build_intermediary_models(self):
        """
        Construct the layer.k, layer.input_names, and layer.model's.
        """
        self.prop_from_dict.clear()
        self.keras_functions.clear()
        sequence = topological_sort(self, self.layers)
        if self.debug: print("topological sort:", [l.name for l in sequence])
        for layer in sequence:
            if layer.kind() == 'input':
                if self.debug: print("making input layer for", layer.name)
                layer.k = layer.make_input_layer_k()
                layer.input_names = set([layer.name])
                layer.model = keras.models.Model(inputs=layer.k, outputs=layer.k) # identity
                self.prop_from_dict[(layer.name, layer.name)] = layer.model
            else:
                if self.debug: print("making layer for", layer.name)
                if len(layer.incoming_connections) == 0:
                    raise Exception("non-input layer '%s' with no incoming connections" % layer.name)
                kfuncs = layer.make_keras_functions()
                self.keras_functions[layer.name] = kfuncs
                if len(layer.incoming_connections) == 1:
                    if self.debug: print("single input", layer.incoming_connections[0])
                    k = layer.incoming_connections[0].k
                    layer.input_names = layer.incoming_connections[0].input_names
                else: # multiple inputs, some type of merge:
                    if self.debug: print("Merge detected!", [l.name for l in layer.incoming_connections])
                    if layer.handle_merge:
                        k = layer.make_keras_function()
                    else:
                        k = keras.layers.Concatenate()([incoming.k for incoming in layer.incoming_connections])
                    # flatten:
                    layer.input_names = set([item for sublist in
                                             [incoming.input_names for incoming in layer.incoming_connections]
                                             for item in sublist])
                if self.debug: print("input names for", layer.name, layer.input_names)
                if self.debug: print("applying k's", kfuncs)
                for f in kfuncs:
                    k = f(k)
                layer.k = k
                ## get the inputs to this branch, in order:
                input_ks = self._get_input_ks_in_order(layer.input_names)
                ## From all inputs to this layer:
                layer.model = keras.models.Model(inputs=input_ks, outputs=layer.k)
        ## Build all prop_from models:
        if self.build_propagate_from_models:
            for in_layer_name in self.input_bank_order:
                for out_layer_name in self.output_bank_order:
                    layer = self[out_layer_name]
                    if (in_layer_name, layer.name) in self.prop_from_dict:
                        continue
                    if self.debug: print("from %s to %s" % (in_layer_name, layer.name))
                    all_paths = find_all_paths(self, self[in_layer_name], layer)
                    for path in all_paths:
                        abort_path = False
                        for i in range(len(path) - 1):
                            if abort_path:
                                break
                            path_layer = path[i]
                            if (path_layer.name, layer.name) in self.prop_from_dict:
                                continue
                            if self.debug: print("   %s to %s" % (path_layer.name, layer.name))
                            if path_layer.shape is None:
                                ## Skips FlattenLayer, Concat, etc. as from_layer
                                if self.debug: print("   aborting this path; try next")
                                continue
                            k = starting_k = keras.layers.Input(path_layer.shape, name=path_layer.name)
                            rest_of_path = path[i + 1:]
                            for rest_of_path_layer in rest_of_path:
                                if abort_path:
                                    break
                                if self.debug: print("      %s to %s" % (path_layer.name, rest_of_path_layer.name))
                                kfuncs = self.keras_functions[rest_of_path_layer.name]
                                for f in kfuncs:
                                    try:
                                        k = f(k)
                                    except:
                                        ## Can't make this pathway; probably a merge
                                        abort_path = True
                                        break
                                ## FIXME: could be multiple paths
                                self.prop_from_dict[
                                    (path_layer.name, rest_of_path_layer.name)
                                ] = keras.models.Model(inputs=starting_k, outputs=k)

    def _get_input_ks_in_order(self, layer_names):
        """
        Get the Keras function for each of a set of layer names.
        [in3, in4] sorted by input bank ordering
        """
        sorted_layer_names = self._get_sorted_input_names(set(layer_names))
        layer_ks = [self[layer_name].k for layer_name in sorted_layer_names]
        if len(layer_ks) == 1:
            layer_ks = layer_ks[0]
        return layer_ks

    def _get_sorted_input_names(self, layer_names):
        """
        Given a set of input names, give them back in order.
        """
        return [name for (index, name) in sorted([(self.input_bank_order.index(name), name)
                                                  for name in layer_names])]

    def _get_output_ks_in_order(self):
        """
        Get the Keras function for each output layer, in order.
        """
        layer_ks = [self[layer_name].k for layer_name in self.output_bank_order]
        if len(layer_ks) == 1:
            layer_ks = layer_ks[0]
        return layer_ks

    def _image_to_uri(self, img_src):
        # Convert to binary data:
        b = io.BytesIO()
        try:
            img_src.save(b, format='gif')
        except:
            return ""
        data = b.getvalue()
        data = base64.b64encode(data)
        if not isinstance(data, str):
            data = data.decode("latin1")
        return "data:image/gif;base64,%s" % html.escape(data)

    def vshape(self, layer_name):
        """
        Find the vshape of layer.
        """
        layer = self[layer_name]
        vshape = layer.vshape if layer.vshape else layer.shape if layer.shape else None
        if vshape is None:
            vshape = layer.get_output_shape()
        return vshape

    def _pre_process_struct(self, inputs, config, ordering):
        """
        Determine sizes and pre-compute images.
        """
        ### find max_width, image_dims, and row_height
        # Go through and build images, compute max_width:
        row_heights = []
        max_width = 0
        max_height = 0
        images = {}
        image_dims = {}
        #######################################################################
        ## For each level:
        #######################################################################
        hiding = {}
        for level_tups in ordering: ## output to input:
            # first make all images at this level
            row_width = 0 # for this row
            row_height = 0 # for this row
            #######################################################################
            ## For each column:
            #######################################################################
            for column in range(len(level_tups)):
                (layer_name, anchor, fname) = level_tups[column]
                if not self[layer_name].visible:
                    if not hiding.get(column, False):
                        row_height = max(row_height, config["vspace"]) # space for hidden indicator
                    hiding[column] = True # in the middle of hiding some layers
                    row_width += config["hspace"] # space between
                    max_width = max(max_width, row_width) # of all rows
                    continue
                elif anchor:
                    # No need to handle anchors here
                    # as they occupy no vertical space
                    hiding[column] = False
                    # give it some hspace for this column
                    # in case there is nothing else in this column
                    row_width += config["hspace"]
                    max_width = max(max_width, row_width)
                    continue
                hiding[column] = False
                #######################################################################
                ## The rest of this for loop is handling image of bank
                #######################################################################
                if inputs is not None:
                    v = inputs
                elif len(self.dataset.inputs) > 0:
                    v = self.dataset.inputs[0]
                else:
                    if self.num_input_layers > 1:
                        v = []
                        for in_name in self.input_bank_order:
                            v.append(self[in_name].make_dummy_vector())
                    else:
                        in_layer = [layer for layer in self.layers if layer.kind() == "input"][0]
                        v = in_layer.make_dummy_vector()
                if self[layer_name].model:
                    try:
                        orig_svg_rotate = self.config["svg_rotate"]
                        self.config["svg_rotate"] = config["svg_rotate"]
                        image = self._propagate_to_image(layer_name, v)
                        self.config["svg_rotate"] = orig_svg_rotate
                    except:
                        image = self[layer_name].make_image(np.array(self[layer_name].make_dummy_vector()), config=config)
                else:
                    image = self[layer_name].make_image(np.array(self[layer_name].make_dummy_vector()), config=config)
                (width, height) = image.size
                images[layer_name] = image ## little image
                ### Layer settings:
                if self[layer_name].image_maxdim:
                    image_maxdim = self[layer_name].image_maxdim
                else:
                    image_maxdim = config["image_maxdim"]
                if self[layer_name].image_pixels_per_unit:
                    image_pixels_per_unit = self[layer_name].image_pixels_per_unit
                else:
                    image_pixels_per_unit = config["image_pixels_per_unit"]
                ## First, try based on shape:
                #pwidth, pheight = np.array(image.size) * image_pixels_per_unit
                vshape = self.vshape(layer_name)
                if vshape is None or self[layer_name].keep_aspect_ratio:
                    pass ## let the image set the shape
                elif len(vshape) == 1:
                    if vshape[0] is not None:
                        width = vshape[0] * image_pixels_per_unit
                        height = image_pixels_per_unit
                elif len(vshape) >= 2:
                    if vshape[0] is not None:
                        height = vshape[0] * image_pixels_per_unit
                        if vshape[1] is not None:
                            width = vshape[1] * image_pixels_per_unit
                    else:
                        if len(vshape) > 2:
                            if vshape[1] is not None:
                                height = vshape[1] * image_pixels_per_unit
                                width = vshape[2] * image_pixels_per_unit
                        elif vshape[1] is not None: # flatten
                            width = vshape[1] * image_pixels_per_unit
                            height = image_pixels_per_unit
                ## keep aspect ratio:
                if self[layer_name].keep_aspect_ratio:
                    scale = image_maxdim / max(width, height)
                    image = image.resize((int(width * scale), int(height * scale)))
                    width, height = image.size
                else:
                    ## Change aspect ratio if too big/small
                    if width < image_pixels_per_unit:
                        width = image_pixels_per_unit
                    if height < image_pixels_per_unit:
                        height = image_pixels_per_unit
                    ## make sure not too big:
                    if height > image_maxdim:
                        height = image_maxdim
                    if width > image_maxdim:
                        width = image_maxdim
                image_dims[layer_name] = (width, height)
                row_width += width + config["hspace"] # space between
                row_height = max(row_height, height)
            row_heights.append(row_height)
            max_width = max(max_width, row_width) # of all rows
        return max_width, max_height, row_heights, images, image_dims

    def _find_spacing(self, row, ordering, max_width):
        """
        Find the spacing for a row number
        """
        return max_width / (len(ordering[row]) + 1)

    def build_struct(self, inputs, class_id, config):
        ordering = list(reversed(self._get_level_ordering())) # list of names per level, input to output
        max_width, max_height, row_heights, images, image_dims = self._pre_process_struct(inputs, config, ordering)
        ### Now that we know the dimensions:
        struct = []
        cheight = config["border_top"] # top border
        #######################################################################
        ## Display targets?
        #######################################################################
        if config["show_targets"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # draw the row of targets:
            cwidth = 0
            for (layer_name, anchor, fname) in ordering[0]: ## no anchors in output
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                cwidth += (spacing - width/2)
                struct.append(["image_svg", {"name": layer_name + "_targets",
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
                                             "rw": width + 2}])
                ## show a label
                struct.append(["label_svg", {"x": cwidth + width + 5,
                                             "y": cheight + height/2 + 2,
                                             "label": "targets",
                                             "font_size": config["font_size"],
                                             "font_color": "black",
                                             "font_family": config["font_family"],
                                             "text_anchor": "start",
                }])
                cwidth += width/2
            ## Then we need to add height for output layer again, plus a little bit
            cheight += row_heights[0] + 10 # max height of row, plus some
        #######################################################################
        ## Display error?
        #######################################################################
        if config["show_errors"]:
            spacing = self._find_spacing(0, ordering, max_width)
            # draw the row of errors:
            cwidth = 0
            for (layer_name, anchor, fname) in ordering[0]: # no anchors in output
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
                cwidth += (spacing - (width/2))
                struct.append(["image_svg", {"name": layer_name + "_errors",
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
                                             "rw": width + 2}])
                ## show a label
                struct.append(["label_svg", {"x": cwidth + width + 5,
                                             "y": cheight + height/2 + 2,
                                             "label": "errors",
                                             "font_size": config["font_size"],
                                             "font_color": "black",
                                             "font_family": config["font_family"],
                                             "text_anchor": "start",
                }])
                cwidth += width/2
            ## Then we need to add height for output layer again, plus a little bit
            cheight += row_heights[0] + 10 # max height of row, plus some
        #######################################################################
        ## Show a separator that takes no space between output and targets/errors
        #######################################################################
        if config["show_errors"] or config["show_targets"]:
            spacing = self._find_spacing(0, ordering, max_width)
            ## Draw a line for each column in putput:
            cwidth = spacing/2 + spacing/2 # border + middle of first column
            # number of columns:
            for level_tups in ordering[0]:
                struct.append(["line_svg", {"x1":cwidth - spacing/2,
                                            "y1":cheight - 5, # half the space between them
                                            "x2":cwidth + spacing/2,
                                            "y2":cheight - 5,
                                            "arrow_color": "green",
                                            "tooltip": "",
                }])
                cwidth += spacing
        #######################################################################
        # Now we go through again and build SVG:
        #######################################################################
        positioning = {}
        level_num = 0
        #######################################################################
        ## For each level:
        #######################################################################
        hiding = {}
        for row in range(len(ordering)):
            level_tups = ordering[row]
            ## how many space at this level for this column?
            spacing = self._find_spacing(row, ordering, max_width)
            cwidth = 0
            # See if there are any connections up:
            any_connections_up = False
            for (layer_name, anchor, fname) in level_tups:
                if not self[layer_name].visible:
                    continue
                elif anchor:
                    continue
                for out in self[layer_name].outgoing_connections:
                    if out.name not in positioning:  ## is it drawn yet? if not, continue,
                        ## if yes, we need vertical space for arrows
                        continue
                    any_connections_up = True
            if any_connections_up:
                cheight += config["vspace"] # for arrows
            else: # give a bit of room:
                ## FIXME: determine if there were spaces drawn last layer
                ## Right now, just skip any space at all
                ## cheight += 5
                pass
            row_height = 0 # for row of images
            #######################################################################
            # Draw each column:
            #######################################################################
            for  column in range(len(level_tups)):
                (layer_name, anchor, fname) = level_tups[column]
                if not self[layer_name].visible:
                    if not hiding.get(column, False): # not already hiding, add some space:
                        struct.append(["label_svg", {"x": cwidth + spacing - 80, ## center the text
                                                     "y": cheight + 15,
                                                     "label": "[layer(s) not visible]",
                                                     "font_size": config["font_size"],
                                                     "font_color": "green",
                                                     "font_family": config["font_family"],
                                                     "text_anchor": "start",
                                                     "rotate": False,
                        }])
                        row_height = max(row_height, config["vspace"])
                    hiding[column] = True
                    cwidth += spacing # leave full column width
                    continue
                ## end run of hiding
                hiding[column] = False
                #######################################################################
                ## Anchor
                #######################################################################
                if anchor:
                    anchor_name = "%s-%s-anchor%s" % (layer_name, fname, level_num)
                    cwidth += spacing
                    positioning[anchor_name] = {"x": cwidth, "y": cheight + row_heights[row]}
                    x1 = cwidth
                    ## now we are at an anchor. Is the thing that it anchors in the
                    ## lower row? level_num is increasing
                    prev = [(oname, oanchor, lfname) for (oname, oanchor, lfname) in ordering[level_num - 1] if
                            (((layer_name == oname) and (oanchor is False)) or
                             ((layer_name == oname) and (oanchor is True) and (fname == lfname)))]
                    if prev:
                        tooltip = html.escape(self.describe_connection_to(self[fname], self[layer_name]))
                        if prev[0][1]: # anchor
                            anchor_name2 = "%s-%s-anchor%s" % (layer_name, fname, level_num - 1)
                            ## draw a line to this anchor point
                            x2 = positioning[anchor_name2]["x"]
                            y2 = positioning[anchor_name2]["y"]
                            struct.append(["line_svg", {"x1":cwidth,
                                                        "y1":cheight,
                                                        "x2":x2,
                                                        "y2":y2,
                                                        "arrow_color": config["arrow_color"],
                                                        "tooltip": tooltip
                            }])
                            struct.append(["line_svg", {"x1":cwidth,
                                                        "y1":cheight + row_heights[row],
                                                        "x2":cwidth,
                                                        "y2":cheight,
                                                        "arrow_color": config["arrow_color"],
                                                        "tooltip": tooltip
                            }])
                        else:
                            ## draw a line to this bank
                            x2 = positioning[layer_name]["x"] + positioning[layer_name]["width"]/2
                            y2 = positioning[layer_name]["y"] + positioning[layer_name]["height"]
                            tootip ="TODO"
                            struct.append(["arrow_svg", {"x1":cwidth,
                                                         "y1":cheight,
                                                         "x2":x2,
                                                         "y2":y2,
                                                         "arrow_color": config["arrow_color"],
                                                         "tooltip": tooltip
                            }])
                            struct.append(["line_svg",  {"x1":cwidth,
                                                         "y1":cheight + row_heights[row],
                                                         "x2":cwidth,
                                                         "y2":cheight,
                                                         "arrow_color": config["arrow_color"],
                                                         "tooltip": tooltip
                            }])
                    else:
                        print("that's weird!", layer_name, "is not in", prev)
                    continue
                else:
                    #######################################################################
                    ## Bank positioning
                    #######################################################################
                    image = images[layer_name]
                    (width, height) = image_dims[layer_name]
                    cwidth += (spacing - (width/2))
                    positioning[layer_name] = {"name": layer_name + ("-rotated" if config["svg_rotate"] else ""),
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
                #######################################################################
                ## Arrows going up
                #######################################################################
                for out in self[layer_name].outgoing_connections:
                    if out.name not in positioning:
                        continue
                    # draw an arrow between layers:
                    anchor_name = "%s-%s-anchor%s" % (out.name, layer_name, level_num - 1)
                    ## Don't draw this error, if there is an anchor in the next level
                    if anchor_name in positioning:
                        tooltip = html.escape(self.describe_connection_to(self[layer_name], out))
                        x2 = positioning[anchor_name]["x"]
                        y2 = positioning[anchor_name]["y"]
                        struct.append(["line_svg", {"x1":x1,
                                                    "y1":y1,
                                                    "x2":x2,
                                                    "y2":y2,
                                                    "arrow_color": config["arrow_color"],
                                                    "tooltip": tooltip
                        }])
                        continue
                    else:
                        tooltip = html.escape(self.describe_connection_to(self[layer_name], out))
                        x2 = positioning[out.name]["x"] + positioning[out.name]["width"]/2
                        y2 = positioning[out.name]["y"] + positioning[out.name]["height"]
                        struct.append(["arrow_svg", {"x1":x1,
                                                     "y1":y1,
                                                     "x2":x2,
                                                     "y2":y2 + 2,
                                                     "arrow_color": config["arrow_color"],
                                                     "tooltip": tooltip
                        }])
                #######################################################################
                ## Bank images
                #######################################################################
                struct.append(["image_svg", positioning[layer_name]])
                struct.append(["label_svg", {"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] + 5,
                                             "y": positioning[layer_name]["y"] + positioning[layer_name]["height"]/2 + 2,
                                             "label": layer_name,
                                             "font_size": config["font_size"],
                                             "font_color": "black",
                                             "font_family": config["font_family"],
                                             "text_anchor": "start",
                }])
                output_shape = self[layer_name].get_output_shape()
                if (isinstance(output_shape, tuple) and len(output_shape) == 4 and
                    self[layer_name].__class__.__name__ != "ImageLayer"):
                    features = str(output_shape[3])
                    feature = str(self[layer_name].feature)
                    if config["svg_rotate"]:
                        struct.append(["label_svg", {"x": positioning[layer_name]["x"] + 5,
                                                     "y": positioning[layer_name]["y"] - 10 - 5,
                                                     "label": features,
                                                     "font_size": config["font_size"],
                                                     "font_color": "black",
                                                     "font_family": config["font_family"],
                                                     "text_anchor": "start",
                        }])
                        struct.append(["label_svg", {"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] - 10,
                                                     "y": positioning[layer_name]["y"] + positioning[layer_name]["height"] + 10 + 5,
                                                     "label": feature,
                                                     "font_size": config["font_size"],
                                                     "font_color": "black",
                                                     "font_family": config["font_family"],
                                                     "text_anchor": "start",
                        }])
                    else:
                        struct.append(["label_svg", {"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] + 5,
                                                     "y": positioning[layer_name]["y"] + 5,
                                                     "label": features,
                                                     "font_size": config["font_size"],
                                                     "font_color": "black",
                                                     "font_family": config["font_family"],
                                                     "text_anchor": "start",
                        }])
                        struct.append(["label_svg", {"x": positioning[layer_name]["x"] - (len(feature) * 7) - 5 - 5,
                                                     "y": positioning[layer_name]["y"] + positioning[layer_name]["height"] - 5,
                                                     "label": feature,
                                                     "font_size": config["font_size"],
                                                     "font_color": "black",
                                                     "font_family": config["font_family"],
                                                     "text_anchor": "start",
                        }])
                if (self[layer_name].dropout > 0):
                    label = "&#10683;"
                    struct.append(["label_svg", {"x": positioning[layer_name]["x"] - len(label) * 2.0 - 5,
                                                 "y": positioning[layer_name]["y"] + 5,
                                                 "label": label,
                                                 "font_size": config["font_size"] * 2.0,
                                                 "font_color": "black",
                                                 "font_family": config["font_family"],
                                                 "text_anchor": "start",
                    }])
                cwidth += width/2
                row_height = max(row_height, height)
                self._svg_counter += 1
            cheight += row_height
            level_num += 1
        cheight += config["border_bottom"]
        ### DONE!
        ## Draw live/static sign
        if (class_id is None):
            label = "*" # lightning bold, dynamic image
            if config["svg_rotate"]:
                struct.append(["label_svg", {"x": 10,
                                             "y": cheight - 10,
                                             "label": label,
                                             "font_size": config["font_size"] * 2.0,
                                             "font_color": "red",
                                             "font_family": config["font_family"],
                                             "text_anchor": "middle",
                }])
            else:
                struct.append(["label_svg", {"x": 10,
                                             "y": 10,
                                             "label": label,
                                             "font_size": config["font_size"] * 2.0,
                                             "font_color": "red",
                                             "font_family": config["font_family"],
                                             "text_anchor": "middle",
                }])
        ## Draw the title:
        if config["svg_rotate"]:
            struct.append(["label_svg", {"x": 10, ## really border_left
                                         "y": cheight/2,
                                         "label": self.name,
                                         "font_size": config["font_size"] + 3,
                                         "font_color": "black",
                                         "font_family": config["font_family"],
                                         "text_anchor": "middle",
            }])
        else:
            struct.append(["label_svg", {"x": max_width/2,
                                         "y": config["border_top"]/2,
                                         "label": self.name,
                                         "font_size": config["font_size"] + 3,
                                         "font_color": "black",
                                         "font_family": config["font_family"],
                                         "text_anchor": "middle",
            }])
        ## figure out scale optimal, if scale is None
        ## the fraction:
        if config["svg_scale"] is not None: ## scale is given:
            if config["svg_rotate"]:
                scale_value = (config["svg_max_width"] / cheight) * config["svg_scale"]
            else:
                scale_value = (config["svg_max_width"] / max_width) * config["svg_scale"]
        else:
            if config["svg_rotate"]:
                scale_value = config["svg_max_width"] / max(cheight, max_width)
            else:
                scale_value = config["svg_preferred_size"] / max(cheight, max_width)
        svg_scale = "%s%%" % int(scale_value * 100)
        scaled_width = (max_width * scale_value)
        scaled_height = (cheight * scale_value)
        #######################################################################
        ### Need a top-level width, height because Jupyter peeks at it
        #######################################################################
        if config["svg_rotate"]:
            svg_transform = """ transform="rotate(90) translate(0 -%s)" """ % scaled_height
            ### Swap them:
            top_width = scaled_height
            top_height = scaled_width
        else:
            svg_transform = ""
            top_width = scaled_width
            top_height = scaled_height
        struct.append(["svg_head", {
            "viewbox_width": max_width,  # view port width
            "viewbox_height": cheight,   # view port height
            "width": scaled_width, ## actual pixels of image in page
            "height": scaled_height, ## actual pixels of image in page
            "netname": self.name,
            "top_width": top_width,
            "top_height": top_height,
            "arrow_color": config["arrow_color"],
            "arrow_width": config["arrow_width"],
            "svg_transform": svg_transform,
        }])
        return struct

    def _initialize_javascript(self):
        from IPython.display import Javascript, display
        js = """
require(['base/js/namespace'], function(Jupyter) {
    Jupyter.notebook.kernel.comm_manager.register_target('conx_svg_control', function(comm, msg) {
        comm.on_msg(function(msg) {
            var data = msg["content"]["data"];
            var images = document.getElementsByClassName(data["class"]);
            for (var i = 0; i < images.length; i++) {
                if (data["href"]) {
                    images[i].setAttributeNS(null, "href", data["href"]);
                }
                if (data["src"]) {
                    images[i].setAttributeNS(null, "src", data["src"]);
                }
            }
        });
    });
});
"""
        display(Javascript(js))
        self._initialized_javascript = True

    def to_svg(self, inputs=None, class_id=None, **kwargs):
        """
        opts - temporary override of config

        includes:
            "font_size": 12,
            "border_top": 25,
            "border_bottom": 25,
            "hspace": 100,
            "vspace": 50,
            "image_maxdim": 200
            "image_pixels_per_unit": 50

        See .config for all options.
        """
        if any([(layer.kind() == "unconnected") for layer in self.layers]) or len(self.layers) == 0:
            return None
        # defaults:
        config = copy.copy(self.config)
        config.update(kwargs)
        struct = self.build_struct(inputs, class_id, config)
        ### Define the SVG strings:
        image_svg = """<rect x="{{rx}}" y="{{ry}}" width="{{rw}}" height="{{rh}}" style="fill:none;stroke:{border_color};stroke-width:{border_width}"/><image id="{netname}_{{name}}_{{svg_counter}}" class="{netname}_{{name}}" x="{{x}}" y="{{y}}" height="{{height}}" width="{{width}}" preserveAspectRatio="none" href="{{image}}"><title>{{tooltip}}</title></image>""".format(
            **{
                "netname": class_id if class_id is not None else self.name,
                "border_color": config["border_color"],
                "border_width": config["border_width"],
            })
        line_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}"><title>{{tooltip}}</title></line>""".format(**config)
        arrow_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}" marker-end="url(#arrow)"><title>{{tooltip}}</title></line>""".format(**config)
        arrow_rect = """<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" style="fill:white;stroke:none"><title>{tooltip}</title></rect>"""
        label_svg = """<text x="{x}" y="{y}" font-family="{font_family}" font-size="{font_size}" text-anchor="{text_anchor}" fill="{font_color}" alignment-baseline="central" {transform}>{label}</text>"""
        svg_head = """<svg id='{netname}' xmlns='http://www.w3.org/2000/svg' image-rendering="pixelated" width="{top_width}px" height="{top_height}px">
 <g {svg_transform}>
  <svg viewBox="0 0 {viewbox_width} {viewbox_height}" width="{width}px" height="{height}px">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="{arrow_color}" />
        </marker>
    </defs>"""
        templates = {
            "image_svg": image_svg,
            "line_svg": line_svg,
            "arrow_svg": arrow_svg,
            "arrow_rect": arrow_rect,
            "label_svg": label_svg,
            "svg_head": svg_head,
        }
        ## get the header:
        svg = None
        for (template_name, dict) in struct:
            if template_name == "svg_head":
                svg = svg_head.format(**dict)
        ## build the rest:
        for (template_name, dict) in struct:
            if template_name != "svg_head" and not template_name.startswith("_"):
                rotate = dict.get("rotate", config["svg_rotate"])
                if template_name == "label_svg" and rotate:
                    dict["x"] += 8
                    dict["text_anchor"] = "middle"
                    dict["transform"] = """ transform="rotate(-90 %s %s) translate(%s)" """ % (dict["x"], dict["y"], 2)
                else:
                    dict["transform"] = ""
                t = templates[template_name]
                svg += t.format(**dict)
        svg += """</svg></g></svg>"""
        if (not self._initialized_javascript and get_ipython()):
            self._initialize_javascript()
        return svg

    def _get_level_ordering(self):
        """
        Returns a list of lists of tuples from
        input to output of levels.

        Each tuple contains: (layer_name, anchor?, from_name/None)

        If anchor is True, this is just an anchor point.
        """
        ## First, get a level for all layers:
        levels = {}
        for layer in topological_sort(self, self.layers):
            if not hasattr(layer, "model"):
                continue
            level = max([levels[lay.name] for lay in layer.incoming_connections] + [-1])
            levels[layer.name] = level + 1
        max_level = max(levels.values())
        ordering = []
        for i in range(max_level + 1): # input to output
            layer_names = [layer.name for layer in self.layers if levels[layer.name] == i]
            ordering.append([(name, False, None) for name in layer_names]) # (going_to/layer_name, anchor, coming_from)
        ## promote all output banks to last row:
        for level in range(len(ordering)): # input to output
            tuples = ordering[level]
            for (name, anchor, none) in tuples[:]: # go through copy
                if self[name].kind() == "output":
                    ## move it to last row
                    ## find it and remove
                    index = tuples.index((name, anchor, None))
                    ordering[-1].append(tuples.pop(index))
        ## insert anchor points for any in next level
        ## that doesn't go to a bank in this level
        order_cache = {}
        for level in range(len(ordering)): # input to output
            tuples = ordering[level]
            for (name, anchor, fname) in tuples:
                if anchor:
                    ## is this in next? if not add it
                    next_level = [(n, hfname) for (n, anchor, hfname) in ordering[level + 1]]
                    if (name, None) not in next_level and (name, fname) not in next_level:
                        ordering[level + 1].append((name, True, fname)) # add anchor point
                    else:
                        pass ## finally!
                else:
                    ## if next level doesn't contain an outgoing
                    ## connection, add it to next level as anchor point
                    for layer in self[name].outgoing_connections:
                        next_level = [(n,fname) for (n, anchor, fname) in ordering[level + 1]]
                        if (layer.name, None) not in next_level:
                            ordering[level + 1].append((layer.name, True, name)) # add anchor point
            ## replace level with sorted level:
            lev = sorted([(self._column_order(fname if anchor else name, order_cache), name, anchor, fname)
                          for (name, anchor, fname) in ordering[level]])
            ordering[level] = [(name, anchor, fname) for (index, name, anchor, fname) in lev]
        return ordering

    def _column_order(self, layer_name, order_cache):
        """
        Get the column order of a layer_name. Note that in this
        version, the path grows on each split, and never shrinks.
        """
        ## special case to get started:
        if layer_name in self.input_bank_order:
            order_cache[layer_name] = [self.input_bank_order.index(layer_name)]
        ## Get path to this node:
        path = order_cache[layer_name]
        ## Put next layer in cache:
        if len(self[layer_name].outgoing_connections) > 1: ## split!
            count = 0
            for layer in self[layer_name].outgoing_connections:
                order_cache[layer.name] = path + [count]
                count += 1
        elif len(self[layer_name].outgoing_connections) == 0: ## output layer
            pass
        else:
            ## just one output, no split:
            order_cache[self[layer_name].outgoing_connections[0].name] = path
        ## should we worry about merges at all?
        return order_cache[layer_name]

    def describe_connection_to(self, layer1, layer2):
        """
        Returns a textual description of the weights for the SVG tooltip.
        """
        retval = "Weights from %s to %s" % (layer1.name, layer2.name)
        if self.model is None:
            return retval
        for klayer in self.model.layers:
            if klayer.name == layer2.name:
                weights = klayer.get_weights()
                for w in range(len(klayer.weights)):
                    retval += "\n %s has shape %s" % (
                        klayer.weights[w].name, weights[w].shape)
        return retval

    def saved(self, dir=None):
        """
        Return True if network has been saved.
        """
        if dir is None:
            dir = "%s.conx" % self.name.replace(" ", "_")
        return (os.path.isdir(dir) and
                os.path.isfile("%s/network.pickle" % dir) and
                os.path.isfile("%s/model.h5" % dir) and
                os.path.isfile("%s/weights.h5" % dir))

    def delete(self, dir=None):
        """
        Delete network save folder.
        """
        if dir is None:
            dir = "%s.conx" % self.name.replace(" ", "_")
        import shutil
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        else:
            print("Nothing to delete.")

    def load(self, dir=None):
        """
        Load the model and the weights/history into an existing conx network.
        """
        import pickle
        if self is None:
            raise Exception("Network.load() requires a directory name")
        elif isinstance(self, str):
            dir = self
            with open("%s/network.pickle" % (("%s.conx" % self.name.replace(" ", "_"))
                                             if dir is None else dir), "rb") as fp:
                network = pickle.load(fp)
            network.load(dir)
            return network
        else:
            self.load_model(dir)
            self.load_weights(dir)
            self.load_config(dir)

    def save(self, dir=None):
        """
        Save the model and the weights/history (if compiled) to a dir.
        """
        if self.model:
            self.save_model(dir)
            self.save_weights(dir)
            self.save_config(dir)
            with open("%s/network.pickle" % (("%s.conx" % self.name.replace(" ", "_"))
                                             if dir is None else dir), "wb") as fp:
                pickle.dump(self, fp)
        else:
            raise Exception("need to compile network before saving")

    def load_model(self, dir=None, filename=None):
        """
        Load a model from a dir/filename.
        """
        from keras.models import load_model
        if dir is None:
            dir = "%s.conx" % self.name.replace(" ", "_")
        if filename is None:
            filename = "model.h5"
        self.model = load_model(os.path.join(dir, filename))
        if self.compile_options:
            self.reset()

    def save_model(self, dir=None, filename=None):
        """
        Save a model (if compiled) to a dir/filename.
        """
        if self.model:
            if dir is None:
                dir = "%s.conx" % self.name.replace(" ", "_")
            if filename is None:
                filename = "model.h5"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            self.model.save(os.path.join(dir, filename))
        else:
            raise Exception("need to compile network before saving")

    def load_history(self, dir=None, filename=None):
        """
        Load the history from a dir/file.

        network.load_history()
        """
        if dir is None:
            dir = "%s.conx" % self.name.replace(" ", "_")
        if filename is None:
            filename = "history.pickle"
        full_filename = os.path.join(dir, filename)
        if os.path.isfile(full_filename):
            with open(os.path.join(dir, filename), "rb") as fp:
                self.history = pickle.load(fp)
                self.weight_history = pickle.load(fp)
                self.epoch_count = (len(self.history) - 1) if self.history else 0
        else:
            print("WARNING: no such history file '%s'" % full_filename, file=sys.stderr)

    def save_history(self, dir=None, filename=None):
        """
        Save the history to a file.

        network.save_history()
        """
        if dir is None:
            dir = "%s.conx" % self.name.replace(" ", "_")
        if filename is None:
            filename = "history.pickle"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, filename), "wb") as fp:
            pickle.dump(self.history, fp)
            pickle.dump(self.weight_history, fp)

    def load_weights(self, dir=None, filename=None):
        """
        Load the network weights and history from dir/files.

        network.load_weights()
        """
        if self.model:
            if dir is None:
                dir = "%s.conx" % self.name.replace(" ", "_")
            if filename is None:
                filename = "weights.h5"
            self.model.load_weights(os.path.join(dir, filename))
            self.load_history(dir)
        else:
            raise Exception("need to compile network before loading weights")

    def save_weights(self, dir=None, filename=None):
        """
        Save the network weights and history to dir/files.

        network.save_weights()
        """
        if self.model:
            if dir is None:
                dir = "%s.conx" % self.name.replace(" ", "_")
            if filename is None:
                filename = "weights.h5"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            self.model.save_weights(os.path.join(dir, filename))
            self.save_history(dir)
        else:
            raise Exception("need to compile network before saving weights")

    def dashboard(self, width="95%", height="550px", play_rate=0.5):
        """
        Build the dashboard for Jupyter widgets. Requires running
        in a notebook/jupyterlab.
        """
        from .widgets import Dashboard
        return Dashboard(self, width, height, play_rate)

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
        print(label + self.pf(vector[:20], **opts))

    def pf_matrix(self, matrix, force=False, **opts):
        """
        Pretty-fromat a matrix. If a list, then that implies multi-bank.
        """
        if isinstance(matrix, list): ## multiple output banks
            rows = []
            for r in range(len(matrix[0])):
                row = []
                for c in range(len(matrix)):
                    row.append(self.pf(matrix[c][r], **opts))
                    if c > 99 and not force:
                        row.append("...")
                rows.append("[" + (",".join(row)) + "]")
                if r > 99 and not force:
                    rows.append("...")
                    break
            return rows
        else:
            rows = []
            for r in range(len(matrix)):
                rows.append(self.pf(matrix[r], **opts))
                if r > 99 and not force:
                    rows.append("...")
                    break
            return rows

    def pf(self, vector, **opts):
        """
        Pretty-format a vector. Returns string.

        Arguments:
            vector (list): The first parameter.
            precision (int): Number of decimal places to show for each
                value in vector.

        Returns:
            str: Returns the vector formatted as a short string.

        Examples:
            These examples demonstrate the net.pf formatting function:

            >>> import conx
            >>> net = Network("Test")
            >>> net.pf([1.01])
            '[1.01]'

            >>> net.pf(range(10), precision=2)
            '[0,1,2,3,4,5,6,7,8,9]'

            >>> net.pf([0]*10000) # doctest: +ELLIPSIS
            '[0,0,0,...]'
        """
        if isinstance(vector, collections.Iterable):
            vector = list(vector)
        if isinstance(vector, (list, tuple)):
            vector = np.array(vector)
        config = copy.copy(self.config)
        config.update(opts)
        precision  = "{0:.%df}" % config["precision"]
        return np.array2string(
            vector,
            formatter={'float_kind': precision.format},
            separator=",",
            max_line_width=79).replace("\n", "")

    def set_weights(self, weights, layer_name=None):
        """
        Set the model's weights, or a particular layer's weights.

        >>> net = Network("Weight Set Test", 2, 2, 1, activation="sigmoid")
        >>> net.compile(error="mse", optimizer="sgd")
        >>> net.set_weights(net.get_weights())

        >>> hw = net.get_weights("hidden")
        >>> net.set_weights(hw, "hidden")
        """
        if self.model is None:
            raise Exception("need to compile network")
        if layer_name is None:
            self.model.set_weights(weights)
        else:
            for i in range(len(self.model.layers)):
                if self.model.layers[i].name == layer_name:
                    w = [np.array(x) for x in
                         self.model.layers[i].get_weights()]
                    self.model.layers[i].set_weights(w)

    def to_array(self) -> list:
        """
        Get the weights of a network as a flat, one-dimensional list.

        Example:
            >>> from conx import Network
            >>> net = Network("Deep", 3, 4, 5, 2, 3, 4, 5)
            >>> net.compile(optimizer="adam", error="mse")
            >>> array = net.to_array()
            >>> len(array)
            103

        Returns:
            All of weights and biases of the network in a single, flat list.
        """
        if self.model is None:
            raise Exception("need to compile network")
        array = []
        for layer in self.model.layers:
            for weight in layer.get_weights():
                array.extend(weight.flatten())
        return array

    def from_array(self, array: list):
        """
        Load the weights from a list.

        Arguments:
            array: a sequence (e.g., list, np.array) of numbers

        Example:
            >>> from conx import Network
            >>> net = Network("Deep", 3, 4, 5, 2, 3, 4, 5)
            >>> net.compile(optimizer="adam", error="mse")
            >>> net.from_array([0] * 103)
            >>> array = net.to_array()
            >>> len(array)
            103
        """
        if self.model is None:
            raise Exception("need to compile network")
        position = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            new_weights = []
            for i in range(len(weights)):
                w = weights[i]
                size = reduce(operator.mul, w.shape)
                new_w = np.array(array[position:position + size]).reshape(w.shape)
                new_weights.append(new_w)
                position += size
            layer.set_weights(new_weights)

    ### Config methods:
    def load_config(self, datadir=None, config_file=None):
        """
        """
        if datadir is None:
            datadir = "%s.conx" % self.name.replace(" ", "_")
        if config_file is None:
            config_file = "config.json"
        datadir = os.path.expanduser(datadir)
        if not os.path.exists(datadir):
            ## second try, here
            datadir = os.path.join('/tmp', datadir)
        full_config_file = os.path.join(datadir, config_file)
        if os.path.isfile(full_config_file):
            with open(full_config_file) as fp:
                config_data = json.load(fp)
            self.update_config(config_data)
        ## give up, fail silently

    def save_config(self, datadir=None, config_file=None):
        """
        """
        if datadir is None:
            datadir = "%s.conx" % self.name.replace(" ", "_")
        if config_file is None:
            config_file = "config.json"
        if not os.path.exists(datadir):
            try:
                os.makedirs(datadir)
            except:
                datadir = os.path.join('/tmp', datadir)
                os.makedirs(datadir)
        full_config_file = os.path.join(datadir, config_file)
        self.rebuild_config()
        with open(full_config_file, "w") as fp:
            json.dump(self.config, fp, indent="    ")

    def update_config(self, config):
        """
        """
        self.config.update(config)
        for layer in self.layers:
            self.update_layer_from_config(layer)

    def rebuild_config(self):
        """
        """
        self.config["config_layers"].clear()
        for layer in self.layers:
            d = {}
            self.config["config_layers"][layer.name] = d
            for item in ["visible",
                         "minmax",
                         "vshape",
                         "image_maxdim",
                         "image_pixels_per_unit",
                         "colormap",
                         "feature",
                         "max_draw_units"]:
                d[item] = getattr(layer, item)

    def update_layer_from_config(self, layer):
        """
        """
        if layer.name in self.config["config_layers"]:
            for item in self.config["config_layers"][layer.name]:
                setattr(layer, item, self.config["config_layers"][layer.name][item])

class _InterruptHandler():
    """
    Class for handling interrupts so that state is not left
    in inconsistant situation.
    """
    def __init__(self, network, sig=signal.SIGINT):
        self.network = network
        self.sig = sig
        self.interrupted = None
        self.released = None
        self.original_handler = None

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
            self.network.model.stop_training = True

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
