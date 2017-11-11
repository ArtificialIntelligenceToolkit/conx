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
import importlib
from functools import reduce
import signal
import numbers
import threading
import time
import random
import pickle
import base64
import html
import copy
import sys
import io
import os
import re
import PIL
from typing import Any

import numpy as np
import keras
from keras.callbacks import Callback, History

from .utils import *
from .layers import Layer
from .dataset import Dataset


try:
    from IPython import get_ipython
except:
    get_ipython = lambda: None

#------------------------------------------------------------------------

class ReportCallback(Callback):
    def __init__(self, network, report_rate, mpl_backend):
        # mpl_backend is matplotlib backend
        super().__init__()
        self.network = network
        self.report_rate = report_rate
        self.mpl_backend = mpl_backend
        self.in_console = self.network.in_console(mpl_backend)

    def on_epoch_end(self, epoch, results=None):
        #print("in ReportCallback with epoch = %d" % epoch)
        self.network.history.append(results)
        self.network.epoch_count += 1
        #print("epoch_count is now", self.network.epoch_count)
        #print("history is now", self.network.history)
        #print("ReportCallback got:", epoch, results)
        if self.in_console and (epoch+1) % self.report_rate == 0:
            self.network.report_epoch(self.network.epoch_count, results)

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

    def on_epoch_end(self, epoch, results=None):
        #print("in PlotCallback with epoch = %d" % epoch)
        if epoch == -1:
            # training loop finished, so make a final update to plot
            # in case the number of loop cycles wasn't a multiple of
            # report_rate
            self.network.plot_loss_acc(self, epoch)
            if not self.in_console:
                plt.close(self.figure[0])
        elif (epoch+1) % self.report_rate == 0:
            self.network.plot_loss_acc(self, epoch)

class StoppingCriteria(Callback):
    def __init__(self, item, op, value, use_validation_to_stop):
        super().__init__()
        self.item = item
        self.op = op
        self.value = value
        self.use_validation_to_stop = use_validation_to_stop

    def on_epoch_end(self, epoch, results=None):
        key = ("val_" + self.item) if self.use_validation_to_stop else self.item
        if key in results: # we get what we need directly:
            if self.compare(results[key], self.op, self.value):
                self.model.stop_training = True
        else:
            ## ok, then let's sum/average anything that matches
            total = 0
            count = 0
            for item in results:
                if self.use_validation_to_stop:
                    if item.startswith("val_") and item.endswith("_" + self.item):
                        count += 1
                        total += results[item]
                else:
                    if item.endswith("_" + self.item) and not item.startswith("val_"):
                        count += 1
                        total += results[item]
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
        >>> net.add(Layer("hidden", 5))
        >>> net.add(Layer("output", 2))
        >>> net.connect()
        >>> len(net.layers)
        3

        >>> net = Network("XOR3")
        >>> net.add(Layer("input", 2))
        >>> net.add(Layer("hidden", 5))
        >>> net.add(Layer("output", 2))
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

    def __init__(self, name: str, *sizes: int, **config: Any):
        import keras.backend as K
        if not isinstance(name, str):
            raise Exception("first argument should be a name for the network")
        self.debug = False
        ## Pick a place in the random stream, and remember it:
        ## (can override randomness with a particular seed):
        if "seed" in config:
            seed = config["seed"]
            del config["seed"]
        else:
            seed = np.random.randint(int(2 ** 32 - 1))
        self.seed = seed
        np.random.seed(self.seed)
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
            "minmax": None,
            "pixels_per_unit": 1,
            "precision": 2,
            "svg_height": 780, # for svg
        }
        if not isinstance(name, str):
            raise Exception("conx layers need a name as a first parameter")
        self.num_input_layers = 0
        self.num_target_layers = 0
        self.input_bank_order = []
        self.output_bank_order = []
        self.config.update(config)
        self.dataset = Dataset(self)
        self.compile_options = {}
        self.train_options = {}
        self._tolerance = K.variable(0.1, dtype='float32', name='tolerance')
        self.name = name
        self.layers = []
        self.layer_dict = {}
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
        self.epoch_count = 0
        self.history = []
        self.visualize = get_ipython() is not None
        self._comm = None
        self.model = None
        self.prop_from_dict = {}
        self._svg_counter = 1
        self._need_to_show_headings = True

    def _get_tolerance(self):
        import keras.backend as K
        return K.get_value(self._tolerance)

    def _set_tolerance(self, value):
        import keras.backend as K
        K.set_value(self._tolerance, value)

    tolerance = property(_get_tolerance,
                         _set_tolerance)

    def __getitem__(self, layer_name):
        if layer_name not in self.layer_dict:
            return None
        else:
            return self.layer_dict[layer_name]

    def _repr_html_(self):
        if all([layer.model for layer in self.layers]):
            return self.build_svg(opts={"svg_height": "780px"}) ## will fill width
        else:
            return None

    def __repr__(self):
        return "<Network name='%s' (%s)>" % (
            self.name, ("uncompiled" if not self.model else "compiled"))

    def snapshot(self, inputs=None, class_id=None, height="780px", opts={}):
        from IPython.display import HTML
        if class_id is None:
            r = random.randint(1,1000000)
            class_id = "snapshot-%s-%s" % (self.name, r)
        if height is not None:
            opts["svg_height"] = height
        return HTML(self.build_svg(class_id=class_id, inputs=inputs, opts=opts))

    def in_console(self, mpl_backend):
        """
        Return True if running connected to a console; False if connected
        to notebook, or other non-console system.

        Possible values:
            'TkAgg' - console with Tk
            'Qt5Agg' - console with Qt
            'MacOSX' - mac console
            'module://ipykernel.pylab.backend_inline` - default for notebook
                          and non-console, and when using %matplotlib inline
            'NbAgg` - notebook, using %matplotlib notebook

        Here, None means not plotting, or just use text.

        NOTE: if you are running ipython without a DISPLAY with the QT
        background, you may wish to:
            export QT_QPA_PLATFORM='offscreen'
        """
        return mpl_backend not in [
            'module://ipykernel.pylab.backend_inline',
            'NbAgg',
        ]

    def add(self, layer: Layer):
        """
        Add a layer to the network layer connections. Order is not
        important, unless calling :any:`Network.connect` without any
        arguments.

        Arguments:
            layer: A layer instance.

        Examples:
            >>> net = Network("XOR2")
            >>> net.add(Layer("input", 2))
            >>> len(net.layers)
            1

            >>> net = Network("XOR3")
            >>> net.add(Layer("input", 2))
            >>> net.add(Layer("hidden", 5))
            >>> net.add(Layer("output", 2))
            >>> len(net.layers)
            3

        Note:
            See :any:`Network` for more information.
        """
        if layer.name in self.layer_dict:
            raise Exception("duplicate layer name '%s'" % layer.name)
        self.layers.append(layer)
        self.layer_dict[layer.name] = layer

    def connect(self, from_layer_name:str=None, to_layer_name:str=None):
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
            >>> net.add(Layer("hidden", 5))
            >>> net.add(Layer("output", 2))
            >>> net.connect()
            >>> [layer.name for layer in net["input"].outgoing_connections]
            ['hidden']
        """
        if len(self.layers) == 0:
            raise Exception("no layers have been added")
        if from_layer_name is None and to_layer_name is None:
            if (any([layer.outgoing_connections for layer in self.layers]) or
                any([layer.incoming_connections for layer in self.layers])):
                raise Exception("layers already have connections")
            for i in range(len(self.layers) - 1):
                self.connect(self.layers[i].name, self.layers[i+1].name)
        else:
            ## FIXME: check for cycle here
            if from_layer_name == to_layer_name:
                raise Exception("self connections are not allowed")
            if from_layer_name not in self.layer_dict:
                raise Exception('unknown layer: %s' % from_layer_name)
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

    def summary(self):
        """
        Print out a summary of the network.
        """
        print("Network Summary")
        print("---------------")
        print("Network name:", self.name)
        for layer in self.layers:
            layer.summary()

    def reset(self, clear=False, **overrides):
        """
        Reset all of the weights/biases in a network.
        The magnitude is based on the size of the network.
        """
        self.epoch_count = 0
        self.history = []
        self.prop_from_dict = {}
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
             filter="all"):
        """
        Test a dataset.
        """
        tolerance = tolerance if tolerance is not None else self.tolerance
        if len(self.dataset.inputs) == 0:
            raise Exception("nothing to test")
        if self.dataset._split == 1.0: ## special case; use entire set
            inputs = self.dataset._inputs
            targets = self.dataset._targets
        else:
            ## need to split; check format based on output banks:
            length = len(self.dataset.train_targets)
            if self.num_target_layers == 1:
                targets = self.dataset._targets[:length]
            else:
                targets = [column[:length] for column in self.dataset._targets]
            if self.num_input_layers == 1:
                inputs = self.dataset._inputs[:length]
            else:
                inputs = [column[:length] for column in self.dataset._inputs]
        self._test(inputs, targets, "train dataset", batch_size, show,
                   tolerance, force, show_inputs, show_outputs, filter)
        if self.dataset._split in [1.0, 0.0]: ## special case; use entire set
            return
        else: # split is greater than 0, less than 1
            ## need to split; check format based on output banks:
            length = len(self.dataset.test_targets)
            if self.num_target_layers == 1:
                targets = self.dataset._targets[-length:]
            else:
                targets = [column[-length:] for column in self.dataset._targets]
            if self.num_input_layers == 1:
                inputs = self.dataset._inputs[-length:]
            else:
                inputs = [column[-length:] for column in self.dataset._inputs]
            val_values = self.model.evaluate(inputs, targets, verbose=0)
        self._test(inputs, targets, "validation dataset", batch_size, show,
                   tolerance, force, show_inputs, show_outputs, filter)

    def _test(self, inputs, targets, dataset, batch_size=32, show=False,
              tolerance=None, force=False,
              show_inputs=True, show_outputs=True,
              filter="all"):
        print("=" * 56)
        print("Testing %s with tolerance %.6s..." % (dataset, tolerance))
        outputs = self.model.predict(inputs, batch_size=batch_size)
        ## FYI: outputs not shaped
        correct = self.compute_correct(outputs, targets, tolerance)
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
        print("Total count:", len(correct))
        print("      correct:", len([c for c in correct if c]))
        print("      incorrect:", len([c for c in correct if not c]))
        print("Total percentage correct:", list(correct).count(True)/len(correct))

    def compute_correct(self, outputs, targets, tolerance=None):
        """
        Both are np.arrays. Return [True, ...].
        """
        tolerance = tolerance if tolerance is not None else self.tolerance
        if self.num_target_layers > 1: ## multiple output banks
            correct = []
            for r in range(len(outputs[0])):
                row = []
                for c in range(len(outputs)):
                    row.extend(list(map(lambda v: v <= tolerance, np.abs(outputs[c][r] - targets[c][r]))))
                correct.append(all(row))
            return correct
        else:
            correct = [x.all() for x in map(lambda v: v <= tolerance, np.abs(outputs - targets))]
        return correct

    def train_one(self, inputs, targets, batch_size=32):
        """
        Train on one input/target pair. Requires internal format.

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
            >>> net.add(Layer("input1", shape=1))
            >>> net.add(Layer("input2", shape=1))
            >>> net.add(Layer("hidden1", shape=2, activation="sigmoid"))
            >>> net.add(Layer("hidden2", shape=2, activation="sigmoid"))
            >>> net.add(Layer("shared-hidden", shape=2, activation="sigmoid"))
            >>> net.add(Layer("output1", shape=1, activation="sigmoid"))
            >>> net.add(Layer("output2", shape=1, activation="sigmoid"))
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
        outputs = self.propagate(inputs, batch_size=batch_size, visualize=False)
        errors = (np.array(outputs) - np.array(targets)).tolist() # FYI: multi outputs
        if self.visualize and get_ipython():
            if self.config["show_targets"]:
                self.display_component([targets], "targets") # FIXME: use output layers' minmax
            if self.config["show_errors"]:
                self.display_component([errors], "errors", minmax=(-1, 1))
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
        if len(self.dataset.inputs) == 0:
            raise Exception("no dataset loaded")
        (train_inputs, train_targets), (test_inputs, test_targets) = self.dataset._split_data()
        train_metrics = self.model.evaluate(train_inputs, train_targets, batch_size=batch_size, verbose=0)
        results = {k:v for k, v in zip(self.model.metrics_names, train_metrics)}
        if len(test_inputs) > 0:
            test_metrics = self.model.evaluate(test_inputs, test_targets, batch_size=batch_size, verbose=0)
            results.update({"val_"+k: v for k, v in zip(self.model.metrics_names, test_metrics)})
        return results

    def train(self, epochs=1, accuracy=None, error=None, batch_size=32,
              report_rate=1, verbose=1, kverbose=0, shuffle=True, tolerance=None,
              class_weight=None, sample_weight=None, use_validation_to_stop=False,
              plot=False):
        """
        Train the network.

        To stop before number of epochs, give either error=VALUE, or accuracy=VALUE.

        Normally, it will check training info to stop, unless you
        use_validation_to_stop = True.
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
            }
        if plot:
            import matplotlib
            mpl_backend = matplotlib.get_backend()
        else:
            mpl_backend = None
        if not isinstance(report_rate, numbers.Integral) or report_rate < 1:
            raise Exception("bad report rate: %s" % (report_rate,))
        if not (isinstance(batch_size, numbers.Integral) or batch_size is None):
            raise Exception("bad batch size: %s" % (batch_size,))
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
            self._tolerance.set_value(tolerance)
        ## Going to need evaluation on training set in any event:
        if self.dataset._split == 1.0: ## special case; use entire set
            inputs = self.dataset._inputs
            targets = self.dataset._targets
        else:
            ## need to split; check format based on output banks:
            length = len(self.dataset.train_targets)
            if self.num_target_layers == 1:
                targets = self.dataset._targets[:length]
            else:
                targets = [column[:length] for column in self.dataset._targets]
            if self.num_input_layers == 1:
                inputs = self.dataset._inputs[:length]
            else:
                inputs = [column[:length] for column in self.dataset._inputs]
        if len(self.history) > 0:
            results = self.history[-1]
        else:
            print("Evaluating initial training metrics...")
            values = self.model.evaluate(inputs, targets, batch_size=batch_size, verbose=0)
            if not isinstance(values, list): # if metrics is just a single value
                values = [values]
            results = {metric: value for metric,value in zip(self.model.metrics_names, values)}
        results_acc = self._compute_result_acc(results)
        val_results = {}
        if use_validation_to_stop:
            if ((self.dataset._split > 0) and
                ((accuracy is not None) or (error is not None))):
                ## look at split, use validation subset:
                if self.dataset._split == 1.0: ## special case; use entire set; already done!
                    val_results = {"val_%s" % key: results[key] for key in results}
                else: # split is greater than 0, less than 1
                    print("Evaluating initial validation metrics...")
                    ## need to split; check format based on output banks:
                    length = len(self.dataset.test_targets)
                    if self.num_target_layers == 1:
                        targets = self.dataset._targets[-length:]
                    else:
                        targets = [column[-length:] for column in self.dataset._targets]
                    if self.num_input_layers == 1:
                        inputs = self.dataset._inputs[-length:]
                    else:
                        inputs = [column[-length:] for column in self.dataset._inputs]
                    val_values = self.model.evaluate(inputs, targets, batch_size=batch_size, verbose=0)
                    val_results = {metric: value for metric,value in zip(self.model.metrics_names, val_values)}
                val_results_acc = self._compute_result_acc(val_results, use_validation=True)
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
                    return
        else: ## regular training to stop, use_validation_to_stop is False
            if ((accuracy is not None) and (results_acc >= accuracy)):
                print("No training required: accuracy already to desired value")
                print("Training dataset status:")
                self.report_epoch(self.epoch_count, results)
                return
            elif ((error is not None) and (results["loss"] <= error)):
                print("No training required: error already to desired value")
                print("Training dataset status:")
                self.report_epoch(self.epoch_count, results)
                return
        ## Ok, now we know we need to train:
        if len(self.history) == 0:
            results.update(val_results)
            self.history = [results]
        print("Training...")
        if self.in_console(mpl_backend):
            self.report_epoch(self.epoch_count, self.history[-1])
        interrupted = False
        callbacks=[
            History(),
            ReportCallback(self, report_rate, mpl_backend),
        ]
        if accuracy is not None:
            callbacks.append(StoppingCriteria("acc", ">=", accuracy, use_validation_to_stop))
        if error is not None:
            callbacks.append(StoppingCriteria("loss", "<=", error, use_validation_to_stop))
        if plot:
            pc = PlotCallback(self, report_rate, mpl_backend)
            callbacks.append(pc)
        with _InterruptHandler(self) as handler:
            if self.dataset._split == 1:
                result = self.model.fit(self.dataset._inputs,
                                        self.dataset._targets,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_data=(self.dataset._inputs,
                                                         self.dataset._targets),
                                        callbacks=callbacks,
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
                                        callbacks=callbacks,
                                        shuffle=shuffle,
                                        class_weight=class_weight,
                                        sample_weight=sample_weight,
                                        verbose=kverbose)
            if plot:
                pc.on_epoch_end(-1)
            if handler.interrupted:
                interrupted = True
        last_epoch = self.history[-1]
        assert len(self.history) == self.epoch_count+1  # +1 is for epoch 0
        if verbose:
            print("=" * 72)
            self.report_epoch(self.epoch_count, last_epoch)
        if interrupted:
            raise KeyboardInterrupt
        if verbose == 0:
            return (self.epoch_count, result)

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

    def set_activation(self, layer_name, activation):
        """
        Swap activation function of a layer after compile.
        """
        from keras.models import load_model
        import keras.activations
        import tempfile
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
        """
        from matplotlib import cm
        import PIL
        weights = [layer.get_weights() for layer in self.model.layers
                   if layer_name == layer.name][0]
        weights = weights[0] # get the weight matrix, not the biases
        vector = self[layer_name].scale_output_for_image(weights, (-5,5), truncate=True)
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

    def get_weights(self, layer_name):
        """
        Get the weights from the model in an easy to read format.
        """
        weights = [layer.get_weights() for layer in self.model.layers
                   if layer_name == layer.name][0]
        return [m.tolist() for m in weights]

    def propagate(self, input, batch_size=32, visualize=None):
        """
        Propagate an input (in human API) through the network.
        If visualizing, the network image will be updated.
        """
        import keras.backend as K
        visualize = visualize if visualize is not None else self.visualize
        if isinstance(input, dict):
            input = [input[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                input = input[0]
        elif isinstance(input, PIL.Image.Image):
            input = image2array(input)
        if self.num_input_layers == 1:
            outputs = self.model.predict(np.array([input]), batch_size=batch_size)
        else:
            inputs = [np.array([x], "float32") for x in input]
            outputs = self.model.predict(inputs, batch_size=batch_size)
        ## Shape the outputs:
        if self.num_target_layers == 1:
            shape = self[self.output_bank_order[0]].shape
            try:
                outputs = outputs[0].reshape(shape).tolist()
            except:
                outputs = outputs[0].tolist()  # can't reshape; maybe a dynamically changing output
        else:
            shapes = [self[layer_name].shape for layer_name in self.output_bank_order]
            ## FIXME: may not be able to reshape; dynamically changing output
            outputs = [outputs[i].reshape(shapes[i]).tolist() for i in range(len(self.output_bank_order))]
        if visualize and get_ipython():
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            if self._comm.kernel:
                for layer in self.layers:
                    image = self.propagate_to_image(layer.name, input, batch_size, visualize=False)
                    data_uri = self._image_to_uri(image)
                    self._comm.send({'class': "%s_%s" % (self.name, layer.name), "href": data_uri})
        return outputs

    def propagate_from(self, layer_name, input, output_layer_names=None, batch_size=32, visualize=None):
        """
        Propagate activations from the given layer name to the output layers.
        """
        visualize = visualize if visualize is not None else self.visualize
        if layer_name not in self.layer_dict:
            raise Exception("No such layer '%s'" % layer_name)
        if isinstance(input, dict):
            input = [input[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                input = input[0]
        elif isinstance(input, PIL.Image.Image):
            input = image2array(input)
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
            if (layer_name, output_layer_name) not in self.prop_from_dict:
                path = find_path(self, layer_name, output_layer_name)
                # Make a new Input to start here:
                input_k = k = keras.layers.Input(self[layer_name].shape, name=self[layer_name].name)
                # So that we can display activations here:
                if (layer_name, layer_name) not in self.prop_from_dict:
                    self.prop_from_dict[(layer_name, layer_name)] = keras.models.Model(inputs=input_k,
                                                                                       outputs=k)
                for layer in path: # FIXME: this should be a straight path between incoming and outgoing
                    k = layer.keras_layer(k)
                    if (layer_name, layer.name) not in self.prop_from_dict:
                        self.prop_from_dict[(layer_name, layer.name)] = keras.models.Model(inputs=input_k,
                                                                                           outputs=k)
            # Now we should be able to get the prop_from model:
            prop_model = self.prop_from_dict.get((layer_name, output_layer_name), None)
            inputs = np.array([input])
            outputs.append([list(x) for x in prop_model.predict(inputs)][0])
            ## FYI: outputs not shaped
        if visualize and get_ipython():
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            ## Update from start to rest of graph
            if self._comm.kernel:
                for output_layer_name in output_layer_names:
                    for layer in find_path(self, layer_name, output_layer_name):
                        model = self.prop_from_dict[(layer_name, layer.name)]
                        vector = model.predict(inputs)[0]
                        ## FYI: outputs not shaped
                        image = layer.make_image(vector, config=self.config)
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
        output_names = self.output_bank_order
        if self._comm.kernel:
            for (target, layer_name) in zip(vector, output_names):
                array = np.array(target)
                if component == "targets":
                    colormap = self[layer_name].colormap
                else:
                    colormap = get_error_colormap()
                image = self[layer_name].make_image(array, colormap, config) # minmax=minmax, colormap=colormap)
                data_uri = self._image_to_uri(image)
                self._comm.send({'class': "%s_%s_%s" % (self.name, layer_name, component), "href": data_uri})

    def propagate_to(self, layer_name, inputs, batch_size=32, visualize=None):
        """
        Computes activation at a layer. Side-effect: updates visualized SVG.
        """
        visualize = visualize if visualize is not None else self.visualize
        if layer_name not in self.layer_dict:
            raise Exception('unknown layer: %s' % (layer_name,))
        if isinstance(inputs, dict):
            inputs = [inputs[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                inputs = inputs[0]
        elif isinstance(inputs, PIL.Image.Image):
            inputs = image2array(inputs)
        if self.num_input_layers == 1:
            outputs = self[layer_name].model.predict(np.array([inputs]), batch_size=batch_size)
        else:
            # get just inputs for this layer, in order:
            vector = [np.array([inputs[self.input_bank_order.index(name)]]) for name in
                      self._get_sorted_input_names(self[layer_name].input_names)]
            outputs = self[layer_name].model.predict(vector, batch_size=batch_size)
        ## output shaped below:
        if visualize and get_ipython():
            if not self._comm:
                from ipykernel.comm import Comm
                self._comm = Comm(target_name='conx_svg_control')
            # Update path from input to output
            if self._comm.kernel:
                for layer in self.layers: # FIXME??: update all layers for now
                    out = self.propagate_to(layer.name, inputs, visualize=False)
                    image = self[layer.name].make_image(np.array(out), config=self.config) # single vector, as an np.array
                    data_uri = self._image_to_uri(image)
                    self._comm.send({'class': "%s_%s" % (self.name, layer.name), "href": data_uri})
        ## Shape the outputs:
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
        output_shape = self[layer_name].keras_layer.output_shape
        return (isinstance(output_shape, tuple) and len(output_shape) == 4)


    def propagate_to_features(self, layer_name, inputs, cols=5, scale=1.0, html=True, size=None, display=True):
        """
        if html is True, then generate HTML, otherwise send images.
        """
        from IPython.display import HTML
        if isinstance(inputs, dict):
            inputs = [inputs[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                inputs = inputs[0]
        elif isinstance(inputs, PIL.Image.Image):
            inputs = image2array(inputs)
        output_shape = self[layer_name].keras_layer.output_shape
        retval = """<table><tr>"""
        if self._layer_has_features(layer_name):
            if html:
                orig_feature = self[layer_name].feature
                for i in range(output_shape[3]):
                    self[layer_name].feature = i
                    image = self.propagate_to_image(layer_name, inputs, visualize=False)
                    if scale != 1.0:
                        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
                    #if size:
                    #    image = image.resize(size)
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
                    image = self.propagate_to_image(layer_name, inputs, visualize=False)
                    if scale != 1.0:
                        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
                    data_uri = self._image_to_uri(image)
                    if not self._comm:
                        from ipykernel.comm import Comm
                        self._comm = Comm(target_name='conx_svg_control')
                    if self._comm.kernel:
                        self._comm.send({'class': "%s_%s_feature%s" % (self.name, layer_name, i), "src": data_uri})
                self[layer_name].feature = orig_feature

    def propagate_to_image(self, layer_name, input, batch_size=32, scale=1.0, visualize=None):
        """
        Gets an image of activations at a layer.
        """
        if isinstance(input, dict):
            input = [input[name] for name in self.input_bank_order]
            if self.num_input_layers == 1:
                input = input[0]
        elif isinstance(input, PIL.Image.Image):
            input = image2array(input)
        outputs = self.propagate_to(layer_name, input, batch_size, visualize=visualize)
        array = np.array(outputs)
        image = self[layer_name].make_image(array, config=self.config)
        if scale != 1.0:
            image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
        return image

    def plot_activation_map(self, from_layer='input', from_units=(0,1), to_layer='output',
                            to_unit=0, colormap=None, default_from_layer_value=0,
                            resolution=None, act_range=(0,1), show_values=False):
        from .graphs import plot_activations
        return plot_activations(self, from_layer, from_units, to_layer, to_unit,
                                colormap, default_from_layer_value, resolution,
                                act_range, show_values)

    def plot_layer_weights(self, layer_name, units='all', wrange=None, wmin=None, wmax=None,
                           cmap='gray', vshape=None, cbar=True, ticks=5, figsize=(12,3)):
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
                  % (rows, cols))
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
            return self.plot_layer_weights(layer_name, units, None, wmin, wmax, cmap, vshape,
                                           cbar, ticks, figsize)
        if wmin >= wmax:
            raise Exception("specified weight range is empty")
        if not isinstance(ticks, numbers.Integral) or ticks < 2:
            raise Exception("invalid number of colorbar ticks: %s" % (ticks,))
        # clip weights to the range [wmin, wmax] and normalize to [0, 1]:
        scaled_W = (np.clip(W, wmin, wmax) - wmin) / (wmax - wmin)
        # FIXME: need a better way to set the figure size
        fig, axes = plt.subplots(1, len(units), figsize=figsize)#, tight_layout=True)
        if len(units) == 1:
            axes = [axes]
        for unit, ax in zip(units, axes):
            ax.axis('off')
            ax.set_title('weights to %s[%d]' % (layer_name, unit))
            im = scaled_W[unit,:].reshape((rows, cols))
            axim = ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        if cbar:
            tick_locations = np.linspace(0, 1, ticks)
            tick_values = tick_locations * (wmax - wmin) + wmin
            colorbar = fig.colorbar(axim, ticks=tick_locations)
            cbar_labels = ['0' if t == 0 else '%+.2f' % (t,) for t in tick_values]
            cbar_labels[0] = wmin_label
            cbar_labels[-1] = wmax_label
            colorbar.ax.set_yticklabels(cbar_labels)
        plt.show(block=False)

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
        """returns a list of the metrics available in the Network's history"""
        metrics = set()
        for epoch in self.history:
            metrics = metrics.union(set(epoch.keys()))
        return sorted(metrics)

    def plot(self, metrics=None, ymin=None, ymax=None, start=0, end=None, legend='best',
             title=None, svg=False):
        """Plots the current network history for the specific epoch range and
        metrics. metrics is '?', 'all', a metric keyword, or a list of metric keywords.
        if metrics is None, loss and accuracy are plotted on separate graphs.

        >>> net = Network("Plot Test", 1, 3, 1)
        >>> net.compile(error="mse", optimizer="rmsprop")
        >>> net.dataset.add([0.0], [1.0])
        >>> net.dataset.add([1.0], [0.0])
        >>> net.train()  # doctest: +ELLIPSIS
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
        if metrics is None:
            return self.plot('loss') # FIXME: change this to plot loss and acc on separate graphs
        elif metrics is '?':
            print("Available metrics:", ", ".join(self.get_metrics()))
            return
        elif metrics == 'all':
            metrics = self.get_metrics()
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif isinstance(metrics, (list, tuple)):
            pass
        else:
            print("metrics: expected a list or a string but got %s" % (metrics,))
            return
        fig, ax = plt.subplots(1)
        x_values = range(self.epoch_count+1)
        x_values = x_values[start:end]
        ax.set_xlabel('Epoch')
        data_found = False
        for metric in metrics:
            y_values = [epoch[metric] if metric in epoch else None for epoch in self.history]
            y_values = y_values[start:end]
            if y_values.count(None) == len(y_values):
                print("WARNING: No %s data available for the specified epochs" % (metric,))
            else:
                ax.plot(x_values, y_values, label=metric)
                data_found = True
        if not data_found:
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
        if svg:
            from IPython.display import SVG
            bytes = io.BytesIO()
            plt.savefig(bytes, format='svg')
            img_bytes = bytes.getvalue()
            plt.close(fig)
            return SVG(img_bytes.decode())
        else:
            plt.show(block=False)

    def plot_loss_acc(self, callback, epoch):
        """plots loss and accuracy on separate graphs, ignoring any other metrics"""
        #print("called on_epoch_end with epoch =", epoch)
        metrics = self.get_metrics()
        if callback.figure is not None:
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
            callback.figure = fig, loss_ax, acc_ax
        x_values = range(self.epoch_count+1)
        for metric in metrics:
            y_values = [epoch[metric] if metric in epoch else None for epoch in self.history]
            if metric == 'loss':
                loss_ax.plot(x_values, y_values, label='Training set')
            elif metric == 'val_loss':
                loss_ax.plot(x_values, y_values, label='Validation set')
            elif metric == 'acc' and acc_ax is not None:
                acc_ax.plot(x_values, y_values, label='Training set')
            elif metric == 'val_acc' and acc_ax is not None:
                acc_ax.plot(x_values, y_values, label='Validation set')
        loss_ax.set_ylim(bottom=0)
        loss_ax.set_title("%s: Loss" % (self.name,))
        loss_ax.set_xlabel('Epoch')
        loss_ax.legend(loc='best')
        if acc_ax is not None:
            acc_ax.set_ylim([-0.1, 1.1])
            acc_ax.set_title("%s: Accuracy" % (self.name,))
            acc_ax.set_xlabel('Epoch')
            acc_ax.legend(loc='best')
        if not callback.in_console:
            from IPython.display import SVG, clear_output, display
            bytes = io.BytesIO()
            plt.savefig(bytes, format='svg')
            img_bytes = bytes.getvalue()
            clear_output(wait=True)
            display(SVG(img_bytes.decode()))
            #return SVG(img_bytes.decode())
        else:
            plt.pause(0.01)
            #plt.show(block=False)

    def compile(self, **kwargs):
        """
        Check and compile the network.

        See https://keras.io/ `Model.compile()` method for more details.
        """
        import keras.backend as K
        ## Error checking:
        if len(self.layers) == 0:
            raise Exception("network has no layers")
        for layer in self.layers:
            if layer.kind() == 'unconnected':
                raise Exception("'%s' layer is unconnected" % layer.name)
        if "error" in kwargs: # synonym
            kwargs["loss"] = kwargs["error"]
            del kwargs["error"]
        if "loss" in kwargs and kwargs["loss"] == 'sparse_categorical_crossentropy':
            raise Exception("'sparse_categorical_crossentropy' is not a valid error metric in conx; use 'categorical_crossentropy' with proper targets")
        if "optimizer" in kwargs:
            optimizer = kwargs["optimizer"]
            if (not ((isinstance(optimizer, str) and optimizer in self.OPTIMIZERS) or
                     (isinstance(optimizer, object) and issubclass(optimizer.__class__, keras.optimizers.Optimizer)))):
                raise Exception("invalid optimizer '%s'; use valid function or one of %s" %
                                (optimizer, Network.OPTIMIZERS,))
        ## FIXME: redo checks to separate dataset:
        # if len(input_layers) == 1 and len(self.input_layer_order) == 0:
        #     pass # ok!
        # elif len(input_layers) == len(self.dataset._input_layer_order):
        #     # check to make names all match
        #     for layer in input_layers:
        #         if layer.name not in self.dataset._input_layer_order:
        #             raise Exception("layer '%s' is not listed in dataset._input_layer_order" % layer.name)
        # else:
        #     raise Exception("improper dataset._input_layer_order names")
        ## FIXME: add new dataset-based checks:
        # if len(output_layers) == 1 and len(self.output_layer_order) == 0:
        #     pass # ok!
        # elif len(output_layers) == len(self.output_layer_order):
        #     # check to make names all match
        #     for layer in output_layers:
        #         if layer.name not in self.output_layer_order:
        #             raise Exception("layer '%s' is not listed in set_output_layer_order()" % layer.name)
        # else:
        #     raise Exception("improper set_output_layer_order() names")
        self._build_intermediary_models()
        output_k_layers = self._get_output_ks_in_order()
        input_k_layers = self._get_input_ks_in_order(self.input_bank_order)
        self.model = keras.models.Model(inputs=input_k_layers, outputs=output_k_layers)
        if "metrics" in kwargs and kwargs["metrics"] is not None:
            pass ## ok allow override
        else:
            kwargs['metrics'] = [self.acc] ## the default
        self.compile_options = copy.copy(kwargs)
        self.model.compile(**kwargs)
        # set each conx layer to point to corresponding keras model layer
        for layer in self.layers:
            layer.keras_layer = self._find_keras_layer(layer.name)

    def acc(self, targets, outputs):
        # This is only used on non-multi-output-bank training:
        import keras.backend as K
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
            layer.input_names = []
            layer.model = None

    def update_model(self):
        """
        Useful if you change, say, an activation function after training.
        """
        self._build_intermediary_models()

    def generate_keras_script(self):
        sequence = topological_sort(self, self.layers)
        program = "## Autogenerated by conx\n\n"
        program += "import keras\n\n"
        program += "kfunc = {} # dictionary to keep track of k's by layer\n\n"
        for layer in sequence:
            if layer.kind() == 'input':
                ## an input vector:
                program += "kfunc['%s'] = %s\n" % (layer.name, layer.make_input_layer_k_text())
            else:
                if len(layer.incoming_connections) == 1:
                    program += "k = kfunc['%s']\n" % layer.incoming_connections[0].name
                else: # multiple inputs, need to merge
                    program += "k = keras.layers.Concatenate()([kfunc[layer] for layer in %s])\n" % [layer.name for layer in layer.incoming_connections]
                kfuncs = layer.make_keras_functions_text()
                program += "for f in %s:\n" % kfuncs
                program += "    k = f(k)\n"
                program += "kfunc['%s'] = k\n" % layer.name
        input_ks = ",".join(["kfunc['%s']" % name for name in self.input_bank_order])
        output_ks = ",".join(["kfunc['%s']" % name for name in self.output_bank_order])
        ## FIXME: not a list, if only one?
        program += "model = keras.models.Model(inputs=[%s], outputs=[%s])\n" % (input_ks, output_ks)
        return program

    def _build_intermediary_models(self):
        """
        Construct the layer.k, layer.input_names, and layer.model's.
        """
        sequence = topological_sort(self, self.layers)
        if self.debug: print("topological sort:", [l.name for l in sequence])
        for layer in sequence:
            if layer.kind() == 'input':
                if self.debug: print("making input layer for", layer.name)
                layer.k = layer.make_input_layer_k()
                layer.input_names = [layer.name]
                layer.model = keras.models.Model(inputs=layer.k, outputs=layer.k) # identity
            else:
                if self.debug: print("making layer for", layer.name)
                if len(layer.incoming_connections) == 0:
                    raise Exception("non-input layer '%s' with no incoming connections" % layer.name)
                kfuncs = layer.make_keras_functions()
                if len(layer.incoming_connections) == 1:
                    if self.debug: print("single input", layer.incoming_connections[0])
                    k = layer.incoming_connections[0].k
                    layer.input_names = layer.incoming_connections[0].input_names
                else: # multiple inputs, need to merge
                    if self.debug: print("Merge detected!", [l.name for l in layer.incoming_connections])
                    k = keras.layers.Concatenate()([incoming.k for incoming in layer.incoming_connections])
                    # flatten:
                    layer.input_names = [item for sublist in
                                         [incoming.input_names for incoming in layer.incoming_connections]
                                         for item in sublist]
                if self.debug: print("input names for", layer.name, layer.input_names)
                if self.debug: print("applying k's", kfuncs)
                for f in kfuncs:
                    k = f(k)
                layer.k = k
                ## get the inputs to this branch, in order:
                input_ks = self._get_input_ks_in_order(layer.input_names)
                layer.model = keras.models.Model(inputs=input_ks, outputs=layer.k)
                ## IS THIS A BETTER WAY?:
                #layer.model = keras.models.Model(inputs=input_ks, outputs=layer.keras_layer.output)

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

    def build_struct(self, inputs, class_id, config):
        ordering = list(reversed(self._get_level_ordering())) # list of names per level, input to output
        ### find max_width, image_dims, and row_height
        max_width = 0
        images = {}
        image_dims = {}
        row_height = []
        # Go through and build images, compute max_width:
        for level_tups in ordering: ## output to input:
            # first make all images at this level
            total_width = 0 # for this row
            max_height = 0
            for (layer_name, anchor, fname) in level_tups:
                if not self[layer_name].visible or anchor: # not need to handle anchors here
                    continue
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
                image = self.propagate_to_image(layer_name, v, visualize=False)
                (width, height) = image.size
                images[layer_name] = image ## little image
                max_dim = max(width, height)
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
                pwidth, pheight = np.array(image.size) * image_pixels_per_unit
                if max(pwidth, pheight) < image_maxdim:
                    width, height = pwidth, pheight
                else:
                    width, height = (int(width/max_dim * image_maxdim),
                                     int(height/max_dim * image_maxdim))
                # make sure not too small:
                if min(width, height) < 25:
                    width, height = (image_maxdim, 25)
                image_dims[layer_name] = (width, height)
                total_width += width + config["hspace"] # space between
                max_height = max(max_height, height)
            row_height.append(max_height)
            max_width = max(max_width, total_width)
        ### Now that we know the dimensions:
        struct = []
        ## Draw the title:
        struct.append(["label_svg", {"x": max_width/2,
                                     "y": config["border_top"]/2,
                                     "label": self.name,
                                     "font_size": config["font_size"] + 3,
                                     "font_family": config["font_family"],
                                     "text_anchor": "middle",
        }])
        cheight = config["border_top"] # top border
        ## Display targets?
        if config["show_targets"]:
            # Find the spacing for row:
            for (layer_name, anchor, fname) in ordering[0]:
                if not self[layer_name].visible:
                    continue
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
            spacing = max_width / (len(ordering[0]) + 1)
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
                                             "font_family": config["font_family"],
                                             "text_anchor": "start",
                }])
                cwidth += width/2
            ## Then we need to add height for output layer again, plus a little bit
            cheight += row_height[0] + 10 # max height of row, plus some
        ## Display error?
        if config["show_errors"]:
            # Find the spacing for row:
            for (layer_name, anchor, fname) in ordering[0]: # no anchors in output
                if not self[layer_name].visible:
                    continue
                image = images[layer_name]
                (width, height) = image_dims[layer_name]
            spacing = max_width / (len(ordering[0]) + 1)
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
                                             "font_family": config["font_family"],
                                             "text_anchor": "start",
                }])
                cwidth += width/2
            ## Then we need to add height for output layer again, plus a little bit
            cheight += row_height[0] + 10 # max height of row, plus some
        # Now we go through again and build SVG:
        positioning = {}
        level_num = 0
        for level_tups in ordering:
            spacing = max_width / (len(level_tups) + 1)
            cwidth = 0
            # See if there are any connections up:
            any_connections_up = False
            last_connections_up = False
            for (layer_name, anchor, fname) in level_tups:
                if not self[layer_name].visible or anchor:
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
            for (layer_name, anchor, fname) in level_tups:
                if not self[layer_name].visible:
                    continue
                if anchor:
                    anchor_name = "%s-%s-anchor%s" % (layer_name, fname, level_num)
                    cwidth += spacing
                    positioning[anchor_name] = {"x": cwidth, "y": cheight}
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
                                                        "arrow_color": config["arrow_color"] if self[fname].dropout == 0 else "red",
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
                                                         "arrow_color": config["arrow_color"] if self[fname].dropout == 0 else "red",
                                                         "tooltip": tooltip
                            }])
                    else:
                        print("that's weird!", layer_name, "is not in", prev)
                    continue
                else:
                    image = images[layer_name]
                    (width, height) = image_dims[layer_name]
                    cwidth += (spacing - (width/2))
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
                #### Background rects for arrow mouseovers
                # for out in self[layer_name].outgoing_connections:
                #     if out.name not in positioning:
                #         continue
                #     # draw background to arrows to allow mouseover tooltips:
                #     x2 = positioning[out.name]["x"] + positioning[out.name]["width"]/2
                #     y2 = positioning[out.name]["y"] + positioning[out.name]["height"]
                #     rect_width = abs(x1 - x2)
                #     rect_extra = 0
                #     if rect_width < 20:
                #         rect_extra = 10
                #     tooltip = html.escape(self.describe_connection_to(self[layer_name], out))
                #     svg += arrow_rect.format(**{"tooltip": tooltip,
                #                                 "rx": min(x2, x1) - rect_extra,
                #                                 "ry": min(y2, y1) + 2, # bring down
                #                                 "rw": rect_width + rect_extra * 2,
                #                                 "rh": abs(y1 - y2) - 2})
                # Draw all of the connections up from here:
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
                                                    "arrow_color": config["arrow_color"] if self[layer_name].dropout == 0 else "red",
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
                                                     "arrow_color": config["arrow_color"] if self[layer_name].dropout == 0 else "red",
                                                     "tooltip": tooltip
                        }])
                struct.append(["image_svg", positioning[layer_name]])
                struct.append(["label_svg", {"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] + 5,
                                             "y": positioning[layer_name]["y"] + positioning[layer_name]["height"]/2 + 2,
                                             "label": layer_name,
                                             "font_size": config["font_size"],
                                             "font_family": config["font_family"],
                                             "text_anchor": "start",
                }])
                output_shape = self[layer_name].keras_layer.output_shape
                if (isinstance(output_shape, tuple) and len(output_shape) == 4 and
                    "ImageLayer" != self[layer_name].__class__.__name__):
                    features = str(output_shape[3])
                    feature = str(self[layer_name].feature)
                    struct.append(["label_svg", {"x": positioning[layer_name]["x"] + positioning[layer_name]["width"] + 5,
                                                 "y": positioning[layer_name]["y"] + 5,
                                                 "label": features,
                                                 "font_size": config["font_size"],
                                                 "font_family": config["font_family"],
                                                 "text_anchor": "start",
                    }])
                    struct.append(["label_svg", {"x": positioning[layer_name]["x"] - (len(feature) * 7) - 5,
                                                 "y": positioning[layer_name]["y"] + positioning[layer_name]["height"] - 5,
                                                 "label": feature,
                                                 "font_size": config["font_size"],
                                                 "font_family": config["font_family"],
                                                 "text_anchor": "start",
                    }])
                cwidth += width/2
                max_height = max(max_height, height)
                self._svg_counter += 1
            cheight += max_height
            level_num += 1
        cheight += config["border_bottom"]
        struct.append(["svg_head", {
            "svg_height": config["svg_height"],
            "width": max_width,  # view port width
            "height": cheight,   # view port height
            "netname": self.name,
            "arrow_color": config["arrow_color"],
            "arrow_width": config["arrow_width"],
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

    def build_svg(self, inputs=None, class_id=None, opts={}):
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
        if any([(layer.kind() == "unconnected") for layer in self.layers]):
            raise Exception("can't build display with layers that aren't connected; use Network.connect(...)")
        if self.model is None:
            raise Exception("can't build display before Network.compile(...) as been run")
        self.visualize = False # so we don't try to update previously drawn images
        # defaults:
        config = copy.copy(self.config)
        config.update(opts)
        struct = self.build_struct(inputs, class_id, config)
        ### Define the SVG strings:
        image_svg = """<rect x="{{rx}}" y="{{ry}}" width="{{rw}}" height="{{rh}}" style="fill:none;stroke:{border_color};stroke-width:{border_width}"/><image id="{netname}_{{name}}_{{svg_counter}}" class="{netname}_{{name}}" x="{{x}}" y="{{y}}" height="{{height}}" width="{{width}}" preserveAspectRatio="none" href="{{image}}"><title>{{tooltip}}</title></image>""".format(
            **{
                "netname": class_id if class_id is not None else self.name,
                "border_color": config["border_color"],
                "border_width": config["border_width"],
            })
        line_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}"><title>{{tooltip}}</title></line>""".format(**self.config)
        arrow_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}" marker-end="url(#arrow)"><title>{{tooltip}}</title></line>""".format(**self.config)
        arrow_rect = """<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" style="fill:white;stroke:none"><title>{tooltip}</title></rect>"""
        label_svg = """<text x="{x}" y="{y}" font-family="{font_family}" font-size="{font_size}" text-anchor="{text_anchor}" alignment-baseline="central">{label}</text>"""
        svg_head = """<svg id='{netname}' xmlns='http://www.w3.org/2000/svg' viewBox="0 0 {width} {height}" height="{svg_height}" image-rendering="pixelated">
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
            if template_name != "svg_head":
                t = templates[template_name]
                svg += t.format(**dict)
        svg += """</svg>"""
        self.visualize = True
        if get_ipython():
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
            def input_index(name):
                return min([self.input_bank_order.index(iname) for iname in self[name].input_names])
            lev = sorted([(input_index(fname if anchor else name), name, anchor, fname) for (name, anchor, fname) in ordering[level]])
            ordering[level] = [(name, anchor, fname) for (index, name, anchor, fname) in lev]
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
                    retval += "\n %s has shape %s" % (
                        klayer.weights[w].name, weights[w].shape)
        ## FIXME: how to show merged layer weights?
        return retval

    def load(self, dir=None):
        self.load_model(dir)
        self.load_weights(dir)

    def save(self, dir=None):
        if self.model:
            self.save_model(dir)
            self.save_weights(dir)
        else:
            raise Exception("need to compile network before saving")

    def load_model(self, dir=None, filename=None):
        from keras.models import load_model
        if dir is None:
            dir = "%s.conx" % self.name.replace(" ", "_")
        if filename is None:
            filename = "model.h5"
        self.model = load_model(os.path.join(dir, filename))
        if self.compile_options:
            self.reset()

    def save_model(self, dir=None, filename=None):
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
        Load the history from a file.

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
                self.epoch_count = (len(self.history) - 1) if self.history else 0
        else:
            print("WARNING: no such history file '%s'" % full_filename)

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

    def load_weights(self, dir=None, filename=None):
        """
        Load the network weights from a file.

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
        Save the network weights to a file.

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
        from .dashboard import Dashboard
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

        Parameters:
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

            >>> net.pf([0]*10000)
            '[0,0,0,..., 0,0,0]'
        """
        from IPython.lib.pretty import pretty
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


class _InterruptHandler():
    """
    Class for handling interrupts so that state is not left
    in inconsistant situation.
    """
    def __init__(self, network, sig=signal.SIGINT):
        self.network = network
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
