from __future__ import print_function, division

# conx - a neural network library
#
# Copyright (c) 2016 Douglas S. Blank <dblank@cs.brynmawr.edu>
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

import theano
import theano.tensor as T
from theano import function, pp
import theano.tensor.nnet as nnet
import numpy as np
import random

__version__ = "0.0.4"

class Network():
    """
    Backpropagation of error neural network on top of
    Theano.
    """
    def __init__(self, *sizes, **kwargs):
        """
        Initialize network, given sizes of layers from
        input to output (inclusive).
        """
        self.sizes = sizes
        epsilon = 0.1
        for key in kwargs:
            if key == "epsilon":
                epsilon = kwargs["epsilon"]
            else:
                raise Exception("unknown argument: '%s'" % key)
        # learning rate
        self._epsilon = theano.shared(epsilon, name='epsilon')
        self.th_inputs = T.dvector('inputs') # inputs
        self.th_target = T.dvector('target') # target/output
        self.weights = []
        self.layer = []
        for i in range(len(self.sizes) - 1):
            self.weights.append(
                theano.shared(self.make_weights(self.sizes[i],
                                                self.sizes[i+1]),
                              name='self.weights[%s]' % i))
            if i == 0:
                self.layer.append(
                    self.compute_activation(self.th_inputs,
                                            self.weights[0]))
            else:
                self.layer.append(
                    self.compute_activation(self.layer[i - 1],
                                            self.weights[i]))
        # Theano function:
        self.compute_error = T.sum((self.layer[-1] - self.th_target) ** 2)
        # Dynamic Methods:
        self.train_one = function(
            inputs=[self.th_inputs, self.th_target],
            outputs=self.compute_error,
            updates=self.update_weights())
        self._propagate = function([self.th_inputs], self.layer[-1])
        self.compute_target = None
        # Properties:
        self.inputs = None
        self.targets = None

    def propagate(self, inputs):
        return self._propagate(inputs)

    def update_weights(self):
        """
        Returns [(weights, Theano update function), ...]
        """
        updates = []
        for i in range(len(self.weights)):
            updates.append(
                (self.weights[i],
                 self.compute_delta_weights(self.compute_error,
                                            self.weights[i])))
        return updates

    def make_weights(self, ins, outs):
        """
        Makes a 2D matrix of random weights centered around 0.0.
        """
        # Add one to ins for bias units:
        return np.array(2 * np.random.rand(ins + 1, outs) - 1,
                      dtype='float64')

    def set_epsilon(self, epsilon):
        """
        Method to set epsilon (learning rate).
        """
        self._epsilon.set_value(epsilon)

    def get_epsilon(self):
        """
        Method to get epsilon (learning rate).
        """
        return self._epsilon.get_value()

    epsilon = property(get_epsilon, set_epsilon)

    def compute_activation(self, inputs, weights):
        """
        Theano function to compute activation at a layer.
        """
        bias = np.array([1], dtype='float64')
        all_inputs = T.concatenate([inputs, bias])
        net_input = T.dot(weights.T, all_inputs)
        activation = nnet.sigmoid(net_input)
        return activation

    def compute_delta_weights(self, compute_error, weights):
        """
        Theano function to change weights.
        """
        return weights - (self._epsilon * T.grad(compute_error, wrt=weights))

    def train(self, max_epoch=5000, stop_percentage=None, tolerance=0.1,
             report_rate=500):
        """
        Method to train network.
        """
        for e in range(max_epoch):
            total = 0
            correct = 0
            random.shuffle(self.inputs)
            for i in range(len(self.inputs)):
                if hasattr(self.inputs[0][0], "__len__"):
                    # inputs is input and target
                    target = self.inputs[i][1]
                    inputs = self.inputs[i][0]
                else:
                    target = self.compute_target(self.inputs[i])
                    inputs = self.inputs[i]
                error = self.train_one(inputs, target)
                total += 1
                output = self._propagate(inputs)
                if np.sum(np.abs(output - target)) <= tolerance:
                    correct += 1
            if (e + 1) % report_rate == 0 or e == 0:
                print('Epoch:', e + 1,
                      'TSS error:', error,
                      '%correct:', correct/total)
            if stop_percentage is not None:
                if correct/total >= stop_percentage:
                    break
        print("-" * 50)
        print('Epoch:', e + 1,
              'TSS error:', error,
              '%correct:', correct/total)

    def test(self, stop=None, start=0):
        """
        Method to test network.
        """
        if stop is None:
            stop = len(self.inputs)
        for inputs in self.inputs[start:stop]:
            if self.compute_target:
                target = self.compute_target(inputs)
            else:
                # inputs is input and target
                target = inputs[1]
                inputs = inputs[0]
            output = self.propagate(inputs)
            self.display_input(inputs)
            self.display_output(output)

    def reset(self):
        """
        Reset the weights of a network.
        """
        for i in range(len(self.sizes) - 1):
            self.weights[i].set_value(
                self.make_weights(self.sizes[i], self.sizes[i+1]))

    def set_inputs(self, inputs):
        """
        Set the inputs to train on.
        """
        self.inputs = inputs

    def set_target_function(self, func):
        """
        Set the target function, if one.
        """
        self.compute_target = func

    def change_layer_size(self, layer, size):
        """
        Change the size of a layer. Should call
        reset at some point after this.
        """
        self.sizes[layer] = size

    def display_input(self, v):
        """
        Method to display input pattern.
        """
        print("Input:", v)

    def display_output(self, v):
        """
        Method to display output pattern.
        """
        print("Output:", v)
        print()

class SRN(Network):
    """
    Simple Recurrent Network
    """
    def __init__(self, *sizes, **kwargs):
        """
        Initialize network, given sizes of layers from
        input to output (inclusive).
        """
        self.sizes = sizes
        epsilon = 0.1
        for key in kwargs:
            if key == "epsilon":
                epsilon = kwargs["epsilon"]
            else:
                raise Exception("unknown argument: '%s'" % key)
        # learning rate
        self._epsilon = theano.shared(epsilon, name='epsilon')
        self.th_inputs = T.dvector('inputs') # inputs
        self.th_target = T.dvector('target') # target/output
        self.weights = []
        self.layer = []
        self.context_layer = []
        self.context_weights = []
        # Just the hidden layers:
        for i in range(1,len(self.sizes) - 1):
            self.context_weights.append(
                theano.shared(
                    self.make_weights(self.sizes[i],
                                      self.sizes[i]),
                    name='self.context_weights[%s]' % (i - 1))
            )
            self.context_layer.append(
                theano.shared(
                    np.array([0.5] * self.sizes[i],
                             dtype='float64'),
                    name="self.context_layer[%s]" % (i - 1))
            )
        for i in range(len(self.sizes) - 1):
            self.weights.append(
                theano.shared(self.make_weights(self.sizes[i],
                                                self.sizes[i+1]),
                              name='self.weights[%s]' % i))
            if i == 0:
                self.layer.append(
                    self.compute_activation(self.th_inputs,
                                            self.context_layer[0],
                                            self.context_weights[0],
                                            self.weights[0]))
            elif i < len(self.sizes) - 2:
                self.layer.append(
                    self.compute_activation(self.layer[i - 1],
                                            self.context_layer[i],
                                            self.context_weights[i],
                                            self.weights[i]))
            else:
                self.layer.append(
                    self.compute_activation(self.layer[i - 1],
                                            None,
                                            None,
                                            self.weights[i]))
        # Theano function:
        self.compute_error = T.sum((self.layer[-1] - self.th_target) ** 2)
        # Dynamic Methods:
        self.train_one = function(
            inputs=[self.th_inputs, self.th_target],
            outputs=self.compute_error,
            updates=self.update_weights())
        self._propagate = function([self.th_inputs], self.layer[-1])
        self.compute_target = None
        # Properties:
        self.inputs = None
        self.targets = None

    def compute_activation(self, inputs, context, cweights, weights):
        """
        Theano function to compute activation at a layer.
        """
        bias = np.array([1], dtype='float64')
        all_inputs = T.concatenate([inputs, bias])
        if context:
            all_context = T.concatenate([context, bias])
            net_input = (T.dot(weights.T, all_inputs) +
                         T.dot(cweights.T, all_context))
        else:
            net_input = T.dot(weights.T, all_inputs)
        activation = nnet.sigmoid(net_input)
        return activation

    def propagate(self, inputs):
        """
        Propagate activation, and copy contexts.
        """
        retval = self._propagate(inputs)
        self.copy_context(inputs)
        return retval

    def copy_context(self, inputs):
        """
        Copy hidden activations to context layers.
        """
        for i in range(len(self.context_layer)):
            value = function([self.th_inputs], self.layer[i])(inputs)
            self.context_layer[i].set_value(value)

    def update_weights(self):
        """
        Returns [(weights, Theano update function), ...]
        """
        updates = Network.update_weights(self)
        # Hidden layers:
        for i in range(len(self.context_layer)):
            updates.append(
                (self.context_layer[i], self.layer[i])
            )
            # FIXME: can I backprop error through both
            # context_weights and weights?
            updates.append(
                (self.context_weights[i],
                 self.compute_delta_weights(self.compute_error,
                                            self.context_weights[i]))
            )
        return updates
