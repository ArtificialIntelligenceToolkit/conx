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

from .network import Network

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
