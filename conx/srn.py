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

from .network import Network, Layer

class SRNLayer(Layer):
    def __init__(self, n_input, n_output, activation_function):
        cweights = self.make_weights(n_output, n_output)
        self.cweights = theano.shared(
            value=cweights.astype(theano.config.floatX),
            # The name parameter is solely for printing purporses
            name='self.cweights',
            # Setting borrow=True allows Theano to use user memory for this object.
            # It can make code slightly faster by avoiding a deep copy on construction.
            # For more details, see
            # http://deeplearning.net/software/theano/tutorial/aliasing.html
            borrow=True
        )
        # Initialize context to 0.5:
        last_outputs = np.array([0.5] * n_output,
                                dtype=theano.config.floatX)
        self.last_outputs = theano.shared(
            value=last_outputs.astype(theano.config.floatX),
            name="self.last_outputs",
            borrow=True
        )
        super(SRNLayer, self).__init__(n_input, n_output, activation_function)
        self.params += [self.cweights]
        inputs = T.vector(dtype=theano.config.floatX)
        # Replaces Layer's _pypropagate:
        self._pypropagate = function([inputs], self._propagate(inputs),
                                     allow_input_downcast=True)

    def propagate(self, inputs):
        """
        Overrides Layer's propagate method to save the outputs.
        """
        outputs = self._pypropagate(inputs)
        self.last_outputs.set_value(outputs)
        return outputs

    def _propagate(self, inputs):
        '''
        Compute this layer's output given an input

        :parameters:
            - inputs : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated inputs
        '''
        # Compute linear mix
        lin_output = (T.dot(self.weights, inputs) + self.biases +
                      T.dot(self.cweights, self.last_outputs))
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        activation = (lin_output if self.activation_function is None
                      else self.activation_function(lin_output))
        return activation

    def reset(self):
        """
        Resets a layer's learned values and context.
        """
        super(SRNLayer, self).reset()
        self.last_outputs.set_value([0.5] * n_output)
        out_size, in_size = self.cweights.get_value().shape
        self.cweights.set_value(
            self.make_weights(out_size, out_size)
        )

    def change_size(self, ins, outs):
        """
        Change the size of the weights/biases for this layer.
        """
        super(SRNLayer, self).change_size(ins, outs)
        self.cweights.set_value(self.make_weights(outs, outs))


class SRN(Network):
    """
    Create a Elman-style recurrent network. All hidden layers
    get a context layer.
    """
    def make_layers(self, sizes):
        """
        Puts a context layer on hidden layers.
        """
        self.layer = [] # [0, 1, 2], [1, 2, 3] for 4 layers
        i = 0
        for n_input, n_output in zip(sizes[:-1], sizes[1:]):
            if i < len(sizes) - 1: # hidden
                self.layer.append(SRNLayer(n_input, n_output, self.settings["activation_function"]))
            else:
                self.layer.append(Layer(n_input, n_output, self.settings["activation_function"]))
            i += 1

    def propagate(self, inputs, copy_context=True):
        """
        This override explicitly calls the propagate in each layer
        so to copy the context values.
        """
        if copy_context:
            activations = inputs
            for layer in self.layer:
                activations = layer.propagate(activations)
        else:
            activations = self._pypropagate(inputs)
        return activations
