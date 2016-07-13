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

class Layer(object):
    def __init__(self, weights, biases, activation_function):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a
        nonlinearity and x is the input vector.

        :parameters:
            - weights : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - biases : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation_function : theano.tensor.elemwise.Elemwise
                Activation function for layer output

        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_output, n_input = weights.shape
        # Make sure b is n_output in size
        assert biases.shape == (n_output,)
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output,
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that weights has the theano.config.floatX dtype
        self.weights = theano.shared(
            value=weights.astype(theano.config.floatX),
            # The name parameter is solely for printing purporses
            name='self.weights',
            # Setting borrow=True allows Theano to use user memory for this object.
            # It can make code slightly faster by avoiding a deep copy on construction.
            # For more details, see
            # http://deeplearning.net/software/theano/tutorial/aliasing.html
            borrow=True
        )
        # We can force our bias vector b to be a column vector using numpy's reshape method.
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting (described below)
        self.biases = theano.shared(
            value=biases.reshape(n_output, 1).astype(theano.config.floatX),
            name='self.biases',
            borrow=True,
            # Theano allows for broadcasting, similar to numpy.
            # However, you need to explicitly denote which axes can be broadcasted.
            # By setting broadcastable=(False, True), we are denoting that b
            # can be broadcast (copied) along its second dimension in order to be
            # added to another variable.  For more information, see
            # http://deeplearning.net/software/theano/library/tensor/basic.html
            broadcastable=(False, True)
        )
        self.activation_function = activation_function
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.weights, self.biases]

    def _propagate(self, inputs):
        '''
        Compute this layer's output given an input

        :parameters:
            - inouts : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated inputs
        '''
        # Compute linear mix
        lin_output = T.dot(self.weights, inputs) + self.biases
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation_function is None
                else self.activation_function(lin_output))

class Network():
    def __init__(self, *sizes, **kwargs):
        '''
        Multi-layer perceptron class, computes the composition of a
        sequence of Layers

        :parameters:
            - weights : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in weights
            - biases : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activation_functions : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer

        '''
        activation_function = T.nnet.sigmoid
        epsilon = 0.1
        momentum = 0.9
        for key in kwargs:
            if key == "activation_function":
                activation_function = kwargs[key]
            elif key == "epsilon":
                activation_function = kwargs[key]
            elif key == "momentum":
                activation_function = kwargs[key]
            else:
                raise Exception("unknown argument: '%s'" % key)
        self._epsilon = theano.shared(epsilon, name='self.epsilon')
        self._momentum = theano.shared(momentum, name='self.momentum')
        weights = []
        biases = []
        activation_functions = []
        for n_input, n_output in zip(sizes[:-1], sizes[1:]):
            weights.append(self.make_weights(n_input, n_output))
            # Set initial biases to 1
            biases.append(np.ones(n_output))
            # We'll use sigmoid activation for all layers
            # Note that this doesn't make a ton of sense when using squared distance
            # because the sigmoid function is bounded on [0, 1].
            activation_functions.append(activation_function)

        # Make sure the input lists are all of the same length
        assert len(weights) == len(biases) == len(activation_functions)

        # Initialize lists of layers
        self.layer = []
        # Construct the layers
        for weight, bias, activation_function in zip(weights, biases, activation_functions):
            self.layer.append(Layer(weight, bias, activation_function))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layer:
            self.params += layer.params # [weights, biases]

        # Create Theano variables for the MLP input
        self.th_inputs = T.matrix('self.th_inputs')
        # ... and the desired output
        self.th_targets = T.vector('self.th_targets')

        error = self.tss_error(self.th_inputs, self.th_targets)
        # Dynamic Python methods:
        self._pytrain_one = function(
            [self.th_inputs, self.th_targets],
            error,
            updates=self.compute_delta_weights(error))
        self._pypropagate = function([self.th_inputs], self._propagate(self.th_inputs))
        self.compute_target = None

    def train_one(self, inputs, targets):
        return self._pytrain_one(self.make_matrix(inputs), targets)

    def propagate(self, inputs):
        return self._pypropagate(self.make_matrix(inputs))

    def compute_delta_weights(self, error):
        '''
        Compute updates for gradient descent with momentum

        :returns:
            updates : list
                List of updates, one for each parameter
        '''
        # Make sure momentum is a sane value
        assert self.momentum < 1 and self.momentum >= 0
        # List of update steps for each parameter
        updates = []
        # Just gradient descent on cost
        for param in self.params: # [weights, biases]
            # For each parameter, we'll create a param_update shared variable.
            # This variable will keep track of the parameter's update step across iterations.
            # We initialize it to 0
            param_update = theano.shared(param.get_value() * 0.,
                                         broadcastable=param.broadcastable)
            # Each parameter is updated by taking a step in the direction of the gradient.
            # However, we also "mix in" the previous step according to the given momentum value.
            # Note that when updating param_update, we are using its old value and also the new gradient step.
            updates.append((param, param - self._epsilon * param_update))
            # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
            updates.append((param_update,
                            self._momentum * param_update + (1. - self._momentum) * T.grad(error, param)))
        return updates

    def _set_epsilon(self, epsilon):
        """
        Method to set epsilon (learning rate).
        """
        self._epsilon.set_value(epsilon)

    def _get_epsilon(self):
        """
        Method to get epsilon (learning rate).
        """
        return self._epsilon.get_value()

    def _set_momentum(self, momentum):
        """
        Method to set momentum.
        """
        self._momentum.set_value(momentum)

    def _get_momentum(self):
        """
        Method to get mementum.
        """
        return self._momentum.get_value()

    epsilon = property(_get_epsilon, _set_epsilon)
    momentum = property(_get_momentum, _set_momentum)

    def _propagate(self, inputs):
        '''
        Compute the MLP's output given an input

        :parameters:
            - inputs : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                inputs passed through the MLP
        '''
        # Recursively compute output
        activations = inputs
        for layer in self.layer:
            activations = layer._propagate(activations)
        return activations

    def tss_error(self, inputs, targets):
        '''
        Compute the squared euclidean error of the network output against the "true" output y

        :parameters:
            - inputs : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - targets : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self._propagate(inputs) - targets)**2)

    def make_weights(self, ins, outs):
        """
        Makes a 2D matrix of random weights centered around 0.0.
        """
        return np.array(2 * np.random.rand(outs, ins) - 1,
                        dtype=theano.config.floatX)

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
                if self.compute_target:
                    target = self.compute_target(self.inputs[i])
                    inputs = self.inputs[i]
                else:
                    # inputs is input and target
                    target = self.inputs[i][1]
                    inputs = self.inputs[i][0]
                error = self.train_one(inputs, target)
                total += 1
                output = self.propagate(inputs)
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
                inputs = inputs[0]
                target = inputs[1]
            output = self.propagate(inputs)
            self.display_input(inputs)
            self.display_output(output)

    def reset(self):
        """
        Reset the weights of a network.
        """
        for i in range(len(self.layer)):
            out_size, in_size = self.layer[i].weights.get_value().shape
            self.layer[i].weights.set_value(
                self.make_weights(in_size, out_size)
            )

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

    def make_matrix(self, vector):
        return [[v] for v in vector]
