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
import copy

# Based on code from:
# http://colinraffel.com/talks/next2015theano.pdf
# http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb#example-mlp
# http://mlg.eng.cam.ac.uk/yarin/620bdeb168a59f1b072e4173ac867e79/Ex4_MLP_answer.py

class Layer(object):
    def __init__(self, n_input, n_output, activation_function):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a
        nonlinearity and x is the input vector.

        Properties:
            - weights : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - biases : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation_function : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        weights = self.make_weights(n_input, n_output)
        biases = np.ones(n_output, dtype=theano.config.floatX)

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
            value=biases.astype(theano.config.floatX),
            name='self.biases',
            borrow=True,
            # Theano allows for broadcasting, similar to numpy.
            # However, you need to explicitly denote which axes can be broadcasted.
            # By setting broadcastable=(False, True), we are denoting that b
            # can be broadcast (copied) along its second dimension in order to be
            # added to another variable.  For more information, see
            # http://deeplearning.net/software/theano/library/tensor/basic.html
            #broadcastable=(False, True)
        )
        self.activation_function = activation_function
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.weights, self.biases]
        self.n_output, self.n_input = (n_output, n_input)
        # Dynamic functions
        inputs = T.vector(dtype=theano.config.floatX)
        self._pypropagate = function([inputs], self._propagate(inputs),
                                     allow_input_downcast=True)

    def __str__(self):
        retval = "    Type: %s\n" % type(self)
        retval += "    Act : %s\n" % self.activation_function
        retval += "    In  : %s\n" % self.n_input
        retval += "    Out : %s\n" % self.n_output
        return retval

    def propagate(self, inputs):
        """
        Layer's propagate method. May be overridden in subclasses.
        """
        return self._pypropagate(inputs)

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
        lin_output = T.dot(self.weights, inputs) + self.biases
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation_function is None
                else self.activation_function(lin_output))

    def make_weights(self, ins, outs):
        """
        Makes a 2D matrix of random weights centered around 0.0.
        """
        return np.array(2 * np.random.rand(outs, ins) - 1,
                        dtype=theano.config.floatX)

    def reset(self):
        """
        Resets a layer's learned values.
        """
        out_size, in_size = self.weights.get_value().shape
        self.weights.set_value(
            self.make_weights(in_size, out_size)
        )
        self.biases.set_value(
            np.ones(out_size, dtype=theano.config.floatX)
        )

    def change_size(self, ins, outs):
        """
        Change the size of the weights/biases for this layer.
        """
        self.n_output, self.n_input = (outs, ins)
        self.weights.set_value(self.make_weights(ins, outs))
        self.biases.set_value(np.ones(outs, dtype=theano.config.floatX))

class Network(object):
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
        self.settings = {
            "max_training_epochs": 5000,  # train
            "stop_percentage": None,  #train
            "tolerance": 0.1, # train
            "report_rate": 500,# train
            "activation_function": T.nnet.sigmoid, # init
            "epsilon": 0.1, # train
            "momentum": 0.9, # train
        }
        self.defaults = copy.copy(self.settings)
        settings = self.settings
        for key in kwargs:
            if key == "activation_function":
                settings["activation_function"] = kwargs[key]
            elif key == "epsilon":
                settings["epsilon"] = kwargs[key]
            elif key == "momentum":
                settings["momentum"] = kwargs[key]
            else:
                raise Exception("unknown argument: '%s'" % key)
        self._epsilon = theano.shared(settings["epsilon"], name='self.epsilon')
        self._momentum = theano.shared(settings["momentum"], name='self.momentum')
        self._set_momentum(settings["momentum"])
        self._set_epsilon(settings["epsilon"])
        self.make_layers(sizes)
        # Combine parameters from all layers
        self.params = []
        for layer in self.layer:
            self.params += layer.params # [weights, biases]

        # Create Theano variables for the input
        self.th_inputs = T.vector('self.th_inputs', dtype=theano.config.floatX)
        # ... and the desired output
        self.th_targets = T.vector('self.th_targets', dtype=theano.config.floatX)

        error = self._tss_error(self.th_inputs, self.th_targets)
        # Dynamic Python methods:
        self._pytrain_one = function(
            [self.th_inputs, self.th_targets],
            error,
            updates=self.compute_delta_weights(error),
            allow_input_downcast=True)
        self._pypropagate = function([self.th_inputs],
                                     self._propagate(self.th_inputs),
                                     allow_input_downcast=True)
        self.tss_error = function([self.th_inputs, self.th_targets],
                                  self._tss_error(self.th_inputs, self.th_targets),
                                  allow_input_downcast=True)
        self.target_function = None
        self.epoch = 0
        self.history = {}

    def make_layers(self, sizes):
        """
        Create the layers for the network.
        """
        self.layer = []
        for n_input, n_output in zip(sizes[:-1], sizes[1:]):
            self.layer.append(Layer(n_input, n_output,
                                    self.settings["activation_function"]))

    def _get_report_rate(self):
        return self.settings["report_rate"]

    def _set_report_rate(self, value):
        self.settings["report_rate"] = value

    def _set_epsilon(self, epsilon):
        """
        Method to set epsilon (learning rate).
        """
        self.settings["epsilon"] = epsilon
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
        self.settings["momentum"] = momentum
        self._momentum.set_value(momentum)

    def _get_momentum(self):
        """
        Method to get mementum.
        """
        return self._momentum.get_value()

    def _get_max_training_epochs(self):
        return self.settings["max_training_epochs"]

    def _set_max_training_epochs(self, value):
        self.settings["max_training_epochs"] = value

    def _get_stop_percentage(self):
        return self.settings["stop_percentage"]

    def _set_stop_percentage(self, value):
        self.settings["stop_percentage"] = value

    def _get_tolerance(self):
        return self.settings["tolerance"]

    def _set_tolerance(self, value):
        self.settings["tolerance"] = value

    def _get_activation_function(self):
        return self.settings["activation_function"]

    def _set_activation_function(self, value):
        self.settings["activation_function"] = value
        for i in range(len(self.layer)):
            self.layer[i].activation_function = value

    epsilon = property(_get_epsilon, _set_epsilon)
    momentum = property(_get_momentum, _set_momentum)
    report_rate = property(_get_report_rate, _set_report_rate)
    max_training_epochs = property(_get_max_training_epochs, _set_max_training_epochs)
    stop_percentage = property(_get_stop_percentage, _set_stop_percentage)
    tolerance = property(_get_tolerance, _set_tolerance)
    activation_function = property(_get_activation_function, _set_activation_function)

    def train_one(self, inputs, targets):
        return self._pytrain_one(inputs, targets)

    def propagate(self, inputs):
        return self._pypropagate(inputs)

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
            updates.append((param, T.cast(param - self._epsilon * param_update,
                                          theano.config.floatX)))
            # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
            updates.append((param_update,
                            T.cast((self._momentum * param_update) +
                                   (1. - self._momentum) * T.grad(error, param),
                                   theano.config.floatX)))
        return updates

    def _propagate(self, inputs):
        '''
        Compute the networks's output given an input

        :parameters:
            - inputs : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                inputs passed through the network
        '''
        # Recursively compute output
        activations = inputs
        for layer in self.layer:
            activations = layer._propagate(activations)
        return activations

    def _tss_error(self, inputs, targets):
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
        return T.sum((self._propagate(inputs) - targets)**2,
                     dtype=theano.config.floatX)

    def train(self, **kwargs):
        """
        Method to train network.
        """
        self.config(kwargs)
        # Get initial error, before training:
        error = 0
        correct = 0
        total = 0
        print("-" * 50)
        print("Training for max trails:", self.settings["max_training_epochs"], "...")
        for i in range(len(self.inputs)):
            if self.target_function:
                target = self.target_function(self.inputs[i])
                inputs = self.inputs[i]
            else:
                # inputs is input and target
                target = self.inputs[i][1]
                inputs = self.inputs[i][0]
            output = self.propagate(inputs)
            error += self.tss_error(inputs, target)
            if all(map(lambda v: v <= self.settings["tolerance"],
                       np.abs(output - target, dtype=theano.config.floatX))):
                correct += 1
            total += 1
        print('Epoch:', self.epoch,
              'TSS error:', error,
              '%correct:', correct/total)
        self.history[self.epoch] = [error, correct/total]
        if correct/total < self.settings["stop_percentage"]:
            for e in range(self.settings["max_training_epochs"]):
                error = 0
                correct = 0
                total = 0
                random.shuffle(self.inputs)
                for i in range(len(self.inputs)):
                    if self.target_function:
                        target = self.target_function(self.inputs[i])
                        inputs = self.inputs[i]
                    else:
                        # inputs is input and target
                        target = self.inputs[i][1]
                        inputs = self.inputs[i][0]
                    error += self.train_one(inputs, target)
                    total += 1
                    output = self.propagate(inputs)
                    if all(map(lambda v: v <= self.settings["tolerance"],
                               np.abs(output - target, dtype=theano.config.floatX))):
                        correct += 1
                self.epoch += 1
                if self.epoch % self.settings["report_rate"] == 0:
                    self.history[self.epoch] = [error, correct/total]
                    print('Epoch:', self.epoch,
                          'TSS error:', error,
                          '%correct:', correct/total)
                if self.settings["stop_percentage"] is not None:
                    if correct/total >= self.settings["stop_percentage"]:
                        break
        self.history[self.epoch] = [error, correct/total]
        print("-" * 50)
        print('Epoch:', self.epoch,
              'TSS error:', error,
              '%correct:', correct/total)

    def test(self, stop=None, start=0):
        """
        Method to test network.
        """
        if stop is None:
            stop = len(self.inputs)
        error = 0
        total = 0
        correct = 0
        print("-" * 50)
        print("Test:")
        for inputs in self.inputs[start:stop]:
            if self.target_function:
                target = self.target_function(inputs)
            else:
                # inputs is input and target
                target = inputs[1]
                inputs = inputs[0]
            output = self.propagate(inputs)
            error += self.tss_error(inputs, target)
            if all(map(lambda v: v <= self.settings["tolerance"],
                       np.abs(output - target, dtype=theano.config.floatX))):
                correct += 1
            total += 1
            self.display_test_input(inputs)
            self.display_test_output(output)
        print("-" * 50)
        print('Epoch:', self.epoch,
              'TSS error:', error,
              '%correct:', correct/total)

    def reset(self):
        """
        Resets a network's learned values.
        """
        self.epoch = 0
        self.history = {}
        for i in range(len(self.layer)):
            self.layer[i].reset()

    def config(self, kwargs):
        self.settings.update(kwargs)
        self._set_momentum(self.settings["momentum"])
        self._set_epsilon(self.settings["epsilon"])

    def reinit(self):
        """
        Restore network to inital state.
        """
        self.reset()
        self.config(copy.copy(self.defaults))
        self.target_function = None
        self.inputs = None

    def get_history(self):
        """
        Get the history in order.
        """
        epochs = sorted(self.history.keys())
        return [[key] + self.history[key] for key in epochs]

    def set_inputs(self, inputs):
        """
        Set the inputs and optionally targets to train on.

        inputs may be list of inputs, or list of inputs/targets.

        If inputs is just inputs, then you need to set
        net.set_target_function.
        """
        self.inputs = inputs

    def set_target_function(self, func):
        """
        Set the target function, if one.
        """
        self.target_function = func

    def change_layer_size(self, layer, size):
        """
        Change the size of a layer. Should call
        reset at some point after this.
        """
        self.sizes[layer] = size

    def display_test_input(self, v):
        """
        Method to display input pattern.
        """
        print("Input:", v)

    def display_test_output(self, v):
        """
        Method to display output pattern.
        """
        print("Output:", v)
        print()

    def __repr__(self):
        retval = "Network:"
        retval += ("-" * 50) + "\n"
        for i in range(len(self.layer)):
            layer = self.layer[i]
            retval += "Layer %s:\n" % i
            retval += str(layer)
            retval += ("-" * 50) + "\n"
        return retval
