from __future__ import print_function, division

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

import theano
import theano.tensor as T
from theano import function, pp
import io
import operator
import functools
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

    def save_deltas_and_reset_weights(self):
        # compute delta weights/biases
        # save deltas
        self.delta_weights = self.weights.get_value() - self.orig_weights
        self.delta_biases = self.biases.get_value() - self.orig_biases
        # reset the weights/biases
        self.weights.set_value(self.orig_weights)
        self.biases.set_value(self.orig_biases)

    def save_weights(self):
        self.orig_weights = self.weights.get_value()
        self.orig_biases = self.biases.get_value()

    def update_weights_from_deltas(self):
        self.weights.set_value(self.orig_weights + self.delta_weights)
        self.biases.set_value(self.orig_biases + self.delta_biases)

    def make_weights(self, ins, outs):
        """
        Makes a 2D matrix of random weights centered around 0.0.
        """
        min, max = -0.5, 0.5
        range = (max - min)
        return np.array(range * np.random.rand(outs, ins) - range/2.0,
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

    def save(self, fp):
        """
        Save the weights to a file.
        """
        np.save(fp, self.weights.get_value())
        np.save(fp, self.biases.get_value())

    def load(self, fp):
        """
        Load the weights from a file.
        """
        self.weights.set_value(np.load(fp))
        self.biases.set_value(np.load(fp))

class Network(object):
    def __init__(self, *sizes, **kwargs):
        ## args in old style because Python2 limitations
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
        # Construct theano variables:
        self._epsilon = theano.shared(0.0, name='self.epsilon')
        self._momentum = theano.shared(0.0, name='self.momentum')
        # Create Theano variables for the input
        self.th_inputs = T.vector('self.th_inputs', dtype=theano.config.floatX)
        # ... and the desired output
        self.th_targets = T.vector('self.th_targets', dtype=theano.config.floatX)
        # Set defaults:
        self.defaults = {"epsilon": 0.1,
                         "momentum": 0.9,
                         "activation_function": T.nnet.sigmoid,
                         "max_training_epochs": 5000,
                         "stop_percentage": 1.0,
                         "tolerance": 0.1,
                         "report_rate": 500,
                         "batch": False,
                         "shuffle": True}
        # First, set defaults from self.defaults:
        for key in self.defaults:
            if key == "activation_function": # don't use property yet, layers not made
                self._activation_function = self.defaults[key]
            else:
                setattr(self, key, self.defaults[key])
        # Then, set args:
        for key in kwargs:
            if key in self.defaults:
                if key == "activation_function": # don't use property yet, layers not made
                    self._activation_function = kwargs[key]
                else:
                    setattr(self, key, kwargs[key])
                self.defaults[key] = kwargs[key]
            else:
                raise Exception("invalid keyword argument '%s'" % key)
        # Make the layers:
        self.make_layers(sizes)
        # Combine parameters from all layers
        self.params = []
        for layer in self.layer:
            self.params += layer.params # [weights, biases]
        # Theano expressions:
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
                                    self.activation_function))

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

    def _set_batch(self, batch):
        """
        Method to set batch.
        """
        self._batch = batch
        if self._batch:
            self.save_weights()

    def _get_batch(self):
        """
        Method to get batch.
        """
        return self._batch

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

    def _get_activation_function(self):
        return self._activation_function

    def _set_activation_function(self, value):
        self._activation_function = value
        for i in range(len(self.layer)):
            self.layer[i].activation_function = value

    epsilon = property(_get_epsilon, _set_epsilon)
    momentum = property(_get_momentum, _set_momentum)
    activation_function = property(_get_activation_function, _set_activation_function)
    batch = property(_get_batch, _set_batch)

    def train_one(self, inputs, targets):
        """
        Given an input vector and a target vector, update the weights (if
        not batch) and return error.
        """
        retval = self._pytrain_one(inputs, targets)
        if self.batch:
            self.save_deltas_and_reset_weights()
        return retval

    def save_deltas_and_reset_weights(self):
        """
        """
        for layer in self.layer:
            layer.save_deltas_and_reset_weights()

    def save_weights(self):
        """
        """
        for layer in self.layer:
            layer.save_weights()

    def save(self, filename=None):
        """
        Save network weights and biases to a file.
        """
        if filename is None:
            fp = io.BytesIO()
            for layer in self.layer:
                layer.save(fp)
            return fp
        else:
            with open(filename, "wb") as fp:
                for layer in self.layer:
                    layer.save(fp)

    def load(self, filename):
        """
        Load network weights and biases from a file.
        """
        if isinstance(filename, str):
            with open(filename, "rb") as fp:
                for layer in self.layer:
                    layer.load(fp)
        else:
            fp.seek(0)
            for layer in net.layer:
                layer.load(fp)

    def update_weights_from_deltas(self):
        """
        """
        for layer in self.layer:
            layer.update_weights_from_deltas()

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
                                   T.grad(error, param),
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

    def initialize_inputs(self):
        if self.shuffle:
            self.shuffle_inputs()

    def inputs_size(self):
        return len(self.inputs)

    def get_inputs(self, i):
        return self.inputs[i]

    def cross_validate(self):
        """
        Run through all, computing error, correct, total
        """
        error = 0
        correct = 0
        total = 0
        for i in range(self.inputs_size()):
            self.current_input_index = i
            inputs = self.get_inputs(i)
            if self.target_function:
                target = self.target_function(inputs)
            else:
                # inputs is input and target
                target = inputs[1]
                inputs = inputs[0]
            output = self.propagate(inputs)
            error += self.tss_error(inputs, target)
            if all(map(lambda v: v <= self.tolerance,
                       np.abs(output - target, dtype=theano.config.floatX))):
                correct += 1
            total += 1
        self.last_cv_error, self.last_cv_correct, self.last_cv_total = error, correct, total
        return error, correct, total
    
    def train(self, **kwargs):
        """
        Method to train network.
        """
        # Get initial error, before training:
        for key in kwargs:
            if key in ["epsilon", "momentum", "activation_function",
                       "max_training_epochs", "stop_percentage",
                       "tolerance", "report_rate", "batch", "shuffle"]:
                setattr(self, key, kwargs[key])
            else:
                raise AttributeError("Invalid option: '%s'" % key)
        print("-" * 50)
        print("Training for max trails:", self.max_training_epochs, "...")
        self.initialize_inputs()
        error, correct, total = self.cross_validate()
        print('Epoch:', self.epoch,
              'TSS error:', error,
              '%correct:', correct/total)
        self.history[self.epoch] = [error, correct/total]
        if correct/total < self.stop_percentage:
            for e in range(self.max_training_epochs):
                if self.batch:
                    self.save_weights()
                self.initialize_inputs()
                for i in range(self.inputs_size()):
                    self.current_input_index = i
                    inputs = self.get_inputs(i)
                    if self.target_function:
                        target = self.target_function(inputs)
                    else:
                        # inputs is input and target
                        target = inputs[1]
                        inputs = inputs[0]
                    self.train_one(inputs, target)
                    output = self.propagate(inputs)
                if self.batch:
                    self.update_weights_from_deltas()
                self.epoch += 1
                error, correct, total = self.cross_validate()
                if self.epoch % self.report_rate == 0:
                    self.history[self.epoch] = [error, correct/total]
                    print('Epoch:', self.epoch,
                          'TSS error:', error,
                          '%correct:', correct/total)
                if self.stop_percentage is not None:
                    if correct/total >= self.stop_percentage:
                        break
        self.history[self.epoch] = [error, correct/total]
        print("-" * 50)
        print('Epoch:', self.epoch,
              'TSS error:', error,
              '%correct:', correct/total)

    def shuffle_inputs(self):
        """
        Shuffle the input/target patterns.
        """
        random.shuffle(self.inputs)

    def test(self, stop=None, start=0):
        """
        Method to test network.
        """
        if stop is None:
            stop = self.inputs_size()
        error = 0
        total = 0
        correct = 0
        print("-" * 50)
        print("Test:")
        self.initialize_inputs()
        for i in range(start, stop):
            self.current_input_index = i
            inputs = self.get_inputs(i)
            if self.target_function:
                target = self.target_function(inputs)
            else:
                # inputs is input and target
                target = inputs[1]
                inputs = inputs[0]
            output = self.propagate(inputs)
            error += self.tss_error(inputs, target)
            answer = "Incorrect"
            if all(map(lambda v: v <= self.tolerance,
                       np.abs(output - target, dtype=theano.config.floatX))):
                correct += 1
                answer = "Correct"
            total += 1
            print("*" * 30)
            self.display_test_input(inputs)
            self.display_test_output(output)
            self.display_test_output(target, result=answer, label='Target: ')
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

    def reinit(self):
        """
        Restore network to inital state.
        """
        self.reset()
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

    def pp(self, label, vector, result=''):
        """
        Pretty printer for a label, vector and optional result
        """
        print(label + (",".join(["% .1f" % v for v in vector])) + " " + result)
                
    def display_test_input(self, v):
        """
        Method to display input pattern.
        """
        self.pp("Input : ", v)

    def display_test_output(self, v, result='', label='Output: '):
        """
        Method to display output pattern.
        """
        self.pp(label, v, result)

    def __repr__(self):
        retval = "Network:"
        retval += ("-" * 50) + "\n"
        for i in range(len(self.layer)):
            layer = self.layer[i]
            retval += "Layer %s:\n" % i
            retval += str(layer)
            retval += ("-" * 50) + "\n"
        return retval

    def get_device(self):
        """
        Returns 'cpu' or 'gpu' indicating which device
        the network will use.
        """
        if np.any([isinstance(x.op, T.Elemwise) for x
                   in self._pypropagate.maker.fgraph.toposort()]):
            return "cpu"
        else:
            return "gpu"

    def to_array(self):
        """
        Turn weights and biases into a 1D vector of values.
        """
        fp = self.save()
        fp.seek(0)
        array = []
        while True:
            try:
                layer = np.load(fp)
            except:
                break
            layer.shape = tuple([functools.reduce(operator.mul, layer.shape)])
            array.extend(layer)
        return array

    def from_array(self, array):
        """
        Given an array (perhaps formed from Network.to_array) load
        all of the weights and biases.
        """
        current = 0
        for layer in self.layer:
            w = layer.n_input * layer.n_output
            weights = np.array(array[current: current + w])
            weights.shape = (layer.n_output, layer.n_input)
            assert layer.weights.get_value().shape == weights.shape
            layer.weights.set_value(weights)
            current += w
            b = layer.n_output
            biases = np.array(array[current: current + b])
            assert layer.biases.get_value().shape == biases.shape
            layer.biases.set_value(biases)
            current += b
