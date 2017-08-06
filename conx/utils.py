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

import numbers
from keras.utils import to_categorical

#------------------------------------------------------------------------
# utility functions

def one_hot(vector, categories):
    """
    Given a vector of integers (i.e. labels), return a numpy array of one-hot vectors.
    """
    return to_categorical(vector, categories)[0].tolist()

def topological_sort(net, layers):
    """
    Given a conx network and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    ## Initilize all:
    for layer in net.layers:
        layer.visited = False
    stack = []
    # Only track and sort these:
    for layer in layers:
        if not layer.visited:
            visit(layer, stack)
    stack.reverse()
    return stack

def visit(layer, stack):
    """
    Utility function for topological_sort.
    """
    layer.visited = True
    for outgoing_layer in layer.outgoing_connections:
        if not outgoing_layer.visited:
            visit(outgoing_layer, stack)
    stack.append(layer)

def autoname(index, sizes):
    """
    Given an index and list of sizes, return a
    name for the layer.
    """
    if index == 0:
        n = "input"
    elif index == sizes - 1:
        n = "output"
    elif sizes == 3:
        n = "hidden"
    else:
        n = "hidden%d" % index
    return n

def valid_shape(x):
    """
    Is this a valid shape for Keras layers?
    """
    return isinstance(x, numbers.Integral) and x > 0 \
        or isinstance(x, (tuple, list)) and len(x) > 1 and all([isinstance(n, numbers.Integral) and n > 0 for n in x])

def valid_vshape(x):
    """
    Is this a valid shape (i.e., size) to display vectors using PIL?
    """
    # vshape must be a single int or a 2-dimensional tuple
    return valid_shape(x) and (isinstance(x, numbers.Integral) or len(x) == 2)

def rescale_numpy_array(a, old_range, new_range, new_dtype):
    """
    Given a vector, old min/max, a new min/max and a numpy type,
    create a new vector scaling the old values.
    """
    assert isinstance(old_range, (tuple, list)) and isinstance(new_range, (tuple, list))
    old_min, old_max = old_range
    if a.min() < old_min or a.max() > old_max:
        raise Exception('array values are outside range %s' % (old_range,))
    new_min, new_max = new_range
    old_delta = old_max - old_min
    new_delta = new_max - new_min
    if old_delta == 0:
        return ((a - old_min) + (new_min + new_max)/2).astype(new_dtype)
    else:
        return (new_min + (a - old_min)*new_delta/old_delta).astype(new_dtype)

'''

def evaluate(model, test_inputs, test_targets, threshold=0.50, indices=None, show=False):
    assert len(test_targets) == len(test_inputs), "number of inputs and targets must be the same"
    if type(indices) not in (list, tuple) or len(indices) == 0:
        indices = range(len(test_inputs))
    # outputs = [np.argmax(t) for t in model.predict(test_inputs[indices]).round()]
    # targets = list(test_labels[indices])
    wrong = 0
    for i in indices:
        target_vector = test_targets[i]
        target_class = np.argmax(target_vector)
        output_vector = model.predict(test_inputs[i:i+1])[0]
        output_class = np.argmax(output_vector)  # index of highest probability in output_vector
        probability = output_vector[output_class]
        if probability < threshold or output_class != target_class:
            if probability < threshold:
                output_class = '???'
            print('image #%d (%s) misclassified as %s' % (i, target_class, output_class))
            wrong += 1
            if show:
                plt.imshow(test_images[i], cmap='binary', interpolation='nearest')
                plt.draw()
                answer = raw_input('RETURN to continue, q to quit...')
                if answer in ('q', 'Q'):
                    return
    total = len(indices)
    correct = total - wrong
    correct_percent = 100.0*correct/total
    wrong_percent = 100.0*wrong/total
    print('%d test images: %d correct (%.1f%%), %d wrong (%.1f%%)' %
          (total, correct, correct_percent, wrong, wrong_percent))

'''
