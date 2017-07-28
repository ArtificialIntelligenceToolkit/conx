from __future__ import print_function, division, with_statement

from konx import Network, Layer
from keras.utils import to_categorical
import pickle
import gzip
import numpy
import os

directory, filname = os.path.split(__file__)

with gzip.open(os.path.join(directory, 'mnist.pkl.gz'), 'rb') as f:
    try: # Python3
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    except: # Python2
        data = pickle.load(f)
    train_set, validation_set, test_set = data

net = Network(
    Layer("input", shape=784),
    Layer("hidden1", shape=512, activation='relu', dropout=0.2),
    Layer("hidden2", shape=512, activation='relu', dropout=0.2),
    Layer("output", shape=10, activation='softmax')
    )

net.connect("input", "hidden1")
net.connect("hidden1", "hidden2")
net.connect("hidden2", "output")

net.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

inputs = [train_set[0][i] for i in range(len(train_set[0]))]
targets = [train_set[1][i] for i in range(len(train_set[0]))]

inputs = inputs[:100]
targets = to_categorical(targets[:100], 10).astype('uint8')

def display_digit(vector):
    for r in range(28):
        for c in range(28):
            v = int(vector[r * 28 + c] * 10)
            ch = " .23456789"[v]
            print(ch, end="")
        print()
        
# net.display_test_input = display_digit
# net.set_inputs(list(zip(inputs, targets)))

# net.test(2)
# net.train(report_rate=1, tolerance=0.05)
# net.test(2)

def showoutput():
    for i in range(100):
        output = net.propagate(inputs[i])
        target = targets[i]
        display_digit(inputs[i])
        output = '[' + ' '.join(['%.2f' % o for o in output]) + ']'
        print("target:", target, "\noutput:", output)
        
