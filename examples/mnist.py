from conx import Network
import pickle
import gzip
import numpy

with gzip.open('mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()
    train_set, validation_set, test_set = data

net = Network(784, 100, 1)

inputs = [train_set[0][i] for i in range(len(train_set[0]))]
targets = [[train_set[1][i]/9.0] for i in range(len(train_set[0]))]

inputs = inputs[:100]
targets = targets[:100]

def display_digit(vector):
    for r in range(28):
        for c in range(28):
            v = int(vector[r * 28 + c] * 10)
            ch = " .23456789"[v]
            print(ch, end="")
        print()
        
net.display_input = display_digit
net.set_inputs(list(zip(inputs, targets)))

net.test(2)
net.train(stop_percentage=1.0, report_rate=1, tolerance=0.05)
net.test(2)

for i in range(100):
    output = net.propagate(inputs[i])
    target = int(targets[i][0] * 9)
    print("target:", target, 
          "output:", output, 
          "correct?", int(output * 10) == target)
