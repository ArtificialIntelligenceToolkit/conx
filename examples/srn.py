from conx import SRN
import theano.tensor as T

inputs = [[0],
          [0],
          [0],
          [1],
          [1],
          [0],
          [1],
          [1]]

last = 0

def xor(inputs):
    global last
    a = inputs[0]
    b = last
    last = a
    # if a == -1 and b == -1:
    #     return [-1]
    # elif a == -1 and b == 1:
    #     return [1]
    # elif a == 1 and b == -1:
    #     return [1]
    # elif a == 1 and b == 1:
    #     return [-1]
    if a == 0 and b == 0:
        return [0]
    elif a == 0 and b == 1:
        return [1]
    elif a == 1 and b == 0:
        return [1]
    elif a == 1 and b == 1:
        return [0]
    else:
        raise Exception("invalid XOR inputs: %s, %s" % (a, b))

net = SRN(1, 2, 2, 1, activation_function=lambda inputs: T.nnet.sigmoid(inputs - 0.5))
net.set_inputs(inputs)
net.set_target_function(xor)
net.train(max_training_epochs=5000,
          tolerance=0.3,
          epsilon=0.1,
          shuffle=False)
net.test()

net = SRN(1, 10, 1, activation_function=lambda inputs: T.nnet.sigmoid(inputs - 0.5))
net.set_inputs(inputs)
net.set_target_function(xor)
net.train(max_training_epochs=5000,
          tolerance=0.3,
          epsilon=0.1,
          shuffle=False)

net.test()
