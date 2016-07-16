from conx import Network

inputs = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

def xor(inputs):
    a = inputs[0]
    b = inputs[1]
    return [[0.1, 0.9][int((a or b) and not(a and b))]]

net = Network(2, 2, 1)
net.set_inputs(inputs)
net.set_target_function(xor)
net.train(stop_percentage=1.0)
net.test()

net = Network(2, 2, 2, 1)
net.set_inputs(inputs)
net.set_target_function(xor)
net.train(stop_percentage=1.0, max_epoch=10000)
net.test()
