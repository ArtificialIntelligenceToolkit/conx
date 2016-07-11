from conx import SRN

inputs = [[0],
          [1]]

last = 0

def xor(inputs):
    global last
    a = inputs[0]
    b = last
    last = a
    return [int((a or b) and not(a and b))]

net = SRN(1, 10, 1)
net.set_inputs(inputs)
net.set_target_function(xor)
net.train()
net.test()
