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
net.train()
net.test()

net = Network(2, 2, 2, 1)
net.set_inputs(inputs)
net.set_target_function(xor)
net.train(max_training_epochs=10000)
net.test()

inputs = [[[0, 0], [0, 0]],
          [[0, 1], [1, 1]],
          [[1, 0], [1, 1]],
          [[1, 1], [0, 0]]]

net = Network(2, 10, 2)
net.set_inputs(inputs)
net.train(max_training_epochs=10000)
net.test()

#---------------------------------------------------------------------
net = Network(
    Layer("input1", shape=2),
    Layer("hidden", shape=2, activation="sigmoid"),
    Layer("output1", shape=1, activation="sigmoid")
)

net.connect("input1", "hidden")
net.connect("hidden", "output1")

net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9),
            metrics=['accuracy'])

XOR_inputs = np.array([[0,0], [0,1], [1,0], [1,1]], 'float32')
XOR_targets = np.array([[0], [1], [1], [0]], 'float32')


