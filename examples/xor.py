from conx import Network, SGD, Dataset

# Method 1:
dataset = Dataset()
ds = [[[0, 0], [0]],
      [[0, 1], [1]],
      [[1, 0], [1]],
      [[1, 1], [0]]]
dataset.load(ds)

net = Network("XOR", 2, 2, 1, activation="sigmoid")
net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))

# NOTE:
#    net = Network(2, 3, 4, 1, activation="sigmoid")
# is the same as:
#    net = Network()
#    net.add(Layer("input", shape=2))
#    net.add(Layer("hidden1", shape=3, activation="sigmoid"))
#    net.add(Layer("hidden2", shape=4, activation="sigmoid"))
#    net.add(Layer("output", shape=1, activation="sigmoid"))
#    net.connect("input", "hidden1")
#    net.connect("hidden1", "hidden2")
#    net.connect("hidden2", "output")

net.set_dataset(dataset)
net.train(2000, report_rate=10, accuracy=1)
net.test()

# Method 2:
net.reset()
net.dataset = None
net.set_dataset(ds)
net.train(2000, report_rate=10, accuracy=1)
net.test()
