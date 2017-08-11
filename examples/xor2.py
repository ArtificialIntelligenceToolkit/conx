from conx import Network, Layer, SGD, Dataset

net = Network("XOR2")
net.add(Layer("input1", shape=1))
net.add(Layer("input2", shape=1))
net.add(Layer("hidden1", shape=2, activation="sigmoid"))
net.add(Layer("hidden2", shape=2, activation="sigmoid"))
net.add(Layer("shared-hidden", shape=2, activation="sigmoid"))
net.add(Layer("output1", shape=1, activation="sigmoid"))
net.add(Layer("output2", shape=1, activation="sigmoid"))

net.connect("input1", "hidden1")
net.connect("input2", "hidden2")
net.connect("hidden1", "shared-hidden")
net.connect("hidden2", "shared-hidden")
net.connect("shared-hidden", "output1")
net.connect("shared-hidden", "output2")

net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))

# Method 1:
dataset = Dataset([1, 1], [1, 1]) # input layer sizes, target layer sizes
ds = [
    ([[0],[0]], [[0],[0]]),
    ([[0],[1]], [[1],[1]]),
    ([[1],[0]], [[1],[1]]),
    ([[1],[1]], [[0],[0]])
]
dataset.load(ds)
net.set_dataset(dataset)
net.train(2000, report_rate=10, accuracy=1)
net.test()

# Method 2
net.reset()
net.dataset = None
net.set_dataset(ds)
net.train(2000, report_rate=10, accuracy=1)
net.test()
