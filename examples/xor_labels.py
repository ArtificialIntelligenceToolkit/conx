import conx as cx

ds = [[[0, 0], [0], "one"],
      [[0, 1], [1], "two"],
      [[1, 0], [1], "three"],
      [[1, 1], [0], "four"]]

net = cx.Network("XOR", 2, 2, 1, activation="sigmoid")
net.compile(error='mean_squared_error',
            optimizer=cx.SGD(lr=0.3, momentum=0.9))

# NOTE:
#    net = cx.Network("XOR", 2, 3, 4, 1, activation="sigmoid")
# is the same as:
#    net = Network("XOR")
#    net.add(Layer("input", shape=2))
#    net.add(Layer("hidden1", shape=3, activation="sigmoid"))
#    net.add(Layer("hidden2", shape=4, activation="sigmoid"))
#    net.add(Layer("output", shape=1, activation="sigmoid"))
#    net.connect("input", "hidden1")
#    net.connect("hidden1", "hidden2")
#    net.connect("hidden2", "output")

net.dataset.load(ds)
net.train(2000, report_rate=10, accuracy=1)
net.test()
