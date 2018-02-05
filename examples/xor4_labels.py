import conx as cx

net = cx.Network("XOR2")
net.add(cx.Layer("input1", 2))
net.add(cx.Layer("input2", 2))
net.add(cx.Layer("hidden1", 2, activation="sigmoid"))
net.add(cx.Layer("hidden2", 2, activation="sigmoid"))
net.add(cx.Layer("shared-hidden", 2, activation="sigmoid"))
net.add(cx.Layer("output1", 2, activation="sigmoid"))
net.add(cx.Layer("output2", 2, activation="sigmoid"))

net.connect("input1", "hidden1")
net.connect("input2", "hidden2")
net.connect("hidden1", "shared-hidden")
net.connect("hidden2", "shared-hidden")
net.connect("shared-hidden", "output1")
net.connect("shared-hidden", "output2")

net.compile(loss='mean_squared_error',
            optimizer=cx.SGD(lr=0.3, momentum=0.9))

ds = [
    ([[0, 0],[0, 0]], [[0, 0],[0, 0]], ["one", "one"]),
    ([[0, 0],[1, 1]], [[1, 1],[1, 1]], ["two", "two"]),
    ([[1, 1],[0, 0]], [[1, 1],[1, 1]], ["three", "three"]),
    ([[1, 1],[1, 1]], [[0, 0],[0, 0]], ["four", "four"])
]
net.dataset.load(ds)
net.train(2000, report_rate=10, accuracy=1)
net.test()
