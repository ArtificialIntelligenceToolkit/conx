from konx import Network, Layer, SGD

net = Network(
    Layer("input", shape=2),
    Layer("hidden", shape=2, activation="sigmoid"),
    Layer("output", shape=1, activation="sigmoid")
)

net.connect("input", "hidden")
net.connect("hidden", "output")

net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))

net.set_dataset([
    ([0,0], [0]),
    ([0,1], [1]),
    ([1,0], [1]),
    ([1,1], [0])
])

net.train(2000, report_rate=10, accuracy=1)
net.test(net.train_inputs)
