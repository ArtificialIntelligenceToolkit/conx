from konx import Network, Layer, SGD

net = Network(
    Layer("input1", shape=1),
    Layer("input2", shape=1),
    Layer("shared-hidden", shape=2, activation="sigmoid"),
    Layer("hidden1", shape=2, activation="sigmoid"),
    Layer("hidden2", shape=2, activation="sigmoid"),
    Layer("output1", shape=1, activation="sigmoid"),
    Layer("output2", shape=1, activation="sigmoid"),
)

net.connect("input1", "hidden1")
net.connect("input2", "hidden2")
net.connect("hidden1", "shared-hidden")
net.connect("hidden2", "shared-hidden")
net.connect("shared-hidden", "output1")
net.connect("shared-hidden", "output2")

net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))

# net.set_dataset([
#     ([0,0], [0]),
#     ([0,1], [1]),
#     ([1,0], [1]),
#     ([1,1], [0])
# ])

# net.train(2000, report_rate=10, accuracy=1)
# net.test(net.train_inputs)
