from conx import *

net = Network("Image")
net.add(ImageLayer("input", (28, 28), 1))
net.add(Conv2DLayer("conv1", 10, (5,5), activation="relu"))
net.add(Conv2DLayer("conv2", 10, (5,5), activation="relu"))
net.add(MaxPool2DLayer("pool1", pool_size=(2, 2)))
net.add(FlattenLayer("flatten"))
net.add(Layer("hidden1", 20, activation="relu"))
net.add(Layer("output", 10, activation="softmax"))

net.connect()

net.compile(error='categorical_crossentropy',
            optimizer="adam")

net.dataset.get("mnist")

# NOTE:
#    net = Network("XOR", 2, 3, 4, 1, activation="sigmoid")
# is the same as:
#    net = Network("XOR")
#    net.add(Layer("input", shape=2))
#    net.add(Layer("hidden1", shape=3, activation="sigmoid"))
#    net.add(Layer("hidden2", shape=4, activation="sigmoid"))
#    net.add(Layer("output", shape=1, activation="sigmoid"))
#    net.connect("input", "hidden1")
#    net.connect("hidden1", "hidden2")
#    net.connect("hidden2", "output")

#net.dataset.load(ds)
#net.train(2000, report_rate=10, accuracy=1)
#net.test()

assert net.propagate(net.dataset.inputs[0]) == net.propagate_to("output", net.dataset.inputs[0]), "before"

net.train(1)

assert net.propagate(net.dataset.inputs[0]) == net.propagate_to("output", net.dataset.inputs[0]), "after"
