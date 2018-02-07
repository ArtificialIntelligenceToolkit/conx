from conx import *

def test_network_constructor():
    """
    Network constructor.
    """
    net = Network("Constructor", 2, 5, 2)
    assert net is not None

def test_xor1():
    """
    Standard XOR.
    """
    net = Network("XOR")
    net.add(Layer("input", 2))
    net.add(Layer("hidden", 5))
    net.add(Layer("output", 1))
    net.connect("input", "hidden")
    net.connect("hidden", "output")
    net.compile(error="binary_crossentropy", optimizer="adam")
    net.summary()
    net.model.summary()
    net.dataset.load([[[0, 0], [0]],
                      [[0, 1], [1]],
                      [[1, 0], [1]],
                      [[1, 1], [0]]])
    net.train(epochs=2000, accuracy=1, report_rate=25, plot=False)
    net.test()
    net.save_weights("/tmp")
    net.load_weights("/tmp")
    svg = net.to_svg()
    assert net is not None

def test_xor2():
    """
    Two inputs, two outputs.
    """
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
    net.compile(error='mean_squared_error',
                optimizer=SGD(lr=0.3, momentum=0.9))

    net.dataset.load([
        ([[0],[0]], [[0],[0]]),
        ([[0],[1]], [[1],[1]]),
        ([[1],[0]], [[1],[1]]),
        ([[1],[1]], [[0],[0]])
    ])
    net.train(2000, report_rate=10, accuracy=1, plot=False)
    net.test()
    net.propagate_to("shared-hidden", [[1], [1]])
    net.propagate_to("output1", [[1], [1]])
    net.propagate_to("output2", [[1], [1]])
    net.propagate_to("hidden1", [[1], [1]])
    net.propagate_to("hidden2", [[1], [1]])
    net.propagate_to("output1", [[1], [1]])
    net.propagate_to("output2", [[1], [1]])
    net.save_weights("/tmp")
    net.load_weights("/tmp")
    net.test()
    svg = net.to_svg()
    assert net is not None

def test_dataset():
    """
    Load MNIST dataset after network creation.
    """
    net = Network("MNIST")
    net.add(Layer("input", shape=784, vshape=(28, 28), colormap="hot", minmax=(0,1)))
    net.add(Layer("hidden1", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("hidden2", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("output", shape=10, activation='softmax'))
    net.connect('input', 'hidden1')
    net.connect('hidden1', 'hidden2')
    net.connect('hidden2', 'output')
    net.compile(optimizer="adam", error="binary_crossentropy")
    net.dataset.get("mnist")
    assert net is not None

def test_dataset2():
    """
    Load data before adding network.
    """
    net = Network("MNIST")
    net.add(Layer("input", shape=784, vshape=(28, 28), colormap="hot", minmax=(0,1)))
    net.add(Layer("hidden1", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("hidden2", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("output", shape=10, activation='softmax'))
    net.connect('input', 'hidden1')
    net.connect('hidden1', 'hidden2')
    net.connect('hidden2', 'output')
    net.compile(optimizer="adam", error="binary_crossentropy")
    net.dataset.get("mnist")
    net.dataset.split(100)
    net.dataset.slice(100)
    assert net is not None

def test_images():
    net = Network("MNIST")
    net.dataset.get("mnist")
    assert net.dataset.inputs.shape == [(28,28,1)]
    net.add(Layer("input", shape=(28, 28, 1), colormap="hot", minmax=(0,1)))
    net.add(FlattenLayer("flatten"))
    net.add(Layer("hidden1", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("hidden2", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("output", shape=10, activation='softmax'))
    net.connect('input', 'flatten')
    net.connect('flatten', 'hidden1')
    net.connect('hidden1', 'hidden2')
    net.connect('hidden2', 'output')
    net.compile(optimizer="adam", error="binary_crossentropy")
    svg = net.to_svg()
    assert svg is not None

def test_cifar10():
    """
    Test the cifar10 API and training.
    """
    from conx import Network, Layer, Conv2DLayer, MaxPool2DLayer, FlattenLayer

    batch_size = 32
    num_classes = 10
    epochs = 200
    data_augmentation = True
    num_predictions = 20

    net = Network("CIRAR10")
    net.add(Layer("input", (32, 32, 3)))
    net.add(Conv2DLayer("conv1", 32, (3, 3), padding='same', activation='relu'))
    net.add(Conv2DLayer("conv2", 32, (3, 3), activation='relu'))
    net.add(MaxPool2DLayer("pool1", pool_size=(2, 2), dropout=0.25))
    net.add(Conv2DLayer("conv3", 64, (3, 3), padding='same', activation='relu'))
    net.add(Conv2DLayer("conv4", 64, (3, 3), activation='relu'))
    net.add(MaxPool2DLayer("pool2", pool_size=(2, 2), dropout=0.25))
    net.add(FlattenLayer("flatten"))
    net.add(Layer("hidden1", 512, activation='relu', vshape=(16, 32), dropout=0.5))
    net.add(Layer("output", num_classes, activation='softmax'))
    net.connect()

    # initiate RMSprop optimizer
    opt = RMSprop(lr=0.0001, decay=1e-6)
    net.compile(error='categorical_crossentropy',
                optimizer=opt)
    net.dataset.get("cifar10")
    widget = net.dashboard()
    widget.goto("begin")
    widget.goto("next")
    widget.goto("end")
    widget.goto("prev")
    widget.prop_one()
    net.dataset.slice(10)
    net.dataset.shuffle()
    net.dataset.split(.5)
    net.train(plot=False)
    net.propagate(net.dataset.inputs[0])
