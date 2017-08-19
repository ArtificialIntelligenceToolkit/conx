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
    net.compile(loss="binary_crossentropy", optimizer="adam")
    net.summary()
    net.model.summary()
    dataset = Dataset()
    dataset.load([[[0, 0], [0]],
                  [[0, 1], [1]],
                  [[1, 0], [1]],
                  [[1, 1], [0]]])
    net.set_dataset(dataset)
    net.train(epochs=2000, accuracy=1, report_rate=25)
    net.test()
    net.save("/tmp/XOR.conx")
    net.load("/tmp/XOR.conx")
    svg = net.build_svg()
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
    net.compile(loss='mean_squared_error',
                optimizer=SGD(lr=0.3, momentum=0.9))

    dataset = Dataset()
    dataset.load([
        ([[0],[0]], [[0],[0]]),
        ([[0],[1]], [[1],[1]]),
        ([[1],[0]], [[1],[1]]),
        ([[1],[1]], [[0],[0]])
    ])
    net.set_dataset(dataset)
    net.train(2000, report_rate=10, accuracy=1)
    net.test()
    net.propagate_to("shared-hidden", [[1], [1]])
    net.propagate_to("output1", [[1], [1]])
    net.propagate_to("output2", [[1], [1]])
    net.propagate_to("hidden1", [[1], [1]])
    net.propagate_to("hidden2", [[1], [1]])
    net.propagate_to("output1", [[1], [1]])
    net.propagate_to("output2", [[1], [1]])
    net.save("/tmp/XOR2.conx")
    net.load("/tmp/XOR2.conx")
    net.test()
    svg = net.build_svg()
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
    net.compile(optimizer="adam", loss="binary_crossentropy")
    dataset = Dataset.get("mnist")
    net.set_dataset(dataset)
    assert net is not None

def test_dataset2():
    """
    Load data before adding network.
    """
    net = Network("MNIST")
    dataset = Dataset.get("mnist")
    net.add(Layer("input", shape=784, vshape=(28, 28), colormap="hot", minmax=(0,1)))
    net.add(Layer("hidden1", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("hidden2", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
    net.add(Layer("output", shape=10, activation='softmax'))
    net.connect('input', 'hidden1')
    net.connect('hidden1', 'hidden2')
    net.connect('hidden2', 'output')
    net.compile(optimizer="adam", loss="binary_crossentropy")
    dataset.split(100)
    dataset.slice(100)
    assert net is not None


## FIXME: doesn't work
# def test_images():
#     net = Network("MNIST")
#     net.load_mnist()
#     net.add(Layer("input", shape=784, vshape=(28, 28), colormap="hot", minmax=(0,1)))
#     net.add(Layer("hidden1", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
#     net.add(Layer("hidden2", shape=512, vshape=(16,32), activation='relu', dropout=0.2))
#     net.add(Layer("output", shape=10, activation='softmax'))
#     net.connect('input', 'hidden1')
#     net.connect('hidden1', 'hidden2')
#     net.connect('hidden2', 'output')
#     net.compile(optimizer="adam", loss="binary_crossentropy")
#     svg = net.build_svg() ## FAIL!
#     assert svg is not None

def test_cifar10():
    """
    Test the cifar10 API and training.
    """
    from conx import Network, Layer, Conv2DLayer, MaxPool2DLayer, FlattenLayer, Dataset

    batch_size = 32
    num_classes = 10
    epochs = 200
    data_augmentation = True
    num_predictions = 20

    ds = Dataset.get("cifar10")

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

    net.compile(loss='categorical_crossentropy',
                optimizer=opt)

    # Let's train the model using RMSprop
    net.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    net.set_dataset(ds)
    net.dashboard()
    net.dataset.slice(10)
    net.dataset.shuffle()
    net.dataset.split(.5)
    net.train()
    net.propagate(ds.inputs[0])
