from conx import Network, SGD

dataset = [[[0, 0], [0]],
          [[0, 1], [1]],
          [[1, 0], [1]],
          [[1, 1], [0]]]

net = Network(2, 2, 1)
net.set_dataset(dataset)
net.compile(loss='mean_squared_error',
            optimizer=SGD(lr=0.3, momentum=0.9))
net.train(2000, report_rate=10, accuracy=1)
net.test()
