from conx import Network, LSTMLayer, Layer

tsteps = 1
batch_size = 25

net = Network("LSTM")
net.add(Layer("input", shape=(tsteps, 1), batch_shape=(tsteps, 1, 1)))
net.add(LSTMLayer("lstm1",
                  shape=50,
                  return_sequences=True,
                  stateful=True))
net.add(LSTMLayer("lstm2",
                  shape=50,
                  return_sequences=False,
                  stateful=True))
net.add(Layer("output", shape=1))
net.connect()
net.compile(loss='mse', optimizer='rmsprop')

