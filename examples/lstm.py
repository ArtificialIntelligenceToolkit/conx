from conx import Network, LSTMLayer, Layer
import numpy as np
import matplotlib.pyplot as plt

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 25
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 1

def gen_cosine_amp(amp=100, period=1000, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing
    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos

print('Generating Data...')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape:', expected_output.shape)

print('Creating Model...')
net = Network("LSTM")
net.add(Layer("input", None, batch_shape=(batch_size, tsteps, 1))) #, batch_shape=(batch_size, tsteps, 1)))
net.add(LSTMLayer("lstm1",
                  shape=50,
                  #batch_input_shape=batch_size,
                  return_sequences=True,
                  stateful=True))
net.add(LSTMLayer("lstm2",
                  shape=50,
                  return_sequences=False,
                  stateful=True))
net.add(Layer("output", shape=1))
net.connect()
net.compile(loss='mse', optimizer='rmsprop')

net.set_dataset_direct(cos, expected_output)

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)

    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in cos.
    # Each of these series are offset by one step and can be
    # extracted with cos[i::batch_size].

    net.train(batch_size=batch_size,
              epochs=1,
              shuffle=False)
    net.model.reset_states()

print('Predicting')
predicted_output = net.model.predict(cos, batch_size=batch_size)

print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()
