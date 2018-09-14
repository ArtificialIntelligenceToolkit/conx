import h5py
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
inputs = np.concatenate((x_train,x_test)) / 255
labels = np.concatenate((y_train,y_test)) # ints, 0 to 10
###########################################
# fix mis-labeled image(s) in Keras dataset
labels[10994] = 9
###########################################
targets = to_categorical(labels).astype("uint8")
string = h5py.special_dtype(vlen=str)
labels = np.array([str(label) for label in labels], dtype=string)

print("creating h5...")
with h5py.File("mnist.h5", "w") as h5:
    dset = h5.create_dataset('inputs', data=[inputs], compression='gzip', compression_opts=9)
    dset = h5.create_dataset('targets', data=[targets], compression='gzip', compression_opts=9)
    dset = h5.create_dataset('labels', data=[labels], compression='gzip', compression_opts=9)
print("done!")
