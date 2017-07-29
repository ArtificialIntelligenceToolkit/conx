from konx import *

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from scipy import misc
import glob, random

#---------------------------------------------------------------------------

def save_files(images, labels, dir_name, start=0):
    assert type(dir_name) is str and len(images) == len(labels)
    for i in range(len(images)):
        category = labels[i]
        filename = '%s/%d_%d.png' % (dir_name, category, start+i)
        misc.imsave(filename, images[i])
        if i > 0 and i % 5000 == 0:
            print('%d images saved' % i)
    print('all done')

# returns images and labels in dir_name folder, split and randomly shuffled
def myload_data(dir_name, limit=None, split=None):
    filenames = glob.glob('%s/*.png' % dir_name)
    num_files = len(filenames)
    if num_files == 0:
        print('ERROR: no files found')
        return None
    if limit is None:
        limit = num_files
    elif type(limit) is not int or not 0 < limit <= num_files:
        print('ERROR: invalid limit')
        return None
    if split is None or type(split) is int and 0 < split < limit:
        pass
    elif type(split) is float and 0 < split < 1:
        split = int(split*limit)
    else:
        print('ERROR: invalid split')
        return None
    random.shuffle(filenames)
    img = misc.imread(filenames[0])
    images = np.empty((limit,) + img.shape).astype('uint8')
    labels = np.empty((limit,)).astype('uint8')
    print('loading images...')
    for i, filename in enumerate(filenames):
        if i >= limit: break
        images[i] = misc.imread(filename)
        labels[i] = int(filename.split('/')[1].split('_')[0])
        if i > 0 and i % 1000 == 0:
            print('%d images loaded' % i)
    if split is None:
        print('images:', images.shape)
        print('image bytes:', images.nbytes)
        print('labels:', labels.shape)
        print('label bytes:', labels.nbytes)
        return images, labels
    else:
        train_images = images[:split]
        train_labels = labels[:split]
        test_images = images[split:]
        test_labels = labels[split:]
        print('train_images: %s, test_images: %s' % (train_images.shape, test_images.shape))
        print('image bytes:', train_images.nbytes + test_images.nbytes)
        print('train_labels %s, test_labels: %s' % (train_labels.shape, test_labels.shape))
        print('label bytes:', train_labels.nbytes + test_labels.nbytes)
        return (train_images, train_labels), (test_images, test_labels)

def info():
    print('images: %s %s' % (images.shape, images.dtype))
    print('image bytes: %d' % images.nbytes)
    print('labels: %s %s' % (labels.shape, labels.dtype))
    print('label bytes: %d' % labels.nbytes)

def info2():
    print('train_images: %s %s' % (train_images.shape, train_images.dtype))
    print('test_images: %s %s' % (test_images.shape, test_images.dtype))
    print('total image bytes: %d' % (train_images.nbytes + test_images.nbytes))
    print('train_labels: %s %s' % (train_labels.shape, train_labels.dtype))
    print('test_labels: %s %s' % (test_labels.shape, test_labels.dtype))
    print('total label bytes: %d' % (train_labels.nbytes + test_labels.nbytes))

def reshuffle(images, labels):
    num_images = images.shape[0]
    num_labels = labels.shape[0]
    assert num_images == num_labels
    indices = np.random.permutation(num_images)
    return images[indices], labels[indices]

def view(images, start=0, cmap=None, interpolation=None):
    plt.axis('off')
    k = start
    while True:
        print('image %d' % k)
        plt.imshow(images[k], cmap=cmap, interpolation=interpolation)
        plt.draw()
        k += 1
        answer = raw_input('RETURN to continue, q to quit...')
        if answer == 'q': break

# to save:
#np.savez_compressed('mydataset.npz', images=images, labels=labels)

plt.ion()
#---------------------------------------------------------------------------

net = Network(
    Layer("input", shape=784),
    Layer("hidden1", shape=512, activation='relu', dropout=0.2),
    Layer("hidden2", shape=512, activation='relu', dropout=0.2),
    Layer("output", shape=10, activation='softmax')
    )

net.connect('input', 'hidden1')
net.connect('hidden1', 'hidden2')
net.connect('hidden2', 'output')

net.compile(loss='mean_squared_error',
            optimizer='sgd')

# net.load_keras_dataset('mnist')
# net.rescale_inputs((0,255), (0,1), 'float32')
# net.shuffle_dataset()
# net.reshape_inputs(784)
# net.split_dataset(100)
# net.set_targets(10)
# net.show_dataset()

#-------------------------------------------------------------------------

