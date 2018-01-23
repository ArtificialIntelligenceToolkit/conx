import numpy as np
from keras.utils import to_categorical

def cifar100(dataset):
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    inputs = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))
    targets = to_categorical(labels, 100)
    labels = np.array([str(label[0]) for label in labels], dtype=str)
    inputs = inputs.astype('float32')
    inputs /= 255
    dataset.name = "CIFAR-100"
    dataset.description = """
Original source: https://www.cs.toronto.edu/~kriz/cifar.html

This dataset is just like the CIFAR-10, except it has 100 classes
containing 600 images each. The 100 classes in the CIFAR-100 are grouped
into 20 superclasses. Each image comes with a "fine" label (the class
to which it belongs) and a "coarse" label (the superclass to which it
belongs).  Here is the list of classes in the CIFAR-100:

Superclass                     | Classes
-------------------------------|-----------------------------------------------------
aquatic mammals	               | beaver, dolphin, otter, seal, whale
fish                           | aquarium fish, flatfish, ray, shark, trout
flowers	                       | orchids, poppies, roses, sunflowers, tulips
food containers                | bottles, bowls, cans, cups, plates
fruit and vegetables           | apples, mushrooms, oranges, pears, sweet peppers
household electrical devices   | clock, computer keyboard, lamp, telephone, television
household furniture            | bed, chair, couch, table, wardrobe
insects	                       | bee, beetle, butterfly, caterpillar, cockroach
large carnivores               | bear, leopard, lion, tiger, wolf
large man-made outdoor things  | bridge, castle, house, road, skyscraper
large natural outdoor scenes   | cloud, forest, mountain, plain, sea
large omnivores and herbivores | camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals           | fox, porcupine, possum, raccoon, skunk
non-insect invertebrates       | crab, lobster, snail, spider, worm
people	                       | baby, boy, girl, man, woman
reptiles                       | crocodile, dinosaur, lizard, snake, turtle
small mammals                  | hamster, mouse, rabbit, shrew, squirrel
trees                          | maple, oak, palm, pine, willow
vehicles 1                     | bicycle, bus, motorcycle, pickup truck, train
vehicles 2                     | lawn-mower, rocket, streetcar, tank, tractor

"""
    dataset.load_direct([inputs], [targets], [labels])
