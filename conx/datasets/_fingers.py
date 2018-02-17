import numpy as np

def fingers(dataset, path='fingers.npz'):
    inputs, labels = load_dataset_npz(
        path,
        "https://raw.githubusercontent.com/Calysto/conx/master/data/fingers.npz")
    inputs = inputs.astype('float32')
    inputs /= 255
    make_target_vector = lambda label: [int(label == n) for n in range(6)]
    targets = np.array([make_target_vector(l) for l in labels]).astype('uint8')
    dataset.name = "Fingers"
    dataset.description = """
This dataset contains 12,000 RGB images of human hands showing different
numbers of fingers, from zero to five.  The same fingers are always used
to represent each number category (e.g., all images of "two" have raised
index and middle fingers).  Each image is a 30 x 40 x 3 array of
floating-point numbers in the range 0 to 1.  The target data consists of
one-hot binary vectors of size 6 corresponding to the classification
categories "zero" through "five".  There are 2000 images for each category.

Created by Shreeda Segan and Albert Yu at Sarah Lawrence College.
"""
    dataset.load_direct([inputs], [targets], [labels])


def load_dataset_npz(path, url):
    """loads an .npz file of saved image data, and returns the images and their
    associated labels as numpy arrays
    """
    from keras.utils import get_file
    path = get_file(path, origin=url)
    f = np.load(path)
    images, labels = f['data'], f['labels']
    return images, labels
