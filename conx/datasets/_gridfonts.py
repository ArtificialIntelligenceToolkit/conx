import numpy as np
from keras.utils import get_file

def gridfonts(dataset):
    url = "https://raw.githubusercontent.com/Calysto/conx/master/data/gridfonts.npy"
    path = get_file("gridfonts.npy", origin=url)
    ds = np.load(path)
    ## [[[letter], [brim, body]], ...]
    letters = np.array([pair[0] for pair in ds])
    brims = np.array([pair[1][0] for pair in ds])
    bodies = np.array([pair[1][1] for pair in ds])
    dataset.load_direct([letters], [brims, bodies])
