import numpy as np
from keras.utils import get_file

def gridfonts(dataset):
    url = "https://raw.githubusercontent.com/Calysto/conx/master/data/gridfonts.npy"
    path = get_file("gridfonts.npy", origin=url)
    ds = np.load(path)
    ## [letters, labels]
    letters = np.array([matrix for matrix in ds[0]])
    targets = np.array([matrix for matrix in ds[0]])
    labels = np.array([char for char in ds[1]], dtype=str)
    dataset.name = "Gridfonts"
    dataset.description = """
This dataset originates from Douglas Hofstadter's research
group:

http://goosie.cogsci.indiana.edu/pub/gridfonts.data

![Gridfont Grid](https://github.com/Calysto/conx/raw/master/data/grid.png)

These data have been processed to make them neural
network friendly:

https://github.com/Calysto/conx/blob/master/data/gridfonts.py

The dataset is composed of letters on a 25 row x 9 column
grid. The inputs and targets are identical, and the labels
contain a string identifying the letter.

You can read a thesis using part of this dataset here:
https://repository.brynmawr.edu/compsci_pubs/78/
"""
    dataset.load_direct([letters], [targets], [labels])

def figure_ground_a(dataset):
    url = "https://raw.githubusercontent.com/Calysto/conx/master/data/figure_ground_a.npy"
    path = get_file("figure_ground_a.npy", origin=url)
    ds = np.load(path)
    ## [[[letter], [brim, body]], ...]
    letters = np.array([pair[0] for pair in ds])
    brims = np.array([pair[1][0] for pair in ds])
    bodies = np.array([pair[1][1] for pair in ds])
    dataset.name = "Figure-Ground A"
    dataset.description = """
This dataset (the so-called a-tabase) originates from Douglas
Hofstadter's research group:

http://goosie.cogsci.indiana.edu/pub/gridfonts.data

![Gridfont Grid](https://github.com/Calysto/conx/raw/master/data/grid.png)

These data (all the letter A) have been processed to make them neural
network friendly:

https://github.com/Calysto/conx/blob/master/data/gridfonts.py

The brim and body parts have been idenified manually.  The dataset is
composed of letters on a 17 row x 9 column grid (4 lines not used on
top and another 4 not used on the bottom of each letter were removed
from the original 25x9 latter images). The inputs are composed of the
full letter. The targets are composed of a picture of the body and
the brim.

You can read a thesis using part of this dataset here:
https://repository.brynmawr.edu/compsci_pubs/78/
"""
    dataset.load_direct([letters], [brims, bodies])
