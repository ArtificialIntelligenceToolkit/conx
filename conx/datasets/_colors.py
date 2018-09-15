import conx as cx
import numpy as np
import csv

def colors(*args, path='colors.csv',
           url="https://raw.githubusercontent.com/Calysto/conx-data/master/colors/colors.csv",
           **kwargs):
    dataset = cx.Dataset()
    from keras.utils import get_file
    path = get_file(path, origin=url)
    fp = open(path, "r")
    reader = csv.reader(fp)
    inputs = []
    labels = []
    targets = []
    count = 1
    for line in reader:
        name, r, g, b = line
        if name == "name": continue # first line is header
        inputs.append([float(int(r)/255), float(int(g)/255), float(int(b)/255)])
        targets.append([count])
        labels.append(name)
        count += 1
    inputs = np.array(inputs, dtype='float32')
    targets = np.array(targets, dtype='uint16')
    dataset.name = "Colors"
    dataset.description = """
Original source: https://github.com/andrewortman/colorbot

This dataset also includes some ignored in original data.

Inspired by:

* http://aiweirdness.com/tagged/paint-colors

When initially loaded, this database has the following format:

* labels: [color_name_string, ...] # order matches target
* inputs: [[red, green, blue], ...] # scaled between 0 and 1
* targets: [[int], ...] # number of label

For example:

```
>>> import conx as cx
>>> ds = cx.Dataset.get("colors")
>>> ds.labels[0], ds.inputs[0], ds.targets[0]
('tidewater',
 [0.7686274647712708, 0.843137264251709, 0.8352941274642944],
 [1])
```
"""
    dataset.load_direct([inputs], [targets], [labels])
    return dataset
