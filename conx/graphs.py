# conx - a neural network library
#
# Copyright (c) 2017 Douglas S. Blank <dblank@cs.brynmawr.edu>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301  USA


import base64
import io
import numpy as np
from .utils import get_colormap

from IPython.display import SVG

try:
    import matplotlib.pyplot as plt
except:
    plt = None

def plot_f(f, frange=(-1, 1, .1), symbol="o-"):
    """
    Plot a function.
    """
    xs = np.arange(*frange)
    ys = [f(x) for x in xs]
    fig = plt.figure()
    plt.plot(xs, ys, symbol)
    bytes = io.BytesIO()
    plt.savefig(bytes, format='svg')
    svg = bytes.getvalue()
    plt.close(fig)
    return SVG(svg.decode())

def plot(lines, width=8.0, height=4.0, xlabel="time", ylabel=""):
    """
    SVG(plot([["Error", "+", [1, 2, 4, 6, 1, 2, 3]]],
             ylabel="error",
             xlabel="hello"))
    """
    if plt is None:
        raise Exception("matplotlib was not loaded")
    plt.rcParams['figure.figsize'] = (width, height)
    fig = plt.figure()
    for (label, symbol, data) in lines:
        kwargs = {}
        args = [data]
        if label:
            kwargs["label"] = label
        if symbol:
            args.append(symbol)
        plt.plot(*args, **kwargs)
    if any([line[0] for line in lines]):
        plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    bytes = io.BytesIO()
    plt.savefig(bytes, format='svg')
    svg = bytes.getvalue()
    plt.close(fig)
    return SVG(svg.decode())

def plot_activations(net, output_layer="output", output_index=0,
                     input_layer="input",input_index1=0, input_index2=1,
                     colormap=None, default_input_value=0, resolution=0.1):
    if colormap is None: colormap = get_colormap()
    if plt is None:
        raise Exception("matplotlib was not loaded")
    act_range = net[input_layer].minmax
    if act_range is None:
        slice = (-1, 1, resolution)
    else:
        slice = (act_range[0], act_range[1], resolution)
    min1, max1, step1 = slice
    min2, max2, step2 = slice
    fig = plt.figure()
    resolution1 = int((max1 - min1) / step1)
    resolution2 = int((max2 - min2) / step2)
    mat = np.zeros((resolution1, resolution2))
    for i1 in range(resolution1):
        for i2 in range(resolution2):
            input1 = i1/resolution1 * (max1 - min1) + min1
            input2 = i2/resolution2 * (max2 - min2) + min2
            vector = net[input_layer].make_dummy_vector(default_input_value)
            vector[input_index1] = input1
            vector[input_index2] = input2
            mat[i1, i2] = net.propagate_to(output_layer, vector,
                                           visualize=False)[output_index]
    figure = plt.matshow(mat, origin="lower", cmap=colormap)
    plt.title("Activation of %s[%s]" % (output_layer, output_index))
    plt.xlabel("%s[%s]" % (input_layer, input_index1))
    plt.ylabel("%s[%s]" % (input_layer, input_index2))
    figure.axes.set_xticks([0, 0.25 * resolution1, 0.50 * resolution1, 0.75 * resolution1])
    figure.axes.set_xticklabels([min1,
                                 (max1 - min1) * 0.25,
                                 (max1 - min1) * 0.50,
                                 (max1 - min1) * 0.75,])
    figure.axes.set_yticks([0, 0.25 * resolution2, 0.50 * resolution2, 0.75 * resolution2])
    figure.axes.set_yticklabels([min2,
                                 (max2 - min2) * 0.25,
                                 (max2 - min2) * 0.50,
                                 (max2 - min2) * 0.75,])
    ##plt.ylim([min2,max2])
    ##plt.xlim([min1,max1])
    ## Turn off ticks
    ##figure.axes.get_xaxis().set_visible(False)
    ##figure.axes.get_yaxis().set_visible(False)
    bytes = io.BytesIO()
    plt.savefig(bytes, format='svg')
    svg = bytes.getvalue()
    plt.close(fig)
    return SVG(svg.decode())
