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
import copy
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
    plt.plot(xs, ys, symbol)
    plt.show()
    #bytes = io.BytesIO()
    #plt.savefig(bytes, format='svg')
    #svg = bytes.getvalue()
    #plt.close(fig)
    #return SVG(svg.decode())

def plot(lines, width=8.0, height=4.0, xlabel="time", ylabel=""):
    """
    SVG(plot([["Error", "+", [1, 2, 4, 6, 1, 2, 3]]],
             ylabel="error",
             xlabel="hello"))
    """
    if plt is None:
        raise Exception("matplotlib was not loaded")
    plt.rcParams['figure.figsize'] = (width, height)
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
    plt.show()
    #bytes = io.BytesIO()
    #plt.savefig(bytes, format='svg')
    #svg = bytes.getvalue()
    #plt.close(fig)
    #return SVG(svg.decode())


def plot_activations(net, output_layer="output", output_index=0,
                     input_layer="input",input_index1=0, input_index2=1,
                     colormap=None, default_input_value=0, resolution=None,
                     act_range=None):
    if colormap is None: colormap = get_colormap()
    if plt is None:
        raise Exception("matplotlib was not loaded")
    if act_range is not None:
        act_min, act_max = act_range
    else:
        act_min, act_max = net[input_layer].get_act_minmax()
    out_min, out_max = net[output_layer].get_act_minmax()
    if resolution is None:
        default_pixels = 50
        resolution = (act_max - act_min) / default_pixels
    xmin, xmax, xstep = act_min, act_max, resolution
    ymin, ymax, ystep = act_min, act_max, resolution
    xspan = xmax - xmin
    yspan = ymax - ymin
    xpixels = int(xspan/xstep)+1
    ypixels = int(yspan/ystep)+1
    mat = np.zeros((ypixels, xpixels))
    ovector = net[input_layer].make_dummy_vector(default_input_value)
    for row in range(ypixels):
        for col in range(xpixels):
            # (x,y) corresponds to lower left corner point of pixel
            x = xmin + xstep*col
            y = ymin + ystep*row
            vector = copy.copy(ovector)
            vector[input_index1] = x
            vector[input_index2] = y
            mat[row,col] = net.propagate_from(input_layer, vector,
                                              output_layer, visualize=False)[output_index]
    fig, ax = plt.subplots()
    axim = ax.imshow(mat, origin='lower', cmap=colormap, vmin=out_min, vmax=out_max)
    ax.set_title("Activation of %s[%s]" % (output_layer, output_index))
    ax.set_xlabel("%s[%s]" % (input_layer, input_index1))
    ax.set_ylabel("%s[%s]" % (input_layer, input_index2))
    ax.xaxis.tick_bottom()
    ax.set_xticks([i*(xpixels-1)/4 for i in range(5)])
    ax.set_xticklabels([xmin+i*xspan/4 for i in range(5)])
    ax.set_yticks([i*(ypixels-1)/4 for i in range(5)])
    ax.set_yticklabels([ymin+i*yspan/4 for i in range(5)])
    cbar = fig.colorbar(axim)
    plt.show()
