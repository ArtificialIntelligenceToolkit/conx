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
    
    # FIXME: why is input_layer even an argument here, since
    # propagate_to always requires starting from "input"???
    assert input_layer == 'input', "for now, input_layer must be 'input'"

    default_ranges = {'sigmoid': (0, 1),
                      'tanh': (-1, 1),
                      'relu': (0, 2),
                      'linear': (-2, 2),
                      'softmax': (0, 1)}

    if act_range is not None:
        act_min, act_max = act_range
    elif net[input_layer].activation in default_ranges:
        act_min, act_max = default_ranges[net[input_layer].activation]
    # elif net[input_layer].minmax is not None:
    #     act_min, act_max = net[input_layer].minmax
    else:
        act_min, act_max = -1, 1

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
    for row in range(ypixels):
        for col in range(xpixels):
            # (x,y) corresponds to lower left corner point of pixel
            x = xmin + xstep*col
            y = ymin + ystep*row
            vector = net[input_layer].make_dummy_vector(default_input_value)
            vector[input_index1] = x
            vector[input_index2] = y
            mat[row,col] = net.propagate_to(output_layer, vector, visualize=False)[output_index]
    # print("matrix is:")
    # for row in mat:
    #     print(''.join(['%.6f ' % x for x in row]))
    fig, ax = plt.subplots()
    cax = ax.imshow(mat, origin='lower', cmap=colormap)
    ax.set_title("Activation of %s[%s]" % (output_layer, output_index))
    ax.set_xlabel("%s[%s]" % (input_layer, input_index1))
    ax.set_ylabel("%s[%s]" % (input_layer, input_index2))
    ax.xaxis.tick_bottom()
    ax.xaxis.set_ticks([i*(xpixels-1)/4 for i in range(5)])
    ax.xaxis.set_ticklabels([xmin+i*xspan/4 for i in range(5)])
    ax.yaxis.set_ticks([i*(ypixels-1)/4 for i in range(5)])
    ax.yaxis.set_ticklabels([ymin+i*yspan/4 for i in range(5)])
    # added a colorbar. FIXME: ideally, maximum red should always map to
    # act_min and maximum blue should map to act_max, but i don't know how
    # to do this. matplotlib automatically rescales the colorbar as it likes.
    cbar = fig.colorbar(cax)
    plt.show()

    ##plt.ylim([min2,max2])
    ##plt.xlim([min1,max1])
    ## Turn off ticks
    ##figure.axes.get_xaxis().set_visible(False)
    ##figure.axes.get_yaxis().set_visible(False)
    #bytes = io.BytesIO()
    #plt.savefig(bytes, format='svg')
    #svg = bytes.getvalue()
    #plt.close(fig)
    #return SVG(svg.decode())
