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


def plot_activations(net, from_layer, from_units, to_layer, to_unit,
                     colormap, default_from_layer_value, resolution,
                     act_range, show_values):
    # first do some error checking
    assert net[from_layer] is not None, "unknown layer: %s" % (from_layer,)
    assert type(from_units) in (tuple, list) and len(from_units) == 2, \
        "expected a pair of ints for the %s units but got %s" % (from_layer, from_units)
    ix, iy = from_units
    assert 0 <= ix < net[from_layer].size, "no such %s layer unit: %d" % (from_layer, ix)
    assert 0 <= iy < net[from_layer].size, "no such %s layer unit: %d" % (from_layer, iy)
    assert net[to_layer] is not None, "unknown layer: %s" % (to_layer,)
    assert type(to_unit) is int, "expected an int for the %s unit but got %s" % (to_layer, to_unit)
    assert 0 <= to_unit < net[to_layer].size, "no such %s layer unit: %d" % (to_layer, to_unit)

    if colormap is None: colormap = get_colormap()
    if plt is None:
        raise Exception("matplotlib was not loaded")
    act_min, act_max = net[from_layer].get_act_minmax() if act_range is None else act_range
    out_min, out_max = net[to_layer].get_act_minmax()
    if resolution is None:
        resolution = (act_max - act_min) / 50  # 50x50 pixels by default
    xmin, xmax, xstep = act_min, act_max, resolution
    ymin, ymax, ystep = act_min, act_max, resolution
    xspan = xmax - xmin
    yspan = ymax - ymin
    xpixels = int(xspan/xstep)+1
    ypixels = int(yspan/ystep)+1
    mat = np.zeros((ypixels, xpixels))
    ovector = net[from_layer].make_dummy_vector(default_from_layer_value)
    for row in range(ypixels):
        for col in range(xpixels):
            # (x,y) corresponds to lower left corner point of pixel
            x = xmin + xstep*col
            y = ymin + ystep*row
            vector = copy.copy(ovector)
            vector[ix] = x
            vector[iy] = y
            activations = net.propagate_from(from_layer, vector, to_layer, visualize=False)
            mat[row,col] = activations[to_unit]
    fig, ax = plt.subplots()
    axim = ax.imshow(mat, origin='lower', cmap=colormap, vmin=out_min, vmax=out_max)
    ax.set_title("Activation of %s[%s]" % (to_layer, to_unit))
    ax.set_xlabel("%s[%s]" % (from_layer, ix))
    ax.set_ylabel("%s[%s]" % (from_layer, iy))
    ax.xaxis.tick_bottom()
    ax.set_xticks([i*(xpixels-1)/4 for i in range(5)])
    ax.set_xticklabels([xmin+i*xspan/4 for i in range(5)])
    ax.set_yticks([i*(ypixels-1)/4 for i in range(5)])
    ax.set_yticklabels([ymin+i*yspan/4 for i in range(5)])
    cbar = fig.colorbar(axim)
    plt.show(block=False)
    # optionally print out a table of activation values
    if show_values:
        s = '\n'
        for y in np.linspace(act_max, act_min, 20):
            for x in np.linspace(act_min, act_max, 20):
                vector = [default_from_layer_value] * net[from_layer].size
                vector[ix] = x
                vector[iy] = y
                out = net.propagate_from(from_layer, vector, to_layer)[to_unit]
                s += '%4.2f ' % out
            s += '\n'
        separator = 100 * '-'
        s += separator
        print("%s\nActivation of %s[%d] as a function of %s[%d] and %s[%d]" %
              (separator, to_layer, to_unit, from_layer, ix, from_layer, iy))
        print("rows: %s[%d] decreasing from %.2f to %.2f" % (from_layer, iy, act_max, act_min))
        print("cols: %s[%d] increasing from %.2f to %.2f" % (from_layer, ix, act_min, act_max))
        print(s)

