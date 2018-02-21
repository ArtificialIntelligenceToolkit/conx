# conx - a neural network library
#
# Copyright (c) 2016-2017 Douglas S. Blank <dblank@cs.brynmawr.edu>
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import matplotlib
## If no DISPLAY, then set the matplotlib backend
## to an alternate to work if in console (Tk, Qt, etc).
if False: # sys.platform == "darwin":
    pass # let's not mess with OSX
else:
    if (("DISPLAY" not in os.environ) or
        (os.environ["DISPLAY"] == "")):
        if (matplotlib.get_backend() in [
                'module://ipykernel.pylab.backend_inline',
                'NbAgg',
                ]):
            pass  ## Don't change if server has no DISPLAY but is connected to notebook
        else:
            matplotlib.use('Agg') # something that will work
from ._version import __version__, VERSION
from .network import *
from .layers import *
from .dataset import *

print("Conx, version %s" % __version__, file=sys.stderr)
