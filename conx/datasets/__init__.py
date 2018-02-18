
## All functions here must be loadable datasets
## All modules must be named differently from their functions!
## Otherwise, confuses tools like nose, inspect, etc.

from ._mnist import mnist
from ._cifar10 import cifar10
from ._cifar100 import cifar100

from .cmu_faces import cmu_faces_full_size
from .cmu_faces import cmu_faces_half_size
from .cmu_faces import cmu_faces_quarter_size

from ._gridfonts import gridfonts, figure_ground_a

from ._fingers import fingers
