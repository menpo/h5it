from .base import (load, save,  # main API for saving and loading files.
                   HDF5able,    # subclass to make things HDF5able
                   SerializableCallable)  # specialist class for callables

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
