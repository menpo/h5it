from .base import load, dump  # main API for saving and loading files.
from .stdpickle import H5itPicklingError, H5itUnpicklingError

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
