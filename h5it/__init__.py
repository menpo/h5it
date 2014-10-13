from .base import load, dump  # main API for saving and loading files.

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
