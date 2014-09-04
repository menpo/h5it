from __future__ import unicode_literals
import tempfile

import numpy as np
from nose.tools import raises
from pathlib import Path, PosixPath

from hdf5able import save, load
from hdf5able.base import isPython2, u

if isPython2:
    unicode_type = unicode
else:
    unicode_type = str

path = tempfile.mkstemp()[1]


def test_save_integer():
    save(path, 1)


def test_save_float():
    save(path, -14.1512)


def test_save_complex():
    save(path, 25.1512 + 7j)


def test_save_unicode():
    save(path, u'unicode str')


def test_save_byte_str():
    save(path, b'byte str')


def test_save_bool():
    save(path, True)


def test_save_none():
    save(path, None)


def test_save_ndarray():
    x = np.random.rand(151, 16, 16, 1)
    save(path, x)


def test_save_path():
    save(path, Path('/some/path/here'))


def test_save_empty_list():
    save(path, [])


def test_save_list():
    save(path, [1, 'a', None, True])


def test_save_recursive_list():
    save(path, [1, 'a', None, True, [b'another', -125.14]])


def test_save_empty_tuple():
    save(path, tuple())


def test_save_tuple():
    save(path, (1, 'xyx', 15.161))


def test_save_recursive_tuple():
    save(path, (1, 'xyx', 15.161, (None, {'a': 1}, [1, 3, 4], True)))


def test_save_empty_dict():
    save(path, {})


@raises(ValueError)
def test_save_dict_with_non_string_keys():
    save(path, {1: 'a', None: True})


def test_save_dict():
    save(path, {'b': 2, 'c': True})


def test_save_dict_recursive():
    save(path, {'b': 2, 'c': True, 'd': [1, None, {'key': 2.5012343}]})


def test_load_integer():
    x = 1
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == int


def test_load_float():
    x = +11241.151214541
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == float


def test_load_complex():
    x = -1589.151214541 - 1390815.24155j
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == complex


def test_load_unicode():
    x = "some unicode"
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == unicode_type


def test_load_byte_str():
    x = b"some byte str"
    save(path, x)
    y = load(path)
    assert y == u(x)
    assert type(y) == unicode_type


def test_load_bool():
    x = False
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == bool


def test_load_none():
    x = None
    save(path, x)
    y = load(path)
    assert y == x
    assert isinstance(y, type(None))
    assert y is None


def test_load_ndarray():
    x = np.random.rand(151, 16, 16, 1)
    save(path, x)
    y = load(path)
    assert np.all(y == x)
    assert type(y) == np.ndarray


def test_load_posix_path():
    x = Path('/some/path/here')
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == PosixPath


def test_load_empty_list():
    x = []
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == list


def test_load_list():
    x = [1, 'a', None, True]
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == list


def test_load_recursive_list():
    x = [1, 'a', None, True, ['another', -125.14]]
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == list


def test_load_empty_tuple():
    x = tuple()
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == tuple


def test_load_tuple():
    x = (1, 'xyx', 15.161)
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == tuple


def test_load_recursive_tuple():
    x = (1, 'xyx', 15.161, (None, {'a': 1}, [1, 3, 4], True))
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == tuple


def test_load_empty_dict():
    x = {}
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == dict


def test_load_dict():
    x = {'b': 2, 'c': True}
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == dict


def test_load_recursive_dict():
    x = {'b': 2, 'c': True, 'd': [1, None, {'key': 2.5012343}]}
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == dict
