# coding=utf-8
from __future__ import unicode_literals

import tempfile
from nose.tools import raises
import os
from os.path import join as j
import numpy as np
from pathlib import (Path, PosixPath, PurePosixPath,
                     WindowsPath, PureWindowsPath)

from h5it import dump, load, H5itUnpicklingError
from h5it.base import is_py2, host_is_posix, host_is_windows
import pickle

test_dir = os.path.dirname(os.path.realpath(__file__))
files_dir = os.path.join(test_dir, 'files')

unicode_test_str = (u'σκουλικομερμυγκότρυπα ασπρη πέτρα '
                    u'ξέξασπρη κι από τον ήλιο ξεξασπρότερη')

py2_unicode_pickle = j(files_dir, 'py2_unicode_proto2_bin.pickle')
py3_unicode_pickle = j(files_dir, 'py3_unicode_proto2_bin.pickle')
py2_unicode_h5it = j(files_dir, 'py2_unicode.hdf5')
py3_unicode_h5it = j(files_dir, 'py3_unicode.hdf5')

py2_bytes_pickle = j(files_dir, 'py2_bytes_proto2_bin.pickle')
py3_bytes_pickle = j(files_dir, 'py3_bytes_proto2_bin.pickle')
py2_bytes_h5it = j(files_dir, 'py2_bytes.hdf5')
py3_bytes_h5it = j(files_dir, 'py3_bytes.hdf5')

if is_py2:
    unicode_type = unicode
    bytes_type = str
else:
    unicode_type = str
    bytes_type = bytes

path = tempfile.mkstemp()[1]


def test_save_integer():
    dump(1, path)


def test_save_float():
    dump(-14.1512, path)


def test_save_complex():
    dump(25.1512 + 7j, path)


def test_save_unicode():
    dump(unicode_test_str, path)


def test_save_byte_str():
    dump(b'byte str', path)


def test_save_bool():
    dump(True, path)


def test_save_none():
    dump(None, path)


def test_save_ndarray():
    x = np.random.rand(151, 16, 16, 1)
    dump(x, path)


def test_save_path():
    dump(Path('/some/path/here'), path)


def test_save_empty_list():
    dump([], path)


def test_save_list():
    dump([1, 'a', None, True], path)


def test_save_recursive_list():
    dump([1, 'a', None, True, [b'another', -125.14]], path)


def test_save_empty_tuple():
    dump(tuple(), path)


def test_save_tuple():
    dump((1, 'xyx', 15.161), path)


def test_save_recursive_tuple():
    dump((1, 'xyx', 15.161, (None, {'a': 1}, [1, 3, 4], True)), path)


def test_save_empty_dict():
    dump({}, path)


def test_save_dict_with_non_string_keys():
    dump({1: 'a', None: True}, path)


def test_save_dict():
    dump({'b': 2, 'c': True}, path)


def test_save_dict_recursive():
    dump({'b': 2, 'c': True, 'd': [1, None, {'key': 2.5012343}]}, path)


def test_load_integer():
    x = 1
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == int


def test_load_float():
    x = +11241.151214541
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == float


def test_load_complex():
    x = -1589.151214541 - 1390815.24155j
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == complex


def test_load_unicode():
    dump(unicode_test_str, path)
    y = load(path)
    assert y == unicode_test_str
    assert type(y) == unicode_type


def test_load_byte_str():
    x = b"some byte str"
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == bytes_type


def test_load_bool():
    x = False
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == bool


def test_load_none():
    x = None
    dump(x, path)
    y = load(path)
    assert y == x
    assert isinstance(y, type(None))
    assert y is None


def test_load_ndarray():
    x = np.random.rand(151, 16, 16, 1)
    dump(x, path)
    y = load(path)
    assert np.all(y == x)
    assert type(y) == np.ndarray


if host_is_posix:
    def test_load_posix_path_on_posix():
        x = PosixPath('/some/path/here')
        dump(x, path)
        y = load(path)
        assert y == x
        assert type(y) == PosixPath

    def test_load_windows_path_on_posix():
        x = PureWindowsPath('C:\some\path\here')
        dump(x, path)
        y = load(path)
        assert y == x
        assert type(y) == PureWindowsPath


if host_is_windows:
    def test_load_posix_path_on_windows():
        x = PurePosixPath('/some/path/here')
        dump(x, path)
        y = load(path)
        assert y == x
        assert type(y) == PurePosixPath

    def test_load_windows_path_on_windows():
        x = WindowsPath('C:\some\path\here')
        dump(x, path)
        y = load(path)
        assert y == x
        assert type(y) == WindowsPath


def test_load_empty_list():
    x = []
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == list


def test_load_list():
    x = [1, 'a', None, True]
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == list


def test_load_recursive_list():
    x = [1, 'a', None, True, ['another', -125.14]]
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == list


def test_load_empty_tuple():
    x = tuple()
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == tuple


def test_load_tuple():
    x = (1, 'xyx', 15.161)
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == tuple


def test_load_recursive_tuple():
    x = (1, 'xyx', 15.161, (None, {'a': 1}, [1, 3, 4], True))
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == tuple


def test_load_empty_dict():
    x = {}
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == dict


def test_load_dict():
    x = {'b': 2, 'c': True}
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == dict


def test_load_dict_with_non_string_keys():
    x = {1: 'a', None: True}
    dump(x, path)
    y = load(path)
    assert y == x


def test_load_recursive_dict():
    x = {'b': 2, 'c': True, 'd': [1, None, {'key': 2.5012343}]}
    dump(x, path)
    y = load(path)
    assert y == x
    assert type(y) == dict


def test_load_reference():
    c = [1, 2, 3]
    a = {'c_from_a': c}
    b = {'c_from_b': c}
    dump((a, b), path)
    a_l, b_l = load(path)
    assert id(a_l['c_from_a']) == id(b_l['c_from_b'])


def test_save_with_path():
    dump(None, Path(path))


def test_save_with_path_load_with_str():
    dump(None, Path(path))
    assert load(path) is None


def test_save_with_path_load_with_path():
    dump(None, Path(path))
    assert load(Path(path)) is None


def test_save_with_str_load_with_path():
    dump(None, path)
    assert load(Path(path)) is None


if is_py2:

    def test_load_unicode_from_py3_on_py2():
        with open(py3_unicode_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py3_unicode_h5it)
        assert type(y) == type(x)
        assert x == y

    def test_load_bytes_from_py3_on_py2():
        with open(py3_bytes_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py3_bytes_h5it)
        assert type(y) == type(x)
        assert x == y

    def test_load_unicode_from_py2_on_py2():
        with open(py2_unicode_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py2_unicode_h5it)
        assert type(y) == type(x)
        assert x == y

    def test_load_bytes_from_py2_on_py2():
        with open(py2_bytes_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py2_bytes_h5it)
        assert type(y) == type(x)
        assert x == y

else:

    def test_load_unicode_from_py3_on_py3():
        with open(py3_unicode_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py3_unicode_h5it)
        assert type(y) == type(x)
        assert x == y

    def test_load_bytes_from_py3_on_py3():
        with open(py3_bytes_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py3_bytes_h5it)
        assert type(y) == type(x)
        assert x == y

    def test_load_bytes_from_py3_on_py3_encoding_ascii_shouldnt_matter():
        with open(py3_bytes_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py3_bytes_h5it, encoding='ASCII')
        assert type(y) == type(x)
        assert x == y

    def test_load_unicode_from_py2_on_py3():
        with open(py2_unicode_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py2_unicode_h5it)
        assert type(y) == type(x)
        assert x == y

    def test_load_bytes_from_py2_on_py3_encoding_default():
        with open(py2_bytes_pickle, 'rb') as f:
            x = pickle.load(f)
        y = load(py2_bytes_h5it)
        assert type(y) == type(x)
        assert x == y

    def test_load_bytes_from_py2_on_py3_encoding_ascii():
        with open(py2_bytes_pickle, 'rb') as f:
            x = pickle.load(f, encoding='ASCII')
        y = load(py2_bytes_h5it, encoding='ASCII')
        assert type(y) == type(x)
        assert x == y

    def test_load_bytes_from_py2_on_py3_encoding_bytes():
        with open(py2_bytes_pickle, 'rb') as f:
            x = pickle.load(f, encoding='bytes')
        y = load(py2_bytes_h5it, encoding='bytes')
        assert type(y) == type(x)
        assert x == y

    @raises(H5itUnpicklingError)
    def test_load_invalid_encoding():
        load(py2_bytes_pickle, encoding='asda')
