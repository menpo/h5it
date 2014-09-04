from __future__ import unicode_literals
import tempfile
from nose.tools import raises
from hdf5able import save, load, HDF5able


path = tempfile.mkstemp()[1]


class Foo(HDF5able):

    def __init__(self):
        self.a = 1
        self.b = 'a'
        self.c = {'x': None}
        self.d = None
        self.e = False
        self.f = (1, 5, 1)
        self.g = ['h', 142+32j, -159081.1340]

    def __eq__(self, other):
        return (self.a == other.a and
                self.b == other.b and
                self.c == other.c and
                self.d == other.d and
                self.e == other.e and
                self.f == other.f and
                self.g == other.g)


class FooCustom(Foo):

    def h5_dict_to_serializable_dict(self):
        d = self.__dict__.copy()
        d['f'] = list(d['f'])
        d['e_another_name'] = not d.pop('e')
        return d

    @classmethod
    def h5_dict_from_serialized_dict(cls, d, version):
        d['f'] = tuple(d['f'])
        d['e'] = not d.pop('e_another_name')
        return d


class FooNonHDF5able(object):

    def __init__(self):
        self.a = None


def test_save_hdf5able():
    save(path, Foo())

@raises(ValueError)
def test_save_object_raises():
    save(path, FooNonHDF5able())


def test_load_hdf5able():
    x = Foo()
    save(path, x)
    y = load(path)
    assert y == x
    assert type(y) == Foo


def test_load_custom_hdf5able():
    x = FooCustom()
    save(path, x)
    y = load(path)
    assert y == x
    print(type(y))
    assert type(y) == FooCustom
