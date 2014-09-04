import tempfile
from nose.tools import raises
from hdf5able import SerializableCallable, save, load

path = tempfile.mkstemp()[1]


def test_serialize_callable_and_test_basic():
    from hdf5able.callable import serialize_callable_and_test as sct

    def a_mock_function(x, y):
        pass

    sct(a_mock_function, [])


def test_get_source():
    from hdf5able.callable import extract_source

    def a_mock_function(x, y):
        pass

    source = """def a_mock_function(x, y):\n    pass"""
    assert source == extract_source(a_mock_function)


def test_use_allowed_global():
    from hdf5able.callable import serialize_callable_and_test
    from itertools import product
    import itertools

    def a_mock_function(*args):
        return product(args)

    serialize_callable_and_test(a_mock_function, [itertools])


@raises(NameError)
def test_use_unallowed_global_raises_never_present():
    from hdf5able.callable import serialize_callable_and_test

    def a_mock_function(*args):
        return product(args)

    serialize_callable_and_test(a_mock_function, [])


@raises(NameError)
def test_use_unallowed_global_raises_was_present():
    from hdf5able.callable import serialize_callable_and_test
    from itertools import product

    def a_mock_function(*args):
        return product(args)

    serialize_callable_and_test(a_mock_function, [])


def test_save_serializable_callable():
    from itertools import product
    import itertools

    def a_mock_function(*args):
        return product(args)

    sc = SerializableCallable(a_mock_function, [itertools])
    save(path, sc)


def test_load_serializable_callable():

    def a_function(*args):
        import numpy as np
        return np.sum(args)

    sc = SerializableCallable(a_function, [])
    save(path, sc)
    f = load(path)
    assert a_function(2, 4) == f(2, 4)
