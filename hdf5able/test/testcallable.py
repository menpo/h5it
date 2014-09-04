from nose.tools import raises


def test_serialize_callable_and_test_basic():
    from hdf5able.callable import serialize_callable_and_test as sct

    def a_mock_function(x, y):
        pass

    sct(a_mock_function, [])


def test_get_source():
    from hdf5able.callable import extract_source

    def a_mock_function(x, y):
        pass

    sc = """def a_mock_function(x, y):\n    pass"""
    assert sc == extract_source(a_mock_function)


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
