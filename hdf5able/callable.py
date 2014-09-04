import inspect
import importlib
from collections import namedtuple, Hashable
from functools import partial
try:
    from unittest.mock import MagicMock, patch  # Python 3
except ImportError:
    from mock import MagicMock, patch  # Python 2
from inspect import getargspec

SerializedCallable = namedtuple('SerializedCallable',
                                ['name', 'source', 'modules'])


def serialize_callable_and_test(c, modules):
    # save the callable down
    serialized_c = serialize_callable(c, modules)
    # attempt to re-serialize
    deserialize_callable(*serialized_c)
    if serialized_c.source is not None:
        # test the callable source
        namespace = namespace_for_modules(modules)
        mock_namespace = {k: MagicMock() for k in namespace
                          if not (k.startswith('__') and k.endswith('__'))}
        # mock namespace means the funciton has access to the desired
        # namespace only, but everything in there is a MagicMock instance
        mock_c_rebuilt = deserialize_callable_in_namespace(
            serialized_c.name, serialized_c.source, mock_namespace)
        test_callable(mock_c_rebuilt)
    return serialized_c


def test_callable(c):
    nargs = len(getargspec(c).args)
    args = [MagicMock() for _ in range(nargs)]

    # Store original __import__
    orig_import = __import__

    def import_mock(name, *args):
        orig_import(name)
        return MagicMock()

    from .base import isPython2
    if isPython2:
        import_string = '__builtin__.__import__'
    else:
        import_string = 'builtins.__import__'

    with patch(import_string, side_effect=import_mock):
        c(*args)


def serialize_callable(c, modules):
    # build the namespace mapping {name: callable}
    name_to_callable = {n: s for n, s in namespace_for_modules(modules).items()
                        if callable(s) and isinstance(s, Hashable)}
    module_names = [module_to_str(m) for m in modules]
    # build the inverse mapping for callables {callable: name}
    callable_to_name = {s: n for n, s in name_to_callable.items()}
    # see if c is in the module namespace
    name = callable_to_name.get(c)
    if name is not None:
        # c is directly in the namespace - easy to serialize.
        return SerializedCallable(name, None, module_names)
    elif hasattr(c, 'h5_source'):
        # f is a novel function that has it's own source attached.
        return SerializedCallable(c.__name__, c.source, module_names)
    elif isinstance(c, partial):
        # Special case: c is a partially applied function (that isn't directly
        # in the namespace of the modules)
        # currently not supported, could be added
        raise ValueError("Partial function serialization is not yet supported")
    else:
        # c is a novel function and needs to be introspected for it's
        # definition
        source = extract_source(c)
        return SerializedCallable(c.__name__, source, module_names)


def extract_source(c):
    source = inspect.getsource(c)
    lines = source.splitlines()
    l = lines[0]
    # find any leading whitespace on the function and strip it
    leading_space = len(l) - len(l.lstrip())
    return '\n'.join([l[leading_space:] for l in lines])


def deserialize_callable(name, source, modules):
    namespace = namespace_for_modules([str_to_module(m) for m in modules])
    return deserialize_callable_in_namespace(name, source, namespace)


def deserialize_callable_in_namespace(name, source, namespace):
    if source is None:
        # must be directly in namespace
        return namespace[name]
    else:
        # exec the callable in this namespace
        return safe_exec(source, namespace, name)


def str_to_module(module_str):
    return importlib.import_module(module_str)


def module_to_str(module):
    return module.__name__


def namespace_for_module(module):
    return dict(inspect.getmembers(module))


def namespace_for_modules(modules):
    namespace = {}
    for m in modules:
        namespace.update(namespace_for_module(m))
    return namespace


def safe_exec(source, namespace, name):
    r"""
    Execs a function definition a certain namespace, returning the
    function.
    """
    namespace = namespace.copy()
    exec(source, namespace)
    f = namespace[name]
    f.h5_source = source
    return f

# rough idea of partial support, difficult though.
# Python 3.3+ makes this trivial:
#     https://docs.python.org/3/library/inspect.html#inspect.signature
#
# def source_for_partial(p):
#     arg_str = [str(a) for a in p.args]
#     kwarg_str = ['{}={}'.format(*i) for i in p.keywords.items()]
#     args = ', '.join([p.func.__name__] + arg_str + kwarg_str)
#     return "partial({})".format(args)
