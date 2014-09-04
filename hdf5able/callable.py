import inspect
import importlib
from collections import namedtuple
from functools import partial
from mock import Mock, patch
from inspect import getargspec

SerializedCallable = namedtuple('SerializedCallable',
                                ['name', 'source', 'modules'])


def serialize_callable_and_test(c, modules, args=None, kwargs=None):
    # save the callable down
    serialized_c = serialize_callable(c, modules)
    # attempt to re-serialize
    c_rebuilt = deserialize_callable(*serialized_c)
    if serialized_c.source is not None:
        # test the callable
        namespace = namespace_for_modules(modules)
        #test_callable(c_rebuilt, namespace, args, kwargs)
        print('callable was successfully rebuilt')
    else:
        print('callable is in namespace - no need to test')
    return serialized_c


def test_callable(c, namespace, args=None, kwargs=None):
    if kwargs is None:
        kwargs = {}
    if args is None:
        # user is not supplying args, so we need to mock
        nargs = len(getargspec(c).args)
        args = [Mock() for _ in range(nargs)]

        # Store original __import__
        orig_import = __import__

        def import_mock(name, *args):
            print('using import_mock')
            #orig_import(name, globals=namespace)
            return Mock()
        __import__ = import_mock
        c(*args, **kwargs)
    else:
        c(*args, **kwargs)


def serialize_callable(c, modules):
    # build the namespace mapping {name: symbol}
    name_to_symbol = namespace_for_modules(modules)
    module_names = [module_to_str(m) for m in modules]
    # build the inverse mapping for callables {callable: name}
    callable_to_name = {s: n for n, s in name_to_symbol.iteritems()
                        if callable(s)}
    # see if c is in the module namespace
    name = callable_to_name.get(c)
    if name is not None:
        # c is directly in the namespace - easy to serialize.
        print("{} is in the namespace - being saved directly".format(c))
        return SerializedCallable(name, None, module_names)
    elif hasattr(c, 'h5_source'):
        # f is a novel function that has it's own source attached.
        print("{} has been previously serialized, reusing source".format(c))
        return SerializedCallable(c.__name__, c.source, module_names)
    elif isinstance(c, partial):
        # Special case: c is a partially applied function (that isn't directly
        # in the namespace of the modules)
        # currently not supported, could be added
        raise ValueError("Partial function serialization is not yet supported")
    else:
        # c is a novel function and needs to be introspected for it's
        # definition
        print("{} is an alien symbol - source code required".format(c))
        source = inspect.getsource(c)
        return SerializedCallable(c.__name__, source, module_names)


def deserialize_callable(name, source, modules):
    namespace = namespace_for_modules([str_to_module(m) for m in modules])
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
