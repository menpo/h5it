import inspect
import importlib
from collections import namedtuple
from functools import partial

SerializedFunc = namedtuple('SerializedFunc', ['name', 'source', 'modules'])


def serialize_f_and_test(f, modules, args=None, kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    # save the function down
    f = serialize_f(f, modules)
    # attempt to re-serialize
    f_rebuilt = deserialise_f(*f)
    f_rebuilt(*args, **kwargs)
    print('function was succesfully rebuilt')
    return f


def serialize_f(f, modules):
    # build the namespace mapping {name: item}
    name_to_entity = namespace_for_modules(modules)
    module_names = [module_to_str(m) for m in modules]
    # build the inverse mapping {item: name}
    entity_to_name = {v: k for k, v in name_to_entity.iteritems()}
    # see if f is in the module namespace
    name = entity_to_name.get(f)
    if name is not None:
        # f is directly in the namespace - easy to serialize.
        print("{} is in the namespace - being saved directly".format(f))
        return SerializedFunc(name, '', module_names)
    elif hasattr(f, 'source'):
        # f is a novel function that has it's own source attached.
        print("{} has been previously exported, reusing source".format(f))
        return SerializedFunc(f.__name__, f.source, module_names)
    elif isinstance(f, partial):
        # Special case: f is a partially applied function (that isn't directly
        # in the namespace of the modules)
        raise ValueError("Partial function serialization is not yet supported")
    else:
        # f is a novel function and needs to be introspected for it's
        # definition
        print("{} is an alien symbol - source code required".format(f))
        source = inspect.getsource(f)
        return SerializedFunc(f.__name__, source, module_names)


def deserialise_f(name, source, module_strs):
    modules = [str_to_module(m) for m in module_strs]
    namespace = namespace_for_modules(modules)
    # exec the function in this namespace
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
    f.source = source
    return f

# def source_for_partial(p):
#     arg_str = [str(a) for a in boo1.args]
#     kwarg_str = ['{}={}'.format(*i) for i in boo1.keywords.items()]
#     args = ', '.join([p.func.__name__] + arg_str + kwarg_str)
#     return "partial({})".format(args)