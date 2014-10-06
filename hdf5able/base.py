import sys
import importlib
from collections import namedtuple
from pathlib import Path, PosixPath

import numpy as np
import h5py


isPy2 = sys.version_info.major == 2
isPy3 = not isPy2

if isPy2:
    strTypes = (str, unicode)
    numberTypes = (int, long, float, complex)
else:
    strTypes = (str, bytes)
    numberTypes = (int, float, complex)


def u(s):
    if isPy2:
        return unicode(s)
    else:
        return str(s)

# ------------------------------ IMPORTS ------------------------------ #

ListItem = namedtuple('ListItem', ['i', 'item'])


def load_list(parent, name, memo):
    node = parent[name]
    unordered = []
    for i, j in enumerate(node.keys()):
        unordered.append(ListItem(int(j), h5_import(node, j, memo)))
    ordered = sorted(unordered)
    counts = [x.i for x in ordered]
    items = [x.item for x in ordered]
    if counts != list(range(len(counts))):
        raise ValueError("Attempted to import a list that is missing elements")
    return items


def load_tuple(parent, name, memo):
    return tuple(load_list(parent, name, memo))


def load_unicode_dict(parent, name, memo):
    node = parent[name]
    imported_dict = {}
    for k in node.keys():
        imported_dict[k] = h5_import(node, k, memo)
    return imported_dict


def load_dict(parent, name, memo):
    node = parent[name]
    return dict(h5_import(node, k, memo) for k in node.keys())


def load_set(parent, name, memo):
    node = parent[name]
    return set(h5_import(node, k, memo) for k in node.keys())


def load_instance(parent, name, memo):
    # import the class and new it up
    cls = import_symbol(parent[name].attrs[attr_key_instance_cls])
    inst = cls.__new__(cls)

    state = load_unicode_dict(parent, name, memo)
    setstate = getattr(inst, "__setstate__", None)
    if setstate is not None:
        setstate(state)
        return inst
    # we should support slots
    # slotstate = None
    # if isinstance(state, tuple) and len(state) == 2:
    #     state, slotstate = state
    if state:
        inst_dict = inst.__dict__
        if isPy3:
            intern = sys.intern
        for k, v in state.items():
            if isPy3:
                k = intern(k)
            inst_dict[k] = v
    # Again, for future __slot__ support
    # if slotstate:
    #     for k, v in slotstate.items():
    #         setattr(inst, k, v)
    return inst


def load_ndarray(parent, name, _):
    return np.array(parent[name])


def load_none(parent, name, _):
    return None


def load_str(parent, name, _):
    return u(np.array(parent[name]))


def load_bool(parent, name, _):
    return bool(parent[name].attrs[attr_key_bool_value])


def load_number(parent, name, _):
    return np.asscalar(parent[name].attrs[attr_key_number_value])


def load_path(parent, name, _):
    return Path(load_str(parent, name, _))


# ------------------------------ EXPORTS ------------------------------ #

zero_padded = lambda x: "{:0" + u(len(u(x))) + "}"


def save_list(l, parent, name, memo):
    list_node = parent.create_group(name)
    padded = zero_padded(len(l))
    for i, x in enumerate(l):
        h5_export(x, list_node, padded.format(i), memo)


def save_unicode_dict(d, parent, name, memo):
    dict_node = parent.create_group(name)
    if sum(not isinstance(k, strTypes) for k in d.keys()) != 0:
        raise ValueError("Only dictionaries with string keys can be "
                         "serialized")
    for k, v in d.items():
        h5_export(v, dict_node, str(k), memo)


def save_dict(d, parent, name, memo):
    dict_node = parent.create_group(name)
    for k, v in d.items():
        h5_export((k, v), dict_node, str(hash(k)), memo)


def save_set(s, parent, name, memo):
    set_node = parent.create_group(name)
    for x in s:
        h5_export(x, set_node, str(hash(x)), memo)


def get_instance_state(x):
    try:
        state = x.__getstate__()
    except AttributeError:
        state = x.__dict__
    return state


def instance_is_hdf5able(x):
    return not ((hasattr(x, '__slots__') and not hasattr(x, '__getstate__')) or
                hasattr(x, '__getnewargs__') or
                hasattr(x, '__getnewargs_ex__') or
                hasattr(x, '__getinitargs__') or
                x.__class__.__reduce__ != object.__reduce__ or
                x.__class__.__reduce_ex__ != object.__reduce_ex__)


def save_instance(instance, parent, name, memo):
    if not instance_is_hdf5able(instance):
        raise ValueError('instance {} cannot be saved as it '
                         'implements unsupported parts of the pickle protocol')
    d = get_instance_state(instance)
    save_unicode_dict(d, parent, name, memo)
    # unicode dict added itself to the parent. Grab the node
    node = parent[name]
    # And set the attribute so it can be decoded.
    node.attrs[attr_key_instance_cls] = str_of_cls(instance.__class__)


def save_ndarray(a, parent, name, _):
    # fletcher32 is a checksum, gzip compression is supported by Matlab
    parent.create_dataset(name, data=a, compression='gzip', fletcher32=True)


def save_none(none, parent, name, _):
    parent.create_group(name)  # A blank group


def save_str(s, parent, name, _):
    parent.create_dataset(name, data=s)


def save_bool(a_bool, parent, name, _):
    group = parent.create_group(name)  # A blank group
    group.attrs[attr_key_bool_value] = a_bool


def save_number(a_number, parent, name, _):
    group = parent.create_group(name)  # A blank group
    group.attrs[attr_key_number_value] = a_number


def save_path(path, parent, name, _):
    parent.create_dataset(name, data=u(path))


str_of_cls = lambda x: "{}.{}".format(x.__module__, x.__name__)


def import_symbol(name):
    symbol_name = name.split('.')[-1]
    module_name = '.'.join(name.split('.')[:-1])
    m = importlib.import_module(module_name)
    return m.__getattribute__(symbol_name)


attr_key_type = u'type'
attr_key_instance_cls = u'cls'
attr_key_number_value = u'number_value'
attr_key_bool_value = u'bool_value'


top_level_key = 'hdf5able'
T = namedtuple('T', ["type", "str", "importer", "exporter"])


types = [T(list, "list", load_list, save_list),
         T(tuple, "tuple", load_tuple, save_list),  # export is as list
         T(dict, "dict", load_dict, save_dict),
         T(set, "set", load_set, save_set),
         T(np.ndarray, "ndarray", load_ndarray, save_ndarray),
         T(type(None), "NoneType", load_none, save_none),
         T(strTypes, "unicode", load_str, save_str),
         T(bool, "bool", load_bool, save_bool),
         T(PosixPath, "pathlib.PosixPath", load_path, save_path),
         T(numberTypes, "Number", load_number, save_number)]

type_to_exporter = dict()
str_to_importer = dict()
type_to_str = dict()

for t in types:
    # tuples of types are allowed, just add each type in turn with the same
    # rule
    if isinstance(t.type, tuple):
        for t_i in t.type:
            type_to_exporter[t_i] = t.exporter
            type_to_str[t_i] = t.str
    else:
        type_to_exporter[t.type] = t.exporter
        type_to_str[t.type] = t.str
    str_to_importer[t.str] = t.importer
    type_to_str[t.type] = t.str

str_to_importer['instance'] = load_instance


def link_path_if_softlink(node, name):
    if node.get(name, getclass=True, getlink=True) == h5py.SoftLink:
        # this node is a softlink - grab it's path
        return node.get(name, getlink=True).path


def h5_import(parent, name, memo):
    link_path = link_path_if_softlink(parent, name)
    if link_path is not None:
        # this node is a softlink - memoize the link destination path
        memo_path = link_path
    else:
        memo_path = parent[name].name
    if memo_path in memo:
        # this object has already been loaded - just return it
        return memo[memo_path]
    node = parent[name]
    Type = node.attrs.get(attr_key_type)
    if Type is not None:
        # node type is specific
        importer = str_to_importer.get(Type)
        if importer is not None:
            obj = importer(parent, name, memo)
            # remember we imported this already
            memo[memo_path] = obj
            return obj
        else:
            raise ValueError("Don't know how to import type {}"
                             " for node {}".format(Type, node))
    else:
        raise ValueError("Cannot find Type attribute on {}".format(node))


def h5_export(x, parent, name, memo):
    if id(x) in memo:
        # this object is already exported, just softlink to it.
        parent[name] = h5py.SoftLink(memo[id(x)].name)
        return
    type_x = type(x)
    exporter = type_to_exporter.get(type_x)
    if exporter is None:
        # hmm hopefully it's an object instance, otherwise we will be unable
        # to proceed
        if isinstance(x, object):
            exporter = save_instance
            type_str = 'instance'
        else:
            raise ValueError("Cannot export {} named "
                             "'{}' of type {}".format(x, name, type(x)))
    else:
        type_str = type_to_str.get(type_x)
    # definitely have type_str and exporter
    exporter(x, parent, name, memo)
    new_node = parent[name]
    new_node.attrs[attr_key_type] = type_str
    # remember we have exported this object
    memo[id(x)] = new_node
    # make sure that object doesn't die, as otherwise future objects
    # might reuse the same id. We steal the standard lib approach, and
    # just append on a special list in the memo (the list is stored on
    # the hash of the memo, so we are pretty guaranteed that no one
    # will use it!)
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # I'm the first!
        memo[id(memo)] = [x]


def save(path, x):
    with h5py.File(path, "w") as f:
        h5_export(x, f, top_level_key, {})


def load(path):
    with h5py.File(path, "r") as f:
        return h5_import(f, top_level_key, {})
