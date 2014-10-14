from __future__ import unicode_literals

import os
from collections import namedtuple
from pathlib import PosixPath, WindowsPath, PurePosixPath, PureWindowsPath
import numpy as np
import h5py

from .stdpickle import (H5itPicklingError, H5itUnpicklingError,
                        is_py2, is_py3,
                        r_key_func, r_key_cls, r_key_args,
                        r_key_state, r_key_listitems, r_key_dictitems,
                        pickle_load_global, pickle_save_global, GlobalTuple,
                        pickle_load_build, pickle_save)

if is_py2:
    from types import ClassType, FunctionType, BuiltinFunctionType, TypeType
    strType = unicode
    py2_bytesType = str
    py3_bytesType = tuple()  # will never encounter Py3 bytes str on Py2
    as_unicode_str = unicode

    numberTypes = int, long, float, complex
    globalTypes = ClassType, FunctionType, BuiltinFunctionType, TypeType
elif is_py3:
    from types import FunctionType
    strType = str
    py2_bytesType = tuple()  # will never encounter Py2 bytes str on Py3
    py3_bytesType = bytes
    as_unicode_str = str

    numberTypes = int, float, complex
    globalTypes = FunctionType
else:
    raise Exception('h5it is only compatible with Python 2 or Python 3')

host_is_posix = os.name == 'posix'
host_is_windows = os.name == 'nt'

attr_key_type = 'type'
attr_key_type_reduction = 'reduction'

attr_key_reduction_cls_module = 'cls_module'
attr_key_reduction_cls_name = 'cls_name'
attr_key_reduction_func_module = 'func_module'
attr_key_reduction_func_name = 'func_name'

attr_key_global_module = 'module'
attr_key_global_name = 'name'

attr_key_instance_has_custom_getstate = 'instance_has_custom_getstate'
attr_key_number_value = 'number_value'
attr_key_bool_value = 'bool_value'

top_level_group_namespace = 'h5it'


# ------------------------------ IMPORTS ------------------------------ #

ListItem = namedtuple('ListItem', ['i', 'item'])


def load_list(parent, name, memo, encoding):
    node = parent[name]
    unordered = []
    for i, j in enumerate(node.keys()):
        unordered.append(ListItem(int(j), h5_import(node, j, memo, encoding)))
    ordered = sorted(unordered)
    counts = [x.i for x in ordered]
    items = [x.item for x in ordered]
    if counts != list(range(len(counts))):
        raise H5itUnpicklingError("Attempted to import a list "
                                  "that is missing elements")
    return items


def load_tuple(parent, name, memo, encoding):
    return tuple(load_list(parent, name, memo, encoding))


def load_unicode_dict(parent, name, memo, encoding):
    node = parent[name]
    imported_dict = {}
    for k in node.keys():
        imported_dict[k] = h5_import(node, k, memo, encoding)
    return imported_dict


def load_dict(parent, name, memo, encoding):
    node = parent[name]
    return dict(h5_import(node, k, memo, encoding) for k in node.keys())


def load_reducible(parent, name, memo, encoding):
    node = parent[name]
    # import the class and new it up
    if attr_key_global_name in node.attrs:
        # reduction actually saved out a global
        return load_global(parent, name, memo, encoding)

    # if not, we are loading with NEWOBJ or REDUCE
    args = load_tuple(node, r_key_args, memo, encoding)
    if attr_key_reduction_cls_module in node.attrs:
        # reduction using the NEWOBJ protocol
        cls_module = node.attrs[attr_key_reduction_cls_module]
        cls_name = node.attrs[attr_key_reduction_cls_name]
        cls = pickle_load_global(cls_module, cls_name)
        obj = cls.__new__(cls, *args)
    elif attr_key_reduction_func_module in node.attrs:
        # reduction using the REDUCE protocol
        func_module = node.attrs[attr_key_reduction_func_module]
        func_name = node.attrs[attr_key_reduction_func_name]
        func = pickle_load_global(func_module, func_name)
        obj = func(*args)
    else:
        raise H5itUnpicklingError(
            "error loading reduction - can't find {} or {} in attrs".format(
            attr_key_reduction_cls_module, attr_key_reduction_func_module))

    if r_key_state in node:
        state = h5_import(node, r_key_state, memo, encoding)
        pickle_load_build(obj, state)

    if r_key_listitems in node:
        listitems = h5_import(node, r_key_listitems, memo, encoding)
        for i in listitems:
            obj.append(i)

    if r_key_dictitems in node:
        iteritems = h5_import(node, r_key_dictitems, memo, encoding)
        for k, v in iteritems:
            obj[k] = v

    return obj


def load_global(parent, name, memo, encoding):
    module = parent[name].attrs[attr_key_global_module]
    m_name = parent[name].attrs[attr_key_global_name]
    return pickle_load_global(module, m_name)


def load_ndarray(parent, name, memo, encoding):
    return np.array(parent[name])


def load_none(parent, name, memo, encoding):
    return None


def load_str(parent, name, memo, encoding):
    return parent[name].value


def load_bytes(parent, name, memo, encoding):
    return parent[name].value


def load_py2_bytes_on_py3(parent, name, memo, encoding):
    if encoding == 'ASCII':
        return parent[name].value.decode('ASCII')
    elif encoding == 'bytes':
        return parent[name].value
    else:
        raise H5itUnpicklingError("The only valid encodings are 'ASCII' or "
                                  "'bytes'")

if is_py3:
    load_py2_bytes = load_py2_bytes_on_py3
else:
    load_py2_bytes = load_bytes


def load_bool(parent, name, memo, encoding):
    return bool(parent[name].attrs[attr_key_bool_value])


def load_number(parent, name, memo, encoding):
    return np.asscalar(parent[name].attrs[attr_key_number_value])


def load_posix_path(parent, name, memo, encoding):
    str_path = load_str(parent, name, memo, encoding)
    if host_is_posix:
        return PosixPath(str_path)
    else:
        return PurePosixPath(str_path)


def load_windows_path(parent, name, memo, encoding):
    str_path = load_str(parent, name, memo, encoding)
    if host_is_windows:
        return WindowsPath(str_path)
    else:
        return PureWindowsPath(str_path)


# ------------------------------ EXPORTS ------------------------------ #

zero_padded = lambda x: "{:0" + as_unicode_str(len(as_unicode_str(x))) + "}"


def save_list(l, parent, name, memo):
    list_node = parent.create_group(name)
    padded = zero_padded(len(l))
    for i, x in enumerate(l):
        h5_export(x, list_node, padded.format(i), memo)


is_string_keyed_dict = lambda d: (sum(not isinstance(k, strType)
                                      for k in d.keys()) == 0)


def save_unicode_dict(d, parent, name, memo):
    dict_node = parent.create_group(name)
    if not is_string_keyed_dict(d):
        raise ValueError("Only dictionaries with string keys can be "
                         "serialized")
    for k, v in d.items():
        h5_export(v, dict_node, str(k), memo)


def save_dict(d, parent, name, memo):
    dict_node = parent.create_group(name)
    for k, v in d.items():
        h5_export((k, v), dict_node, str(hash(k)), memo)


def save_reducible(x, parent, name, memo):
    # save down the object: we'll either get back a global or a reduction state
    reduction = pickle_save(x)
    if type(reduction) == GlobalTuple:
        save_global(reduction, parent, name, memo)
        return

    # Reduction is a dict that is ready to directly be saved. Let's make a
    # group
    node = parent.create_group(name)

    if r_key_cls in reduction:
        cls_module, cls_name = reduction[r_key_cls]
        node.attrs[attr_key_reduction_cls_module] = cls_module
        node.attrs[attr_key_reduction_cls_name] = cls_name
    elif r_key_func in reduction:
        func_module, func_name = reduction[r_key_func]
        node.attrs[attr_key_reduction_func_module] = func_module
        node.attrs[attr_key_reduction_func_name] = func_name
    else:
        H5itPicklingError("reduction state is missing a 'func' or 'cls'")

    # save out the reduction args
    save_list(reduction[r_key_args], node, r_key_args, memo)

    if r_key_state in reduction:
        h5_export(reduction[r_key_state], node, r_key_state, memo)

    if r_key_listitems in reduction:
        h5_export(reduction[r_key_listitems], node, r_key_listitems, memo)

    if r_key_dictitems in reduction:
        h5_export(reduction[r_key_dictitems], node, r_key_dictitems, memo)


def save_global(g, parent, name, _):
    node = parent.create_group(name)  # A blank group
    module, g_name = pickle_save_global(g)
    node.attrs[attr_key_global_module] = module
    node.attrs[attr_key_global_name] = g_name


def save_ndarray(a, parent, name, _):
    # fletcher32 is a checksum, gzip compression is supported by Matlab
    parent.create_dataset(name, data=a, compression='gzip', fletcher32=True)


def save_none(none, parent, name, _):
    parent.create_group(name)  # A blank group


def save_str(s, parent, name, _):
    dt = h5py.special_dtype(vlen=as_unicode_str)
    parent.create_dataset(name, data=s, dtype=dt)


def save_bytes(s, parent, name, _):
    parent.create_dataset(name, data=s)


def save_bool(a_bool, parent, name, _):
    group = parent.create_group(name)  # A blank group
    group.attrs[attr_key_bool_value] = a_bool


def save_number(a_number, parent, name, _):
    group = parent.create_group(name)  # A blank group
    group.attrs[attr_key_number_value] = a_number


def save_path(path, parent, name, _):
    parent.create_dataset(name, data=as_unicode_str(path))


T = namedtuple('T', ["type", "str", "importer", "exporter"])


types = [T(list, "list", load_list, save_list),
         T(tuple, "tuple", load_tuple, save_list),  # export is as list
         T(dict, "dict", load_dict, save_dict),
         T(np.ndarray, "ndarray", load_ndarray, save_ndarray),
         T(type(None), "NoneType", load_none, save_none),
         T(strType, "str", load_str, save_str),
         T(py2_bytesType, "py2_bytes", load_py2_bytes, save_bytes),
         T(py3_bytesType, "bytes", load_bytes, save_bytes),
         T(bool, "bool", load_bool, save_bool),
         T(globalTypes, "global", load_global, save_global),
         T(numberTypes, "Number", load_number, save_number),
         T((PosixPath, PurePosixPath), "pathlib.PosixPath",
           load_posix_path, save_path),
         T((WindowsPath, PureWindowsPath), "pathlib.WindowsPath",
           load_windows_path, save_path)]

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

# add on the reduction importer
str_to_importer[attr_key_type_reduction] = load_reducible


def link_path_if_softlink(node, name):
    if node.get(name, getclass=True, getlink=True) == h5py.SoftLink:
        # this node is a softlink - grab it's path
        return node.get(name, getlink=True).path


def h5_import(parent, name, memo, encoding):
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
    type_ = node.attrs.get(attr_key_type)
    if type_ is not None:
        # node type is specific
        importer = str_to_importer.get(type_)
        if importer is not None:
            obj = importer(parent, name, memo, encoding)
            # remember we imported this already
            memo[memo_path] = obj
            return obj
        else:
            raise H5itUnpicklingError(
                "Don't know how to import type "
                "{} for node {}".format(type_, node))
    else:
        raise H5itUnpicklingError("Cannot find {} "
                                  "attribute on {}".format(attr_key_type,
                                                           node))


def h5_export(x, parent, name, memo):
    if id(x) in memo:
        # this object is already exported, just softlink to it.
        parent[name] = h5py.SoftLink(memo[id(x)].name)
        return
    type_x = type(x)
    exporter = type_to_exporter.get(type_x)
    if exporter is None:
        # use the pickle protocol to save a reducible/global
        exporter = save_reducible
        type_str = attr_key_type_reduction
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


def norm_path(path):
    r"""
    Uses all the tricks in the book to expand a path out to an absolute one.
    """
    return os.path.abspath(os.path.normpath(
        os.path.expandvars(os.path.expanduser(as_unicode_str(path)))))


def dump(x, path):
    with h5py.File(norm_path(path), "w") as f:
        h5_export(x, f, top_level_group_namespace, {})


def load_py2(path):
    with h5py.File(norm_path(path), "r") as f:
        # encoding is not used on Python 2, set to a dummy string
        return h5_import(f, top_level_group_namespace, {}, '')


def load_py3(path, encoding='ASCII'):
    if encoding not in ['ASCII', 'bytes']:
        raise H5itUnpicklingError("The only valid encodings are 'ASCII' or "
                                  "'bytes'")
    with h5py.File(norm_path(path), "r") as f:
        return h5_import(f, top_level_group_namespace, {}, encoding)

if is_py3:
    load = load_py3
else:
    load = load_py2
