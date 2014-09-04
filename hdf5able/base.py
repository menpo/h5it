from collections import namedtuple, OrderedDict
from numbers import Number
import importlib
from pathlib import Path
import sys

import numpy as np
import h5py

from .callable import (serialize_callable, serialize_callable_and_test,
                       deserialize_callable)


isPython2 = sys.version_info.major == 2

if isPython2:
    strTypes = (str, unicode)
else:
    strTypes = (str, bytes)


def u(s):
    if isPython2:
        return unicode(s)
    else:
        return str(s)

# ------------------------------ IMPORTS ------------------------------ #

ListItem = namedtuple('ListItem', ['i', 'item'])


def import_list(node):
    unordered = []
    for i, (j, x) in enumerate(node.items()):
        unordered.append(ListItem(int(j), h5_import(x)))
    ordered = sorted(unordered)
    counts = [x.i for x in ordered]
    items = [x.item for x in ordered]
    if counts != range(len(counts)):
        raise ValueError("Attempted to import a list that is missing elements")
    return items


def import_tuple(node):
    return tuple(import_list(node))


def import_dict(node):
    imported_dict = {}
    for k, v in node.items():
        imported_dict[k] = h5_import(v)
    return imported_dict


def h_import_hdf5able(node):
    cls = import_hdf5able(node.attrs[attr_key_hdf5able_cls])
    return cls.h5_rebuild_from_dict(cls.h5_import_as_dict(node))


def import_ndarray(node):
    return np.array(node)


def import_none(_):
    return None


def import_str(node):
    return u(np.array(node))


def import_bool(node):
    return bool(node.attrs[attr_key_bool_value])


def import_number(node):
    return np.asscalar(node.attrs[attr_key_number_value])


def import_path(node):
    return Path(import_str(node))


# ------------------------------ EXPORTS ------------------------------ #

zero_padded = lambda x: "{:0" + u(len(u(x))) + "}"


def export_list(parent, l, name):
    list_node = parent.create_group(name)
    padded = zero_padded(len(l))
    for i, x in enumerate(l):
        h5_export(list_node, x, padded.format(i))


def export_dict(parent, d, name):
    dict_node = parent.create_group(name)
    if sum(not isinstance(k, strTypes) for k in d.keys()) != 0:
        raise ValueError("Only dictionaries with string keys can be "
                         "serialized")
    for k, v in d.items():
        h5_export(dict_node, v, str(k))


def export_hdf5able(parent, hdf5able, name):
    # Objects behave a lot like dictionaries
    hdf5able.h5_export_as_dict(parent, name)
    # HDF5able added itself to the parent. Grab the node
    node = parent[name]
    # And set the attribute so it can be decoded.
    node.attrs[attr_key_hdf5able_cls] = str_of_cls(hdf5able.__class__)


def export_ndarray(parent, a, name):
    # fletcher32 is a checksum, lzf is fast OK compression
    parent.create_dataset(name, data=a, compression='lzf', fletcher32=True)


def export_none(parent, _, name):
    parent.create_group(name)  # A blank group


def export_str(parent, s, name):
    parent.create_dataset(name, data=s)


def export_bool(parent, a_bool, name):
    group = parent.create_group(name)  # A blank group
    group.attrs[attr_key_bool_value] = a_bool


def export_number(parent, a_number, name):
    group = parent.create_group(name)  # A blank group
    group.attrs[attr_key_number_value] = a_number


def export_path(parent, path, name):
    parent.create_dataset(name, data=u(path))


str_of_cls = lambda x: "{}.{}".format(x.__module__, x.__name__)


def import_hdf5able(name):
    callable_name = name.split('.')[-1]
    module_name = '.'.join(name.split('.')[:-1])
    m = importlib.import_module(module_name)
    return m.__getattribute__(callable_name)


class HDF5able(object):

    @classmethod
    def h5_rebuild_from_dict(cls, d):
        # by default, __new__ cls -> set dict.
        instance = cls.__new__(cls)
        instance.__dict__ = d
        return instance

    def h5_dict_representation(self):
        return self.__dict__

    @classmethod
    def h5_import_as_dict(cls, node):
        # default behavior - import this node as a dict
        return import_dict(node)

    def h5_export_as_dict(self, parent, name):
        export_dict(parent, self.h5_dict_representation(), name)


class SerializableCallable(HDF5able):

    def __init__(self, f, modules):
        self.f = f
        self.modules = modules

    def h5_dict_representation(self):
        serialized_c = serialize_callable_and_test(self.f, self.modules)
        return serialized_c._asdict()

    @classmethod
    def h5_rebuild_from_dict(cls, d):
        # just return directly the function
        return deserialize_callable(**d)


class TestedSerializableCallable(SerializableCallable):

    def __init__(self, f, modules, testargs=None, testkwargs=None):
        SerializableCallable.__init__(self, f, modules)
        self.testargs = testargs
        self.testkwargs = testkwargs

    def h5_dict_representation(self):
        serialized_c = serialize_callable_and_test(self.f,
                                                   self.modules,
                                                   args=self.testargs,
                                                   kwargs=self.testkwargs)
        return serialized_c._asdict()


attr_key_type = u'type'
attr_key_hdf5able_cls = u'cls'
attr_key_number_value = u'number_value'
attr_key_bool_value = u'bool_value'


top_level_key = 'hdf5able'
T = namedtuple('T', ["type", "str", "importer", "exporter"])


types = [T(list, "list", import_list, export_list),
         T(tuple, "tuple", import_tuple, export_list),  # export is as list
         T(dict, "dict", import_dict, export_dict),
         T(HDF5able, "HDF5able", h_import_hdf5able, export_hdf5able),
         T(np.ndarray, "ndarray", import_ndarray, export_ndarray),
         T(type(None), "NoneType", import_none, export_none),
         T(strTypes, "unicode", import_str, export_str),
         T(bool, "bool", import_bool, export_bool),
         T(Path, "pathlib.Path", import_path, export_path),
         T(Number, "Number", import_number, export_number)]

type_to_exporter = OrderedDict()
str_to_importer = OrderedDict()
type_to_str = OrderedDict()

for t in types:
    type_to_exporter[t.type] = t.exporter
    str_to_importer[t.str] = t.importer
    type_to_str[t.type] = t.str


def h5_import(node):
    Type = node.attrs.get(attr_key_type)
    if Type is not None:
        # node type is specific
        importer = str_to_importer.get(Type)
        if importer is not None:
            return importer(node)
        else:
            print("Don't know how to import type {}".format(Type))


def h5_export(parent, x, name):
    for Type, exporter in type_to_exporter.items():
        if isinstance(x, Type):
            exporter(parent, x, name)
            new_node = parent[name]
            new_node.attrs[attr_key_type] = type_to_str[Type]
            return
    raise ValueError("Cannot export {} named "
                     "'{}' of type {}".format(x, name, type(x)))


def save(path, x):
    with h5py.File(path, "w") as f:
        h5_export(f, x, top_level_key)


def load(path):
    with h5py.File(path, "r") as f:
        return h5_import(f[top_level_key])
