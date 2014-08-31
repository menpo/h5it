import abc
from collections import namedtuple, OrderedDict
from enum import Enum
from pathlib import Path

import numpy as np
from numpy import ndarray
import h5py


# IMPORT

ListItem = namedtuple('ListItem', ['i', 'item'])


def h_import_list(node):
    unordered = []
    for i, (j, x) in enumerate(node.iteritems()):
        unordered.append(ListItem(int(j), h_import(x)))
    ordered = sorted(unordered)
    counts = [x.i for x in ordered]
    items = [x.item for x in ordered]
    if counts != range(len(counts)):
        raise ValueError("Attempted to import a list that is missing elements")
    return items


def h_import_dict(node):
    imported_dict = {}
    for k, v in node.iteritems():
        imported_dict[k] = h_import(v)
    return imported_dict


def h_import_hdf5able(node):
    cls = str_to_cls[node.attrs[AttrKey.hdf5able_cls]]
    return cls.h5_rebuild(cls.h5_import_to_dict(node))


def h_import_ndarray(node):
    return np.array(node)


def h_import_none(_):
    return None


def h_import_str(node):
    return str(np.array(node))


def h_import_bool(node):
    return node.attrs[AttrKey.bool_value]


def h_import_number(node):
    return node.attrs[AttrKey.number_value]


def h_import_path(node):
    return Path(str(np.array(node)))


# EXPORT

zero_padded = lambda x: "{:0" + str(len(str(x))) + "}"


def h_export_list(parent, l, name):
    list_node = parent.create_group(name)
    padded = zero_padded(len(l))
    for i, x in enumerate(l):
        h_export(list_node, x, padded.format(i))


def h_export_dict(parent, d, name):
    dict_node = parent.create_group(name)
    if sum(not isinstance(k, (str, unicode)) for k in d.keys()) != 0:
        raise ValueError("Only dictionaries with string keys can be "
                         "serialized")
    for k, v in d.iteritems():
        h_export(dict_node, v, str(k))


def h_export_hdf5able(parent, hdf5able, name):
    # Objects behave a lot like dictionaries
    hdf5able.h5_export_to_dict(parent, name)
    # HDF5able added itself to the parent. Grab the node
    node = parent[name]
    # And set the attribute so it can be decoded.
    node.attrs[AttrKey.hdf5able_cls] = str_of_cls(hdf5able.__class__)


def h_export_ndarray(parent, a, name):
    # fletcher32 is a checksum, lzf is fast OK compression
    parent.create_dataset(name, data=a, compression='lzf', fletcher32=True)


def h_export_none(parent, _, name):
    parent.create_group(name)  # A blank group


def h_export_str(parent, s, name):
    parent.create_dataset(name, data=s)


def h_export_bool(parent, a_bool, name):
    group = parent.create_group(name)  # A blank group
    group.attrs[AttrKey.bool_value] = a_bool


def h_export_number(parent, a_number, name):
    group = parent.create_group(name)  # A blank group
    group.attrs[AttrKey.number_value] = a_number


def h_export_path(parent, path, name):
    parent.create_dataset(name, data=str(path))


str_of_cls = lambda x: "{}.{}".format(x.__module__, x.__name__)

str_to_cls = {}
cls_to_str = {}


def h5_register_cls(cls):
    cls_str = str_of_cls(cls)
    print("Registering {}: {}".format(cls_str, cls))
    str_to_cls[cls_str] = cls
    cls_to_str[cls] = cls_str


class HDF5Meta(type):

    def __new__(cls, name, bases, attrs):
        new_cls = super(HDF5Meta, cls).__new__(cls, name, bases, attrs)
        h5_register_cls(new_cls)
        return new_cls


class HDF5ABCMeta(abc.ABCMeta):

    def __new__(cls, name, bases, attrs):
        new_cls = super(HDF5ABCMeta, cls).__new__(cls, name, bases, attrs)
        h5_register_cls(new_cls)
        return new_cls


class HDF5able(object):

    __metaclass__ = HDF5ABCMeta

    @classmethod
    def h5_rebuild(cls, d):
        # by default, __new__ cls -> set dict.
        instance = cls.__new__(cls)
        instance.__dict__ = d
        return instance

    @classmethod
    def h5_import_to_dict(cls, node):
        # default behavior - import this node as a dict
        return h_import_dict(node)

    def h5_export_to_dict(self, parent, name):
        h_export_dict(parent, self.__dict__, name)


class AttrKey(Enum):
    type = 'type'
    hdf5able_cls = 'cls'
    bool_value = 'bool_value'
    number_value = 'number_value'


top_level_key = 'hdf5able'
T = namedtuple('T', ["type", "str", "importer", "exporter"])

from numbers import Number

types = [T(list, "list", h_import_list, h_export_list),
         T(dict, "dict", h_import_dict, h_export_dict),
         T(HDF5able, "HDF5able", h_import_hdf5able, h_export_hdf5able),
         T(ndarray, "ndarray", h_import_ndarray, h_export_ndarray),
         T(type(None), "NoneType", h_import_none, h_export_none),
         T(str, "str", h_import_str, h_export_str),
         T(bool, "bool", h_import_bool, h_export_bool),
         T(Path, "pathlib.Path", h_import_path, h_export_path),
         T(Number, "Number", h_import_number, h_export_number)]

type_to_exporter = OrderedDict()
str_to_importer = OrderedDict()
type_to_str = OrderedDict()

for t in types:
    type_to_exporter[t.type] = t.exporter
    str_to_importer[t.str] = t.importer
    type_to_str[t.type] = t.str


def h5export(path, x):
    with h5py.File(path, "w") as f:
        h_export(f, x, top_level_key)


def h5import(path):
    with h5py.File(path, "r") as f:
        return h_import(f[top_level_key])


def h_import(node):
    Type = node.attrs.get(AttrKey.type)
    if Type is not None:
        # node type is specific
        importer = str_to_importer.get(Type)
        if importer is not None:
            return importer(node)
        else:
            print("Don't know how to import type {}".format(Type))


def h_export(parent, x, name):
    for Type, exporter in type_to_exporter.iteritems():
        if isinstance(x, Type):
            exporter(parent, x, name)
            new_node = parent[name]
            new_node.attrs[AttrKey.type] = type_to_str[Type]
            return
    print("Cannot find exporter for {} named {} of type {}".format(
          name, x, type(x)))

