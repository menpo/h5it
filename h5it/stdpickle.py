# the module includes adapted parts of the Python 2 and Python 3 standard
# libraries. It is intended to provide a clean implementation of the pickle
# protocol minus the actual file format on disk. This module contains no
# code specific to HDF5 files.
import sys
from collections import namedtuple
from pickle import (whichmodule, PicklingError, UnpicklingError,
                    dispatch_table, _extension_registry)


class H5itPicklingError(PicklingError):
    pass


class H5itUnpicklingError(UnpicklingError):
    pass


GlobalTuple = namedtuple(u'Global', (u'module', u'name'))


r_key_func = u'func'
r_key_cls = u'cls'
r_key_args = u'args'
r_key_state = u'state'
r_key_listitems = u'listitems'
r_key_dictitems = u'dictitems'


is_py2 = sys.version_info.major == 2
is_py3 = sys.version_info.major == 3


if is_py2:
    from types import TypeType, StringType, TupleType

if is_py3:
    from pickle import _getattribute, _compat_pickle


# adapted from Python 3 save_global
def save_global_py3(obj, name=None, proto=2, fix_imports=True):

    if name is None and proto >= 4:
        name = getattr(obj, '__qualname__', None)
    if name is None:
        name = obj.__name__

    module_name = whichmodule(obj, name, allow_qualname=proto >= 4)
    try:
        __import__(module_name, level=0)
        module = sys.modules[module_name]
        obj2 = _getattribute(module, name, allow_qualname=proto >= 4)
    except (ImportError, KeyError, AttributeError):
        raise PicklingError(
            "Can't pickle %r: it's not found as %s.%s" %
            (obj, module_name, name))
    else:
        if obj2 is not obj:
            raise PicklingError(
                "Can't pickle %r: it's not the same object as %s.%s" %
                (obj, module_name, name))

    if proto >= 2:
        code = _extension_registry.get((module_name, name))
        if code:
            # assert code > 0
            # if code <= 0xff:
            #     write(EXT1 + pack("<B", code))
            # elif code <= 0xffff:
            #     write(EXT2 + pack("<H", code))
            # else:
            #     write(EXT4 + pack("<i", code))
            # return
            raise H5itPicklingError("h5it Can't pickle %r: extension codes are not"
                            " supported yet." % obj)
    # Non-ASCII identifiers are supported only with protocols >= 3.
    if proto >= 4:
        # self.save(module_name)
        # self.save(name)
        # write(STACK_GLOBAL)
        raise H5itPicklingError("h5it Can't pickle %r: protocol %i is not "
                        "supported yet." % (obj, proto))
    elif proto >= 3:
        # write(GLOBAL + bytes(module_name, "utf-8") + b'\n' +
        #       bytes(name, "utf-8") + b'\n')
        raise H5itPicklingError("h5it Can't pickle %r: protocol %i is not "
                        "supported yet." % (obj, proto))
    else:
        if fix_imports:
            r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
            r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
            if (module_name, name) in r_name_mapping:
                module_name, name = r_name_mapping[(module_name, name)]
            if module_name in r_import_mapping:
                module_name = r_import_mapping[module_name]
        try:
            return GlobalTuple(bytes(module_name, "ascii"),
                               bytes(name, "ascii"))
        except UnicodeEncodeError:
            raise PicklingError(
                "can't pickle global identifier '%s.%s' using "
                "pickle protocol %i" % (module, name, proto))


# adapted from Python 2 save_global
def save_global_py2(obj, name=None, proto=2):

    if name is None:
        name = obj.__name__

    module = getattr(obj, "__module__", None)
    if module is None:
        module = whichmodule(obj, name)

    try:
        __import__(module)
        mod = sys.modules[module]
        klass = getattr(mod, name)
    except (ImportError, KeyError, AttributeError):
        raise PicklingError(
            "Can't pickle %r: it's not found as %s.%s" %
            (obj, module, name))
    else:
        if klass is not obj:
            raise PicklingError(
                "Can't pickle %r: it's not the same object as %s.%s" %
                (obj, module, name))

    if proto >= 2:
        code = _extension_registry.get((module, name))
        if code:
            # assert code > 0
            # if code <= 0xff:
            #     write(EXT1 + chr(code))
            # elif code <= 0xffff:
            #     write("%c%c%c" % (EXT2, code&0xff, code>>8))
            # else:
            #     write(EXT4 + pack("<i", code))
            # return
            raise H5itPicklingError(
                "h5it Can't pickle %r: extension codes "
                "are not supported yet." % obj)
    return GlobalTuple(module, name)


def save_py3(obj, proto=2):
    t = type(obj)

    # Check copyreg.dispatch_table only.
    reduce = dispatch_table.get(t)
    if reduce is not None:
        rv = reduce(obj)
    else:
        # Check for a class with a custom metaclass; treat as regular class
        try:
            issc = issubclass(t, type)
        except TypeError: # t is not a class (old Boost; see SF #502085)
            issc = False
        if issc:
            return save_global_py3(obj)

        # Check for a __reduce_ex__ method, fall back to __reduce__
        reduce = getattr(obj, "__reduce_ex__", None)
        if reduce is not None:
            rv = reduce(proto)
        else:
            reduce = getattr(obj, "__reduce__", None)
            if reduce is not None:
                rv = reduce()
            else:
                raise PicklingError("Can't pickle %r object: %r" %
                                    (t.__name__, obj))

    # Check for string returned by reduce(), meaning "save as global"
    if isinstance(rv, str):
        return save_global_py3(obj, rv)

    # Assert that reduce() returned a tuple
    if not isinstance(rv, tuple):
        raise PicklingError("%s must return string or tuple" % reduce)

    # Assert that it returned an appropriately sized tuple
    l = len(rv)
    if not (2 <= l <= 5):
        raise PicklingError("Tuple returned by %s must have "
                            "two to five elements" % reduce)

    # return the reduce() output for serializing
    return save_reduce_py3(obj=obj, *rv)


def save_py2(obj, proto=2):
    t = type(obj)

    # Check copy_reg.dispatch_table
    reduce = dispatch_table.get(t)
    if reduce:
        rv = reduce(obj)
    else:
        # Check for a class with a custom metaclass; treat as regular class
        try:
            issc = issubclass(t, TypeType)
        except TypeError: # t is not a class (old Boost; see SF #502085)
            issc = 0
        if issc:
            return save_global_py2(obj, proto=proto)

        # Check for a __reduce_ex__ method, fall back to __reduce__
        reduce = getattr(obj, "__reduce_ex__", None)
        if reduce:
            rv = reduce(proto)
        else:
            reduce = getattr(obj, "__reduce__", None)
            if reduce:
                rv = reduce()
            else:
                raise PicklingError("Can't pickle %r object: %r" %
                                    (t.__name__, obj))

    # Check for string returned by reduce(), meaning "save as global"
    if type(rv) is StringType:
        return save_global_py2(obj, rv)

    # Assert that reduce() returned a tuple
    if type(rv) is not TupleType:
        raise PicklingError("%s must return string or tuple" % reduce)

    # Assert that it returned an appropriately sized tuple
    l = len(rv)
    if not (2 <= l <= 5):
        raise PicklingError("Tuple returned by %s must have "
                            "two to five elements" % reduce)

    # Return the reduce() output for serialization
    return save_reduce_py2(obj=obj, *rv)


def save_reduce_py3(func, args, state=None, listitems=None, dictitems=None,
                    obj=None, proto=2):
    reduced = {}

    if not isinstance(args, tuple):
        raise PicklingError("args from save_reduce() must be a tuple")
    if not callable(func):
        raise PicklingError("func from save_reduce() must be callable")

    func_name = getattr(func, "__name__", "")
    if proto >= 4 and func_name == "__newobj_ex__":
        # cls, args, kwargs = args
        # if not hasattr(cls, "__new__"):
        #     raise PicklingError("args[0] from {} args has no __new__"
        #                         .format(func_name))
        # if obj is not None and cls is not obj.__class__:
        #     raise PicklingError("args[0] from {} args has the wrong class"
        #                         .format(func_name))
        # save(cls)
        # save(args)
        # save(kwargs)
        # write(NEWOBJ_EX)
        raise H5itPicklingError("h5it can't reduce {} {}: __newobj_ex__ is not "
                        "supported.".format(obj, func_name))
    elif proto >= 2 and func_name == "__newobj__":
        # A __reduce__ implementation can direct protocol 2 or newer to
        # use the more efficient NEWOBJ opcode, while still
        # allowing protocol 0 and 1 to work normally.  For this to
        # work, the function returned by __reduce__ should be
        # called __newobj__, and its first argument should be a
        # class.  The implementation for __newobj__
        # should be as follows, although pickle has no way to
        # verify this:
        #
        # def __newobj__(cls, *args):
        #     return cls.__new__(cls, *args)
        #
        # Protocols 0 and 1 will pickle a reference to __newobj__,
        # while protocol 2 (and above) will pickle a reference to
        # cls, the remaining args tuple, and the NEWOBJ code,
        # which calls cls.__new__(cls, *args) at unpickling time
        # (see load_newobj below).  If __reduce__ returns a
        # three-tuple, the state from the third tuple item will be
        # pickled regardless of the protocol, calling __setstate__
        # at unpickling time (see load_build below).
        #
        # Note that no standard __newobj__ implementation exists;
        # you have to provide your own.  This is to enforce
        # compatibility with Python 2.2 (pickles written using
        # protocol 0 or 1 in Python 2.3 should be unpicklable by
        # Python 2.2).
        cls = args[0]
        if not hasattr(cls, "__new__"):
            raise PicklingError(
                "args[0] from __newobj__ args has no __new__")
        if obj is not None and cls is not obj.__class__:
            raise PicklingError(
                "args[0] from __newobj__ args has the wrong class")
        args = args[1:]
        reduced[r_key_cls] = save_global_py3(cls)
        reduced[r_key_args] = args
    else:
        reduced[r_key_func] = save_global_py3(func)
        reduced[r_key_args] = args

    # if obj is not None:
    #     # If the object is already in the memo, this means it is
    #     # recursive. In this case, throw away everything we put on the
    #     # stack, and fetch the object back from the memo.
    #     if id(obj) in memo:
    #         write(POP + self.get(self.memo[id(obj)][0]))
    #     else:
    #         memoize(obj)

    # More new special cases (that work with older protocols as
    # well): when __reduce__ returns a tuple with 4 or 5 items,
    # the 4th and 5th item should be iterators that provide list
    # items and dict items (as (key, value) tuples), or None.

    if listitems is not None:
        reduced[r_key_listitems] = listitems

    if dictitems is not None:
        reduced[r_key_dictitems] = dictitems

    if state is not None:
        reduced[r_key_state] = state

    return reduced


def save_reduce_py2(func, args, state=None, listitems=None, dictitems=None,
                    obj=None, proto=2):
    reduced = {}
    # Assert that args is a tuple or None
    if not isinstance(args, TupleType):
        raise PicklingError("args from reduce() should be a tuple")

    # Assert that func is callable
    if not hasattr(func, '__call__'):
        raise PicklingError("func from reduce should be callable")

    # Protocol 2 special case: if func's name is __newobj__, use NEWOBJ
    if proto >= 2 and getattr(func, "__name__", "") == "__newobj__":
        # A __reduce__ implementation can direct protocol 2 to
        # use the more efficient NEWOBJ opcode, while still
        # allowing protocol 0 and 1 to work normally.  For this to
        # work, the function returned by __reduce__ should be
        # called __newobj__, and its first argument should be a
        # new-style class.  The implementation for __newobj__
        # should be as follows, although pickle has no way to
        # verify this:
        #
        # def __newobj__(cls, *args):
        #     return cls.__new__(cls, *args)
        #
        # Protocols 0 and 1 will pickle a reference to __newobj__,
        # while protocol 2 (and above) will pickle a reference to
        # cls, the remaining args tuple, and the NEWOBJ code,
        # which calls cls.__new__(cls, *args) at unpickling time
        # (see load_newobj below).  If __reduce__ returns a
        # three-tuple, the state from the third tuple item will be
        # pickled regardless of the protocol, calling __setstate__
        # at unpickling time (see load_build below).
        #
        # Note that no standard __newobj__ implementation exists;
        # you have to provide your own.  This is to enforce
        # compatibility with Python 2.2 (pickles written using
        # protocol 0 or 1 in Python 2.3 should be unpicklable by
        # Python 2.2).
        cls = args[0]
        if not hasattr(cls, "__new__"):
            raise PicklingError(
                "args[0] from __newobj__ args has no __new__")
        if obj is not None and cls is not obj.__class__:
            raise PicklingError(
                "args[0] from __newobj__ args has the wrong class")
        args = args[1:]
        reduced[r_key_cls] = save_global_py2(cls)
        reduced[r_key_args] = args
    else:
        reduced[r_key_func] = save_global_py2(func)
        reduced[r_key_args] = args

    # More new special cases (that work with older protocols as
    # well): when __reduce__ returns a tuple with 4 or 5 items,
    # the 4th and 5th item should be iterators that provide list
    # items and dict items (as (key, value) tuples), or None.

    if listitems is not None:
        reduced[r_key_listitems] = listitems

    if dictitems is not None:
        reduced[r_key_dictitems] = dictitems

    if state is not None:
        reduced[r_key_state] = state

    return reduced


def find_class_py3(module, name, proto=2, fix_imports=True):
    if proto < 3 and fix_imports:
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        if module in _compat_pickle.IMPORT_MAPPING:
            module = _compat_pickle.IMPORT_MAPPING[module]
    __import__(module, level=0)
    return _getattribute(sys.modules[module], name,
                         allow_qualname=proto >= 4)


def find_class_py2(module, name):
    # Subclasses may override this
    __import__(module)
    mod = sys.modules[module]
    klass = getattr(mod, name)
    return klass


def load_build_py2(inst, state):
    setstate = getattr(inst, "__setstate__", None)
    if setstate:
        setstate(state)
        return
    slotstate = None
    if isinstance(state, tuple) and len(state) == 2:
        state, slotstate = state
    if state:
        try:
            d = inst.__dict__
            try:
                for k, v in state.iteritems():
                    d[intern(k)] = v
            # keys in state don't have to be strings
            # don't blow up, but don't go out of our way
            except TypeError:
                d.update(state)

        except RuntimeError:
            # XXX In restricted execution, the instance's __dict__
            # is not accessible.  Use the old way of unpickling
            # the instance variables.  This is a semantic
            # difference when unpickling in restricted
            # vs. unrestricted modes.
            # Note, however, that cPickle has never tried to do the
            # .update() business, and always uses
            #     PyObject_SetItem(inst.__dict__, key, value) in a
            # loop over state.items().
            for k, v in state.items():
                setattr(inst, k, v)
    if slotstate:
        for k, v in slotstate.items():
            setattr(inst, k, v)
    return inst


def load_build_py3(inst, state):
    setstate = getattr(inst, "__setstate__", None)
    if setstate is not None:
        setstate(state)
        return
    slotstate = None
    if isinstance(state, tuple) and len(state) == 2:
        state, slotstate = state
    if state:
        inst_dict = inst.__dict__
        intern = sys.intern
        for k, v in state.items():
            if type(k) is str:
                inst_dict[intern(k)] = v
            else:
                inst_dict[k] = v
    if slotstate:
        for k, v in slotstate.items():
            setattr(inst, k, v)

    return inst


def load_global_py3(module, name):
    module = module.decode("utf-8")
    name = name.decode("utf-8")
    return find_class_py3(module, name)


def load_global_py2(module, name):
    return find_class_py2(module, name)


if is_py2:
    pickle_save_global = save_global_py2
    pickle_save = save_py2
    pickle_load_build = load_build_py2
    pickle_load_global = load_global_py2
elif is_py3:
    pickle_save_global = save_global_py3
    pickle_save = save_py3
    pickle_load_build = load_build_py3
    pickle_load_global = load_global_py3

############################### DISPATCH TABLES ###############################
#
# PYTHON 2 TYPE         PYTHON 3 TYPE   FUNCTION            NOTES
# -----------------------------------------------------------------------------
# NoneType              type(None)      save_none           -
# bool                  bool            save_bool           -
# IntType               -               save_int            -
# LongType              int             save_long           -
# FloatType             float           save_float          -
# StringType            bytes           save_{string,bytes} Jython (Py2 only)
# UnicodeType           str             save_{unicode,str}  -
# TupleType             tuple           save_tuple          recursive stuff
# ListType              list            save_list           -
# DictionaryType        dict            save_dict           -
# PyStringMap           PyStringMap     save_dict           Jython only
# InstanceType          -               save_instance       Old style class?
# -                     set             save_set            -
# -                     frozen_set      save_frozenset      -
# ClassType             -               save_global         -
# FunctionType          FunctionType    save_global         -
# BuiltinFunctionType   -               save_global         -
# TypeType              -               save_global         -
# -                     type            save_type           why not Py2?
###############################################################################


######################### RECONSTRUCTION OPCODES ##############################
#
# PYTHON 2  PYTHON 3    FUNCTION            NOTES
# -----------------------------------------------------------------------------
# NEWOBJ    NEWOBJ      load_newobj     -
# -         NEWOBJ_EX   load_newobj_ex  Protocol 4+
# REDUCE    REDUCE      load_reduce     -
# BUILD     BUILD       load_build      -
# INST      INST        load_inst       Old style instance
###############################################################################
