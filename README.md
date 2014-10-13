h5it
====

An implimentation of the Pickle protocol 2 to HDF5 files.



Pickle Support
==============

h5it achieves excellent coverage of the pickle protocol as it actually includes
modified parts of the pickle standard library module (pickle.py). pickle.py
is a well laid out body of code but it mixes two concerns, namely:

1. How to save out base types (strings, numbers, tuples, lists...). pickle.py
uses a stack-based approach that is fundamentally different to the
hierarchical nature of HDF5, so we redefine all these functions in h5it
so that we get a clean layout of data in HDF5 files.

2. How to deal with a number of subtle complexities like:
 a. Parsing a large object graph, dispatching the correct method for the
 correct type
 b. Memoizing what has already been found (taking care to deal with recursive
 structures)
 c. How to give globals a safe unique string identifier
 d. Saving out objects through the __reduce__ protocol

It would be great if the second set of challenging concerns were separate
from the first, then creating something like h5it would be straightforward.
Instead the two are intertwined, hence the need to include modified
functions from the standard library (we basically drop all the stack saving
stuff but keep the pickle.py code for handing all the complexity).

Having said all that, defining the separation is not all that easy. There are
a number of things that aren't yet supported by h5it, but could be in the
future:

1. Protocols other than 2. Protocol 2 is compatible with both Python 2 and 3
hence it's choice as the basis of h5it. In the future newer protocols
could definitely be added.

2. The saving of extension codes. These are integer codes stored in copyreg
which are identifiers for saving out extension modules. Again support can
absolutely be added in the future

3. The use of persistent ids. This only exists for subclasses of
Pickler/Unpickler so it's not surprising we don't support it.

4. Python 3.3 added private dispatch tables to picklers - this is not
presently supported in h5it (you can customise __reduce__ methods but only
in the global copyreg module).