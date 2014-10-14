# coding=utf-8
import h5it
import pickle
import sys
import os

unicode_test_str = (u'σκουλικομερμυγκότρυπα ασπρη πέτρα '
                    u'ξέξασπρη κι από τον ήλιο ξεξασπρότερη')

byte_test_str = b'the rain in spain falls mainly on the plain'

v = sys.version_info.major

test_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
print(test_dir)

with open(test_dir + 'py{}_unicode_proto2_bin.pickle'.format(v), 'wb') as f:
    pickle.dump(unicode_test_str, f, protocol=2)

with open(test_dir + 'py{}_bytes_proto2_bin.pickle'.format(v), 'wb') as f:
    pickle.dump(byte_test_str, f, protocol=2)

h5it.dump(unicode_test_str, test_dir + 'py{}_unicode.hdf5'.format(v))
h5it.dump(byte_test_str, test_dir + 'py{}_bytes.hdf5'.format(v))
