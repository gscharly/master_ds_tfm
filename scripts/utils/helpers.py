import hashlib

import numpy as np


def hashlib_hash(obj):
    identifier = str(obj).encode('utf-8')
    return hashlib.sha256(identifier).hexdigest()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
