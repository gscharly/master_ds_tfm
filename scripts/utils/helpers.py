import hashlib


def hashlib_hash(obj):
    identifier = str(obj).encode('utf-8')
    return hashlib.sha256(identifier).hexdigest()
