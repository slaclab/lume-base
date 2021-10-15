import hashlib
import json
import numpy as np
import os


class NpEncoder(json.JSONEncoder):
    """
    Custom encoder to serialize Numpy data types.

    [StackOverflow reference](https://stackoverflow.com/q/50916422)
    """
    def default(self, obj):
        """
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data.
    Used JSON dumps to form strings, and the blake2b algorithm to hash.

    Parameters
    ----------
    keyed_data : dict
        dict with the keys to generate a fingerprint
    digest_size : int, optional
        Digest size for blake2b hash code, by default 16

    Returns
    -------
    str
        The hexadecimal digest
    """
    h = hashlib.blake2b(digest_size=digest_size)
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()


def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path

    Parameters
    ----------
    path : str
        A path possibly containing environment variables and user (~) shortcut

    Returns
    -------
    str
        The expanded absolute path
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
