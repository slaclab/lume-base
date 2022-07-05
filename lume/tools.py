import hashlib
import json
import numpy as np
import os
import shutil


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


def make_executable(path):
    """
    Makes a file executable.

    https://stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python
    """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def find_executable(exename=None, envname=None):
    """
    Finds an executable from a given name or environmental variable.

    If neither are files, the path will be searched for exename

    """

    # Simply return if this exists
    if exename and os.path.isfile(exename):
        assert os.access(exename, os.X_OK), f'File is not executable: {exename}'
        return full_path(exename)

    envexe = os.environ.get(envname)
    if envexe and os.path.isfile(envexe):
        assert os.access(envexe, os.X_OK), f'File is not executable: {envexe}'
        return full_path(envexe)

    if not exename and not envname:
         raise ValueError('No exename or envname ')

    # Start searching
    search_path = []
    #search_path.append(os.environ.get(envname))
    search_path.append(os.getcwd())
    search_path.append(os.environ.get('PATH'))
    search_path_str = os.pathsep.join(search_path)
    bin_location = shutil.which(exename, path=search_path_str)

    if bin_location and os.path.isfile(bin_location):
        return full_path(bin_location)

    raise ValueError(f'Could not find executable: exename={exename}, envname={envname}')
