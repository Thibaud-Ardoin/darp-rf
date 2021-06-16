import scipy.io
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
import h5py

def _map_numpy_to_list(arr, func=None):
    """like `np.vectorize(func)(arr).tolist()`, except that it works."""
    if arr.ndim == 0:
        raise ValueError
    elif arr.ndim == 1:
        if func is None:
            return arr.tolist()
        else:
            return list(map(func, arr))
    else:
        return [_map_numpy_to_list(arr[i, ...], func) for i in range(arr.shape[0])]

def _build_objects(f, o):
    if isinstance(o, h5py.Reference):
        return _build_objects(f, f[o])

    c = o.attrs['MATLAB_class'].decode()

    if c == 'cell':
        x = _map_numpy_to_list(o[()], lambda y : _build_objects(f, y))
        return x

    elif c == 'struct':
        s = mat_struct()
        for k,v in o.items():
            setattr(s, k, _build_objects(f, v))
        return s

    else:
        x = o[()]
        assert np.issubdtype(x.dtype, np.number)
        if x.shape == (1,1):
            return x[0,0]
        elif x.shape == (1, x.size):
            return x.flatten()
        else:
            return x

def _loadmat_v7(filename, variable_names = None):
    if variable_names is  None:
        variable_names = set()
    else:
        variable_names = set(variable_names)
    vardict = {}

    h5 = h5py.File(filename, 'r')
    ignore = {'#refs#'} | variable_names
    for k,v in h5.items():
        if k in ignore:
            continue
        vardict[k] = _build_objects(h5, v)
    h5.close()
    return vardict

def _clean(x):
    """Fix up scipy.io output - Matlab cells should become lists"""
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.object_) and isinstance(x.flat[0], np.ndarray):
            return _map_numpy_to_list(x, _clean)
    return x

def loadmat(filename, variable_names = None):
    """
    Read a MATLAB mat file, for >=v7.3 uses h5py, otherwise uses scipy.io.loadmat.  MATLAB Cells become lists of lists,
    MATLAB arrays become numpy arrays and MATLAB structs become anonymous `mat_struct` objects.
    :param filename: Path to filename
    :param variable_names: Only load variables whose names are in this sequence.
    :return: A dictionary of MATLAB variables in the saved workspace.
    """
    try:
        matvars = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False, variable_names =variable_names)
        return dict(zip(matvars.keys(), map(_clean, matvars.values())))
    except NotImplementedError:
        return _loadmat_v7(filename, variable_names=variable_names)

def whosmat(filename):
    """List the variables in a MATLAB mat file."""
    try:
        return scipy.io.whosmat(filename)
    except NotImplementedError:
        h5 = h5py.File(filename, 'r')
        varlist = list(h5.keys())
        h5.close()
        try:
            varlist.remove('#refs#')
        except ValueError:
            pass
        return varlist
