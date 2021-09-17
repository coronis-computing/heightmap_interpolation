import numpy as np
import scipy.ndimage.filters as spfilters

# --> Kernels <--
backward_diff_kernel = [-1, 1, 0]
forward_diff_kernel = [0, -1, 1]
centered_diff_kernel = [-1, 0, 1]


# --> Differential operators <--
def gradient(f, **kwargs):
    axis = kwargs.pop("axis", -1)
    order = kwargs.pop("order", 0)

    # First derivatives in X/Y
    if axis == 0:
        if order == 0:
            ux = spfilters.convolve1d(f, forward_diff_kernel, axis=0)
        else:
            ux = spfilters.convolve1d(f, backward_diff_kernel, axis=0)
        return ux
    elif axis == 1:
        if order == 0:
            uy = spfilters.convolve1d(f, forward_diff_kernel, axis=1)
        else:
            uy = spfilters.convolve1d(f, backward_diff_kernel, axis=1)
        return uy
    else:
        if order == 0:
            ux = spfilters.convolve1d(f, forward_diff_kernel, axis=0)
            uy = spfilters.convolve1d(f, forward_diff_kernel, axis=1)
        else:
            ux = spfilters.convolve1d(f, backward_diff_kernel, axis=0)
            uy = spfilters.convolve1d(f, backward_diff_kernel, axis=1)
        return [ux, uy]


def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    Code extracted from: https://stackoverflow.com/questions/11435809/compute-divergence-of-vector-field-using-python (Daniel's answer)
                         https://stackoverflow.com/questions/67970477/compute-divergence-with-python/67971515#67971515

    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    # return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])
    # return np.ufunc.reduce(np.add, [np.diff(f[i], axis=i) for i in range(num_dims)])
    # return np.ufunc.reduce(np.add, [scipy.ndimage.filters.sobel(f[i], axis=i)/8.0 for i in range(num_dims)])
    return np.ufunc.reduce(np.add, [gradient(f[i], axis=i, order=1) for i in range(num_dims)])
