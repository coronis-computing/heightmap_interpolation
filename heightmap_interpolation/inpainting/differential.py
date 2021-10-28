import numpy as np
import scipy.ndimage.filters as spfilters
from numpy.fft  import fft2, ifft2
import cv2

# --> Kernels <--
backward_diff_kernel_1d = [-1, 1, 0]
forward_diff_kernel_1d = [0, -1, 1]
centered_diff_kernel_1d = [-1, 0, 1]
backward_diff_kernel_2d_horz = np.array([[0, 0, 0],
                                    [-1, 1, 0],
                                    [0, 0, 0]])
forward_diff_kernel_2d_horz = np.array([[0, 0, 0],
                                   [0, -1, 1],
                                   [0, 0, 0]])
centered_diff_kernel_2d_horz = np.array([[0, 0, 0],
                                    [-1, 0, 1],
                                    [0, 0, 0]])
backward_diff_kernel_2d_vert = np.array([[0, -1, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]])
forward_diff_kernel_2d_vert = np.array([[0, 0, 0],
                                        [0, -1, 0],
                                        [0, 1, 0]])
centered_diff_kernel_2d_vert = np.array([[0, -1, 0],
                                         [0, 0, 0],
                                         [0, 1, 0]])
laplacian_kernel_2d = np.array([[0.0, 1.0, 0],
                                [1.0, -4.0, 1.0],
                                [0.0, 1.0, 0]])
laplacian_kernel_1d_cv_horz = np.array([[0, 0, 0],
                                        [1.0, -2.0, 1.0],
                                        [0, 0, 0]])
laplacian_kernel_1d_cv_vert = np.array([[0, 1.0, 0],
                                        [0, -2.0, 0],
                                        [0, 1.0, 0]])

# --> Differential operators <--
def gradient(f, **kwargs):
    axis = kwargs.pop("axis", -1)
    order = kwargs.pop("order", 0)

    # # First derivatives in X/Y
    # (scipy version, slower than using cv2.filter2D)
    # if axis == 0:
    #     if order == 0:
    #         ux = spfilters.convolve1d(f, forward_diff_kernel_1d, axis=0)
    #     else:
    #         ux = spfilters.convolve1d(f, backward_diff_kernel_1d, axis=0)
    #     return ux
    # elif axis == 1:
    #     if order == 0:
    #         uy = spfilters.convolve1d(f, forward_diff_kernel_1d, axis=1)
    #     else:
    #         uy = spfilters.convolve1d(f, backward_diff_kernel_1d, axis=1)
    #     return uy
    # else:
    #     if order == 0:
    #         ux = spfilters.convolve1d(f, forward_diff_kernel_1d, axis=0)
    #         uy = spfilters.convolve1d(f, forward_diff_kernel_1d, axis=1)
    #     else:
    #         ux = spfilters.convolve1d(f, backward_diff_kernel_1d, axis=0)
    #         uy = spfilters.convolve1d(f, backward_diff_kernel_1d, axis=1)
    #     return [ux, uy]

    # First derivatives in X/Y
    if axis == 0:
        if order == 0:
            ux = cv2.filter2D(f, -1, forward_diff_kernel_2d_horz)
        else:
            ux = cv2.filter2D(f, -1, backward_diff_kernel_2d_horz)
        return ux
    elif axis == 1:
        if order == 0:
            uy = cv2.filter2D(f, -1, forward_diff_kernel_2d_vert)
        else:
            uy = cv2.filter2D(f, -1, backward_diff_kernel_2d_vert)
        return uy
    else:
        if order == 0:
            ux = cv2.filter2D(f, -1, forward_diff_kernel_2d_horz)
            uy = cv2.filter2D(f, -1, forward_diff_kernel_2d_vert)
        else:
            ux = cv2.filter2D(f, -1, backward_diff_kernel_2d_horz)
            uy = cv2.filter2D(f, -1, backward_diff_kernel_2d_vert)
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


# Code from https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html
# def np_fftconvolve(A, B):
    # return np.real(ifft2(fft2(A)*fft2(B, s=A.shape)))

# Code from https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution
# def np_fftconvolve(x, y):
#     s1 = np.array(x.shape)
#     s2 = np.array(y.shape)
#
#     size = s1 + s2 - 1
#
#     fsize = 2 ** np.ceil(np.log2(size)).astype(int)
#     fslice = tuple([slice(0, int(sz)) for sz in size])
#
#     new_x = np.fft.fft2(x, fsize)
#
#     new_y = np.fft.fft2(y, fsize)
#
#     result = np.fft.ifft2(new_x * new_y)[fslice].copy()
#
#     return result.real

def my_laplacian(f):
    # ux = spfilters.convolve1d(f, laplacian_kernel_1d, axis=0)
    # uy = spfilters.convolve1d(f, laplacian_kernel_1d, axis=1)
    # return ux+uy
    ux = cv2.filter2D(f, -1, laplacian_kernel_1d_cv_horz)
    uy = cv2.filter2D(f, -1, laplacian_kernel_1d_cv_vert)
    return ux+uy


class DifferentialOperators(object):
    """Differential operators implemented using a custom convolver"""

    def __init__(self, convolver):
        self.convolver = convolver

    def gradient(self, f, mask=None, axis=-1, order=0):

        # First derivatives in X/Y
        if axis == 0:
            if order == 0:
                ux = self.convolver(f, forward_diff_kernel_2d_horz, mask)
            else:
                ux = self.convolver(f, backward_diff_kernel_2d_horz, mask)
            return ux
        elif axis == 1:
            if order == 0:
                uy = self.convolver(f, forward_diff_kernel_2d_vert, mask)
            else:
                uy = self.convolver(f, backward_diff_kernel_2d_vert, mask)
            return uy
        else:
            if order == 0:
                ux = self.convolver(f, forward_diff_kernel_2d_horz, mask)
                uy = self.convolver(f, forward_diff_kernel_2d_vert, mask)
            else:
                ux = self.convolver(f, backward_diff_kernel_2d_horz, mask)
                uy = self.convolver(f, backward_diff_kernel_2d_vert, mask)
            return [ux, uy]

    def divergence(self, f, mask=None):
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
        return np.ufunc.reduce(np.add, [self.gradient(f[i], mask, axis=i, order=1) for i in range(num_dims)])