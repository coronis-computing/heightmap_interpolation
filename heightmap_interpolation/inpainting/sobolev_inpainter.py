from heightmap_interpolation.inpainting.fd_pde_inpainter import FDPDEInpainter
from scipy.ndimage.filters import laplace
from heightmap_interpolation.inpainting.differential import divergence, gradient, my_laplacian
import heightmap_interpolation.inpainting.differential as diff
import scipy.signal
import numpy as np
import cv2


class SobolevInpainter(FDPDEInpainter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step_fun(self, f, mask):
        # return cv2.filter2D(f, -1, diff.laplacian_kernel_2d)
        return self.convolver(f, diff.laplacian_kernel_2d, mask)

        # DevNote: Other ways of doing the same operation (sorted from best to lowest performance):
        # - Our implementation of a laplacian with 2 1D filters:
        # return my_laplacian(f)
        # - Using opencv's Laplacian filter:
        # return cv2.Laplacian(f, -1)
        # - Using scipy.ndimage.filters.laplace function:
        # return laplace(f)
        # - Using the functions from heightmap_interpolation.inpainting.differential:
        # return divergence(gradient(f))
        # - Using scipy.signal.convolve function:
        # return scipy.signal.convolve(f, self.laplacian_stencil, mode='same')
        # - Using scipy.signal.convolve2d function:
        # return scipy.signal.convolve2d(f, self.laplacian_stencil, mode='same')
        # - Using scipy.signal.oaconvolve function:
        # return scipy.signal.oaconvolve(f, self.laplacian_stencil, mode='same')
        # - Using scipy.signal.fftconvolve function
        # return scipy.signal.fftconvolve(f, self.laplacian_stencil, mode='same')

