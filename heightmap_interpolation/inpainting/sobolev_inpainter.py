from heightmap_interpolation.inpainting.fd_pde_inpainter import FDPDEInpainter
from scipy.ndimage.filters import laplace
from heightmap_interpolation.inpainting.differential import divergence, gradient
import scipy.signal
import numpy as np
import cv2


class SobolevInpainter(FDPDEInpainter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.laplacian_stencil = np.array([[0.0, 1.0, 0],
                                           [1.0, -4.0, 1.0],
                                           [0.0, 1.0, 0]])

    def step_fun(self, f):
        return cv2.filter2D(f, -1, self.laplacian_stencil)

        # DevNote: Other ways of doing the same operation (sorted from best to lowest performance):
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