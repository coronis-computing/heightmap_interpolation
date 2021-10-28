from abc import ABC, abstractmethod
import numpy as np
import scipy
import cv2
from heightmap_interpolation.inpainting.convolve_at_mask import convolve_at_mask, nb_convolve_at_mask, nb_convolve_at_mask_parallel, nb_convolve_at_mask_guvec


class Convolver(object):
    # Common point for defining a convolution operation. Allows switching between different implementations of the convolution operation.

    def __init__(self, method):
        # DevNote: This method is actually a kind of "Factory" method, and in conjunction with ConvolverBase, they form a Strategy pattern.
        # if method.lower() == "numpy":
        #     self.actual_convolver = NumpyConvolver()
        if method.lower() == "scipy-signal":
            self.actual_convolver = ScipySignalConvolver()
        elif method.lower() == "scipy-ndimage":
            self.actual_convolver = ScipySignalConvolver()
        elif method.lower() == "opencv":
            self.actual_convolver = OpenCVConvolver()
        elif method.lower() == "masked":
            self.actual_convolver = MaskedConvolver()
        elif method.lower() == "masked-parallel":
            self.actual_convolver = ParallelMaskedConvolver()
        else:
            raise ValueError("Unknown convolver type '{:s}'".format(method))

    def __call__(self, image, kernel, mask):
        return self.actual_convolver(image, kernel, mask)


class ConvolverBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, image, kernel, mask):
        pass


class NumpyConvolver(ConvolverBase):
    # WARNING: this does not work!
    def __call__(self, image, kernel, mask):
        return np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel, s=image.shape)))


class ScipySignalConvolver(ConvolverBase):
    def __call__(self, image, kernel, mask):
        return scipy.signal.convolve(image, kernel, mode='same') # Using this instead of convolve2d because it automatically decides whether to use a 'direct' or an 'fft' convolution


class ScipyNDImageConvolver(ConvolverBase):
    def __call__(self, image, kernel, mask):
        return scipy.ndimage.convolve(image, kernel, mode='reflect')


class OpenCVConvolver(ConvolverBase):
    def __call__(self, image, kernel, mask):
        return cv2.filter2D(image, -1, kernel)


class MaskedConvolver(ConvolverBase):
    def __call__(self, image, kernel, mask):
        return nb_convolve_at_mask(image, mask, kernel)


class ParallelMaskedConvolver(ConvolverBase):
    def __call__(self, image, kernel, mask):
        return nb_convolve_at_mask_parallel(image, mask, kernel)


class GuvecMaskedConvolver(ConvolverBase):
    def __call__(self, image, kernel, mask):
        return nb_convolve_at_mask_guvec(image, mask, kernel)