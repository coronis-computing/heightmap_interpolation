import cv2
import numpy as np


class OpenCVInpainter():
    """Interphase for using OpenCV's builtin inpainting functions"""

    def __init__(self, **kwargs):
        method = kwargs.pop("method", "navier-stokes")
        self.radius = kwargs.pop("radius", 25)
        if method == "navier-stokes":
            self.method = cv2.INPAINT_NS
        elif method == "telea":
            self.method = cv2.INPAINT_TELEA
        else:
            raise ValueError("Unknown method")
        if self.radius <= 0:
            raise ValueError("Radius should be a positive value > 0")

    def inpaint(self, f, mask):
        mask = ~mask
        mask_cv = mask.astype(np.uint8)  # convert to an unsigned byte
        mask_cv *= 255
        return cv2.inpaint(f, mask_cv, self.radius, self.method)

    def get_config(self):
        config = {"radius": self.radius}
        return config


class OpenCVXPhotoInpainter():
    def __init__(self, **kwargs):
        method = kwargs.pop("method", "shiftmap")
        if method == "shiftmap":
            self.method = cv2.xphoto.INPAINT_SHIFTMAP

    def inpaint(self, f, mask):
        mask = ~mask # Do not invert! xphoto.inpaint expects the mask as the inverse of the photo module...
        mask_cv = mask.astype(np.uint8)  # convert to an unsigned byte
        mask_cv *= 255
        mask_cv = cv2.bitwise_not(mask_cv)
        result = np.zeros_like(f)
        cv2.xphoto.inpaint(f, mask_cv, result, self.method)
        return result