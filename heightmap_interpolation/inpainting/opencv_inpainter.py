import cv2


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
        return cv2.inpaint(f, mask, self.radius, self.method)

