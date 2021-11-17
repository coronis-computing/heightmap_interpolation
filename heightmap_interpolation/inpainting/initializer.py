import heightmap_interpolation.inpainting.sobolev_inpainter
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

class Initializer():

    def __init__(self, init_with):
        self.init_with = init_with
        available_options = ["zeros", "mean", "nearest", "linear", "cubic", "sobolev"]
        if self.init_with.lower() not in available_options:
            raise ValueError("init_with must be one of these options: " + ", ".join(available_options))

    def initialize(self, image, mask):
        if self.init_with.lower() == "zeros":
            image[~mask] = 0
        elif self.init_with.lower() == "mean":
            image[~mask] = np.mean(image[mask])
        elif self.init_with.lower() == "nearest" or self.init_with.lower() == "linear" or self.init_with.lower() == "cubic":
            if self.init_with.lower() == "nearest":
                fill_value = 0 # Not important, will be disregarded...
            else:
                fill_value = np.mean(image[mask]) # Fill value equal to the mean
            # Grid
            x = np.arange(0, image.shape[1])
            y = np.arange(0, image.shape[0])
            xx, yy = np.meshgrid(x, y)
            # Get reference values
            x1 = xx[mask]
            y1 = yy[mask]
            ref = image[mask]
            # Use griddata to interpolate
            interp = interpolate.griddata((x1, y1), ref.ravel(), (xx, yy), method=self.init_with.lower(), fill_value=fill_value)
            image[~mask] = interp[~mask]
        elif self.init_with.lower() == "sobolev" or self.init_with.lower() == "harmonic":
            params_dict = {"update_step_size": 0.8/4,
                           "rel_change_tolerance": 1e-5,
                           "max_iters": 1e5}
            inpainter = heightmap_interpolation.inpainting.sobolev_inpainter.SobolevInpainter(**params_dict)
            image = inpainter.inpaint(image, mask)
            print("initialized with sobolev")
        else:
            raise ValueError("Unknown initialization type")
        return image