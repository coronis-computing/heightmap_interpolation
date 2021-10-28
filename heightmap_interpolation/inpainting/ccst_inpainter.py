from heightmap_interpolation.inpainting.fd_pde_inpainter import FDPDEInpainter
from scipy.ndimage.filters import laplace
import heightmap_interpolation.inpainting.differential as diff
import cv2


class CCSTInpainter(FDPDEInpainter):
    """Continous Curvature Splines in Tension (CCST) inpainter

    Implements the method in:
      Smith, W. H. F, and P. Wessel, 1990, Gridding with continuous curvature splines in tension, Geophysics, 55, 293-305.

    Should mimic GMT surface (http://gmt.soest.hawaii.edu/doc/latest/surface.html)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --- Gather and check the input parameters ---
        self.tension = kwargs.pop("tension", 0.0)
        if self.tension < 0. or self.tension > 1.:
            raise ValueError("tension parameter must be a number between 0 and 1 (included)")

    def step_fun(self, f, mask):
        # Version using scipy, slower than using OpenCV below
        # harmonic = laplace(f)
        # biharmonic = laplace(harmonic)

        # harmonic = cv2.filter2D(f, -1, diff.laplacian_kernel_2d)
        # biharmonic = cv2.filter2D(harmonic, -1, diff.laplacian_kernel_2d)

        harmonic = self.convolver(f, diff.laplacian_kernel_2d, mask)
        biharmonic = self.convolver(harmonic, diff.laplacian_kernel_2d, mask)

        return -1*((1-self.tension)*biharmonic - self.tension*harmonic)

    def get_config(self):
        config = super(CCSTInpainter, self).get_config()
        config["tension"] = self.tension
        return config
