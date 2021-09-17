from heightmap_interpolation.inpainting.fd_pde_inpainter import FDPDEInpainter
from scipy.ndimage.filters import laplace


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

    def step_fun(self, f):
        harmonic = laplace(f)
        biharmonic = laplace(harmonic)

        return -1*((1-self.tension)*biharmonic - self.tension*harmonic)
