from heightmap_interpolation.inpainting.fd_pde_inpainter import FDPDEInpainter
from scipy.ndimage.filters import convolve1d
import numpy as np
import heightmap_interpolation.inpainting.differential as diff


class AMLEInpainter(FDPDEInpainter):
    """ Absolutely Minimizing Lipschitz Extension (AMLE) Inpainter
    Implements the method in:
        Andrés Almansa, Frédéric Cao, Yann Gousseau, and Bernard Rougé.
        Interpolation of Digital Elevation Models Using AMLE and Related
        Methods. IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 40,
        NO. 2, FEBRUARY 2002
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --- Gather and check the input parameters ---
        self.convolve_in_1d = kwargs.pop("convolve_in_1d", False)

    def step_fun(self, f, mask):
        if self.convolve_in_1d:
            # --- Using 1-dimensional convolutions ---
            # First derivatives in X/Y
            ux = convolve1d(f, diff.forward_diff_kernel_1d, axis=0)
            uy = convolve1d(f, diff.forward_diff_kernel_1d, axis=1)

            # Second derivatives
            uxx = convolve1d(ux, diff.backward_diff_kernel_1d, axis=0)
            uxy = convolve1d(ux, diff.backward_diff_kernel_1d, axis=1)
            uyx = convolve1d(uy, diff.backward_diff_kernel_1d, axis=0)
            uyy = convolve1d(uy, diff.backward_diff_kernel_1d, axis=1)

            # Du/|Du| with central differences
            v0 = convolve1d(f, diff.centered_diff_kernel_1d, axis=0)
            v1 = convolve1d(f, diff.centered_diff_kernel_1d, axis=1)
        else:
            # --- Same as above but with the 2D convolver ---
            # First derivatives in X/Y
            ux = self.convolver(f, diff.forward_diff_kernel_2d_vert, mask)
            uy = self.convolver(f, diff.forward_diff_kernel_2d_horz, mask)

            # Second derivatives
            uxx = self.convolver(ux, diff.backward_diff_kernel_2d_vert, mask)
            uxy = self.convolver(ux, diff.backward_diff_kernel_2d_horz, mask)
            uyx = self.convolver(uy, diff.backward_diff_kernel_2d_vert, mask)
            uyy = self.convolver(uy, diff.backward_diff_kernel_2d_horz, mask)

            # Du/|Du| with central differences
            v0 = self.convolver(f, diff.centered_diff_kernel_2d_vert, mask)
            v1 = self.convolver(f, diff.centered_diff_kernel_2d_horz, mask)

        # Normalize the direction field
        dennormal = np.sqrt(v0*v0 + v1*v1 + 1e-15)
        v0 = v0/dennormal
        v1 = v1/dennormal

        return uxx*v0*v0 + uyy*v1*v1 + (uxy+uyx)*v0*v1

