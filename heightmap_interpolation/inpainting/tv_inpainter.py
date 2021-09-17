from heightmap_interpolation.inpainting.fd_pde_inpainter import FDPDEInpainter
import heightmap_interpolation.inpainting.differential as diff
import numpy as np
import matplotlib.pyplot as plt


class TVInpainter(FDPDEInpainter):
    """
    Inpainter minimizing Total Variation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --- Gather and check the input parameters ---
        self.epsilon = kwargs.pop("epsilon", 1e-2)
        if self.epsilon <= 0:
            raise ValueError("Epsilon parameter must be greater than zero")

    def step_fun(self, f):
        # DevNotes:
        #   - We are not using the np.gradient() function because it uses 2nd order approximation (central differences) to compute the gradients, and this leads to "blocky" solutions for the PDE... Need to check why!
        #   - The first gradient uses forward differences, and the gradients inside "divergence" use backward differences. In this way, the resulting divergence is "aligned" to f
        return diff.divergence(self.neps(diff.gradient(f)))

    def amplitude(self, g):
        return np.sqrt(g[0]*g[0] + g[1]*g[1] + self.epsilon*self.epsilon)

    def neps(self, g):
        ampl = self.amplitude(g)
        g[0] = np.divide(g[0], ampl)
        g[1] = np.divide(g[1], ampl)
        return g