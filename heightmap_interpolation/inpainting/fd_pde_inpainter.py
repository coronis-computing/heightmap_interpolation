# Copyright (c) 2020 Coronis Computing S.L. (Spain)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Ricard Campos (ricard.campos@coronis.es)

import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class FDPDEInpainter(ABC):
    """ Abstract base class for Finite-Differences Partial Differential Equation (FDPDE) Inpainters
        Common interphase for PDE-based inpainting methods. Solves the problem using finite differences

        Attributes:
            relChangeTolerance: Relative tolerance, stop the gradient descent when the energy descent between iterations is less than this value
            maxIters: Maximum number of gradient descent iterations to perform
            dt: Gradient descent step size
            relax: Over-relaxation parameter
        """
    def __init__(self, **kwargs):
        """ Constructor """
        super().__init__()
        # --- Gather and check the input parameters ---
        self.dt = kwargs.pop("update_step_size", 0.01)
        self.rel_change_tolerance = kwargs.pop("rel_change_tolerance", 1e-8)
        self.max_iters = int(kwargs.pop("max_iters", 1e8))
        self.relaxation = kwargs.pop("relaxation", 0)
        self.print_progress = kwargs.pop("print_progress", False)
        self.print_progress_iters = kwargs.pop("print_progress_iters", 1000)
        self.show_progress = kwargs.pop("show_progress", False)

        if self.dt <= 0:
            raise ValueError("update_step_size must be larger than zero")
        if self.rel_change_tolerance <= 0:
            raise ValueError("rel_change_tolerance must be larger than zero")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be larger than zero")
        if self.relaxation != 0.0 and (self.relaxation < 1.0 or self.relaxation > 2.0):
            raise ValueError("relaxation must be a number between 1 and 2 (0 to deactivate)")
        if not isinstance(self.max_iters, int):
            raise ValueError("max_iters must be an integer")

    def inpaint(self, image, mask):
        # Inpainting of an "image" by iterative minimization of a PDE functional
        #
        # Input:
        #   img: input image to be inpainted
        #   mask: logical mask of the same size as the input image.
        #         True == known pixels, False == unknown pixels to be inpainted
        # Output:
        #   f: inpainted image

        print(self.dt)

        pi_fun = lambda f: f*(1-mask) + image*mask

        # Initialize
        f = image
        f[~mask] = 0 # Just in case the values not filled in the image are NaNs!

        # Iterate
        last_diff = 0
        for i in range(1, self.max_iters):
            # Perform a step in the optimization
            fnew = pi_fun(f + self.dt*self.step_fun(f))

            # Over-relaxation?
            if self.relaxation > 1:
                fnew = pi_fun(f * (1 - self.relaxation) + fnew * self.relaxation)

            # Compute the difference with the previous step
            diff = np.linalg.norm(fnew.flatten()-f.flatten(), 2)/np.linalg.norm(fnew.flatten(), 2)

            # Update the function
            f = fnew

            if self.print_progress and i % self.print_progress_iters == 0:
                print("Iter %d: function relative change = %f" % (i, diff))

            if self.show_progress and i % self.print_progress_iters == 0:
                imgplot = plt.imshow(f)
                plt.pause(0)

            #  % Stop if "almost" no change
            if diff < self.rel_change_tolerance:
                return f

            # Check for increasing relative changes... should not happen in a convex optimization!
            # if last_diff > diff:
            #     print("[ERROR] Residuals increased from the last iteration. Sins this should be a convex optimization, this probably means the step size is too large!")
            #     return f
            # last_diff = diff

        # If we got here, issue a warning because the maximum number of iterations has been reached (normally means that
        # the solution will not be useful because it did not converge...)
        print("[WARNING] Maximum number of iterations reached")

        return f

    @abstractmethod
    def step_fun(self, f, mask):
        pass