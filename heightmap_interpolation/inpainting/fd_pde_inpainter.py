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

class FDPDEInpainter:
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
        # --- Gather and check the input parameters ---
        self.dt = kwargs.pop("update_step_size", 0.01)
        self.rel_change_tolerance = kwargs.pop("rel_change_tolerance", 1e-8)
        self.max_iters = kwargs.pop("max_iters", 1e8)
        self.relaxation = kwargs.pop("relaxation", 0)

        if self.dt <= 0:
            raise ValueError("update_step_size must be larger than zero")
        if self.rel_change_tolerance <= 0:
            raise ValueError("rel_change_tolerance must be larger than zero")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be larger than zero")
        if self.relax < 1 | self.relax > 2:
            raise ValueError("relaxation must be a number betwee 1 and 2")
        if not isinstance(self.max_iters, int):
            raise ValueError("max_iters must be an integer")

    def __call__(self, image, mask):
        # Inpainting of an "image" by iterative minimization of a PDE functional
        #
        # Input:
        #   img: input image to be inpainted
        #   mask: logical mask of the same size as the input image.
        #         1 == known pixels, 0 == unknown pixels to be inpainted
        # Output:
        #   f: inpainted image

        pi_fun = lambda f: f*(1-mask) + image*mask

        # Initialize
        f = image

        # Iterate
        for i in range(1, self.max_iters):
            # Perform a step in the optimization
            fnew = pi_fun(f - self.dt*self.step_fun(f))

            # Over-relaxation?
            if self.relax > 1:
                fnew = pi_fun(f * (1 - self.relax) + fnew * self.relax)

            # Compute the difference with the previous step
            diff = np.linalg.norm(fnew.reshape(-1, 1)-f.reshape(-1, 1),2)/np.linalg.norm(fnew.reshape(-1, 1),2);

            # Update the function
            f = fnew

            #  % Stop if "almost" no change
            if diff < self.rel_change_tolerance:
                return f

        # If we got here, issue a warning because the maximum number of iterations has been reached (normally means that
        # the solution will not be useful because it did not converge...)
        print("[WARNING] Maximum number of iterations reached")

        return f
