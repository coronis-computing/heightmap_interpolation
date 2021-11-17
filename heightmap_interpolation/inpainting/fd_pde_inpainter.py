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
import os
import cv2
import math
from timeit import default_timer as timer
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from heightmap_interpolation.inpainting.initializer import Initializer
from heightmap_interpolation.inpainting.convolver import Convolver
from heightmap_interpolation.inpainting.update_at_mask import update_at_mask


class FDPDEInpainter(ABC):
    """Abstract base class for Finite-Differences Partial Differential Equation (FDPDE) Inpainters

        Common interphase for PDE-based inpainting methods. Solves the problem using finite differences in a gradient-descent manner.

        Attributes:
            dt (float): Gradient descent step size.
            rel_change_iters (int): Check the relative change between iterations of the optimizer every this number of iterations.
            rel_change_tolerance (float): Stop the optimization when the energy descent between iterations is less than
            max_iters (int): Maximum number of iterations for the optimizer.
            relaxation (float): Over-relaxation parameter. It is still under testing, use with care.
            mgs_levels (int): Number of levels of detail to use in the Mult-Grid Solver (MGS). Setting it to 1 deactivates the MGS.
            mgs_min_res (int): minimum resolution (width or height) allowed for a level in the MGS. If the level of detail in the pyramid gets to a value lower than this, the pyramid construction will stop.
            print_progress (bool): Print information about the progress of the optimization on screen.
            print_progress_iters (int): If print_progress==True, the information will be printed every this number of iterations.
            init_with (str): initializer for the unknown data before applying the optimization.
            convolver_type (str): the convolver used for all the convolutions required by the solver.
            debug_dir (str): a debug directory where the intermediate steps will be rendered. Useful to create a video of the evolution of the solver.
        """
    def __init__(self, **kwargs):
        """Constructor

        Keyword Args:
            dt (float): Gradient descent step size.
            rel_change_iters (int): Check the relative change between iterations of the optimizer every this number of iterations.
            rel_change_tolerance (float): Stop the optimization when the energy descent between iterations is less than
            max_iters (int): Maximum number of iterations for the optimizer.
            relaxation (float): Over-relaxation parameter. It is still under testing, use with care.
            mgs_levels (int): Number of levels of detail to use in the Mult-Grid Solver (MGS). Setting it to 1 deactivates the MGS.
            mgs_min_res (int): minimum resolution (width or height) allowed for a level in the MGS. If the level of detail in the pyramid gets to a value lower than this, the pyramid construction will stop.
            print_progress (bool): Print information about the progress of the optimization on screen.
            print_progress_iters (int): If print_progress==True, the information will be printed every this number of iterations.
            init_with (str): initializer for the unknown data before applying the optimization.
            convolver_type (str): the convolver used for all the convolutions required by the solver.
            debug_dir (str): a debug directory where the intermediate steps will be rendered. Useful to create a video of the evolution of the solver.
        """
        super().__init__()
        # --- Gather and check the input parameters ---
        self.dt = kwargs.pop("update_step_size", 0.01)
        self.rel_change_iters = kwargs.pop("rel_change_iters", 1000)
        self.rel_change_tolerance = kwargs.pop("rel_change_tolerance", 1e-8)
        self.max_iters = int(kwargs.pop("max_iters", 1e8))
        self.relaxation = kwargs.pop("relaxation", 0)
        self.print_progress = kwargs.pop("print_progress", False)
        self.print_progress_iters = kwargs.pop("print_progress_iters", 1000)
        self.mgs_levels = kwargs.pop("mgs_levels", 1)
        self.mgs_min_res = kwargs.pop("mgs_min_res", 100)
        self.init_with = kwargs.pop("init_with", "zeros")
        self.convolver_type = kwargs.pop("convolver", "masked")
        self.convolver = Convolver(self.convolver_type)
        self.debug_dir = kwargs.pop("debug_dir", "")
        self.ts = 0 # ts is just a timer used to print the execution time of some of the steps

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
        if not isinstance(self.mgs_levels, int):
            raise ValueError("mgs_levels must be an integer")
        if not isinstance(self.mgs_min_res, int):
            raise ValueError("mgs_min_res must be an integer")

        # Some convenience variables to print progress
        #decimal_places_to_show = self.get_decimal_places(self.rel_change_tolerance) # DevNote: this does not work as expected yet...
        decimal_places_to_show = 10
        self.print_progress_table_row_str = "|{:>11d}|{:>" + str(26) + "." + str(decimal_places_to_show) + "f}|"
        self.print_progress_last_table_row_str = "| CONVERGED |{:>" + str(26) + "." + str(
            decimal_places_to_show) + "f}|"

        # Create the debug dir, if needed
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "progress"), exist_ok=True)
        self.current_level_debug_dir = self.debug_dir

        # Init the initializer object
        self.initializer = Initializer(self.init_with)

    def get_config(self):
        # Convert the internal configuration of the inpainter into a dictionary
        config = {"update_step_size": self.dt,
                  "rel_change_iters": self.rel_change_iters,
                  "rel_change_tolerance": self.rel_change_tolerance,
                  "max_iters": self.max_iters,
                  "relaxation": self.relaxation,
                  "print_progress": self.print_progress,
                  "print_progress_iters": self.print_progress_iters,
                  "mgs_levels": self.mgs_levels,
                  "mgs_min_res": self.mgs_min_res,
                  "init_with": self.init_with,
                  "convolver": self.convolver_type}
        return config

    def inpaint(self, image, mask):
        # Inpainting of an "image" by iterative minimization of a PDE functional.
        #
        # Input:
        #   img: input image to be inpainted
        #   mask: logical mask of the same size as the input image.
        #         True == known pixels, False == unknown pixels to be inpainted
        # Output:
        #   f: inpainted image

        self.print_msg("*** INPAINTING ***")

        if self.mgs_levels > 1:
            self.print_msg("* Using a multi-grid solver")
            inpainted = self.inpaint_multigrid(image, mask)
        else:
            # Init
            self.print_msg("* Initializing the inpainting problem using the '{:s}' filler".format(self.init_with))
            image = self.initializer.initialize(image, mask)
            if self.debug_dir:
                imgplot = plt.imshow(image)
                plt.savefig(os.path.join(self.current_level_debug_dir, "initialization.png"), bbox_inches="tight")
            # Inpaint
            self.print_msg("* Optimization:")
            inpainted = self.inpaint_grid(image, mask)
        return inpainted

    def inpaint_multigrid(self, image, mask):
        """Multigrid solver for inpainting problems.

        Args:
            img: input image to be inpainted
            mask: logical mask of the same size as the input image.
                  True == known pixels, False == unknown pixels to be inpainted
        Returns:
            inpainted image
        """

        # Check if it is worth applying a multi-grid solver for the resolution of the image
        if image.shape[0] < self.mgs_min_res or image.shape[1] < self.mgs_min_res:
            print("[WARNING] A multigrid solver was requested, but the size of the image is too small, defaulting to a single-scale inpainting")
            return self.inpaint_grid(image, mask)

        # Create the multi-scale pyramid
        self.print_start("Creating the image pyramid... ")
        image_pyramid = [image]
        mask_pyramid = [mask]
        num_levels = self.mgs_levels
        for level in range(1, self.mgs_levels):
            # Resize the image
            width = math.ceil(image_pyramid[level-1].shape[1] / 2)
            height = math.ceil(image_pyramid[level-1].shape[0] / 2)
            if width < self.mgs_min_res or height < self.mgs_min_res:
                print("[WARNING] Stopping pyramid construction at level {:d}, image resolution would be too small at this level (width or height < {:d})".format(level, self.mgs_min_res))
                num_levels = level
                break
            dim = (width, height)
            image_rs = cv2.resize(image_pyramid[level-1], dim)
            image_pyramid.append(image_rs)
            mask_rs = cv2.resize(np.asarray(mask_pyramid[level-1], dtype="uint8"), dim) == 1
            # Special case! If the image contains nans, resizing may increase those nans out of the resized mask, so we extend it to include the positions which are NaN in the image
            mask_valid = ~np.isnan(image_rs)
            mask_rs = np.logical_and(mask_rs, mask_valid)
            mask_pyramid.append(mask_rs)
        self.print_end()

        # Solve the inpainting problem at each level of the pyramid, using as initial guess the upscaled solution of
        # the previous level in the pyramid
        self.print_start("[Pyramid Level {:d}] Initializing the deepest level... ".format(num_levels - 1))
        init_lower_scale = self.initializer.initialize(image_pyramid[num_levels-1], mask_pyramid[num_levels-1])
        if self.debug_dir:
            self.current_level_debug_dir = os.path.join(self.debug_dir, str(num_levels-1))
            os.makedirs(self.current_level_debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.current_level_debug_dir, "progress"), exist_ok=True)
            imgplot = plt.imshow(init_lower_scale)
            plt.savefig(os.path.join(self.current_level_debug_dir, "initialization.png"), bbox_inches="tight")
        self.print_end()
        self.print_start("[Pyramid Level {:d}] Inpainting...\n".format(num_levels-1))
        inpainted_lower_scale = self.inpaint_grid(init_lower_scale, mask_pyramid[num_levels-1] > 0)
        self.print_end()
        if num_levels == 1:
            return inpainted_lower_scale
        for level in range(num_levels-2, -1, -1):
            self.print_start("[Pyramid Level {:d}] Inpainting...\n".format(level))

            image = image_pyramid[level]
            mask = mask_pyramid[level]

            # Upscale the previous solution
            upscaled_inpainted_lower_scale = cv2.resize(inpainted_lower_scale, dsize=(image.shape[1], image.shape[0]))

            # Special case: the first level of the pyramid may contain NaNs! (because we did not initialize it)
            # This will make the masking below to fail, so remove and substitute by zeros
            if level == 0 and np.any(np.isnan(image)):
                image[np.isnan(image)] = 0

            # Use the upscaled solution as initial guess
            image = upscaled_inpainted_lower_scale*(~mask) + image*mask

            # Prepare the debug folder
            if self.debug_dir:
                self.current_level_debug_dir = os.path.join(self.debug_dir, str(level))
                os.makedirs(self.current_level_debug_dir, exist_ok=True)
                os.makedirs(os.path.join(self.current_level_debug_dir, "progress"), exist_ok=True)

                # Inpaint
            inpainted = self.inpaint_grid(image, mask)

            self.print_end()

            if level > 0:
                inpainted_lower_scale = inpainted

        return inpainted

    def inpaint_grid(self, image, mask):
        # Actual inpainting function on a single-channel, single-scale image
        #
        # Input:
        #   img: input image to be inpainted
        #   mask: logical mask of the same size as the input image.
        #         True == known pixels, False == unknown pixels to be inpainted
        # Output:
        #   f: inpainted image

        mask_inv = 1-mask

        if self.convolver_type.startswith("masked"):
            mask_inp = cv2.dilate(np.asarray(mask_inv, dtype="uint8"), np.ones((3, 3))) == 1
        else:
            mask_inp = None
        # if self.convolver_type.startswith("masked"):
        #     pi_fun = lambda f: f
        # else:
        pi_fun = lambda f: f*mask_inv + image*mask

        # Initialize
        f = image
        #f[~mask] = 0 # Just in case the values not filled in the image are NaNs!

        # Iterate
        diff = 100000
        # last_diff = 0
        for i in range(0, self.max_iters):
            # Perform a step in the optimization
            # fnew = pi_fun(f + self.dt*self.step_fun(f, mask_inv))
            fnew = update_at_mask(image, f+self.dt*self.step_fun(f, mask_inp), mask_inv)

            # Over-relaxation?
            if self.relaxation > 1:
                fnew = pi_fun(f * (1 - self.relaxation) + fnew * self.relaxation)

            # Compute the difference with the previous step
            # This is a costly operation, do it every now and then:
            if i % self.rel_change_iters == 0:
                # diff = np.linalg.norm(fnew.flatten()-f.flatten(), 2)/np.linalg.norm(fnew.flatten(), 2) # DevNote: by profiling, we found this way to be much slower than the following line!
                diff = self.fast_norm(fnew.flatten() - f.flatten()) / self.fast_norm(fnew.flatten())

            # Update the function
            f = fnew

            if self.print_progress and i % self.print_progress_iters == 0:
                if i == 0:
                    print("+-----------+--------------------------+")
                    print("| Iteration | Function relative change |")
                    print("+-----------+--------------------------+")
                # print("Iter. %d, function relative change = %.10f" % (i, diff))
                # print(("|{:<11d}|{:>" + str(self.get_decimal_places(self.rel_change_tolerance)) + "f}|").format(i, diff))
                print(self.print_progress_table_row_str.format(i, diff))

            if self.debug_dir and i % self.print_progress_iters == 0:
                imgplot = plt.imshow(f)
                plt.savefig(os.path.join(self.current_level_debug_dir, "progress", "{:010d}.png".format(i)), bbox_inches="tight")

            #  % Stop if "almost" no change
            if i % self.rel_change_iters == 0 and diff < self.rel_change_tolerance:
                if self.print_progress:
                    print("+-----------+--------------------------+")
                    print(self.print_progress_last_table_row_str.format(diff))
                    print("+-----------+--------------------------+")
                if self.debug_dir:
                    imgplot = plt.imshow(f)
                    plt.savefig(os.path.join(self.current_level_debug_dir, "progress", "{:010d}.png".format(i)), bbox_inches="tight")
                return f

            # if i % self.rel_change_iters == 0:
            #     last_diff = diff

            # Check for increasing relative changes... should not happen in a convex optimization!
            # if last_diff > diff:
            #     print("[ERROR] Residuals increased from the last iteration. Sins this should be a convex optimization, this probably means the step size is too large!")
            #     return f
            # last_diff = diff

        # If we got here, issue a warning because the maximum number of iterations has been reached (normally means that
        # the solution will not be useful because it did not converge...)
        print("[WARNING] Inpainting did NOT converge: Maximum number of iterations reached...")

        return f

    def fast_norm(self, vector):
        return np.sqrt(np.sum(np.square(vector)))

    # Printing utilities...
    def print_start(self, msg):
        if self.print_progress:
            print(msg, end='')
            self.ts = timer()

    def print_end(self):
        if self.print_progress:
            te = timer()
            print("done ({:.2f} s)".format(te-self.ts))

    def print_msg(self, msg):
        if self.print_progress:
            print(msg)

    def get_decimal_places(self, number):
        if math.floor(number) == number:
            return 0
        return len("{:f}".format(number).split(".")[1])

        # --- The method to be implemented by each FDPDE inpainter ---
    @abstractmethod
    def step_fun(self, f, mask):
        pass