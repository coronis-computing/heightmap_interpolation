from heightmap_interpolation.inpainting.fd_pde_inpainter import FDPDEInpainter
from heightmap_interpolation.inpainting.initializer import Initializer
import heightmap_interpolation.inpainting.differential as diff
import taichi as ti
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
import heightmap_interpolation.inpainting.differential as diff


@ti.data_oriented
class TaichiFDPDEInpainter():
    """Taichi Lang implementation of the Continous Curvature Splines in Tension (CCST) inpainter

    Implements the method in:
      Smith, W. H. F, and P. Wessel, 1990, Gridding with continuous curvature splines in tension, Geophysics, 55, 293-305.

    Should mimic GMT surface (http://gmt.soest.hawaii.edu/doc/latest/surface.html)
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
            [NOT USED] convolver_type (str): the convolver used for all the convolutions required by the solver.
            debug_dir (str): a debug directory where the intermediate steps will be rendered. Useful to create a video of the evolution of the solver.
            ti_arch: Taichi Lang architecture where the expensive parts of the processing will be executed.
        """
        super().__init__()
        # --- Gather and check the input parameters ---
        self.method = kwargs.pop("method", 0.01)
        self.dt = kwargs.pop("update_step_size", 0.01)
        self.term_check_iters = kwargs.pop("term_check_iters", 1000)
        self.term_criteria = kwargs.pop("term_criteria", "relative")
        self.term_thres = kwargs.pop("term_thres", 1e-8)        
        self.max_iters = int(kwargs.pop("max_iters", 1e8))
        self.relaxation = kwargs.pop("relaxation", 0)
        self.print_progress = kwargs.pop("print_progress", False)
        self.print_progress_iters = kwargs.pop("print_progress_iters", 1000)
        self.mgs_levels = kwargs.pop("mgs_levels", 1)
        self.mgs_min_res = kwargs.pop("mgs_min_res", 100)
        self.init_with = kwargs.pop("init_with", "zeros")
        self.convolver_type = kwargs.pop("convolver", "masked")
        #self.convolver = Convolver(self.convolver_type)
        self.debug_dir = kwargs.pop("debug_dir", "")
        self.ts = 0 # ts is just a timer used to print the execution time of some of the steps
        self.tension = kwargs.pop("tension", 0.0)
        self.ti_arch = kwargs.pop("ti_arch", "gpu")

        if self.dt <= 0:
            raise ValueError("update_step_size must be larger than zero")
        if self.term_thres <= 0:
            raise ValueError("term_thres must be larger than zero")
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
        if self.tension < 0. or self.tension > 1.:
            raise ValueError("tension parameter must be a number between 0 and 1 (included)")

        self.map_term_criteria_str_to_int = {
            "relative": 0,
            "absolute": 1,
            "absolute_percent": 2
        }

        # Some convenience variables to print progress
        #decimal_places_to_show = self.get_decimal_places(self.rel_change_tolerance) # DevNote: this does not work as expected yet...
        decimal_places_to_show = 10
        self.print_progress_table_row_str = "|{:>11d}|{:>" + str(17) + "." + str(decimal_places_to_show) + "f}|"
        self.print_progress_last_table_row_str = "| CONVERGED |{:>" + str(17) + "." + str(
            decimal_places_to_show) + "f}|"

        # Create the debug dir, if needed
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "progress"), exist_ok=True)
        self.current_level_debug_dir = self.debug_dir

        # Init the initializer object
        self.initializer = Initializer(self.init_with)

        # Initialize Taichi Lang    
        if self.ti_arch == 'cpu':
            ti.init(arch=ti.cpu)
        elif self.ti_arch == 'gpu':
            ti.init(arch=ti.gpu)
        elif self.ti_arch == 'cuda':
            ti.init(arch=ti.cuda)
        elif self.ti_arch == 'vulkan':
            ti.init(arch=ti.vulkan)
        elif self.ti_arch == 'opengl':
            ti.init(arch=ti.opengl)
        elif self.ti_arch == 'metal':
            ti.init(arch=ti.metal)

        # Define Taichi constants 
        self.laplacian_kernel_2d_ti = ti.field(dtype=ti.f32, shape=(3, 3))
        self.laplacian_kernel_2d_ti.from_numpy(diff.laplacian_kernel_2d.astype(np.float32))

    def get_config(self):
        # Convert the internal configuration of the inpainter into a dictionary
        config = {"update_step_size": self.dt,
                  "term_criteria": self.term_criteria,  
                  "term_check_iters": self.term_check_iters,
                  "term_thres": self.term_thres,
                  "max_iters": self.max_iters,
                  "relaxation": self.relaxation,
                  "print_progress": self.print_progress,
                  "print_progress_iters": self.print_progress_iters,
                  "mgs_levels": self.mgs_levels,
                  "mgs_min_res": self.mgs_min_res,
                  "init_with": self.init_with,
                  "convolver": self.convolver_type}
        return config

    def set_term_criteria(self):
        self.term_criteria_int = self.map_term_criteria_str_to_int.get(self.term_criteria, -1)
        if self.term_criteria_int == 0:
            self.print_msg("* Termination criteria --> relative change = {:f}".format(self.term_thres))
        elif self.term_criteria_int == 1:
            self.print_msg("* Termination criteria --> absolute change = {:f}".format(self.term_thres))
        elif self.term_criteria_int == 2:
            # Adapt the termination threshold based on the range of depth values on the map 
            max_val = np.max(self.f.to_numpy())
            min_val = np.min(self.f.to_numpy())
            z_range = abs(max_val-min_val)
            self.print_msg("* Termination criteria --> absolute percent change:")
            self.print_msg(f"    - Abs. Z range: {z_range:f}")
            self.print_msg(f"    - Percent = {self.term_thres:f}")
            self.term_thres = z_range*self.term_thres
            self.print_msg(f"    - Terminate if absolute change < {self.term_thres:f}")
        else:
            raise ValueError("Unknown termination criteria!")

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


    def inpaint_grid(self, image_np, mask_np):
        # Actual inpainting function on a single-channel, single-scale image (for the moment, we do not consider multi-grid solvers)
        #
        # Input:
        #   img: input image to be inpainted
        #   mask: logical mask of the same size as the input image.
        #         True == known pixels, False == unknown pixels to be inpainted
        # Output:
        #   f: inpainted image

        mask_inv_np = 1-mask_np

        if self.convolver_type.startswith("masked"):
            mask_inp_np = cv2.dilate(np.asarray(mask_inv_np, dtype="uint8"), np.ones((3, 3))) == 1
        else:
            mask_inp_np = None
        # !!! TODO: use the mask_inp...
        
        # Convert to Taichi types
        h, w = image_np.shape[0:2]
        self.image = ti.field(dtype=ti.f32, shape=(h, w))
        self.f = ti.field(dtype=ti.f32, shape=(h, w))
        self.image.from_numpy(image_np)
        self.f.from_numpy(image_np)            
        
        # Boolean types do not exist, so we convert them to integers
        self.mask = ti.field(dtype=ti.i8, shape=(h, w))
        self.mask.from_numpy(mask_np.astype(int))
        self.mask_inv = ti.field(dtype=ti.i8, shape=(h, w))
        self.mask_inv.from_numpy(mask_inv_np.astype(int))
        mask_inp_np = cv2.dilate(np.asarray(mask_inv_np, dtype="uint8"), np.ones((3, 3))) == 1
        self.mask_inp = ti.field(dtype=ti.i8, shape=(h, w))
        self.mask_inp.from_numpy(mask_inp_np.astype(int))

        # Some helper intermediate matrices
        self.harmonic = ti.field(dtype=ti.f32, shape=(h, w))
        self.biharmonic = ti.field(dtype=ti.f32, shape=(h, w))
        self.step_f = ti.field(dtype=ti.f32, shape=(h, w))          
        self.fprev = ti.field(dtype=ti.f32, shape=(h, w))
        self.fprev.from_numpy(image_np) 
        # self.fnew = ti.field(dtype=ti.f32, shape=(h, w))
        # self.fnew.from_numpy(f_np) 

        # Set the termination criteria
        self.set_term_criteria()

        # Inpaint!
        # self.inpaint_loop()
        self.inpaint_loop_2()

        # Return the results
        return self.f.to_numpy()
 
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
    
    def inpaint_loop(self):
        # Iterate until convergence (or max number of iterations reached)
        iter = 0
        converged = False
        while not converged and iter < self.max_iters:
            # Perform a step in the optimization
            #converged = self.inpaint_step(iter)
            self.update()
            if iter % self.term_check_iters == 0: 
                converged, diff = self.term_check()
            if self.print_progress and iter % self.print_progress_iters == 0:
                if iter == 0:
                    print("+-----------+-----------------+")
                    print("| Iteration | Function change |")
                    print("+-----------+-----------------+")
                    print(self.print_progress_table_row_str.format(iter, diff))

            if iter > 0 and (iter+1) % self.rel_change_iters == 0: 
                # In the following iteration, we will have to check the relative change, so save the last value
                self.fprev.copy_from(self.f)
                #self.update_f()
            # self.fnew.copy_from(self.f)
            iter += 1
        if iter >= self.max_iters:
            print("[WARNING] Inpainting did NOT converge: Maximum number of iterations reached...")

    def inpaint_loop_2(self):
        # Iterate until convergence (or max number of iterations reached)
        iter = 0
        converged = False
        print("+-----------+-----------------+")
        print("| Iteration | Function change |")
        print("+-----------+-----------------+")
        if self.debug_dir:
            imgplot = plt.imshow(self.f.to_numpy())
            plt.savefig(os.path.join(self.current_level_debug_dir, "progress", "{:010d}.png".format(iter)), bbox_inches="tight")
        while not converged and iter < self.max_iters:
            for i in range(self.term_check_iters):
                self.update()
            self.fprev.copy_from(self.f)
            self.update()
            # diff = self.step_diff()
            converged, diff = self.term_check()
            
            iter = iter + self.term_check_iters            
            if self.print_progress and iter % self.print_progress_iters == 0:
                print(self.print_progress_table_row_str.format(iter, diff))
                if self.debug_dir:
                    imgplot = plt.imshow(self.f.to_numpy())
                    plt.savefig(os.path.join(self.current_level_debug_dir, "progress", "{:010d}.png".format(iter)), bbox_inches="tight")
                
        if iter >= self.max_iters:
            print("[WARNING] Inpainting did NOT converge: Maximum number of iterations reached...")
        else:
            print("+-----------+-----------------+")
            print(self.print_progress_last_table_row_str.format(diff))
            print("+-----------+-----------------+")

    def inpaint_loop_no_out(self):
        # Iterate until convergence (or max number of iterations reached)
        iter = 0
        converged = False                
        while not converged and iter < self.max_iters:
            for i in range(self.term_check_iters):
                self.update()
            self.fprev.copy_from(self.f)
            self.update()
            converged, diff = self.term_check()
            
            iter = iter + self.term_check_iters            
            if self.print_progress and iter % self.print_progress_iters == 0:
                print(self.print_progress_table_row_str.format(iter, diff))
                if self.debug_dir:
                    imgplot = plt.imshow(self.f.to_numpy())
                    plt.savefig(os.path.join(self.current_level_debug_dir, "progress", "{:010d}.png".format(iter)), bbox_inches="tight")
                
        if iter >= self.max_iters:
            print("[WARNING] Inpainting did NOT converge: Maximum number of iterations reached...")
        else:
            print("+-----------+-----------------+")
            print(self.print_progress_last_table_row_str.format(diff))
            print("+-----------+-----------------+")

    def term_check(self):
        if self.term_criteria_int == 0: 
            # Relative change
            # diff = np.linalg.norm(fnew.flatten()-f.flatten(), 2)/np.linalg.norm(fnew.flatten(), 2) # DevNote: by profiling, we found this way to be much slower than the following line!
            diff = self.step_rel_diff()
        elif self.term_criteria_int == 1 or self.term_criteria_int == 2:
            # Absolute change
            diff = self.step_abs_diff()
        else:
            raise RuntimeError("Invalid termination criteria!")
        terminate = diff < self.term_thres
        return terminate, diff

    # @ti.kernel
    # def inpaint_step(self, iter: ti.types.i32) -> ti.types.i32:
    #     self.update()
    #     converged = self.converged(iter)        
    #     return converged

    @ti.func
    def laplacian(self, img, img_out):
        return convolve_taichi(img, img_out, self.laplacian_kernel_2d_ti)

    @ti.kernel
    def update(self):
        # self.ccst_step_fun()
        self.step_fun()
        for i in ti.grouped(self.f):        
            new_val = self.f[i] + self.dt*self.step_f[i]
            self.f[i] = self.pi_fun(new_val, i)
            # DevNote: for the moment, we sacrifice relaxation, as it slows down processing a lot as it was originally implemented... maybe we could run it every now and then?
            # if self.relaxation > 1:
                #     self.fnew[i] = self.pi_fun(self.f[i] * (1 - self.relaxation) + self.fnew[i] * self.relaxation)
    
    @ti.kernel
    def step_rel_diff(self) -> ti.types.f32:
        diff_fprev_sq_sum = 0.0
        fprev_sq_sum = 0.0
        for i in ti.grouped(self.f):
            diff_fprev = self.f[i]-self.fprev[i]
            diff_fprev_sq = diff_fprev*diff_fprev
            fprev_sq = self.fprev[i]*self.fprev[i]
            diff_fprev_sq_sum += diff_fprev_sq
            fprev_sq_sum += fprev_sq
        norm_diff_fprev = ti.sqrt(diff_fprev_sq_sum)
        norm_fprev = ti.sqrt(fprev_sq_sum)
        diff = norm_diff_fprev/norm_fprev
        return diff
    
    @ti.kernel
    def step_abs_diff(self) -> ti.types.f32:
        max_abs_diff = 0.0
        for i in ti.grouped(self.f):
            ti.atomic_max(max_abs_diff, abs(self.f[i]-self.fprev[i]))
        
        return max_abs_diff

    @ti.kernel
    def update_f(self):
        for i in ti.grouped(self.f):
            self.fprev[i] = self.f[i]

    @ti.func
    def pi_fun(self, val, i):
        return val*self.mask_inv[i] + self.image[i]*self.mask[i]

    @ti.func
    def step_fun(self):
        if ti.static(self.method == "ccst"):
            self.ccst_step_fun()
        elif ti.static(self.method == "harmonic"):
            print("Harmonic")
            self.harmonic_step_fun()
        else:
            print("[ERROR] Unknown method!")

    # Step functions for each method
    @ti.func
    def harmonic_step_fun(self):
        self.laplacian(self.f, self.step_f)

    @ti.func
    def ccst_step_fun(self):
        self.laplacian(self.f, self.harmonic) # , mask) !!!Ignoring mask for the moment!!!
        self.laplacian(self.harmonic, self.biharmonic)

        for i in ti.grouped(self.step_f):
            self.step_f[i] = -1*((1-self.tension)*self.biharmonic[i] - self.tension*self.harmonic[i])

@ti.func
def convolve_taichi(img, img_out, kernel):
    h, w = img.shape
    fh, _ = kernel.shape
    fr = fh // 2 # Radius of the filter (should be odd!)

    for i, j in ti.ndrange(h, w):
        k_begin, k_end = ti.max(0, i - fr), ti.min(h, i + fr + 1)
        l_begin, l_end = ti.max(0, j - fr), ti.min(w, j + fr + 1)

        total = 0.0
        for k, l in ti.ndrange((k_begin, k_end), (l_begin, l_end)):
            total += img[k, l] * kernel[k-i+fr, l-j+fr]
        
        img_out[i, j] = total