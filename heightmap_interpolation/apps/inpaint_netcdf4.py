#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from heightmap_interpolation.misc.conditional_print import ConditionalPrint
import cvxpy as cp
from timeit import default_timer as timer
from heightmap_interpolation.apps.common import load_data, write_results
from heightmap_interpolation.inpainting.fd_pde_inpainter_factory import create_fd_pde_inpainter
import scipy

# def total_variation(arr):
#     dx = cp.vec(arr[1:, :-1] - arr[:-1, :-1])
#     dy = cp.vec(arr[:-1, 1:] - arr[:-1, :-1])
#     D = cp.vstack((dx, dy))
#     norm = cp.norm(D, p=1, axis=0)
#     return cp.sum(norm)


def inpaint(param):
    # Conditional print
    condp = ConditionalPrint(param.verbose)

    # Load the data
    condp.print("- Loading input data")
    elevation, mask_int, lats_mat, lons_mat, mask_ref = load_data(param)

    # Compute the percentage of missing data to interpolate w.r.t. the full image
    if param.verbose:
        total_pixels = elevation.shape[0]*elevation.shape[1]
        num_pixels_to_inpaint = np.count_nonzero(mask_int)
        inpaint_percent = (num_pixels_to_inpaint/total_pixels)*100
        condp.print("Pixels to inpaint represent a {:.2f}% of the image ({:d}/{:d})".format(inpaint_percent, num_pixels_to_inpaint, total_pixels))

    # # TV interpolant
    # condp.print("- Defining the inpainting problem")
    # rows, cols = np.where(mask_int == False)
    # x = cp.Variable(elevation.shape)
    # objective = cp.Minimize(total_variation(x))
    # knowledge = x[rows, cols] == elevation[rows, cols]
    # constraints = [knowledge]
    # prob = cp.Problem(objective, constraints)
    # condp.print("- Solving the problem:")
    # prob.solve(solver=cp.SCS, verbose=param.verbose)

    # # Collect common parameters
    # params_dict = {"update_step_size": 0.01,
    #                "rel_change_tolerance": 1e-8,
    #                "max_iters": 1e8,
    #                "relaxation": 0,
    #                "mgs_levels": param.mgs_levels,
    #                "print_progress": True,
    #                "print_progress_iters": 1000,
    #                "init_with": param.init_with,
    #                "convolver": param.convolver}

    # Sobolev inpainter
    # condp.print("- Inpainting")
    # if param.method.lower() == "sobolev":
    #     condp.print("   - Using Sobolev inpainter")
    #     params_dict["update_step_size"] = 0.8/4
    #     params_dict["rel_change_tolerance"] = 1e-5
    #     params_dict["max_iters"] = 1e5
    #     inpainter = SobolevInpainter(**params_dict)
    #     inpaint_mask = ~mask_int
    # elif param.method.lower() == "tv":
    #     params_dict["epsilon"] = 1
    #     params_dict["update_step_size"] = .9*params_dict["epsilon"]/4
    #     params_dict["rel_change_tolerance"] = 1e-5
    #     params_dict["max_iters"] = 1e5
    #     params_dict["show_progress"] = False
    #     inpainter = TVInpainter(**params_dict)
    #     inpaint_mask = ~mask_int
    # elif param.method.lower() == "ccst":
    #     params_dict["update_step_size"] = 0.01
    #     # params_dict["rel_change_tolerance"] = 1e-8
    #     params_dict["rel_change_tolerance"] = 1e-8
    #     params_dict["max_iters"] = 1e8
    #     # params_dict["relaxation"] = 1.4
    #     params_dict["tension"] = 0.3
    #     inpainter = CCSTInpainter(**params_dict)
    #     inpaint_mask = ~mask_int
    # elif param.method.lower() == "amle":
    #     params_dict["update_step_size"] = 0.01
    #     params_dict["rel_change_tolerance"] = 1e-5
    #     params_dict["max_iters"] = 1e8
    #     inpainter = AMLEInpainter(**params_dict)
    #     inpaint_mask = ~mask_int
    # elif param.method.lower() == "navier-stokes":
    #     # Convert mask to opencv's inpaint function expected format
    #     inpainter = OpenCVInpainter(method="navier-stokes", radius=25)
    #     inpaint_mask = mask_int.astype(np.uint8)  # convert to an unsigned byte
    #     inpaint_mask *= 255
    # elif param.method.lower() == "telea":
    #     inpainter = OpenCVInpainter(method="telea", radius=5)
    #     inpaint_mask = mask_int.astype(np.uint8)  # convert to an unsigned byte
    #     inpaint_mask *= 255
    # elif param.method.lower() == "transport":
    #     params_dict["update_step_size"] = 0.05
    #     params_dict["rel_change_tolerance"] = 1e-5
    #     params_dict["max_iters"] = 50
    #     params_dict["iters_inpainting"] = 40
    #     params_dict["iters_anisotropic"] = 2
    #     params_dict["epsilon"] = 1e-10
    #     # inpainter = BertalmioInpainter(**params_dict)
    #     # inpaint_mask = scipy.float64((~mask_int))
    # else:
    #     print("[ERROR] The required method (" + param.method + ") is unknown. Available options: sobolev, ccst")

    # Setup the options from the script's input args
    options = {
        "mgs_levels": param.mgs_levels,
        "print_progress": param.verbose,
        "init_with": param.init_with,
        "convolver": param.convolver,
        "debug_dir": param.debug_dir
    }

    # Create the inpainter
    inpaint_mask = ~mask_int
    inpainter = create_fd_pde_inpainter(param.method, options)

    # Inpaint!
    if param.verbose:
        ts = timer()
    elevation_inpainted = inpainter.inpaint(elevation, inpaint_mask)
    if param.verbose:
        te = timer()
        condp.print("- Inpainting took a total of {:.2f} sec.".format(te - ts))

    # Write the results
    if param.output_file:
        condp.print("- Writing the results to disk")
        write_results(param, elevation, mask_int)

    # Show results
    if param.show:
        condp.print("- Showing results")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        images = [elevation, elevation_inpainted]
        titles = ['Original', 'Inpainted']
        for (ax, image, title) in zip(axes, images, titles):
            ax.imshow(image)
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Interpolate terrain data in a SeaDataNet_1.0 CF1.6-compliant netCDF4 file via inpainting")
    parser.add_argument("input_file", action="store", type=str,
                        help="Input NetCDF file")
    parser.add_argument("-o", "--output_file", action="store", type=str, dest="output_file",
                        help="Output NetCDF file with interpolated values for cells in which the interpolation_flag was not false, if not specified the input file is modified")
    parser.add_argument("--areas", action="store", type=str, default="",
                        help="KML file containing the areas that will be interpolated.")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    parser.add_argument("--interpolation_flag_var", action="store", type=str, default="interpolation_flag",
                        help="Name of the variable storing the per-cell interpolation flag in the input file")
    parser.add_argument("--method", action="store", type=str, default="sobolev",
                        help="Name of the inpainting method to use. Available: sobolev, tv, ccst, amle")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                        help="Verbosity flag, activate it to have feedback of the current steps of the process in the command line")
    parser.add_argument("--interpolate_missing_values", action="store_true", default=False,
                        help="Missing value flag, activate it to interpolate missing values instead of re interpolate previously interpolated values")
    parser.add_argument("-s", "--show", action="store_true", dest="show", default=False,
                        help="Show interpolation problem and results on screen")
    parser.add_argument("--mgs_levels", action="store", dest="mgs_levels", default=1, type=int,
                        help="If larger than 1, the PDE will be solved using a Multigrid Solver with the number of levels specified in this parameter")
    parser.add_argument("--init_with", action="store", dest="init_with", default="zeros", type=str,
                        help="Indicates how to initialize the unknown values before inpainting. If using a Multi-Grid Solver, the initialization will only happen at the deepest level of the pyramid. Available options: zeros (fill with zeros), linear (linear interpolation), sobolev (use the sobolev inpainter)")
    parser.add_argument("--convolver", action="store", dest="convolver", default="opencv", type=str,
                        help="Convolution implementation to use. Available: 'scipy-signal', 'scipy-ndimage', 'opencv', 'masked', 'masked-parallel'")
    parser.add_argument("--debug_dir", action="store", dest="debug_dir", default="", type=str,
                        help="If set, debugging information will be stored in this directory (useful to visualize the inpainting progress)")
    param = parser.parse_args(args)

    return param

# Main function
if __name__ == "__main__":
    inpaint(parse_args())