import argparse
import matplotlib.pyplot as plt
import numpy as np
from heightmap_interpolation.misc.conditional_print import ConditionalPrint
import cvxpy as cp
from timeit import default_timer as timer
from heightmap_interpolation.apps.common import load_data, write_results
from heightmap_interpolation.inpainting.sobolev_inpainter import SobolevInpainter
from heightmap_interpolation.inpainting.tv_inpainter import TVInpainter
from heightmap_interpolation.inpainting.ccst_inpainter import CCSTInpainter
from heightmap_interpolation.inpainting.amle_inpainter import AMLEInpainter
from heightmap_interpolation.inpainting.opencv_inpainter import OpenCVInpainter
# from heightmap_interpolation.inpainting.transport_inpainter import TransportInpainter
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

    # Collect common parameters
    params_dict = {"update_step_size": 0.01,
                   "rel_change_tolerance": 1e-8,
                   "max_iters": 1e8,
                   "relaxation": 0,
                   "print_progress": True,
                   "print_progress_iters": 1000}

    # Sobolev inpainter
    condp.print("- Inpainting")
    if param.method.lower() == "sobolev":
        condp.print("   - Using Sobolev inpainter")
        params_dict["update_step_size"] = 0.8/4
        params_dict["rel_change_tolerance"] = 1e-5
        params_dict["max_iters"] = 1e5
        inpainter = SobolevInpainter(**params_dict)
        inpaint_mask = ~mask_int
    elif param.method.lower() == "tv":
        params_dict["epsilon"] = 1
        params_dict["update_step_size"] = .9*params_dict["epsilon"]/4
        params_dict["rel_change_tolerance"] = 1e-5
        params_dict["max_iters"] = 1e5
        params_dict["show_progress"] = False
        inpainter = TVInpainter(**params_dict)
        inpaint_mask = ~mask_int
    elif param.method.lower() == "ccst":
        params_dict["update_step_size"] = 0.01
        params_dict["rel_change_tolerance"] = 1e-8
        params_dict["max_iters"] = 1e8
        params_dict["relaxation"] = 1.4
        params_dict["tension"] = 0.3
        inpainter = CCSTInpainter(**params_dict)
        inpaint_mask = ~mask_int
    elif param.method.lower() == "amle":
        params_dict["update_step_size"] = 0.01
        params_dict["rel_change_tolerance"] = 1e-5
        params_dict["max_iters"] = 1e8
        inpainter = AMLEInpainter(**params_dict)
        inpaint_mask = ~mask_int
    elif param.method.lower() == "navier-stokes":
        # Convert mask to opencv's inpaint function expected format
        inpainter = OpenCVInpainter(method="navier-stokes", radius=25)
        inpaint_mask = mask_int.astype(np.uint8)  # convert to an unsigned byte
        inpaint_mask *= 255
    elif param.method.lower() == "telea":
        inpainter = OpenCVInpainter(method="telea", radius=5)
        inpaint_mask = mask_int.astype(np.uint8)  # convert to an unsigned byte
        inpaint_mask *= 255
    elif param.method.lower() == "transport":
        params_dict["update_step_size"] = 0.05
        params_dict["rel_change_tolerance"] = 1e-5
        params_dict["max_iters"] = 50
        params_dict["iters_inpainting"] = 40
        params_dict["iters_anisotropic"] = 2
        params_dict["epsilon"] = 1e-10
        # inpainter = BertalmioInpainter(**params_dict)
        # inpaint_mask = scipy.float64((~mask_int))
    else:
        print("[ERROR] The required method (" + param.method + ") is unknown. Available options: sobolev, ccst")

    if param.verbose:
        ts = timer()
    elevation_inpainted = inpainter.inpaint(elevation, inpaint_mask)
    if param.verbose:
        te = timer()
        condp.print(condp.print("done, {:.2f} sec.".format(te - ts)))

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
    parser.add_argument("--output_file", action="store", type=str,
                        help="Output NetCDF file with interpolated values for cells in which the interpolation_flag was not false, if not specified the input file is modified")
    parser.add_argument("--areas", action="store", type=str, default="",
                        help="KML file containing the areas that will be interpolated.")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    parser.add_argument("--interpolation_flag_var", action="store", type=str, default="interpolation_flag",
                        help="Name of the variable storing the per-cell interpolation flag in the input file")
    parser.add_argument("--method", action="store", type=str, default="sobolev",
                        help="Name of the inpainting method to use. Available: sobolev, ccst")
    parser.add_argument("-v, --verbose", action="store_true", dest="verbose", default=False,
                        help="Verbosity flag, activate it to have feedback of the current steps of the process in the command line")
    parser.add_argument("--interpolate_missing_values", action="store_true", default=False,
                        help="Missing value flag, activate it to interpolate missing values instead of re interpolate previously interpolated values")
    parser.add_argument("-s, --show", action="store_true", dest="show", default=False,
                        help="Show interpolation problem and results on screen")
    param = parser.parse_args(args)

    return param

# Main function
if __name__ == "__main__":
    inpaint(parse_args())