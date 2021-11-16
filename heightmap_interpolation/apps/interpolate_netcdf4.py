#!/usr/bin/env python3

# Copyright (c) 2021 Coronis Computing S.L. (Spain)
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

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from timeit import default_timer as timer
from heightmap_interpolation.misc.conditional_print import ConditionalPrint
from heightmap_interpolation.apps.netcdf_data_io import load_interpolation_input_data, write_interpolation_results
# All interpolation methods
from heightmap_interpolation.interpolants.nearest_neighbor_interpolant import NearestNeighborInterpolant
from heightmap_interpolation.interpolants.linear_interpolant import LinearInterpolant
from heightmap_interpolation.interpolants.cubic_interpolant import CubicInterpolant
from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant
from heightmap_interpolation.interpolants.quad_tree_pu_rbf_interpolant import QuadTreePURBFInterpolant
from heightmap_interpolation.inpainting.sobolev_inpainter import SobolevInpainter
from heightmap_interpolation.inpainting.tv_inpainter import TVInpainter
from heightmap_interpolation.inpainting.ccst_inpainter import CCSTInpainter
from heightmap_interpolation.inpainting.amle_inpainter import AMLEInpainter
from heightmap_interpolation.inpainting.opencv_inpainter import OpenCVInpainter
from heightmap_interpolation.inpainting.opencv_inpainter import OpenCVXPhotoInpainter


def add_common_fd_pde_inpainters_args(parser):
    """Adds to the ArgumentParser parser the set of options common to all FD-PDE inpainting methods"""

    # The following two commented parameters are common... but with different default values!
    # parser.add_argument("--update_step_size", default=0.01, help="Update step size")
    # parser.add_argument("--rel_change_tolerance", default=0.01,
    #                              help="If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop")
    parser.add_argument("--rel_change_iters", type=int, default=1000, help="Number of iterations in the optimization after which we will check if the relative tolerance is below the threshold")
    parser.add_argument("--max_iters", type=int, default=1000000, help="Maximum number of iterations in the optimization.")
    parser.add_argument("--relaxation", type=float, default=0, help="Set to >1 to perform over-relaxation at each iteration")
    # The following parameter gest its value from "verbose" global argument
    # parser.add_argument("--print_progress", action="store_true",
    #                              help="Flag indicating if some info about the optimization progress should be printed on screen")
    parser.add_argument("--print_progress_iters", type=int, default=1000, help="If '--print_progress True', the optimization progress will be shown after this number of iterations")
    parser.add_argument("--mgs_levels", type=int, default=1, help="Levels of the Multi-grid solver. I.e., number of levels of detail used in the solving pyramid")
    parser.add_argument("--mgs_min_res", type=int, default=100, help="If during the construction of the pyramid of the Multi-Grid Solver one of the dimensions of the grid drops below this size, the pyramid construction will stop at that level")
    parser.add_argument("--init_with", type=str, default="nearest", help="Initialize the unknown values to inpaint using a simple interpolation function. If using a MGS, this will be used with the lowest level on the pyramid. Available initializers: 'nearest' (default), 'linear', 'cubic', 'sobolev'")
    parser.add_argument("--convolver", type=str, default="opencv", help="The convolution method to use. Available: 'opencv' (default),'scipy-signal', 'scipy-ndimage', 'masked', 'masked-parallel'")
    parser.add_argument("--debug_dir", action="store", dest="debug_dir", default="", type=str, help="If set, debugging information will be stored in this directory (useful to visualize the inpainting progress)")
    return parser


def get_common_fd_pde_inpainters_params_from_args(params):
    """Gets the set of common parameters/options of all FD-PDE inpainters from the parameters structure derived from ArgumentParser"""
    options = {"update_step_size": params.update_step_size,
               "rel_change_iters": params.rel_change_iters,
               "rel_change_tolerance": params.rel_change_tolerance,
               "max_iters": params.max_iters,
               "relaxation": params.relaxation,
               "print_progress": params.verbose,
               "print_progress_iters": params.print_progress_iters,
               "mgs_levels": params.mgs_levels,
               "mgs_min_res": params.mgs_min_res,
               "init_with": params.init_with,
               "convolver": params.convolver,
               "debug_dir": params.debug_dir}
    return options


def interpolate(params):
    condp = ConditionalPrint(params.verbose)

    # Load the data of the interpolation problem
    if params.verbose:
        condp.print("- Loading data...", end='', flush=True)
        ts = timer()
    lats_mat, lons_mat, elevation, mask_int, mask_ref, work_areas = load_interpolation_input_data(params.input_file,
                                                                                                  params.elevation_var,
                                                                                                  params.interpolation_flag_var,
                                                                                                  params.areas)
    elevation_int = np.copy(elevation)
    if params.verbose:
        te = timer()
        condp.print(" done, {:.2f} sec.".format(te-ts))

    # Show a bit of information regarding the interpolation problem (percentage of missing data to interpolate w.r.t. the full image)
    if params.verbose:
        condp.print("- Summary of input data:")
        condp.print("    - Elevation grid has a size of {:d}x{:d} cells".format(elevation.shape[0], elevation.shape[1]))
        if params.areas:
            condp.print("    - Data will be interpolated just at the user-defined areas")
        else:
            total_cells = elevation.shape[0] * elevation.shape[1]
            num_cells_to_interpolate = np.count_nonzero(mask_int)
            interp_percent = (num_cells_to_interpolate / total_cells) * 100
            condp.print("    - Cells to interpolate represent a {:.2f}% of the image:".format(interp_percent))
            condp.print("        - Total cells = {:d}".format(total_cells)),
            condp.print("        - Number of reference cells = {:d}".format(total_cells-num_cells_to_interpolate))
            condp.print("        - Number of cells to interpolate = {:d}".format(num_cells_to_interpolate))

    for i in range(work_areas.shape[2]):
        # Get the current working area
        cur_work_area = work_areas[:, :, i]

        # --- Scattered data interpolation ---
        scattered_methods = ['nearest', 'linear', 'cubic', 'rbf', 'purbf']
        if params.subparser_name.lower() in scattered_methods:
            # Get the reference points from the current working area
            cur_mask_ref = np.logical_and(mask_ref, cur_work_area)
            cur_mask_int = np.logical_and(mask_int, cur_work_area)

            # Cast the matrices to a set of "scattered" data points and references
            lats_ref = lats_mat[cur_mask_ref]
            lons_ref = lons_mat[cur_mask_ref]
            elevation_ref = elevation[cur_mask_ref]
            lats_int = lats_mat[cur_mask_int]
            lons_int = lons_mat[cur_mask_int]

            # Show a bit of information regarding the current area interpolation problem (percentage of missing data to interpolate w.r.t. the full image)
            if params.verbose and params.areas:
                condp.print("- Interpolating area {:d}/{:d}:".format(i+1, work_areas.shape[2]))
                condp.print("    - Number of reference cells = {:d}".format(len(lats_ref)))
                condp.print("    - Number of cells to interpolate = {:d}".format(len(lats_int)))

            # Create the interpolant
            if params.verbose:
                endl = '\n' if params.subparser_name.lower() == "purbf" else ''
                condp.print("- Creating the interpolant...", end=endl)
                ts = timer()
            if params.subparser_name.lower() == "nearest":
                interpolant = NearestNeighborInterpolant(lons_ref, lats_ref, elevation_ref,
                                                         params.rescale)
            elif params.subparser_name.lower() == "linear":
                interpolant = LinearInterpolant(lons_ref, lats_ref, elevation_ref,
                                                params.fill_value, params.rescale)
            elif params.subparser_name.lower() == "cubic":
                interpolant = CubicInterpolant(lons_ref, lats_ref, elevation_ref,
                                               params.fill_value, params.tolerance, params.max_iters, params.rescale)
            elif params.subparser_name.lower() == "rbf":
                # Warn the user if the PURBF is better suited for this problem
                if len(lats_ref) > 10000:
                    print("\n!!!WARNING!!! You are trying to build a RBF intepolant from a large number of data points, and this may require large computational cost and memory consumption.\nPlease consider using the PURBF interpolant instead!")
                interpolant = RBFInterpolant(lons_ref, lats_ref, elevation_ref,
                                             rbf_type=params.rbf_type,
                                             distance_type=params.rbf_distance_type,
                                             epsilon=params.rbf_epsilon,
                                             regularization=params.rbf_regularization,
                                             polynomial_degree=params.rbf_polynomial_degree)
            elif params.subparser_name.lower() == "purbf":
                # Compute the query domain to be that of the points to interpolate
                minX = np.min(lons_int)
                maxX = np.max(lons_int)
                minY = np.min(lats_int)
                maxY = np.max(lats_int)
                w = maxX - minX
                h = maxY - minY
                wh = max(w, h)
                domain = [np.min(lons_int), np.min(lats_int), wh]
                interpolant = QuadTreePURBFInterpolant(lons_ref, lats_ref, elevation_ref,
                                                       domain=domain,
                                                       min_points_in_cell=params.pu_min_point_in_cell,
                                                       overlap=params.pu_overlap,
                                                       overlap_increment=params.pu_overlap_increment,
                                                       min_cell_size_percent=params.pu_min_cell_size_percent,
                                                       rbf_type=params.rbf_type,
                                                       distance_type=params.rbf_distance_type,
                                                       epsilon=params.rbf_epsilon,
                                                       regularization=params.rbf_regularization,
                                                       polynomial_degree=params.rbf_polynomial_degree)
            if params.verbose:
                te = timer()
                condp.print(" done, {:.2f} sec.".format(te - ts))

            # Interpolate at the grid points
            if params.verbose:
                condp.print("- Applying the interpolant at the query points...", end=endl)
                ts = timer()
            if params.subparser_name.lower() != "rbf" and params.subparser_name.lower() != "purbf":
                zi = interpolant(lons_int, lats_int)
            else:
                # For RBF and PURBF, apply the interpolant in blocks to avoid large memory consumption

                # Divide the data into blocks
                query_block_size = params.query_block_size
                zi = np.zeros(lons_int.shape)
                num_int = np.sum(mask_int)
                num_blocks = math.ceil(num_int/query_block_size)
                block_start = 0
                block_end = min([num_int, query_block_size])

                # Interpolate
                for i in range(num_blocks):
                    message = "    - Querying block {}/{}".format(i + 1, num_blocks)
                    condp.print(message)
                    # condp.backspace(len(message))

                    zi[block_start:block_end] = interpolant(lons_int[block_start:block_end],
                                                            lats_int[block_start:block_end])
                    block_end = min([block_end + query_block_size, num_int])
                    block_start = block_start + query_block_size
            if params.verbose:
                te = timer()
                condp.print(" done, {:.2f} sec.".format(te - ts))

            # Put the interpolated values back into the elevation matrix
            elevation_int[cur_mask_int] = zi

        # --- Gridded data interpolation/inpainting ---
        gridded_methods = ['harmonic', 'tv', 'ccst', 'amle', 'navier-stokes', 'telea', 'shiftmap']
        if params.subparser_name.lower() in gridded_methods:
            # if params.areas:
            # Get the bounding box of the current working area (inpainters work on full 2D grids...)
            rows = np.any(cur_work_area, axis=1)
            cols = np.any(cur_work_area, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Extract this region from the image
            cur_inpaint_mask = np.copy(~mask_int[rmin:rmax+1, cmin:cmax+1]) # Inpainting mask (inverse of mask_int by our internal convention)
            cur_elevation = np.copy(elevation[rmin:rmax+1, cmin:cmax+1])
            # However, we will not interpolate those points in the rectangular region that do not fall within the marked area
            cur_inpaint_mask = np.logical_or(cur_inpaint_mask, ~cur_work_area[rmin:rmax+1, cmin:cmax+1])

            cur_elevation[np.isnan(cur_elevation)] = 0 # Initializer, as well as boundary conditions when the areas to interpolate do not cover all the cells with unknown data

            if params.verbose and params.areas:
                condp.print("- Interpolating area {:d}/{:d}:".format(i+1, work_areas.shape[2]))
                condp.print("    - Number of reference cells = {:d}".format(np.count_nonzero(cur_inpaint_mask)))
                condp.print("    - Number of cells to interpolate = {:d}".format(np.count_nonzero(~cur_inpaint_mask)))

            # Create the inpainter
            if params.subparser_name.lower() == "harmonic":
                options = get_common_fd_pde_inpainters_params_from_args(params)
                inpainter = SobolevInpainter(**options)
            elif params.subparser_name.lower() == "tv":
                options = get_common_fd_pde_inpainters_params_from_args(params)
                options["epsilon"] = params.epsilon
                inpainter = TVInpainter(**options)
            elif params.subparser_name.lower() == "ccst":
                options = get_common_fd_pde_inpainters_params_from_args(params)
                options["tension"] = params.tension
                inpainter = CCSTInpainter(**options)
            elif params.subparser_name.lower() == "amle":
                options = get_common_fd_pde_inpainters_params_from_args(params)
                options["convolve_in_1d"] = params.convolve_in_1d
                inpainter = AMLEInpainter(**options)
            elif params.subparser_name.lower() == "navier-stokes":
                inpainter = OpenCVInpainter(method="navier-stokes", radius=params.radius)
            elif params.subparser_name.lower() == "telea":
                inpainter = OpenCVInpainter(method="telea", radius=params.radius)
            elif params.subparser_name.lower() == "shiftmap":
                inpainter = OpenCVXPhotoInpainter(method="shiftmap")
            # Inpaint!
            if params.verbose:
                ts = timer()
            cur_elevation_int = inpainter.inpaint(cur_elevation, cur_inpaint_mask)
            if params.verbose:
                te = timer()
                condp.print("- Inpainting took a total of {:.2f} sec.".format(te - ts))

            # "Paste" the results into the original elevation matrix
            # elevation_int[rmin:rmax+1, cmin:cmax+1] = cur_elevation_int
            elevation_slice = elevation_int[rmin:rmax + 1, cmin:cmax + 1] # Do not copy! we want to refer to that part in elevation_int matrix
            elevation_slice[~cur_inpaint_mask] = cur_elevation_int[~cur_inpaint_mask] # Only modify the inpainted part! (This way we preserve "unknown"/NaN values in areas we did not interpolate

    # Write the results
    if params.output_file:
        condp.print("- Writing the results to disk")
        write_interpolation_results(params.input_file, params.output_file,
                                    elevation_int, mask_int,
                                    params.elevation_var, params.interpolation_flag_var, params.areas)

    # Show results
    if params.show:
        condp.print("- Showing results")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        images = [elevation, elevation_int]
        titles = ['Original', 'Interpolated']
        for (ax, image, title) in zip(axes, images, titles):
            ax.imshow(image)
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        plt.show()


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Interpolate elevation data in a SeaDataNet_1.0 CF1.6-compliant netCDF4 file")
    # Create a sub-parser for each possible interpolator, with its own options
    subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')
    parser.add_argument("input_file", action="store", type=str,
                        help="Input NetCDF file")
    parser.add_argument("-o","--output_file", dest="output_file", action="store", type=str,
                        help="Output NetCDF file with interpolated values")
    parser.add_argument("--areas", action="store", type=str, default=None,
                        help="KML file containing the areas that will be interpolated.")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    parser.add_argument("--interpolation_flag_var", action="store", type=str, default=None,
                        help="Name of the variable storing the per-cell interpolation flag in the input file (0 == known value, 1 == interpolated/to interpolate cell). If not set, it will interpolate the locations in the elevation variable containing an invalid (NaN) value.")
    parser.add_argument("-v, --verbose", action="store_true", dest="verbose", default=False,
                        help="Verbosity flag, activate it to have feedback of the current steps of the process in the command line")
    parser.add_argument("-s, --show", action="store_true", dest="show", default=False,
                        help="Show interpolation problem and results on screen")

    # Parser for the "nearest" method
    parser_nearest = subparsers.add_parser("nearest", help="Nearest-neighbor interpolator")
    parser_nearest.add_argument("--rescale", action="store_true", dest="rescale",
                               help="Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.")

    # Parser for the "linear" method
    parser_linear = subparsers.add_parser("linear", help="Linear interpolator")
    parser_linear.add_argument("--fill_value", type=float, default=np.nan, help="Value used to fill in for requested points outside of the convex hull of the input points. If not provided, the default is NaN.")
    parser_linear.add_argument("--rescale", action="store_true", dest="rescale",
                               help="Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.")

    # Parser for the "cubic" method
    parser_cubic = subparsers.add_parser("cubic", help="Piecewise cubic, C1 smooth, curvature-minimizing (Clough-Tocher) nterpolator")
    parser_cubic.add_argument("--fill_value", type=float, default=np.nan,
                               help="Value used to fill in for requested points outside of the convex hull of the input points. If not provided, the default is NaN.")
    parser_cubic.add_argument("--rescale", action="store_true", dest="rescale",
                               help="Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.")
    parser_cubic.add_argument("--tolerance", type=float, default=1e-6, help="Absolute/relative tolerance for gradient estimation.")
    parser_cubic.add_argument("--max_iters", type=int, default=400, help="Maximum number of iterations in gradient estimation.")

    # Parser for the "rbf" method
    parser_rbf = subparsers.add_parser("rbf", help="Radial Basis Function interpolant")
    parser_rbf.add_argument("--query_block_size", action="store", type=int, default=1000, help="Apply the interpolant using maximum this number of points at a time to avoid large memory consumption")
    parser_rbf.add_argument("--rbf_distance_type", action="store", type=str, default="euclidean",
                        help="Distance type. Available: euclidean, haversine, vincenty(default)")
    parser_rbf.add_argument("--rbf_type", action="store", type=str, default="thinplate",
                        help="RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland")
    parser_rbf.add_argument("--rbf_epsilon", action="store", type=float, default=1,
                        help="Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland")
    parser_rbf.add_argument("--rbf_regularization", action="store", type=float, default=0,
                        help="Regularization scalar to use in the RBF (optional)")
    parser_rbf.add_argument("--rbf_polynomial_degree", action="store", type=int, default=1,
                        help="Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)")

    # Parser for the "pu-rbf" method
    parser_purbf = subparsers.add_parser("purbf", help="Partition of Unity Radial Basis Function interpolant")
    parser_purbf.add_argument("--query_block_size", action="store", type=int, default=1000, help="Apply the interpolant using maximum this number of points at a time to avoid large memory consumption")
    parser_purbf.add_argument("--rbf_distance_type", action="store", type=str, default="euclidean", help="Distance type. Available: euclidean, haversine, vincenty(default)")
    parser_purbf.add_argument("--rbf_type", action="store", type=str, default="thinplate", help="RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland")
    parser_purbf.add_argument("--rbf_epsilon", action="store", type=float, default=1, help="Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland")
    parser_purbf.add_argument("--rbf_regularization", action="store", type=float, default=0, help="Regularization scalar to use in the RBF (optional)")
    parser_purbf.add_argument("--rbf_polynomial_degree", action="store", type=int, default=1, help="Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)")
    parser_purbf.add_argument("--pu_overlap", action="store", type=float, default=0.25, help="(Just if PU is used) Overlap factor between circles in neighboring sub-domains in the partition. The radius of a QuadTree cell, computed as half its diagonal, is enlarged by this factor")
    parser_purbf.add_argument("--pu_min_point_in_cell", action="store", type=int, default=1000, help="(Just if PU is used) Minimum number of points in a QuadTree cell")
    parser_purbf.add_argument("--pu_min_cell_size_percent", action="store", type=float, default=0.005, help="(Just if PU is used) Minimum cell size, specified as a percentage [0..1] of the max(width, height) of the query domain")
    parser_purbf.add_argument("--pu_overlap_increment", action="store", type=float, default=0.001, help="(Just if PU is used) If, after creating the QuadTree, a cell contains less than pu_min_point_in_cell, the radius will be iteratively incremented until this condition is satisfied. This parameter specifies how much the radius of a cell increments at each iteration")

    # Parser for the "harmonic" method
    parser_harmonic = subparsers.add_parser("harmonic", help="Harmonic inpainter")
    parser_harmonic.add_argument("--update_step_size", type=float, default=0.2, help="Update step size")
    parser_harmonic.add_argument("--rel_change_tolerance", type=float, default=1e-5, help="If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop")
    parser_harmonic = add_common_fd_pde_inpainters_args(parser_harmonic)

    # Parser for the "tv" method
    parser_tv = subparsers.add_parser("tv", help="Inpainter minimizing Total-Variation (TV) across the 'image'")
    parser_tv.add_argument("--update_step_size", type=float, default=0.225, help="Update step size")
    parser_tv.add_argument("--rel_change_tolerance", type=float, default=1e-5, help="If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop")
    parser_tv = add_common_fd_pde_inpainters_args(parser_tv)
    parser_tv.add_argument("--epsilon", type=float, default=1, help="A small value to be added when computing the norm of the gradients during optimization, to avoid a division by zero")

    # Parser for the "ccst" method
    parser_ccst = subparsers.add_parser("ccst", help="Continous Curvature Splines in Tension (CCST) inpainter")
    parser_ccst.add_argument("--update_step_size", type=float, default=0.01, help="Update step size")
    parser_ccst.add_argument("--rel_change_tolerance", type=float, default=1e-8, help="If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop")
    parser_ccst = add_common_fd_pde_inpainters_args(parser_ccst)
    parser_ccst.add_argument("--tension", type=float, default=0.3, help="Tension parameter weighting the contribution between a harmonic and a biharmonic interpolation (see the docs and the original reference for more details)")

    # Parser for the "amle" method
    parser_amle = subparsers.add_parser("amle", help="Absolutely Minimizing Lipschitz Extension (AMLE) inpainter")
    parser_amle.add_argument("--update_step_size", type=float, default=0.01, help="Update step size")
    parser_amle.add_argument("--rel_change_tolerance", type=float, default=1e-7, help="If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop")
    parser_amle = add_common_fd_pde_inpainters_args(parser_amle)
    parser_amle.add_argument("--convolve_in_1d", action="store_true", help="Perform 1D convolutions instead of using the 2D convolution indicated in --convolver")

    # Parser for the "navier-stokes" method
    parser_ns = subparsers.add_parser("navier-stokes", help="OpenCV's Navier-Stokes inpainter")
    parser_ns.add_argument("--radius", type=int, default=25, help="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm")

    # Parser for the "telea" method
    parser_ns = subparsers.add_parser("telea", help="OpenCV's Telea inpainter")
    parser_ns.add_argument("--radius", type=int, default=25, help="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm")

    # Parser for the "shiftmap" method
    parser_shiftmap = subparsers.add_parser("shiftmap", help="OpenCV's xphoto module's Shiftmap inpainter")

    return parser.parse_args(args)

# Main function
if __name__ == "__main__":
    interpolate(parse_args())