# Copyright (c) 2024 Coronis Computing S.L. (Spain)
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
# Inpainting methods
from heightmap_interpolation.inpainting.sobolev_inpainter import SobolevInpainter
from heightmap_interpolation.inpainting.tv_inpainter import TVInpainter
from heightmap_interpolation.inpainting.ccst_inpainter import CCSTInpainter
from heightmap_interpolation.inpainting.taichi_fd_pde_inpainter import TaichiFDPDEInpainter
from heightmap_interpolation.inpainting.amle_inpainter import AMLEInpainter
from heightmap_interpolation.inpainting.opencv_inpainter import OpenCVInpainter
from heightmap_interpolation.inpainting.opencv_inpainter import OpenCVXPhotoInpainter
from heightmap_interpolation.inpainting.exemplar_based_inpainter import ExemplarBasedInpainter

# Common functions to use in the apps main functions

def add_common_fd_pde_inpainters_args(parser):
    """Adds to the ArgumentParser parser the set of options common to all FD-PDE inpainting methods"""

    # The following two commented parameters are common... but with different default values!
    # parser.add_argument("--update_step_size", default=0.01, help="Update step size")
    # parser.add_argument("--term_thres", default=0.01,
    #                              help="If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop")
    parser.add_argument("--term_criteria", type=str, default='absolute_percent', help="The termination criteria to use. Available: 'relative': stop if the relative change between the inpainted elevations in the current and a previous step is smaller than this value. " +
                                                                                     "'absolute': stop if all cells absolute change between the inpainted elevations in the current and a previous step is smaller than this value. " +
                                                                                     "'absolute_percent' (default): stop if all cells absolute change between the inpainted elevations in the current and a previous step is smaller than this value multiplied by the absolute range of depths in the dataset (i.e., the absolute value is range_depths * absolute_change_percent).")
    parser.add_argument("--term_check_iters", type=int, default=1000, help="Number of iterations in the optimization after which we will check for the termination condition")    
    parser.add_argument("--max_iters", type=int, default=1000000, help="Maximum number of iterations in the optimization.")
    parser.add_argument("--relaxation", type=float, default=0, help="Set to > 1 to perform over-relaxation at each iteration")
    parser.add_argument("--backend", type=str, default="cpu", help="The desired backend where computations should take place. If the requested backend is not available in the machine, will fallback to 'cpu'. Options: 'cpu', 'gpu'")
    parser.add_argument("--ti_arch", type=str, default="gpu", help="When '--backend' is 'gpu', this parameter sets the actual GPU architecture to use. Available options: 'cpu' (i.e., runs the GPU implementation in the CPU), 'gpu', 'cuda', 'vulkan', 'metal'")
    # The following parameter gest its value from "verbose" global argument
    # parser.add_argument("--print_progress", action="store_true",
    #                              help="Flag indicating if some info about the optimization progress should be printed on screen")
    parser.add_argument("--print_progress_iters", type=int, default=1000, help="If set to > 0, the optimization progress will be shown after this number of iterations")
    parser.add_argument("--mgs_levels", type=int, default=1, help="Levels of the Multi-grid solver. I.e., number of levels of detail used in the solving pyramid")
    parser.add_argument("--mgs_min_res", type=int, default=100, help="If during the construction of the pyramid of the Multi-Grid Solver one of the dimensions of the grid drops below this size, the pyramid construction will stop at that level")
    parser.add_argument("--init_with", type=str, default="nearest", help="Initialize the unknown values to inpaint using a simple interpolation function. If using a MGS, this will be used with the lowest level on the pyramid. Available initializers: 'nearest' (default), 'linear', 'cubic', 'harmonic'")
    parser.add_argument("--convolver", type=str, default="opencv", help="The convolution method to use. Available: 'opencv' (default),'scipy-signal', 'scipy-ndimage', 'masked', 'masked-parallel'")
    parser.add_argument("--debug_dir", action="store", dest="debug_dir", default="", type=str, help="If set, debugging information will be stored in this directory (useful to visualize the inpainting progress)")
    return parser

def get_common_fd_pde_inpainters_params_from_args(params):
    """Gets the set of common parameters/options of all FD-PDE inpainters from the parameters structure derived from ArgumentParser"""
    options = {"update_step_size": params.update_step_size,
               "term_criteria": params.term_criteria,
               "term_check_iters": params.term_check_iters,
               "term_thres": params.term_thres,
               "max_iters": params.max_iters,
               "relaxation": params.relaxation,
               "backend": params.backend,
               "ti_arch": params.ti_arch,
               "print_progress": params.verbose,
               "print_progress_iters": params.print_progress_iters,
               "mgs_levels": params.mgs_levels,
               "mgs_min_res": params.mgs_min_res,
               "init_with": params.init_with,
               "convolver": params.convolver,
               "debug_dir": params.debug_dir}
    return options

def add_inpainting_subparsers(subparsers):
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
                        help="Distance type. Available: euclidean (default), haversine, vincenty")
    parser_rbf.add_argument("--rbf_type", action="store", type=str, default="thinplate",
                        help="RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland")
    parser_rbf.add_argument("--rbf_epsilon", action="store", type=float, default=1,
                        help="Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland")
    parser_rbf.add_argument("--rbf_regularization", action="store", type=float, default=0,
                        help="Regularization scalar to use while creating the RBF interpolant (optional)")
    parser_rbf.add_argument("--rbf_polynomial_degree", action="store", type=int, default=1,
                        help="Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)")

    # Parser for the "pu-rbf" method
    parser_purbf = subparsers.add_parser("purbf", help="Partition of Unity Radial Basis Function interpolant")
    parser_purbf.add_argument("--query_block_size", action="store", type=int, default=1000, help="Apply the interpolant using maximum this number of points at a time to avoid large memory consumption")
    parser_purbf.add_argument("--rbf_distance_type", action="store", type=str, default="euclidean", help="Distance type. Available: euclidean (default), haversine, vincenty")
    parser_purbf.add_argument("--rbf_type", action="store", type=str, default="thinplate", help="RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland")
    parser_purbf.add_argument("--rbf_epsilon", action="store", type=float, default=1, help="Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland")
    parser_purbf.add_argument("--rbf_regularization", action="store", type=float, default=0, help="Regularization scalar to use while creating the RBF interpolant (optional)")
    parser_purbf.add_argument("--rbf_polynomial_degree", action="store", type=int, default=1, help="Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)")
    parser_purbf.add_argument("--pu_overlap", action="store", type=float, default=0.25, help="Overlap factor between circles in neighboring sub-domains in the partition. The radius of a QuadTree cell, computed as half its diagonal, is enlarged by this factor")
    parser_purbf.add_argument("--pu_min_point_in_cell", action="store", type=int, default=1000, help="Minimum number of points in a QuadTree cell")
    parser_purbf.add_argument("--pu_min_cell_size_percent", action="store", type=float, default=0.005, help="Minimum cell size, specified as a percentage [0..1] of the max(width, height) of the query domain")
    parser_purbf.add_argument("--pu_overlap_increment", action="store", type=float, default=0.001, help="If, after creating the QuadTree, a cell contains less than pu_min_point_in_cell, the radius will be iteratively incremented until this condition is satisfied. This parameter specifies how much the radius of a cell increments at each iteration")

    # Parser for the "harmonic" method
    parser_harmonic = subparsers.add_parser("harmonic", help="Harmonic inpainter")
    parser_harmonic.add_argument("--update_step_size", type=float, default=0.2, help="Update step size")
    parser_harmonic.add_argument("--term_thres", type=float, default=1e-5, help="Termination threshold. Its meaning depends on the --term_criteria parameter.")
    parser_harmonic = add_common_fd_pde_inpainters_args(parser_harmonic)

    # Parser for the "tv" method
    parser_tv = subparsers.add_parser("tv", help="Inpainter minimizing Total-Variation (TV) across the 'image'")
    parser_tv.add_argument("--update_step_size", type=float, default=0.225, help="Update step size")
    parser_tv.add_argument("--term_thres", type=float, default=1e-5, help="Termination threshold. Its meaning depends on the --term_criteria parameter.")
    parser_tv = add_common_fd_pde_inpainters_args(parser_tv)
    parser_tv.add_argument("--epsilon", type=float, default=1, help="A small value to be added when computing the norm of the gradients during optimization, to avoid a division by zero")

    # Parser for the "ccst" method
    parser_ccst = subparsers.add_parser("ccst", help="Continous Curvature Splines in Tension (CCST) inpainter")
    parser_ccst.add_argument("--update_step_size", type=float, default=0.01, help="Update step size")
    parser_ccst.add_argument("--term_thres", type=float, default=1e-5, help="Termination threshold. Its meaning depends on the --term_criteria parameter.")
    parser_ccst = add_common_fd_pde_inpainters_args(parser_ccst)
    parser_ccst.add_argument("--tension", type=float, default=0.3, help="Tension parameter weighting the contribution between a harmonic and a biharmonic interpolation (see the docs and the original reference for more details)")

    # Parser for the "amle" method
    parser_amle = subparsers.add_parser("amle", help="Absolutely Minimizing Lipschitz Extension (AMLE) inpainter")
    parser_amle.add_argument("--update_step_size", type=float, default=0.01, help="Update step size")
    parser_amle.add_argument("--term_thres", type=float, default=1e-5, help="Termination threshold. Its meaning depends on the --term_criteria parameter.")
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

    # Parser for the "ebi" (Exemplar-Based Inpainter) method
    parser_ebi = subparsers.add_parser("ebi", help="Exemplar-based inpainter")
    parser_ebi.add_argument("--patch_size", type=int, default=9, help="Size of the inpainting patch.")
    parser_ebi.add_argument("--search_original_source_only", action='store_true', help="If true, just the original source image - mask will be searched for inpainting patches. Otherwise, the growing inpainting area will also be taken into account.")
    # search_color_space (str, optional): Color space to use when searching for the next best filler patch. Options available: "bgr", "hsv", "lab", "gray". In case gray is selected, the input image must also be grayscale. Defaults to "bgr".
    parser_ebi.add_argument("--plot_progress", action='store_true', help="Activates the plotting of the inpainting process (internal of the inpainter library)")
    parser_ebi.add_argument("--out_progress_dir", type=str, help="Set to a directory to get the same output as with --plot_progress, but stored in files.")
    parser_ebi.add_argument("--show_progress_bar", action='store_true', help="Activates the progress bar (internal of the inpainter library).")
    parser_ebi.add_argument("--patch_preference", type=str, help="When more than a patch has the same similarity score, this parameter selects which one to choose. Available: \"any\", \"closest\", \"random\".")

def create_inpainter_from_params(params):
    if (params.subparser_name.lower() != "navier-stokes" and 
        params.subparser_name.lower() != "telea" and 
        params.subparser_name.lower() != "shiftmap" and
        params.subparser_name.lower() != "ebi"):
        options = get_common_fd_pde_inpainters_params_from_args(params)
        if options["backend"] == "gpu" and (params.subparser_name.lower() != "harmonic" and params.subparser_name.lower() != "ccst"):
            raise ValueError("Currently the GPU backend is only available for harmonic and ccst methods.")    
    if params.subparser_name.lower() == "harmonic":        
        if options["backend"] == "gpu":
            options["method"] = "harmonic"
            inpainter = TaichiFDPDEInpainter(**options)
        else:
            inpainter = SobolevInpainter(**options)
    elif params.subparser_name.lower() == "tv":
        options["epsilon"] = params.epsilon
        inpainter = TVInpainter(**options)
    elif params.subparser_name[0:4].lower() == "ccst":
        options["tension"] = params.tension                
        if options["backend"] == "gpu":
            options["method"] = "ccst"
            inpainter = TaichiFDPDEInpainter(**options)
        else:                    
            inpainter = CCSTInpainter(**options)
    elif params.subparser_name.lower() == "amle":
        options["convolve_in_1d"] = params.convolve_in_1d
        inpainter = AMLEInpainter(**options)
    elif params.subparser_name.lower() == "navier-stokes":
        inpainter = OpenCVInpainter(method="navier-stokes", radius=params.radius)
    elif params.subparser_name.lower() == "telea":
        inpainter = OpenCVInpainter(method="telea", radius=params.radius)
    elif params.subparser_name.lower() == "shiftmap":
        inpainter = OpenCVXPhotoInpainter(method="shiftmap")
    elif params.subparser_name.lower() == "ebi":
        options = {
            "patch_size": params.patch_size,
            "search_original_source_only": params.search_original_source_only,
            "plot_progress": params.plot_progress,
            "out_progress_dir": params.out_progress_dir,
            "show_progress_bar": params.show_progress_bar,
            "patch_preference": params.patch_preference}
        inpainter = ExemplarBasedInpainter(**options)

    return inpainter