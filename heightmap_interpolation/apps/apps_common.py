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
import argparse
import math
from timeit import default_timer as timer

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from heightmap_interpolation.inpainting.amle_inpainter import AMLEInpainter
from heightmap_interpolation.inpainting.ccst_inpainter import CCSTInpainter
from heightmap_interpolation.inpainting.exemplar_based_inpainter import (
    ExemplarBasedInpainter,
)
from heightmap_interpolation.inpainting.opencv_inpainter import (
    OpenCVInpainter,
    OpenCVXPhotoInpainter,
)

# Inpainting methods
from heightmap_interpolation.inpainting.sobolev_inpainter import SobolevInpainter
from heightmap_interpolation.inpainting.taichi_fd_pde_inpainter import (
    TaichiFDPDEInpainter,
)
from heightmap_interpolation.inpainting.tv_inpainter import TVInpainter
from heightmap_interpolation.interpolants.ams_interpolant import AMSInterpolant
from heightmap_interpolation.interpolants.cubic_interpolant import CubicInterpolant
from heightmap_interpolation.interpolants.linear_interpolant import LinearInterpolant
from heightmap_interpolation.interpolants.nearest_neighbor_interpolant import (
    NearestNeighborInterpolant,
)
from heightmap_interpolation.interpolants.quad_tree_pu_rbf_interpolant import (
    QuadTreePURBFInterpolant,
)
from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant

# Common functions to use in the apps main functions

SCATTERED_METHODS = [
    "nearest",
    "linear",
    "cubic",
    "rbf",
    "purbf",
    "ams",
]
EXPERIMENTAL_SCATTERED_METHODS = ["mlp"]
GRIDDED_METHODS = [
    "harmonic",
    "tv",
    "ccst",
    "amle",
    "navier-stokes",
    "telea",
]
EXPERIMENTAL_GRIDDED_METHODS = ["shiftmap", "ebi"]


def get_available_scattered_methods():
    methods = list(SCATTERED_METHODS)
    if experimental_features_available():
        methods += EXPERIMENTAL_SCATTERED_METHODS
    if pygmt_available():
        methods += ["gmt_surface"]
    return methods


def get_available_gridded_methods():
    if experimental_features_available():
        return GRIDDED_METHODS + EXPERIMENTAL_GRIDDED_METHODS
    else:
        return GRIDDED_METHODS


def show_interpolation_results(
    elevation,
    elevation_int,
    mask_int,
    xs_mat,
    ys_mat,
    x_var_name="x",
    y_var_name="y",
    colormap="terrain",
    highlight_interpolated_area=False,
    scatter_xs=None,
    scatter_ys=None,
    scatter_values=None,
    truncate_to_input_range=False,
):
    """Show interpolation results side by side.

    If scatter_xs/ys/values are provided, the left panel shows the scattered
    input points instead of the elevation grid (useful for XYZ input data).

    If truncate_to_input_range is set, the colormap range is derived from the
    input data (scatter_values when provided, otherwise the elevation grid)
    instead of the interpolated result, which visually clamps overshoots.
    """
    extent = [xs_mat.min(), xs_mat.max(), ys_mat.min(), ys_mat.max()]
    if truncate_to_input_range:
        input_data = scatter_values if scatter_values is not None else elevation
        vmin = np.nanmin(input_data)
        vmax = np.nanmax(input_data)
    else:
        vmin = np.nanmin(elevation_int)
        vmax = np.nanmax(elevation_int)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), layout="compressed")

    # Left panel: scattered input points or original grid
    if scatter_xs is not None:
        sc = axes[0].scatter(
            scatter_xs,
            scatter_ys,
            c=scatter_values,
            s=5,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_xlim(extent[0], extent[1])
        axes[0].set_ylim(extent[2], extent[3])
        im = sc
    else:
        im = axes[0].imshow(
            elevation,
            origin="lower",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
        )
        if highlight_interpolated_area:
            mask_overlay = np.where(mask_int == 1, 1.0, np.nan)
            axes[0].imshow(
                mask_overlay,
                origin="lower",
                cmap="autumn",
                alpha=0.4,
                extent=extent,
                aspect="equal",
            )
            axes[0].legend(
                handles=[
                    mpatches.Patch(color="red", alpha=0.4, label="Area to interpolate")
                ],
                loc="lower right",
            )
    axes[0].set_title("Original")
    axes[0].set_xlabel(x_var_name)
    axes[0].set_ylabel(y_var_name)

    # Right panel: interpolated grid
    axes[1].imshow(
        elevation_int,
        origin="lower",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )
    axes[1].set_title("Interpolated")
    axes[1].set_xlabel(x_var_name)
    axes[1].set_ylabel(y_var_name)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Elevation (m)")
    plt.show(block=True)


def add_common_args(parser, interpolation_flag_var_default=None):
    """Adds CLI arguments shared by all interpolation apps to the given ArgumentParser."""
    parser.add_argument(
        "-o",
        "--output_file",
        dest="output_file",
        action="store",
        type=str,
        help="Output NetCDF file with interpolated values",
    )
    parser.add_argument(
        "--areas",
        action="store",
        type=str,
        default=None,
        help="KML file containing the areas that will be interpolated.",
    )
    parser.add_argument(
        "--elevation_var",
        action="store",
        type=str,
        default="elevation",
        help="Name of the variable storing the elevation grid.",
    )
    parser.add_argument(
        "--x_var",
        action="store",
        type=str,
        default="lon",
        help="Name of the variable storing the columns' coordinates of the elevation grid.",
    )
    parser.add_argument(
        "--y_var",
        action="store",
        type=str,
        default="lat",
        help="Name of the variable storing the rows' coordinates of the elevation grid.",
    )
    parser.add_argument(
        "--x_dim",
        action="store",
        type=str,
        default="",
        help="Name of the dimension for the columns' coordinates."
        " Defaults to x_var if not set.",
    )
    parser.add_argument(
        "--y_dim",
        action="store",
        type=str,
        default="",
        help="Name of the dimension for the rows' coordinates."
        " Defaults to y_var if not set.",
    )
    parser.add_argument(
        "--interpolation_flag_var",
        action="store",
        type=str,
        default=interpolation_flag_var_default,
        help="Name of the variable storing the per-cell interpolation flag.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Verbosity flag, activate it to have feedback of the current"
        " steps of the process in the command line",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        dest="show",
        default=False,
        help="Show interpolation problem and results on screen",
    )
    parser.add_argument(
        "--colormap",
        action="store",
        type=str,
        dest="colormap",
        default="terrain",
        help="Matplotlib colormap used when showing results (default: terrain)",
    )
    parser.add_argument(
        "--highlight_interpolated_area",
        action="store_true",
        dest="highlight_interpolated_area",
        default=False,
        help="Highlight the area to interpolate in the results plot",
    )
    parser.add_argument(
        "--truncate_to_input_range",
        action="store_true",
        dest="truncate_to_input_range",
        default=False,
        help="Clip the interpolated values to the [min, max] range of the input"
        " data, both in the results plot and in the written output file. Useful"
        " to remove overshoots produced by some interpolation methods.",
    )
    parser.add_argument(
        "--max_distance_to_data",
        action="store",
        type=float,
        dest="max_distance_to_data",
        default=None,
        help="If set, output cells whose nearest input data point is farther than"
        " this distance (in the coordinate units of the grid, e.g. degrees for"
        " lon/lat) are left as NaN/nodata instead of being interpolated. Off by"
        " default.",
    )
    return parser


def _too_far_from_data(xs_query, ys_query, xs_ref, ys_ref, max_distance):
    """Boolean array, True where the query point is farther than max_distance
    (Euclidean, coordinate units) from the nearest reference point."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack((np.ravel(xs_ref), np.ravel(ys_ref))))
    dists, _ = tree.query(
        np.column_stack((np.ravel(xs_query), np.ravel(ys_query))), workers=-1
    )
    return dists > max_distance


def run_scattered_interpolation(
    params, xs_ref, ys_ref, elevation_ref, xs_int, ys_int, condp
):
    """Creates a scattered interpolant and applies it to the query points.

    Returns zi, the interpolated values at (xs_int, ys_int).
    """
    method = params.subparser_name.lower()

    # Create the interpolant
    endl = "\n" if method == "purbf" else ""
    if params.verbose:
        condp.print("- Creating the interpolant...", end=endl)
        ts = timer()
    if method == "nearest":
        interpolant = NearestNeighborInterpolant(
            xs_ref, ys_ref, elevation_ref, params.rescale
        )
    elif method == "linear":
        interpolant = LinearInterpolant(
            xs_ref, ys_ref, elevation_ref, params.fill_value, params.rescale
        )
    elif method == "cubic":
        interpolant = CubicInterpolant(
            xs_ref,
            ys_ref,
            elevation_ref,
            params.fill_value,
            params.tolerance,
            params.max_iters,
            params.rescale,
        )
    elif method == "rbf":
        if len(ys_ref) > 10000:
            print(
                "\n!!!WARNING!!! You are trying to build a RBF interpolant from a large"
                " number of data points, and this may require large computational cost"
                " and memory consumption.\n"
                "Please consider using the PURBF interpolant instead!"
            )
        interpolant = RBFInterpolant(
            xs_ref,
            ys_ref,
            elevation_ref,
            rbf_type=params.rbf_type,
            distance_type=params.rbf_distance_type,
            epsilon=params.rbf_epsilon,
            regularization=params.rbf_regularization,
            polynomial_degree=params.rbf_polynomial_degree,
        )
    elif method == "purbf":
        w = np.max(xs_int) - np.min(xs_int)
        h = np.max(ys_int) - np.min(ys_int)
        wh = max(w, h)
        if wh == 0:
            # Special case: single cell to interpolate
            wh = xs_int[0] * 1e-6 if xs_int[0] != 0 else 1e-6
        domain = [np.min(xs_int), np.min(ys_int), wh]
        interpolant = QuadTreePURBFInterpolant(
            xs_ref,
            ys_ref,
            elevation_ref,
            domain=domain,
            min_points_in_cell=params.pu_min_point_in_cell,
            overlap=params.pu_overlap,
            overlap_increment=params.pu_overlap_increment,
            min_cell_size_percent=params.pu_min_cell_size_percent,
            rbf_type=params.rbf_type,
            distance_type=params.rbf_distance_type,
            epsilon=params.rbf_epsilon,
            regularization=params.rbf_regularization,
            polynomial_degree=params.rbf_polynomial_degree,
        )
    elif method == "mlp":
        from heightmap_interpolation.interpolants.mlp_interpolant import MLPInterpolant

        interpolant = MLPInterpolant(
            xs_ref,
            ys_ref,
            elevation_ref,
            hidden_dims=params.hidden_dims,
            use_fourier=params.use_fourier,
            num_frequencies=params.num_frequencies,
            fourier_scale=params.fourier_scale,
            lr=params.lr,
            smoothness_weight=params.smoothness_weight,
            use_density_weighting=params.use_density_weighting,
            epochs=params.epochs,
            device=params.device,
            verbose=params.verbose,
            show_loss_plots=params.show_loss_plots,
        )
    elif method == "ams":
        suggested_scale = AMSInterpolant.preferred_scale_factor(
            xs_ref, ys_ref, xs_int, ys_int
        )
        if suggested_scale > params.scale:
            print(
                f"\n[WARNING] The requested scale parameter is too small to include"
                f" some of the points to interpolate within the query domain."
                f" Changing it to {suggested_scale}"
            )
            ams_scale = suggested_scale
        else:
            ams_scale = params.scale
        interpolant = AMSInterpolant(
            xs_ref,
            ys_ref,
            elevation_ref,
            depth=params.depth,
            degree=params.degree,
            solve_depth=params.solve_depth,
            full_depth=params.full_depth,
            base_depth=params.base_depth,
            boundary_type=params.boundary_type,
            iters=params.iters,
            base_v_cycles=params.base_v_cycles,
            max_memory_gb=params.max_memory_gb,
            parallel_type=params.parallel_type,
            parallel_schedule=params.parallel_schedule,
            parallel_thread_chunk_size=params.parallel_thread_chunk_size,
            value_weight=params.value_weight,
            gradient_weight=params.gradient_weight,
            scale=ams_scale,
            width=params.width,
            cg_accuracy=params.cg_accuracy,
            iso=params.iso,
            laplacian_weight=params.laplacian_weight,
            bi_laplacian_weight=params.bi_laplacian_weight,
            show_performance=params.show_performance,
            show_residual=params.show_residual,
            exact_interpolation=params.exact_interpolation,
            verbose=params.ams_verbose,
            transform_file=params.transform_file,
            estimate_gradients=params.estimate_gradients,
            gradient_neighbors=params.gradient_neighbors,
            gradient_max_distance=params.gradient_max_distance,
            gradient_min_planarity=params.gradient_min_planarity,
        )
    elif method == "gmt_surface":
        from heightmap_interpolation.interpolants.gmt_surface_interpolant import (
            GMTSurfaceInterpolant,
        )

        # The grid region must cover both the reference and the query points so that the
        # gridded result can be sampled everywhere it is needed.
        region = [
            float(min(np.min(xs_int), np.min(xs_ref))),
            float(max(np.max(xs_int), np.max(xs_ref))),
            float(min(np.min(ys_int), np.min(ys_ref))),
            float(max(np.max(ys_int), np.max(ys_ref))),
        ]
        # The XYZ app provides --cell_size; the netCDF4 app does not, so derive the grid
        # spacing from the coordinates (both ref and query points lie on the same grid).
        cell_size = getattr(params, "cell_size", None)
        if cell_size is not None:
            spacing = cell_size
        else:
            dx = _grid_spacing(np.concatenate((xs_ref.ravel(), xs_int.ravel())))
            dy = _grid_spacing(np.concatenate((ys_ref.ravel(), ys_int.ravel())))
            spacing = "{}/{}".format(dx, dy)
        interpolant = GMTSurfaceInterpolant(
            xs_ref,
            ys_ref,
            elevation_ref,
            spacing=spacing,
            region=region,
            tension=params.tension,
            convergence_limit=params.convergence_limit,
            max_radius=params.max_radius,
            max_iterations=params.max_iterations,
            verbose=params.verbose,
        )
    else:
        raise ValueError("Unknown interpolant type: {}".format(params.subparser_name))
    if params.verbose:
        condp.print(" done, {:.2f} sec.".format(timer() - ts))

    # Restrict the query points to those close enough to the input data: cells too
    # far from any input point are left as nodata and never interpolated.
    max_dist = getattr(params, "max_distance_to_data", None)
    if max_dist is not None:
        near = ~_too_far_from_data(xs_int, ys_int, xs_ref, ys_ref, max_dist)
        xs_q = xs_int[near]
        ys_q = ys_int[near]
        if params.verbose:
            condp.print(
                "- Discarding {:d}/{:d} query cells farther than {} from the input"
                " data".format(int(np.count_nonzero(~near)), len(xs_int), max_dist)
            )
    else:
        xs_q = xs_int
        ys_q = ys_int

    # Apply the interpolant at the (kept) query points
    if params.verbose:
        condp.print("- Applying the interpolant at the query points...", end=endl)
        ts = timer()
    if len(xs_q) == 0:
        zq = np.zeros(xs_q.shape)
    elif method not in ("rbf", "purbf"):
        zq = interpolant(xs_q, ys_q)
    else:
        # Apply in blocks to avoid large memory consumption
        query_block_size = params.query_block_size
        num_int = len(xs_q)
        zq = np.zeros(xs_q.shape)
        num_blocks = math.ceil(num_int / query_block_size)
        block_start = 0
        block_end = min(num_int, query_block_size)
        for b in range(num_blocks):
            condp.print("    - Querying block {}/{}".format(b + 1, num_blocks))
            zq[block_start:block_end] = interpolant(
                xs_q[block_start:block_end], ys_q[block_start:block_end]
            )
            block_start += query_block_size
            block_end = min(block_end + query_block_size, num_int)
    if params.verbose:
        condp.print(" done, {:.2f} sec.".format(timer() - ts))

    # Scatter the results back into the full set of query points (discarded cells
    # remain nodata)
    if max_dist is not None:
        zi = np.full(xs_int.shape, np.nan)
        zi[near] = zq
    else:
        zi = zq

    interpolant.cleanup()
    return zi


def run_gridded_inpainting(
    params,
    elevation_src,
    elevation_int,
    mask_int,
    xs_mat,
    ys_mat,
    cur_work_area,
    area_idx,
    num_areas,
    condp,
):
    """Runs the gridded inpainting method for a single work area.

    Modifies elevation_int in-place.
    """
    rows = np.any(cur_work_area, axis=1)
    cols = np.any(cur_work_area, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Extract this region; inpainting mask is the inverse of mask_int by convention
    cur_inpaint_mask = np.copy(~mask_int[rmin : rmax + 1, cmin : cmax + 1])
    cur_elevation = np.copy(elevation_src[rmin : rmax + 1, cmin : cmax + 1])
    # Exclude cells outside the marked area from inpainting
    cur_inpaint_mask = np.logical_or(
        cur_inpaint_mask, ~cur_work_area[rmin : rmax + 1, cmin : cmax + 1]
    )
    # Initializer / boundary condition for cells with unknown data
    cur_elevation[np.isnan(cur_elevation)] = 0

    if params.verbose and params.areas:
        condp.print("- Interpolating area {:d}/{:d}:".format(area_idx + 1, num_areas))
        condp.print(
            "    - Number of reference cells = {:d}".format(
                np.count_nonzero(cur_inpaint_mask)
            )
        )
        condp.print(
            "    - Number of cells to interpolate = {:d}".format(
                np.count_nonzero(~cur_inpaint_mask)
            )
        )

    inpainter = create_inpainter_from_params(params)
    if params.verbose:
        ts = timer()
    cur_elevation_int = inpainter.inpaint(cur_elevation, cur_inpaint_mask)
    if params.verbose:
        condp.print("- Inpainting took a total of {:.2f} sec.".format(timer() - ts))

    # Leave inpainted cells too far from any known-data cell as nodata
    if getattr(params, "max_distance_to_data", None) is not None:
        xs_region = xs_mat[rmin : rmax + 1, cmin : cmax + 1]
        ys_region = ys_mat[rmin : rmax + 1, cmin : cmax + 1]
        # Real known-data cells within this work area are the reference set. Do not
        # use cur_inpaint_mask directly: cells outside the work area were folded into
        # it above and are not real data.
        known = np.logical_and(
            ~mask_int[rmin : rmax + 1, cmin : cmax + 1],
            cur_work_area[rmin : rmax + 1, cmin : cmax + 1],
        )
        filled = ~cur_inpaint_mask
        if np.any(known) and np.any(filled):
            far_full = np.zeros_like(cur_inpaint_mask, dtype=bool)
            far_full[filled] = _too_far_from_data(
                xs_region[filled],
                ys_region[filled],
                xs_region[known],
                ys_region[known],
                params.max_distance_to_data,
            )
            cur_elevation_int[far_full] = np.nan

    # Paste results back (slice reference avoids a copy)
    elevation_slice = elevation_int[rmin : rmax + 1, cmin : cmax + 1]
    elevation_slice[~cur_inpaint_mask] = cur_elevation_int[~cur_inpaint_mask]


def add_common_fd_pde_inpainters_args(parser):
    """Adds to the ArgumentParser parser the set of options common to all FD-PDE inpainting methods"""

    # The following two commented parameters are common... but with different default values!
    # parser.add_argument("--update_step_size", default=0.01, help="Update step size")
    # parser.add_argument("--term_thres", default=0.01,
    #                              help="If the relative change between the inpainted elevations in the current and a previous step is smaller than this value, the optimization will stop")
    parser.add_argument(
        "--term_criteria",
        type=str,
        default="absolute_percent",
        help="The termination criteria to use. Available: 'relative': stop if the relative change between the inpainted elevations in the current and a previous step is smaller than this value. "
        + "'absolute': stop if all cells absolute change between the inpainted elevations in the current and a previous step is smaller than this value. "
        + "'absolute_percent' (default): stop if all cells absolute change between the inpainted elevations in the current and a previous step is smaller than this value multiplied by the absolute range of depths in the dataset (i.e., the absolute value is range_depths * absolute_change_percent).",
    )
    parser.add_argument(
        "--term_check_iters",
        type=int,
        default=1000,
        help="Number of iterations in the optimization after which we will check for the termination condition",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=1000000,
        help="Maximum number of iterations in the optimization.",
    )
    parser.add_argument(
        "--relaxation",
        type=float,
        default=0,
        help="Set to > 1 to perform over-relaxation at each iteration",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        help="The desired backend where computations should take place. If the requested backend is not available in the machine, will fallback to 'cpu'. Options: 'cpu', 'gpu'",
    )
    parser.add_argument(
        "--ti_arch",
        type=str,
        default="gpu",
        help="When '--backend' is 'gpu', this parameter sets the actual GPU architecture to use. Available options: 'cpu' (i.e., runs the GPU implementation in the CPU), 'gpu', 'cuda', 'vulkan', 'metal'",
    )
    # The following parameter gest its value from "verbose" global argument
    # parser.add_argument("--print_progress", action="store_true",
    #                              help="Flag indicating if some info about the optimization progress should be printed on screen")
    parser.add_argument(
        "--print_progress_iters",
        type=int,
        default=1000,
        help="If set to > 0, the optimization progress will be shown after this number of iterations",
    )
    parser.add_argument(
        "--mgs_levels",
        type=int,
        default=5,
        help="Levels of the Multi-grid solver. I.e., number of levels of detail used in the solving pyramid",
    )
    parser.add_argument(
        "--mgs_min_res",
        type=int,
        default=100,
        help="If during the construction of the pyramid of the Multi-Grid Solver one of the dimensions of the grid drops below this size, the pyramid construction will stop at that level",
    )
    parser.add_argument(
        "--init_with",
        type=str,
        default="nearest",
        help="Initialize the unknown values to inpaint using a simple interpolation function. If using a MGS, this will be used with the lowest level on the pyramid. Available initializers: 'nearest' (default), 'linear', 'cubic', 'harmonic'",
    )
    parser.add_argument(
        "--convolver",
        type=str,
        default="opencv",
        help="The convolution method to use. Available: 'opencv' (default),'scipy-signal', 'scipy-ndimage', 'masked', 'masked-parallel'",
    )
    parser.add_argument(
        "--debug_dir",
        action="store",
        dest="debug_dir",
        default="",
        type=str,
        help="If set, debugging information will be stored in this directory (useful to visualize the inpainting progress)",
    )
    parser.add_argument(
        "--use_direct_solver",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a direct solver instead of an iterative one",
    )
    parser.add_argument(
        "--direct_solver",
        type=str,
        default="cg",
        choices=["cg", "minres"],
        help="Sparse solver to use when --use_direct_solver is set: 'cg' "
        "(conjugate gradient, for SPD systems) or 'minres' (for symmetric "
        "indefinite systems) (default: cg)",
    )
    parser.add_argument(
        "--cg_term_thres",
        type=float,
        default=1e-4,
        help="Convergence tolerance (rtol) for the sparse CG/minres solver "
        "when --use_direct_solver is set. This is independent of "
        "--term_thres and --term_criteria, which only apply to the "
        "iterative solver (default: 1e-4)",
    )
    return parser


def get_common_fd_pde_inpainters_params_from_args(params):
    """Gets the set of common parameters/options of all FD-PDE inpainters from the parameters structure derived from ArgumentParser"""
    options = {
        "update_step_size": params.update_step_size,
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
        "debug_dir": params.debug_dir,
        "use_direct_solver": params.use_direct_solver,
        "direct_solver": params.direct_solver,
        "cg_term_thres": params.cg_term_thres,
    }
    return options


def add_subparsers(subparsers):
    # Parser for the "nearest" method
    parser_nearest = subparsers.add_parser(
        "nearest", help="Nearest-neighbor interpolator"
    )
    parser_nearest.add_argument(
        "--rescale",
        action="store_true",
        dest="rescale",
        help="Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.",
    )

    # Parser for the "linear" method
    parser_linear = subparsers.add_parser("linear", help="Linear interpolator")
    parser_linear.add_argument(
        "--fill_value",
        type=float,
        default=np.nan,
        help="Value used to fill in for requested points outside of the convex hull of the input points. If not provided, the default is NaN.",
    )
    parser_linear.add_argument(
        "--rescale",
        action="store_true",
        dest="rescale",
        help="Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.",
    )

    # Parser for the "cubic" method
    parser_cubic = subparsers.add_parser(
        "cubic",
        help="Piecewise cubic, C1 smooth, curvature-minimizing (Clough-Tocher) nterpolator",
    )
    parser_cubic.add_argument(
        "--fill_value",
        type=float,
        default=np.nan,
        help="Value used to fill in for requested points outside of the convex hull of the input points. If not provided, the default is NaN.",
    )
    parser_cubic.add_argument(
        "--rescale",
        action="store_true",
        dest="rescale",
        help="Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.",
    )
    parser_cubic.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute/relative tolerance for gradient estimation.",
    )
    parser_cubic.add_argument(
        "--max_iters",
        type=int,
        default=400,
        help="Maximum number of iterations in gradient estimation.",
    )

    # Parser for the "rbf" method
    parser_rbf = subparsers.add_parser("rbf", help="Radial Basis Function interpolant")
    parser_rbf.add_argument(
        "--query_block_size",
        action="store",
        type=int,
        default=1000,
        help="Apply the interpolant using maximum this number of points at a time to avoid large memory consumption",
    )
    parser_rbf.add_argument(
        "--rbf_distance_type",
        action="store",
        type=str,
        default="euclidean",
        help="Distance type. Available: euclidean (default), haversine, vincenty",
    )
    parser_rbf.add_argument(
        "--rbf_type",
        action="store",
        type=str,
        default="thinplate",
        help="RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland",
    )
    parser_rbf.add_argument(
        "--rbf_epsilon",
        action="store",
        type=float,
        default=1,
        help="Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland",
    )
    parser_rbf.add_argument(
        "--rbf_regularization",
        action="store",
        type=float,
        default=0,
        help="Regularization scalar to use while creating the RBF interpolant (optional)",
    )
    parser_rbf.add_argument(
        "--rbf_polynomial_degree",
        action="store",
        type=int,
        default=1,
        help="Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)",
    )

    # Parser for the "pu-rbf" method
    parser_purbf = subparsers.add_parser(
        "purbf", help="Partition of Unity Radial Basis Function interpolant"
    )
    parser_purbf.add_argument(
        "--query_block_size",
        action="store",
        type=int,
        default=1000,
        help="Apply the interpolant using maximum this number of points at a time to avoid large memory consumption",
    )
    parser_purbf.add_argument(
        "--rbf_distance_type",
        action="store",
        type=str,
        default="euclidean",
        help="Distance type. Available: euclidean (default), haversine, vincenty",
    )
    parser_purbf.add_argument(
        "--rbf_type",
        action="store",
        type=str,
        default="thinplate",
        help="RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland",
    )
    parser_purbf.add_argument(
        "--rbf_epsilon",
        action="store",
        type=float,
        default=1,
        help="Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland",
    )
    parser_purbf.add_argument(
        "--rbf_regularization",
        action="store",
        type=float,
        default=0,
        help="Regularization scalar to use while creating the RBF interpolant (optional)",
    )
    parser_purbf.add_argument(
        "--rbf_polynomial_degree",
        action="store",
        type=int,
        default=1,
        help="Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)",
    )
    parser_purbf.add_argument(
        "--pu_overlap",
        action="store",
        type=float,
        default=0.25,
        help="Overlap factor between circles in neighboring sub-domains in the partition. The radius of a QuadTree cell, computed as half its diagonal, is enlarged by this factor",
    )
    parser_purbf.add_argument(
        "--pu_min_point_in_cell",
        action="store",
        type=int,
        default=1000,
        help="Minimum number of points in a QuadTree cell",
    )
    parser_purbf.add_argument(
        "--pu_min_cell_size_percent",
        action="store",
        type=float,
        default=0.005,
        help="Minimum cell size, specified as a percentage [0..1] of the max(width, height) of the query domain",
    )
    parser_purbf.add_argument(
        "--pu_overlap_increment",
        action="store",
        type=float,
        default=0.001,
        help="If, after creating the QuadTree, a cell contains less than pu_min_point_in_cell, the radius will be iteratively incremented until this condition is satisfied. This parameter specifies how much the radius of a cell increments at each iteration",
    )

    # Parser for the "mlp" method
    parser_mlp = subparsers.add_parser("mlp", help="Multi-Layer Perceptron interpolant")
    parser_mlp.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 128, 128, 128, 128, 64],
        help="Sizes of the hidden layers of the MLP, as a space-separated list (default: 64 128 128 128 128 64)",
    )
    parser_mlp.add_argument(
        "--use_fourier",
        action="store_true",
        help="Use Fourier feature encoding of the input coordinates, which helps the network learn high-frequency details (default: false)",
    )
    parser_mlp.add_argument(
        "--num_frequencies",
        type=int,
        default=10,
        help="Number of Fourier frequencies to use. Only relevant if --use_fourier is set (default: 10)",
    )
    parser_mlp.add_argument(
        "--fourier_scale",
        type=float,
        default=1.0,
        help="Scale of the Fourier features. Only relevant if --use_fourier is set (default: 1.0)",
    )
    parser_mlp.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the Adam optimizer (default: 1e-3)",
    )
    parser_mlp.add_argument(
        "--smoothness_weight",
        type=float,
        default=0.0,
        help="Weight of the smoothness (Laplacian) regularization loss. Larger values yield smoother interpolations (default: 0.0, i.e., disabled)",
    )
    parser_mlp.add_argument(
        "--use_density_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weight the training samples by the inverse of their local density, so that sparse regions get more importance. Disable with --no-use_density_weighting (default: true)",
    )
    parser_mlp.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of training epochs (default: 10000)",
    )
    parser_mlp.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the MLP on (e.g. 'cpu', 'cuda', 'cuda:0'). By default it uses the GPU if available, and falls back to the CPU otherwise",
    )
    parser_mlp.add_argument(
        "--show_loss_plots",
        action="store_true",
        help="Show live matplotlib plots of the training losses. Requires a GUI backend, so keep it off on headless machines (default: false)",
    )

    # Parser for the "poisson" method
    parser_ams = subparsers.add_parser(
        "ams",
        description="Adaptive Multi-grid Solver interpolant",
        help="PointInterpolant/AdaptiveTreeVisualization tools from the PoissonRecon project (https://github.com/mkazhdan/PoissonRecon)",
    )
    parser_ams.add_argument(
        "--depth",
        type=int,
        default=10,
        help="This integer is the maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x ... Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound. (default: 8)",
    )
    parser_ams.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Degree of the B-spline that is to be used to define the finite elements system. Larger degrees support higher order approximations, but come at the cost of denser system matrices (incurring a cost in both space and time). (default: 2)",
    )
    parser_ams.add_argument(
        "--solve_depth",
        type=int,
        default=-1,
        help=" the depth up to which the solver will solve the numerical system. It will still show the results at the finest resolution, but no additional high-frequency data will be introduced at the finest resolutions. It could also be the case that aliasing that occurs at the coarser resolutions will not get corrected. (default = -1, i.e., --depth)",
    )
    parser_ams.add_argument(
        "--full_depth",
        type=int,
        default=5,
        help="The depth up to which the octree is completely refined, i.e. a regular grid (default: 5)",
    )
    parser_ams.add_argument(
        "--base_depth",
        type=int,
        default=-1,
        help="The coarsest depth at which the system will be solved over an octree. (At coarser levels it will be solved using a standard MG solver, with multiple V-Cycles, defined over a regular grid.) As such, the assumption is that BaseDepth<=FullDepth (default = -1, i.e., not used)",
    )
    parser_ams.add_argument(
        "--boundary_type",
        type=str,
        default="free",
        help="Boundary type (default: free, available: free, dirichlet, neumann)",
    )
    parser_ams.add_argument(
        "--iters",
        type=int,
        default=8,
        help="The number of Gauss-Seidel relaxations to be performed at every level of the hierarchy (default: 8)",
    )
    parser_ams.add_argument(
        "--base_v_cycles",
        type=int,
        default=4,
        help="coarse MG solver v-cycles (default: 4)",
    )
    parser_ams.add_argument(
        "--max_memory_gb",
        type=int,
        default=0,
        help="Maximum memory to use in GB (default: 0, i.e., no limit)",
    )
    parser_ams.add_argument(
        "--parallel_type",
        default="openmp",
        help="Parallel mode (default: openmp, available: openmp, threads, none",
    )
    parser_ams.add_argument(
        "--parallel_schedule",
        type=str,
        default="static",
        help="Parallel schedule (default: static, available: static, dynamic)",
    )
    parser_ams.add_argument(
        "--parallel_thread_chunk_size",
        type=int,
        default=128,
        help="Parallel thread chunk size (default: 128)",
    )
    parser_ams.add_argument(
        "--value_weight",
        type=float,
        default=1000.0,
        help="Importance that interpolation of the samples' values is given in the fitting of the function (default: 1000.0)",
    )
    parser_ams.add_argument(
        "--gradient_weight",
        type=float,
        default=1.0,
        help="Importance that interpolation of the samples' gradients is given in the fitting of the function (default: 1.0)",
    )
    parser_ams.add_argument(
        "--estimate_gradients",
        action="store_true",
        help="Estimate gradients at the input points (via a local plane fit) and"
        " also fit them, in addition to the values. Only reliable gradients are"
        " kept. Their influence is controlled by --gradient_weight (default: false)",
    )
    parser_ams.add_argument(
        "--gradient_neighbors",
        type=int,
        default=8,
        help="Number of nearest neighbors used to estimate the gradient at each"
        " input point (default: 8)",
    )
    parser_ams.add_argument(
        "--gradient_max_distance",
        type=float,
        default=None,
        help="Maximum distance (in coordinate units) to the furthest neighbor used"
        " for a point's gradient to be considered reliable. If not set, it is"
        " derived automatically from the data spacing (default: auto)",
    )
    parser_ams.add_argument(
        "--gradient_min_planarity",
        type=float,
        default=0.0,
        help="Minimum coefficient of determination (R^2) of the local plane fit to"
        " keep a point's gradient, in [0, 1]. Set to 0 to disable this filter"
        " (default: 0.0)",
    )
    parser_ams.add_argument(
        "--scale",
        type=float,
        default=1.1,
        help="The ratio between the diameter of the cube used for reconstruction and the diameter of the samples' bounding cube. (default: 1.1)",
    )
    parser_ams.add_argument(
        "--width",
        type=float,
        default=0.0,
        help="Target width of the finest level octree cells. This parameter is ignored if the --depth is also specified. (default: 0.0, i.e., ignore and use --depth)",
    )
    parser_ams.add_argument(
        "--cg_accuracy",
        type=float,
        default=1e-3,
        help="Conjugate Gradient solver accuracy (default: 1e-3)",
    )
    parser_ams.add_argument(
        "--iso", type=float, default=0.0, help="Iso-value (default=0.0)"
    )
    parser_ams.add_argument(
        "--laplacian_weight",
        type=float,
        default=0.0,
        help="Importance that Laplacian regularization is given in the fitting of the function (default: 0.0)",
    )
    parser_ams.add_argument(
        "--bi_laplacian_weight",
        type=float,
        default=1.0,
        help="Importance that bi-Laplacian regularization is given in the fitting of the function (default: 1.0)",
    )
    parser_ams.add_argument(
        "--show_performance",
        action="store_true",
        help="Show performance statistics (default: false)",
    )
    parser_ams.add_argument(
        "--show_residual", action="store_true", help="Show residuals (default: false)"
    )
    parser_ams.add_argument(
        "--exact_interpolation",
        action="store_true",
        help="Use exact interpolation (default: false)",
    )
    parser_ams.add_argument(
        "--ams_verbose",
        action="store_true",
        help="Verbose mode for AMS, will print information during the creation of the interpolant (default: false)",
    )
    parser_ams.add_argument(
        "--transform_file", type=str, default="", help="Transform file (default: none)"
    )

    # Parser for the "gmt_surface" method
    parser_gmt = subparsers.add_parser(
        "gmt_surface",
        help="GMT continuous-curvature spline-in-tension gridder (requires pygmt/GMT)",
    )
    parser_gmt.add_argument(
        "--tension",
        type=float,
        default=0.0,
        help="Tension factor in [0..1] (GMT -T). 0 = minimum curvature; higher values"
        " reduce overshoot near steep gradients (default: 0.0)",
    )
    parser_gmt.add_argument(
        "--convergence_limit",
        type=float,
        default=0.0,
        help="Convergence limit (GMT -C). 0 = GMT default (default: 0.0)",
    )
    parser_gmt.add_argument(
        "--max_radius",
        type=str,
        default=None,
        help="Search radius for nearest-data initialization (GMT -M), e.g. '5c'"
        " (default: GMT default)",
    )
    parser_gmt.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum number of iterations (GMT -N). None = GMT default (default: None)",
    )

    # Parser for the "harmonic" method
    parser_harmonic = subparsers.add_parser("harmonic", help="Harmonic inpainter")
    parser_harmonic.add_argument(
        "--update_step_size", type=float, default=0.2, help="Update step size"
    )
    parser_harmonic.add_argument(
        "--term_thres",
        type=float,
        default=1e-5,
        help="Termination threshold. Its meaning depends on the --term_criteria parameter.",
    )
    parser_harmonic = add_common_fd_pde_inpainters_args(parser_harmonic)

    # Parser for the "tv" method
    parser_tv = subparsers.add_parser(
        "tv", help="Inpainter minimizing Total-Variation (TV) across the 'image'"
    )
    parser_tv.add_argument(
        "--update_step_size", type=float, default=0.225, help="Update step size"
    )
    parser_tv.add_argument(
        "--term_thres",
        type=float,
        default=1e-5,
        help="Termination threshold. Its meaning depends on the --term_criteria parameter.",
    )
    parser_tv = add_common_fd_pde_inpainters_args(parser_tv)
    parser_tv.add_argument(
        "--epsilon",
        type=float,
        default=1,
        help="A small value to be added when computing the norm of the gradients during optimization, to avoid a division by zero",
    )

    # Parser for the "ccst" method
    parser_ccst = subparsers.add_parser(
        "ccst", help="Continous Curvature Splines in Tension (CCST) inpainter"
    )
    parser_ccst.add_argument(
        "--update_step_size", type=float, default=0.01, help="Update step size"
    )
    parser_ccst.add_argument(
        "--term_thres",
        type=float,
        default=1e-5,
        help="Termination threshold. Its meaning depends on the --term_criteria parameter.",
    )
    parser_ccst = add_common_fd_pde_inpainters_args(parser_ccst)
    parser_ccst.add_argument(
        "--tension",
        type=float,
        default=0.3,
        help="Tension parameter weighting the contribution between a harmonic and a biharmonic interpolation (see the docs and the original reference for more details)",
    )

    # Parser for the "amle" method
    parser_amle = subparsers.add_parser(
        "amle", help="Absolutely Minimizing Lipschitz Extension (AMLE) inpainter"
    )
    parser_amle.add_argument(
        "--update_step_size", type=float, default=0.01, help="Update step size"
    )
    parser_amle.add_argument(
        "--term_thres",
        type=float,
        default=1e-5,
        help="Termination threshold. Its meaning depends on the --term_criteria parameter.",
    )
    parser_amle = add_common_fd_pde_inpainters_args(parser_amle)
    parser_amle.add_argument(
        "--convolve_in_1d",
        action="store_true",
        help="Perform 1D convolutions instead of using the 2D convolution indicated in --convolver",
    )

    # Parser for the "navier-stokes" method
    parser_ns = subparsers.add_parser(
        "navier-stokes", help="OpenCV's Navier-Stokes inpainter"
    )
    parser_ns.add_argument(
        "--radius",
        type=int,
        default=25,
        help="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm",
    )

    # Parser for the "telea" method
    parser_ns = subparsers.add_parser("telea", help="OpenCV's Telea inpainter")
    parser_ns.add_argument(
        "--radius",
        type=int,
        default=25,
        help="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm",
    )

    # Parser for the "shiftmap" method
    parser_shiftmap = subparsers.add_parser(
        "shiftmap", help="OpenCV's xphoto module's Shiftmap inpainter"
    )

    # Parser for the "ebi" (Exemplar-Based Inpainter) method
    parser_ebi = subparsers.add_parser("ebi", help="Exemplar-based inpainter")
    parser_ebi.add_argument(
        "--patch_size", type=int, default=9, help="Size of the inpainting patch."
    )
    parser_ebi.add_argument(
        "--search_original_source_only",
        action="store_true",
        help="If true, just the original source image - mask will be searched for inpainting patches. Otherwise, the growing inpainting area will also be taken into account.",
    )
    # search_color_space (str, optional): Color space to use when searching for the next best filler patch. Options available: "bgr", "hsv", "lab", "gray". In case gray is selected, the input image must also be grayscale. Defaults to "bgr".
    parser_ebi.add_argument(
        "--plot_progress",
        action="store_true",
        help="Activates the plotting of the inpainting process (internal of the inpainter library)",
    )
    parser_ebi.add_argument(
        "--out_progress_dir",
        type=str,
        help="Set to a directory to get the same output as with --plot_progress, but stored in files.",
    )
    parser_ebi.add_argument(
        "--show_progress_bar",
        action="store_true",
        help="Activates the progress bar (internal of the inpainter library).",
    )
    parser_ebi.add_argument(
        "--patch_preference",
        type=str,
        help='When more than a patch has the same similarity score, this parameter selects which one to choose. Available: "any", "closest", "random".',
    )


def create_inpainter_from_params(params):
    if (
        params.subparser_name.lower() != "navier-stokes"
        and params.subparser_name.lower() != "telea"
        and params.subparser_name.lower() != "shiftmap"
        and params.subparser_name.lower() != "ebi"
    ):
        options = get_common_fd_pde_inpainters_params_from_args(params)
        if options["backend"] == "gpu" and (
            params.subparser_name.lower() != "harmonic"
            and params.subparser_name.lower() != "ccst"
        ):
            raise ValueError(
                "Currently the GPU backend is only available for harmonic and ccst methods."
            )
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
            "patch_preference": params.patch_preference,
        }
        inpainter = ExemplarBasedInpainter(**options)

    return inpainter


def experimental_features_available():
    try:
        import torch

        return True
    except ImportError:
        return False


def pygmt_available():
    try:
        import pygmt  # noqa: F401

        return True
    except ImportError:
        return False


def _grid_spacing(coords):
    """Estimates the spacing of a regular grid from a set of (possibly partial) coordinates.

    Returns the minimum positive difference between consecutive unique coordinate values.
    """
    unique = np.unique(coords)
    if unique.size < 2:
        raise ValueError(
            "Cannot derive a grid spacing from fewer than 2 distinct coordinates."
        )
    return float(np.min(np.diff(unique)))
