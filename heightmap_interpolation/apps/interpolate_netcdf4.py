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
import math
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

# All interpolation methods
from heightmap_interpolation.apps.apps_common import (
    add_subparsers,
    create_inpainter_from_params,
    experimental_features_available,
)
from heightmap_interpolation.apps.netcdf_data_io import (
    load_interpolation_input_data,
    write_interpolation_results,
)
from heightmap_interpolation.interpolants.cubic_interpolant import CubicInterpolant
from heightmap_interpolation.interpolants.linear_interpolant import LinearInterpolant
from heightmap_interpolation.interpolants.nearest_neighbor_interpolant import (
    NearestNeighborInterpolant,
)
from heightmap_interpolation.interpolants.quad_tree_pu_rbf_interpolant import (
    QuadTreePURBFInterpolant,
)
from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant
from heightmap_interpolation.misc.conditional_print import ConditionalPrint

if experimental_features_available():
    from heightmap_interpolation.interpolants.mlp_interpolant import MLPInterpolant

from heightmap_interpolation.interpolants.ams_interpolant import (
    AMSInterpolant,
)


def interpolate(params):
    condp = ConditionalPrint(params.verbose)

    # Load the data of the interpolation problem
    if params.verbose:
        condp.print("- Loading data...", end="", flush=True)
        ts = timer()
    xs_mat, ys_mat, elevation, mask_int, mask_ref, work_areas = (
        load_interpolation_input_data(
            params.input_file,
            params.elevation_var,
            x_var=params.x_var,
            y_var=params.y_var,
            x_dim=(params.x_dim if params.x_dim else params.x_var),
            y_dim=(params.y_dim if params.y_dim else params.y_var),
            interpolation_flag_var=params.interpolation_flag_var,
            areas_kml_file=params.areas,
        )
    )
    elevation_int = np.copy(elevation)
    if params.verbose:
        te = timer()
        condp.print(" done, {:.2f} sec.".format(te - ts))

    # Show a bit of information regarding the interpolation problem (percentage of missing data to interpolate w.r.t. the full image)
    if params.verbose:
        condp.print("- Summary of input data:")
        condp.print(
            "    - Elevation grid has a size of {:d}x{:d} cells".format(
                elevation.shape[0], elevation.shape[1]
            )
        )
        if params.areas:
            condp.print(
                "    - Data will be interpolated just at the user-defined areas"
            )
        else:
            total_cells = elevation.shape[0] * elevation.shape[1]
            num_cells_to_interpolate = np.count_nonzero(mask_int)
            interp_percent = (num_cells_to_interpolate / total_cells) * 100
            condp.print(
                "    - Cells to interpolate represent a {:.2f}% of the image:".format(
                    interp_percent
                )
            )
            (condp.print("        - Total cells = {:d}".format(total_cells)),)
            condp.print(
                "        - Number of reference cells = {:d}".format(
                    total_cells - num_cells_to_interpolate
                )
            )
            condp.print(
                "        - Number of cells to interpolate = {:d}".format(
                    num_cells_to_interpolate
                )
            )

    for i in range(work_areas.shape[2]):
        # Get the current working area
        cur_work_area = work_areas[:, :, i]

        # --- Scattered data interpolation ---
        scattered_methods = [
            "nearest",
            "linear",
            "cubic",
            "rbf",
            "purbf",
            "ams",
        ]
        if experimental_features_available:
            scattered_methods.append("mlp")

        if params.subparser_name.lower() in scattered_methods:
            # Get the reference points from the current working area
            cur_mask_ref = np.logical_and(mask_ref, cur_work_area)
            cur_mask_int = np.logical_and(mask_int, cur_work_area)

            # Cast the matrices to a set of "scattered" data points and references
            ys_ref = ys_mat[cur_mask_ref]
            xs_ref = xs_mat[cur_mask_ref]
            elevation_ref = elevation[cur_mask_ref]
            ys_int = ys_mat[cur_mask_int]
            xs_int = xs_mat[cur_mask_int]

            # Show a bit of information regarding the current area interpolation problem (percentage of missing data to interpolate w.r.t. the full image)
            if params.verbose and params.areas:
                condp.print(
                    "- Interpolating area {:d}/{:d}:".format(i + 1, work_areas.shape[2])
                )
                condp.print(
                    "    - Number of reference cells = {:d}".format(len(ys_ref))
                )
                condp.print(
                    "    - Number of cells to interpolate = {:d}".format(len(ys_int))
                )

            # Create the interpolant
            if params.verbose:
                endl = "\n" if params.subparser_name.lower() == "purbf" else ""
                condp.print("- Creating the interpolant...", end=endl)
                ts = timer()
            if params.subparser_name.lower() == "nearest":
                interpolant = NearestNeighborInterpolant(
                    xs_ref, ys_ref, elevation_ref, params.rescale
                )
            elif params.subparser_name.lower() == "linear":
                interpolant = LinearInterpolant(
                    xs_ref, ys_ref, elevation_ref, params.fill_value, params.rescale
                )
            elif params.subparser_name.lower() == "cubic":
                interpolant = CubicInterpolant(
                    xs_ref,
                    ys_ref,
                    elevation_ref,
                    params.fill_value,
                    params.tolerance,
                    params.max_iters,
                    params.rescale,
                )
            elif params.subparser_name.lower() == "rbf":
                # Warn the user if the PURBF is better suited for this problem
                if len(ys_ref) > 10000:
                    print(
                        "\n!!!WARNING!!! You are trying to build a RBF intepolant from a large number of data points, and this may require large computational cost and memory consumption.\nPlease consider using the PURBF interpolant instead!"
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
            elif params.subparser_name.lower() == "purbf":
                # Compute the query domain to be that of the points to interpolate
                minX = np.min(xs_int)
                maxX = np.max(xs_int)
                minY = np.min(ys_int)
                maxY = np.max(ys_int)
                w = maxX - minX
                h = maxY - minY
                wh = max(w, h)
                if wh == 0:
                    # Special case: when there is a single cell to interpolate!
                    wh = xs_mat[0, 1] - xs_mat[0, 0]  # a small number, pseudo-cell size
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
            elif params.subparser_name.lower() == "mlp":
                interpolant = MLPInterpolant(
                    xs_ref, ys_ref, elevation_ref
                )  # TODO: set parameters from command line!
            elif params.subparser_name.lower() == "ams":
                suggested_scale = AMSInterpolant.preferred_scale_factor(
                    xs_ref, ys_ref, xs_int, ys_int
                )
                if suggested_scale > params.scale:
                    print(
                        f"\n[WARNING] The requested scale parameter is too small to include some of the points to interpolate within the query domain. Changing it to {suggested_scale}"
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
                )
            else:
                raise ValueError(
                    "Unknown interpolant type: {}".format(params.subparser_name)
                )
            if params.verbose:
                te = timer()
                condp.print(" done, {:.2f} sec.".format(te - ts))

            # Interpolate at the grid points
            if params.verbose:
                condp.print(
                    "- Applying the interpolant at the query points...", end=endl
                )
                ts = timer()
            if (
                params.subparser_name.lower() != "rbf"
                and params.subparser_name.lower() != "purbf"
            ):
                zi = interpolant(xs_int, ys_int)
            else:
                # For RBF and PURBF, apply the interpolant in blocks to avoid large memory consumption

                # Divide the data into blocks
                query_block_size = params.query_block_size
                zi = np.zeros(xs_int.shape)
                num_int = np.sum(mask_int)
                num_blocks = math.ceil(num_int / query_block_size)
                block_start = 0
                block_end = min([num_int, query_block_size])

                # Interpolate
                for i in range(num_blocks):
                    message = "    - Querying block {}/{}".format(i + 1, num_blocks)
                    condp.print(message)
                    # condp.backspace(len(message))

                    zi[block_start:block_end] = interpolant(
                        xs_int[block_start:block_end], ys_int[block_start:block_end]
                    )
                    block_end = min([block_end + query_block_size, num_int])
                    block_start = block_start + query_block_size
            if params.verbose:
                te = timer()
                condp.print(" done, {:.2f} sec.".format(te - ts))

            # Put the interpolated values back into the elevation matrix
            elevation_int[cur_mask_int] = zi

            # Cleanup, if needed
            interpolant.cleanup()

        # --- Gridded data interpolation/inpainting ---
        gridded_methods = [
            "harmonic",
            "tv",
            "ccst",
            "ccst-ti",
            "amle",
            "navier-stokes",
            "telea",
            "shiftmap",
            "ebi",
        ]
        if params.subparser_name.lower() in gridded_methods:
            # if params.areas:
            # Get the bounding box of the current working area (inpainters work on full 2D grids...)
            ys = np.any(cur_work_area, axis=1)
            xs = np.any(cur_work_area, axis=0)
            rmin, rmax = np.where(ys)[0][[0, -1]]
            cmin, cmax = np.where(xs)[0][[0, -1]]

            # Extract this region from the image
            cur_inpaint_mask = np.copy(
                ~mask_int[rmin : rmax + 1, cmin : cmax + 1]
            )  # Inpainting mask (inverse of mask_int by our internal convention)
            cur_elevation = np.copy(elevation[rmin : rmax + 1, cmin : cmax + 1])
            # However, we will not interpolate those points in the rectangular region that do not fall within the marked area
            cur_inpaint_mask = np.logical_or(
                cur_inpaint_mask, ~cur_work_area[rmin : rmax + 1, cmin : cmax + 1]
            )

            cur_elevation[np.isnan(cur_elevation)] = (
                0  # Initializer, as well as boundary conditions when the areas to interpolate do not cover all the cells with unknown data
            )

            if params.verbose and params.areas:
                condp.print(
                    "- Interpolating area {:d}/{:d}:".format(i + 1, work_areas.shape[2])
                )
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

            # Create the inpainter
            inpainter = create_inpainter_from_params(params)
            # Inpaint!
            if params.verbose:
                ts = timer()
            cur_elevation_int = inpainter.inpaint(cur_elevation, cur_inpaint_mask)
            if params.verbose:
                te = timer()
                condp.print("- Inpainting took a total of {:.2f} sec.".format(te - ts))

            # "Paste" the results into the original elevation matrix
            # elevation_int[rmin:rmax+1, cmin:cmax+1] = cur_elevation_int
            elevation_slice = elevation_int[
                rmin : rmax + 1, cmin : cmax + 1
            ]  # Do not copy! we want to refer to that part in elevation_int matrix
            elevation_slice[~cur_inpaint_mask] = cur_elevation_int[
                ~cur_inpaint_mask
            ]  # Only modify the inpainted part! (This way we preserve "unknown"/NaN values in areas we did not interpolate

    # Write the results
    if params.output_file:
        condp.print("- Writing the results to disk")
        write_interpolation_results(
            params.input_file,
            params.output_file,
            elevation_int,
            mask_int,
            params.elevation_var,
            (params.x_dim if params.x_dim else params.x_var),
            (params.y_dim if params.y_dim else params.y_var),
            params.interpolation_flag_var,
            params.areas,
        )

    # Show results
    if params.show:
        condp.print("- Showing results")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        images = [elevation, elevation_int]
        titles = ["Original", "Interpolated"]
        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image, origin="lower")
            ax.set_title(title)
        fig.tight_layout()
        plt.show(block=True)


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Interpolate elevation data in a netCDF4 file (defaults set to read an EMODnet Bathymetry SeaDataNet_1.0 CF1.6-compliant file)"
    )
    # Create a sub-parser for each possible interpolator, with its own options
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")
    parser.add_argument(
        "input_file", action="store", type=str, help="Input NetCDF file"
    )
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
        help="Name of the variable storing the elevation grid in the input file.",
    )
    parser.add_argument(
        "--x_var",
        action="store",
        type=str,
        default="lon",
        help="Name of the variable storing the columns' coordinates of the elevation grid in the input file.",
    )
    parser.add_argument(
        "--y_var",
        action="store",
        type=str,
        default="lat",
        help="Name of the variable storing the rows' coordinates of the elevation grid in the input file.",
    )
    parser.add_argument(
        "--x_dim",
        action="store",
        type=str,
        default="",
        help="Name of the dimension for the columns' coordinates of the elevation grid in the input file. Defaults to x_var if not set.",
    )
    parser.add_argument(
        "--y_dim",
        action="store",
        type=str,
        default="",
        help="Name of the dimension for the rows' coordinates of the elevation grid in the input file. Defaults to y_var if not set.",
    )
    parser.add_argument(
        "--interpolation_flag_var",
        action="store",
        type=str,
        default=None,
        help="Name of the variable storing the per-cell interpolation flag in the input file (0 == known value, 1 == interpolated/to interpolate cell). If not set, it will interpolate the locations in the elevation variable containing an invalid (NaN) value.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Verbosity flag, activate it to have feedback of the current steps of the process in the command line",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        dest="show",
        default=False,
        help="Show interpolation problem and results on screen",
    )

    add_subparsers(subparsers)

    return parser.parse_args(args)


def main():
    interpolate(parse_args())


# Main function
if __name__ == "__main__":
    main()
