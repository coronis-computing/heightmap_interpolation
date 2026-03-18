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

from heightmap_interpolation.apps.apps_common import (
    add_subparsers,
    create_inpainter_from_params,
)
from heightmap_interpolation.apps.netcdf_data_io import (
    create_work_areas,
    write_interpolation_results_new_file,
)
from heightmap_interpolation.interpolants.cubic_interpolant import CubicInterpolant
from heightmap_interpolation.interpolants.linear_interpolant import LinearInterpolant
from heightmap_interpolation.interpolants.mlp_interpolant import MLPInterpolant

# All interpolation methods
from heightmap_interpolation.interpolants.nearest_neighbor_interpolant import (
    NearestNeighborInterpolant,
)
from heightmap_interpolation.interpolants.quad_tree_pu_rbf_interpolant import (
    QuadTreePURBFInterpolant,
)
from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant
from heightmap_interpolation.misc.conditional_print import ConditionalPrint


def load_interpolation_input_data_xyz(
    input_file,
    delimiter,
    raster_step,
    min_y=np.nan,
    max_y=np.nan,
    min_x=np.nan,
    max_x=np.nan,
    areas_kml_file=None,
):
    # Load the xyz (x, y, elevation) from a coma(or other)-separated file
    data = np.genfromtxt(input_file, delimiter=delimiter)

    xs = data[:, 0]
    ys = data[:, 1]
    elev = data[:, 2]

    # The result will be a raster, so we need to set its bounds
    if np.isnan(min_y):  # or np.isnan(max_y) or np.isnan(min_x) or np.isnan(max_x):
        min_y = np.min(ys)
    elif min_y > np.min(ys):
        print(
            "[WARNING] the minimum Y requested ({:f}) is larger than the minimum Y of the samples ({:f})",
            min_y,
            np.min(ys),
        )
    if np.isnan(min_x):
        min_x = np.min(xs)
    elif min_x > np.min(xs):
        print(
            "[WARNING] the minimum X requested ({:f}) is larger than the minimum X of the samples ({:f})",
            min_x,
            np.min(xs),
        )
    if np.isnan(max_y):
        max_y = np.max(ys)
    elif max_y < np.max(ys):
        print(
            "[WARNING] the maximum Y requested ({:f}) is smaller than the maximum Y of the samples ({:f})",
            max_y,
            np.max(ys),
        )
    if np.isnan(max_x):
        max_x = np.max(xs)
    elif max_x < np.max(xs):
        print(
            "[WARNING] the maximum X requested ({:f}) is smaller than the maximum X of the samples ({:f})",
            max_x,
            np.max(xs),
        )

    # Check the values...
    if min_y > max_y:
        raise Exception("the minimum Y cannot be larger than the maximum Y")
    if min_x > max_x:
        raise Exception("the minimum X cannot be larger than the maximum X")

    ys_1d = np.linspace(min_y, max_y, math.ceil((max_y - min_y) / raster_step))
    xs_1d = np.linspace(min_x, max_x, math.ceil((max_x - min_x) / raster_step))

    # Get the dimensions of the grid
    num_y = len(ys_1d)
    num_x = len(xs_1d)

    # Create the matrix of x/y coordinates out of the 1D arrays
    ys_mat = np.tile(ys_1d.reshape(-1, 1), (1, num_x))
    xs_mat = np.tile(xs_1d, (num_y, 1))

    # Create an EMPTY elevation data
    elevation = np.zeros_like(ys_mat)

    work_areas = create_work_areas(elevation, areas_kml_file, xs_1d, ys_1d)

    return xs, ys, elev, xs_mat, ys_mat, elevation, work_areas


def samples_to_grid(xs_ref, ys_ref, elevation_ref, ys_mat, xs_mat, elevation):
    ys_1d = ys_mat[:, 0]
    xs_1d = xs_mat[0, :]

    accum = np.zeros_like(elevation)
    num_elems = np.zeros_like(elevation)
    for x, y, elev in zip(xs_ref, ys_ref, elevation_ref):
        x_ind = find_nearest_ind(xs_1d, x)
        y_ind = find_nearest_ind(ys_1d, y)
        accum[y_ind, x_ind] += elev
        num_elems[y_ind, x_ind] += 1

    return accum / num_elems, num_elems == 0


def find_nearest_ind(array, value):
    # Modified version of the snippet in: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array (answer by Demitri)
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def rasterize(params):
    condp = ConditionalPrint(params.verbose)

    # Load the data of the interpolation problem
    if params.verbose:
        condp.print("- Loading data...", end="", flush=True)
        ts = timer()
    xs_ref, ys_ref, elevation_ref, xs_mat, ys_mat, elevation_int, work_areas = (
        load_interpolation_input_data_xyz(
            params.input_file,
            params.delimiter,
            params.cell_size,
            params.min_y,
            params.max_y,
            params.min_x,
            params.max_x,
            params.areas,
        )
    )

    if params.verbose:
        te = timer()
        condp.print(" done, {:.2f} sec.".format(te - ts))

    # Show a bit of information regarding the interpolation problem (percentage of missing data to interpolate w.r.t. the full image)
    if params.verbose:
        condp.print("- Summary of input data:")
        condp.print("    - Input XYZ has {:d} data points".format(xs_ref.shape[0]))
        condp.print(
            "    - Output elevation grid has a size of {:d}x{:d} cells".format(
                elevation_int.shape[0], elevation_int.shape[1]
            )
        )
        if params.areas:
            condp.print(
                "    - Data will be interpolated just at the user-defined areas"
            )
        else:
            total_cells = elevation_int.shape[0] * elevation_int.shape[1]
            num_cells_to_interpolate = len(elevation_int)
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
        scattered_methods = ["nearest", "linear", "cubic", "rbf", "purbf", "mlp"]
        if params.subparser_name.lower() in scattered_methods:
            mask_int = np.ones_like(
                elevation_int
            )  # Note: when using a scattered data interpolation method, ALL grid points will be interpolated

            # Cast the matrices to a set of "scattered" data points and references
            ys_int = ys_mat[cur_work_area]
            xs_int = xs_mat[cur_work_area]

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
                num_int = np.sum(cur_work_area)
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
            elevation_int[cur_work_area] = zi

        # --- Gridded data interpolation/inpainting ---
        gridded_methods = [
            "harmonic",
            "tv",
            "ccst",
            "amle",
            "navier-stokes",
            "telea",
            "shiftmap",
        ]
        if params.subparser_name.lower() in gridded_methods:
            elevation_int, mask_int = samples_to_grid(
                xs_ref, ys_ref, elevation_ref, ys_mat, xs_mat, elevation_int
            )

            # if params.areas:
            # Get the bounding box of the current working area (inpainters work on full 2D grids...)
            rows = np.any(cur_work_area, axis=1)
            cols = np.any(cur_work_area, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Extract this region from the image
            cur_inpaint_mask = np.copy(
                ~mask_int[rmin : rmax + 1, cmin : cmax + 1]
            )  # Inpainting mask (inverse of mask_int by our internal convention)
            cur_elevation = np.copy(elevation_int[rmin : rmax + 1, cmin : cmax + 1])
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
        write_interpolation_results_new_file(
            params.output_file,
            elevation_int,
            mask_int,
            ys_mat[:, 0],
            xs_mat[0, :],
            params.elevation_var,
            params.interpolation_flag_var,
        )

    # Show results
    if params.show:
        condp.print("- Showing results (close the window to continue)")

        # plt.imshow(elevation_int, origin='lower')
        plt.contourf(xs_mat, ys_mat, elevation_int, levels=50, cmap="viridis")
        plt.scatter(xs_ref, ys_ref, c=elevation_ref, s=10, alpha=0.3, cmap="viridis")
        plt.show(block=True)


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Interpolate/grid XYZ data to a netCDF4 file"
    )
    # Create a sub-parser for each possible interpolator, with its own options
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")
    parser.add_argument("input_file", action="store", type=str, help="Input XYZ file")
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
        help="Name of the variable storing the elevation grid in the OUTPUT file.",
    )
    parser.add_argument(
        "--x_var",
        action="store",
        type=str,
        default="lon",
        help="Name of the variable storing the columns' coordinates of the elevation grid in the output file.",
    )
    parser.add_argument(
        "--y_var",
        action="store",
        type=str,
        default="lat",
        help="Name of the variable storing the rows' coordinates of the elevation grid in the output file.",
    )
    parser.add_argument(
        "--x_dim",
        action="store",
        type=str,
        default="",
        help="Name of the dimension for the columns' coordinates of the elevation grid in the output file. Defaults to y_var if not set.",
    )
    parser.add_argument(
        "--y_dim",
        action="store",
        type=str,
        default="",
        help="Name of the dimension for the rows' coordinates of the elevation grid in the output file. Defaults to x_var if not set.",
    )
    parser.add_argument(
        "--interpolation_flag_var",
        action="store",
        type=str,
        default="interpolation_flag",
        help="Name of the variable storing the per-cell interpolation flag in the output file",
    )
    parser.add_argument(
        "--min_y",
        action="store",
        type=float,
        default=np.nan,
        help="Minimum Y of the output raster grid (if not set, will be computed from the samples' bounds).",
    )
    parser.add_argument(
        "--max_y",
        action="store",
        type=float,
        default=np.nan,
        help="Maximum Y of the output raster grid (if not set, will be computed from the samples' bounds).",
    )
    parser.add_argument(
        "--min_x",
        action="store",
        type=float,
        default=np.nan,
        help="Minimum X of the output raster grid (if not set, will be computed from the samples' bounds).",
    )
    parser.add_argument(
        "--max_x",
        action="store",
        type=float,
        default=np.nan,
        help="Maximum X of the output raster grid (if not set, will be computed from the samples' bounds).",
    )
    parser.add_argument(
        "--cell_size",
        action="store",
        type=float,
        required=True,
        help="Cell size of each cell in the interpolated raster, in the units of the XYZ",
    )
    parser.add_argument(
        "--delimiter",
        action="store",
        type=str,
        required=False,
        default=None,
        help="The string used to separate values. By default, any consecutive whitespaces act as delimiter. An integer or sequence of integers can also be provided as width(s) of each field.",
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
    rasterize(parse_args())


# Main function
if __name__ == "__main__":
    main()
