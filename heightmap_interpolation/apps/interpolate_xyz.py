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
from timeit import default_timer as timer

import numpy as np

from heightmap_interpolation.apps.apps_common import (
    add_common_args,
    add_subparsers,
    get_available_gridded_methods,
    get_available_scattered_methods,
    run_gridded_inpainting,
    run_scattered_interpolation,
    show_interpolation_results,
)
from heightmap_interpolation.apps.netcdf_data_io import (
    create_work_areas,
    write_interpolation_results_new_file,
)
from heightmap_interpolation.misc.conditional_print import ConditionalPrint
import math


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
            "[WARNING] the minimum Y requested ({:f}) is larger than the minimum Y of the samples ({:f})".format(
                min_y,
                np.min(ys),
            )
        )
    if np.isnan(min_x):
        min_x = np.min(xs)
    elif min_x > np.min(xs):
        print(
            "[WARNING] the minimum X requested ({:f}) is larger than the minimum X of the samples ({:f})".format(
                min_x,
                np.min(xs),
            )
        )
    if np.isnan(max_y):
        max_y = np.max(ys)
    elif max_y < np.max(ys):
        print(
            "[WARNING] the maximum Y requested ({:f}) is smaller than the maximum Y of the samples ({:f})".format(
                max_y,
                np.max(ys),
            )
        )
    if np.isnan(max_x):
        max_x = np.max(xs)
    elif max_x < np.max(xs):
        print(
            "[WARNING] the maximum X requested ({:f}) is smaller than the maximum X of the samples ({:f})".format(
                max_x,
                np.max(xs),
            )
        )

    # Check the values...
    if min_y > max_y:
        raise Exception("the minimum Y cannot be larger than the maximum Y")
    if min_x > max_x:
        raise Exception("the minimum X cannot be larger than the maximum X")

    # Grid cells at exactly raster_step spacing, starting at the minimum coordinate and
    # covering up to the maximum one (to the nearest cell)
    num_y = int(round((max_y - min_y) / raster_step)) + 1
    num_x = int(round((max_x - min_x) / raster_step)) + 1
    ys_1d = min_y + raster_step * np.arange(num_y)
    xs_1d = min_x + raster_step * np.arange(num_x)

    # Create the matrix of x/y coordinates out of the 1D arrays
    ys_mat = np.tile(ys_1d.reshape(-1, 1), (1, num_x))
    xs_mat = np.tile(xs_1d, (num_y, 1))

    # Create an EMPTY elevation data
    elevation = np.full_like(ys_mat, np.nan)

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

    empty = num_elems == 0
    elevation = np.divide(
        accum, num_elems, out=np.full_like(accum, np.nan), where=~empty
    )
    return elevation, empty


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

    scattered_methods = get_available_scattered_methods()
    gridded_methods = get_available_gridded_methods()

    requested_method = params.subparser_name.lower()
    if (
        requested_method not in scattered_methods
        and requested_method not in gridded_methods
    ):
        raise ValueError(
            f"Experimental method {requested_method} requested, but the experimental dependencies were not installed (use pip install heightmap_interpolation[experimental])"
        )
    is_scattered = requested_method in scattered_methods

    # Bin the input samples into the output grid: cells containing samples get their
    # (mean) value, the rest are NaN and are the ones to interpolate (mask_int). Cells
    # outside the work areas keep these values.
    elevation_src, mask_int = samples_to_grid(
        xs_ref, ys_ref, elevation_ref, ys_mat, xs_mat, elevation_int
    )
    elevation_int = np.copy(elevation_src)
    in_work_areas = np.any(work_areas, axis=2)

    # Cells that will be interpolated: all the cells within the work areas for the
    # scattered methods, just the empty ones within the work areas for the gridded ones
    if is_scattered:
        interpolated = in_work_areas
    else:
        interpolated = np.logical_and(mask_int, in_work_areas)

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
        total_cells = elevation_int.shape[0] * elevation_int.shape[1]
        num_cells_to_interpolate = int(np.count_nonzero(interpolated))
        interp_percent = (num_cells_to_interpolate / total_cells) * 100
        condp.print(
            "    - Cells to interpolate represent {:.2f}% of the image:".format(
                interp_percent
            )
        )
        condp.print("        - Total cells = {:d}".format(total_cells))
        condp.print(
            "        - Number of cells with samples = {:d}".format(
                int(np.count_nonzero(~mask_int))
            )
        )
        condp.print(
            "        - Number of cells to interpolate = {:d}".format(
                num_cells_to_interpolate
            )
        )

    for i in range(work_areas.shape[2]):
        cur_work_area = work_areas[:, :, i]

        # --- Scattered data interpolation ---
        if is_scattered:
            # All grid points within the work area will be interpolated
            xs_int = xs_mat[cur_work_area]
            ys_int = ys_mat[cur_work_area]
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
            zi = run_scattered_interpolation(
                params, xs_ref, ys_ref, elevation_ref, xs_int, ys_int, condp
            )
            elevation_int[cur_work_area] = zi

        # --- Gridded data interpolation/inpainting ---
        else:
            run_gridded_inpainting(
                params,
                elevation_src,
                elevation_int,
                mask_int,
                xs_mat,
                ys_mat,
                cur_work_area,
                i,
                work_areas.shape[2],
                condp,
            )

    if is_scattered:
        # Some scattered methods return NaN at some points (e.g., linear/cubic on the
        # boundary of the convex hull of the data): keep the samples' value there
        restore = interpolated & np.isnan(elevation_int) & ~mask_int
        elevation_int[restore] = elevation_src[restore]
        interpolated = interpolated & ~restore

    # Clip the interpolated values to the input data range to remove overshoots
    if params.truncate_to_input_range:
        np.clip(
            elevation_int, np.nanmin(elevation_ref), np.nanmax(elevation_ref),
            out=elevation_int,
        )

    # Write the results
    if params.output_file:
        condp.print("- Writing the results to disk")
        write_interpolation_results_new_file(
            params.output_file,
            elevation_int,
            interpolated,
            ys_mat[:, 0],
            xs_mat[0, :],
            params.elevation_var,
            params.interpolation_flag_var,
        )

    # Show results
    if params.show:
        condp.print("- Showing results (close the window to continue)")
        show_interpolation_results(
            elevation_src,
            elevation_int,
            interpolated,
            xs_mat,
            ys_mat,
            x_var_name=params.x_var,
            y_var_name=params.y_var,
            colormap=params.colormap,
            scatter_xs=xs_ref,
            scatter_ys=ys_ref,
            scatter_values=elevation_ref,
            truncate_to_input_range=params.truncate_to_input_range,
        )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Interpolate/grid XYZ data to a netCDF4 file"
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")
    parser.add_argument("input_file", action="store", type=str, help="Input XYZ file")
    add_common_args(parser, interpolation_flag_var_default="interpolation_flag")
    parser.add_argument(
        "--min_y",
        action="store",
        type=float,
        default=np.nan,
        help="Minimum Y of the output raster grid"
        " (if not set, will be computed from the samples' bounds).",
    )
    parser.add_argument(
        "--max_y",
        action="store",
        type=float,
        default=np.nan,
        help="Maximum Y of the output raster grid"
        " (if not set, will be computed from the samples' bounds).",
    )
    parser.add_argument(
        "--min_x",
        action="store",
        type=float,
        default=np.nan,
        help="Minimum X of the output raster grid"
        " (if not set, will be computed from the samples' bounds).",
    )
    parser.add_argument(
        "--max_x",
        action="store",
        type=float,
        default=np.nan,
        help="Maximum X of the output raster grid"
        " (if not set, will be computed from the samples' bounds).",
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
        help="The string used to separate values. By default, any consecutive"
        " whitespaces act as delimiter. An integer or sequence of integers can"
        " also be provided as width(s) of each field.",
    )
    add_subparsers(subparsers)
    return parser.parse_args(args)


def main():
    rasterize(parse_args())


# Main function
if __name__ == "__main__":
    main()
