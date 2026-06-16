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
    load_interpolation_input_data,
    write_interpolation_results,
)
from heightmap_interpolation.misc.conditional_print import ConditionalPrint


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
                "    - Cells to interpolate represent {:.2f}% of the image:".format(
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

    # Methods available
    scattered_methods = get_available_scattered_methods()
    gridded_methods = get_available_gridded_methods()

    requested_method = params.subparser_name.lower()
    if (
        requested_method not in scattered_methods
        and requested_method not in gridded_methods
    ):
        # Since the requested method is specified via subparser, the only option for not being present on the list is that is an experimental feature and the package was not requested to include them when installed
        raise ValueError(
            f"Experimental method {requested_method} requested, but the experimental dependencies were not installed (use pip install heightmap_interpolation[experimental])"
        )

    for i in range(work_areas.shape[2]):
        cur_work_area = work_areas[:, :, i]

        # --- Scattered data interpolation ---
        if params.subparser_name.lower() in scattered_methods:
            cur_mask_ref = np.logical_and(mask_ref, cur_work_area)
            cur_mask_int = np.logical_and(mask_int, cur_work_area)
            xs_ref = xs_mat[cur_mask_ref]
            ys_ref = ys_mat[cur_mask_ref]
            elevation_ref = elevation[cur_mask_ref]
            xs_int = xs_mat[cur_mask_int]
            ys_int = ys_mat[cur_mask_int]
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
            elevation_int[cur_mask_int] = zi

        # --- Gridded data interpolation/inpainting ---
        if params.subparser_name.lower() in gridded_methods:
            run_gridded_inpainting(
                params, elevation, elevation_int, mask_int,
                cur_work_area, i, work_areas.shape[2], condp
            )

    # Clip the interpolated values to the input data range to remove overshoots
    if params.truncate_to_input_range:
        np.clip(
            elevation_int, np.nanmin(elevation), np.nanmax(elevation),
            out=elevation_int,
        )

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
        condp.print("- Showing results (close the emerging window to finish)")
        show_interpolation_results(
            elevation,
            elevation_int,
            mask_int,
            xs_mat,
            ys_mat,
            x_var_name=params.x_var,
            y_var_name=params.y_var,
            colormap=params.colormap,
            highlight_interpolated_area=params.highlight_interpolated_area,
            truncate_to_input_range=params.truncate_to_input_range,
        )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Interpolate elevation data in a netCDF4 file"
        " (defaults set to read an EMODnet Bathymetry"
        " SeaDataNet_1.0 CF1.6-compliant file)"
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")
    parser.add_argument(
        "input_file", action="store", type=str, help="Input NetCDF file"
    )
    add_common_args(parser, interpolation_flag_var_default=None)
    add_subparsers(subparsers)
    return parser.parse_args(args)


def main():
    interpolate(parse_args())


# Main function
if __name__ == "__main__":
    main()
