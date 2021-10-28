#!/usr/bin/env python3

# Copyright (c) 2020 Coronis Computing S.L. (Spain)
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

# Common functionalities of the inpaint and interpolate apps

import netCDF4 as nc
import numpy as np
from PIL import Image, ImageDraw
import geopandas as gpd
import shutil
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'


def imageToArray(i):
    """
    Converts a Python Imaging Library array to a numpy array.
    """
    a = np.fromstring(i.tobytes(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a


def load_data(param):
    """Loads the data required for interpolation"""
    return load_data_impl(param.input_file, param.elevation_var, param.interpolate_missing_values, param.areas)


def load_data_impl(input_file, elevation_var, interpolate_missing_values, areas = None):

    # Read the file
    ds = nc.Dataset(input_file, "r", format="NETCDF4")

    # Get the dimensions of the grid
    num_lat = len(ds.dimensions["lat"])
    num_lon = len(ds.dimensions["lon"])

    # Get the lat/lon coordinates
    lats_1d = ds.variables["lat"][:]
    lons_1d = ds.variables["lon"][:]

    # Get the elevation data
    elevation = ds.variables[elevation_var][:]

    # Get a mask with the values to interpolate and the reference (known) values
    if not interpolate_missing_values:
        # Get a mask with the values to interpolate and the reference (known) valuesfrom the interpolation flag per-cell
        # we do not recompute interpolated area if interpolate_missing_values is set to true
        mask_int = ds.variables["interpolation_flag"][:]
        mask_int = mask_int == 1  # Convert to boolean!
        mask_ref = np.logical_not(mask_int)

        # If the elevation field is masked, we just focus on the values of reference/to interpolate
        # that are in the valid area
        if np.ma.is_masked(elevation):
            mask_int[elevation.mask] = False  # turn to true to interpolate everywhere bathymetry is empty
            mask_ref[elevation.mask] = False
    else:
        if np.ma.is_masked(elevation):
            mask_ref = ~elevation.mask
            mask_int = elevation.mask
        else:
            # no invalid value, exit
            return
    ds.close()

    # Change the masked array to a simple array
    elevation = elevation.data
    # Change NaNs by 0s
    elevation[np.isnan(elevation)] = 0

    # Create the matrix of lat/lon coordinates out of the 1D arrays
    lats_mat = np.tile(lats_1d.reshape(-1, 1), (1, num_lon))
    lons_mat = np.tile(lons_1d, (num_lat, 1))

    # Are we using a KML to restrict the interpolation?
    if areas:
        # Then, the areas to interpolate are only defined by the polygons in the file
        mask_int.fill(False)

        # Read the KML file using geopandas
        df = gpd.read_file(param.areas, driver='KML')

        # Get minimum lat/lon and pixel resolution
        xmin, ymin, xmax, ymax = [lons_1d.min(), lats_1d.min(), lons_1d.max(), lats_1d.max()]
        xres = (xmax - xmin) / float(num_lon)
        yres = (ymax - ymin) / float(num_lat)

        # Create a raster with the same size as the input map
        rasterPoly = Image.new("L", (num_lon, num_lat), 0)
        rasterize = ImageDraw.Draw(rasterPoly)

        for poly in df.geometry:
            # Get the coordinates of the polygon
            x, y = poly.exterior.coords.xy

            # Convert the coordinates to pixels (assuming all in the same CRS)
            x = np.round((x - xmin) / xres).astype(np.int)
            y = np.round((y - ymin) / yres).astype(np.int)

            # Rasterize the polygon
            listdata = [(x[i], y[i]) for i in range(len(x))]
            rasterize.polygon(listdata, 1)
        # Extract the mask out of the raster
        mask_int = imageToArray(rasterPoly) == 1
        mask_int = np.logical_and(mask_int, ~mask_ref)

    return elevation, mask_int, lats_mat, lons_mat, mask_ref


def write_results(param, elevation, mask_int):
    write_results_impl(param.output_file, param.input_file, elevation, mask_int, elevation_var=param.elevation_var, areas=param.areas, interpolate_missing_values=param.interpolate_missing_values)


def write_results_impl(output_file, input_file, elevation, mask_int, elevation_var = "elevation", areas = None, interpolate_missing_values=False):
    # We just want to modify the elevation variable, while retaining the rest of the dataset as is, so the easiest
    # solution is to copy the input file to the destination file, and open it in write mode to change the elevation
    # variable
    if output_file is not None:
        shutil.copy(input_file, output_file)
        out_ds = nc.Dataset(output_file, "r+")
    else:
        out_ds = nc.Dataset(input_file, "r+")

    out_ds.variables[elevation_var][:] = elevation
    if areas or interpolate_missing_values:
        # Also update the interpolated areas
        if "interpolation_flag" not in out_ds.variables.keys():
            out_ds.createVariable('interpolation_flag', 'int8', ('lat', 'lon'))
            new_cell_interpolated_flag = out_ds.variables["interpolation_flag"][:]
            new_cell_interpolated_flag[~mask_int] = 0
        else:
            new_cell_interpolated_flag = out_ds.variables["interpolation_flag"][:]
        new_cell_interpolated_flag[mask_int] = 1

        # # Create the interpolation flag, if was not present in the input
        # new_cell_interpolated_flag[mask_int] = 1
        out_ds.variables["interpolation_flag"][:] = new_cell_interpolated_flag

