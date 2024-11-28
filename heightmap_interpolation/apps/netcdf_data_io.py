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

import netCDF4 as nc
import numpy as np
from PIL import Image, ImageDraw
import geopandas as gpd
import shutil
import cv2
from shapely.geometry import Point
from fiona.drvsupport import supported_drivers
supported_drivers['KML'] = 'rw'


def imageToArray(i):
    """
    Converts a Python Imaging Library array to a numpy array.
    """
    a = np.fromstring(i.tobytes(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a


def load_interpolation_input_data(input_file, elevation_var, interpolation_flag_var=None, areas_kml_file=None):
    """Loads the data required for interpolation from the NetCDF file

    :param input_file: path to the input file (expected NetCDF4 format)
    :param elevation_var: the name of the variable within the netcdf4 file representing the elevation
    :param interpolation_flag_var: the name of the variable representing an "interpolation flag", where 0 == reference data, 1 == interpolated point (or point to interpolate).
    :param areas_kml_file: the KML file with a set of areas where the interpolation should take place. Areas out of the described polygons will be left as they are.
    """

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
    if interpolation_flag_var:
        # Get a mask with the values to interpolate and the reference (known) valuesfrom the interpolation flag per-cell
        # we do not recompute interpolated area if interpolate_missing_values is set to true
        mask_int = ds.variables[interpolation_flag_var][:]
        mask_int = mask_int == 1  # Convert to boolean!
        mask_ref = np.logical_not(mask_int)

        # If the elevation field is masked, we just focus on the values of reference/to interpolate
        # that are in the valid area
        if np.ma.is_masked(elevation):
            mask_int[elevation.mask] = False  # turn to true to interpolate everywhere bathymetry is empty
            mask_ref[elevation.mask] = False
    else:
        # Get the "invalid" values out of the elevation matrix
        if np.ma.is_masked(elevation):
            mask_ref = ~elevation.mask
            mask_int = elevation.mask
        else:
            # no invalid value, exit
            raise RuntimeError("The elevation var on the input NetCDF does neither have an interpolation_flag variable nor a set of masked values. Therefore, there is nothing to interpolate!")
    ds.close()

    # Create the matrix of lat/lon coordinates out of the 1D arrays
    lats_mat = np.tile(lats_1d.reshape(-1, 1), (1, num_lon))
    lons_mat = np.tile(lons_1d, (num_lat, 1))

    work_areas = create_work_areas(elevation, areas_kml_file, lons_1d, lats_1d)

    return lats_mat, lons_mat, elevation, mask_int, mask_ref, work_areas


def write_interpolation_results(input_file, output_file, elevation, mask_int, elevation_var, interpolation_flag_var=None, areas_kml_file=None):
    # We just want to modify the elevation variable, while retaining the rest of the dataset as is, so the easiest
    # solution is to copy the input file to the destination file, and open it in write mode to change the elevation
    # variable
    if output_file:
        shutil.copy(input_file, output_file)
    else:
        raise ValueError("Missing output file path!")

    out_ds = nc.Dataset(output_file, "r+")
    out_ds.variables[elevation_var][:] = elevation
    if areas_kml_file or not interpolation_flag_var:
        # Also update the interpolated areas
        if "interpolation_flag" not in out_ds.variables.keys():

            # retrieve compression parameters
            elev_var = out_ds.variables[elevation_var]
            complib = "zlib" if elev_var.filters() is not None and elev_var.filters().get("zlib", False) else None
            complevel = elev_var.filters().get("complevel", 0) if elev_var.filters() is not None else 0
            chunksizes = (
                tuple(elev_var.chunking())
                if elev_var.chunking() is not None and elev_var.chunking() != "contiguous"
                else None
            )

            out_ds.createVariable(
                "interpolation_flag",
                "int8",
                ("lat", "lon"),
                compression=complib,
                complevel=complevel,
                chunksizes=chunksizes,
            )
            new_cell_interpolated_flag = out_ds.variables["interpolation_flag"][:]
            new_cell_interpolated_flag[~mask_int] = 0
        else:
            new_cell_interpolated_flag = out_ds.variables["interpolation_flag"][:]
        new_cell_interpolated_flag[mask_int] = 1

        # # Create the interpolation flag, if was not present in the input
        # new_cell_interpolated_flag[mask_int] = 1
        out_ds.variables["interpolation_flag"][:] = new_cell_interpolated_flag

    out_ds.close()


def write_interpolation_results_new_file(output_file, elevation, mask_int, lats, lons, elevation_var_name, interpolation_flag_var_name):
    # We just want to modify the elevation variable, while retaining the rest of the dataset as is, so the easiest
    # solution is to copy the input file to the destination file, and open it in write mode to change the elevation
    # variable
    
    # Open the output dataset for writing
    out_ds = nc.Dataset(output_file, "w", format='NETCDF4')

    # Create the dimensions (lat/lon)
    lat_dim = out_ds.createDimension('lat', len(lats))     # latitude axis
    lon_dim = out_ds.createDimension('lon', len(lons))

    # Create attributes
    out_ds.flag_values = np.array([0, 1], dtype=np.int16)
    out_ds.flag_meaning = "not_interpolated interpolated"

    # Create variables
    lat = out_ds.createVariable('lat', np.float64, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lat[:] = lats
    lon = out_ds.createVariable('lon', np.float64, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    lon[:] = lons
    elevation_var = out_ds.createVariable(elevation_var_name, np.float64, ('lat', 'lon'))
    elevation_var[:, :] = elevation
    interpolation_flag_var = out_ds.createVariable(interpolation_flag_var_name, 'int8', ('lat', 'lon'))
    interpolation_flag_var[:, :] = mask_int

    out_ds.close()


def create_work_areas(elevation, areas_kml_file, lons_1d, lats_1d):
        # Are we using a KML to restrict the interpolation?
    if areas_kml_file:
        # Read the KML file using geopandas
        df = gpd.read_file(areas_kml_file, driver='KML')

        # Collect all polygons
        polys = []
        types = df.geom_type
        for ind in range(len(types)):
            type = types[ind]
            poly = df.geometry[ind]
            if not poly.is_empty:
                if type == 'Polygon':
                    polys.append(poly)
                elif type == 'MultiPolygon':
                    polys.extend(poly.geoms)

        # Create an array of work areas, each 3rd dimension plane represents a mask delimiting the working area
        work_areas = np.full((elevation.shape[0], elevation.shape[1], len(polys)), False)

        # i = 0
        # for poly in df.geometry:
        #     # Check all the cells falling within the polygon
        #     for ix, iy in np.ndindex(elevation.shape):
        #         work_areas[ix, iy, i] = poly.contains(Point(lons_mat[ix, iy], lats_mat[ix, iy]))
        #
        #     # Sanity check
        #     if not np.any(work_areas[:, :, i]):
        #         raise ValueError("One of the input areas does not contain any point within the map!")
        #
        #     i = i+1

        # Get minimum lat/lon and pixel resolution
        xmin, ymin, xmax, ymax = [lons_1d.min(), lats_1d.min(), lons_1d.max(), lats_1d.max()]
        num_lon = len(lons_1d)
        num_lat = len(lats_1d)
        xres = (xmax - xmin) / float(num_lon)
        yres = (ymax - ymin) / float(num_lat)

        i = 0
        for poly in polys:
            # Get the coordinates of the polygon
            x, y = poly.exterior.coords.xy

            # Convert the coordinates to pixels (assuming all in the same CRS)
            x = np.round((x - xmin) / xres).astype(np.int32)
            y = np.round((y - ymin) / yres).astype(np.int32)

            poly_points_cv = np.zeros((len(x), 2))
            for p in range(len(x)):                                                                                                                                                                                                                                                                     
                poly_points_cv[p, :] = np.array([x[p], y[p]])
            poly_points_cv = np.int32([poly_points_cv])  # Bug with fillPoly, needs explict cast to 32bit

            # Rasterize
            color = 255
            area_mask = np.zeros((elevation.shape[0], elevation.shape[1]), dtype='uint8')
            cv2.fillPoly(area_mask, poly_points_cv, color, lineType=cv2.LINE_AA)

            work_areas[:, :, i] = area_mask > 0

            # Sanity check
            if not np.any(work_areas[:, :, i]):
                raise ValueError("One of the polygon areas of the KML does not contain any point within the map!")

            i = i+1

        # # Create a raster with the same size as the input map
        # rasterPoly = Image.new("L", (num_lon, num_lat), 0)
        # rasterize = ImageDraw.Draw(rasterPoly)
        #
        # i = 0
        # for poly in df.geometry:
        #     # Get the coordinates of the polygon
        #     x, y = poly.exterior.coords.xy
        #
        #     # Convert the coordinates to pixels (assuming all in the same CRS)
        #     x = np.round((x - xmin) / xres).astype(np.int)
        #     y = np.round((y - ymin) / yres).astype(np.int)
        #
        #     # Rasterize the polygon
        #     listdata = [(x[i], y[i]) for i in range(len(x))]
        #     rasterize.polygon(listdata, 1)
        #     # Extract the mask out of the raster
        #     work_areas[:, :, i] = imageToArray(rasterPoly) == 1
        #     i = i+1


    else:
        work_areas = np.full((elevation.shape[0], elevation.shape[1], 1), True)

    return work_areas
