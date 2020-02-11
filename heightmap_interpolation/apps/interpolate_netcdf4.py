#!/usr/bin/env python3

# Copyright (c) 2020 Coronis Computing S.L. (Spain)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Ricard Campos (ricard.campos@coronis.es)

import argparse
import netCDF4 as nc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import shutil
from heightmap_interpolation.interpolants.rbf_interpolant import  RBFInterpolant
import heightmap_interpolation.apps.bivariate_functions as demo_functions
from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant
from heightmap_interpolation.interpolants.quad_tree_pu_rbf_interpolant import QuadTreePURBFInterpolant
from timeit import default_timer as timer
from heightmap_interpolation.misc.conditional_print import ConditionalPrint
import geopandas as gpd
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
from PIL import Image, ImageDraw


def imageToArray(i):
    """
    Converts a Python Imaging Library array to a
    numpy array.
    """
    a=np.fromstring(i.tobytes(),'b')
    a.shape=i.im.size[1], i.im.size[0]
    return a

# Main function
if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser(
        description="Interpolate terrain data in a SeaDataNet_1.0 CF1.6-compliant netCDF4 file")
    parser.add_argument("input_file", action="store", type=str,
                        help="Input NetCDF file")
    parser.add_argument("output_file", action="store", type=str,
                        help="Output NetCDF file with interpolated values for cells in which the cell_interpolation_flag was not false")
    parser.add_argument("--areas", action="store", type=str, default="",
                        help="KML file containing the areas that will be interpolated.")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    parser.add_argument("--cell_interpolation_flag_var", action="store", type=str, default="cell_interpolation_flag",
                        help="Name of the variable storing the per-cell interpolation flag in the input file")
    parser.add_argument("--query_block_size", action="store", default=1000, type=int,
                        help="Query the interpolant in blocks of maximum this size, in order to avoid having to store large matrices in memory")
    parser.add_argument("--rbf_max_ref_points", action="store", type=int, default=10000,
                        help="Maximum number of data points to use a single RBF interpolation. Datasets with a number of reference points greater than this will use a partition of unity")
    parser.add_argument("--rbf_distance_type", action="store", type=str, default="euclidean",
                        help="Distance type. Available: euclidean, haversine, vincenty(default)")
    parser.add_argument("--rbf_type", action="store", type=str, default="thinplate",
                        help="RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland")
    parser.add_argument("--rbf_epsilon", action="store", type=float, default=1,
                        help="Epsilon parameter of the RBF. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland")
    parser.add_argument("--rbf_regularization", action="store", type=float, default=0,
                        help="Regularization scalar to use in the RBF (optional)")
    parser.add_argument("--rbf_polynomial_degree", action="store", type=int, default=1,
                        help="Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic)")
    parser.add_argument("--pu_overlap", action="store", type=float, default=0.25,
                        help="(Just if PU is used) Overlap factor between circles in neighboring sub-domains in the partition. The radius of a QuadTree cell, computed as half its diagonal, is enlarged by this factor")
    parser.add_argument("--pu_min_point_in_cell", action="store", type=int, default=1000,
                        help="(Just if PU is used) Minimum number of points in a QuadTree cell")
    parser.add_argument("--pu_min_cell_size_percent", action="store", type=float, default=0.005,
                        help="(Just if PU is used) Minimum cell size, specified as a percentage [0..1] of the max(width, height) of the query domain")
    parser.add_argument("--pu_overlap_increment", action="store", type=float, default=0.001,
                        help="(Just if PU is used) If, after creating the QuadTree, a cell contains less than pu_min_point_in_cell, the radius will be iteratively incremented until this condition is satisfied. This parameter specifies how much the radius of a cell increments at each iteration")
    parser.add_argument("-v, --verbose", action="store_true", dest="verbose", default=False,
                        help="Verbosity flag, activate it to have feedback of the current steps of the process in the command line")
    parser.add_argument("-s, --show", action="store_true", dest="show", default=False,
                        help="Show interpolation problem and results on screen")
    param = parser.parse_args()

    # Conditional print
    cp = ConditionalPrint(param.verbose)

    # Read the file
    ds = nc.Dataset(param.input_file, "r", format="NETCDF4")

    # Get the dimensions of the grid
    num_lat = len(ds.dimensions["lat"])
    num_lon = len(ds.dimensions["lon"])

    # Get the lat/lon coordinates
    lats_1d = ds.variables["lat"][:]
    lons_1d = ds.variables["lon"][:]

    # Get the elevation data
    elevation = ds.variables[param.elevation_var][:]

    # Get a mask with the values to interpolate and the reference (known) valuesfrom the interpolation flag per-cell
    mask_int = ds.variables["cell_interpolation_flag"][:]
    mask_int = mask_int == 1 # Convert to boolean!
    mask_ref = np.logical_not(mask_int)

    # If the elevation field is masked, we just focus on the values of reference/to interpolate
    # that are in the valid area
    if np.ma.is_masked(elevation):
        mask_int[elevation.mask] = False
        mask_ref[elevation.mask] = False

    # Create the matrix of lat/lon coordinates out of the 1D arrays
    lats_mat = np.tile(lats_1d.reshape(-1, 1), (1, num_lon))
    lons_mat = np.tile(lons_1d, (num_lat, 1))

    # Are we using a KML to restrict the interpolation?
    if param.areas:
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

    # Compute the number of reference points and points to interpolate
    num_int = np.sum(mask_int)
    num_ref = np.sum(mask_ref)

    cp.print("Interpolation problem:")
    cp.print("  - Queries   =", str(num_int))
    cp.print("  - Reference =", str(num_ref))

    # And mask them to get the reference lat/lon/depth
    lats_ref = lats_mat[mask_ref]
    lons_ref = lons_mat[mask_ref]
    elevation_ref = elevation[mask_ref]
    lats_int = lats_mat[mask_int]
    lons_int = lons_mat[mask_int]

    # Use the reference values to create the interpolant
    if num_ref < param.rbf_max_ref_points:
        # Use a RBF interpolant
        cp.print("Creating the interpolant (RBF)...", end='', flush=True)
        ts = timer()
        interpolant = RBFInterpolant(lons_ref, lats_ref, elevation_ref,
                                     rbf_type=param.rbf_type,
                                     distance_type=param.rbf_distance_type,
                                     epsilon=param.rbf_epsilon,
                                     regularization=param.rbf_regularization,
                                     polynomial_degree=param.rbf_polynomial_degree
                                     )
        te = timer()
        cp.print("done, {:.2f} sec.".format(te-ts))
    else:
        # Use a QuadTreePURBF interpolant
        cp.print("Creating the interpolant (QuadTreePURBF)...", end='', flush=True)
        ts = timer()
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
                                               min_points_in_cell=param.pu_min_point_in_cell,
                                               overlap=param.pu_overlap,
                                               overlap_increment=param.pu_overlap_increment,
                                               min_cell_size_percent=param.pu_min_cell_size_percent,
                                               rbf_type=param.rbf_type,
                                               distance_type=param.rbf_distance_type,
                                               epsilon=param.rbf_epsilon,
                                               regularization=param.rbf_regularization,
                                               polynomial_degree=param.rbf_polynomial_degree
                                               )
        te = timer()
        cp.print("done, {:.2f} sec.".format(te - ts))
        if param.verbose:
            interpolant.show_interpolant_stats()

    # Apply the interpolant at query locations in chuncks (to avoid storing too large matrices in memory)
    cp.print("Interpolating...", end='', flush=True)

    # Divide the data into blocks
    zi = np.zeros(lons_int.shape)
    num_blocks = math.ceil(num_int/param.query_block_size)
    block_start = 0
    block_end = min([num_int, param.query_block_size])

    # Interpolate
    ts = timer()
    for i in range(num_blocks):
        message = "Querying block {}/{}".format(i+1, num_blocks)
        cp.print(message)
        cp.backspace(len(message))

        zi[block_start:block_end] = interpolant(lons_int[block_start:block_end], lats_int[block_start:block_end])
        block_end = min([block_end+param.query_block_size, num_int])
        block_start = block_start+param.query_block_size
    te = timer()
    cp.print("done, {:.2f} sec.".format(te-ts))

    elevation[mask_int] = zi

    # Replace the elevation data on the NetCDF dataset by the new one
    cp.print("Writing results to disk...", end='', flush=True)
    ts = timer()

    # We just want to modify the elevation variable, while retaining the rest of the dataset as is, so the easiest
    # solution is to copy the input file to the destination file, and open it in write mode to change the elevation
    # variable
    shutil.copy(param.input_file, param.output_file)
    out_ds = nc.Dataset(param.output_file, "r+")
    out_ds.variables["elevation"][:] = elevation
    if param.areas:
        # Also update the interpolated areas
        new_cell_interpolated_flag = ds.variables["cell_interpolation_flag"][:]
        new_cell_interpolated_flag[mask_int] = 1
        out_ds.variables["cell_interpolation_flag"][:] = new_cell_interpolated_flag
    out_ds.close()

    te = timer()
    cp.print("done, {:.2f} sec.".format(te - ts))

    if param.show:
        cp.print("Showing results")
        # Show the initial data
        fig = plt.figure()
        ax = []
        sp_rows = 1
        if num_ref < param.rbf_max_ref_points:
            sp_cols = 3
        else:
            sp_cols = 4
        sp_ind = 0
        # Show the original elevation map
        ax.append(fig.add_subplot(sp_rows, sp_cols, sp_ind+1, projection="rectilinear"))
        elevation_ref_mat = ds.variables[param.elevation_var][:]
        elevation_ref_mat[~mask_ref] = float('nan')
        vmin = elevation.min()
        vmax = elevation.max()
        ax[sp_ind].imshow(elevation_ref_mat, origin='lower', vmin=vmin, vmax=vmax)
        ax[sp_ind].axis('equal')
        ax[sp_ind].set_aspect('equal', 'box')
        ax[sp_ind].title.set_text('Reference Data')
        plt.show(block=False)
        sp_ind = sp_ind + 1
        # Show the mask of points to interpolate
        ax.append(fig.add_subplot(sp_rows, sp_cols, sp_ind + 1, projection="rectilinear"))
        ax[sp_ind].imshow(mask_int, origin='lower', cmap='gray')
        ax[sp_ind].axis('equal')
        ax[sp_ind].set_aspect('equal', 'box')
        ax[sp_ind].title.set_text('Points to interpolate')
        plt.show(block=False)
        sp_ind = sp_ind + 1
        if num_ref >= param.rbf_max_ref_points:
            # Show the QuadTree structure
            ax.append(fig.add_subplot(sp_rows, sp_cols, sp_ind + 1, projection="rectilinear"))
            interpolant.plot(ax[sp_ind])
            ax[sp_ind].axis('equal')
            ax[sp_ind].set_aspect('equal', 'box')
            ax[sp_ind].title.set_text('Query Domain Decomposition')
            plt.show(block=False)
            sp_ind = sp_ind+1
        # Show the final result
        ax.append(fig.add_subplot(sp_rows, sp_cols, sp_ind+1, projection="rectilinear"))
        elevation[~mask_int] = float('nan')
        ax[sp_ind].imshow(elevation, origin='lower', vmin=vmin, vmax=vmax)
        ax[sp_ind].axis('equal')
        ax[sp_ind].set_aspect('equal', 'box')
        ax[sp_ind].title.set_text('Interpolated')
        # plt.colorbar()
        plt.show()

    ds.close()
