import sys
import argparse
import cv2 
import numpy as np
import netCDF4 as nc
import shutil
from heightmap_interpolation.apps.netcdf_data_io import load_interpolation_input_data


def main():
    parser = argparse.ArgumentParser("Converts a NetCDF4 grid to an XYZ file")
    parser.add_argument('input_file', help="The input elevation map to inpaint (in NetCDF4 format)")
    parser.add_argument("-o","--output_file", dest="output_file", action="store", type=str, required=True,
                        help="Output XYZ file with the cells with non-nan/interpolated values")
    parser.add_argument("--output_missing", dest="output_missing", action="store", type=str, required=False,
                        help="Output XY file with missing cells on the grid (i.e., the ones that would be interpolated by the other apps in this package)")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    parser.add_argument("--x_var", action="store", type=str, default="lon",
                        help="Name of the variable storing the columns' coordinates of the elevation grid in the input file.")
    parser.add_argument("--y_var", action="store", type=str, default="lat",
                        help="Name of the variable storing the rows' coordinates of the elevation grid in the input file.")    
    parser.add_argument("--x_dim", action="store", type=str, default="",
                        help="Name of the dimension for the columns' coordinates of the elevation grid in the input file. Defaults to x_var if not set.")
    parser.add_argument("--y_dim", action="store", type=str, default="",
                        help="Name of the dimension for the rows' coordinates of the elevation grid in the input file. Defaults to y_var if not set.")
    parser.add_argument("--interpolation_flag_var", action="store", type=str, default=None,
                        help="Name of the variable storing the per-cell interpolation flag in the input file (0 == known value, 1 == interpolated/to interpolate cell). If not set, it will interpolate the locations in the elevation variable containing an invalid (NaN) value.")
    parser.add_argument("--delimiter", action="store", type=str, default=" ", help="Delimeter for the output file")
    args = parser.parse_args()

    xs_mat, ys_mat, elevation, mask_int, mask_ref, _ = load_interpolation_input_data(args.input_file,
                                                                              args.elevation_var,
                                                                              x_var=args.x_var,
                                                                              y_var=args.y_var,
                                                                              x_dim=(args.x_dim if args.x_dim else args.x_var),
                                                                              y_dim=(args.y_dim if args.y_dim else args.y_var),
                                                                              interpolation_flag_var=args.interpolation_flag_var)

    xs_ref = xs_mat[mask_ref]
    ys_ref = ys_mat[mask_ref]
    elevation_ref = elevation[mask_ref]
    np.savetxt(args.output_file, np.column_stack((xs_ref, ys_ref, elevation_ref)), delimiter=" ")

    if args.output_missing:
        xs_int = xs_mat[mask_int]
        ys_int = ys_mat[mask_int]
        np.savetxt(args.output_missing, np.column_stack((xs_int, ys_int)), delimiter=" ")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
