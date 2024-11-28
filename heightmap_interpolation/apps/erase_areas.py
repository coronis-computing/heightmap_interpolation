import sys
import argparse
import cv2 
import numpy as np
import netCDF4 as nc
import shutil


def simple_image_masker(img, brush_color=(0, 0, 255)):
    img_cp = img.copy()

    cv2.namedWindow('Draw a mask', cv2.WINDOW_GUI_NORMAL)

    brush_size = 5
    def change_brush_size(val):
        nonlocal brush_size
        brush_size = val

    cv2.createTrackbar('Brush size', 'Draw a mask', 1, 1000, change_brush_size)
    
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    drawing = False
    def click_event(event, x, y, flags, param):
        nonlocal mask, img_cp, drawing
        if event == cv2.EVENT_LBUTTONDOWN:            
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(img_cp, (x, y), brush_size, brush_color, -1)
            cv2.circle(mask, (x, y), brush_size, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    cv2.setMouseCallback('Draw a mask', click_event)

    print("Draw a mask over the image. Usage:")
    print("  - Adapt brush size with the trackbar.")
    print("  - Press 'c' to clear the current mask.")
    print("  - Press 'a', 'ESC' or 'Return' to finish.")
    while True:
        cv2.imshow('Draw a mask', img_cp)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k == 13 or k == ord('a'):
            break
        if k == ord("c"):
            img_cp = img.copy()
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)        

    return mask


def write_mask(input_file, output_file, new_interpolation_flag, elevation_var, interpolation_flag_var=None):
    # We just want to modify the elevation variable, while retaining the rest of the dataset as is, so the easiest
    # solution is to copy the input file to the destination file, and open it in write mode to change the elevation
    # variable
    if output_file:
        shutil.copy(input_file, output_file)
    else:
        raise ValueError("Missing output file path!")

    # Modify the elevation variable
    out_ds = nc.Dataset(output_file, "r+")
    elevation = out_ds.variables[elevation_var][:]
    elevation.mask = np.ma.mask_or(np.ma.make_mask(new_interpolation_flag), elevation.mask)
    out_ds.variables[elevation_var][:] = elevation

    # Modify/create the interpolation_flag variable
    if interpolation_flag_var and interpolation_flag_var not in out_ds.variables.keys():
        raise RuntimeError(f"The input NetCDF4 dataset does not contain a variable named {interpolation_flag_var}.")
    # Also update the interpolated areas
    if "interpolation_flag" not in out_ds.variables.keys():
        out_ds.createVariable(
            "interpolation_flag",
            "int8",
            ("lat", "lon"))
        new_cell_interpolated_flag = out_ds.variables["interpolation_flag"][:]        
    else:
        new_cell_interpolated_flag = out_ds.variables["interpolation_flag"][:]
    new_cell_interpolated_flag[new_interpolation_flag > 0] = 1
    new_cell_interpolated_flag[new_interpolation_flag <= 0] = 0

    out_ds.variables["interpolation_flag"][:] = new_cell_interpolated_flag

    out_ds.close()


def main():
    """Simple GUI for removing parts of a map, in order to test the interpolation algorithms on known data"""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="The input elevation map to inpaint (in NetCDF4 format)")
    parser.add_argument("-o","--output_file", dest="output_file", action="store", type=str, required=True,
                        help="Output NetCDF file with erased areas")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    parser.add_argument("--interpolation_flag_var", action="store", type=str, default=None,
                        help="Name of the variable storing the per-cell interpolation flag in the input file (0 == known value, 1 == interpolated/to interpolate cell). If it exist on the dataset, the result will be the LOGICAL OR between the original and the selected mask. If not in the dataset, it will create one in the output netcdf file with the \"interpolation_flag\" id containing the selected mask.")
    args = parser.parse_args()

    # Read the file
    ds = nc.Dataset(args.input_file, "r", format="NETCDF4")
    
    # Get the elevation data
    elevation = ds.variables[args.elevation_var][:]

    if args.interpolation_flag_var:
        interpolation_flag = ds.variables[args.interpolation_flag_var][:].astype(np.uint8) 
    else:
        interpolation_flag = np.zeros_like(elevation, np.uint8)
    
    # display = cv2.convertScaleAbs(elevation)
    display = cv2.normalize(elevation, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    mask = simple_image_masker(display)

    new_interpolation_flag = cv2.bitwise_or(mask, interpolation_flag)

    write_mask(args.input_file, args.output_file, new_interpolation_flag, args.elevation_var, args.interpolation_flag_var)
    



if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
