import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage.filters import laplace
# from heightmap_interpolation.inpainting.differential import divergence
from heightmap_interpolation.apps.common import load_data
import heightmap_interpolation.inpainting.differential as diff

# ---
# Test to check if the scipy's "laplace" function is equal to our implementation of "divergence" of "gradient"
# ---

def array2cmpa(X):
    # Assuming array is Nx3, where x3 gives RGB values
    # Append 1's for the alpha channel, to make X Nx4
    X = np.c_[X,np.ones(len(X))]

    return matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', X)


def my_fun(param):
    # Load the data
    print("- Loading input data")
    elevation, mask_int, lats_mat, lons_mat, mask_ref = load_data(param)

    # Initialize
    elevation[mask_int] = 0  # Just in case the values not filled in the image are NaNs!

    # Compute the Laplacian
    option1 = laplace(elevation)
    option2 = diff.divergence(diff.gradient(elevation))

    print(option1-option2)
    # option2 = divergence([np.diff(elevation, axis=0), np.diff(elevation, axis=1)])
    #option2 = np.diff(elevation, axis=0, prepend=0) + np.diff(elevation, axis=1, prepend=0)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
    images = [option1, option2]
    titles = ['Laplace', 'Divergence(gradients)']
    for (ax, image, title) in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    # plt.colorbar()
    fig.tight_layout()
    plt.show()


# Main function

def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Interpolate terrain data in a SeaDataNet_1.0 CF1.6-compliant netCDF4 file via inpainting")
    parser.add_argument("input_file", action="store", type=str,
                        help="Input NetCDF file")
    parser.add_argument("--areas", action="store", type=str, default="",
                        help="KML file containing the areas that will be interpolated.")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    parser.add_argument("--interpolation_flag_var", action="store", type=str, default="interpolation_flag",
                        help="Name of the variable storing the per-cell interpolation flag in the input file")
    parser.add_argument("--method", action="store", type=str, default="sobolev",
                        help="Name of the inpainting method to use. Available: sobolev, ccst")
    parser.add_argument("-v, --verbose", action="store_true", dest="verbose", default=False,
                        help="Verbosity flag, activate it to have feedback of the current steps of the process in the command line")
    parser.add_argument("--interpolate_missing_values", action="store_true", default=False,
                        help="Missing value flag, activate it to interpolate missing values instead of re interpolate previously interpolated values")
    parser.add_argument("-s, --show", action="store_true", dest="show", default=False,
                        help="Show interpolation problem and results on screen")
    param = parser.parse_args(args)

    return param

if __name__ == "__main__":
    my_fun(parse_args())
