import argparse
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


def create_sample(params):
    # Read the elevation data
    ds = nc.Dataset(params.input_file, "r", format="NETCDF4")
    elevation = ds[params.elevation_var]
    lats_1d = ds.variables["lat"][:]
    lons_1d = ds.variables["lon"][:]

    valid_pts = np.argwhere(~np.isnan(elevation))

    # Take random samples
    valid_pts_inds = np.random.choice(valid_pts.shape[0], params.num_samples)
    
    samples = np.zeros((len(valid_pts_inds), 3))    
    for i, ind in zip(range(params.num_samples), valid_pts_inds):
        x = valid_pts[ind, 1]
        y = valid_pts[ind, 0]
        samples[i, :] = [lons_1d[x], lats_1d[y], elevation[y, x]]        

    # Show sample
    if params.show:
        lons_selected = np.zeros(params.num_samples)
        lats_selected = np.zeros(params.num_samples)
        for i, ind in zip(range(params.num_samples), valid_pts_inds):
            x = valid_pts[ind, 1]
            y = valid_pts[ind, 0]
            lons_selected[i] = lons_1d[x]
            lats_selected[i] = lats_1d[y]
        print("Showing the samples taken, close the window to continue...")
        plt.scatter(lons_selected.T, lats_selected.T)
        plt.show(block=True)        

    # Write to file
    np.savetxt(params.output_file, samples)


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(description="Creates a random XYZ sample from a raster file")
    parser.add_argument("input_file", action="store", type=str,
                        help="Input NetCDF file")
    parser.add_argument("-o","--output_file", dest="output_file", action="store", type=str,
                        help="Output NetCDF file with interpolated values")
    parser.add_argument("-n","--num_samples", dest="num_samples", action="store", type=int,
                        help="Number of random samples to draw from the input dataset")
    parser.add_argument("-s", "--show", action="store_true", dest="show", default=False,
                        help="Show a 2D plot of the sample points")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    return parser.parse_args(args)


def main():
    create_sample(parse_args())


# Main function
if __name__ == "__main__":
    main()