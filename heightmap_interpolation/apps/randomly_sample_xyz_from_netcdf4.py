import argparse
import numpy as np
import netCDF4 as nc


def create_sample(params):
    # Read the elevation data
    ds = nc.Dataset(params.input_file, "r", format="NETCDF4")
    elevation = ds[params.elevation_var]
    lats_1d = ds.variables["lat"][:]
    lons_1d = ds.variables["lon"][:]

    # Take random samples
    xs = np.choice(elevation.size[0], params.num_samples)
    ys = np.choice(elevation.size[1], params.num_samples)

    samples = np.zeros((len(xs), 3))
    for i, x, y in zip(range(params.num_samples), xs, ys):
        samples[i, :] = [lons_1d[x], lats_1d[y], elevation[x, y]]
    
    # Write to file
    np.savetxt(params.output_file, samples)


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(description="Creates a random XYZ sample from a raster file")
    parser.add_argument("input_file", action="store", type=str,
                        help="Input NetCDF file")
    parser.add_argument("-o","--output_file", dest="output_file", action="store", type=str,
                        help="Output NetCDF file with interpolated values")
    parser.add_argument("-n","--num_samples", dest="num_samples", action="store", type=str,
                        help="Number of random samples to draw from the input dataset")
    parser.add_argument("--elevation_var", action="store", type=str, default="elevation",
                        help="Name of the variable storing the elevation grid in the input file.")
    return parser.parse_args(args)


def main():
    create_sample(parse_args())


# Main function
if __name__ == "__main__":
    main()