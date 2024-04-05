import argparse
import netCDF4 as nc

def main():
    # Parameters
    parser = argparse.ArgumentParser(description="Prints a summary of the information in a netcdf file.")
    parser.add_argument("input_file", action="store", type=str, help="Input NetCDF file")
    params = parser.parse_args()

    ds = nc.Dataset(params.input_file, "r")
    print(ds)

# Main function
if __name__ == "__main__":
    main()