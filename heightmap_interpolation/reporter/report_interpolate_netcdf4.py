#!/usr/bin/env python3

import argparse
from heightmap_interpolation.reporter.interpolate_netcdf4_reporter import InterpolateNetCDF4Reporter, create_default_config
import json


def report(param):

    if param.generate_default_config:
        # Generate a default configuration
        config = create_default_config()
        with open(param.config_file, 'w') as fp:
            json.dump(config, fp, indent=4)
        return

    # Load config
    f = open(param.config_file)
    config = json.load(f)
    f.close()

    # Create the reporter
    reporter = InterpolateNetCDF4Reporter(config)

    # Report!
    reporter.report()


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Creates a report applying different interpolation methods to different datasets, according to the options set in the configuration file")
    parser.add_argument("config_file", action="store", type=str,
                        help="Reporter configuration file in JSON format")
    parser.add_argument("--generate_default_config", action="store_true", help="If using this flag, the 'config_file' parameter should point to an OUTPUT file, where the default config of the reporter will be written (easy to then modify and extend at user's convenience)")

    param = parser.parse_args(args)
    return param


# Main function
if __name__ == "__main__":
    report(parse_args())
