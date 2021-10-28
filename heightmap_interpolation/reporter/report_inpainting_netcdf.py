import argparse
from heightmap_interpolation.reporter.inpaint_netcdf_reporter import InpaintingReporter
import json


def report(param):
    # Load config
    f = open(param.config_file)
    config = json.load(f)
    f.close()

    # Create the reporter
    reporter = InpaintingReporter(config)

    # Report!
    reporter.report()

    # # Render report to PDF
    # reporter.render_report_to_pdf(param.output_file)


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Creates a report applying different inpainting methods to different datasets, according to the options set in the configuration file")
    parser.add_argument("config_file", action="store", type=str,
                        help="Reporter configuration file in JSON format")

    param = parser.parse_args(args)
    return param

# Main function
if __name__ == "__main__":
    report(parse_args())
