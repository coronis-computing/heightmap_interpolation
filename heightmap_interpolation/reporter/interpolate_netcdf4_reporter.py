#!/usr/bin/env python3

import os
import sys
import copy
import shutil
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import json
from heightmap_interpolation.apps.netcdf_data_io import load_interpolation_input_data
from heightmap_interpolation.apps.interpolate_netcdf4 import parse_args
from timeit import default_timer as timer
import time
import psutil
import multiprocessing as mp

def create_default_config():
    # Create the default configurations for each method
    methods = ['nearest', 'linear', 'cubic', 'rbf', 'purbf', 'harmonic', 'tv', 'ccst', 'amle', 'navier-stokes', 'telea', 'shiftmap']
    tests = []
    for method in methods:
        # Get defaults of this method
        defaults = vars(parse_args([method, "dummy_file"]))

        # Remove from these parameters those corresponding to the input
        keys_to_delete = ["input_file", "output_file", "areas", "elevation_var", "interpolation_flag_var", "verbose", "show", "subparser_name"]
        for key in keys_to_delete:
            del defaults[key]

        # Create a test structure with all the parameters
        test = {
            "name": "",
            "method": method,
            "parameters": defaults
        }
        tests.append(test)

    config = {
        # Configuration of the reporter
        "work_dir": "./report", # The working directory, here results as well as the report doc sources will be stored
        "title": "Execution report", # The title of the report
        "intro_text": "", # Use this option to introduce further custom text in the intro section
        "re-write": False, # If set, all the text sections will be re-written
        "re-execute": False,  # If set, all the tests will be re-executed and tests sections will be re-written
        "output_file": "./report.pdf", # The final report file
        "output_tex_file": "./report.tex",  # The final report file in LaTex format (in case you want to manually edit it)
        "datasets": [ # A list of datasets. A report can contain different datasets
            {
                "name": "Example dataset", # A name/ID/alias for the dataset
                "netcdf_file": "", # The actual NetCDF file containing the data
                "elevation_var": "elevation", # The name of the variable inside the netcdf file to be considered as the elevation/bathymetry
                "interpolation_flag_var": "",
                "areas": "",
                "tests": tests
            }
        ]
    }

    return config


class InterpolateNetCDF4Reporter():
    # Executes a series of tests described in the config file, and creates a document in PDF with the results
    # Each section is written in a separate markdown file, and joined at the end in a single file

    def __init__(self, config):
        # Store the configuration
        self.config = config
        # Different paths used while reporting
        self.results_dir = os.path.join(self.config["work_dir"], "results")
        self.docs_dir = os.path.join(self.config["work_dir"], "docs")
        self.report_file_md = os.path.join(self.config["work_dir"], "report.md")
        # Data/info of the current dataset
        self.ds_config = None
        self.ds_counter = 0
        self.ds_elevation = []
        self.ds_interpolate_mask = []
        self.ds_elevation_var = None
        self.ds_interpolation_flag_var = None
        self.ds_min_elevation = 0
        self.ds_max_elevation = 0
        self.ds_areas = ""
        self.ds_work_areas = None
        # Data/info of the current test
        self.test_counter = 0
        self.test_subdir = ""
        self.test_base_name = ""
        self.run_duration = 0
        self.interpolation_results_netcdf = ""
        # Current document data
        self.section_dir = ""
        self.doc_path = ""
        # Attributes needed for accounting resources used during execution
        self.res_num_queries = 0
        self.res_once_cpu_mean = True
        self.res_cpu_percent_mean = 0
        self.res_cpu_percent_max = 0
        self.res_memory_mean = 0
        self.res_once_memory_mean = 0
        self.res_memory_max = 0
        self.res_cpu_time = 0
        self.update_period = 0.1 # Account every self.update_period seconds

    def init_folders(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)

    def call_external_and_account_resources(self, args, log_to):
        # Init resources' stats
        self.res_num_queries = 0
        self.res_once_cpu_mean = True
        self.res_cpu_percent_mean = 0
        self.res_cpu_percent_max = 0
        self.res_memory_mean = 0
        self.res_once_memory_mean = 0
        self.res_memory_max = 0

        # Call psutil on the created thread
        p = psutil.Popen(args, stdout=log_to, stderr=log_to)

        # Monitor each period
        while True:
            time.sleep(self.update_period)
            self.update_resources_stats(p)

            # Check if the process has finished
            ret_code = p.poll()
            if ret_code is not None:
                break

        return ret_code

    def update_resources_stats(self, process):
        """Update the statistics accounted for a given process"""

        # Compute some CPU/Memory stats
        self.res_num_queries += 1
        cpu_percent = process.cpu_percent()  # Make sure to do this before the call to poll()
        if self.res_once_cpu_mean:
            if cpu_percent > sys.float_info.epsilon:
                self.res_cpu_percent_mean = cpu_percent
                self.res_once_cpu_mean = False
        else:
            if cpu_percent > sys.float_info.epsilon:
                self.res_cpu_percent_mean = self.res_cpu_percent_mean + (cpu_percent - self.res_cpu_percent_mean) / self.res_num_queries

        if self.res_cpu_percent_max < cpu_percent:
            self.res_cpu_percent_max = cpu_percent

        cpu_times = process.cpu_times()
        self.res_cpu_time = cpu_times.user + cpu_times.system

        memory = process.memory_info()
        if self.res_once_memory_mean:
            if memory.rss > sys.float_info.epsilon:
                self.res_memory_mean = float(memory.rss)
                self.res_once_memory_mean = False
        else:
            if memory.rss > sys.float_info.epsilon:
                self.res_memory_mean = self.res_memory_mean + (float(memory.rss) - self.res_memory_mean) / self.res_num_queries

        if self.res_memory_max < memory.rss:
            self.res_memory_max = float(memory.rss)

    # def prepare_section(self, section_name):
    #     section_dir = os.path.join(self.docs_dir, section_name)
    #     doc_path = os.path.join(section_dir, section_name+".md")
    #     if os.path.exists(doc_path) and not self.config["re-write"]:
    #         return None, None
    #     os.makedirs(section_dir, exist_ok=True)
    #
    #     return section_dir, doc_path

    def write_intro_section(self):
        section_dir = os.path.join(self.docs_dir, "intro")
        doc_path = os.path.join(section_dir, "intro.md")
        if os.path.exists(doc_path) and not self.config["re-write"]:
            return
        os.makedirs(section_dir, exist_ok=True)
        # section_dir, doc_path = self.prepare_section("intro")
        # if not section_dir:
        #     return

        # Open the file
        f = open(doc_path, 'w')

        # Writes the main title
        f.write("# " + self.config["title"] + "\n\n")

        # Writes the intro section
        if not self.config["intro_text"]:
            f.write("This report presents the execution results using interpolation methods of the `heightmap_interpolation` toolbox.\n\n")
        else:
            f.write(self.config["intro_text"]+"\n\n")

        # Close the file
        f.close()

    def write_machine_resources_section(self):
        section_dir = os.path.join(self.docs_dir, "resources")
        doc_path = os.path.join(section_dir, "resources.md")
        if os.path.exists(doc_path) and not self.config["re-write"]:
            return
        os.makedirs(section_dir, exist_ok=True)
        # section_dir, doc_path = self.prepare_section("intro")
        # if not section_dir:
        #     return

        # Open the file
        f = open(doc_path, 'w')

        # Writes the main title
        f.write("# Resources\n\n")

        # Writes the intro section
        f.write("All the tests in this report were executed on a computer with the following resources:\n\n")

        f.write("| **CPU cores** | **RAM memory (Gb)**\n" +
                "| :---: | :---: |\n" +
                "| {:d} | {:.2f} |\n\n".format(mp.cpu_count(), psutil.virtual_memory().total/(1024**3)))

        # Close the file
        f.close()

    def load_dataset(self, ds_config):
        # Read the file
        lats_mat, lons_mat, elevation, mask_int, mask_ref, work_areas = load_interpolation_input_data(ds_config["netcdf_file"],
                                                                                                      ds_config["elevation_var"],
                                                                                                      ds_config["interpolation_flag_var"],
                                                                                                      ds_config["areas"])

        # Store the elevation and interpolation mask (will be used later in all the tests to be performed on this dataset)
        self.ds_interpolate_mask = mask_int
        self.ds_reference_mask = mask_ref
        self.ds_elevation = elevation
        self.ds_areas = ds_config["areas"]
        self.ds_work_areas = work_areas

        # Some other vars/statistics used by other sections
        self.ds_min_elevation = np.amin(self.ds_elevation[self.ds_reference_mask])
        self.ds_max_elevation = np.amax(self.ds_elevation[self.ds_reference_mask])
        self.ds_elevation_var = ds_config["elevation_var"]
        self.ds_interpolation_flag_var = ds_config["interpolation_flag_var"]
        self.ds_netcdf_file = ds_config["netcdf_file"]

    def set_dataset_paths(self, ds_config):
        self.section_dir = os.path.join(self.docs_dir, "dataset_{:d}".format(self.ds_counter), "data_analysis")
        self.doc_path = os.path.join(self.section_dir, "intro.md")
        os.makedirs(self.section_dir, exist_ok=True)
        self.ds_config = ds_config # TODO: do we need this? When using it in other functions, we should decide whether we use it as a function parameter or we rely on the attributes of the class...

    def set_tests_paths(self, tst_config):
        self.test_subdir = os.path.join("dataset_{:d}".format(self.ds_counter), "test_{:d}".format(self.test_counter))
        self.section_dir = os.path.join(self.docs_dir, self.test_subdir)
        self.current_results_dir = os.path.join(self.results_dir, self.test_subdir)
        self.doc_path = os.path.join(self.section_dir, "test_{:d}.md".format(self.test_counter))
        os.makedirs(self.section_dir, exist_ok=True)
        os.makedirs(self.current_results_dir, exist_ok=True)
        if tst_config["name"]:
            self.test_base_name = "{:s}_test_{:d}".format(tst_config["name"], self.test_counter)
        else:
            self.test_base_name = "test_{:d}".format(self.test_counter)

    def write_data_analysis_section(self, ds_config):
        # Set the current config
        self.ds_config = ds_config

        # Check if this section was already written
        self.section_dir = os.path.join(self.docs_dir, "dataset_{:d}".format(self.ds_counter), "data_analysis")
        self.doc_path = os.path.join(self.section_dir, "intro.md")
        if self.data_analysis_section_was_already_written(ds_config) and not self.config["re-write"]:
            print("Dataset analysis section was already written, skipping...")
            return
        # Create dirs if needed
        os.makedirs(self.section_dir, exist_ok=True)

        f = open(self.doc_path, 'w')
        f.write("# Dataset {:d}".format(self.ds_counter))
        if ds_config["name"]:
            f.write(": {:s}\n\n".format(ds_config["name"]))
        else:
            f.write("\n\n")

        # Show the image
        fig, axes = plt.subplots(nrows=1, ncols=2)
        im = axes[0].imshow(self.ds_elevation)
        # divider = make_axes_locatable(axes[0])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(im, cax=cax, orientation="vertical")
        axes[0].set_title("Elevation")
        axes[0].set_axis_off()
        divider = make_axes_locatable(axes[1])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        if not self.ds_areas:
            if not np.all(np.logical_or(self.ds_reference_mask, self.ds_interpolate_mask)):
                rgb_interpolate_mask = cv2.cvtColor(np.asarray(self.ds_interpolate_mask, dtype="uint8") * 255,
                                                    cv2.COLOR_GRAY2RGB)
                ref_or_interpolate_mask = np.logical_or(self.ds_reference_mask, self.ds_interpolate_mask)
                rgb_interpolate_mask[self.ds_reference_mask] = [0, 255, 0]
                rgb_interpolate_mask[self.ds_interpolate_mask] = [255, 0, 0]
                rgb_interpolate_mask[~ref_or_interpolate_mask] = [0, 0, 255]
                axes[1].imshow(rgb_interpolate_mask)
                axes[1].set_title("- Reference data (green)\n- Unknown data to interpolate (red)\n- Unknown data to obviate (blue)")
                axes[1].set_axis_off()
                plt.tight_layout()
            else:
                axes[1].imshow(~self.ds_interpolate_mask, cmap=plt.cm.gray)
                axes[1].set_title("Known (white) / Unknown (black) data")
                axes[1].set_axis_off()
                plt.tight_layout()
        else:
            # Convert to color image the "data to interpolate" mask
            rgb_interpolate_mask = cv2.cvtColor(np.asarray(self.ds_interpolate_mask, dtype="uint8")*255, cv2.COLOR_GRAY2RGB)
            for i in range(self.ds_work_areas.shape[2]):
                # Extract the contour of the mask
                work_area_mask_cv = np.asarray(self.ds_work_areas[:, :, i], dtype="uint8")*255
                contours, hierarchy = cv2.findContours(work_area_mask_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb_interpolate_mask, contours, 0, (0, 0, 255), 2)

            axes[1].imshow(rgb_interpolate_mask)
            divider = make_axes_locatable(axes[1])
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            axes[1].set_title("- Reference data (white)\n- Unknown data (black)\n- Areas to interpolate (blue contours)")
            axes[1].set_axis_off()
            # Also plot the regions where we are going to interpolate
            plt.tight_layout()

        # Save it
        images_dir = os.path.join(self.section_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        input_data_image_path = os.path.join(images_dir, "input_data.png")
        plt.savefig(input_data_image_path, bbox_inches="tight")

        # Render it on the doc (as a Markdown link)
        tex_figure_label = "fig:dataset_{:d}".format(self.ds_counter)
        fig_caption = "Input dataset {:d}".format(self.ds_counter)
        if ds_config["name"]:
            fig_caption += " - {:s}".format(ds_config["name"])

        f.write("The input dataset {:s} can be seen in figure \\ref{{{:s}}}.\n\n".format(ds_config["name"],
                                                                                         tex_figure_label))
        f.write("![{:s}\label{{{:s}}}]({:s})\n\n".format(fig_caption, tex_figure_label, input_data_image_path))

        f.write("It has the following properties:\n\n")

        storage_size = os.path.getsize(input_data_image_path) / (1024*1024)
        total_pixels = self.ds_elevation.shape[0] * self.ds_elevation.shape[1]
        num_pixels_to_interpolate = np.count_nonzero(~self.ds_interpolate_mask)
        interpolate_percent = (num_pixels_to_interpolate / total_pixels) * 100

        f.write("| **Width** | **Height** | **Size (MBytes)** | **Min. elevation** | **Max. elevation** |\n"+
                "| :---: | :---: | :---: | :---: | :---: | \n" +
                "| {:d} | {:d} | {:.2f} | {:.2f} | {:.2f} |\n\n".format(self.ds_elevation.shape[1], self.ds_elevation.shape[0], storage_size, self.ds_min_elevation, self.ds_max_elevation))

        f.write("And it presents the following interpolation problem:\n\n")

        f.write("| **Num. cells** | **Unknown cells** | **Known cells** | **\% to interpolate** |\n"+
                "| :---: | :---: | :---: | :---: |\n" +
                "| {:d} | {:d} | {:d} | {:.2f} |\n\n".format(total_pixels, total_pixels-num_pixels_to_interpolate, num_pixels_to_interpolate, interpolate_percent))

        f.close()

        # Write the dataset config in the section folder as an indicator that the section was already written
        config = os.path.join(self.section_dir, "config.json")

        ds_config_copy = copy.deepcopy(ds_config)
        ds_config_copy.pop("tests")
        with open(config, 'w') as fp:
            json.dump(ds_config_copy, fp, indent=4)

    def data_analysis_section_was_already_written(self, ds_config):
        """Checks if the data analysis section seems to be written already by a previous execution"""
        config_file = os.path.join(self.section_dir, "config.json")
        if not os.path.exists(config_file):
            return False

        # If the configuration file exists, check that the dataset is the same as in the config file
        f = open(config_file)
        saved_config = json.load(f)

        ds_config_copy = copy.deepcopy(ds_config)
        ds_config_copy.pop("tests")
        return saved_config == ds_config_copy

    def run_test(self, tst_config):

        # Call the interpolate_netcdf4.py script (assumed to be on the PYTHONPATH!)
        args = ["interpolate_netcdf4.py"]

        # The dataset-related config
        if self.ds_elevation_var:
            args.append("--elevation_var")
            args.append(self.ds_elevation_var)
        if self.ds_interpolation_flag_var:
            args.append("--interpolation_flag_var")
            args.append(self.ds_interpolation_flag_var)
        if self.ds_areas:
            args.append("--areas")
            args.append(self.ds_areas)

        # Always in verbose mode
        args.append("-v")

        # Where to save the results
        self.interpolation_results_netcdf = os.path.join(self.current_results_dir, self.test_base_name + ".nc")
        args.append("--output_file")
        args.append(self.interpolation_results_netcdf)

        # The method
        args.append(tst_config["method"])

        # Convert the key-value pairs in the "parameters" value from the config dict to a set of arguments
        flags = ["rescale", "convolve_in_1d"]
        for key, value in tst_config["parameters"].items():
            if key in flags:
                if value:
                    args.append("--" + key)
            else:
                if value: # Avoids writing empty parameters
                    args.append("--" + key)
                    args.append(str(value))

        # The input file
        args.append(self.ds_netcdf_file)

        # Create a cmd file with the actual command executed
        cmd_file_path = os.path.join(self.current_results_dir, self.test_base_name + ".cmd")
        cmd_file = open(cmd_file_path, 'w')
        cmd_file.write("* Command:\n\n```\n" + ' '.join(args) + "\n```\n\n")
        cmd_file.close()

        # Create a log file with the output
        log_file_path = os.path.join(self.current_results_dir, self.test_base_name + ".log")
        log_file = open(log_file_path, 'w')

        # Call the script and account the resources required for executing it
        print(args)
        ts = timer()
        ret_code = self.call_external_and_account_resources(args, log_file)
        te = timer()
        run_duration = te-ts
        # Close the log file
        log_file.close()

        # Check if execution went well...
        if ret_code != 0:
            raise ValueError("The return code is not 0! Aborting...")

        # Save some statistics about the execution, that will be used for writing the section
        stats_dict = {"run_duration": run_duration,
                      "res_cpu_percent_mean": self.res_cpu_percent_mean,
                      "res_cpu_percent_max": self.res_cpu_percent_max,
                      "res_cpu_time": self.res_cpu_time,
                      "res_memory_mean": self.res_memory_mean,
                      "res_memory_max": self.res_memory_max}
        test_stats = os.path.join(self.current_results_dir, self.test_base_name + "_stats.json")
        with open(test_stats, 'w') as fp:
            json.dump(stats_dict, fp, indent=4)

        # Save the configuration also, so that we have a reference, and a mark that the test was executed
        test_config = os.path.join(self.current_results_dir, self.test_base_name + "_config.json")
        with open(test_config, 'w') as fp:
            json.dump(tst_config, fp, indent=4)

    def test_was_already_executed(self, tst_config):
        if self.config["re-execute"]:
            return False

        # Compare the current config with the one stored from a previous execution
        # Save the configuration also, so that we have a reference, and a mark that the test was executed
        prev_config_file = os.path.join(self.current_results_dir, self.test_base_name + "_config.json")
        if not os.path.exists(prev_config_file):
            return False
        f = open(prev_config_file)
        prev_config = json.load(f)

        # Check that both configurations are equal
        if prev_config["name"] != tst_config["name"]:
            print("The 'name' variable changed!")
            return False
        if prev_config["method"] != tst_config["method"]:
            print("The 'method' changed!")
            return False
        if prev_config["parameters"] != tst_config["parameters"]:
            print("The configuration of the interpolator changed!")
            return False

        return True

    def write_test_section(self, tst_config):
        # test_subdir = os.path.join("dataset_{:d}".format(self.ds_counter), "test_{:d}".format(self.test_counter))
        # section_dir = os.path.join(self.docs_dir, test_subdir)
        # doc_path = os.path.join(section_dir, "test_{:d}.md".format(self.test_counter))
        # if os.path.exists(doc_path) and not self.config["re-write"]:
        #     return
        # os.makedirs(section_dir, exist_ok=True)
        #
        # if tst_config["name"]:
        #     test_base_name = "{:s}_test_{:d}".format(tst_config["name"], self.test_counter)
        # else:
        #     test_base_name = "test_{:d}".format(self.test_counter)

        # Write the section
        f = open(self.doc_path, 'w')
        f.write("## Test {:d}".format(self.test_counter))
        if tst_config["name"]:
            f.write(": {:s}\n\n".format(tst_config["name"]))
        else:
            f.write("\n\n")

        if tst_config["parameters"]:
            f.write("Executed the *{:s}* interpolation method with the following parameters (defaults not included):\n\n".format(tst_config["method"]))

            for key, value in tst_config["parameters"].items():
                f.write("* `--{:s} {:s}`\n".format(key, str(value)))
        else:
            f.write("Executed the *{:s}* interpolation method with the default parameters.\n\n".format(tst_config["method"]))

        f.write("\n")

        f.write("The actual call to `interpolate_netcdf4.py` was:\n\n")
        cmd_file_path = os.path.join(self.current_results_dir, self.test_base_name + ".cmd")
        cmd_file = open(cmd_file_path, 'r')
        for line in cmd_file:
            f.write(line)
        cmd_file.close()
        f.write("\n\n")

        # Load the interpolated elevation
        ds = nc.Dataset(self.interpolation_results_netcdf, "r", format="NETCDF4")
        interpolated_elevation = ds.variables[self.ds_elevation_var][:]

        # Show the result as an image
        fig, ax = plt.subplots()
        ax.imshow(interpolated_elevation)
        ax.set_axis_off()
        plt.tight_layout()

        # Save it
        images_dir = os.path.join(self.section_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        results_image_path = os.path.join(images_dir, self.test_base_name + "_result.png")
        plt.savefig(results_image_path, bbox_inches="tight")

        # Render it on the doc (as a Markdown link)
        tex_figure_label = "fig:interpolation_test_{:d}".format(self.test_counter)
        fig_caption = "Interpolation results of test {:d}".format(self.test_counter)
        if tst_config["name"]:
            fig_caption += " - {:s}".format(tst_config["name"])
        f.write("The results of the interpolation can be seen in figure \\ref{{{:s}}}.\n\n".format(tex_figure_label))
        f.write("![{:s}\label{{{:s}}}]({:s})\n\n".format(fig_caption, tex_figure_label, results_image_path))

        # Check that the interpolated part did not overshoot the input data
        interpolated_elevation_min = np.amin(interpolated_elevation)
        interpolated_elevation_max = np.amax(interpolated_elevation)
        overshooted_min = interpolated_elevation_min < self.ds_min_elevation
        overshooted_max = interpolated_elevation_max > self.ds_max_elevation
        if overshooted_min or overshooted_max:
            f.write("**WARNING**: the interpolated areas present some **overshooting** with respect to the original known input values:\n\n")
        if overshooted_min:
            f.write("* Original minimum elevation was {:.2f}, but in the interpolated result is {:.2f}\n".format(self.ds_min_elevation, interpolated_elevation_min))
        if overshooted_max:
            f.write("* Original maximum elevation was {:.2f}, but in the interpolated result is {:.2f}\n".format(self.ds_max_elevation, interpolated_elevation_max))

        f.write("\n\n")

        # Add statistics regarding the resources used
        # Load the test's statistics from file
        test_stats = os.path.join(self.current_results_dir, self.test_base_name + "_stats.json")
        stats_file = open(test_stats)
        stats = json.load(stats_file)
        stats_file.close()

        duration_unit = "seconds"
        duration = stats["run_duration"]
        if stats["run_duration"] > 60:
            duration = duration / 60
            duration_unit = "minutes"
            if duration > 60:
                duration = duration / 60
                duration_unit = "hours"
        f.write("The execution took a total of **{:.2f} {}**. This timing includes the loading of the data, the interpolation itself and the writing of results to disk. Also, it required the following resources:\n\n".format(duration, duration_unit))
        f.write(
            "| **Mean CPU %** | **Max. CPU %** | **Total CPU time (s)** | **Mean Memory (MB)** | **Max Memory (MB)** |\n"
            "| :---: | :---: | :---: | :---: | :---: |\n"
            "| {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |\n\n".format(stats["res_cpu_percent_mean"],
                                                                        stats["res_cpu_percent_max"],
                                                                        stats["res_cpu_time"],
                                                                        stats["res_memory_mean"] / (1024*1024),
                                                                        stats["res_memory_max"] / (1024*1024)))
        f.write(
            "Note that the CPU statistics above are computed accross all used CPUs, so multithreaded "
            "calls may potentially exceed the 100% CPU usage and cause the total CPU time to be larger "
            "than the actual execution time.\n\n")

        # Close the file
        f.close()

    def write_tests_comparison_section(self, ds_config):
        self.section_dir = os.path.join(self.docs_dir, "dataset_{:d}".format(self.ds_counter), "tests_comparison")
        self.doc_path = os.path.join(self.section_dir, "comparison.md")
        os.makedirs(self.section_dir, exist_ok=True)

        f = open(self.doc_path, 'w')
        f.write("## Comparison between tests performed to Dataset {:d}".format(self.ds_counter))
        if ds_config["name"]:
            f.write(": {:s}\n\n".format(ds_config["name"]))
        else:
            f.write("\n\n")

        f.write("The following table presents at a glance some of the statistics computed for the different methods/tests applied to the present dataset:\n\n")

        # Table header
        f.write("| **Test Num.** | **Method** | **Run time (s) ** | **Mean CPU %** | **Max. CPU %** | **Total CPU time (s)** | **Mean Memory (MB)** | **Max Memory (MB)** |\n" +
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        self.test_counter = 0
        for tst_config in ds_config["tests"]:
            # Get the statistics for this test
            self.set_tests_paths(tst_config)
            test_stats = os.path.join(self.current_results_dir, self.test_base_name + "_stats.json")
            stats_file = open(test_stats)
            stats = json.load(stats_file)
            stats_file.close()

            # Fill Table row
            f.write("| {:d} | {:s} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |\n".format(self.test_counter,
                                                                                        tst_config["method"],
                                                                                        stats["run_duration"],
                                                                                        stats["res_cpu_percent_mean"],
                                                                                        stats["res_cpu_percent_max"],
                                                                                        stats["res_cpu_time"],
                                                                                        stats["res_memory_mean"] / (1024 * 1024),
                                                                                        stats["res_memory_max"] / (1024 * 1024)))
            self.test_counter = self.test_counter+1
        f.write("\n\n")
        f.close()

    def report(self):
        """Main entry point, will execute the desired tests and render the docs"""

        # Create the main dirs
        self.init_folders()

        # Intro
        print("- Writing generic intro section")
        self.write_intro_section()
        
        # Resources
        print("- Writing machine resources section")
        self.write_machine_resources_section()

        # Run over each dataset
        for ds_config in self.config["datasets"]:
            print("- Processing dataset {:d}".format(self.ds_counter))
            # Load the input dataset
            self.load_dataset(ds_config)

            # Set the paths to the docs
            self.set_dataset_paths(ds_config)

            if self.data_analysis_section_was_already_written(ds_config) and not self.config["re-write"]:
                print("(Dataset {:d} analysis section was already written, skipping...)".format(self.ds_counter))
            else:
                # Write a data analysis section
                self.write_data_analysis_section(ds_config)

            # Some common variables for all tests
            self.ds_elevation_var = ds_config["elevation_var"]
            self.ds_intepolation_flag_var = ds_config["interpolation_flag_var"]

            # Run over each test
            for tst_config in ds_config["tests"]:
                print("    - Processing test {:d}".format(self.test_counter))
                # Set the paths to the test
                self.set_tests_paths(tst_config)

                if not self.test_was_already_executed(tst_config) or self.config["re-execute"]:
                    # Note that, in this case, both the execution and the document generation are considered to be created IF the test was executed

                    # Run the test
                    self.run_test(tst_config)

                    # Write a section with the results
                    self.write_test_section(tst_config)
                else:

                    if self.config["re-write"]:
                        # Write a section with the results of the execution
                        self.write_test_section(tst_config)

                    print("(Test {:d} was already executed, skipping...)".format(self.test_counter))
                self.test_counter += 1

            # If there is more than a test, compare them in a comparison section
            if len(ds_config["tests"]) > 1:
                self.write_tests_comparison_section(ds_config)

            self.ds_counter += 1
            self.test_counter = 0  # Reset the test counter

        # Compile all the sections in a single file
        self.merge_sections()

        # Render the final PDF
        print("- Rendering the report to PDF")
        self.render_report_to_pdf()

    def merge_sections(self):
        section_files = []

        # Intro
        section_dir = os.path.join(self.docs_dir, "intro")
        doc_path = os.path.join(section_dir, "intro.md")
        section_files.append(doc_path)

        # Resources
        section_dir = os.path.join(self.docs_dir, "resources")
        doc_path = os.path.join(section_dir, "resources.md")
        section_files.append(doc_path)

        self.ds_counter = 0
        self.test_counter = 0
        for ds_config in self.config["datasets"]:
            # Set the paths to the docs
            self.set_dataset_paths(ds_config)

            # Append to the list of sections
            section_files.append(self.doc_path)

            # Run over each test
            for tst_config in ds_config["tests"]:
                # Set the paths to the test
                self.set_tests_paths(tst_config)
                # Append to the list of sections
                section_files.append(self.doc_path)
                self.test_counter += 1

            # If there is more than a test, compare them in a comparison section
            if len(ds_config["tests"]) > 1:
                section_dir = os.path.join(self.docs_dir, "dataset_{:d}".format(self.ds_counter), "tests_comparison")
                doc_path = os.path.join(section_dir, "comparison.md")
                section_files.append(doc_path)

            self.ds_counter += 1
            self.test_counter = 0  # Reset the test counter

        with open(self.report_file_md, 'w') as outfile:
            for fname in section_files:
                with open(fname) as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")

    def render_report_to_pdf(self):
        # Requires pandoc and texlive!
        # args = ["pandoc",
        #         os.path.join(report_folder, "report.md"),
        #         "-H", "header.tex",
        #         "-B", "before_body.tex",
        #         "--listings",
        #         "-N", "-o", os.path.join(report_folder, "report.pdf")]
        # print("Rendering report to PDF")
        # ret_code = subprocess.call(args, cwd=config.PANDOC_TEMPLATES_DIR)

        # Get the templates into the report folder
        report_folder = os.path.dirname(self.report_file_md)
        pandoc_templates_dir = os.path.join(
            os.path.dirname(sys.modules["heightmap_interpolation.reporter.interpolate_netcdf4_reporter"].__file__),
            "templates")

        header_template_path = os.path.join(report_folder, 'header.tex')
        shutil.copyfile(os.path.join(pandoc_templates_dir, 'header.tex'), header_template_path)

        # Call pandoc
        args = ["pandoc", self.report_file_md,
                "-H", header_template_path,
                #"-B", "before_body.tex",
                "--listings",
                #"--latex-engine", "xelatex",
                "-N",
                #"-f", "markdown-implicit_figures",
                "-o", self.config["output_file"]]
        ret_code = subprocess.call(args)
        if ret_code != 0:
            raise ValueError("Rendering the report to a PDF failed with return code = {:d}".format(ret_code))

        # Write the tex file also (in case the user wants to manually edit it)
        if self.config["output_tex_file"]:
            args = ["pandoc", self.report_file_md,
                    "-H", header_template_path,
                    # "-B", "before_body.tex",
                    "--listings",
                    "-N",
                    "-s",
                    "-o", self.config["output_tex_file"]]
            ret_code = subprocess.call(args)

            if ret_code != 0:
                raise ValueError("Rendering the report to a TEX file failed with return code = {:d}".format(ret_code))
