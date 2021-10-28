#!/usr/bin/env python3

import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import json
from heightmap_interpolation.apps.common import load_data_impl, write_results_impl
from heightmap_interpolation.inpainting.fd_pde_inpainter_factory import create_fd_pde_inpainter
from timeit import default_timer as timer
import time
import psutil
import multiprocessing as mp
import queue
# import billiard as mp

def create_default_config():
    config = {
        # Configuration of the reporter
        "work_dir": "./report", # The working directory, here results as well as the report doc sources will be stored
        "title": "Execution report", # The title of the report
        "further_intro_text": "", # Use this option to introduce further custom text in the intro section
        "re-write": False, # If set, all the text sections will be re-written
        "re-execute": False,  # If set, all the tests will be re-executed and tests sections will be re-written
        "output_file": "./report.pdf", # The final report file
        "datasets": [ # A list of datasets. A report can contain different datasets
            {
                "name": "Example dataset", # A name/ID/alias for the dataset
                "netcdf_file": "", # The actual NetCDF file containing the data
                "elevation_var": "elevation", # The name of the variable inside the netcdf file to be considered as the elevation/bathymetry
                "interpolate_missing_values": True,
                "tests": [ # A list of tests. Each dataset can be used in multiple tests
                    {
                        "name": "", # # A name/ID/alias for the test, if needed
                        "inpainting_method": "sobolev",
                        "inpainting_config": # The configuration for the inpainter, in the same format as expected by fd_pde_inpainter.
                            {
                                "update_step_size": 0.01,
                                "rel_change_tolerance": 1e-8,
                                "max_iters": 1e8,
                                "relaxation": 0,
                                "mgs_levels": 1,
                                "print_progress": True,
                                "print_progress_iters": 1000,
                                "init_with": "zeros",
                                "convolver": "opencv"
                            }
                    }
                ]
            }
        ]

    }


def inpainting_thread(inpainter, elevation, inpainting_mask, inpainted_elevation_mp, queue):
    # Inpaint (time the execution time)
    ts = timer()
    inpainted_elevation = inpainter.inpaint(elevation, inpainting_mask)
    te = timer()
    duration = te-ts

    # Convert the results to the shared mp.Array
    inpainted_elevation_mp_np = np.reshape(np.frombuffer(inpainted_elevation_mp.get_obj(), dtype=elevation.dtype),
                                           elevation.shape)
    np.copyto(inpainted_elevation_mp_np, inpainted_elevation)

    # Queue other return data (in this case, the duration of the execution)
    queue.put([duration])
    # queue.put([True])
    # queue.cancel_join_thread()

class InpaintingReporter():
    # Executes a series of tests described in the config file, and creates a document in PDF with the results
    # Each section is written in a separate markdown file, and joined at the end in a single file

    def __init__(self, config):
        # Store the configuration
        self.config = config
        # Different paths used while reporting
        self.results_dir = os.path.join(self.config["work_dir"], "results")
        self.docs_dir = os.path.join(self.config["work_dir"], "docs")
        self.report_file_md = os.path.join(self.config["work_dir"], "report.md")
        self.section_files = [] # Will be populated after writing a given section
        # Data/info of the current dataset
        self.ds_config = None
        self.ds_counter = 0
        self.ds_elevation = []
        self.ds_inpainting_mask = []
        self.ds_min_elevation = 0
        self.ds_max_elevation = 0
        # Data/info of the current test
        self.test_counter = 0
        self.test_subdir = ""
        self.test_base_name = ""
        self.inpainter = None
        self.inpainted_elevation = None
        self.run_duration = 0
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

    def call_function_and_account_resources(self, fun, args):
        # Init resources' stats
        self.res_num_queries = 0
        self.res_once_cpu_mean = True
        self.res_cpu_percent_mean = 0
        self.res_cpu_percent_max = 0
        self.res_memory_mean = 0
        self.res_once_memory_mean = 0
        self.res_memory_max = 0

        # Call function in a thread
        worker_process = mp.Process(target=fun, args=args)
        worker_process.start()

        # Call psutil on the created thread
        p = psutil.Process(worker_process.pid)

        # Monitor each period
        while worker_process.is_alive():
            time.sleep(self.update_period)
            self.update_resources_stats(p)

        worker_process.join()

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
        f.write("This report presents the execution results using the inpainting methods of the `heightmap_interpolation` toolbox.\n\n")

        # Close the file
        f.close()

        # Append to the list of sections
        self.section_files.append(doc_path)

    def load_dataset(self, ds_config):
        # Read the file
        elevation, mask_int, lats_mat, lons_mat, mask_ref = load_data_impl(ds_config["netcdf_file"],
                                                                           ds_config["elevation_var"],
                                                                           ds_config["interpolate_missing_values"])

        # Store the elevation and inpainting mask (will be used later in all the tests to be performed on this dataset)
        self.ds_inpainting_mask = ~mask_int
        self.ds_elevation = elevation

        # Some other statistics used by other sections
        self.ds_min_elevation = np.amin(self.ds_elevation[self.ds_inpainting_mask])
        self.ds_max_elevation = np.amax(self.ds_elevation[self.ds_inpainting_mask])

    def set_dataset_paths(self, ds_config):
        self.section_dir = os.path.join(self.docs_dir, "dataset_{:d}".format(self.ds_counter), "data_analysis")
        self.doc_path = os.path.join(self.section_dir, "intro.md")
        os.makedirs(self.section_dir, exist_ok=True)
        self.ds_config = ds_config # TODO: do we need this? When using it in other functions, we should decide whether we use it as a function parameter or we rely on the attributes of the class...

    def set_tests_paths(self, tst_config):
        self.test_subdir = os.path.join("dataset_{:d}".format(self.ds_counter), "test_{:d}".format(self.test_counter))
        self.section_dir = os.path.join(self.docs_dir, self.test_subdir)
        self.results_dir = os.path.join(self.results_dir, self.test_subdir)
        self.doc_path = os.path.join(self.section_dir, "test_{:d}.md".format(self.test_counter))
        os.makedirs(self.section_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
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
            # Append to the list of sections
            self.section_files.append(self.doc_path)
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
        axes[1].imshow(self.ds_inpainting_mask, cmap=plt.cm.gray)
        divider = make_axes_locatable(axes[1])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        axes[1].set_title("Known (white) / Unknown (black) data")
        axes[1].set_axis_off()
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
            fig_caption += ": {:s}".format(ds_config["name"])

        f.write("The input dataset {:s} can be seen in figure \\ref{{{:s}}}.\n\n".format(ds_config["name"],
                                                                                         tex_figure_label))
        f.write("![{:s}\label{{{:s}}}]({:s})\n\n".format(fig_caption, tex_figure_label, input_data_image_path))

        f.write("It has the following properties:\n\n")

        storage_size = os.path.getsize(input_data_image_path) / (1024*1024)
        total_pixels = self.ds_elevation.shape[0] * self.ds_elevation.shape[1]
        num_pixels_to_inpaint = np.count_nonzero(~self.ds_inpainting_mask)
        inpaint_percent = (num_pixels_to_inpaint / total_pixels) * 100

        f.write("| **Width** | **Height** | **Size (MBytes)** | **Min. elevation** | **Max. elevation** |\n"+
                "| :---: | :---: | :---: | :---: | :---: | \n" +
                "| {:d} | {:d} | {:.2f} | {:.2f} | {:.2f} |\n\n".format(self.ds_elevation.shape[1], self.ds_elevation.shape[0], storage_size, self.ds_min_elevation, self.ds_max_elevation))

        f.write("And it presents the following inpainting problem:\n\n")

        f.write("| **Num. cells** | **Unknown cells** | **Known cells** | **\% to inpaint** |\n"+
                "| :---: | :---: | :---: | :---: |\n" +
                "| {:d} | {:d} | {:d} | {:.2f} |\n\n".format(total_pixels, total_pixels-num_pixels_to_inpaint, num_pixels_to_inpaint, inpaint_percent))

        f.close()

        # Write the dataset config in the section folder as an indicator that the section was already written
        config = os.path.join(self.section_dir, "config.json")

        ds_config_copy = copy.deepcopy(ds_config)
        ds_config_copy.pop("tests")
        with open(config, 'w') as fp:
            json.dump(ds_config_copy, fp, indent=4)

        # Append to the list of sections
        self.section_files.append(self.doc_path)

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
        # Create some output folders
        test_subdir = os.path.join("dataset_{:d}".format(self.ds_counter), "test_{:d}".format(self.test_counter))
        results_dir = os.path.join(self.results_dir, test_subdir)
        os.makedirs(results_dir, exist_ok=True)

        # Create the inpainter
        # inpainter = create_fd_pde_inpainter(tst_config["inpainting_method"], tst_config["inpainting_config"])

        # Check if we already executed the test

        # Create a shared variable for storing the inpainting results, which will be computed on a different process
        inpainted_elevation = np.zeros(self.ds_elevation.shape, dtype=self.ds_elevation.dtype)
        inpainted_elevation_ctype = np.ctypeslib.as_ctypes_type(inpainted_elevation.dtype)  # Same as elevation
        inpainted_elevation_mp = mp.Array(inpainted_elevation_ctype, inpainted_elevation.size)

        # Execute the inpainting (and time it)
        # ts = timer()
        # inpainted_elevation = inpainter.inpaint(self.ds_elevation, self.ds_inpainting_mask)
        # Call inpainting in a thread (to be able to account resources used during execution)
        results_queue = mp.Queue()
        # results_queue = queue.Queue()
        self.call_function_and_account_resources(inpainting_thread, (self.inpainter, self.ds_elevation, self.ds_inpainting_mask, inpainted_elevation_mp, results_queue))
        # te = timer()
        # duration = te-ts

        # Retrieve the results of the threaded call
        # ret_vals = results_queue.get()
        # print(ret_vals)
        # inpainted_elevation = ret_vals[0]
        ret_vals = results_queue.get()
        self.run_duration = ret_vals[0]
        self.inpainted_elevation = np.reshape(
            np.frombuffer(inpainted_elevation_mp.get_obj(), dtype=inpainted_elevation.dtype), inpainted_elevation.shape)

        # Save the results
        if tst_config["name"]:
            test_base_name = "{:s}_test_{:d}".format(tst_config["name"], self.test_counter)
        else:
            test_base_name = "test_{:d}".format(self.test_counter)
        inpainting_results_netcdf = os.path.join(results_dir, test_base_name + ".nc")
        write_results_impl(inpainting_results_netcdf,
                           self.ds_config["netcdf_file"],
                           inpainted_elevation, self.ds_inpainting_mask,
                           elevation_var=self.ds_config["elevation_var"],
                           interpolate_missing_values=self.ds_config["interpolate_missing_values"])

        # Save the configuration also, so that we have a reference, and a mark that the test was executed
        test_config = os.path.join(self.results_dir, self.test_base_name + "_config.json")

        # Update the config to store with the actual (full) inpainter configuration
        inpainter_config = self.inpainter.get_config()
        tst_config["actual_inpainter_config"] = inpainter_config
        with open(test_config, 'w') as fp:
            json.dump(tst_config, fp, indent=4)

    def test_was_already_executed(self, tst_config):
        # Create the inpainter
        self.inpainter = create_fd_pde_inpainter(tst_config["inpainting_method"], tst_config["inpainting_config"])

        if self.config["re-execute"]:
            return False

        # # Check that output paths exist at least
        # test_subdir = os.path.join("dataset_{:d}".format(self.ds_counter), "test_{:d}".format(self.test_counter))
        # section_dir = os.path.join(self.docs_dir, test_subdir)
        # results_dir = os.path.join(self.results_dir, test_subdir)
        # doc_path = os.path.join(section_dir, "test_{:d}.md".format(self.test_counter))
        # if not os.path.exists(results_dir) or not os.path.exists(doc_path):
        #     return False

        # Check if the run requires a previously-executed configuration

        # Get the configuration of the inpainter object
        inpainter_config = self.inpainter.get_config()

        # Compare this config with the one stored from a previous execution
        # Save the configuration also, so that we have a reference, and a mark that the test was executed


        prev_config_file = os.path.join(self.results_dir, self.test_base_name + "_config.json")
        if not os.path.exists(prev_config_file):
            return False
        f = open(prev_config_file)
        prev_config = json.load(f)

        # Check that both configurations are equal
        if prev_config["name"] != tst_config["name"]:
            print("The 'name' variable changed!")
            return False
        if prev_config["inpainting_method"] != tst_config["inpainting_method"]:
            print("The 'inpainting_method' changed!")
            return False
        if prev_config["actual_inpainter_config"] != inpainter_config:
            print("The configuration for the inpainter changed!")
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

        f.write("Executed the *{:s}* inpainting method with the following parameters (includes defaults):\n\n".format(tst_config["inpainting_method"]))

        config = self.inpainter.get_config()
        for key, value in config.items():
            f.write("* {:s} = {:s}\n".format(key, str(value)))

        f.write("\n")

        # Show the result as an image
        fig, ax = plt.subplots()
        ax.imshow(self.inpainted_elevation)
        ax.set_axis_off()
        plt.tight_layout()

        # Save it
        images_dir = os.path.join(self.section_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        results_image_path = os.path.join(images_dir, self.test_base_name + "_result.png")
        plt.savefig(results_image_path, bbox_inches="tight")

        # Render it on the doc (as a Markdown link)
        tex_figure_label = "fig:inpainting_test_{:d}".format(self.test_counter)
        fig_caption = "Inpainting results of test {:d} {:s}".format(self.test_counter, tst_config["name"])
        f.write("The inpainting results can be seen in figure \\ref{{{:s}}}.\n\n".format(tex_figure_label))
        f.write("![{:s}\label{{{:s}}}]({:s})\n\n".format(fig_caption, tex_figure_label, results_image_path))

        # Write further info on the results
        # f.write("The inpainting process took {:.2f} seconds. ".format(duration))

        # Check that the inpainter did not overshoot the input data
        inpainted_elevation_min = np.amin(self.inpainted_elevation)
        inpainted_elevation_max = np.amax(self.inpainted_elevation)
        overshooted_min = inpainted_elevation_min < self.ds_min_elevation
        overshooted_max = inpainted_elevation_max > self.ds_max_elevation
        if overshooted_min or overshooted_max:
            f.write("**WARNING**: the inpainting results present some **overshooting** with respect to the original known input values:\n\n")
        if overshooted_min:
            f.write("* Original minimum elevation was {:.2f}, but in the inpainted result is {:.2f}\n".format(self.ds_min_elevation, inpainted_elevation_min))
        if overshooted_max:
            f.write("* Original maximum elevation was {:.2f}, but in the inpainted result is {:.2f}\n".format(self.ds_max_elevation, inpainted_elevation_max))

        f.write("\n\n")

        # Add statistics regarding the resources used
        # f.write("\n### Execution Summary\n\n")

        duration_unit = "seconds"
        duration = self.run_duration
        if self.run_duration > 60:
            duration = duration / 60
            duration_unit = "minutes"
        if self.run_duration > 60:
            duration = duration / 60
            duration_unit = "hours"
        f.write("The execution took a total of **{:.2f} {}**, and it required the following resources:\n\n".format(duration, duration_unit))
        f.write(
            "| **Mean CPU %** | **Max. CPU %** | **Total CPU time (s)** | **Mean Memory (MB)** | **Max Memory (MB)** |\n"
            "| :---: | :---: | :---: | :---: | :---: |\n"
            "| {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |\n\n".format(self.res_cpu_percent_mean,
                                                                        self.res_cpu_percent_max,
                                                                        self.res_cpu_time,
                                                                        self.res_memory_mean / (1024*1024),
                                                                        self.res_memory_max / (1024*1024)))
        f.write(
            "Note that the CPU statistics above are computed accross all used CPUs, so multithreaded "
            "calls may potentially exceed the 100% CPU usage and cause the total CPU time to be larger "
            "than the actual execution time.\n\n")

        # Close the file
        f.close()

        # Append to the list of sections
        self.section_files.append(self.doc_path)

    def report(self):
        """Main entry point, will execute the desired tests and render the docs"""

        # Create the main dirs
        self.init_folders()

        # Intro
        print("- Writing generic intro section")
        self.write_intro_section()

        # TODO: Describe resources where the tests will take place

        # Run over each dataset
        for ds_config in self.config["datasets"]:
            print("- Processing dataset {:d}".format(self.ds_counter))
            # Load the input dataset
            self.load_dataset(ds_config)

            # Set the paths to the docs
            self.set_dataset_paths(ds_config)

            if self.data_analysis_section_was_already_written(ds_config) and not self.config["re-write"]:
                print("(Dataset {:d} analysis section was already written, skipping...)".format(self.ds_counter))
                # Append to the list of sections
                self.section_files.append(self.doc_path)
            else:
                # Write a data analysis section
                self.write_data_analysis_section(ds_config)

            # Run over each test
            for tst_config in ds_config["tests"]:
                print("    - Processing test {:d}".format(self.test_counter))
                # Set the paths to the test
                self.set_tests_paths(tst_config)

                if not self.test_was_already_executed(tst_config) and not self.config["re-execute"]:
                    # Note that, in this case, both the execution and the document generation are considered to be created IF the test was executed

                    # Run the test
                    self.run_test(tst_config)

                    # Write a section with the results
                    self.write_test_section(tst_config)
                else:
                    # Append to the list of sections
                    self.section_files.append(self.doc_path)
                    print("(Test {:d} was already executed, skipping...)".format(self.test_counter))
                self.test_counter += 1

            self.ds_counter += 1
            self.test_counter = 0  # Reset the test counter

        # Compile all the sections in a single file
        self.merge_sections()

        # Render the final PDF
        print("- Rendering the report to PDF")
        self.render_report_to_pdf()

    def merge_sections(self):
        with open(self.report_file_md, 'w') as outfile:
            for fname in self.section_files:
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

        args = ["pandoc", self.report_file_md,
                "-H", "header.tex",
                #"-B", "before_body.tex",
                "--listings",
                "-N",
                "-o", self.config["output_file"]]

        pandoc_templates_dir=os.path.join(os.path.dirname(sys.modules["heightmap_interpolation.reporter.inpaint_netcdf_reporter"].__file__), "templates")

        ret_code = subprocess.call(args, cwd=pandoc_templates_dir)

        if ret_code != 0:
            raise ValueError("Rendering the report to a PDF failed with return code = {:d}", ret_code)
