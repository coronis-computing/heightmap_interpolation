# Changelog

Starting from v1.1.0, all version-specific notable changes to this project will be documented in this file.

## v1.1.0

* Some signature changes in the CLI tools. Now the interpolation scripts (`interpolate_netcdf4` and `interpolate_xyz`) do not assume any projection for the input, and the names of the variables containing the x/y/elevation values can be defined from the command line with `--x_var`, `--y_var` and `--elevation_var`. This allows using the interpolation on data using other projections, such as UTM. The defaults follow the previous assumptions (i.e., `--x_var lon --y_var lat --elevation_var elevation`), so the scripts should work as before if none of these arguments are passed.
* Added two new methods:
    - `ams`: Adaptive Multi-grid Solver. Using the 
    - `mlp`: Multi-Layer Perceptron. It is now listed as an experimental feature, and it is not enabled by default. To enable, install the `experimental` dependencies with `pip install .[experimental]`. Note that it adds the pytorch (`torch`) dependency, which requires a large amount of disk space...
* Changed the dependencies:
    - Using the headless version of `opencv`/`opencv-contrib` packages.
    - Separated the dependencies that are only used for building the docs and experimental features. Now a default `pip install .` will not include the dependencies needed for building the docs nor the experimental features. These should be installed with `pip install .[docs,experimental]`.
* Moved some internal (undocumented) tools to another project:
    - `grid_to_xyz.py`
    - `randomly_sample_xyz_from_netcdf4.py`
    - `erase_areas.py`
    - `print_netcdf.py`
* Updated github actions with publication to `testpypi` for each commit to the `master` branch and publication to `pypi` for each release.
