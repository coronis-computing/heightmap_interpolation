# Changelog

Starting from v1.1.0, all version-specific notable changes to this project will be documented in this file.

## v1.1.0

* Some signature changes in the CLI tools. Now the interpolation scripts (`interpolate_netcdf4` and `interpolate_xyz`) do not assume any projection for the input, and the names of the variables containing the x/y/elevation values can be defined from the command line with `--x_var`, `--y_var` and `--elevation_var`. This allows using the interpolation on data using other projections, such as UTM. The defaults follow the previous assumptions (i.e., `--x_var lon --y_var lat --elevation_var elevation`), so the scripts should work as before if none of these arguments are passed.
* Moved some of the methods (`ebi`, `shiftmap`) that are not ready to be used by the general public to the experimental tag. To enable, install the `experimental` dependencies with `pip install .[experimental]`. Note that this adds some huge dependencies, such as pytorch (`torch`), which makes the installation larger on disk.
* Added two new methods:
    - `ams`: Adaptive Multi-grid Solver.
    - `mlp`: Multi-Layer Perceptron. It is listed as an experimental feature, and it is not enabled by default. 
* Implemented a direct solver for `sobolev` and `ccst` inpainting methods, which runs much faster than our previous iterative (time-stepping) approach. Set `--use_direct_solver` flag to use it.
* Changed the dependencies:
    - Using the headless version of `opencv`/`opencv-contrib` packages.
    - Separated the dependencies that are only used for building the docs and experimental features. Now a default `pip install .` will not include the dependencies needed for building the docs nor the experimental features. These should be installed with `pip install .[docs,experimental]`.
* Moved some internal (undocumented) tools to another project:
    - `grid_to_xyz.py`
    - `randomly_sample_xyz_from_netcdf4.py`
    - `erase_areas.py`
    - `print_netcdf.py`
* Updated github actions with publication to `testpypi` for each commit to the `master` branch and publication to `pypi` for each release.
* Updated the docs to reflect the new methods/solvers/parameters (excluding experimental methods).
* Changed the way results are shown on screen if --show parameter is used. Also, now the user may change the colormap (`--colormap`) or highlight the areas to interpolate (`--highlight_interpolated_area`).
* Added two demos showing the behaviour of the two types of methods in the package with synthetic data and using default parameters: `tests/demo_scattered_interpolants.py` and `tests/demo_gridded_inpainters.py`.
