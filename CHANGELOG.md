# Changelog

Starting from v1.1.0, all version-specific notable changes to this project will be documented in this file.

## v1.2.0

* The GPU backend of the FD-PDE inpainters (`--backend gpu`) is now an optional dependency: install it with `pip install heightmap_interpolation[gpu]`. It now uses [Taichi Forge](https://pypi.org/project/taichi-forge/), a maintained fork of Taichi supporting Python 3.10-3.14 (the original `taichi` package is still accepted if installed). If no Taichi-compatible package is available, `--backend gpu` falls back to the CPU implementation with a warning, and if the requested GPU architecture cannot be initialized, Taichi falls back to `cpu`.
* Methods depending on compiled/optional packages (`ams`, the GPU backend, `ebi`) are now imported only when used, so a missing or broken package no longer prevents the rest of the methods from running.
* Requires `py_ams_point_interpolant>=0.1.1`, which fixes the import of the `ams` method on Windows and provides wheels for Python 3.13 and 3.14.
* The `experimental` extra restricts OpenCV to versions < 5, since the `exemplar-based-inpainting` package used by `ebi` does not work with OpenCV 5. The core package also works with OpenCV 5.
* Removed the dependency on `fiona`, which has no wheels for Python 3.14: now requires `geopandas>=1.0`, which reads the `--areas` KML files through `pyogrio`.
* Fixed the `ccst`/`ccst-ti` selection in `fd_pde_inpainter_factory.create_fd_pde_inpainter`.
* Fixed the multigrid solver (`--mgs_levels` > 1, the default) of the GPU backend (`--backend gpu`) of `harmonic` and `ccst`, which produced wrong results: the image pyramid now decimates instead of interpolating (unknown cells were blended into known ones), NaNs no longer propagate through the upscaled initial guesses, and the termination threshold no longer compounds across pyramid levels.
* Fixed the GPU backend (`--backend gpu`) of `harmonic` and `ccst` not solving the finer levels of the multigrid solver (`--mgs_levels` > 1, the default): the Taichi kernels stayed bound to the fields of the first (coarsest) pyramid level, so the full-resolution result was just the upscaled coarse solution, and the solver ran until `--max_iters` because the termination check also read the stale fields. Also removed a debug print executed at every iteration.
* Fixed the GPU backend (`--backend gpu`) pulling the inpainted values next to the grid borders towards 0 (e.g., gaps touching the border of the map, or of a `--areas` polygon): its convolutions now handle the borders as the CPU ones (OpenCV's `BORDER_REFLECT_101`) instead of zero-padding.
* Fixed the `ebi` method crashing when `--patch_preference` was not given (it now defaults to `closest`).
* Fixed `interpolate_xyz`: with `--areas` and a scattered method, the cells outside the areas were written as 0 and flagged as interpolated (they now keep the gridded input samples, or nodata); with several areas and a gridded method, each area overwrote the results of the previous ones; and the output grid had one cell less per dimension, so its cell size was slightly larger than `--cell_size`. Cells with input samples where a scattered method returns NaN (e.g., `linear` on the border of the data) now keep the sample value.
* Fixed the interpolation flag written by `interpolate_netcdf4` with `--areas`: missing cells outside the areas (left as nodata) were flagged as interpolated.
* Fixed the FD-PDE inpainters returning uninitialized (NaN) results when a multigrid solver was requested on a grid too small to build the pyramid.
* `ams --exact_interpolation` now works (requires `py_ams_point_interpolant>=0.1.1`; it crashed or returned a flat surface before).
* Added the possibility to call GMT surface (experimental).
* Added `--truncate_to_input_range` option to truncate interpolation results to input min/max.
* Added `--max_distance_to_data` option to leave output cells farther than the given distance (in coordinate units) from the nearest input data point as NaN/nodata instead of interpolating them. Works for both scattered and gridded methods.
* Added `--estimate_gradients` option to the `ams` method: estimates gradients at the input points (via a local plane fit, keeping only the reliable ones) and fits them in addition to the values. Tunable with `--gradient_neighbors`, `--gradient_max_distance` and `--gradient_min_planarity`. The estimation does not depend on the units of the coordinates (e.g., it works with geographic coordinates), and a warning is shown if no reliable gradient is found.
* Sped up the `nearest` initializer used by the FD-PDE inpainters (e.g. `ccst`): it now uses an exact Euclidean distance transform instead of `scipy.interpolate.griddata`, which is orders of magnitude faster on large grids with large areas to fill (~60x in our tests) while producing the same result.
* Changed the defaults for the FD-PDE inpainters (e.g. `ccst`): `--use_direct_solver` is now enabled by default for the methods supporting it (`harmonic` and `ccst`; `tv` and `amle` keep using the iterative solver), use `--no-use_direct_solver` to disable it, `--mgs_levels` defaults to `5` (was `1`), and `--cg_term_thres` defaults to `1e-4` (was `1e-6`).
* Changed default `--depth` of AMS to 10.

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
