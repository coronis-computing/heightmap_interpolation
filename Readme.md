# EMODnet - Heightmap Interpolation (Python)

Interpolation functions for heightmaps developed within the EMODnet Bathymetry (High Resolution Seabed Mapping) project.

## Requirements

This project uses Python 3 and the following libraries:
* numpy
* haversine
* mpmath
* argparse
* scipy
* matplotlib
* netCDF4
* geopy
* geopandas
* Pillow

## Installation

In order to install the library and all requirements just use:

```
python setup.py install 
```

## Usage

This toolbox implements a set of Radial Basis Functions (RBFs) interpolants on elevation data. The set of RBF is the following:
* linear.
* cubic.
* quintic.
* thinplate.
* green.
* multiquadric.
* tension.
* regularized.
* gaussian.
* wendland.

A simple demo of the behaviour of each RBF can be seen in the `demo_rbf.py` script:

```
python demo_rbf.py
```
(run `demo_rbf.py -h` for futher options).

Since the complexity of a RBF interpolant highly depends on the number of input (reference) data points, we provide some 
heuristics in order to allow the interpolation of large-scale data. We use a Partition of Unity (PU) approach, where we subdivide
the domain into overlapping circles containing a minimum number of data points, where we compute a local RBF interpolant.
The contributions of the different interpolants are blended together using PU weighting. Since the domain decomposition 
follows a Quad-Tree space partition, we name it QTPURBF

We provide the app `interpolate_netcdf4.py` to run the interpolation using both the original RBF formulation and/or the QTPURBF on NetCDF4 following 
the specifications of the EMODnet Bathymetry project. In its simple form, one just need to run:

```
python interpolate_netcdf4.py <input_netcdf> <output_netcdf>
```

It takes the non-interpolated data points in the NetCDF to create an interpolant using either RBF (for small datasets) 
or QTPURBF (for large datasets). Then, it evaluates the interpolant at the required query locations to interpolate.  

However, you can list all available options with the `-h` or `--help` flag:

```
python interpolate_netcdf4.py -h
```

While the message generated with the command above already lists the meaning of each individual parameter, we list some 
of them in more detail in the following:

* Related to the definition of the interpolation problem:
    * `--areas <string>`: Path to the KML file containing the areas that will be interpolated. In case it is not specified, the cells for which the cell_interpolation_flag variable is True will be used as query points. 
    * `--elevation_var <string>`: Name of the variable storing the elevation grid in the input file.
    * `--cell_interpolation_flag_var <string>`: Name of the variable storing the per-cell interpolation flag in the input file. It is used in case no `--areas` is specified, but also to define the reference data, i.e., the points that are NOT interpolated.
    * `--query_block_size <integer>`: Query the interpolant in blocks of maximum this size, in order to avoid having to store large matrices in memory.
* Related to the computation of the RBF interpolant. Note that most of them also define the local RBF interpolation in case of using the PU:
    * `--rbf_max_ref_points <integer>`: Maximum number of data points to use a single RBF interpolation. Datasets with a number of reference points greater than this will use a partition of unity.
    * `--rbf_distance_type <string>`: Distance type. Available: euclidean (default), haversine, vincenty. While it may be more correct to use haversine or vincenty functions in geographic coordinates, keep in mind that they are not as optimized as the Euclidean distance, and therefore their use in large datasets is NOT RECOMMENDED.
    * `--rbf_type RBF_TYPE <string>`. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate (default), wendland.
    * `--rbf_epsilon <float>`: Epsilon parameter of the RBF. Also known as "shape parameter" in the RBF literature. Please check each RBF documentation for its meaning. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland.
    * `--rbf_regularization <float>`: Regularization scalar to use in the RBF (optional).
    * `--rbf_polynomial_degree <integer>`: Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic).
* Related to QTPURBF interpolation:
    * `--pu_overlap <float>`: Overlap factor between circles in neighboring sub-domains in the partition. The radius of a QuadTree cell, computed as half its diagonal, is enlarged by this factor
    * `--pu_min_point_in_cell <integer>`: Minimum number of points in a QuadTree cell.
    * `--pu_min_cell_size_percent <float>`: Minimum cell size, specified as a percentage [0..1] of the max(width, height) of the query domain.
    * `--pu_overlap_increment <float>`: If, after creating the QuadTree, a cell contains less than pu_min_point_in_cell, the radius will be iteratively incremented until this condition is satisfied. This parameter specifies how much the radius of a cell increments at each iteration.
* Display results on screen:
    * `-v, --verbose`: Verbosity flag, activate it to have feedback of the current steps of the process in the command line.
    * `-s, --show`: Show/plot interpolation problem and results.

## Acknowledgements

This project has been developed by Coronis Computing S.L. within the EMODnet Bathymetry (High Resolution Seabed Mapping) project.

* EMODnet: http://www.emodnet.eu/
* EMODnet (bathymetry): http://www.emodnet-bathymetry.eu/
* Coronis: http://www.coronis.es