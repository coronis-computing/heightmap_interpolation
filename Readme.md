# EMODnet - Heightmap Interpolation (Python)

Interpolation functions for heightmaps developed within the EMODnet Bathymetry (High Resolution Seabed Mapping) project.

Please visit the documentation at: https://emodnet-heightmap-interpolation.readthedocs.io/en/latest/

## Installation

This package is available through [PyPI](https://pypi.org/project/heightmap-interpolation/):

```
pip install heightmap-interpolation
```

Otherwise, the package and all its requirements can be installed from sources through `setuptools` using:

```
python setup.py install 
```

Note that it requires python >= 3.10 to run.

If you prefer to install it in a virtual environment:

```
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install .
```

### Optional (experimental) methods

Some interpolation methods rely on heavier, optional dependencies and are only
available when these are installed:

```
pip install heightmap_interpolation[experimental]
```

In particular, the `gmt_surface` method wraps GMT's `surface` gridder through
[PyGMT](https://www.pygmt.org). Besides the `pygmt` Python package (pulled in by
the `experimental` extra), it additionally requires the GMT binaries to be present
on the system, e.g. via conda: `conda install -c conda-forge gmt pygmt`. The method
only appears in the list of available methods when `pygmt` can be imported.

## Usage

This package installs two tools for interpolating from the command line: 

* `interpolate_netcdf4`: interpolates an already-gridded dataset in a NetCDF4 file.
* `interpolate_xyz`: takes a set of points (XYZ coordinates in a text file, one line each), creates a grid, and fills this grid with the interpolated values.

You can check the parameters of both tools with the `--help` argument:

```
interpolate_netcdf4 --help
interpolate_xyz --help
```

For other uses, take the code in the `apps/interpolate_netcdf4.py` script as reference and use directly the different interpolation modules at your convenience. 

## Docker

For convenience, we also provide a docker image with all the dependencies intalled at DockerHub. Assuming you have docker installed, you can obtain it by:

```
docker pull coroniscomputing/heightmap_interpolation:<tag_name>
```

Where `<tag_name>` must be a specific version of the package, or `latest`.

Or, if you want to compile the docker image by yourself:

```
docker build -t <image_tag_name> .
```

Then, run it with:

```
docker run -it -v <data_folder>:/data coroniscomputing/heightmap_interpolation:<tag_name>
```

On the one hand, using the `-v` flag we are mounting the directory containing the data to process to the `/data` folder within the container. The container will automatically run the `bash` command, and you will be inside the container. Thus, there we simply run the `interpolate_netcdf4` script with the desired parameters

For instance:

```
interpolate_netcdf4 -o /data/<netcdf_results_file> linear /data/<netcdf_input_file>
```  

Keep in mind that this way of running the docker does not provide visualization, so the "--show" flag will be useless! There are ways of sharing the Xs with docker, but these are out of the scope of this documentation.

## Acknowledgements

This project has been developed by Coronis Computing S.L. within the EMODnet Bathymetry (High Resolution Seabed Mapping) project.

* EMODnet: http://www.emodnet.eu/
* EMODnet (bathymetry): http://www.emodnet-bathymetry.eu/
* Coronis: http://www.coronis.es

