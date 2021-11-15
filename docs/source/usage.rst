Usage
=====

You can use the modules in this package directly within your code (see :ref: `Python Packages and Modules`).

However, since this toolbox was developed within the `EMODnet Bathymetry <https://www.emodnet-bathymetry.eu/>`_ (High Resolution Seabed Mapping) project, we also provide a python script that accepts NetCDF4 files following the specifications of the EMODnet Bathymetry project.
More precisely, we expect a NetCDF4 file describing an elevation map in Geodetic (WGS84) coordinates. While the original format includes several other fields, in this project we just expect the file to contain the following variables:

* ``lat``: vector of type ``float64`` with the latitude coordinates.
* ``lon``: vector of type ``float64`` with the longitude coordinates.
* ``elevation``: elevation data grid of type ``float32`` and size (<number_of_lats>, <number_of_lons>).
* ``interpolation_flag`` (optional): grid of type ``int8`` the same size as that on ``elevation`` indicating if the data on the ``elevation`` grid was interpolated (1) or not (0).

You can download samples of files following this format from the `EMODnet Bathymetry Viewing and Download Service <https://portal.emodnet-bathymetry.eu/>`_.

The main script interpolating this input is ``interpolate_netcdf4.py``. It provides a single entry point to execute the different interpolation methods in the package, also allowing to tune the different parameters of each method to your convenience. Broadly speaking, the signature of the method is as follows: ::

    interpolate_netcdf4.py [<io_data_parameters>] <interpolation_method_name> [<method_parameters>] <input_file>

There are two sets of optional parameters to set: the ones regarding the input data and the ones regarding the specific interpolation method to use.

First get the usage of this command by running it with the ``-h`` flag:

>>> interpolate_netcdf4.py -h

The result is the following: ::

    usage: interpolate_netcdf4.py [-h] [--output_file OUTPUT_FILE] [--areas AREAS]
                                  [--elevation_var ELEVATION_VAR]
                                  [--interpolation_flag_var INTERPOLATION_FLAG_VAR]
                                  [-v, --verbose] [-s, --show]
                                  {nearest,linear,cubic,rbf,purbf,harmonic,tv,ccst,amle,navier-stokes,telea,shiftmap}
                                  ... input_file

    Interpolate elevation data in a SeaDataNet_1.0 CF1.6-compliant netCDF4 file

    positional arguments:
      {nearest,linear,cubic,rbf,purbf,harmonic,tv,ccst,amle,navier-stokes,telea,shiftmap}
                            sub-command help
        nearest             Nearest-neighbor interpolator
        linear              Linear interpolator
        cubic               Piecewise cubic, C1 smooth, curvature-minimizing
                            (Clough-Tocher) nterpolator
        rbf                 Radial Basis Function interpolant
        purbf               Partition of Unity Radial Basis Function interpolant
        harmonic            Harmonic inpainter
        tv                  Inpainter minimizing Total-Variation (TV) across the
                            'image'
        ccst                Continous Curvature Splines in Tension (CCST)
                            inpainter
        amle                Absolutely Minimizing Lipschitz Extension (AMLE)
                            inpainter
        navier-stokes       OpenCV's Navier-Stokes inpainter
        telea               OpenCV's Telea inpainter
        shiftmap            OpenCV's xphoto module's Shiftmap inpainter
      input_file            Input NetCDF file

    optional arguments:
      -h, --help            show this help message and exit
      --output_file OUTPUT_FILE
                            Output NetCDF file with interpolated values
      --areas AREAS         KML file containing the areas that will be
                            interpolated.
      --elevation_var ELEVATION_VAR
                            Name of the variable storing the elevation grid in the
                            input file.
      --interpolation_flag_var INTERPOLATION_FLAG_VAR
                            Name of the variable storing the per-cell
                            interpolation flag in the input file (0 == known
                            value, 1 == interpolated/to interpolate cell). If not
                            set, it will interpolate the locations in the
                            elevation variable containing an invalid (NaN) value.
      -v, --verbose         Verbosity flag, activate it to have feedback of the
                            current steps of the process in the command line
      -s, --show            Show interpolation problem and results on screen

As you can see, this first help shows the parameters regarding the input and output data (under ``optional arguments``) and the different interpolation methods available (under ``positional arguments``).