Usage
=====

Once installed, all the methods in this package can be used in your Python projects as in any other package (see the docs of :mod:`heightmap_interpolation`).

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

First get the usage of this command by running it with the ``-h`` flag: ::

    interpolate_netcdf4.py -h

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

As you can see, this first help shows the parameters regarding the input and output data (under ``optional arguments``) and the different interpolation methods available (under ``positional arguments``). The input file goes at the end of the command (i.e., the last positional parameter is interpreted as the input file).

The parameters related to the input data are the following:

* ``-o`` or ``--output_file``: the output NetCDF with the interpolated values. It will basically copy the input NetCDF file and modify its elevation variable (as well as the interpolation_flag accordingly, if available).
* ``--areas``: a path to a KML file containing the areas where the interpolation will take place.
* ``--elevation_var``: string ID within the NetCDF pointing to the elevation data grid to interpolate.
* ``--interpolation_flag_var``: string ID within the NetCDF pointing to the ``interpolation_flag`` variable. If specified, it will use this variable in the NetCDF as an indicator of where to interpolate (will just interpolate areas where interpolation_flag = 1). Otherwise, it will interpolate "invalid" values within the elevation variable.
* ``-v`` or ``--verbose``: flag to activate verbose output during execution.
* ``-s`` or ``--show``: flag to show the interpolation result on screen after a successful execution.

**Important note:** interpolation at areas contained in the KML file pointed by ``--areas`` will just solve the inpainting using both reference and missing data within those areas. That is, **only** the reference points falling within the area will contribute to the interpolation of the missing data. In other words: do not just mark the "holes" in your data, but also mark the "data you want to use to interpolate those holes".

The parameters specific to each method will appear if you call the function with the ``positional_argument`` and the ``-h`` flag. E.g.: ::

    $ interpolate_netcdf4.py harmonic -h
    usage: interpolate_netcdf4.py harmonic [-h]
                                           [--update_step_size UPDATE_STEP_SIZE]
                                           [--rel_change_tolerance REL_CHANGE_TOLERANCE]
                                           [--rel_change_iters REL_CHANGE_ITERS]
                                           [--max_iters MAX_ITERS]
                                           [--relaxation RELAXATION]
                                           [--print_progress_iters PRINT_PROGRESS_ITERS]
                                           [--mgs_levels MGS_LEVELS]
                                           [--mgs_min_res MGS_MIN_RES]
                                           [--init_with INIT_WITH]
                                           [--convolver CONVOLVER]
                                           [--debug_dir DEBUG_DIR]

    optional arguments:
      -h, --help            show this help message and exit
      --update_step_size UPDATE_STEP_SIZE
                            Update step size
      --rel_change_tolerance REL_CHANGE_TOLERANCE
                            If the relative change between the inpainted
                            elevations in the current and a previous step is
                            smaller than this value, the optimization will stop
      --rel_change_iters REL_CHANGE_ITERS
                            Number of iterations in the optimization after which
                            we will check if the relative tolerance is below the
                            threshold
      --max_iters MAX_ITERS
                            Maximum number of iterations in the optimization.
      --relaxation RELAXATION
                            Set to >1 to perform over-relaxation at each iteration
      --print_progress_iters PRINT_PROGRESS_ITERS
                            If '--print_progress True', the optimization progress
                            will be shown after this number of iterations
      --mgs_levels MGS_LEVELS
                            Levels of the Multi-grid solver. I.e., number of
                            levels of detail used in the solving pyramid
      --mgs_min_res MGS_MIN_RES
                            If during the construction of the pyramid of the
                            Multi-Grid Solver one of the dimensions of the grid
                            drops below this size, the pyramid construction will
                            stop at that level
      --init_with INIT_WITH
                            Initialize the unknown values to inpaint using a
                            simple interpolation function. If using a MGS, this
                            will be used with the lowest level on the pyramid.
                            Available initializers: 'nearest' (default), 'linear',
                            'cubic', 'sobolev'
      --convolver CONVOLVER
                            The convolution method to use. Available: 'opencv'
                            (default),'scipy-signal', 'scipy-ndimage', 'masked',
                            'masked-parallel'
      --debug_dir DEBUG_DIR
                            If set, debugging information will be stored in this
                            directory (useful to visualize the inpainting
                            progress)

The different options and their meaning for each specific method will be listed in the corresponding section at :ref:`methods`.