Installation
============

This package is available through `PyPI <https://pypi.org/project/heightmap-interpolation/>`_, so it can be installed through ``pip``::

    pip install heightmap-interpolation

Installation from sources
*************************

This project expects python3.7 (or above).

Start by cloning the project: ::

    git clone https://github.com/coronis-computing/heightmap_interpolation.git

We provide the requirements of this project in a "requirements.txt"

If you want to do it in a virtual environment: ::

    cd <path_to_this_project>
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements

To ease the calls to the main ``interpolate_netcdf4.py`` script, you can add the container folder to the PATH: ::

    export PATH=$PATH:<path_to_this_package>/heightmap_interpolation/apps

In Ubuntu (and other linux distros), you can set this command as a new line in your ``<home>/.bash.rc`` for these changes to persist on new terminals.

Alternative: use the pre-compiled docker
****************************************

For convenience, we also provide a docker image with all the dependencies installed at `DockerHub <https://hub.docker.com/r/coroniscomputing/heightmap_interpolation>`_. Assuming you have docker installed, you can obtain it by:

::

    docker pull coroniscomputing/heightmap_interpolation:<tag_name>


Where ``<tag_name>`` must be a specific version of the package, or ``latest``.

Then, run it with (tested in Ubuntu): ::


    docker run -it --user $(id -u):$(id -g) -v <data_folder>:/data coroniscomputing/heightmap_interpolation:<tag_name>


On the one hand, using the ``-v`` flag we are mounting the directory containing the data to process to the ``/data`` folder within the container. On the other hand, the ``--user $(id -u):$(id -g)`` part is to achieve that the files you generate within docker in the mounted volume are owned by your user (otherwise they would be owned by the *root* user!).

The container will automatically run the ``bash`` command, and you will be inside the container. Thus, there you can simply run the ``interpolate_netcdf4.py`` script with the desired parameters (it is included in the container's path, so you can call it directly). For instance: ::

    interpolate_netcdf4.py -o /data/<netcdf_results_file> linear /data/<netcdf_input_file>

Keep in mind that this way of running the docker does not provide visualization, so the ``--show`` flag will be useless! There are ways of sharing the Xs with docker, but these are out of the scope of this documentation.