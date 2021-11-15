Installation
============

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

..
    Alternative: use the pre-compiled docker
    ****************************************