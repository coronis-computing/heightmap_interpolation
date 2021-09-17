#!/usr/bin/env python3

# Copyright (c) 2020 Coronis Computing S.L. (Spain)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Ricard Campos (ricard.campos@coronis.es)

import setuptools

setuptools.setup(
      name='Heightmap Interpolation',
      version='1.0',
      description='EMODnet Bathymetry Heightmap Interpolation Package',
      author='Ricard Campos',
      author_email='ricard.campos@coronis.es',
      url='',
      packages=setuptools.find_packages(),
      install_requires=[
            'numpy',
            'haversine',
            'mpmath',
            'argparse',
            'scipy',
            'matplotlib',
            'netCDF4',
            'geopy',
            'geopandas',
            'Pillow',
            'Cython',
            'cvxpy',
            'opencv-python',
            'pyfftw'
      ],
      python_requires='>=3.5',
      )

# 'matplotlib==3.3',
#'scipy==1.5.4',
#'numpy==1.19.5',