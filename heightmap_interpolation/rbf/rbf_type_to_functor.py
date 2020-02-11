# Copyright (c) 2020 Coronis Computing S.L. (Spain)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Ricard Campos (ricard.campos@coronis.es)

from heightmap_interpolation.rbf.linear_rbf import *
from heightmap_interpolation.rbf.cubic_rbf import *
from heightmap_interpolation.rbf.quintic_rbf import *
from heightmap_interpolation.rbf.gaussian_rbf import *
from heightmap_interpolation.rbf.green_rbf import *
from heightmap_interpolation.rbf.multiquadric_rbf import *
from heightmap_interpolation.rbf.regularized_spline_rbf import *
from heightmap_interpolation.rbf.tension_spline_rbf import *
from heightmap_interpolation.rbf.thin_plate_spline_rbf import *
from heightmap_interpolation.rbf.wendland_csrbf import *


def rbf_type_to_functor(rbf_type: str, e):
    """Gets the appropriate function given a string type"""

    # The functions where the 'e' parameter is fixed according to the input parameter (i.e., mask the existence of 'e')
    def fixed_e_gaussian_rbf(r):
        return gaussian_rbf(r, e)

    def fixed_e_multiquadric_rbf(r):
        return multiquadric_rbf(r, e)

    def fixed_e_regularized_spline_rbf(r):
        return regularized_spline_rbf(r, e)

    def fixed_e_tension_spline_rbf(r):
        return tension_spline_rbf(r, e)

    def fixed_e_wendland_csrbf(r):
        return wendland_csrbf(r, e)

    # Use a dictionary as a substitute for a switch case construction
    switcher = {
        "linear": linear_rbf,
        "cubic": cubic_rbf,
        "quintic": quintic_rbf,
        "gaussian": fixed_e_gaussian_rbf,
        "multiquadric": fixed_e_multiquadric_rbf,
        "green": green_rbf,
        "regularized": fixed_e_regularized_spline_rbf,
        "tension": fixed_e_tension_spline_rbf,
        "thinplate": thin_plate_spline_rbf,
        "wendland": fixed_e_wendland_csrbf
    }

    return switcher[rbf_type]
