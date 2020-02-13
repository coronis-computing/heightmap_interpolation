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

import numpy as np


def green_rbf(r):
    """Green's RBF

    Green's RBF, as defined in :
    David T. Sandwell, Biharmonic spline interpolation of GEOS-3 and SEASAT altimeter data, Geophysical Research Letters, 2, 139-142, 1987.
    """

    # Singularity at 0, scalar value
    if np.isscalar(r) and r == 0:
        return 0

    fx = np.power(r, 2)*(np.log(r)-1)

    # Fix singularity of Green's function at 0
    if not np.isscalar(r):
        fx[r == 0] = 0

    return fx
