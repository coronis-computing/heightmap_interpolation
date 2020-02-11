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

import numpy as np


def thin_plate_spline_rbf(r):
    """Thin Plate Spline RBF"""

    if np.isscalar(r) and r == 0:
        return 0

    fx = np.zeros(r.shape)
    ind = r != 0 # Non-zero indices
    fx[ind] = np.power(r[ind], 2) * np.log(r[ind])

    if not np.isscalar(r):
        fx[~ind] = 0

    return fx
