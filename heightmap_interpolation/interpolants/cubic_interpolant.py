# Copyright (c) 2021 Coronis Computing S.L. (Spain)
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

from heightmap_interpolation.interpolants.interpolant import Interpolant
from scipy.interpolate import CloughTocher2DInterpolator
import numpy as np


class CubicInterpolant(Interpolant):

    def __init__(self, x, y, z, fill_value=np.nan, tol=1e-06, maxiter=400, rescale=False):
        """ Constructor """

        # Base class constructor
        super().__init__(x, y, z)

        # Create the interpolant
        self.interpolant = CloughTocher2DInterpolator(list(zip(x, y)), z,
                                                      fill_value=fill_value, tol=tol, maxiter=maxiter, rescale=rescale)

    def __call__(self, x, y):
        """Evaluates the interpolant at the x, y locations"""
        return self.interpolant(x, y)