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
from abc import ABC, abstractmethod


class Interpolant(ABC):
    """Abstract base class for Interpolant(s)

    Attributes:
        data: The known x, y, z values of the function to interpolate (size numPts x 3)
    """

    def __init__(self, x, y, z):
        """Constructor

        Args:
            x, y, z: Coordinates of the known points
        """

        # Check sizes
        if x.ndim != y.ndim or x.ndim != z.ndim:
            raise ValueError("x, y and z should have the same number of dimensions.")
        if x.ndim > 2 or y.ndim > 2 or z.ndim > 2:
            raise ValueError("x, y and z should have, at most, 2 dimensions.")
        if x.size != y.size or x.size != z.size:
            raise ValueError("x, y and z should have the same number of elements.")

        # Flatten the input arrays, and create a numPts x 3 matrix
        self.data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

        # Ensure there are no repetitions in the input data
        self.data = np.unique(self.data, axis=0)

    # This way of defining Abstract classes requires Python 3.4+
    @abstractmethod
    def __call__(self, x, y):
        """Interpolator function to be implemented by each individual interpolant

        Args:
            x, y: Coordinates of the points to interpolate

        Returns:
            Interpolated values at (x, y)
        """
        pass
