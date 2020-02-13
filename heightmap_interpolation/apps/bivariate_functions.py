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


def franke(x, y):
    """ Franke's bivariate function

    Franke's bivariate function from:
    Franke, R. (1979). A critical comparison of some methods for interpolation of scattered data (No. NPS53-79-003). NAVAL POSTGRADUATE SCHOOL MONTEREY CA.
    """

    term1 = 0.75 * np.exp(-np.power(9 * x - 2, 2) / 4 - np.power(9 * y - 2, 2) / 4)
    term2 = 0.75 * np.exp(-np.power(9 * x + 1, 2) / 49 - (9 * y + 1) / 10)
    term3 = 0.5 * np.exp(-np.power(9 * x - 7, 2) / 4 - np.power(9 * y - 3, 2) / 4)
    term4 = -0.2 * np.exp(-np.power(9 * x - 4, 2) - np.power(9 * y - 7, 2))

    return term1 + term2 + term3 + term4


def flower(x, y):
    """Flower-shaped function

    Flower-shaped function found in the following example from scipy docs:
    https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolategriddata/
    """
    s = np.hypot(x, y)
    phi = np.arctan2(y, x)
    tau = s + s*(1-s)/5 * np.sin(6*phi)
    return 5*(1-tau) + tau
