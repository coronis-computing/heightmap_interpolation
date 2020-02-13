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


def wendland_csrbf(r, e):
    """Wendland's Compactly-Supported RBF"""

    if np.isscalar(r):
        return (max(1 - r / e, 0)**4) * (1 + 4 * r / e)

    original_shape = r.shape
    rf = r.reshape(-1, 1)
    a = np.hstack(((1-rf/e), np.zeros(rf.shape)))
    a = np.amax(a, axis=1)
    a = a.reshape(-1, 1)
    fx = np.power(a, 4)*(1+4*rf/e)

    return fx.reshape(original_shape)
