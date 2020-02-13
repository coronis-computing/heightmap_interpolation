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


def terms(degree: int, x, y):
    """Computes the terms of a bivariate polynomial function of a given degree"""

    # Check the input parameters
    if x.ndim < 2 or x.shape[1] != 1:
        raise ValueError("x should be a column vector.")

    if y.ndim < 2 or y.shape[1] != 1:
        raise ValueError("y should be a column vector.")

    if x.size != y.size:
        raise ValueError("x and y should have the same size.")

    if degree < 0:
        raise ValueError("Polynomial degree should be greater than zero.")

    if degree > 3:
        raise ValueError("Polynomial degree not implemented.")

    # Deal with each individual case
    if degree == 0:
        terms = np.ones(x.shape)  # Constant
    elif degree == 1:
        terms = np.hstack((x, y, np.ones(x.shape)))  # Linear
    elif degree == 2:
        terms = np.hstack((x*x, y*y, x*y, x, y, np.ones(x.shape)))  # Quadratic
    elif degree == 3:
        terms = np.hstack((x*x*x, y*y*y, x*x*y, x*y*y, x*x, y*y, x*y, x, y, np.ones(x.shape)))  # Cubic

    return terms


def eval(coeffs, x, y):
    # Check the input parameters
    if x.ndim < 2 or x.shape[1] != 1:
        raise ValueError("x should be a column vector.")

    if y.ndim < 2 or y.shape[1] != 1:
        raise ValueError("y should be a column vector.")

    if x.size != y.size:
        raise ValueError("x and y should have the same size.")

    # Deal with special case where the coeffs is empty
    if coeffs.size == 0:
        return np.zeros(x.shape)

    # Infer the degree of the polynomial from the number of coefficients
    switcher = {
        1: 0,   # Constant
        3: 1,   # Linear
        6: 2,   # Quadratic
        10: 3,  # Cubic
    }
    degree = switcher[coeffs.size]

    # Get the terms of this polynomial
    t = terms(degree, x, y)

    # Evaluate using the coefficients
    mul = t.dot(coeffs)
    return mul
