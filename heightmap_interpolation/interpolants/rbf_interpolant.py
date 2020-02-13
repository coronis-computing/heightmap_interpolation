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

from heightmap_interpolation.interpolants.interpolant import Interpolant
from heightmap_interpolation.rbf.rbf_type_to_functor import *
from heightmap_interpolation.polynomials import bivariate_polynomials
from heightmap_interpolation.interpolants.distance_type_to_functor import distance_type_to_cdist_functor
from heightmap_interpolation.interpolants.distance_type_to_functor import distance_type_to_pdist_functor
import numpy as np
import scipy.spatial


class RBFInterpolant(Interpolant):
    """ Radial Basis Function Interpolant

    Interpolates using the classical Radial Basis Function (RBF) interpolant

    Attributes:
        weights: The weights of the RBF equation
        poly: The constant polynomial part
        dist_fun: Distance functor
        rbf_fun: RBF functor
    """

    def __init__(self, x, y, z, **kwargs):
        """Constructor"""

        # Base class constructor
        super().__init__(x, y, z)

        # --- Gather and check the input parameters ---
        dist_type = kwargs.pop("distance_type", "euclidean")
        rbf_type = kwargs.pop("rbf_type", "thinplate")
        e = kwargs.pop("epsilon", 0)
        regularization = kwargs.pop("regularization", 0)
        poly_deg = kwargs.pop("polynomial_degree", 1)

        # Get the distance function from its type
        self.cdist_fun = distance_type_to_cdist_functor(dist_type)
        self.pdist_fun = distance_type_to_pdist_functor(dist_type)

        # Get the RBF from its type
        self.rbf_fun = rbf_type_to_functor(rbf_type, e)

        # --- Compute the weights and polynomial part ---
        n = self.data.shape[0]  # Number of samples

        # Compute all pair-wise distances
        # dists = scipy.spatial.distance.pdist(self.data[:, :2], self.dist_fun)
        # dists = scipy.spatial.distance.pdist(self.data[:, :2])
        dists = self.pdist_fun(self.data[:, :2])

        # Evaluate the RBF for all the distances
        rbf_evals = self.rbf_fun(dists)

        # Compose the system of equations
        # - RBF part
        A = np.zeros((n, n))
        A[np.triu_indices(n, 1)] = rbf_evals # Fill the upper triangular part of the matrix (indexing is row-wise in Python!)
        A = A+A.T  # Mirror over the diagonal (matrix A is symmetric)
        # Compute RBF values at the diagonal (radius == 0)
        A[np.diag_indices(n)] = self.rbf_fun(0.)

        # Regularization?
        if regularization != 0:
            A = A + np.eye(n)*regularization

        b = self.data[:, [2]]

        # Polynomial part
        terms = bivariate_polynomials.terms(poly_deg, self.data[:, [0]], self.data[:, [1]])
        num_terms = terms.shape[1]
        A = np.append(A, terms, 1)
        termsa = np.hstack((terms.T, np.zeros((num_terms, num_terms))))
        A = np.append(A, termsa, 0)

        b = np.vstack((b, np.zeros((num_terms, 1))))

        # Solve the system of equations
        solution = np.linalg.solve(A, b)

        # Recover the results
        self.weights = solution[:n]
        self.poly = solution[n:]

    def __call__(self, x, y):
        # Check sizes
        if x.size != y.size:
            raise ValueError("x and y should have the same number of elements")

        if x.shape != y.shape:
            print("[WARNING] x.shape != y.shape. The size of the output matrix will be that of x")

        # Flatten the data into a column vector
        original_shape = x.shape
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Compute all pair-wise distances
        # dists = scipy.spatial.distance.cdist(np.hstack((x, y)), self.data[:, :2], self.dist_fun)
        # dists = scipy.spatial.distance.cdist(np.hstack((x, y)), self.data[:, :2])
        dists = self.cdist_fun(np.hstack((x, y)), self.data[:, :2])

        # Evaluate the RBF for all the distances
        A = self.rbf_fun(dists)

        # Evaluate the polynomials
        poly_eval = bivariate_polynomials.eval(self.poly, x, y)

        # Compute the evaluation
        z = A@self.weights + poly_eval

        # Return z with the same shape as input x
        z = np.reshape(z, original_shape)

        return z
