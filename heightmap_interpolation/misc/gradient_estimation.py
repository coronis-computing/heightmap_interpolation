# Copyright (c) 2026 Coronis Computing S.L. (Spain)
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
from scipy.spatial import cKDTree


def estimate_scattered_gradients(
    x,
    y,
    z,
    num_neighbors=8,
    max_distance=None,
    min_planarity=0.7,
    max_condition=1e6,
):
    """Estimates per-point gradients of scattered (x, y, z) data via a local
    least-squares plane fit, keeping only the reliable ones.

    For each input point, a plane z ~= a + b*dx + c*dy is fitted (in the least
    squares sense) to its ``num_neighbors`` nearest neighbors, where (dx, dy) are
    the neighbor offsets from the point. The estimated gradient at the point is
    then (b, c). A gradient is only kept (considered reliable) when all of the
    following hold:

    - Enough close neighbors: the distance to the furthest of the requested
      neighbors is <= ``max_distance`` (so the estimate is local).
    - Non-degenerate geometry: the local fit is well conditioned (neighbors are
      not nearly collinear), i.e. its condition number is <= ``max_condition``.
    - Good fit residual: the plane explains the local variation well, i.e. the
      coefficient of determination R^2 is >= ``min_planarity``.

    Args:
        x, y, z: Coordinates and values of the known points (any shape, flattened
            internally).
        num_neighbors: Number of nearest neighbors used for each local fit.
        max_distance: Maximum distance (coordinate units) to the furthest used
            neighbor for a point to be considered reliable. If None, it is derived
            automatically as twice the median furthest-neighbor distance.
        min_planarity: Minimum R^2 of the local plane fit to keep the gradient,
            in [0, 1]. Set to 0 to disable the residual filter.
        max_condition: Maximum condition number of the local normal matrix to keep
            the gradient (geometry/degeneracy filter).

    Returns:
        positions_gradients: (M, 2) float64 array with the XY coordinates of the
            points with a reliable gradient.
        gradients: (M, 2) float64 array with the (dz/dx, dz/dy) gradients at those
            points. Both arrays are empty ((0, 2)) when no reliable gradient is found.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()

    pts = np.column_stack((x, y))
    n = pts.shape[0]

    # A plane fit needs at least 3 points; cap the requested neighbors to what is
    # available (query includes the point itself, hence the +1).
    k = min(num_neighbors + 1, n)
    if n < 3 or k < 3:
        return np.empty((0, 2)), np.empty((0, 2))

    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=k, workers=-1)  # (n, k) each

    # Local design tensor A = [1, dx, dy] for every point's neighborhood
    nbr_x = x[idx]  # (n, k)
    nbr_y = y[idx]
    nbr_z = z[idx]
    dx = nbr_x - x[:, None]
    dy = nbr_y - y[:, None]
    A = np.stack((np.ones_like(dx), dx, dy), axis=2)  # (n, k, 3)

    # Batched normal equations: (A^T A) coeffs = A^T z, solved per point
    ata = np.einsum("nki,nkj->nij", A, A)  # (n, 3, 3)
    atz = np.einsum("nki,nk->ni", A, nbr_z)  # (n, 3)

    # Geometry filter: drop ill-conditioned (near-collinear) neighborhoods. These
    # would also make np.linalg.solve fail, so solve only the valid ones.
    cond = np.linalg.cond(ata)
    well_conditioned = np.isfinite(cond) & (cond <= max_condition)

    coeffs = np.zeros((n, 3))
    if np.any(well_conditioned):
        coeffs[well_conditioned] = np.linalg.solve(
            ata[well_conditioned], atz[well_conditioned][..., None]
        )[..., 0]
    grads = coeffs[:, 1:]  # (n, 2) -> (dz/dx, dz/dy)

    # Locality filter: the furthest requested neighbor must be close enough
    furthest = dist[:, -1]
    if max_distance is None:
        max_distance = 2.0 * np.median(furthest)
    close_enough = furthest <= max_distance

    # Residual filter: coefficient of determination of the local plane fit. A flat
    # neighborhood (ss_tot ~ 0) is perfectly planar, so it is kept.
    z_pred = np.einsum("nki,ni->nk", A, coeffs)
    ss_res = np.sum((nbr_z - z_pred) ** 2, axis=1)
    z_mean = np.mean(nbr_z, axis=1, keepdims=True)
    ss_tot = np.sum((nbr_z - z_mean) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 1.0)
    planar_enough = r2 >= min_planarity

    keep = well_conditioned & close_enough & planar_enough
    return pts[keep], grads[keep]
