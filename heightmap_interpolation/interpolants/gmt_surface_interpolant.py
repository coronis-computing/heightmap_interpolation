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

from heightmap_interpolation.interpolants.interpolant import Interpolant


class GMTSurfaceInterpolant(Interpolant):
    """Interpolant wrapping GMT's `surface` spline-in-tension gridder.

    GMT `surface` is a gridder (scattered XYZ -> regular grid), not a point
    evaluator. To fit the Interpolant interface, this class runs `surface` over
    the bounding box of the query points at the requested grid spacing, and then
    samples the resulting grid at the query locations. In this project the query
    points originate from the same regular grid used as `spacing`, so
    nearest-cell sampling is effectively exact.

    Requires the `pygmt` package and the GMT binaries installed on the system.
    """

    def __init__(
        self,
        x,
        y,
        z,
        spacing,
        tension=0.0,
        convergence_limit=0.0,
        max_radius=None,
        max_iterations=None,
        region=None,
        verbose=False,
    ):
        """Constructor

        Args:
            x, y, z: Coordinates of the known points.
            spacing: Grid spacing (GMT -I), e.g. the output raster cell size.
            tension: Tension factor in [0, 1] (GMT -T). 0 = minimum curvature.
            convergence_limit: Convergence limit (GMT -C). 0 = GMT default.
            max_radius: Search radius for nearest-data init (GMT -M), e.g. "5c".
            max_iterations: Maximum number of iterations (GMT -N). None = default.
            region: [xmin, xmax, ymin, ymax] grid region (GMT -R). If None, it is
                computed from the query points at evaluation time.
            verbose: If True, let GMT print progress information.
        """
        # Base class constructor (validates sizes and removes duplicates into self.data)
        super().__init__(x, y, z)

        # Deferred import so that importing this module never fails when GMT is absent.
        try:
            import pygmt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'gmt_surface' interpolant requires the 'pygmt' package and the GMT"
                " binaries. Install them, e.g. via:"
                " 'conda install -c conda-forge gmt pygmt'"
                " (or 'pip install heightmap_interpolation[experimental]' plus the GMT"
                " binaries)."
            ) from exc

        self.spacing = spacing
        self.tension = tension
        self.convergence_limit = convergence_limit
        self.max_radius = max_radius
        self.max_iterations = max_iterations
        self.region = region
        self.verbose = verbose

    def __call__(self, x, y):
        """Evaluates the interpolant at the x, y locations"""
        import pygmt
        import xarray as xr

        xq = np.asarray(x).flatten()
        yq = np.asarray(y).flatten()

        region = self.region
        if region is None:
            region = [
                float(np.min(xq)),
                float(np.max(xq)),
                float(np.min(yq)),
                float(np.max(yq)),
            ]

        # Compose the optional GMT arguments
        kwargs = {
            "x": self.data[:, 0],
            "y": self.data[:, 1],
            "z": self.data[:, 2],
            "region": region,
            "spacing": self.spacing,
            "tension": self.tension,
            "verbose": "i" if self.verbose else "e",
        }
        if self.convergence_limit:
            kwargs["convergence"] = self.convergence_limit
        if self.max_radius is not None:
            kwargs["maxradius"] = self.max_radius
        if self.max_iterations is not None:
            kwargs["N"] = str(self.max_iterations)

        grid = pygmt.surface(**kwargs)

        # Sample the resulting grid at the query points (pointwise nearest-cell lookup).
        ydim, xdim = grid.dims[0], grid.dims[1]
        sampled = grid.interp(
            {
                xdim: xr.DataArray(xq, dims="points"),
                ydim: xr.DataArray(yq, dims="points"),
            },
            method="nearest",
        )

        return np.reshape(sampled.values, np.asarray(x).shape)
