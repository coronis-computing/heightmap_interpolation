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
from py_ams_point_interpolant.ams_point_interpolant import (
    ams_point_interpolant_2d,
    get_parallel_types,
)

from heightmap_interpolation.interpolants.interpolant import Interpolant
from heightmap_interpolation.misc.conditional_print import ConditionalPrint


class AMSInterpolant(Interpolant):
    def __init__(
        self,
        x,
        y,
        z,
        depth=8,
        degree=2,
        solve_depth=-1,
        full_depth=5,
        base_depth=-1,
        boundary_type="free",
        iters=8,
        base_v_cycles=4,
        max_memory_gb=0,
        parallel_type="openmp",
        parallel_schedule="dynamic",
        parallel_thread_chunk_size=128,
        value_weight=1000.0,
        gradient_weight=100.0,
        scale=1.1,
        width=0.0,
        cg_accuracy=0.001,
        iso=0.0,
        laplacian_weight=0.0,
        bi_laplacian_weight=1.0,
        show_performance=False,
        show_residual=False,
        exact_interpolation=False,
        verbose=False,
        transform_file="",
    ):
        """Constructor"""
        # Base class constructor
        super().__init__(x, y, z)

        self.cp = ConditionalPrint(verbose)
        self.cp.print("")

        # Validate input arguments
        parallel_types = get_parallel_types()
        if parallel_type not in parallel_types:
            raise ValueError(
                f"Invalid parallel type: {parallel_type}, available: "
                + " ".join(parallel_types)
            )

        if parallel_schedule not in ["static", "dynamic"]:
            raise ValueError(
                f"Invalid parallel schedule: {parallel_schedule}, available: static, dynamic"
            )

        if boundary_type not in ["free", "dirichlet", "neumann"]:
            raise ValueError(
                f"Invalid boundary type: {boundary_type}, available: free, dirichlet, neumann"
            )

        if degree < 2 or degree > 3:
            raise ValueError(f"Invalid degree: {degree}, available: 2, 3")

        b_type = ["free", "neumann", "dirichlet"].index(boundary_type) + 1
        schedule = ["static", "dynamic"].index(parallel_schedule)
        parallel_type_id = parallel_types.index(parallel_type)

        # Create the interpolant
        points = np.asarray(np.column_stack((x, y)), dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        self.interp = ams_point_interpolant_2d(
            points,
            z,
            np.empty((0, 2), dtype=np.float64),  # Not used for the moment
            np.empty(0, dtype=np.float64),        # Not used for the moment
            depth=depth,
            degree=degree,
            solve_depth=solve_depth,
            full_depth=full_depth,
            base_depth=base_depth,
            b_type=b_type,
            iters=iters,
            base_v_cycles=base_v_cycles,
            max_memory_gb=max_memory_gb,
            parallel=parallel_type_id,
            schedule=schedule,
            thread_chunk_size=parallel_thread_chunk_size,
            value_weight=value_weight,
            gradient_weight=gradient_weight,
            scale=scale,
            width=width,
            cg_accuracy=cg_accuracy,
            iso=iso,
            lap_weight=laplacian_weight,
            bi_lap_weight=bi_laplacian_weight,
            show_performance=show_performance,
            show_residual=show_residual,
            exact_interpolation=exact_interpolation,
            verbose=verbose,
            transform_file=transform_file,
        )

    def __call__(self, x, y):
        """Evaluates the interpolant at the x, y locations"""
        eval_pts = np.column_stack((x.flatten(), y.flatten()))
        return self.interp.eval(eval_pts).reshape(x.shape)

    @staticmethod
    def preferred_scale_factor(x_ref, y_ref, x_int, y_int):
        """
        Computes the scale factor needed to cover the query domain
        given the points used to create the interpolant.

        Parameters:
            x_ref, y_ref: (N, 1) arrays with the XY coordinates of the points used to create the interpolant
            query_points: (N, 1) arrays with the XY coordinates of the points to query the interpolant at

        Returns:
            scale_factor: float
        """
        input_points = np.column_stack((x_ref, y_ref))
        query_points = np.column_stack((x_int, y_int))

        min_input = np.min(input_points, axis=0)
        max_input = np.max(input_points, axis=0)

        center = (min_input + max_input) / 2
        max_input_extent = np.max(max_input - min_input)

        max_dist = np.max(np.abs(query_points - center))

        scale_factor = (2 * max_dist) / max_input_extent

        return float(scale_factor)
