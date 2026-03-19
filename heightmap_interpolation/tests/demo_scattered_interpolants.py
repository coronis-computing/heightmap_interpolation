#!/usr/bin/env python3

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

"""Demo script comparing all scattered data interpolation methods available in
the package on a synthetic bivariate function sampled at random points."""

import argparse
import math
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from heightmap_interpolation.apps.apps_common import (
    experimental_features_available,
    get_available_scattered_methods,
)
from heightmap_interpolation.interpolants.ams_interpolant import AMSInterpolant
from heightmap_interpolation.interpolants.cubic_interpolant import CubicInterpolant
from heightmap_interpolation.interpolants.linear_interpolant import LinearInterpolant
from heightmap_interpolation.interpolants.nearest_neighbor_interpolant import (
    NearestNeighborInterpolant,
)
from heightmap_interpolation.interpolants.quad_tree_pu_rbf_interpolant import (
    QuadTreePURBFInterpolant,
)
from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant
from heightmap_interpolation.tests.bivariate_functions import FUNCTIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare all scattered data interpolation methods on a synthetic function."
    )
    parser.add_argument(
        "--function",
        default="sin_cos",
        choices=list(FUNCTIONS.keys()),
        metavar="NAME",
        help="Ground-truth function (default: sin_cos). Use --list-functions to see all.",
    )
    parser.add_argument(
        "--list-functions",
        action="store_true",
        help="Print available functions and exit.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=200,
        help="Evaluation grid resolution (default: 200)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of scattered samples (default: 500)",
    )
    parser.add_argument(
        "--rbf_type",
        default="thinplate",
        help="RBF kernel type for the RBF and PURBF methods (default: thinplate)",
    )
    parser.add_argument(
        "--rbf_epsilon",
        type=float,
        default=0.5,
        help="Epsilon for RBF kernels that require it (default: 0.5)",
    )
    parser.add_argument(
        "--rbf_polynomial_degree",
        type=int,
        default=1,
        help="Polynomial degree for RBF/PURBF (default: 1)",
    )
    return parser.parse_args()


def build_interpolant(method, x, y, z, xi, yi, args):
    """Instantiate and apply the given interpolant. Returns (zi, elapsed_sec)."""
    ts = timer()
    if method == "nearest":
        interp = NearestNeighborInterpolant(x, y, z)
    elif method == "linear":
        interp = LinearInterpolant(x, y, z)
    elif method == "cubic":
        interp = CubicInterpolant(x, y, z)
    elif method == "rbf":
        interp = RBFInterpolant(
            x,
            y,
            z,
            rbf_type=args.rbf_type,
            polynomial_degree=args.rbf_polynomial_degree,
            epsilon=args.rbf_epsilon,
        )
    elif method == "purbf":
        w = xi.max() - xi.min()
        h = yi.max() - yi.min()
        domain = [xi.min(), yi.min(), max(w, h)]
        interp = QuadTreePURBFInterpolant(
            x,
            y,
            z,
            domain=domain,
            rbf_type=args.rbf_type,
            polynomial_degree=args.rbf_polynomial_degree,
            epsilon=args.rbf_epsilon,
        )
    elif method == "mlp":
        from heightmap_interpolation.interpolants.mlp_interpolant import MLPInterpolant

        interp = MLPInterpolant(x, y, z)
    elif method == "ams":
        suggested_scale = AMSInterpolant.preferred_scale_factor(
            x, y, xi.ravel(), yi.ravel()
        )
        interp = AMSInterpolant(x, y, z, scale=suggested_scale)
    else:
        raise ValueError(f"Unknown method: {method}")

    zi = interp(xi, yi)
    elapsed = timer() - ts
    return zi, elapsed


def main():
    args = parse_args()

    if args.list_functions:
        print("\nAvailable 2-D ground-truth functions:\n")
        max_len = max(len(k) for k in FUNCTIONS)
        for name, meta in FUNCTIONS.items():
            d = meta["domain"]
            print(f"  {name:<{max_len}}   {meta['description']}")
            print(
                f"  {'':<{max_len}}   domain: x∈[{d[0]:.2g},{d[1]:.2g}]"
                f"  y∈[{d[2]:.2g},{d[3]:.2g}]\n"
            )
        return 0

    # Sample function
    entry = FUNCTIONS[args.function]
    f = entry["fn"]
    xmin, xmax, ymin, ymax = entry["domain"]
    extent = [xmin, xmax, ymin, ymax]

    rng = np.random.default_rng(seed=42)
    x = rng.uniform(xmin, xmax, args.num_samples)
    y = rng.uniform(ymin, ymax, args.num_samples)
    z = f(x, y)

    # Evaluation grid
    xg = np.linspace(xmin, xmax, args.grid_size)
    yg = np.linspace(ymin, ymax, args.grid_size)
    xi, yi = np.meshgrid(xg, yg)
    gt_z = f(xi, yi)

    methods = get_available_scattered_methods()

    # Grid layout: 4 cols; GT first, then one cell per method
    sp_cols = 4
    sp_total = 1 + len(methods)
    sp_rows = math.ceil(sp_total / sp_cols)
    cell_size = 4  # inches per subplot

    def make_figure(title, gt_colorbar=False):
        fig, axes_all = plt.subplots(
            sp_rows,
            sp_cols,
            figsize=(sp_cols * cell_size, sp_rows * cell_size),
            layout="constrained",
        )
        fig.suptitle(title)
        axes_flat = axes_all.flatten()
        # GT panel
        ax_gt = axes_flat[0]
        im_gt = ax_gt.imshow(gt_z, extent=extent, origin="lower")
        if gt_colorbar:
            fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        ax_gt.scatter(x, y, s=3, alpha=0.6, color="black")
        ax_gt.set_title("Sample points on f(x,y)")
        # Hide trailing empty cells
        for ax in axes_flat[sp_total:]:
            ax.axis("off")
        return fig, list(axes_flat[1:sp_total])

    fig_interp, axes_interp = make_figure("Scattered interpolation methods")
    fig_err, axes_err = make_figure("Absolute errors", gt_colorbar=True)

    print(f"Function : {args.function}  ({entry['description']})")
    print(f"Samples  : {args.num_samples}  |  grid: {args.grid_size}×{args.grid_size}")
    print(f"RBF type : {args.rbf_type}  (used by 'rbf' and 'purbf')\n")

    for idx, method in enumerate(methods):
        print(f"  [{idx + 1}/{len(methods)}] {method}...", end=" ", flush=True)
        try:
            zi, elapsed = build_interpolant(method, x, y, z, xi, yi, args)
            err = np.abs(zi - gt_z)
            mean_err = np.nanmean(err)
            max_err = np.nanmax(err)
            print(f"{elapsed:.2f}s  |  mean_err={mean_err:.3g}  max_err={max_err:.3g}")

            axes_interp[idx].imshow(zi, extent=extent, origin="lower")
            axes_interp[idx].set_title(method)

            im = axes_err[idx].imshow(err, extent=extent, origin="lower")
            fig_err.colorbar(im, ax=axes_err[idx], fraction=0.046, pad=0.04)
            axes_err[idx].set_title(
                f"{method}\n(mean={mean_err:.3g}, max={max_err:.3g})"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            axes_interp[idx].set_title(f"{method}\n(failed)")
            axes_interp[idx].axis("off")
            axes_err[idx].set_title(f"{method}\n(failed)")
            axes_err[idx].axis("off")

    print("\nClose the plot windows to exit.")
    plt.show(block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
