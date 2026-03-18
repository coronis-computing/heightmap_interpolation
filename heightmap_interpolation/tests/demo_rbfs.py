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

import argparse
import math
import sys
from heightmap_interpolation.tests.bivariate_functions import FUNCTIONS
import matplotlib.pyplot as plt
import numpy as np

from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant


def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(
        description="Demo function showing the bivariate function interpolation using RBFs"
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
        default=1000,
        help="Number of scattered samples (default: 1000)",
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="plot_3d",
        default=False,
        help="Set this flag to plot the results in 3D",
    )
    parser.add_argument(
        "--rbf_polynomial_degree",
        type=int,
        default=1,
        help="Polynomial degree for such term in the RBF interpolation equation (default: 1)",
    )
    parser.add_argument(
        "--rbf_epsilon",
        type=float,
        default=0.5,
        help="Default epsilon value for RBFs that require it (default: 0.5)",
    )
    parser.add_argument(
        "--rbf_epsilon_overrides",
        nargs="*",
        metavar="TYPE=VALUE",
        default=[],
        help=(
            "Per-type epsilon overrides for RBFs that use it, "
            "e.g. gaussian=1.0 multiquadric=2.0. "
            "Valid types: gaussian, multiquadric, regularized, tension, wendland. "
            "Overrides --rbf_epsilon for the specified types."
        ),
    )
    parser.add_argument(
        "--rbf_regularization",
        type=float,
        default=0.0,
        help="Regularization value for the RBF interpolation equation (default: 0.0)",
    )
    return parser.parse_args()


# Main function
def main():
    args = parse_args()

    # List available functions
    if args.list_functions:
        print("\nAvailable 2-D ground-truth functions:\n")
        max_len = max(len(k) for k in FUNCTIONS)
        for name, meta in FUNCTIONS.items():
            d = meta["domain"]
            print(f"  {name:<{max_len}}   {meta['description']}")
            print(
                f"  {'':<{max_len}}   domain: x∈[{d[0]:.2g},{d[1]:.2g}]  "
                f"y∈[{d[2]:.2g},{d[3]:.2g}]\n"
            )
        return 0

    # Get the sample function
    entry = FUNCTIONS[args.function]
    f = entry["fn"]
    xmin, xmax, ymin, ymax = entry["domain"]
    extent = [xmin, xmax, ymin, ymax]

    x = np.random.uniform(xmin, xmax, args.num_samples)
    y = np.random.uniform(ymin, ymax, args.num_samples)
    z = f(x, y)

    # Compute a regular grid, where all data points will be interpolated according to the RBF interpolator
    xg = np.linspace(xmin, xmax, args.grid_size)
    yg = np.linspace(ymin, ymax, args.grid_size)
    xi, yi = np.meshgrid(xg, yg)
    gt_z = f(xi, yi)  # Ground-truth z

    # Draw the initial data
    rbf_types = (
        "linear",
        "cubic",
        "quintic",
        "thinplate",
        "green",
        "multiquadric",
        "tension",
        "regularized",
        "gaussian",
        "wendland",
    )
    num_rbf = len(rbf_types)

    # Grid layout: 4 cols; GT is first cell, RBFs follow in the same grid
    sp_cols = 4
    sp_total = 1 + num_rbf  # GT + all RBFs
    sp_rows = math.ceil(sp_total / sp_cols)
    projection_type = "3d" if args.plot_3d else "rectilinear"
    cell_size = 4  # inches per subplot cell

    def make_figure(title, gt_colorbar=False):
        fig, axes_all = plt.subplots(
            sp_rows,
            sp_cols,
            figsize=(sp_cols * cell_size, sp_rows * cell_size),
            layout="constrained",
            subplot_kw={"projection": projection_type},
        )
        fig.suptitle(title)
        axes_flat = axes_all.flatten()
        # GT in first cell
        ax_gt = axes_flat[0]
        if args.plot_3d:
            ax_gt.plot_surface(xi, yi, gt_z)
            ax_gt.scatter(x, y, z, alpha=0.7, marker=".", color="black")
        else:
            im_gt = ax_gt.imshow(gt_z, extent=extent, origin="lower")
            if gt_colorbar:
                fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            ax_gt.scatter(x, y, alpha=0.7, marker=".", color="black")
        ax_gt.set_title("Sample points on f(x,y)")
        # Hide trailing empty cells
        for ax in axes_flat[sp_total:]:
            ax.axis("off")
        axes_data = list(axes_flat[1:sp_total])
        return fig, axes_data

    fig_interp, axes_interp = make_figure("RBF interpolations")
    fig_err, axes_err = make_figure("Absolute errors", gt_colorbar=True)

    distance_type = "euclidean"
    polynomial_degree = args.rbf_polynomial_degree
    default_epsilon = args.rbf_epsilon
    epsilon_rbf_types = {
        "gaussian", "multiquadric", "regularized", "tension", "wendland"
    }
    epsilon_overrides = {}
    for item in args.rbf_epsilon_overrides:
        if "=" not in item:
            raise ValueError(
                f"--rbf_epsilon_overrides: expected TYPE=VALUE, got '{item}'"
            )
        rbf_name, val = item.split("=", 1)
        if rbf_name not in epsilon_rbf_types:
            raise ValueError(
                f"--rbf_epsilon_overrides: '{rbf_name}' does not use epsilon. "
                f"Valid types: {', '.join(sorted(epsilon_rbf_types))}"
            )
        epsilon_overrides[rbf_name] = float(val)
    regularization = args.rbf_regularization
    print("Using:")
    print("  - distance_type = " + distance_type)
    print("  - polynomial_degree = " + str(polynomial_degree))
    print("  - epsilon (default) = " + str(default_epsilon))
    if epsilon_overrides:
        print("  - epsilon overrides = " + str(epsilon_overrides))
    print("  - regularization = " + str(regularization))
    print("- Computing:")
    for idx, rbf_type in enumerate(rbf_types):
        epsilon = epsilon_overrides.get(rbf_type, default_epsilon)
        if rbf_type in epsilon_rbf_types:
            print("  - " + rbf_type + f" RBF interpolation (epsilon={epsilon})")
        else:
            print("  - " + rbf_type + " RBF interpolation")

        interpolant = RBFInterpolant(
            x,
            y,
            z,
            rbf_type=rbf_type,
            distance_type=distance_type,
            polynomial_degree=polynomial_degree,
            epsilon=epsilon,
            regularization=regularization,
        )

        zi = interpolant(xi, yi)
        err = np.abs(zi - gt_z)
        mean_err = np.mean(err)
        max_err = np.max(err)

        ax_i = axes_interp[idx]
        if args.plot_3d:
            ax_i.plot_surface(xi, yi, zi)
        else:
            ax_i.imshow(zi, extent=extent, origin="lower")
        ax_i.set_title(rbf_type + " RBF")

        ax_e = axes_err[idx]
        if args.plot_3d:
            ax_e.plot_surface(xi, yi, err)
        else:
            im = ax_e.imshow(err, extent=extent, origin="lower")
            fig_err.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)
        ax_e.set_title(f"{rbf_type}\n(mean={mean_err:.3g}, max={max_err:.3g})")

    print("- Showing the results, close the emerging window to finish.")

    plt.show()


if __name__ == "__main__":
    sys.exit(main())
