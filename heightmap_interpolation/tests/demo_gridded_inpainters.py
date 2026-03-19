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

"""Demo script comparing all gridded inpainting methods available in the
package. Holes (continuous missing regions) are punched into a synthetic
ground-truth grid, and each inpainter attempts to recover the original values."""

import argparse
import math
import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from heightmap_interpolation.apps.apps_common import get_available_gridded_methods
from heightmap_interpolation.inpainting.amle_inpainter import AMLEInpainter
from heightmap_interpolation.inpainting.ccst_inpainter import CCSTInpainter
from heightmap_interpolation.inpainting.opencv_inpainter import (
    OpenCVInpainter,
    OpenCVXPhotoInpainter,
)
from heightmap_interpolation.inpainting.sobolev_inpainter import SobolevInpainter
from heightmap_interpolation.inpainting.tv_inpainter import TVInpainter
from heightmap_interpolation.tests.bivariate_functions import FUNCTIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare all gridded inpainting methods on a synthetic function with holes."
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
        help="Grid resolution (default: 200)",
    )
    parser.add_argument(
        "--num_holes",
        type=int,
        default=5,
        help="Number of disk-shaped holes to punch into the grid (default: 5)",
    )
    parser.add_argument(
        "--hole_radius",
        type=float,
        default=0.15,
        help="Radius of each hole as a fraction of the domain's shorter side (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for hole placement (default: 42)",
    )
    return parser.parse_args()


def make_holes(gt_z, xs_mat, ys_mat, num_holes, hole_radius_frac, rng):
    """Return (f_with_holes, mask) where mask is True at missing cells."""
    xmin, xmax = xs_mat.min(), xs_mat.max()
    ymin, ymax = ys_mat.min(), ys_mat.max()
    radius = hole_radius_frac * min(xmax - xmin, ymax - ymin)

    mask = np.zeros(gt_z.shape, dtype=bool)
    cx = rng.uniform(xmin + radius, xmax - radius, num_holes)
    cy = rng.uniform(ymin + radius, ymax - radius, num_holes)
    for x0, y0 in zip(cx, cy):
        dist = np.sqrt((xs_mat - x0) ** 2 + (ys_mat - y0) ** 2)
        mask |= dist <= radius

    f_with_holes = gt_z.copy().astype(np.float32)
    f_with_holes[mask] = np.nan
    return f_with_holes, mask


def build_inpainter(method):
    """Instantiate the inpainter for the given method name with default params.

    All inpainters in this package share the same mask convention:
    True = known pixel, False = unknown pixel to inpaint.
    """
    m = method.lower()
    # Shared options for iterative PDE solvers: reasonable cap for a demo
    iterative_opts = dict(term_thres=1e-4, max_iters=50_000, term_check_iters=100)
    if m == "harmonic":
        return SobolevInpainter(use_direct_solver=True)
    elif m == "tv":
        return TVInpainter(**iterative_opts)
    elif m.startswith("ccst"):
        return CCSTInpainter(use_direct_solver=True)
    elif m == "amle":
        return AMLEInpainter(**iterative_opts)
    elif m == "navier-stokes":
        return OpenCVInpainter(method="navier-stokes")
    elif m == "telea":
        return OpenCVInpainter(method="telea")
    elif m == "shiftmap":
        return OpenCVXPhotoInpainter(method="shiftmap")
    elif m == "ebi":
        from heightmap_interpolation.inpainting.exemplar_based_inpainter import (
            ExemplarBasedInpainter,
        )
        return ExemplarBasedInpainter()
    else:
        raise ValueError(f"Unknown gridded method: {method}")


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

    entry = FUNCTIONS[args.function]
    f = entry["fn"]
    xmin, xmax, ymin, ymax = entry["domain"]
    extent = [xmin, xmax, ymin, ymax]

    xg = np.linspace(xmin, xmax, args.grid_size)
    yg = np.linspace(ymin, ymax, args.grid_size)
    xs_mat, ys_mat = np.meshgrid(xg, yg)
    gt_z = f(xs_mat, ys_mat).astype(np.float32)

    rng = np.random.default_rng(seed=args.seed)
    f_with_holes, mask = make_holes(
        gt_z, xs_mat, ys_mat, args.num_holes, args.hole_radius, rng
    )
    hole_fraction = mask.mean() * 100

    methods = get_available_gridded_methods()

    sp_cols = 4
    sp_total = 1 + len(methods)
    sp_rows = math.ceil(sp_total / sp_cols)
    cell_size = 4  # inches per subplot

    vmin, vmax = np.nanmin(gt_z), np.nanmax(gt_z)

    def make_figure(title, gt_colorbar=False):
        fig, axes_all = plt.subplots(
            sp_rows,
            sp_cols,
            figsize=(sp_cols * cell_size, sp_rows * cell_size),
            layout="constrained",
        )
        fig.suptitle(title)
        axes_flat = axes_all.flatten()
        # GT + holes panel
        ax_gt = axes_flat[0]
        im_gt = ax_gt.imshow(
            f_with_holes,
            extent=extent,
            origin="lower",
            cmap="terrain",
            vmin=vmin,
            vmax=vmax,
        )
        if gt_colorbar:
            fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        ax_gt.set_title(f"Input ({hole_fraction:.1f}% missing)")
        # Hide trailing empty cells
        for ax in axes_flat[sp_total:]:
            ax.axis("off")
        return fig, list(axes_flat[1:sp_total])

    fig_inpaint, axes_inpaint = make_figure("Gridded inpainting methods")
    fig_err, axes_err = make_figure("Absolute errors", gt_colorbar=True)

    print(f"Function  : {args.function}  ({entry['description']})")
    print(
        f"Grid      : {args.grid_size}×{args.grid_size}  |  holes: {args.num_holes} × r={args.hole_radius:.2f} ({hole_fraction:.1f}% missing)\n"
    )

    for idx, method in enumerate(methods):
        print(f"  [{idx + 1}/{len(methods)}] {method}...", end=" ", flush=True)
        try:
            inpainter = build_inpainter(method)
            ts = timer()
            # All inpainters expect True = known, False = unknown.
            # Replace NaNs with 0 so OpenCV-based inpainters (which use the
            # mask, not NaN-detection) get a numerically valid input image.
            f_input = f_with_holes.copy()
            f_input[mask] = 0.0
            result = inpainter.inpaint(f_input, ~mask)
            elapsed = timer() - ts

            err = np.abs(result - gt_z)
            mean_err = np.nanmean(err[mask])
            max_err = np.nanmax(err[mask])
            print(f"{elapsed:.2f}s  |  mean_err={mean_err:.3g}  max_err={max_err:.3g}")

            axes_inpaint[idx].imshow(
                result,
                extent=extent,
                origin="lower",
                cmap="terrain",
                vmin=vmin,
                vmax=vmax,
            )
            axes_inpaint[idx].set_title(method)

            im = axes_err[idx].imshow(
                err,
                extent=extent,
                origin="lower",
                cmap="hot_r",
            )
            fig_err.colorbar(im, ax=axes_err[idx], fraction=0.046, pad=0.04)
            axes_err[idx].set_title(
                f"{method}\n(mean={mean_err:.3g}, max={max_err:.3g})"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            axes_inpaint[idx].set_title(f"{method}\n(failed)")
            axes_inpaint[idx].axis("off")
            axes_err[idx].set_title(f"{method}\n(failed)")
            axes_err[idx].axis("off")

    print("\nClose the plot windows to exit.")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
