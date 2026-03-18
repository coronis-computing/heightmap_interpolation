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

# ---------------------------------------------------------------------------
# Ground-truth function registry
# Each entry stores:
#   fn(x, y)    - scalar field values
#   grad(x, y)  - (dfdx, dfdy) as a tuple of arrays
#   description - human-readable string
#   domain      - (x_lo, x_hi, y_lo, y_hi)
#
# Decorated functions must return (value, (dfdx, dfdy)).
# ---------------------------------------------------------------------------

FUNCTIONS = {}


def register(name, description, domain=(-np.pi, np.pi, -np.pi, np.pi)):
    def decorator(fn):
        FUNCTIONS[name] = {
            "fn": lambda x, y: fn(x, y)[0],
            "grad": lambda x, y: fn(x, y)[1],
            "description": description,
            "domain": domain,
        }
        return fn

    return decorator


@register("sin_cos", "sin(x)*cos(y)  —  smooth, periodic baseline")
def f_sin_cos(x, y):
    v = np.sin(x) * np.cos(y)
    return v, (np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y))


@register(
    "ripple_gaussian", "Radial ripple with a Gaussian hill offset from the origin"
)
def f_ripple_gaussian(x, y):
    eps = 1e-12
    r = np.sqrt(x**2 + y**2)
    rip = np.sin(2.5 * r) * np.exp(-0.4 * r)
    drdr = (2.5 * np.cos(2.5 * r) - 0.4 * np.sin(2.5 * r)) * np.exp(-0.4 * r)
    hill = np.exp(-3 * ((x - 0.5) ** 2 + (y + 0.5) ** 2))
    v = rip + hill
    drdx = x / (r + eps)
    drdy = y / (r + eps)
    dfdx = drdr * drdx - 6 * (x - 0.5) * hill
    dfdy = drdr * drdy - 6 * (y + 0.5) * hill
    return v, (dfdx, dfdy)


@register(
    "peaks", "MATLAB-style peaks: mix of Gaussians with positive and negative lobes"
)
def f_peaks(x, y):
    e1 = np.exp(-(x**2) - (y + 1) ** 2)
    e2 = np.exp(-(x**2) - y**2)
    e3 = np.exp(-((x + 1) ** 2) - y**2)
    v = 3 * (1 - x) ** 2 * e1 - 10 * (x / 5 - x**3 - y**5) * e2 - (1 / 3) * e3
    dt1dx = 3 * ((-2 * (1 - x)) * e1 + (1 - x) ** 2 * (-2 * x) * e1)
    dt1dy = 3 * (1 - x) ** 2 * (-2 * (y + 1)) * e1
    dt2dx = -10 * ((1 / 5 - 3 * x**2) * e2 + (x / 5 - x**3 - y**5) * (-2 * x) * e2)
    dt2dy = -10 * (-5 * y**4 * e2 + (x / 5 - x**3 - y**5) * (-2 * y) * e2)
    dt3dx = -(1 / 3) * (-2 * (x + 1)) * e3
    dt3dy = -(1 / 3) * (-2 * y) * e3
    return v, (dt1dx + dt2dx + dt3dx, dt1dy + dt2dy + dt3dy)


@register(
    "franke",
    "Franke's function: standard scattered-data benchmark\n\n \
    Franke, R. (1979). A critical comparison of some methods for interpolation of scattered data (No. NPS53-79-003). NAVAL POSTGRADUATE SCHOOL MONTEREY CA.",
    domain=(0, 1, 0, 1),
)
def f_franke(x, y):
    e1 = np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    e2 = np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    e3 = np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    e4 = np.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2))
    v = 0.75 * e1 + 0.75 * e2 + 0.5 * e3 - 0.2 * e4
    dfdx = (
        0.75 * e1 * (-(9 / 2) * (9 * x - 2))
        + 0.75 * e2 * (-(18 / 49) * (9 * x + 1))
        + 0.5 * e3 * (-(9 / 2) * (9 * x - 7))
        - 0.2 * e4 * (-18 * (9 * x - 4))
    )
    dfdy = (
        0.75 * e1 * (-(9 / 2) * (9 * y - 2))
        + 0.75 * e2 * (-9 / 10)
        + 0.5 * e3 * (-(9 / 2) * (9 * y - 3))
        - 0.2 * e4 * (-18 * (9 * y - 7))
    )
    return v, (dfdx, dfdy)


@register("saddle_trig", "Saddle surface modulated by trig: (x²-y²)*sin(3x)*cos(3y)")
def f_saddle_trig(x, y):
    s3x, c3x = np.sin(3 * x), np.cos(3 * x)
    s3y, c3y = np.sin(3 * y), np.cos(3 * y)
    v = (x**2 - y**2) * s3x * c3y
    dfdx = 2 * x * s3x * c3y + (x**2 - y**2) * 3 * c3x * c3y
    dfdy = -2 * y * s3x * c3y + (x**2 - y**2) * s3x * (-3 * s3y)
    return v, (dfdx, dfdy)


@register(
    "sharp_ridge", "Smooth plateau crossed by a sharp exponential ridge along y = x"
)
def f_sharp_ridge(x, y):
    diff = x - y
    ridge = np.exp(-10 * diff**2)
    plat = np.tanh(x + y)
    v = ridge + 0.5 * plat
    dr = ridge * (-20 * diff)
    dp = 0.5 * (1 - plat**2)
    return v, (dr + dp, -dr + dp)


@register("checkerboard", "Smooth checkerboard: sin(3x)*sin(3y) with Gaussian envelope")
def f_checkerboard(x, y):
    env = np.exp(-(x**2 + y**2) * 0.15)
    s3x, c3x = np.sin(3 * x), np.cos(3 * x)
    s3y, c3y = np.sin(3 * y), np.cos(3 * y)
    v = s3x * s3y * env
    dfdx = (3 * c3x * s3y - 0.3 * x * s3x * s3y) * env
    dfdy = (s3x * 3 * c3y - 0.3 * y * s3x * s3y) * env
    return v, (dfdx, dfdy)


@register("spiral", "Rotating spiral wave: sin(r² + 4θ) with radial decay")
def f_spiral(x, y):
    eps = 1e-12
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    phi = r**2 + 4 * theta
    decay = np.exp(-0.3 * r)
    v = np.sin(phi) * decay
    r2 = r**2 + eps
    dphidx = 2 * x - 4 * y / r2
    dphidy = 2 * y + 4 * x / r2
    ddx = -0.3 * x / (r + eps) * decay
    ddy = -0.3 * y / (r + eps) * decay
    cp = np.cos(phi)
    dfdx = cp * decay * dphidx + np.sin(phi) * ddx
    dfdy = cp * decay * dphidy + np.sin(phi) * ddy
    return v, (dfdx, dfdy)


@register("multi_scale", "Low + high frequency superimposed sine waves")
def f_multi_scale(x, y):
    v = np.sin(x) * np.cos(y) + 0.2 * np.sin(7 * x) * np.sin(7 * y)
    dfdx = np.cos(x) * np.cos(y) + 1.4 * np.cos(7 * x) * np.sin(7 * y)
    dfdy = -np.sin(x) * np.sin(y) + 1.4 * np.sin(7 * x) * np.cos(7 * y)
    return v, (dfdx, dfdy)


@register("cliff", "Near-discontinuous tanh cliff along the diagonal")
def f_cliff(x, y):
    t = np.tanh(10 * (x + y))
    dt = 10 * (1 - t**2)
    return t, (dt, dt)


def franke(x, y):
    """Franke's function: standard scattered-data benchmark

    Franke's bivariate function from:
    Franke, R. (1979). A critical comparison of some methods for interpolation of scattered data (No. NPS53-79-003). NAVAL POSTGRADUATE SCHOOL MONTEREY CA.
    """

    e1 = np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    e2 = np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    e3 = np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    e4 = np.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2))
    v = 0.75 * e1 + 0.75 * e2 + 0.5 * e3 - 0.2 * e4
    dfdx = (
        0.75 * e1 * (-(9 / 2) * (9 * x - 2))
        + 0.75 * e2 * (-(18 / 49) * (9 * x + 1))
        + 0.5 * e3 * (-(9 / 2) * (9 * x - 7))
        - 0.2 * e4 * (-18 * (9 * x - 4))
    )
    dfdy = (
        0.75 * e1 * (-(9 / 2) * (9 * y - 2))
        + 0.75 * e2 * (-9 / 10)
        + 0.5 * e3 * (-(9 / 2) * (9 * y - 3))
        - 0.2 * e4 * (-18 * (9 * y - 7))
    )
    return v, (dfdx, dfdy)


def flower(x, y):
    """Flower-shaped function

    Flower-shaped function found in the following example from scipy docs:
    https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolategriddata/
    """
    s = np.hypot(x, y)
    phi = np.arctan2(y, x)

    # Avoid division by zero
    eps = 1e-12
    s_safe = np.where(s == 0, eps, s)

    sin6phi = np.sin(6 * phi)
    cos6phi = np.cos(6 * phi)

    # Function value
    tau = s + s * (1 - s) / 5 * sin6phi
    f = 5 * (1 - tau) + tau  # = 5 - 4*tau

    # Derivatives of s
    ds_dx = x / s_safe
    ds_dy = y / s_safe

    # Derivatives of phi
    dphi_dx = -y / (s_safe**2)
    dphi_dy = x / (s_safe**2)

    # A = s(1-s)/5
    A = s * (1 - s) / 5
    dA_ds = (1 - 2 * s) / 5

    dA_dx = dA_ds * ds_dx
    dA_dy = dA_ds * ds_dy

    # d tau
    dtau_dx = ds_dx + dA_dx * sin6phi + A * cos6phi * 6 * dphi_dx

    dtau_dy = ds_dy + dA_dy * sin6phi + A * cos6phi * 6 * dphi_dy

    # Gradient of f = -4 * grad(tau)
    dfdx = -4 * dtau_dx
    dfdy = -4 * dtau_dy

    return f, dfdx, dfdy
