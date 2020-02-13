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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from heightmap_interpolation.interpolants.rbf_interpolant import  RBFInterpolant
import heightmap_interpolation.apps.bivariate_functions as demo_functions


demo_parameters = {
    "franke": {
        "function": demo_functions.franke,
        "domain_xmin": -1,
        "domain_xmax": 1,
        "domain_ymin": -1,
        "domain_ymax": 1,
        "num_samples": 400,
        "interpolated_grid_x": 100,
        "interpolated_grid_y": 100,
        "distance_type": "euclidean",
        "linear_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "cubic_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "quintic_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "thinplate_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "green_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "multiquadric_rbf": {
            "polynomial_degree": 1,
            "epsilon": 1,
            "regularization": 0
        },
        "tension_rbf": {
            "polynomial_degree": 1,
            "epsilon": 1,
            "regularization": 0
        },
        "regularized_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0.1,
            "regularization": 0
        },
        "gaussian_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0.15,
            "regularization": 0
        },
        "wendland_rbf": {
            "polynomial_degree": 1,
            "epsilon": 1,
            "regularization": 0
        }
    },
    "flower": {
        "function": demo_functions.flower,
        "domain_xmin": -1,
        "domain_xmax": 1,
        "domain_ymin": -1,
        "domain_ymax": 1,
        "num_samples": 400,
        "interpolated_grid_x": 100,
        "interpolated_grid_y": 100,
        "distance_type": "euclidean",
        "linear_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "cubic_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "quintic_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "thinplate_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "green_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0,
            "regularization": 0
        },
        "multiquadric_rbf": {
            "polynomial_degree": 1,
            "epsilon": 1,
            "regularization": 0
        },
        "tension_rbf": {
            "polynomial_degree": 1,
            "epsilon": 1,
            "regularization": 0
        },
        "regularized_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0.1,
            "regularization": 0
        },
        "gaussian_rbf": {
            "polynomial_degree": 1,
            "epsilon": 0.15,
            "regularization": 0
        },
        "wendland_rbf": {
            "polynomial_degree": 1,
            "epsilon": 1,
            "regularization": 0
        }
    }
}


# Main function
if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser(
        description="Demo function showing the bivariate function interpolation using RBFs")
    parser.add_argument("-f", action="store", dest='function_type', type=str, default="flower",
                        help="Test function to sample. Possibilities: 'flower', 'franke'")
    parser.add_argument("-3d", action="store_true", dest='plot_3d', default=False,
                        help="Set this flag to plot the results in 3D")
    param = parser.parse_args()

    # Get some samples of the function within the domain
    xmin = demo_parameters[param.function_type]["domain_xmin"]
    xmax = demo_parameters[param.function_type]["domain_xmax"]
    ymin = demo_parameters[param.function_type]["domain_ymin"]
    ymax = demo_parameters[param.function_type]["domain_xmax"]
    num_samples = demo_parameters[param.function_type]["num_samples"]
    f = demo_parameters[param.function_type]["function"]

    x = np.random.uniform(xmin, xmax, num_samples)
    y = np.random.uniform(ymin, ymax, num_samples)
    z = f(x, y)

    # Compute a regular grid, where all data points will be interpolated according to the RBF interpolator
    sx = demo_parameters[param.function_type]["interpolated_grid_x"]
    sy = demo_parameters[param.function_type]["interpolated_grid_y"]
    xi, yi = np.meshgrid(np.linspace(xmin, xmax, sx), np.linspace(xmin, xmax, sx))
    gt_z = f(xi, yi) # Ground-truth z

    # Draw the initial data
    fig = plt.figure()
    ax = []
    if param.plot_3d:
        projection_type = '3d'
    else:
        projection_type = 'rectilinear'
    sp_rows = 3
    sp_cols = 4
    sp_ind = 0
    ax.append(fig.add_subplot(sp_rows, sp_cols, sp_ind+1, projection=projection_type))
    if param.plot_3d:
        ax[sp_ind].plot_surface(xi, yi, gt_z)
    else:
        ax[sp_ind].contourf(xi, yi, gt_z)
    ax[sp_ind].scatter(x, y, z, c='k', alpha=1, marker='.')
    ax[sp_ind].set_title('Sample points on f(x,y)')
    sp_ind = sp_ind+1

    # Interpolate the grid points using different methods
    rbf_types = ('linear', 'cubic', 'quintic', 'thinplate', 'green', 'multiquadric', 'tension', 'regularized', 'gaussian', 'wendland')
    for i, rbf_type in enumerate(rbf_types):
        # Collect rbf_type-dependant parameters
        rbf_type_ext = rbf_type + "_rbf"
        distance_type = demo_parameters[param.function_type]["distance_type"]
        polynomial_degree = demo_parameters[param.function_type][rbf_type_ext]["polynomial_degree"]
        epsilon = demo_parameters[param.function_type][rbf_type_ext]["epsilon"]
        regularization = demo_parameters[param.function_type][rbf_type_ext]["regularization"]

        # Print the current step
        print("- Computing " + rbf_type + " RBF interpolation, parameters:")
        print("    - distance_type = " + distance_type)
        print("    - polynomial_degree = " + str(polynomial_degree))
        print("    - epsilon = " + str(epsilon))
        print("    - regularization = " + str(regularization))

        # Create the interpolation function from these known data points
        interpolant = RBFInterpolant(x, y, z, rbf_type=rbf_type,
                                              distance_type=distance_type,
                                              polynomial_degree=polynomial_degree,
                                              epsilon=epsilon,
                                              regularization=regularization)

        # Interpolate
        zi = interpolant(xi, yi)

        # Plot results
        ax.append(fig.add_subplot(sp_rows, sp_cols, sp_ind+1, projection=projection_type))
        if param.plot_3d:
            ax[sp_ind].plot_surface(xi, yi, zi)
        else:
            ax[sp_ind].contourf(xi, yi, zi)
        ax[sp_ind].set_title(rbf_type + " RBF interpolation")
        sp_ind = sp_ind + 1

    # plt.tight_layout()
    plt.show()
