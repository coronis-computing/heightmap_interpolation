# Copyright (c) 2021 Coronis Computing S.L. (Spain)
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
from heightmap_interpolation.misc.conditional_print import ConditionalPrint
import shutil
import subprocess
import os 
import numpy as np

class PoissonReconExternalInterpolant(Interpolant):

    def __init__(self, x, y, z, 
                 point_interpolant_exe_path = None, 
                 adaptive_tree_visualization_exe_path = None, 
                 workspace=None, 
                 verbose=False,
                 degree=2,
                 boundary_type="free",
                 depth=8,
                 solve_depth=-1,
                 full_depth=5,
                 base_depth=None,
                 base_v_cycles=4,
                 laplacian_weight=0.0,
                 bi_laplacian_weight=1.0,
                 iters=8,
                 exact=False,
                 parallel_type="openmp",
                 schedule_type="dynamic",
                 chunk_size=128,
                 cg_accuracy=0.001,
                 max_memory=0,
                 in_core=False):
        """ Constructor """
        # Base class constructor
        super().__init__(x, y, z)

        self.cp = ConditionalPrint(verbose)
        self.cp.print("")

        # This interpolant is just calling an external tool, so we need to make sure such tool is on the path OR the user
        # has provided the path to the tool        
        self.point_interpolant_exe_path = point_interpolant_exe_path if point_interpolant_exe_path else shutil.which('PointInterpolant')
        if not self.point_interpolant_exe_path:
            raise OSError('PointInterpolant executable not found!\n "ext_poisson" is an external interpolator, it requires you to install the PoissonRecon binaries manually, and then either set the system PATH to include the binaries, or set the "point_interpolant_path" parameter. See https://github.com/mkazhdan/PoissonRecon for more info.')
        self.adaptive_tree_visualization_exe_path = adaptive_tree_visualization_exe_path if adaptive_tree_visualization_exe_path else shutil.which('AdaptiveTreeVisualization')
        if not self.adaptive_tree_visualization_exe_path:
            raise OSError('AdaptiveTreeVisualization executable not found!\n "ext_poisson" is an external interpolator, it requires you to install the PoissonRecon binaries manually, and then either set the system PATH to include the binaries, or set the "point_interpolant_path" parameter. See https://github.com/mkazhdan/PoissonRecon for more info.')

        if workspace is None:
            print("[WARNING] Using current directory as workspace, some intermediate files will be created here! If the execution does not end successfully, delete the workspace directory and try again.")
            workspace = os.path.join(os.getcwd(), 'workspace')            
        self.workspace = workspace

        if os.path.exists(self.workspace):
            raise OSError(f'Workspace directory "{self.workspace}" already exists! If it was created from a previous execution that did not end successfully, delete the workspace directory and try again. Otherwise, choose another folder with the --workspace parameter')

        self.cp.print("Creating a workspace dir at: {} (will be deleted upon correct exit)".format(self.workspace))
        os.makedirs(workspace, exist_ok=True)

        # Store the samples there as XYZ
        ref_values_path = os.path.join(self.workspace, "interpolant_samples.xyz")
        np.savetxt(ref_values_path, np.column_stack((x, y, z)), delimiter=" ")

        # Create the interpolant
        self.interpolant_file_path = os.path.join(self.workspace, "quadratic.2D.tree")
        # Base command
        cmd = [self.point_interpolant_exe_path, "--inValues", ref_values_path, "--tree", self.interpolant_file_path, "--dim", "2",
               "--degree", str(degree), "--depth", str(depth), "--solveDepth", str(solve_depth),
               "--fullDepth", str(full_depth), "--baseVCycles", str(base_v_cycles),
               "--lapWeight", str(laplacian_weight), "--biLapWeight", str(bi_laplacian_weight), "--iters", str(iters),
               "--chunkSize", str(chunk_size),
               "--cgAccuracy", str(cg_accuracy), "--maxMemory", str(max_memory), ]
        # Add further options (flags and things we let the user input as strings)
        cmd.append("--bType")
        if boundary_type == "free":            
            cmd.append("1")
        elif boundary_type == "dirichlet":
            cmd.append("2")
        elif boundary_type == "neumann":
            cmd.append("3")
        else:            
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        if base_depth is not None:
            cmd.append("--baseDepth")
            cmd.append(str(base_depth))
        if exact:
            cmd.append("--exact")
        cmd.append("--parallel")
        if parallel_type == "openmp":
            cmd.append("0")
        elif parallel_type == "async":
            cmd.append("1")
        elif parallel_type == "none":
            cmd.append("2")
        else:
            raise ValueError(f"Unknown parallel type: {parallel_type}")
        if in_core:
            cmd.append("--inCore")       
        
        # Run the command to create the interpolant
        self.cp.print("Running the following command:\n\t {}".format(" ".join(cmd)))
        subprocess.run(cmd, check=True)

    def __call__(self, x, y):
        """Evaluates the interpolant at the x, y locations"""
        self.cp.print("")

        # Write the query points to a file
        query_pts_file_path = os.path.join(self.workspace, "query_points.xy")
        np.savetxt(query_pts_file_path, np.column_stack((x, y)), delimiter=" ")

        # Sample the interpolant at the query points
        results_file_path = os.path.join(self.workspace, "query_points_interpolated.xyz")
        results_file = open(results_file_path, 'w')
        cmd = [self.adaptive_tree_visualization_exe_path, 
               "--in", self.interpolant_file_path, 
               "--samples", query_pts_file_path]
        self.cp.print("Running the following command:\n\t {}".format(" ".join(cmd)))
        subprocess.run(cmd, stdout=results_file, check=True)
        results_file.close()
        res = np.loadtxt(results_file_path)
        
        return res[:,2]
    
    def cleanup(self):
        shutil.rmtree(self.workspace)