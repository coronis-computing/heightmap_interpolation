# Copyright (c) 2020 Coronis Computing S.L. (Spain)
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Ricard Campos (ricard.campos@coronis.es)

import math
import numpy as np
from heightmap_interpolation.interpolants.interpolant import Interpolant
from heightmap_interpolation.interpolants.rbf_interpolant import RBFInterpolant
from heightmap_interpolation.interpolants.distance_type_to_functor import distance_type_to_cdist_functor
from heightmap_interpolation.interpolants.distance_type_to_functor import distance_type_to_functor
from heightmap_interpolation.rbf.rbf_type_to_functor import *
import haversine.haversine
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class QTNode:
    """QuadTree Node of the QuadTreePURBF interpolant

    Attributes:
        x: center of the node in X
        y: center of the node in Y
        wh: width/height of the node (nodes are squares!)
        h: height of the node
        pts: points in this node (of size numPts x 3)
        childs: children of the node (up to 4)
        rbf_interp: the interpolator function (just computed at leaves)
        weighting_rbf: the weighting CSRBF
        dist_fun: distance function between points in the XY plane
        overlap: amount of overlap between circles (scalar between 0 and 1). The base radius of a cell (overlap = 0) in the QuadTree corresponds to half the length of its diagonal. This parameter is a multiplicative factor applied to this base radius.
    """

    def __init__(self, x, y, wh, pts, dist_fun, overlap):
        """Constructor of the class"""
        self.x = x
        self.y = y
        self.wh = wh
        self.childs = []
        self.dist_fun = dist_fun
        self.overlap = overlap
        self.rbf_interp = None
        self.weighting_rbf = None

        # From the input points, just retain those within the limits
        inds = self.points_in_node_inds(pts)
        self.pts = pts[inds, :]

    def get_center(self):
        """Gets the center point of the node"""
        return np.array([self.x+(self.wh/2), self.y+(self.wh/2)])

    def get_radius(self):
        """ Gets the radius of the node using the internal distance function (required while evaluating the RBF) it also applies the "overlap" factor"""
        radius = self.get_diagonal()*0.5
        return radius+radius*self.overlap

    def get_diagonal(self):
        """Gets the length of the diagonal of the square represented by the node (computed using the Node's distance function)"""
        return self.dist_fun(np.array([self.x, self.y]), np.array([self.x+self.wh, self.y+self.wh]))

    def get_euclidean_radius(self):
        """ Gets the radius of the node using Euclidean distance"""
        radius = self.get_euclidean_diagonal() * 0.5
        return radius + radius * self.overlap

    def get_euclidean_diagonal(self):
        """Gets the length of the diagonal of the square represented by the node (computed using Euclidean distance)"""
        return math.sqrt((self.x-(self.x+self.wh))**2+(self.y-(self.y+self.wh))**2)

    def points_in_node_inds(self, pts):
        """Computes the indices of the points falling within the circle that this QTNode represents"""
        center = self.get_center()
        radius = self.get_euclidean_radius()

        rads = np.linalg.norm(pts[:, :2]-center, axis=1)
        return rads <= radius

    def is_leaf(self):
        if self.childs:
            return False
        else:
            return True

    def subdivide(self, min_pts, min_cell_side_length):
        #End of recursion if we have less than minimum number of points in the cell OR if cell size is smaller than the minimum
        if self.pts.shape[0] < min_pts or self.wh < min_cell_side_length:
            return

        # Halve the width/height of the current node
        wh = self.wh/2

        # Create the 4 childrens of this node and span subdivision
        sw_node = QTNode(self.x, self.y, wh, self.pts, self.dist_fun, self.overlap)
        sw_node.subdivide(min_pts, min_cell_side_length)
        se_node = QTNode(self.x + wh, self.y, wh, self.pts, self.dist_fun, self.overlap)
        se_node.subdivide(min_pts, min_cell_side_length)
        nw_node = QTNode(self.x, self.y + wh, wh, self.pts, self.dist_fun, self.overlap)
        nw_node.subdivide(min_pts, min_cell_side_length)
        ne_node = QTNode(self.x + wh, self.y + wh, wh, self.pts, self.dist_fun, self.overlap)
        ne_node.subdivide(min_pts, min_cell_side_length)

        # If, after subdivision, NONE of the children contains a point, we also end recursion
        # if sw_node.pts.shape[0] == 0 and se_node.pts.shape[0] == 0 and nw_node.pts.shape[0] == 0 and ne_node.pts.shape[0] == 0:
        #     return

        self.childs = [sw_node, se_node, nw_node, ne_node]


        # sw_node = QTNode(self.x, self.y, wh, self.pts, self.dist_fun, self.overlap)
        # if sw_node.pts.shape[0] < 5:
        #     return
        # sw_node.subdivide(min_pts, min_cell_side_length)
        # se_node = QTNode(self.x + wh, self.y, wh, self.pts, self.dist_fun, self.overlap)
        # if se_node.pts.shape[0] < 5:
        #     return
        # se_node.subdivide(min_pts, min_cell_side_length)
        # nw_node = QTNode(self.x, self.y + wh, wh, self.pts, self.dist_fun, self.overlap)
        # if nw_node.pts.shape[0] < 5:
        #     return
        # nw_node.subdivide(min_pts, min_cell_side_length)
        # ne_node = QTNode(self.x + wh, self.y + wh, wh, self.pts, self.dist_fun, self.overlap)
        # if ne_node.pts.shape[0] < 5:
        #     return
        # ne_node.subdivide(min_pts, min_cell_side_length)
        #
        # # If, after subdivision, NONE of the children contains a point, we also end recursion
        # if sw_node.pts.shape[0] == 0 and se_node.pts.shape[0] == 0 and nw_node.pts.shape[0] == 0 and ne_node.pts.shape[0] == 0:
        #     return
        #
        # self.childs = [sw_node, se_node, nw_node, ne_node]

    def compute_rbf_interpolant_at_leaves(self, **kwargs):
        if self.is_leaf():
            if self.pts is not None and self.pts.shape[0] > 5:
                # Compute the local RBF interpolant corresponding to this QTNode
                self.rbf_interp = RBFInterpolant(self.pts[:, [0]], self.pts[:, [1]], self.pts[:, [2]], **kwargs)
                # Create the weighting RBF (a Wendland RBF with the support == node's radius)
                r = self.get_radius()
                self.weighting_rbf = rbf_type_to_functor("wendland", r)
            # Else, do nothing, this leaf node will not be used for interpolation
        else:
            # Recurse down the tree
            for node in self.childs:
                node = node.compute_rbf_interpolant_at_leaves(**kwargs)

    def get_leaves(self):
        """Get all leaf nodes in the tree"""
        if self.is_leaf():
            nodes = [self]
        else:
            nodes = []
            for child in self.childs:
                nodes += (child.get_leaves())
        return nodes

    def get_leaves_with_samples(self):
        """Get leaf nodes in the tree containing sample nodes"""
        if self.is_leaf():
            if self.pts is not None and self.pts.shape[0] > 5:
                nodes = [self]
            else:
                nodes = []
        else:
            nodes = []
            for child in self.childs:
                nodes += (child.get_leaves_with_samples())
        return nodes

    def free_memory(self):
        """Remove points stored at non-leaf nodes

        Use this function after creating the tree (i.e., after applying the subdivide method)
        """
        if self.is_leaf():
            return
        else:
            # Non-leaf node, eliminate points
            self.pts = None
            # Continue the traversal down the tree
            self.childs[0].free_memory()
            self.childs[1].free_memory()
            self.childs[2].free_memory()
            self.childs[3].free_memory()

    def contains(self, node):
        """Check if the input node domain is completely contained in the domain of this node"""
        center_a = self.get_center()
        radius_a = self.get_radius()
        center_b = node.get_center()
        radius_b = node.get_radius()

        # Distance between centers
        d = math.sqrt( (center_b[0]-center_a[0])**2 + (center_b[1]-center_a[1])**2)

        # Input node is inside node self if its radius is larger than the sum of the distance between centers and the radius of the input node
        return radius_a >= d+radius_b


class QuadTreePURBFInterpolant(Interpolant):
    """Quad Tree-based Partition of Unity Radial Basis Function interpolant"""

    def __init__(self, x, y, z, **kwargs):
        """Constructor of the class"""

        # Base class constructor
        super().__init__(x, y, z)

        # Check the input parameters
        dist_type = kwargs.pop("distance_type", "euclidean")
        min_pts = kwargs.pop("min_points_in_cell", 25)
        min_cell_size_percent = kwargs.pop("min_cell_size_percent", 0.01)
        overlap = kwargs.pop("overlap", 0.25)
        self.overlap_inc = kwargs.pop("overlap_increment", 0.001)
        domain = kwargs.pop("domain", None)

        if overlap < 0:
            raise ValueError("overlap should be greater than zero")
        if min_cell_size_percent > 1 or min_cell_size_percent < 0:
            raise ValueError("min_cell_size_percent should be a number between zero and one")
        if domain:
            if len(domain) != 3:
                raise ValueError("domain must be a 3-elements list")
        if min_pts > self.data.shape[0]:
            raise ValueError("min_pts must be smaller than the number of input samples")

        # Get the distance function from its type
        self.dist_fun = distance_type_to_functor(dist_type)
        self.cdist_fun = distance_type_to_cdist_functor(dist_type)

        if not domain:
            # If the domain is empty, we infer it from the input samples
            minx = np.min(self.data[:, 0])
            maxx = np.max(self.data[:, 0])
            miny = np.min(self.data[:, 1])
            maxy = np.max(self.data[:, 1])
            w = maxx - minx
            h = maxy - miny
            wh = max(w, h)
            domain = [minx, miny, wh]
        else:
            wh = domain[2]

        # Create the root node of the Quad Tree
        self.root = QTNode(domain[0], domain[1], domain[2], self.data, self.dist_fun, overlap)

        # Subdivide the root node (and effectively create the tree top to bottom)
        self.root.subdivide(min_pts, wh*min_cell_size_percent)

        # Remove stored points in non-leaf nodes, from now on we will just use leaves
        self.root.free_memory()

        # Correct all leaf nodes to contain a minimum number of points, so that all the input domain is queriable
        self.correct(min_pts)

        # --- Debug ---
        # self.plot()
        self.show_interpolant_stats()

        # Compute a local RBF interpolator for each leaf
        self.root.compute_rbf_interpolant_at_leaves(**kwargs)

    def correct(self, min_pts):
        """Correct the leaves so that they cover a minimum number of input points"""

        # Create the search structure
        tree = scipy.spatial.cKDTree(self.data[:, :2])

        # Get the leaves
        leaves = self.root.get_leaves()

        # Sort the leaves from larger to smaller
        leaves.sort(key=lambda x: x.get_radius(), reverse=True)

        i = 0
        while i < len(leaves):
            # If the number of points in the leaf is less than min_pts, increase the radius until it contains the minimum number of points
            num_pts = leaves[i].pts.shape[0]
            if num_pts < min_pts:
                # The current overlap
                overlap = leaves[i].overlap

                # Increase the overlap until the radius contains the required number of points (may be more!)
                center = leaves[i].get_center()
                while num_pts < min_pts:
                    overlap = overlap+self.overlap_inc
                    leaves[i].overlap = overlap
                    radius = leaves[i].get_radius()
                    ind = tree.query_ball_point(center, radius)
                    num_pts = len(ind)
                leaves[i].pts = self.data[ind, :]

                # Check if the enlarged cell completely contains within its domain other cells
                inds_to_delete = []
                inds_except_current = [*range(0, len(leaves))]
                del inds_except_current[i]
                for j in inds_except_current:
                    if leaves[i].contains(leaves[j]):
                        inds_to_delete.append(j)

                # How many indices we have to delete before the current position? (needed to know how to update i)
                num_updated_before_i = sum(x < i for x in inds_to_delete)

                # And delete them
                for index in sorted(inds_to_delete, reverse=True):
                    leaves[index].pts = None # We effectively disregard this leaf by setting its points to None
                    del leaves[index]
                # num_updated_before_i = 0
            i = i-num_updated_before_i+1

    def plot(self, ax=None):
        """Plot the QuadTree"""
        if not ax:
            fig = plt.figure(figsize=(12, 8))
            plt.title("Quadtree")
            ax = fig.gca()
        leaves = self.root.get_leaves()
        max_samples_leaf = 0
        for leaf in leaves:
            if leaf.pts is not None:
                num_pts_leaf = leaf.pts.shape[0]
                if num_pts_leaf > max_samples_leaf:
                    max_samples_leaf = num_pts_leaf
                # if num_pts_leaf > 0:
                #     print("Num. pts in leaf = " + str(num_pts_leaf))
                ax.add_patch(patches.Rectangle((leaf.x, leaf.y), leaf.wh, leaf.wh, fill=False))
                ax.add_patch(patches.Circle((leaf.x+leaf.wh*0.5, leaf.y+leaf.wh*0.5), leaf.get_euclidean_radius(), fill=False))
                # plt.plot(leaf.pts[:, 0], leaf.pts[:, 1], 'ro')
                # ax.axis("equal")
                # plt.show()
        ax.axis("equal")
        plt.plot(self.data[:, 0], self.data[:, 1], 'ro')
        plt.show(block=False)

    def show_interpolant_stats(self):
        leaves = self.root.get_leaves_with_samples()
        min_samples_leaf = 9999999999999
        max_samples_leaf = 0
        min_radius = 99999999999999
        max_radius = 0
        for leaf in leaves:
            if leaf.pts is not None:
                num_pts_leaf = leaf.pts.shape[0]
                if num_pts_leaf > max_samples_leaf:
                    max_samples_leaf = num_pts_leaf
                elif num_pts_leaf < min_samples_leaf:
                    min_samples_leaf = num_pts_leaf
            radius = leaf.get_radius()
            if radius > max_radius:
                max_radius = radius
            elif radius < min_radius:
                min_radius = radius

        print("QuadTreePU Interpolant statistics:")
        print("  - Min. domain radius =", str(min_radius))
        print("  - Max. domain radius =", str(max_radius))
        print("  - Max. samples in a RBF =", str(max_samples_leaf))
        print("  - Num. local RBF =", str(len(leaves)))

    def __call__(self, x, y):
    # def new_call(self, x, y):
        # Check sizes
        if x.size != y.size:
            raise ValueError("x and y should have the same number of elements")

        if x.shape != y.shape:
            print("[WARNING] x.shape != y.shape. The size of the output matrix will be that of x")

        # Reshape input, for convenience
        orig_shape = x.shape
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        num_queries = len(x)

        # Get the leaf nodes
        leaves = self.root.get_leaves_with_samples()

        # Compute the centers and radius of each leaf
        num_leaves = len(leaves)

        # Create the search structure
        tree = scipy.spatial.cKDTree(np.hstack((x, y)))

        f = np.zeros((num_queries, 1))
        w = np.zeros((num_queries, 1))
        for i in range(num_leaves):
            # Find those input points falling in the current leaf
            # ind = dists[:, i] <= radius[i]
            center = leaves[i].get_center()
            radius = leaves[i].get_radius()
            ind = tree.query_ball_point(center, radius)

            # Compute the distances
            center = center.reshape(1, 2)
            # dists = scipy.spatial.distance.cdist(np.hstack((x[ind], y[ind])), center, self.dist_fun)
            # dists = scipy.spatial.distance.cdist(np.hstack((x[ind], y[ind])), center)
            dists = self.cdist_fun(np.hstack((x[ind], y[ind])), center)

            # Compute the RBF value at those points
            rbf_eval = leaves[i].rbf_interp(x[ind], y[ind])

            # Compute the weighting function at those points
            d = dists
            d = d.reshape(-1, 1)
            weights = leaves[i].weighting_rbf(d)

            # Apply the weights to the corresponding function and accumulate
            f[ind, :] = f[ind, :] + rbf_eval*weights

            # Accumulate the weights for the final division
            w[ind, :] = w[ind, :] + weights

        # Apply the final division by the accumulated weights
        z = f/w # Here a division by w = 0 will occur for those points outside the domain covered by the quadtree. Since this will result in a NaN, we use this value as an indicator that the z is undefined at that point.

        # Get z back to the original shape of the input
        return z.reshape(orig_shape)


    # def old_call(self, x, y):
    # # def __call__(self, x, y):
    #     # Check sizes
    #     if x.size != y.size:
    #         raise ValueError("x and y should have the same number of elements")
    #
    #     if x.shape != y.shape:
    #         print("[WARNING] x.shape != y.shape. The size of the output matrix will be that of x")
    #
    #     # Reshape input, for convenience
    #     orig_shape = x.shape
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     num_queries = len(x)
    #
    #     # Get the leaf nodes
    #     leaves = self.root.get_leaves_with_samples()
    #
    #     # Compute the centers and radius of each leaf
    #     num_leaves = len(leaves)
    #     centers = np.zeros((num_leaves, 2))
    #     radius = np.zeros((num_leaves, 1))
    #     for i in range(num_leaves):
    #         centers[i, :] = leaves[i].get_center()
    #         radius[i] = leaves[i].get_radius()
    #
    #     # Compute all pair-wise distances
    #     dists = scipy.spatial.distance.cdist(np.hstack((x, y)), centers, self.dist_fun)
    #
    #     f = np.zeros((num_queries, 1))
    #     w = np.zeros((num_queries, 1))
    #     for i in range(num_leaves):
    #         # Find those input points falling in the current leaf
    #         ind = dists[:, i] <= radius[i]
    #
    #         # Compute the RBF value at those points
    #         rbf_eval = leaves[i].rbf_interp(x[ind], y[ind])
    #
    #         # Compute the weighting function at those points
    #         d = dists[ind, [i]]
    #         d = d.reshape(-1, 1)
    #         weights = leaves[i].weighting_rbf(d)
    #
    #         # Apply the weights to the corresponding function and accumulate
    #         f[ind, :] = f[ind, :] + rbf_eval*weights
    #
    #         # Accumulate the weights for the final division
    #         w[ind, :] = w[ind, :] + weights
    #
    #     # Apply the final division by the accumulated weights
    #     z = f/w # Here a division by w = 0 will occur for those points outside the domain covered by the quadtree. Since this will result in a NaN, we use this value as an indicator that the z is undefined at that point.
    #
    #     # Get z back to the original shape of the input
    #     return z.reshape(orig_shape)