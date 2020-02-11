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

import numpy as np
import haversine.haversine
import geopy.distance
import scipy

# Get the distance function from its type
def distance_type_to_functor(dist_type: str):
    if dist_type == "euclidean":
        dist_fun = lambda u, v: np.sqrt(((u - v) ** 2).sum())
    elif dist_type == "haversine":
        dist_fun = haversine
    elif dist_type == "vincenty":
        dist_fun = geopy.distance.vincenty
    else:
        raise ValueError("distance-type should be either 'euclidean', 'haversine' or 'vincenty'")
    return dist_fun

def distance_type_to_cdist_functor(dist_type: str):
    if dist_type == "euclidean":
        def my_cdist(XA, XB):
            return scipy.spatial.distance.cdist(XA, XB)
        return my_cdist
    elif dist_type == "haversine":
        def my_cdist(XA, XB):
            return scipy.spatial.distance.cdist(XA, XB, haversine)
        return my_cdist
    elif dist_type == "vincenty":
        def my_cdist(XA, XB):
            return scipy.spatial.distance.cdist(XA, XB, geopy.distance.vincenty)
        return my_cdist
    else:
        raise ValueError("distance-type should be either 'euclidean', 'haversine' or 'vincenty'")


def distance_type_to_pdist_functor(dist_type: str):
    if dist_type == "euclidean":
        def my_cdist(X):
            return scipy.spatial.distance.pdist(X)
        return my_cdist
    elif dist_type == "haversine":
        def my_cdist(X):
            return scipy.spatial.distance.pdist(X, haversine)
        return my_cdist
    elif dist_type == "vincenty":
        def my_cdist(X):
            return scipy.spatial.distance.pdist(X, geopy.distance.vincenty)
        return my_cdist
    else:
        raise ValueError("distance-type should be either 'euclidean', 'haversine' or 'vincenty'")

