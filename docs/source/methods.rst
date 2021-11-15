Interpolation Methods
=====================

"Interpolation" is a broad term. In our case, it consists in obtaining elevation values at cells/points given a set of known reference elevation data at known locations. However, depending on the sampling/distribution of the input data, and where do we want to interpolate it, there are several ways of dealing with this problem.

The typical literature for interpolation do not consider any specific distribution for the samples. In this sense, we find the **Scattered-data interpolators**. These methods work in two steps:

1. Take the known data points as reference to create an *interpolator*.
2. Apply the interpolator at whatever query point you desire. For interpolations on a grid, as in our case, the interpolation is queried to all the grid cells to be interpolated.

However, there are several cases in which the interpolation problem consists in filling "missing data", in the sense of having continuous parts of the map that are missing and that we need to fill given the known data that is present on the map.
In these cases, the problem can be seen as "fill the holes in a coherent way". Obviously, the scattered data interpolators can be used for this purpose.
However, there is a wide literature of methods trying to take into account that the "filling" happens on a regular grid. In the computer vision literature, these are called **inpainting** methods.
In this toolbox we use inpainting approaches, usually devised for image processing, to tackle the interpolation problem on elevation grids. As mentioned above, these methods only work on the regular grids, but provide the advantage of providing **higher-degree** approximations **faster** than some similar approaches in the scattered area, and require **much less memory** to execute (they just operate on the input grid, and do not need to build complex systems of equations).

Of course, there are cases in which we have a map that is a mix of both: we have densely sampled areas, as well as sparsely sampled parts.

In these cases, all scattered data interpolators (because they do not care about the shape of your data), while some

For each of the methods in the package, we will briefly describe their behaviour, list the parameters available to tune in each case, provide the data/cases for which a given method is more suitable, and list their pros/cons.

Scattered data Interpolators
****************************

Nearest-Neighbors
-----------------

Each cell to interpolate gets its value from the nearest reference cell.

This method is just an interphase for the `scipy.interpolate.NearestNDInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html>`_.

Parameters
++++++++++

There are no specific parameters for this interpolator.

Suitable for
++++++++++++

* Quick initialization of the interpolation using PDE inpainters (see sections below).
* Quick large-area interpolation.

Advantages
++++++++++

* Fastest interpolator.
* As opposed to the other two fast scattered data interpolation methods (*linear* and *cubic*), it can interpolate outside of the convex hull of the reference data.

Disadvantages
+++++++++++++

* Results look *blocky*, as many points get the same elevation value.

Linear
------

Creates a linear interpolant by creating a 2D Delaunay triangulation using the reference data points.
Upon a given query point, it searches in which of the triangle in the XY plane it falls, and computes a barycentric interpolation of the elevation using the reference values at the vertices of the triangle.

This method is just an interphase for the `scipy.interpolate.LinearNDInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html>`_.

Parameters
++++++++++


Suitable for
++++++++++++

* Quick large-area interpolation.

Advantages
++++++++++

* Provides a smoother interpolation than the *linear* method at a similar computational cost.

Disadvantages
+++++++++++++

Cubic
-----

As in the *linear* method, it creates a 2D Delaunay triangulation using the reference data points and query points are
interpolated within the triangle where they fall in the XY plane. However, as opposed to using a linear barycentric
interpolation within the triangle, it uses a piecewise cubic interpolating Bezier polynomial.

This method is just an interphase for the `scipy.interpolate.CloughTocher2DInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html>`_.

Parameters
++++++++++


Suitable for
++++++++++++

* Quick large-area interpolation.

Advantages
++++++++++

* Provides a smoother interpolation than the *linear* method at a similar computational cost.

Disadvantages
+++++++++++++

* May produce artifacts if samples' density vary rapidly, or if the scattered samples are not uniformly distributed over the inpainting area.
* Does not "extrapolate" in query locations outside of the convex hull of the reference data.

Radial Basis Functions
----------------------

A Radial Basis Funcion (RBF) is a function whose value depends only on the distance between the input and some fixed point. The basic idea of a RBF interpolator is to construct an interpolant of the data using a summation of several RBF centered at the input data points. The formal definition is the following:

.. math:: s(x) = p(x) + \sum^{N}_{i=1} \lambda_i \phi(|x-x_i|)

Where :math:`\phi(|x-x_i|)` is a given radial basis function :math:`\phi` centered at a known/reference data point :math:`x_i`, :math:`p(x)` is a polynomial of small degree, evaluated at point :math:`x`, and :math:`\lambda_i` is a scalar weight.

Thus, basically, we have a polynomial (1st term) capturing the main trend of the data, and the summation of weighted RBFs (2nd term).
Therefore, the unknowns of this interpolant are mainly the few terms of the polynomial :math:`p(x)` and the :math:`\lambda_i` weight of each RBF. These unknowns can be solved using a linear system of equations. In matrix form, this corresponds to:

.. math::
    A = \left( \begin{matrix}
                A & P & \\
                P^T & 0
            \end{matrix}
        \right)
        \left( \begin{matrix}
                \lambda \\
                c
            \end{matrix}
        \right)
    =
    \left( \begin{matrix}
                f \\
                0
            \end{matrix}
        \right)

Where:

* :math:`A_{i,j} = \phi(|x_i-x_j|)`.
* :math:`P_{i,j} = p_j(x_i)` are the coefficients of the polynomial.
* :math:`f` are known elevation values at :math:`x_i`.

While solving this system of equations is conceptually simple, it is important to notice that the matrix A is a square
matrix with side length equal to the number of input data points.
Therefore, this formulation becomes prohibitively complex for large datasets, as the amount of memory and computational
resources required for solving and/or evaluating the interpolant is too large.

However, it has the nice feature of allowing some "tunning" of the properties of the interpolating surface via the RBF type that we choose.

The RBF types available in this package are listed in the following. Note that some of these definitions have an :math:`\epsilon` parameter modifying their "shape":

* linear: :math:`\phi(r) = r`
* cubic: :math:`\phi(r) = r^3`
* quintic: :math:`\phi(r) = r^5`
* thin plate spline: :math:`\phi(r) = r^2 log(r)`. It provides a biharmonic interpolant.
* gaussian: :math:`\phi(r) = e^{-(\epsilon r)^2}`
* green: :math:`\phi(r) = r^2 (log(r)-1)`
* multiquadric: :math:`\phi(r) = \sqrt{1+(\epsilon r)^2}`
* tension spline: :math:`\phi(r) = -\frac{1}{2 \pi \epsilon^2}(log(\frac{r\epsilon}{2} + C_e + K_0(r\epsilon))`, being :math:`C_e` the Euler constant and :math:`K_0` the modified Bessel function (same as in [MITAS1988]_, equation 50).
* regularized spline: :math:`\phi(r) = \frac{1}{s\pi} \left{ \frac{r^4}/4 \left[ log(\frac{r}{2\pi} + C_e - 1 \right] + \epsilon^2 \left[K_0(\frac{r}{\epsilon}) + C_e + log(\frac{r}{2\pi}) \right] \right)` (same as in [MITAS1988]_, equation 56).

.. [MITAS1988] Mitas, L., and H. Mitasova. 1988. General Variational Approach to the Interpolation Problem. Comput. Math. Applic. Vol. 16. No. 12. pp. 983–992. Great Britain.

Parameters
++++++++++



Suitable for
++++++++++++

* Best fidelity for the interpolant.
* Small datasets. They can be small in the number of input reference points, and large number of query points (huge scattered data).

Advantages
++++++++++

* Allows tunning the properties of the interpolating surface by changing the RBF type.

Disadvantages
+++++++++++++

* Depending on the input data and the selected RBF type, the resulting interpolant surface may **overshoot** the input data (minimum and maximum elevation values may be outside the range of the input data).

Partition of Unity Radial Basis Functions
-----------------------------------------

Partition of Unity Radial Basis Functions (*purbf*) is an attempt to lower as much as possible the memory and
computational requirements of the RBF interpolator.

The Partition of Unity Method (PUM) divides the global domain into smaller overlapping subdomains. In each of these subdomains, a RBF interpolant is computed using the formulation presented in Section \ref{sec:rbf}. Then, when evaluating a query location, the contribution of several neighboring RBF interpolations are *blended* together in order to get the final value.

The PU interpolant preserves the local approximation order for the global fit. Therefore, large RBF interpolants can be computed by solving small interpolation problems and then combining them together with the global PU.

..
    images

Parameters
++++++++++

All RBF types available for the RBF interpolator are also availble in this case.

Suitable for
++++++++++++

Advantages
++++++++++

* Tunnable output: as in the RBF interpolator, changing the base RBF will change the shape/properties of the output interpolated surface.
* Preferrable in cases where the number of reference data points is far smaller than the number of points to interpolate.

Disadvantages
+++++++++++++

* While compared to the pure RBF, reduction in computational requirements is huge, it may not be sufficient for processing large datasets (i.e., it will still be slower to compute than other options in this package).

PDE-based Inpainting Interpolators
**********************************

We implement all the methods in this section using the same explicit PDE solver. Therefore, there is a set of parameters that are common to all the methods

Common Parameters
-----------------

The parameters that are common to all PDE-based interpolators affect the behaviour of the Finite-Differences solver:


Speed-Up Tricks
---------------

Harmonic Inpainter
------------------

Parameters
++++++++++

Suitable for
++++++++++++

Advantages
++++++++++

* Fastest of the inpainting methods.
* It will never overshoot the data (minimum and maximum elevation values never below/over the reference ones).

Disadvantages
+++++++++++++

* Does not work well with sparsely sampled data: isolated data points will not contribute to the interpolation (they will be left "as they are").

Total Variation (TV) Inpainter
------------------------------

Parameters
++++++++++

Suitable for
++++++++++++

Advantages
++++++++++

Disadvantages
+++++++++++++

* Does not work well with sparsely sampled data: isolated data points will not contribute to the interpolation (they will be left "as they are").

Continous Curvature Splines in Tension (CCST) Inpainter
-------------------------------------------------------



Note that this is a re-implementation/variant of the method in `GMT surface <http://gmt.soest.hawaii.edu/doc/latest/surface.html>`_.



Parameters
++++++++++

Suitable for
++++++++++++

Advantages
++++++++++

* It provides an "easy to tune" mix of an harmonic and a biharmonic interpolant.

Disadvantages
+++++++++++++

* Slower execution time.

Absolutely Minimizing Lipschitz Extension (AMLE) Inpainter
----------------------------------------------------------





Parameters
++++++++++

Suitable for
++++++++++++

Advantages
++++++++++

Disadvantages
+++++++++++++

* It is the only inpainter method in this package that was originally devised for interpolating heightmaps (the rest come from the image processing literature).
* Slower execution time.


