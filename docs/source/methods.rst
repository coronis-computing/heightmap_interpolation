.. _methods:

Interpolation Methods
=====================

"Interpolation" is a broad term. In our case, it consists in obtaining elevation values at cells/points given a set of
known reference elevation data at known locations. However, depending on the sampling/distribution of the input data,
and where do we want to interpolate it, there are several ways of dealing with this problem.

The typical literature for interpolation does not consider any specific distribution for the samples. In this sense,
we find the **Scattered-data interpolators**. These methods work in two steps:

1. Take the known data points as reference to create an *interpolator*.
2. Apply the interpolator at whatever query point you desire. For interpolations on a grid, as in our case, the interpolation is queried at all the grid cells to be interpolated.

However, there are several cases in which the interpolation problem consists in filling "missing data", in the sense of
having continuous and densely-sampled parts of the map that are missing and that we need to fill given the known data surrounding these parts.
In these cases, the problem can be seen as "filling the holes in a coherent way". Obviously, the scattered data
interpolators can be used for this purpose. However, there is a wide literature of methods trying to take advantage of
the "filling" happening on a regular grid. In the computer vision literature, these are called **inpainting** methods.
In this toolbox we use inpainting approaches, usually devised for image processing, to tackle the interpolation problem
on elevation grids. As mentioned above, these methods only work on the regular grids, but provide the advantage of
providing **higher-degree** approximations **faster** than some similar approaches in the scattered area,
and require **much less memory** to execute (the solver we implement just applies convolution operations on the input grid).

In the following sections, for each of the methods in the package, we will briefly describe their behaviour,
list the parameters available to tune in each case, provide the cases for which a given method is more suitable,
and list their pros/cons.

In addition, in order to get a qualitative evaluation of the behaviour of each method, we will run them with default
parameters on the following dataset:

.. figure:: images/example_dataset.png
    :width: 500
    :align: center

    Example dataset. Colored areas and points represent the known reference elevation data, while the area to interpolate is shown in white. Data by courtesy of the Swedish Maritime Administration

Note that this dataset mixes both scattered and densely-sampled reference data.


.. _scattered_methods:

Scattered data Interpolators
****************************

Nearest Neighbors
-----------------

.. figure:: images/nearest.png
    :width: 500
    :align: center

    Example dataset interpolated using the Nearest Neighbors interpolant (*nearest* option  in ``interpolate_netcdf4.py``).

Each cell to interpolate gets its value from the nearest reference cell.

This method is just an interphase for the `scipy.interpolate.NearestNDInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html>`_.

Parameters
++++++++++

* ``--rescale`` (flag): Rescales the points to the unit cube before performing the interpolation.

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

.. figure:: images/linear.png
    :width: 500
    :align: center

    Example dataset interpolated using the Linear interpolant (*linear* option in ``interpolate_netcdf4.py``).

Computes a linear interpolant by creating a 2D Delaunay triangulation using the reference data points.
Upon a given query point, it searches in which of the triangle in the XY plane it falls, and computes a barycentric interpolation of the elevation using the reference values at the vertices of the triangle.

This method is just an interphase for the `scipy.interpolate.LinearNDInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html>`_.

Parameters
++++++++++

* ``--rescale`` (flag): Rescales the points to the unit cube before performing the interpolation.
* ``--fill_value`` (float): Value used to fill in for requested points outside of the convex hull of the input points. If not provided, the default is NaN.

Suitable for
++++++++++++

* Quick large-area interpolation.

Advantages
++++++++++

* Fast classical interpolation method applicable to large areas.

Disadvantages
+++++++++++++

* May produce artifacts if samples' density vary rapidly, or if the scattered samples are not uniformly distributed over the inpainting area (see figure above).
* Does not "extrapolate" in query locations outside of the convex hull of the reference data.

Cubic
-----

.. figure:: images/cubic.png
    :width: 500
    :align: center

    Example dataset interpolated using the Cubic interpolant (*cubic* option in ``interpolate_netcdf4.py``).

As in the *linear* method, it creates a 2D Delaunay triangulation using the reference data points and query points are
interpolated within the triangle where they fall in the XY plane. However, as opposed to using a linear barycentric
interpolation within the triangle, it uses a piecewise cubic interpolating Bezier polynomial.

This method is just an interphase for the `scipy.interpolate.CloughTocher2DInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html>`_.

Parameters
++++++++++

* ``--rescale`` (flag): Rescales the points to the unit cube before performing the interpolation.
* ``--fill_value`` (float): Value used to fill in for requested points outside of the convex hull of the input points. If not provided, the default is NaN.
* ``--tolerance`` (float): Absolute/relative tolerance for gradient estimation.
* ``--max_iters`` (int): Maximum number of iterations in gradient estimation.

Suitable for
++++++++++++

* Quick large-area interpolation.

Advantages
++++++++++

* Provides a smoother interpolation than the *linear* method at a similar computational cost.

Disadvantages
+++++++++++++

* May produce artifacts if samples' density vary rapidly, or if the scattered samples are not uniformly distributed over the inpainting area (see figure above).
* Does not "extrapolate" in query locations outside of the convex hull of the reference data.

.. _rbf_interpolant:

Radial Basis Functions
----------------------

A Radial Basis Function (RBF) is a function whose value depends only on the distance between the input and some fixed point. The basic idea of a RBF interpolator is to construct an interpolant of the data using a summation of several RBF centered at the input reference data points. The formal definition is the following:

.. math:: s(x) = p(x) + \sum^{N}_{i=1} \lambda_i \phi(|x-x_i|)

Where :math:`\phi(|x-x_i|)` is a given radial basis function :math:`\phi` centered at a known/reference data point :math:`x_i`, :math:`p(x)` is a polynomial of small degree evaluated at point :math:`x`, and :math:`\lambda_i` is a scalar weight.

Thus, basically, we have a polynomial (1st term) capturing the main trend of the data, and the summation of weighted RBFs (2nd term).
Therefore, the unknowns of this interpolant are mainly the few terms of the polynomial :math:`p(x)` and the :math:`\lambda_i` weight of each RBF. These unknowns can be solved using a linear system of equations. In matrix form, this corresponds to:

.. math::
    A = \left( \begin{matrix}
                A & P \\
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
resources required for solving and/or evaluating the interpolant is too large. This is the reason why there is no figure
showing the result in this section: even for a small dataset as the one we are using, **it is not feasible to compute the
interpolant in a reasonable amount of time and resources**.

However, it has the nice feature of allowing some "tunning" of the properties of the interpolating surface via the RBF type that we choose.

The RBF types available in this package are listed in the following. Note that some of these definitions have an :math:`\epsilon` parameter modifying their "shape":

* Polyharmonic:
    * linear: :math:`\phi(r) = r`
    * thin plate spline: :math:`\phi(r) = r^2 log(r)`. It provides a biharmonic interpolant.
    * cubic: :math:`\phi(r) = r^3`
    * quintic: :math:`\phi(r) = r^5`

* Infinitely smooth:
    * gaussian: :math:`\phi(r) = e^{-(\epsilon r)^2}`
    * multiquadric: :math:`\phi(r) = \sqrt{1+(\epsilon r)^2}`

* Other:
    * tension spline: :math:`\phi(r) = -\frac{1}{2 \pi \epsilon^2}(log(\frac{r\epsilon}{2} + C_e + K_0(r\epsilon))`, being :math:`C_e` the Euler constant and :math:`K_0` the modified Bessel function (same as in [MITAS1988]_, equation 50).
    * regularized spline: :math:`\phi(r) = \frac{1}{s\pi} \left( \frac{r^4}{4} \left[ log(\frac{r}{2\pi}) + C_e - 1 \right] + \epsilon^2 \left[K_0(\frac{r}{\epsilon}) + C_e + log(\frac{r}{2\pi}) \right] \right)` (same as in [MITAS1988]_, equation 56).

.. [MITAS1988] Mitas, L., and H. Mitasova. 1988. General Variational Approach to the Interpolation Problem. Comput. Math. Applic. Vol. 16. No. 12. pp. 983–992. Great Britain.

Parameters
++++++++++

* ``--query_block_size`` (int): Apply the interpolant using maximum this number of points at a time to avoid large memory consumption.
* ``--rbf_distance_type`` (str): Distance type. Available: euclidean (default), haversine, vincenty.
* ``--rbf_type`` (str): RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland.
* ``--rbf_epsilon`` (float): Epsilon parameter of the RBF. Its meaning changes with each RBF type, but mainly modifies the "shape" of the RBF. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland.
* ``--rbf_regularization`` (float): Regularization scalar to use while creating the RBF interpolant (optional).
* ``--rbf_polynomial_degree`` (float): Degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic).

Suitable for
++++++++++++

* Best approximation quality for the interpolant.
* Small datasets. They can be small in the number of input reference points, and large in the number of query points (huge scattered data).

Advantages
++++++++++

* Allows tuning the properties of the interpolating surface by changing the RBF type and :math:`\epsilon` parameter.

Disadvantages
+++++++++++++

* Depending on the input data and the selected RBF type, the resulting interpolant surface may **overshoot** the input data (minimum and maximum elevation values may be outside the range of the input data).

Partition of Unity Radial Basis Functions
-----------------------------------------

.. figure:: images/purbf.png
    :width: 500
    :align: center

    Example dataset interpolated using the Partition of Unity Radial Basis Functions interpolant (*purbf* option in ``interpolate_netcdf4.py``).

Based on the low applicability of the original definition of the RBF interpolant, the Partition of Unity Radial Basis
Functions (*purbf*) is an attempt to lower as much as possible the memory and computational requirements of the RBF interpolator.

The Partition of Unity Method (PUM) divides the global domain into smaller overlapping subdomains. In each of these
subdomains, a RBF interpolant is computed using the formulation presented in :ref:`rbf_interpolant`. Then, when evaluating
a query location, the contribution of several neighboring RBF interpolations are *blended* together in order to get the final value.

More precisely, we enforce a quadtree decomposition. In the following figure we can see an example of this decomposition:

.. figure:: images/pu_rbf_domain_decomposition.png
    :width: 400
    :align: center

    An example of the decomposition in *purbf* method. Reference data points are marked as black dots, the quadtree
    decomposition is shown using squares, and the domain of each local RBF corresponding to each square is shown with a colored circle.

Each cell in the quadtree define a local RBF interpolant and its area of influence. Note how the different areas overlap
between them (a condition for continuity) and how the area of influence of each local interpolant adapts to the complexity of the data
(larger regions in more sparse areas, and smaller regions in denser ones). Finally, since the extent of local RBF is limited,
we also ensure that at least one local interpolant covers all the data within the possible query space (i.e., it covers the extent of the input grid).

The PU interpolant preserves the local approximation order for the global fit. Therefore, large RBF interpolants can be
computed by solving small interpolation problems and then combining them together with the global PU.

Parameters
++++++++++

* ``--query_block_size`` (int): apply the interpolant using maximum this number of points at a time to avoid large memory consumption.
* ``--rbf_distance_type`` (str): distance type. Available: euclidean (default), haversine, vincenty.
* ``--rbf_type`` (str): RBF type. Available: linear, cubic, quintic, gaussian, multiquadric, green, regularized, tension, thinplate, wendland.
* ``--rbf_epsilon`` (float): epsilon parameter of the RBF. Its meaning changes with each RBF type, but mainly modifies the "shape" of the RBF. Required just for the following RBF types: gaussian, multiquadric, regularized, tension, wendland.
* ``--rbf_regularization`` (float): regularization scalar to use while creating the RBF interpolant (optional).
* ``--rbf_polynomial_degree`` (float): degree of the global polynomial fit used in the RBF formulation. Valid: -1 (no polynomial fit), 0 (constant), 1 (linear), 2 (quadric), 3 (cubic).
* ``--pu_overlap`` (float): overlap factor between circles in neighboring sub-domains in the partition. The radius of a QuadTree cell, computed as half its diagonal, is enlarged by this factor.
* ``--pu_min_point_in_cell`` (int): minimum number of points in a QuadTree cell.
* ``--pu_min_cell_size_percent`` (float): minimum cell size, specified as a percentage [0..1] of the max(width, height) of the query domain.
* ``--pu_overlap_increment`` (float): if, after creating the QuadTree, a cell contains less than pu_min_point_in_cell, the radius will be iteratively incremented until this condition is satisfied. This parameter specifies how much the radius of a cell increments at each iteration.

Suitable for
++++++++++++

* Datasets for which the basic RBF interpolator required too much memory and computational resources.

Advantages
++++++++++

* Tunnable output: as in the RBF interpolator, changing the base RBF will change the shape/properties of the output interpolated surface.
* Preferrable in cases where the number of reference data points is far smaller than the number of points to interpolate.

Disadvantages
+++++++++++++

* While compared to the pure RBF, reduction in computational requirements is huge, it may not be sufficient for processing large datasets (i.e., it will still be slower to compute than other options in this package).

Adaptive Multi-grid Solver
--------------------------

The Adaptive Multi-grid Solver (AMS) method implements the `PointInterpolant` tool from the `AMS (a.k.a. PoissonRecon) project <https://github.com/mkazhdan/PoissonRecon>`_, via our `Python interface <https://github.com/coronis-computing/py_ams_point_interpolant>`_.

This method solves for coefficients by minimizing an energy of the form :math:`E = D + S`, where the *data fitting* term :math:`D` tries to match values at sample points and the *smoothness regularization* :math:`S` penalizes roughness via Laplacian and bi-Laplacian terms. In this sense, the smoothness term resembles that in the CCST method below, allowing to also set relative weights to both **Laplacian** and **Bi-Laplacian** terms similarly to what the **tension** parameter is doing in :ref:`CCST <ccst_inpainter>`.

Parameters
++++++++++

* ``--depth``: This integer is the maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x ... Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound. (default: 8).
* ``--degree``: Degree of the B-spline that is to be used to define the finite elements system. Larger degrees support higher order approximations, but come at the cost of denser system matrices (incurring a cost in both space and time). (default: 2).
* ``--solve_depth``: the depth up to which the solver will solve the numerical system. It will still show the results at the finest resolution, but no additional high-frequency data will be introduced at the finest resolutions. It could also be the case that aliasing that occurs at the coarser resolutions will not get corrected. (default = -1, i.e., --depth).
* ``--full_depth``: The depth up to which the octree is completely refined, i.e. a regular grid (default: 5).
* ``--base_depth``: The coarsest depth at which the system will be solved over an octree. (At coarser levels it will be solved using a standard MG solver, with multiple V-Cycles, defined over a regular grid.) As such, the assumption is that BaseDepth<=FullDepth (default = -1, i.e., not used).
* ``--boundary_type``: Boundary type (default: free, available: free, dirichlet, neumann).
* ``--iters``: The number of Gauss-Seidel relaxations to be performed at every level of the hierarchy (default: 8).
* ``--base_v_cycles``: coarse MG solver v-cycles (default: 4).
* ``--max_memory_gb``: Maximum memory to use in GB (default: 0, i.e., no limit).
* ``--parallel_type``: Parallel mode (default: openmp, available: openmp, threads, none").
* ``--parallel_schedule``: Parallel schedule (default: static, available: static, dynamic).
* ``--parallel_thread_chunk_size``: Parallel thread chunk size (default: 128).
* ``--value_weight``: Importance that interpolation of the samples' values is given in the fitting of the function (default: 1000.0).
* ``--gradient_weight``: Importance that interpolation of the samples' gradients is given in the fitting of the function (default: 1.0).
* ``--scale``: The ratio between the diameter of the cube used for reconstruction and the diameter of the samples' bounding cube. (default: 1.1).
* ``--width``: Target width of the finest level octree cells. This parameter is ignored if the --depth is also specified. (default: 0.0, i.e., ignore and use --depth).
* ``--cg_accuracy``: Conjugate Gradient solver accuracy (default: 1e-3).
* ``--iso``: Iso-value (default=0.0).
* ``--laplacian_weight``: Importance that Laplacian regularization is given in the fitting of the function (default: 0.0).
* ``--bi_laplacian_weight``: Importance that bi-Laplacian regularization is given in the fitting of the function (default: 1.0).
* ``--show_performance``: Show performance statistics (default: false).
* ``--show_residual``: Show residuals (default: false).
* ``--exact_interpolation``: Use exact interpolation (default: false).
* ``--ams_verbose``: Verbose mode for AMS, will print information during the creation of the interpolant (default: false).
* ``--transform_file``: Transform file (default: none).

Suitable for
++++++++++++

* Very large datasets.

Advantages
++++++++++

* Excellent computation speed, it is the fastest method in the package by far.

Disadvantages
+++++++++++++

* The interpolations it provides for large areas of missing data may not be the best (although you may try tunning to non-default parameters to improve this).

.. _pde_inpainters:

PDE-based Inpainting Interpolators
**********************************

Our heightmaps are bivariate functions of the form :math:`u(x, y) = z`, where x/y are the coordinates in a plane and z the corresponding elevation value.

A simple way of defining the interpolant is to define the properties that the "interpolating surface" :math:`f(u)` must satisfy at interpolated areas using Partial Differential Equations (PDEs).

Once defined a given PDE, we can solve it using finite differences. In this project we implement two solvers described in the :ref:`pde_iterative_solver` and :ref:`pde_direct_solver` sections. 
Moreover, both solvers can benefit from both the :ref:`different initializers <inpainting_initializer>` and the :ref:`Multi-grid solver <inpainting_mgs>` to speed up their execution for large datasets. 
Note that the parameters listed in all these sections are common to all the methods, and that method-specific parameters are defined in their corresponding sections.

.. _pde_iterative_solver:

Iterative Solver
----------------

It uses the gridded nature of the problem to solve it in a gradient-descent manner, where:

.. math::
    f(u)_{t+1} = u_t - \phi*\nabla(f(u_t))

Being the subindex :math:`t` the iteration index, :math:`\nabla(f(u_t))` the PDE or the *gradient* that we need to follow, \phi the size of the update step at each iteration. Given a properly small :math:`\phi`, we can iterate the equation above to *steady state* (i.e., no change) in order to solve for the functional.

Using discretized differential stencils, we can work directly on the input cell grid, and evolve the previous equation using just convolutions or use direct solvers (for those PDEs not containing non-linear terms).

Parameters
++++++++++

All the methods in this section may use the iterative PDE solver. Therefore, there is a set of parameters that are common to all the methods if using this solver:

* ``--update_step_size`` (float): gradient descent step size. A default is provided by each method. However, depending on the problem, you could tune it to a higher value to speed-up convergence (but beware of overshooting and missing the minimum!).
* ``--term_criteria`` (string): the termination criteria to use. Available:

    - ``relative``: stop if the relative change between the inpainted elevations in the current and a previous step is smaller than the value in ``--term_thres``.
    - ``absolute``: stop if all cells absolute change between the inpainted elevations in the current and a previous step is smaller than the value in ``--term_thres``.
    - ``absolute_percent`` (default): stop if all cells absolute change between the inpainted elevations in the current and a previous step is smaller than the value in ``--term_thres`` multiplied by the absolute range of depths in the dataset (i.e., the absolute value is range_depths * absolute_change_percent).

* ``--term_thres`` (float): stop the optimization when the energy descent between iterations is less than this value. Its meaning depends on ``--term_criteria``.
* ``--term_check_iters`` (int): since checking for the termination criteria may be costly, we just perform the check for the relative change between iterations of the optimizer every this number of iterations.
* ``--max_iters`` (int): maximum number of iterations for the optimizer (will end the optimization even if there is no convergence on the minimization).
* ``--relaxation`` (float): over-relaxation parameter. *This paramter  is still under testing, use with care*.
* ``--print_progress`` (bool): print information about the progress of the optimization on screen.
* ``--print_progress_iters`` (int): If print_progress==True, the information will be printed every this number of iterations.
* ``--convolver`` (str): the convolver used for all the convolutions required by the solver. Available: 'opencv' (default),'scipy-signal', 'scipy-ndimage', 'masked', 'masked-parallel'. This parameter is for testing purposes, the 'opencv' convolver was proven the fastest of the options in all cases, so you should leave it in its default value.
* ``--debug_dir`` (str): if set, a debug directory in the specified path will be created, and intermediate steps will be rendered as images. Useful to create a video of the evolution of the solver.

Advantages
++++++++++

* Memory efficient: the solution is found on the same grid that is already loaded in memory.
* Can be used to solve all the PDEs in this package.

Disadvantages
+++++++++++++

* Slower than the direct solver, use the proper initializer and multi-grid approach to speed up for large datasets.

.. _pde_direct_solver:

Direct Solver
-------------

The :ref:`pde_iterative_solver` provides a nice memory footprint (i.e., it solves on the grid, so it does not require more memory than the grid itself). However, it may require thousands or millions of iterations to converge, since :math:`\phi*\nabla(f(u_t))` must be small for stability.

The Direct solver reframes it as an algebra problem. Since :math:`\phi*\nabla(f(u_t))` (the **step function**) is a linear combination of neighbouring pixels (a convolution stencil), the condition :math:`\phi*\nabla(f(u_t)) = 0` at every unknown pixel is just a system of linear equations:

.. math::
    A x = b

where:

* :math:`x` is the vector of unknown pixel values we want to find
* Each row of :math:`A` encodes how one unknown pixel's stencil depends on its unknown neighbours
* :math:`b` encodes the contribution from the known (boundary) pixels

Solving this system gives the converged solution in one shot, bypassing all the iterative steps. However, the main limitation of this solver is that it only works when :math:`\phi*\nabla(f(u_t))` is truly linear in :math:`f(u_t)`. For methods where the stencil coefficients themselves depend on f (as in :ref:`TV <tv_inpainter>` or :ref:`AMLE <amle_inpainter>` below), there is no fixed matrix :math:`A` and the approach is invalid. Therefore, this solver is only available for the :ref:`harmonic <harmonic_inpainter>` and :ref:`CCST <ccst_inpainter>` inpainters.

Parameters
++++++++++

This solver shares some of the parameters seen for the :ref:`pde_iterative_solver`, but their meaning may differ:

* ``--cg_term_thres`` (float): the conjugate gradient solver when the energy is less than this value. Its meaning depends on ``--term_criteria``.
* ``--max_iters`` (int): maximum number of iterations passed to the Conjugate Gradient solver used to solve the linear systems.
* ``--convolver`` (str): the convolver used for all the convolutions required by the solver. Available: 'opencv' (default),'scipy-signal', 'scipy-ndimage', 'masked', 'masked-parallel'. This parameter is for testing purposes, the 'opencv' convolver was proven the fastest of the options in all cases, so you should leave it in its default value.
* ``--print_progress_iters`` (int): If print_progress==True, the information will be printed every this number of iterations.
* ``--debug_dir`` (str): if set, a debug directory in the specified path will be created, and debug results will be generated (testing purposes).

Advantages
++++++++++

* Faster than the iterative solver.

Disadvantages
+++++++++++++

* Only applicable to :ref:`harmonic <harmonic_inpainter>` and :ref:`CCST <ccst_inpainter>` inpainters (does not allow non-linearities on the stencil equation).
* For large empty areas to fill, the initialization may dominate and not change much after solving.

.. _inpainting_initializer:

Initializer
-----------

The convergence speed of both the gradient descent optimization or the direct solver on the inpainted area is highly dependant on the initial values.
It is not the same trying to evolve the solution using the optimization starting from a very vague solution (e.g. all unknowns initial value is zero)
than starting from initial values closer to the solution. In this direction, we provide the following initializers, specified via the ``--init_with`` parameter:

* *zeros*: init unknown values with zeros. This is the worst initializer, just kept here for comparison purposes with the rest.
* *mean*: init unknown values with a constant equal to the mean of the reference elevation values.
* *nearest*: use the nearest interpolant to initialize unknown values.
* *linear*: use the linear interpolant to initialize unknown values. Since this interpolant is just defined over the convex hull of the input data, data outside it will get a constant value equal to the mean of the reference elevation values.
* *cubic*: use the cubic interpolant to initialize unknown values. Same as in *linear*, it will get the mean value of reference value outside the convex hull of the reference data points.
* *harmonic*: uses the harmonic inpainter to fill the missing data. Note that, while being the fastest of the inpainter methods, this involves solving another gradient descent optimization, so depending on the complexity of the data it may be very slow.

.. _inpainting_mgs:

Multi-Grid Solver
-----------------

We also provide an optional Multi-Grid Solver (MGS). Basically, instead of solving the optimization problem at the full resolution grid directly, it will do it in a multi-resolution way.

The MGS starts building a pyramid of different levels of resolution from the original grid, where each level of the pyramid contains a halved resolution version of the previous one:

.. figure:: images/mgs_pyramid.png
    :width: 300
    :align: center

    Schematic of the multi-resolution pyramid created by the Multi-Grid Solver. The original grid (bottom of the pyramid) is halved in resolution recursively to get lower resolution versions of the problem. Then, starting from the top of the pyramid, the inpainting problem is solved in a lower resolution version, and upscaled and propagated to the next (higher resolution) level of the pyramid as initial guess.

Then, starting from the coarser level, we solve the inpainting problem there, and use that solution to initialize the solver in the next (higher resolution) level of the pyramid.

Therefore, we use upscaled versions of the problem solved at coarser resolutions to initialize the inpainting problem at higher resolutions.
In this way, the initial values of the optimization at each level of the pyramid are closer to the final solution, decreasing like this the number of iterations required for convergence.

Parameters
++++++++++

* ``--mgs_levels`` (int): number of levels of detail to use in the Mult-Grid Solver (MGS, see :ref:`inpainting_mgs`). Setting it to 1 deactivates the MGS.
* ``--mgs_min_res`` (int): minimum resolution (width or height) allowed for a level in the MGS. If the level of detail in the pyramid gets to a value lower than this, the pyramid construction will stop.

Note that, when using the MGS, the ``--init_with`` parameter (corresponding to the :ref:`inpainting_initializer`) will just affect the initialization of the lowest-resolution level of the pyramid.

.. _harmonic_inpainter:

Harmonic Inpainter
------------------

.. figure:: images/harmonic.png
    :width: 500
    :align: center

    Example dataset interpolated using the Harmonic inpainter (*harmonic* option in ``interpolate_netcdf4.py``).

An harmonic surface is a twice differentiable function satisfying the Laplace equation:

.. math::
    \nabla(f(u_t)) = \nabla^2 u_t = 0

This method has many analogies:

* It can be seen as an "isotropic diffusion" of the elevation values at the borders surrounding the missing data towards the area to interpolate.
* Its evolution follows the `heat diffusion equation <https://en.wikipedia.org/wiki/Heat_equation>`_.
* It minimizes the Sobolev norm on the grid, constrained to the input reference data.
* The interpolated surface is a "minimum energy surface", and many times it is described as the "shape a film of soap would take if laid over the data points".

Parameters
++++++++++

No specific parameters for this method.

Suitable for
++++++++++++

* Filling large gaps smoothly without overshooting the input data.

Advantages
++++++++++

* Fastest of the inpainting methods.
* It will never overshoot the data (minimum and maximum elevation values never below/over the reference ones).

Disadvantages
+++++++++++++

* Does not work well with sparsely sampled data: isolated data points will not contribute much to the interpolation.

.. _tv_inpainter:

Total Variation (TV) Inpainter
------------------------------

.. figure:: images/tv.png
    :width: 500
    :align: center

    Example dataset interpolated using the Total Variation inpainter (*tv* option in ``interpolate_netcdf4.py``).

Minimizes the Total Variation formula within the area to inpaint:

.. math::
    \nabla(f(u_t)) = -div N_{\epsilon}(\nabla u_t)

Where:

.. math::
    N_{\epsilon}(u) = \frac{u}{\sqrt{\left\| u \right\|^2 + \epsilon^2}}

Intuitively, it tends to preserve/continue high gradients better than *harmonic*, since the evolution of the optimizer can be considered a type of
`anisotropic diffusion <https://en.wikipedia.org/wiki/Anisotropic_diffusion>`_.

However, it will not take into account isolated points, and should only be used for filling gaps with no data fully surrounded with reference data.

Parameters
++++++++++

* ``--epsilon``: the :math:`\epsilon` parameter in the formula above. It is just a small value used in the normalization factor :math:`N_{epsilon}(u)` so that the denominator is never zero.

Suitable for
++++++++++++

* Filling continous gaps of data (i.e., not suitable for scattered data interpolation).

Advantages
++++++++++

* Provides similar results to the *harmonic* inpainter, but tends to better preserve level lines of the surroundings.

Disadvantages
+++++++++++++

* Does not work well with sparsely sampled data: isolated data points will not contribute much to the interpolation.

.. _ccst_inpainter:

Continous Curvature Splines in Tension (CCST) Inpainter
-------------------------------------------------------

.. figure:: images/ccst.png
    :width: 500
    :align: center

    Example dataset interpolated using the Continous Curvature Splines in Tension inpainter (*ccst* option in ``interpolate_netcdf4.py``).

Implements the method in [Smith90]_. The PDE guiding this interpolant is the following:

.. math::
    \nabla(f(u_t)) = (1-t)\nabla^4 u_t - t \nabla^2 u_t = 0
    :label: eq_ccst

If we take a look to equation :eq:`eq_ccst`, we will identify that :math:`\nabla^2 u_t` is the harmonic equation (same as in :ref:`harmonic_inpainter`).
Also, in the other term, we find :math:`\nabla^4 u_t = \nabla^2\nabla^2 u_t`, the "harmonic of the harmonic", that is, the biharmonic surface. And, in both terms, they are affected by a constant :math:`t`.

The *tension* parameter :math:`t` allows tuning the influence of an harmonic and a biharmonic surface in the final result. Therefore:

* :math:`t = 0` equals a biharmonic surface.
* :math:`t = 1` equals an harmonic surface (same result as in :ref:`harmonic_inpainter`!).
* A value of :math:`t` between 0 and 1 is a mixture of both harmonic/biharmonic interpolants.

In a nutshell, if we chop off the peak of a mountain at a given altitude, and we try to interpolate it using this method,
:math:`t=0` would probably reconstruct the peak of the mountain (note that this means that it will **overshoot** the input data),
while :math:`t=1` would reconstruct a flat area. A :math:`t` between 0 and 1 would be a mix of both results.

Note that this is a re-implementation/variant of the method in [Smith90]_, which in turn is the method implemented in `GMT surface <http://gmt.soest.hawaii.edu/doc/latest/surface.html>`_.

.. [Smith90] Smith, W. H. F, and P. Wessel, 1990, Gridding with continuous curvature splines in tension, Geophysics, 55, 293-305.

Parameters
++++++++++

* ``--tension``: the parameter corresponding to the :math:`t` constant in equation :eq:`eq_ccst`, responsible for tuning the mixture between an harmonic and a biharmonic interpolant.

Suitable for
++++++++++++

* Getting a higher order interpolating surface, similar to what we achieve with the *purbf* method.
* Achieving **the same** results than using the *purbf* with a thin plate spline RBF (tension == 0) for datasets where the number of reference data points is much larger than the number of points to interpolate with smaller memory requirements and computational cost.

Advantages
++++++++++

* It provides an "easy to tune" mix of an harmonic and a biharmonic interpolant.

Disadvantages
+++++++++++++

* Slower execution time than other inpainters.
* Depending on the parameters, it may overshoot the data.

.. _amle_inpainter:

Absolutely Minimizing Lipschitz Extension (AMLE) Inpainter
----------------------------------------------------------

.. figure:: images/amle.png
    :width: 500
    :align: center

    Example dataset interpolated using the Absolutely Minimizing Lipschitz Extension inpainter (*amle* option in ``interpolate_netcdf4.py``).

Implements the method in [Almansa02]_. Following the notation of the original reference, The PDE guiding this interpolant is the following:

.. math::
    \nabla(f(u_t)) = D^2u_t \left( \frac{Du_t}{|Du_t|}, \frac{Du_t}{|Du_t|} \right)

Where :math:`Du` denotes the gradient of :math:`u`.

The main effort of the AMLE model is to "avoid oscillations", i.e., avoid the interpolated elevation to overshoot the reference values (min and max elevation value do not change).
Also, it handles "isolated points" in the reference data.

.. [Almansa02] Andrés Almansa, Frédéric Cao, Yann Gousseau, and Bernard Rougé.
            Interpolation of Digital Elevation Models Using AMLE and Related
            Methods. IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 40,
            NO. 2, FEBRUARY 2002

Parameters
++++++++++

This inpainter only depends on the parameters of the solver (see .

Suitable for
++++++++++++

* Interpolating gaps in terrain data using a better interpolant, but trying not to overshoot the original data.
* Scattered data: this is the only approach that always takes into account scattered data properly (*ccst* with a tension close to 1 also does, but not so well if tension approaches 0...).

Advantages
++++++++++

* It is the only inpainter method in this package that was originally devised for interpolating heightmaps without overshooting the data.
* Contribution of isolated points is properly propagated within the area to interpolate.

Disadvantages
+++++++++++++

* Slower execution time than other inpainters.
* Contrary to the other inpainters, for which the default values of ``--update_step_size`` parameter should work in any condition, there might be cases where the default ``--update_step_size`` may lead to no convergence. If you see that the solver does not converge for your data (it gets to the maximum number of iterations set in ``--max_iters`` parameter), try lowering it a bit an re-execute.


.. _other_inpainters:

Other Inpainters
****************

Since one of the dependencies we use is `OpenCV <https://opencv.org/>`_, and this library has some inpainting methods already implemented,
we created interphases for using them on our heightmap interpolation problem. Note that these methods are typically used
for closing small, thin gaps, as the ones you can see in the examples of the `OpenCV documentation <https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_photo/py_inpainting/py_inpainting.html>`_.

OpenCV's Telea
--------------

.. figure:: images/telea.png
    :width: 500
    :align: center

    Example dataset interpolated using the OpenCV's Telea inpainter (*telea* option in ``interpolate_netcdf4.py``).

The Telea variant of `OpenCV's inpaint function <https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085>`_.

Parameters
++++++++++

Only the ``--radius`` integer parameter, corresponding to the one with the same name on the official `OpenCV documentation <https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085>`_.

Suitable for
++++++++++++

* Interpolating "thin" continuous missing data parts fast.

Advantages
++++++++++

* Faster than PDE-based inpainters.

Disadvantages
+++++++++++++

* Does not consider scattered data at all.

OpenCV's Navier-Stokes
----------------------

.. figure:: images/navier-stokes.png
    :width: 500
    :align: center

    Example dataset interpolated using the OpenCV's Navier-Stokes inpainter (*navier-stokes* option in ``interpolate_netcdf4.py``).

The Navier-Stokes variant of `OpenCV's inpaint function <https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085>`_.

Parameters
++++++++++

Only the ``--radius`` integer parameter, corresponding to the one with the same name on the official `OpenCV documentation <https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085>`_.

Suitable for
++++++++++++

* Interpolating "thin" continuous missing data parts fast.

Advantages
++++++++++

* Faster than PDE-based inpainters.

Disadvantages
+++++++++++++

* Does not consider scattered data at all.