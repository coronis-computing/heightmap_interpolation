Which method should I use?
==========================

As a summary of section :ref:`methods`, we present here a reference on suggested methods to use depending on different situations. Note that, from the set of methods recommended, choosing one or the other depend on your speed/computational requirements and the quality/properties of the interpolating surface.

Depending on the sampling of the reference data
***********************************************

Case:
    My data is located with no specific order between their respective locations within the input grid, and at a sampling much lower
    than the resolution of the grid.

Suggested methods:
    Any of the methods in :ref:`scattered_methods`.

Case:
    My data is defined at continous regions, but with large (also continous) missing regions that you need to fill

Suggested methods:
    Any of the methods in :ref:`pde_inpainters`.

Case:
    I have a mixture of both cases above: some areas are sparsely sampled, while some others are densely sampled.

Suggested methods:
    This case, common when mixing different datasets in the same grid, is the most complex one. In this case, the method to choose would depend on the speed requirements, and in the :


Case:
    My reference data, while scattered, is close to the resolution of the grid. Therefore, the cells missing between known reference data points are very few.

Suggested methods:
    Since the cells to interpolate are almost next to the data to interpolate, it does not make sense to apply a fully

Depending on the desired interpolation quality
**********************************************




Depending on the computation time
*********************************

