Which method should I use?
==========================

As a summary of section :ref:`methods`, we present here a reference on suggested methods to use depending on different situations. Note that, from the set of methods recommended, choosing one or the other depends on your speed/computational requirements and the quality/properties of the interpolating surface, so read both sections in this page before deciding.

Depending on the sampling of the reference data
***********************************************

Case:
    The reference data is located with no specific order between their respective locations within the input grid, and at a sampling much lower
    than the resolution of the grid in the area to interpolate.

Suggested methods:
    Basically, this is the definition of scattered data, so any of the methods in :ref:`scattered_methods` is good for you.

Case:
    The reference data is defined at continous regions, but with large (also continous) missing regions that the interpolation must fill.

Suggested methods:
    Any of the methods in :ref:`pde_inpainters` and :ref:`other_inpainters` will work well.

Case:
    I have a mixture of both cases above: some areas are sparsely sampled, while some others are densely sampled.

Suggested methods:
    This case, common when mixing different datasets in the same grid, is the most complex one. In this case, the method
    to choose would depend on its complexity, see the section below :ref:`choice_depending_on_complexity`.

Case:
    My reference data, while scattered, is close to the resolution of the grid. Therefore, the cells missing between known reference data points are very few.

Suggested methods:
    Since the cells to interpolate are almost next to (and surrounded by) the data to interpolate, it does not make sense to apply a complex interpolator.
    Some of the fast scattered data interpolators, namely, *nearest*, *linear* or *cubic*, should provide good enough results.
    Depending on the complexity of your problem, if you are not satisfyed with the results provided by any of these methods,
    you can go for *purbf* from the Scattered data interpolants, or the *ccst* and *amle* from the inpainters (these are the only two allowing inpainting using isolated points).
    However, keep in mind that the processing will take far more time and computational resources in those cases.

.. _choice_depending_on_complexity:

Depending on the complexity of the problem
******************************************

The complexity of the problem depends on the number cells.
For small datasets, meaning grids of resolutions in the order of less than 500 x 500, all interpolation methods should provide
results within a reasonable time (in the order of few minutes for the most costly) in modern PCs. However, for **large datasets**
it is important to know that some methods will be faster in some cases.

For :ref:`scattered_methods`, we have to take into account the ratio between reference cells and cells to interpolate in the problem.
As explained in :ref:`methods`, these algorithms work in two steps: creating the interpolant from reference data, and applying the interpolant to the
query points. From these two steps, the costly one is the creation of the interpolant. Therefore, if you need to fill large areas, but you have a
much smaller number of reference cells in comparison, these methods are a good option.

However, that is not exactly the case for :ref:`pde_inpainters`, since our solver extensively uses **convolutions**. Convolutions
are faster if applied to a square (full) grid, regardless of whether we are using all the cells or not in our problem. Therefore,
if you use an inpainter in a large grid, the ratio of reference/to interpolate cells does not matter. This is why these methods
are recommended when the number of reference cells is larger than the ones to interpolate.

**Important**: remember that, if your dataset is too large but interpolation just matters in smaller areas within the grid, you can use the
``--areas`` flag to specify those locations. In that case, **interpolation complexity has to be calculated based on those areas**. For inpainters,
the "square grid" mentioned above is adapted to the bounding box of the specified area.
