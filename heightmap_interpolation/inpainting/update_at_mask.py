import numpy as np
from numba import double, boolean, njit, prange


#@njit(parallel=True) # DevNote: The overhead of parallelizing does not seem to compensate...
@njit()
def update_at_mask(image, f, mask):
    M, N = image.shape
    result = np.copy(image)
    for i in prange(0, M):
        for j in prange(0, N):
            if mask[i, j]:
                result[i, j] = f[i, j]
    return result