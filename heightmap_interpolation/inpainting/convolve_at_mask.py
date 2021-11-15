import numpy as np
from numba import double, boolean, njit, prange, cuda, guvectorize


#@njit(double[:, :](double[:, :], boolean[:, :], double[:, :]), parallel=True, nogil=True, fastmath=True)
def convolve_at_mask(image, mask, filt):
    """Naive 2D convolution, but applied just at the points defined by the mask with the same behaviour at borders as OpenCV's filter2D default behaviour (BORDER_REFLECT_101)."""
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    result = np.copy(image)
    for i in prange(0, M):
        for j in prange(0, N):
            if mask[i, j]:
                num = 0.0
                for ii in prange(Mf):
                    for jj in prange(Nf):
                        img_ind_i = abs(i-Mf2+ii)
                        if img_ind_i > M-1:
                            mod_i = img_ind_i % (M-1)
                            img_ind_i = M-1-mod_i
                        img_ind_j = abs(j-Nf2+jj)
                        if img_ind_j > N-1:
                            mod_j = img_ind_j % (N-1)
                            img_ind_j = N-1-mod_j
                        num += (filt[Mf-1-ii, Nf-1-jj] * image[img_ind_i, img_ind_j])
                result[i, j] = num
    # for i in prange(Mf2, M - Mf2):
    #     for j in prange(Nf2, N - Nf2):
    #         if mask[i, j]:
    #             num = 0.0
    #             for ii in prange(Mf):
    #                 for jj in prange(Nf):
    #                     num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii, j-Nf2+jj])
    #             result[i, j] = num
    return result

@njit(parallel=False, nogil=True, fastmath=True)
def nb_convolve_at_mask(image, mask, filt):
    """Fast 2D convolution using Numba, but applied just at the points defined by the mask with the same behaviour at borders as OpenCV's filter2D default behaviour (BORDER_REFLECT_101)."""
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    result = np.copy(image)
    for i in prange(0, M):
        for j in prange(0, N):
            if mask[i, j]:
                num = 0.0
                for ii in prange(Mf):
                    for jj in prange(Nf):
                        img_ind_i = abs(i - Mf2 + ii)
                        if img_ind_i > M - 1:
                            mod_i = img_ind_i % (M - 1)
                            img_ind_i = M - 1 - mod_i
                        img_ind_j = abs(j - Nf2 + jj)
                        if img_ind_j > N - 1:
                            mod_j = img_ind_j % (N - 1)
                            img_ind_j = N - 1 - mod_j
                        num += (filt[Mf - 1 - ii, Nf - 1 - jj] * image[img_ind_i, img_ind_j])
                result[i, j] = num
    return result


@njit(parallel=False, nogil=True, fastmath=True)
def nb_convolve_at_mask_where(image, mask, filt):
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    inds_mask = np.where(mask)
    result = np.copy(image)
    for a in range(0, len(inds_mask[0])):
        i = inds_mask[0][a]
        j = inds_mask[1][a]
        num = 0
        for ii in prange(Mf):
            for jj in prange(Nf):
                img_ind_i = abs(i - Mf2 + ii)
                if img_ind_i > M - 1:
                    mod_i = img_ind_i % (M - 1)
                    img_ind_i = M - 1 - mod_i
                img_ind_j = abs(j - Nf2 + jj)
                if img_ind_j > N - 1:
                    mod_j = img_ind_j % (N - 1)
                    img_ind_j = N - 1 - mod_j
                num += (filt[Mf - 1 - ii, Nf - 1 - jj] * image[img_ind_i, img_ind_j])
        result[i, j] = num
    return result


@guvectorize(
    ["void(float32[:,:], boolean[:,:], float32[:,:], float32[:,:])",
     "void(float32[:,:], boolean[:,:], float64[:,:], float32[:,:])",
     "void(float64[:,:], boolean[:,:], float64[:,:], float64[:,:])"],
    "(n,m),(n,m),(k,k)->(n,m)",
    nopython=True)
def nb_convolve_at_mask_guvec(image, mask, filt, result):
    """Fast 2D convolution using Numba as a guvector, but applied just at the points defined by the mask with the same behaviour at borders as OpenCV's filter2D default behaviour (BORDER_REFLECT_101)."""
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    for i in prange(0, M):
        for j in prange(0, N):
            if mask[i, j]:
                num = 0.0
                for ii in prange(Mf):
                    for jj in prange(Nf):
                        img_ind_i = abs(i - Mf2 + ii)
                        if img_ind_i > M - 1:
                            mod_i = img_ind_i % (M - 1)
                            img_ind_i = M - 1 - mod_i
                        img_ind_j = abs(j - Nf2 + jj)
                        if img_ind_j > N - 1:
                            mod_j = img_ind_j % (N - 1)
                            img_ind_j = N - 1 - mod_j
                        num += (filt[Mf - 1 - ii, Nf - 1 - jj] * image[img_ind_i, img_ind_j])
                result[i, j] = num
            else:
                result[i, j] = image[i, j]


@njit(parallel=True, nogil=True, fastmath=True)
def nb_convolve_at_mask_parallel(image, mask, filt):
    """Same as nb_convolve_at_mask, but with parallelized fors (the parallelization overhead may render this method slower for small images)"""
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    result = np.copy(image)
    for i in prange(0, M):
        for j in prange(0, N):
            if mask[i, j]:
                num = 0.0
                for ii in prange(Mf):
                    for jj in prange(Nf):
                        img_ind_i = abs(i - Mf2 + ii)
                        if img_ind_i > M - 1:
                            mod_i = img_ind_i % (M - 1)
                            img_ind_i = M - 1 - mod_i
                        img_ind_j = abs(j - Nf2 + jj)
                        if img_ind_j > N - 1:
                            mod_j = img_ind_j % (N - 1)
                            img_ind_j = N - 1 - mod_j
                        num += (filt[Mf - 1 - ii, Nf - 1 - jj] * image[img_ind_i, img_ind_j])
                result[i, j] = num
    return result


# # The numba-speeded up version
# # nb_filter2d_at_mask = njit(double[:, :](double[:, :], boolean[:, :], double[:, :]))(filter2d_at_mask, parallel=True)
#
# # Now fastfilter_2d runs at speeds as if you had first translated
# # it to C, compiled the code and wrapped it with Python
# width = 10000
# height = 10000
# shape = (height, width)
# image = numpy.random.random(shape)
# filt = numpy.random.random((3, 3))
# mask = numpy.ones(shape, dtype=bool)
# mask[:, 0:(width//4)*3] = False
#
# # ts = timer()
# # res = filter2d_at_mask(image, mask, filt)
# # te = timer()
# # print("nb_filter2d_at_mask = {:.2f} sec.".format(te - ts))
#
# # Force numba compilation before timing
# res = nb_filter2d_at_mask(numpy.random.random((3, 3)), numpy.ones((3, 3), dtype=bool), numpy.random.random((3, 3)))
#
# ts = timer()
# res = nb_filter2d_at_mask(image, mask, filt)
# te = timer()
# print("nb_filter2d_at_mask = {:.2f} sec.".format(te - ts))
#
# ts = timer()
# res = cv2.filter2D(image, -1, filt)
# te = timer()
# print("cv2.filter2D = {:.2f} sec.".format(te - ts))


# --- CUDA version ---
def nb_convolve_at_mask_cuda(image, mask, filt):
    result = np.copy(image)
    threadsperblock = 32
    # blockspergrid = (image.size + (threadsperblock - 1)) // threadsperblock
    # kernel_update_at_mask[block_size, ](image, mask, filt, result)
    return result

@cuda.jit
def kernel_update_at_mask(image, mask, filt, result):
    i, j = cuda.grid(2)
    if mask[i, j]:
        M, N = image.shape
        Mf, Nf = filt.shape
        Mf2 = Mf // 2
        Nf2 = Nf // 2
        num = 0.0
        for ii in range(Mf):
            for jj in range(Nf):
                img_ind_i = abs(i - Mf2 + ii)
                if img_ind_i > M - 1:
                    mod_i = img_ind_i % (M - 1)
                    img_ind_i = M - 1 - mod_i
                img_ind_j = abs(j - Nf2 + jj)
                if img_ind_j > N - 1:
                    mod_j = img_ind_j % (N - 1)
                    img_ind_j = N - 1 - mod_j
                num += (filt[Mf - 1 - ii, Nf - 1 - jj] * image[img_ind_i, img_ind_j])
        result[i, j] = num
