import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from heightmap_interpolation.inpainting.convolve_at_mask import nb_convolve_at_mask, nb_convolve_at_mask_guvec
from timeit import default_timer as timer


def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(float) / info.max # Divide all values by the largest possible value in the datatype


def double2im(im):
    im = im*255
    return im.astype(np.uint8)


def my_fun(param):
    # Load image
    image = cv2.imread(param.input_image)

    if len(image.shape) == 3:
        print("[WARNING] This script expects single-channel images, we will convert the input color image to grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to floats (elevation will be in this format)
    image = im2double(image)

    # Load/create mask
    if param.mask:
        mask = cv2.imread(param.mask)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = np.asarray(mask, dtype="bool")
    else:
        mask = np.full(image.shape, True, dtype=bool)
        mask[:, 0:(mask.shape[1] // 2)] = False # half of the image should not be taken into account

    # Create the filter
    # filter = np.random.random((param.filter_size, param.filter_size))
    # filt = np.ones((param.filter_size, param.filter_size))

    filt = np.array([[0.0, 1.0, 0.0],
                     [1.0, -4.0, 1.0],
                     [0.0, 1.0, 0.0]])

    # Force numba compilation before timing
    res = nb_convolve_at_mask(np.random.random((3, 3)), np.ones((3, 3), dtype=bool), np.random.random((3, 3)))
    res = nb_convolve_at_mask_guvec(np.random.random((3, 3)), np.ones((3, 3), dtype=bool), np.random.random((3, 3)))

    # Apply the convolution using our methods
    num_tests = 1000

    ts = timer()
    for i in range(num_tests):
        our_res = nb_convolve_at_mask(image, mask, filt)
    te = timer()
    print("Our convolution (pure numba) executed in {:f} sec.".format((te - ts)))

    ts = timer()
    for i in range(num_tests):
        our_res = nb_convolve_at_mask_guvec(image, mask, filt)
    te = timer()
    print("Our convolution (guvec) executed in {:f} sec.".format((te - ts)))

    # Apply OpenCV's filter2D method
    ts = timer()
    for i in range(num_tests):
        cv_res = cv2.filter2D(image, -1, filt)
        if param.mask:
            # Mask the results
            cv_res = image*(1-mask) + cv_res*mask
    te = timer()
    print("OpenCV's filter2D executed in {:f} sec.".format((te - ts)))

    # Compute the error
    diff = np.linalg.norm(our_res.flatten() - cv_res.flatten(), 2)
    print("Norm of Differences = {:f}".format(diff))

    # Show results on screen
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))
    images = [image, our_res, cv_res, abs(our_res-cv_res)]
    titles = ['Original', 'Our convolution', 'filter2D', 'Difference']
    for (ax, im, title) in zip(axes, images, titles):
        # Convert to uint8 before showing
        ax.imshow(im)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()


def parse_args(args=None):
    # Parameters
    parser = argparse.ArgumentParser(
        description="Compare the results of our convolution with those in opencv")
    parser.add_argument("input_image", action="store", type=str,
                        help="Input image file")
    parser.add_argument("--mask", "-m", action="store", type=str, dest="mask", default="",
                        help="Mask, of the same size as the image, containing the region where the filter will be applied")
    parser.add_argument("--filter-size", action="store", type=str, dest="filter_size", default=3,
                        help="The size of the convolution filter (will be filled with random values)")
    param = parser.parse_args(args)

    return param


if __name__ == "__main__":
    my_fun(parse_args())