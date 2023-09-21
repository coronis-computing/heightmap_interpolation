import numpy as np


def halve_image(img):
    dims = img.shape
    if len(dims) == 3:
        halved_img = img[0:dims[0]:2, 0:dims[1]:2, :]
    elif len(dims) == 2:
        halved_img = img[0:dims[0]:2, 0:dims[1]:2]
    else:
        raise ValueError("Number of dimensions of the input image not 2 nor 3")
    return halved_img
