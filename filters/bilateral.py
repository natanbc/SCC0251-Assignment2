import math
import numpy as np

from .utils import pad_image

def G(x, sigma):
    """
Calculates a gaussian kernel"""
    return ((1 / (2 * math.pi * pow(sigma, 2))) *
            math.exp(-(pow(x, 2) / (2 * pow(sigma, 2)))))

def spatial_component(n, sigma):
    """
Calculates the spatial gaussian kernel
Arguments:
    n     -- filter size
    sigma -- variance"""
    center = int((n - 1) / 2) # index
    def dist(x, y):
        return math.sqrt(pow(x - center, 2) + pow(y - center, 2))
    matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            matrix[i, j] = G(dist(i, j), sigma)
    return matrix

def range_component(region, n, sigma):
    """
Calculates the range gaussian kernel of a region of the image
Arguments:
    region -- region of the image to use
    n      -- filter size
    sigma  -- variance"""
    center = int((n - 1) / 2)
    pixel = region[center, center]
    matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Ii = region[i, j]
            matrix[i, j] = G(Ii - pixel, sigma)
    return matrix

def filter(image, n, sigma_s, sigma_r):
    """
Bilateral Filter
Arguments:
    image -- input image
    n     -- size of the filter
    σs    -- space variance
    σr    -- range variance"""

    gs = spatial_component(n, sigma_s)

    padding_needed = math.floor(n / 2)

    result = np.empty_like(image)
    working_area = pad_image(image, padding_needed)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = working_area[i:i + n, j:j + n]
            gr = range_component(region, n, sigma_r)
            w = gr * gs
            Wp = np.sum(w)
            If = np.sum(w * region)
            result[i, j] = If / Wp

    return result

