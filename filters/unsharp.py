import math
import numpy as np

from .utils import pad_image

kernels = [
    np.array([[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]]),
    np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
]

def normalize(image):
    return ((image - np.min(image)) * 255) / (np.max(image) - np.min(image))

def filter(image, c, k):
    """
Unsharp mask using the Laplacian Filter
Arguments:
    iamge -- input image
    c     -- scaling factor
    k     -- kernel to use"""
    n = 3   # kernels are n x n

    kernel = kernels[k - 1]

    working_area = pad_image(image, math.floor(n / 2))
    filtered = np.empty_like(image)

    # 1. Convolve original image with kernel, storing into `filtered`
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = working_area[i:i + n, j:j + n]
            filtered[i, j] = np.sum(region * kernel)

    # 2. Scale the filtered image
    filtered = normalize(filtered)

    # 3. Add the filtered image, multiplied by c, to the original image
    result = filtered * c + image

    # 4. Scale the final image
    return normalize(result)
