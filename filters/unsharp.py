import math
import numpy as np

from .utils import pad_image

kernels = [
    np.array([[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]]),
    np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
]

def normalize(image):
    return ((image - np.min(image)) * 255) / np.max(image)

def filter(image):
    c = float(input())
    k = int(input())

    n = 3   # kernels are n x n

    kernel = kernels[k - 1]

    working_area = pad_image(image, math.floor(n / 2))
    filtered = np.empty_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = working_area[i:i + n, j:j + n]
            filtered[i, j] = np.sum(region * kernel)

    filtered = normalize(filtered)

    return normalize(image + filtered * c)
