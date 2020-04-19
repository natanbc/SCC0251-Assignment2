import imageio
import math
import numpy as np

from filters.bilateral import filter as bilateral
from filters.unsharp import filter as unsharp

def RSE(m, r):
    return math.sqrt(np.square(m - r).sum())

filename = str(input()).rstrip()
img = imageio.imread(filename).astype(np.float)
method = int(input())
save = int(input())

if method == 1:
    output_img = bilateral(img)
elif method == 2:
    output_img = unsharp(img)
else:
    raise Exception("Method " + str(method) + " unimplemented")

print("{0:.4f}".format(RSE(output_img, img)))

if save == 1:
    imageio.imwrite("output_img.png", output_img.astype(np.uint8))
