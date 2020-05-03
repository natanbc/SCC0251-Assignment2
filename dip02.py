#!/usr/bin/env python3

import imageio
import math
import numpy as np

from filters.bilateral import filter as bilateral
from filters.unsharp   import filter as unsharp
from filters.vignette  import filter as vinagrette

def RSE(m, r):
    return math.sqrt(np.square(m - r).sum())

filename = str(input()).rstrip()
img = imageio.imread(filename).astype(np.float)
method = int(input())
save = int(input())

if method == 1:
    n = int(input())
    sigma_s = float(input())
    sigma_r = float(input())
    output_img = bilateral(img, n, sigma_s, sigma_r)
elif method == 2:
    c = float(input())
    k = int(input())
    output_img = unsharp(img, c, k)
elif method == 3:
	sigma_row = float(input())
	sigma_col = float(input())
	output_img = vinagrette(img, sigma_row, sigma_col)
else:
    raise Exception("Method " + str(method) + " unimplemented")

output_img = output_img.astype(np.uint8)

print("{0:.4f}".format(RSE(output_img, img)))

if save == 1:
    imageio.imwrite("output_img.png", output_img)
