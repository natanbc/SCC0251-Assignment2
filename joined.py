#!/usr/bin/env python3
# vim: set ts=4:
#
# Assignment:	Image Enhancement and Filtering
# Course:		SCC0251
# Semester:		First of 2020
# Group:
#	<name>					<nUSP>
# 	***REMOVED***                       ***REMOVED***
#	***REMOVED***                       ***REMOVED***
#

import imageio
import math
import numpy as np

def pad_image(image, amount):
	h, w = image.shape						  # height, width
	nh, nw = h + amount * 2, w + amount * 2   # new values
	padded = np.zeros([nh, nw])
	padded[amount:amount + h, amount:amount + w] += image
	return padded

######################
## Bilateral filter ##
######################

def G(x, sigma):
	"""
Calculates a gaussian kernel"""
	return ((1 / (2 * math.pi * pow(sigma, 2))) *
			math.exp(-(pow(x, 2) / (2 * pow(sigma, 2)))))

def spatial_component(n, sigma):
	"""
Calculates the spatial gaussian kernel
Arguments:
	n	  -- filter size
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
	n	   -- filter size
	sigma  -- variance"""
	center = int((n - 1) / 2)
	pixel = region[center, center]
	matrix = np.zeros([n, n])
	for i in range(n):
		for j in range(n):
			Ii = region[i, j]
			matrix[i, j] = G(Ii - pixel, sigma)
	return matrix

def bilateral(image, n, sigma_s, sigma_r):
	"""
Bilateral Filter
Arguments:
	image -- input image
	n	  -- size of the filter
	σs	  -- space variance
	σr	  -- range variance"""

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

####################
## Unsharp filter ##
####################

kernels = [
	np.array([[ 0, -1,	0], [-1,  4, -1], [ 0, -1,	0]]),
	np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
]

def normalize(image):
	return ((image - np.min(image)) * 255) / (np.max(image) - np.min(image))

def unsharp(image, c, k):
	"""
Unsharp mask using the Laplacian Filter
Arguments:
	iamge -- input image
	c	  -- scaling factor
	k	  -- kernel to use"""
	n = 3	# kernels are n x n

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

#####################
## Vignette filter ## 
#####################

def gauss(x, sig):
	"""
	Samples the value at a single point from the Gaussian Curve, parameterized
	using the sig value as its sigma.
		x	- Relative positioning along the curve, which is centered at x = 0.
		sig - Sigma parameter.
	"""
	from math import pi as PI;
	from math import exp;
	return exp(-(x ** 2 / 2 * sig ** 2)) / (2 * PI * sig ** 2)

def vinagrette(img, sigma_row, sigma_col):
	"""
	Applies a vignette filter to a desired source image, returning the result.
		img		  - Source image.
		sigma_row - Sigma parameter for the curve along the vertical axis.
		sigma_col - Sigma parameter for the curve along the horizontal axis.
	"""
	width  = len(img[0])
	height = len(img)
	target = img.astype("float")
	maxval = None
	minval = None

	# Generate the combined gaussian matrix for the image and apply it as we go.
	# We also gather the minimum and maximum resulting values so that we can
	# normalize the results later on back to the 0-255 range.
	for i in range(height):
		for j in range(width):
			a = gauss(int(j - width  / 2), 1 / sigma_col)
			b = gauss(int(i - height / 2), 1 / sigma_row)
			target[i][j] *= a * b

			if maxval is None or target[i][j] > maxval: maxval = target[i][j]
			if minval is None or target[i][j] < minval: minval = target[i][j]

	# Normalize the image
	for i in range(height):
		for j in range(width):
			a = minval
			b = maxval
			c = 0
			d = 255
			target[i][j] = (target[i][j] - a) * ((d - c) / (b - a)) + c
	
	return target

###########################
## General functionality ##
###########################

def RSE(m, r):
	return math.sqrt(np.square(m - r).sum())

if __name__ == "__main__":
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

