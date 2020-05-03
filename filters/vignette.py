
def gauss(x, sig):
	"""
	Samples the value at a single point from the Gaussian Curve, parameterized
	using the sig value as its sigma.
		x   - Relative positioning along the curve, which is centered at x = 0.
		sig - Sigma parameter.
	"""
	from math import pi as PI;
	from math import exp;
	return exp(-(x ** 2 / 2 * sig ** 2)) / (2 * PI * sig ** 2)

def filter(img, sigma_row, sigma_col):
	"""
	Applies a vignette filter to a desired source image, returning the result.
		img       - Source image.
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

