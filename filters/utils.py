import numpy as np

def pad_image(image, amount):
    h, w = image.shape                        # height, width
    nh, nw = h + amount * 2, w + amount * 2   # new values
    padded = np.zeros([nh, nw])
    padded[amount:amount + h, amount:amount + w] += image
    return padded
