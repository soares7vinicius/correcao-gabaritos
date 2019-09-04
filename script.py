import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave


# Load picture and detect edges
image = imread('modelo.png', as_gray=True)
image = img_as_ubyte(image)
edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
plt.figure(figsize=(100,200))
plt.imshow(edges, aspect='auto')

# Detect two radii
hough_radii = np.arange(0, 15, 1)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=200)

# Draw them
#fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

#ax.imshow(image, cmap=plt.cm.gray)
#plt.show()
imsave('out.png', image)
