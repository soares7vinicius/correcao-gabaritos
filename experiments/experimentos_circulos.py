#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, resize, rescale
from skimage.feature import canny
from skimage.draw import circle_perimeter, circle
from skimage.util import img_as_ubyte
from skimage.io import imread, imsave

from pdf2image import convert_from_path, convert_from_bytes


# In[52]:


from skimage import __version__ as ski_version

ski_version


# In[3]:


images = convert_from_path("samples/gabarito (2).pdf")
for i, image in enumerate(images, 4):
    image.save(f"samples/{i}.png", "PNG")


# In[54]:


factor = 3


# In[66]:


# Load picture and detect edges
image = imread("modelo.png", as_gray=True)
height, width = image.shape

new_height, new_width = int(height / factor), int(width / factor)
print(new_height, new_width)

image = resize(image, (730, 550))

image = img_as_ubyte(image)

edges = canny(image=image, sigma=0.1, low_threshold=10, high_threshold=10)

plt.figure(figsize=(25, 30))
plt.imshow(edges, aspect="auto")
# plt.savefig('edges.png')


# In[56]:


image.shape


# In[57]:


# Detect two radii
import time

t0 = time.perf_counter()

# hough_radii = np.arange(15, 15, 2)
hough_radii = [5]  # np.array([5])
hough_res = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=602)

tf = time.perf_counter()
tf - t0


# In[58]:


# display(accums, cx, cy, radii)


# In[59]:


plt.clf()
plt.cla()
plt.close()


# In[60]:


image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle(center_y, center_x, radius, shape=image.shape)
    image[circy, circx] = (0, 255, 0)

plt.figure(figsize=(20, 25))
plt.imshow(image, aspect="auto")
imsave("out.png", image)


# In[ ]:
