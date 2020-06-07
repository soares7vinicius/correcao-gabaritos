#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

from pdf2image import convert_from_path, convert_from_bytes

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks


# In[3]:


images = convert_from_path("modelo.pdf")
for image in images:
    image.save("modelo.png", "PNG")


# In[4]:


image = imread("modelo.png", as_gray=True)
image = img_as_ubyte(image)


# In[ ]:


coords = corner_peaks(corner_harris(image), min_distance=5)
coords_subpix = corner_subpix(image, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(
    coords[:, 1], coords[:, 0], color="cyan", marker="o", linestyle="None", markersize=6
)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], "+r", markersize=15)
ax.axis((0, 310, 200, 0))
plt.show()
