#!/usr/bin/env python
# coding: utf-8

# In[55]:


get_ipython().system(" pip install imutils")


# In[1]:


import json
import numpy as np
from sklearn.cluster import KMeans


# In[2]:


circles = json.load(open("circles.json"))


# In[3]:


circles = np.round(circles).astype("int")
circles


# In[5]:


xs = []
ys = []
for c in circles:
    ys.append(c[0])
    xs.append(c[1])


# In[23]:


circles[0][0]


# In[29]:


xs = np.array(xs).reshape(-1, 1)
kmeans_x = KMeans(n_clusters=20, max_iter=50).fit(xs)
centers_x = np.round(kmeans_x.cluster_centers_).astype("int")

ys = np.array(ys).reshape(-1, 1)
kmeans_y = KMeans(n_clusters=30, max_iter=50).fit(ys)
centers_y = np.round(kmeans_y.cluster_centers_).astype("int")


ncircles = []
for c in circles:
    y = min(centers_y, key=lambda v: abs(v - c[0]))[0]
    x = min(centers_x, key=lambda v: abs(v - c[1]))[0]
    ncircles.append([y, x, c[2]])


# In[30]:


circles = sorted(ncircles, key=lambda v: [v[0], v[1]])


# In[34]:


circles[0:10]
