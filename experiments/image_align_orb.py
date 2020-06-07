#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


ref = cv2.imread("modelo3.png")
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
height, width = ref.shape

factor = 1

height = int(height * factor)
width = int(width * factor)

ref = cv2.resize(ref, (width, height))
_, ref = cv2.threshold(ref, 240, 255, cv2.THRESH_BINARY)


# In[3]:


img = cv2.imread("../samples/n1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img.shape

factor = 1

height = int(height * factor)
width = int(width * factor)

img = cv2.resize(img, (width, height))
_, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)


# In[4]:


cv2.imshow("modelo", ref)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.1

orb = cv2.ORB_create(MAX_FEATURES)
kp1, desc1 = orb.detectAndCompute(ref, None)
kp2, desc2 = orb.detectAndCompute(img, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
matches = matcher.match(desc1, desc2, None)

matches.sort(key=lambda x: x.distance, reverse=False)

num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:num_good_matches]
print(num_good_matches)

im_matches = cv2.drawMatches(ref, kp1, img, kp2, matches, None)
cv2.imwrite("matches.png", im_matches)

pts1 = np.zeros((len(matches), 2), dtype=np.float32)
pts2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    pts1[i, :] = kp1[match.queryIdx].pt
    pts2[i, :] = kp2[match.queryIdx].pt

h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

height, width = ref.shape
img_reg = cv2.warpPerspective(img, h, (width, height))

cv2.imwrite("aligned.png", img_reg)


# In[ ]:
