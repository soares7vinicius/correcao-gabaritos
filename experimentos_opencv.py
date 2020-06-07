#!/usr/bin/env python
# coding: utf-8

# In[40]:


get_ipython().system(" ls | grep -i png")


# In[41]:


# https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d


# In[1]:


# ! conda remove opencv
# ! conda install -c menpo opencv
# ! pip install --upgrade pip
# ! pip install opencv-contrib-python


# In[2]:


import cv2

import numpy as np
import matplotlib.pyplot as plt


# In[4]:


img = cv2.imread("experiments/warped.png",)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img.shape


# In[5]:


factor = 0.5

height = int(height * factor)
width = int(width * factor)
min_circle_radius = int(20 * factor)
max_circle_radius = int(25 * factor)


# In[6]:


# n_height, n_width = int(width/factor), int(height/factor)
img = cv2.resize(img, (width, height))
_, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)


# In[12]:


# plt.figure(figsize=(10, 12))
# plt.imshow(img, aspect="auto")

# cv2.imshow("modelo", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[7]:


# finding lines
import math
import numpy as np

dst = cv2.Canny(img, 300, 400, None, 3)
lines = cv2.HoughLines(
    image=dst,
    rho=1,
    theta=np.pi / 2,
    threshold=210,
    min_theta=np.pi / 2,
    max_theta=np.pi,
)
print(len(lines))

colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #         if theta >= (75/180.0)*np.pi and theta <= (110/180.0)*np.pi:
        #             color = (0, 0, 0)
        #             cv2.rectangle(img, pt1, (pt2[0]-50, pt2[1]-50), color, -1)
        #             cv2.rectangle(img, pt1, (pt2[0]+50, pt2[1]+50), color, -1)
        #         else:
        #             color = (0, 0, 0)
        #             cv2.rectangle(img, pt1, (pt2[0]-50, pt2[1]-50), color, -1)
        cv2.line(colored, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)


# cv2.imshow("modelo", colored)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[8]:


lines = [line[0][0] for line in lines]
lines = [line for line in lines if line <= height / 3]
lines = sorted(lines, reverse=True)
divisor_line = int(lines[0])


# In[9]:


divisor_line


# In[10]:


header = img[:divisor_line, :]
content = img[divisor_line:, :]


# In[11]:


int((2 * min_circle_radius) + (5 * factor))


# In[17]:


circles = cv2.HoughCircles(
    image=content,
    method=cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=int((2 * min_circle_radius) + (5 * factor)),
    param1=200,
    param2=7,
    minRadius=min_circle_radius,
    maxRadius=max_circle_radius,
)
# print(circles)
print(circles.shape)
# content.shape
circles.shape[1]


# In[19]:


# from sklearn.cluster import KMeans

# circles = np.round(circles[0, :]).astype("int")

# xs = []
# ys = []
# for c in circles:
#     ys.append(c[0])
#     xs.append(c[1])


# xs = np.array(xs).reshape(-1, 1)
# kmeans_x = KMeans(n_clusters=30, max_iter=50).fit(xs)
# centers_x = np.round(kmeans_x.cluster_centers_).astype("int")

# ys = np.array(ys).reshape(-1, 1)
# kmeans_y = KMeans(n_clusters=20, max_iter=50).fit(ys)
# centers_y = np.round(kmeans_y.cluster_centers_).astype("int")

# ncircles = []
# for c in circles:
#     y = min(centers_y, key=lambda v: abs(v - c[0]))[0]
#     x = min(centers_x, key=lambda v: abs(v - c[1]))[0]
#     ncircles.append([y, x, c[2]])


# circles = sorted(ncircles, key=lambda v: [v[0], v[1]])


# In[18]:


circles = np.round(circles[0, :]).astype("int")
circles = sorted(circles, key=lambda v: [v[0], v[1]])

NUM_ROWS = 30

sorted_cols = []
for k in range(0, len(circles), NUM_ROWS):
    col = circles[k : k + NUM_ROWS]
    sorted_cols.extend(sorted(col, key=lambda v: v[1]))

circles = sorted_cols


# In[19]:


min_circle_radius, max_circle_radius


# In[20]:


radii = [c[2] for c in circles]
min(radii), max(radii)


# In[21]:


def points_in_circle_np(
    radius, x0=0, y0=0,
):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    for x, y in zip(x_[x], y_[y]):
        yield x, y


def points_mode_value(img, points):
    values = [img[y, x] for y, x in points]
    mode = max(set(values), key=values.count)
    return mode


# In[24]:


# img_colored = cv2.cvtColor(content, cv2.COLOR_GRAY2BGR)
# for i in range(0, 150):
#     y, x, r = circles_sorted[0][i]
#     cv2.circle(img_colored, (y, x), r, (255, 0, 0), -1)

# cv2.imshow("modelo", img_colored)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[22]:


# criando sequencia da passagem vertical das questoes
# 1, ..., 15, 61, ..., 75, 16, ..., 30, ...
seq = []
q = 1
for _ in range(1, 121):
    seq.append(q)
    if q % 15 == 0:
        if q > 60:
            q -= 59
        else:
            q += 46
    else:
        q += 1

# dividindo sequencia em 4 grupos
# cada grupo é replicado em 5 vezes conseguintes, criando a sequencia final de 600 posicoes
nseq = []
for chunk in np.array_split(seq, 4):
    a = np.concatenate([chunk for _ in range(0, 5)])
    nseq += list(a)

# identificando alternativa a cada valor da sequencia
alt_i = 0
alt = "ABCDE"
for i in range(len(nseq)):
    nseq[i] = (nseq[i], alt[alt_i])

    if (i + 1) % 30 == 0:
        if alt_i == 4:
            alt_i = 0
        else:
            alt_i += 1


# In[23]:


ans = {i: [] for i in range(1, 121)}


# In[24]:


img_colored = cv2.cvtColor(content, cv2.COLOR_GRAY2BGR)

# circles = circles_sorted

if circles is not None:
    circles = np.uint16(np.around(circles))
    for cont, i in enumerate(circles, start=0):
        points = points_in_circle_np(x0=i[1], y0=i[0], radius=i[2])
        mode = points_mode_value(content, points)

        if mode == 255:
            cv2.circle(img_colored, (i[0], i[1]), i[2], (0, 0, 255), -1)  # cv2.FILLED)
        else:
            cv2.circle(img_colored, (i[0], i[1]), i[2], (0, 255, 0), -1)  # cv2.FILLED)
            ans[nseq[cont][0]].append(nseq[cont][1])

        cv2.putText(
            img_colored,
            str(cont + 1),
            (i[0] - i[2], i[1] + i[2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


# for i in range(0, 67):
#     y, x, r = circles[0][i]
#     cv2.circle(img_colored, (y, x), r, (255, 0, 0), -1)


cv2.imwrite("out.png", img_colored)


# In[28]:


cv2.imwrite("content.png", content)


# In[29]:


# cv2.imshow("modelo", img_colored)
# cv2.waitKey(0)

# cv2.destroyAllWindows()


# In[30]:


from operator import itemgetter

ans = list(ans.items())
ans = sorted(ans, key=itemgetter(0))
# ans


# In[31]:


ans


# In[32]:


# img_colored = cv2.resize(img_colored, (423, 550))
cv2.imshow("modelo", img_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## header

# In[ ]:


# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
# https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html


# In[33]:


header = img[:divisor_line, :]


# In[34]:


cv2.imshow("modelo", header)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[35]:


contours, hierarchy = cv2.findContours(header, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours_area = []
# hie = [h for h in hierarchy[0] if h[2] == -1 and h[3] == 0]


header_colored = cv2.cvtColor(header, cv2.COLOR_GRAY2BGR)
for contour in contours:
    #     approx = contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    contours_area.append((cv2.contourArea(approx), approx))
    if contours_area[-1][0] > 300 and contours_area[-1][0] < 1000:
        (x, y, w, h) = cv2.boundingRect(approx)
        cv2.rectangle(header_colored, (x, y), (x + w, y + h), (255, 0, 0), 2)


# In[38]:


cv2.imshow("modelo", header_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


# hierarchy


# In[ ]:


len(contours), len(contours_area)


# In[ ]:


areas = sorted([a for a, c in contours_area], reverse=True)
contours_area = [ca for ca in contours_area if ca[0] > 600 and ca[0] < 1000]


# In[ ]:


areas[1:13]


# In[ ]:


header_colored = cv2.cvtColor(header, cv2.COLOR_GRAY2BGR)
for _, cont in contours_area:
    (x, y, w, h) = cv2.boundingRect(approx)
    cv2.rectangle(header_colored, (x, y), (x + w, y + h), (255, 0, 0), 2)


# In[ ]:


sorted(contours_area, reverse=True)[1:14]


# In[ ]:


mode = max(set(contours_area), key=contours_area.count)  # achando area moda

boxes = [contour for contour, area in zip(contours, contours_area) if area == mode]
mode


# In[ ]:


header_colored = cv2.cvtColor(header, cv2.COLOR_GRAY2BGR)
for box in boxes:
    (x, y, w, h) = cv2.boundingRect(box)
    cv2.rectangle(header_colored, (x, y), (x + w, y + h), (0, 0, 255), 2)


# In[ ]:


cv2.imshow("modelo", header_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


from sklearn.cluster import KMeans

areas = np.array(contours_area).reshape(-1, 1)
# areas


# In[ ]:


np.set_printoptions(suppress=True)
kmeans = KMeans(n_clusters=10, random_state=0).fit(areas)

kmeans.cluster_centers_, kmeans.labels_


# In[ ]:


keys = set(kmeans.labels_)
rep = {k: 0 for k in keys}

for lbl in kmeans.labels_:
    rep[lbl] = rep[lbl] + 1

rep
