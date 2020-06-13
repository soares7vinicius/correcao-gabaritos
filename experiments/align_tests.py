#!/usr/bin/env python
# coding: utf-8

# In[92]:


# ! pip install --upgrade imutils


# In[93]:


import numpy as np
import cv2
from imutils.perspective import four_point_transform
from imutils import contours as im_contours
import imutils
import skimage


# In[94]:


image = cv2.imread("../samples/model3/5.png")

factor = 0.4
height, width, _ = image.shape
height = int(height * factor)
width = int(width * factor)
image = cv2.resize(image, (width, height))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

cv2.imwrite("canny.png", edged)

# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# docCnt = None

# if len(cnts) > 0:
# 	# sort the contours according to their size in
# 	# descending order
# 	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# 	# loop over the sorted contours
# 	for c in cnts:
# 		# approximate the contour
# 		peri = cv2.arcLength(c, True)
# 		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# 		# if our approximated contour has four points,
# 		# then we can assume we have found the paper
# 		if len(approx) == 4:
# 			docCnt = approx
# 			break

# gray = four_point_transform(gray, docCnt.reshape(4, 2))


# In[95]:


# image_obj = cv2.imread('../samples/foto.jpg')
# image_obj = imutils.resize(image_obj, height=1200)

# gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)

image_obj = image.copy()

# blur = cv2.GaussianBlur(gray, (5, 5), 0)

# kernel = np.ones((4, 4), np.uint8)
# dilation = cv2.dilate(blurred, kernel, iterations=1)

# thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 5, 2)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
cv2.imwrite("thresh.png", thresh)
# thresh = cv2.(thresh, kernel, iterations=1)

# cv2.imshow("image", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Now finding Contours         ###################
_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
coordinates = []
for cnt in contours:
    # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
    approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        coordinates.append(approx)
# cv2.drawContours(image_obj, [cnt], 0, (0, 0, 255), 3)

print(len(coordinates))
# cv2.imwrite("result.png", image_obj)


# In[96]:


sorted_img = image_obj.copy()

# calculando media de niveis de branco/preto pra cada contorno
triangles = []
for cnt in coordinates:
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    #     cv2.imwrite("mask.png", mask)
    mean = cv2.mean(thresh, mask=mask)
    triangles.append((cnt, mean[0]))

# selecionando os 4 contornos com menores niveis de media (triangulos pretos)
triangles.sort(key=lambda x: x[1], reverse=False)
triangles = [tri[0] for tri in triangles[:4]]

# encontrando centros dos triangulos para futura ordenacao
tri_centers = []
for triangle in triangles:
    M = cv2.moments(triangle)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    tri_centers.append((triangle, (cX, cY)))
tri_centers = np.array(tri_centers)
# print(tri_centers)

# ordenando os triangulos, atraves dos seus centros, de cima pra baixo da esquerda pra direita
tri_centers = sorted(tri_centers, key=lambda v: [v[1][0], v[1][1]])
NUM_ROWS = 2
sorted_cols = []
for k in range(0, len(tri_centers), NUM_ROWS):
    col = tri_centers[k : k + NUM_ROWS]
    sorted_cols.extend(sorted(col, key=lambda v: v[1][1]))
tri_centers = sorted_cols
triangles = [tri[0] for tri in tri_centers]

# desenhando triangulos na imagem, para visualizacao
for i, c in enumerate(triangles):
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the contour and label number on the image
    cv2.drawContours(sorted_img, [c], -1, (0, 255, 0), 2)
    cv2.putText(
        sorted_img,
        "#{}".format(i + 1),
        (cX - 20, cY),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )
cv2.imwrite("result.png", sorted_img)


# In[97]:


trilist = []
for triangle in triangles:
    tripoints = []
    for point in triangle:
        tripoints.append((point[0][0], point[0][1]))
    dtype = [("x", int), ("y", int)]
    a = np.array(tripoints, dtype=dtype)
    trilist.append(a)
triangles = np.array(trilist)
triangles

p00x = min(triangles[0], key=lambda x: x[0])[0]
p00y = min(triangles[0], key=lambda x: x[1])[1]
p00 = (p00x, p00y)

p10x = min(triangles[1], key=lambda x: x[0])[0]
p10y = max(triangles[1], key=lambda x: x[1])[1]
p10 = (p10x, p10y)

p01x = max(triangles[2], key=lambda x: x[0])[0]
p01y = min(triangles[2], key=lambda x: x[1])[1]
p01 = (p01x, p01y)

p11x = max(triangles[3], key=lambda x: x[0])[0]
p11y = max(triangles[3], key=lambda x: x[1])[1]
p11 = (p11x, p11y)

warped = imutils.perspective.four_point_transform(
    thresh, np.array([p00, p11, p10, p01])
)
cv2.imwrite("warped.png", warped)


# In[98]:


# cv2.imshow("image", warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[ ]:
