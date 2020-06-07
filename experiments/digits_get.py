import cv2
import numpy as np
import os
from skimage.morphology import erosion, dilation, opening, closing, disk
from imutils.contours import sort_contours, label_contour
from imutils import grab_contours
# https://stackoverflow.com/questions/45310331/detecting-vertical-lines-using-hough-transforms-in-opencv/45312187

# %%

# os.chdir("experiments")

# %%
# lendo imagem e extraindo roi geral
image = cv2.imread("warped.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)

h, w = image.shape

boxes_area = image[int(h * 0.095) : int(h * 0.14), int(w * 0.09) : int(w * 0.77)]

cv2.imwrite("boxes.png", boxes_area)

# %%
# encontrando contorno externo
boxes_neg = cv2.bitwise_not(boxes_area)
_, contours, hierarchy = cv2.findContours(
    boxes_neg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
del boxes_neg

img = boxes_area.copy()
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.drawContours(img, [contours[0]], -1, (0, 255, 0), 2)
cv2.imwrite("contours.png", img)


# %%

# removendo linhas do contorno externo
contour = contours[0]
mask = np.ones_like(boxes_area) * 255

hull = cv2.convexHull(contour)
cv2.drawContours(mask, [hull], -1, 0, -1)
x,y,w,h = cv2.boundingRect(contour)

mask = cv2.dilate(mask, np.ones((9,9),np.uint8))

boxes_area[mask != 0] = 255
# boxes_area = cv2.bitwise_not(boxes_area)

cv2.imwrite('boxes.png', boxes_area)

#%%

# fechando a imagem, encontrando os contornos, os ordenando e filtrando (pular os traços divisorios)

ref = cv2.bitwise_not(boxes_area)

selem = disk(1)
ref = closing(ref, selem)

_, contours, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contours.sort(key=cv2.contourArea, reverse=True)
contours = contours[:21]

contours = list((sort_contours(contours, "left-to-right"))[0])
contours = contours[::2]

ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
cv2.drawContours(ref, contours, -1, (0, 255, 0), 2)
# for i, cnt in enumerate(contours, start=1):
#     ref = label_contour(ref, cnt, i)

cv2.imwrite('contours.png', ref)