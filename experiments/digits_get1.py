#%%
import cv2
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, disk
from imutils.contours import sort_contours, label_contour
from imutils import grab_contours
from matplotlib import cm

# https://stackoverflow.com/questions/45310331/detecting-vertical-lines-using-hough-transforms-in-opencv/45312187

# usando harris corners
# %%
# lendo imagem e extraindo roi geral
image = cv2.imread("warped.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

h, w = image.shape

boxes_area = image[int(h * 0.095) : int(h * 0.14), int(w * 0.09) : int(w * 0.77)]

cv2.imwrite("boxes.png", boxes_area)

#%%

from matplotlib import pyplot as plt
from skimage import measure
from skimage.feature import corner_harris, corner_peaks, corner_subpix
# from imutils.perspective import four_point_transform

# aplicando detector de harris para encontrar os cantos
coords = corner_peaks(corner_harris(~boxes_area, k=0.03), min_distance=1, threshold_rel=0.3)

# plotando todos os pontos encontrados
fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(boxes_area, cmap=plt.cm.gray)
ax.plot(
    coords[:, 1], coords[:, 0], color="blue", marker="o", linestyle="None", markersize=2
)

# ordenandos os pontos de cantos encontrados
# pegando os dois pontos mais a esquerda e os mais a direita horizantalmente
coords = sorted(coords, key=lambda x: x[1], reverse=False)
most_left = coords[:2]
most_right = coords[-2:]
points = np.array(most_left + most_right)

# plotando os 4 pontos extremos
ax.plot(
    points[:, 1], points[:, 0], color="red", marker="o", linestyle="None", markersize=3
)

#%%

# trocando invertendo x e y (colunas)
points.T[[0, 1]] = points.T[[1, 0]] 

# criando imagem colorida para visualizacao
boxes = cv2.cvtColor(boxes_area.copy(), cv2.COLOR_GRAY2BGR)

# criando poligono convexo composto pelos pontos extremos
# (gift wrapping algorithm)
hull = cv2.convexHull(points)

# desenhando poligono na imagem para visualizacao
cv2.drawContours(boxes, [hull], -1, (255, 0, 0), 1)

# recortando area do poligono
# criando mascara pra recorte
mask = np.zeros_like(boxes_area)
out = np.zeros_like(boxes_area)
cv2.drawContours(mask, [hull], -1, 255, -1)
# aplicando a mascara
out[mask == 255] = boxes_area[mask == 255]
# recortando regiao a imagem
(y, x) = np.where(mask == 255)
(topy, topx) = (np.min(y), np.min(x))
(bottomy, bottomx) = (np.max(y), np.max(x))
out = out[topy:bottomy+1, topx:bottomx+1]

# plotando imagem com poligono
fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(boxes, cmap=cm.gray)

# plotando imagem recortada
fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(out, cmap=cm.gray)

# %%

# removendo linhas do contorno externo
mask = np.ones_like(boxes_area) * 255
cv2.drawContours(mask, [hull], -1, 0, -1)
x, y, w, h = cv2.boundingRect(hull)

mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))

boxes_area[mask != 0] = 255
# boxes_area = cv2.bitwise_not(boxes_area)


fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(boxes_area, cmap=cm.gray)


#%%

# fechando a imagem, encontrando os contornos, os ordenando e filtrando (pular os traços divisorios)

ref = cv2.bitwise_not(boxes_area)

# selem = disk(1)
# ref = closing(ref, selem)

_, contours, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contours.sort(key=cv2.contourArea, reverse=True)
contours = contours[:21]

contours = list((sort_contours(contours, "left-to-right"))[0])
contours = contours[::2]

ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
cv2.drawContours(ref, contours, -1, (0, 0, 255), 2)
# for i, cnt in enumerate(contours, start=1):
#     ref = label_contour(ref, cnt, i)

fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(ref, cmap=cm.gray)

