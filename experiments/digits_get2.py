#%%
import cv2
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing, disk
from imutils.contours import sort_contours, label_contour
from imutils import grab_contours
from matplotlib import cm
from matplotlib import pyplot as plt

# https://stackoverflow.com/questions/45310331/detecting-vertical-lines-using-hough-transforms-in-opencv/45312187

# usando hough lines
# %%
# lendo imagem e extraindo roi geral
image = cv2.imread("warped.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

h, w = image.shape

boxes_area = image[int(h * 0.095) : int(h * 0.14), int(w * 0.09) : int(w * 0.77)]

# plotando imagem com poligono
fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(boxes_area, cmap=cm.gray)

# cv2.imwrite("boxes.png", boxes_area)

#%%


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

# plotando imagem com poligono
fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(boxes, cmap=cm.gray)


#%%

# removendo linhas do contorno externo
mask = np.ones_like(boxes_area) * 255
cv2.drawContours(mask, [hull], -1, 0, -1)
# x, y, w, h = cv2.boundingRect(hull)

mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))

boxes_area[mask != 0] = 255 # pintando area externa da mascara de branco
# boxes_area = cv2.bitwise_not(boxes_area)

fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(boxes_area, cmap=cm.gray)


#%%
# iterando a direita até encontrar divisoria

hull_ys = np.sort(hull[:, 0, 1])
miny, maxy = hull_ys[0], hull_ys[-1]

hull_xs = np.sort(hull[:, 0, 0])
minx, maxx = hull_xs[0], hull_xs[-1]

box_len = maxx - minx
span = box_len//11

_, contours, _ = cv2.findContours(~boxes_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
test_img = cv2.cvtColor(boxes_area, cv2.COLOR_GRAY2BGR)
cv2.drawContours(test_img, contours, -1, (0, 0, 255), 1)

x = minx + span
mid_y = (maxy+miny)//2
cont = 0

div_contours = []
while x <= maxx and cont < 10:
    pixel = boxes_area[mid_y, x]
    cv2.circle(test_img, (x, mid_y), 2, (0, 255, 0), -1)
    # print(pixel)
    if pixel == 0:
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, (x, mid_y), measureDist=False) >= 0:
                div_contours.append(cnt)
                cv2.drawContours(test_img, [cnt], -1, (255, 0, 0), 1)
        x += span
        cont += 1
    else:
        cv2.line(test_img, (x, miny), (x, maxy), (0, 0, 255), 1)
        x += 1
    
# removendo divisorias encontradas
mask = np.ones_like(boxes_area) * 255
cv2.drawContours(mask, div_contours, -1, 0, -1)
boxes_area[mask == 0] = 255

fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(test_img, cmap=cm.gray)

fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(boxes_area, cmap=cm.gray)

#%%

# dilatadando a imagem, encontrando os contornos, e os ordenando

selem = disk(1)
ref = dilation(~boxes_area, selem)

_, contours, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours.sort(key=cv2.contourArea, reverse=True)
contours = contours[:11]
contours = list((sort_contours(contours, "left-to-right"))[0])

ref = cv2.cvtColor(boxes_area, cv2.COLOR_GRAY2BGR)

boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    boxes.append((x, y, w, h))
    cv2.rectangle(ref, (x, y), (x+w, y+h), (255, 0, 0), 1)


fig, ax = plt.subplots(dpi=400)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(ref, cmap=cm.gray)

