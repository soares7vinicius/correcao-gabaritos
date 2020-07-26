import json
import sys
import time
from operator import itemgetter
from pprint import pprint
from typing import List, Tuple

import cv2
import imutils
import numpy as np
from imutils import contours as im_contours
from imutils.contours import sort_contours, label_contour
from imutils.perspective import four_point_transform
from skimage import measure
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.morphology import erosion, dilation, opening, closing, disk
import matplotlib.pyplot as plt
from matplotlib import cm
from keras import backend as K
from tensorflow.keras.models import load_model


class SheetCorrector:
    def __init__(self, factor=0.5, debug=False):
        self.seq = self._generate_circles_index_sequence()
        self.debug = debug
        self.factor = factor
        self.model = load_model("model")

    def _clear(self):
        self.ans = {i: [] for i in range(1, 121)}

    def _generate_circles_index_sequence(self) -> List[Tuple[int, str]]:
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

        return nseq

    def _points_in_circle_np(
        self, radius, x0=0, y0=0,
    ):
        x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
        y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
        x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
        # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
        for x, y in zip(x_[x], y_[y]):
            yield x, y

    def _points_mode_value(self, img, points):
        values = [img[y, x] for y, x in points]
        mode = max(set(values), key=values.count)
        return mode

    def _read_and_preprocess_image(self, filepath: str):
        image = cv2.imread(filepath)

        height, width, _ = image.shape
        height = int(height * self.factor)
        width = int(width * self.factor)

        image = cv2.resize(image, (width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        if self.debug:
            cv2.imwrite("out/thresh.png", thresh)

        # detecting contours
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        coordinates = []
        for cnt in contours:
            # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                coordinates.append(approx)

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

        if self.debug:
            colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            # desenhando triangulos na imagem, para visualizacao
            for i, c in enumerate(triangles):
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # draw the contour and label number on the image
                cv2.drawContours(colored, [c], -1, (0, 0, 255), 2)
                # cv2.putText(
                #     sorted_img,
                #     "#{}".format(i + 1),
                #     (cX - 20, cY),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1.0,
                #     (0, 0, 0),
                #     2,
                # )

            cv2.imwrite("out/triangles.png", colored)

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

        warped = imutils.perspective.four_point_transform(thresh, np.array([p00, p11, p10, p01]))

        if self.debug:
            cv2.imwrite("out/warped.png", warped)

        return warped

    def _resize_image(self, img, size=(20, 20)):

        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1

        if h == w:
            return cv2.resize(img, size, cv2.INTER_AREA)

        dif = h if h > w else w

        interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

        x_pos = (dif - w) // 2
        y_pos = (dif - h) // 2

        if len(img.shape) == 2:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos : y_pos + h, x_pos : x_pos + w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos : y_pos + h, x_pos : x_pos + w, :] = img[:h, :w, :]

        return cv2.resize(mask, size, interpolation)

    def _deskew(self, img, size=28):
        affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

        m = cv2.moments(img)

        if abs(m["mu02"]) < 1e-2:
            return img.copy()

        skew = m["mu11"] / m["mu02"]
        M = np.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (size, size), flags=affine_flags)

        return img

    def _find_center(self, img):
        M = cv2.moments(img)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def _align_center(self, img, size=28):
        pt1 = self._find_center(img)
        pt2 = (size // 2, size // 2)
        dst = np.zeros((size, size))

        dx = abs(pt2[0] - pt1[0])
        dy = abs(pt2[1] - pt1[1])

        dst[dy : dy + img.shape[1], dx : dx + img.shape[0]] = img
        return dst

    def _classify_digits(self, img, digits_coords):
        img_rows, img_cols = 28, 28

        digits = []
        for x, y, w, h in digits_coords:
            digit = img[y : y + h, x : x + w]
            cv2.imwrite("out/digit.png", digit)

            squared_img = self._resize_image(~digit, size=(20, 20))
            centered_img = self._align_center(squared_img, size=28)
            digits.append(centered_img)
        digits = np.array(digits)

        if K.image_data_format() == "channels_first":
            digits = digits.reshape(digits.shape[0], 1, img_rows, img_cols)
        else:
            digits = digits.reshape(digits.shape[0], img_rows, img_cols, 1)

        digits = digits.astype("float32")
        digits /= 255

        pred = self.model.predict_classes(digits)
        return pred

    def _extract_digits(self, img):
        h, w = img.shape
        boxes_area = img[int(h * 0.095) : int(h * 0.14), int(w * 0.09) : int(w * 0.77)]

        # aplicando detector de harris para encontrar os cantos
        coords = corner_peaks(corner_harris(~boxes_area, k=0.03), min_distance=1, threshold_rel=0.3)

        if self.debug:
            # plotando todos os pontos encontrados
            fig, ax = plt.subplots(dpi=400)
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(boxes_area, cmap=plt.cm.gray)
            ax.plot(coords[:, 1], coords[:, 0], color="blue", marker="o", linestyle="None", markersize=2)

        # ordenandos os pontos de cantos encontrados
        # pegando os dois pontos mais a esquerda e os mais a direita horizantalmente
        coords = sorted(coords, key=lambda x: x[1], reverse=False)
        most_left = coords[:2]
        most_right = coords[-2:]
        points = np.array(most_left + most_right)

        if self.debug:
            # plotando os 4 pontos extremos
            ax.plot(points[:, 1], points[:, 0], color="red", marker="o", linestyle="None", markersize=3)
            ax.get_figure().savefig("out/corners.png")

        # trocando invertendo x e y (colunas)
        points.T[[0, 1]] = points.T[[1, 0]]

        # criando imagem colorida para visualizacao
        # boxes = cv2.cvtColor(boxes_area.copy(), cv2.COLOR_GRAY2BGR)

        # criando poligono convexo composto pelos pontos extremos
        # (gift wrapping algorithm)
        hull = cv2.convexHull(points)

        # desenhando poligono na imagem para visualizacao
        # cv2.drawContours(boxes, [hull], -1, (255, 0, 0), 1)

        # plotando imagem com poligono
        # fig, ax = plt.subplots(dpi=400)
        # ax.axis("off")
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.imshow(boxes, cmap=cm.gray)

        # removendo linhas do contorno externo
        mask = np.ones_like(boxes_area) * 255
        cv2.drawContours(mask, [hull], -1, 0, -1)
        # x, y, w, h = cv2.boundingRect(hull)

        dilated = cv2.dilate(mask, np.ones((9, 9), np.uint8))

        old = boxes_area.copy()
        boxes_area[dilated != 0] = 255  # pintando area externa da mascara de branco
        # boxes_area = cv2.bitwise_not(boxes_area)

        if self.debug:
            fig, ax = plt.subplots(4, 1, dpi=400)
            for a in ax:
                # a.axis("off")
                a.set_xticks([])
                a.set_yticks([])
            ax[0].imshow(old, cmap=cm.gray)
            ax[0].set_xlabel("Caixa original")
            ax[1].imshow(mask, cmap=cm.gray)
            ax[1].set_xlabel("Máscara")
            ax[2].imshow(dilated, cmap=cm.gray)
            ax[2].set_xlabel("Máscara erodida")
            ax[3].imshow(boxes_area, cmap=cm.gray)
            ax[3].set_xlabel("Caixa limiarizada")
            fig.savefig("out/box_no_exterior.png")

        # iterando a direita até encontrar divisoria
        hull_ys = np.sort(hull[:, 0, 1])
        miny, maxy = hull_ys[0], hull_ys[-1]

        hull_xs = np.sort(hull[:, 0, 0])
        minx, maxx = hull_xs[0], hull_xs[-1]

        box_len = maxx - minx
        span = box_len // 11

        _, contours, _ = cv2.findContours(~boxes_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            test_img = cv2.cvtColor(boxes_area, cv2.COLOR_GRAY2BGR)
            # cv2.drawContours(test_img, contours, -1, (0, 0, 255), 1)

        x = minx + span
        mid_y = (maxy + miny) // 2
        cont = 0

        div_contours = []
        while x <= maxx and cont < 10:
            pixel = boxes_area[mid_y, x]
            if self.debug:
                cv2.circle(test_img, (x, mid_y), 2, (0, 255, 0), -1)
            # print(pixel)
            if pixel == 0:
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, (x, mid_y), measureDist=False) >= 0:
                        div_contours.append(cnt)
                        if self.debug:
                            cv2.drawContours(test_img, [cnt], -1, (255, 0, 0), 1)
                x += span
                cont += 1
            else:
                if self.debug:
                    cv2.line(test_img, (x, miny), (x, maxy), (0, 0, 255), 1)
                x += 1

        # removendo divisorias encontradas
        mask = np.ones_like(boxes_area) * 255
        cv2.drawContours(mask, div_contours, -1, 0, -1)
        boxes_area[mask == 0] = 255

        if self.debug:
            fig, ax = plt.subplots(dpi=400)
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(test_img, cmap=cm.gray)
            fig.savefig("out/divs.png")

            fig, ax = plt.subplots(dpi=400)
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(boxes_area, cmap=cm.gray)
            fig.savefig("out/digits.png")

        # dilatadando a imagem, encontrando os contornos, e os ordenando
        selem = disk(1)
        ref = dilation(~boxes_area, selem)

        _, contours, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours.sort(key=cv2.contourArea, reverse=True)
        contours = contours[:11]
        contours = list((sort_contours(contours, "left-to-right"))[0])

        if self.debug:
            ref = cv2.cvtColor(boxes_area, cv2.COLOR_GRAY2BGR)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            if self.debug:
                cv2.rectangle(ref, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if self.debug:
            fig, ax = plt.subplots(dpi=400)
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(ref, cmap=cm.gray)
            fig.savefig("out/digits_box.png")

        return boxes_area, boxes

    def correct(self, filepath: str):
        self._clear()

        img = self._read_and_preprocess_image(filepath)

        digits_img, digits = self._extract_digits(img)
        pred = self._classify_digits(digits_img, digits)
        print(pred)

        min_circle_radius = int(20 * self.factor)
        max_circle_radius = int(25 * self.factor)

        h, w = img.shape

        # content = img[divisor_line:, :]
        content = img[int(h * 0.14) :, :]

        if self.debug:
            cv2.imwrite("out/sheet.png", content)

        circles = cv2.HoughCircles(
            image=content,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int((2 * min_circle_radius) + (6 * self.factor)),
            param1=200,
            param2=7,
            minRadius=min_circle_radius,
            maxRadius=max_circle_radius,
        )
        print(circles.shape)

        assert circles.shape[1] == 600, f"Erro: não foram detectados exatos 600 círculos. {circles.shape}"

        # sorting circles
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda v: [v[0], v[1]])

        NUM_ROWS = 30

        sorted_cols = []
        for k in range(0, len(circles), NUM_ROWS):
            col = circles[k : k + NUM_ROWS]
            sorted_cols.extend(sorted(col, key=lambda v: v[1]))

        circles = sorted_cols
        ############

        if self.debug:
            img_colored = cv2.cvtColor(content, cv2.COLOR_GRAY2BGR)

        circles = np.uint16(np.around(circles))
        for cont, i in enumerate(circles, start=0):
            points = self._points_in_circle_np(x0=i[1], y0=i[0], radius=i[2])
            mode = self._points_mode_value(content, points)

            if mode == 255:
                if self.debug:
                    cv2.circle(img_colored, (i[0], i[1]), i[2], (0, 0, 255), 2)
            else:
                if self.debug:
                    cv2.circle(img_colored, (i[0], i[1]), i[2], (0, 255, 0), 2)

                self.ans[self.seq[cont][0]].append(self.seq[cont][1])

            # if self.debug:
            #     cv2.putText(
            #         img_colored,
            #         str(cont + 1),
            #         (i[0] - i[2], i[1] + i[2]),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.4,
            #         (0, 0, 0),
            #         1,
            #         cv2.LINE_AA,
            #     )

        if self.debug:
            cv2.imwrite("out/out.png", img_colored)

        self.ans = list(self.ans.items())
        self.ans = sorted(self.ans, key=itemgetter(0))
        with open("out.json", "w") as fp:
            json.dump(self.ans, fp)
        return self.ans


if __name__ == "__main__":
    file = sys.argv[1]

    t0 = time.perf_counter()
    resp = SheetCorrector(debug=True).correct(file)
    tf = time.perf_counter()

    print(tf - t0)
    # print(pprint(resp[0:30]))
