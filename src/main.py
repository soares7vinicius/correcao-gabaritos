import json
import sys
import time
from operator import itemgetter
from pprint import pprint
from typing import List, Tuple

import cv2
import numpy as np


class SheetCorrector:
    def __init__(self, factor=0.5, debug=True):
        self.seq = self._generate_circles_index_sequence()
        self.debug = debug
        self.factor = factor

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

    def correct(self, filepath: str):
        self._clear()

        img = cv2.imread(filepath,)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape

        height = int(height * self.factor)
        width = int(width * self.factor)
        min_circle_radius = int(20 * self.factor)
        max_circle_radius = int(25 * self.factor)

        img = cv2.resize(img, (width, height))
        _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

        divisor_line = 205
        header = img[:divisor_line, :]
        content = img[divisor_line:, :]

        circles = cv2.HoughCircles(
            image=content,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int((2 * min_circle_radius) + (5 * self.factor)),
            param1=200,
            param2=7,
            minRadius=min_circle_radius,
            maxRadius=max_circle_radius,
        )

        if circles.shape[1] != 600:
            print("Erro: não foram detectados exatos 600 círculos. ", circles.shape)
            return

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
                    cv2.circle(img_colored, (i[0], i[1]), i[2], (0, 0, 255), -1)
            else:
                if self.debug:
                    cv2.circle(img_colored, (i[0], i[1]), i[2], (0, 255, 0), -1)

                self.ans[self.seq[cont][0]].append(self.seq[cont][1])

            if self.debug:
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

        if self.debug:
            cv2.imwrite("out.png", img_colored)

        self.ans = list(self.ans.items())
        self.ans = sorted(self.ans, key=itemgetter(0))
        with open("out.json", "w") as fp:
            json.dump(self.ans, fp)
        return self.ans


if __name__ == "__main__":
    file = sys.argv[1]

    t0 = time.perf_counter()
    resp = SheetCorrector().correct(file)
    tf = time.perf_counter()

    print(tf - t0)
    print(pprint(resp[0:30]))
