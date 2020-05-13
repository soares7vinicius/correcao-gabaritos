import cv2
import numpy as np
from sklearn.cluster import KMeans
import time


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


def main():
    img = cv2.imread("content.png",)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=22,
        param1=200,
        param2=7,
        minRadius=10,
        maxRadius=12,
    )
    print(circles.shape)

    circles = np.round(circles[0, :]).astype("int")

    xs = []
    ys = []
    for c in circles:
        ys.append(c[0])
        xs.append(c[1])

    xs = np.array(xs).reshape(-1, 1)
    kmeans_x = KMeans(n_clusters=30, max_iter=50).fit(xs)
    centers_x = np.round(kmeans_x.cluster_centers_).astype("int")

    ys = np.array(ys).reshape(-1, 1)
    kmeans_y = KMeans(n_clusters=20, max_iter=50).fit(ys)
    centers_y = np.round(kmeans_y.cluster_centers_).astype("int")

    ncircles = []
    for c in circles:
        y = min(centers_y, key=lambda v: abs(v - c[0]))[0]
        x = min(centers_x, key=lambda v: abs(v - c[1]))[0]
        ncircles.append([y, x, c[2]])

    circles = sorted(ncircles, key=lambda v: [v[0], v[1]])

    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cont, i in enumerate(circles, start=1):
            points = points_in_circle_np(x0=i[1], y0=i[0], radius=i[2])
            mode = points_mode_value(img, points)

            if mode == 255:
                cv2.circle(
                    img_colored, (i[0], i[1]), i[2], (0, 0, 255), -1
                )  # cv2.FILLED)
            else:
                cv2.circle(
                    img_colored, (i[0], i[1]), i[2], (0, 255, 0), -1
                )  # cv2.FILLED)

            cv2.putText(
                img_colored,
                str(cont),
                (i[0] - i[2], i[1] + i[2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite("out.png", img_colored)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    tf = time.perf_counter()
    print(tf - t0)
