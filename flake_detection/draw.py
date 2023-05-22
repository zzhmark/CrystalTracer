import cv2
import numpy as np


def draw_circles(img: np.ndarray, y, x, rad, color=((0, 0, 255),)):
    if len(color) == 1:
        color = color * len(y)
    assert len(rad) == len(y) == len(x) == len(color)
    if len(img.shape) == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    for i, j, k, t in zip(y, x, rad, color):
        cv2.circle(out, (j, i), k, t, 1)
    return out


def draw_areas(img: np.ndarray, flood: np.ndarray):
    if len(img.shape) == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    for i in range(flood.max()):
        out[flood == i] = np.random.choice(range(255), size=3)
    return out
