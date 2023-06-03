import cv2
import numpy as np
import cython
cimport cython
cimport numpy as np


def draw_circles(img: np.ndarray, y, x, rad, color=((0, 0, 255),)):
    if len(color) == 1:
        color = color * len(y)
    assert len(rad) == len(y) == len(x) == len(color)
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    for i, j, k, t in zip(y, x, rad, color):
        cv2.circle(out, (j, i), k, t, 1)
    return out


def draw_areas(img: np.ndarray, flood: np.ndarray):
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    for i in range(flood.max()):
        out[flood == i] = np.random.choice(range(255), size=3)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef draw_contour(np.ndarray img, np.ndarray flood, color=(0, 0, 255), id=1):
    cdef np.ndarray[np.uint8_t, ndim=3] out
    cdef np.ndarray[np.npy_bool, ndim=2] seg = flood == id, mask = np.zeros_like(flood, dtype=bool)
    cdef int i, j, i1, j1
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if seg[i, j]:
                for i1 in range(-1, 2):
                    for j1 in range(-1, 2):
                        if not seg[i + i1, j + j1] and (i1 != 0 or j1 != 0):
                            mask[i, j] = True
                            break
                    else:
                        continue
                    break
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    out[mask] = color
    return out