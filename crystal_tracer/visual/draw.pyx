import cv2
import numpy as np
import cython
cimport cython
cimport numpy as cnp
import random
from colorsys import hls_to_rgb


cpdef draw_circles(cnp.ndarray[cnp.uint8_t, ndim=3] img, y: list[int], x: list[int], rad: list[int], color=(255, 0, 0)):
    """

    :param img: the background image
    :param y: y coordinates of circle center
    :param x: x coordinates of circle center
    :param rad: radii of circles
    :param color: color of each circle, in (r, g, b), 0-255. can be set as a single color
    :return:
    """
    assert len(rad) == len(y) == len(x) == len(color)
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cdef i, j, r, c
    for i, j, r, c in zip(y, x, rad, color):
        c = tuple(reversed(c))
        cv2.circle(out, (j, i), r, c, 1)
    return out


cpdef mask_areas(cnp.ndarray[cnp.uint8_t, ndim=2] img, cnp.ndarray mask):
    cdef cnp.ndarray[cnp.int_t, ndim=2] m = mask
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), i
    for i in range(m.max()):
        out[m == i] = np.random.choice(range(255), size=3)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef draw_patches(cnp.ndarray[cnp.uint8_t, ndim=2] img, y: list[int], x: list[int], masks: list[cnp.ndarray],
                   colors=()):
    """
    
    :param img: an image as the background
    :param y: the y shift of the patches, i.e. the y coord of the first element of the patch
    :param x: the x shift of the patches, i.e. the x coord of the first element of the patch
    :param masks: the patches
    :param colors: the color of each patch, in (r, g, b), 0-255. if let empty, will be randomly designated.
    :return: a color image
    """
    assert len(y) == len(x) == len(masks)
    if not colors:
        colors = []
        for c in range(len(y)):
            rgb = hls_to_rgb(random.random(), .5, random.random()*.6 + .2)
            colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    elif len(colors) == 1:
        colors = colors * len(x)
    else:
        raise ValueError("No. of colors doesn't match the number of patches.")
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] mask_temp
    cdef int m, n, hm, hn, i, j, i1, j1, r, g, b
    for cty, ctx, mask, c in zip(y, x, masks, colors):
        m, n = mask.shape
        hm = m // 2
        hn = n // 2
        mask_temp = mask
        r, g, b = c

        for i in range(m):
            for j in range(n):
                if mask_temp[i, j]:
                    i1 = cty + i
                    j1 = ctx + j
                    out[i1, j1, 0] = b
                    out[i1, j1, 1] = g
                    out[i1, j1, 2] = r

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef draw_contour(cnp.ndarray[cnp.uint8_t, ndim=2] img, cnp.ndarray mask, color=(255, 0, 0), id=1):
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cdef cnp.ndarray[cnp.npy_bool, ndim=2] seg = (mask == id), mask2 = np.zeros_like(mask, dtype=bool)
    cdef int i, j, i1, j1
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if seg[i, j]:
                for i1 in range(-1, 2):
                    for j1 in range(-1, 2):
                        if not seg[i + i1, j + j1] and (i1 != 0 or j1 != 0):
                            mask2[i, j] = True
                            break
                    else:
                        continue
                    break
    color = tuple(reversed(color))
    out[mask2] = color
    return out