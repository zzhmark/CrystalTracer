import os

import cython
cimport cython
cimport numpy as np
from libcpp.vector cimport vector
# from libcpp.queue cimport queue
# from libcpp.pair cimport pair
# from sklearn.neighbors import KDTree
import pandas as pd
from skimage.morphology import dilation, disk


from findmaxima2d import find_maxima, find_local_maxima
import cv2
from scipy.ndimage import generate_binary_structure
from gwdt import gwdt
import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour, disk_level_set
from skimage.filters import difference_of_gaussians
from skimage.feature._canny import _preprocess
import scipy.ndimage as ndi


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef frame_detection(np.ndarray[np.uint8_t, ndim=2]  gfp, np.ndarray[np.uint8_t, ndim=2] bf, shift=(0, 0),
                      gfp_scale_range=(.0001, .9999), dog_sigma=1., sobel_sigma=1., gfp_weight=.1, dilation_radius=2):
    """
    Detection chips for a single frame.

    :param gfp: gfp image, uint8
    :param bf: bf image, uint8
    :param shift: the shift of gfp from the bf image, (x, y)
    :param gfp_scale_range: gfp image scaling range
    :param dog_sigma: sigma for difference of gaussian on bf. the larger sigma will be 1.6x this value.
    :param sobel_sigma: sigma on gfp before sobel. makes the edge more smooth
    :param gfp_weight: weight of gfp edges to add on bf
    :param dilation_radius: the radius to dilate the segmentation
    :return: DataFrame, segmentation
    """
    # loading the gfp image, rescale
    m = np.quantile(gfp, gfp_scale_range[0]), np.quantile(gfp, gfp_scale_range[1])
    gfp = ((gfp.clip(m[0], m[1]) - m[0]) / (m[1] - m[0]) * 255).astype(np.uint8)

    # distance transform and find centers
    thr = cv2.threshold(gfp, 0, 255, cv2.THRESH_TRIANGLE)[0]
    fgnd_img = gfp - thr
    fgnd_img[fgnd_img < 0] = 0
    structure = generate_binary_structure(gfp.ndim, 10)
    dt = gwdt(fgnd_img, structure)
    y, x, regs = find_maxima(dt, find_local_maxima(dt), 10)

    # loading the bf image and merge them and get an edge map
    # the gfp image provides the sobel magnitude
    # the bf image provides the edges
    dog = difference_of_gaussians(bf, dog_sigma)
    smoothed, eroded_mask = _preprocess(gfp, None, sobel_sigma, 'constant', 0)
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    magnitude = isobel * isobel
    magnitude += jsobel * jsobel
    np.sqrt(magnitude, out=magnitude)
    magnitude = cv2.warpAffine(magnitude, np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=float),
                               (gfp.shape[1], gfp.shape[0]))
    cdef np.ndarray[np.float64_t, ndim=2] edges = (dog - magnitude * gfp_weight).astype(float)

    # rad is for initiating contours
    rad = radius_estimate(gfp, y, x, lowest_cutoff=thr, cutoff_ratio=0.5, bg_thr=0.0001)
    seg = []
    cdef:
        vector[float] area, new_y, new_x
        int i, win_rad, ys, xs, ye, xe, a
        np.ndarray[np.int64_t, ndim=1] x_, y_
        np.ndarray[np.int8_t, ndim=2] ls
    for i, (yy, xx, rr) in enumerate(zip(y, x, rad)):
        win_rad = int(rr * 2)
        ys = max(0, yy - win_rad)
        xs = max(0, xx - win_rad)
        ye = min(bf.shape[0], yy + win_rad)
        xe = min(bf.shape[1], xx + win_rad)
        sub_img = edges[ys:ye, xs:xe].astype(float)
        ls = morphological_geodesic_active_contour(sub_img, 20, disk_level_set(sub_img.shape, radius=rr))
        ls = dilation(ls, disk(dilation_radius))
        a = np.sum(ls > 0)
        if a > 0:
            area.push_back(a)
            seg.append(ls)
            y_, x_ = np.nonzero(ls)
            new_y.push_back(y_.mean() + ys)
            new_x.push_back(x_.mean() + xs)

    return pd.DataFrame({'y': new_y, 'x': new_x, 'area': area}), seg


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def radius_estimate(img, y, x, cutoff_ratio=0.5, bg_thr=.001, lowest_cutoff=0):
    """
    Estimate radius of d circles

    :param lowest_cutoff: the threshold between chips and background
    :param bg_thr: the rate of bg pixel detected to stop the radius enlargement
    :param img: the input image, gfp
    :param y: y coords
    :param x: x coords
    :param cutoff_ratio: the threshold based on the chip intensity is determined by this rate
    :return: a list of radius
    """
    assert len(x) == len(y)
    cdef:
        int tot_cand = len(y), i, j, k, width = img.shape[1], height = img.shape[0], tot, bg, dy, dx
        float cr = cutoff_ratio, thr, r, bg_th = bg_thr, lc = lowest_cutoff
        np.ndarray[np.uint8_t, ndim=2, cast=True] img_c = img
        vector[int] y_c = y, x_c = x, rad = [0] * tot_cand
    for i in range(tot_cand):
        tot = bg = 0
        thr = img_c[y_c[i], x_c[i]] * cr
        if lc > thr:
            thr = lc
        while True:
            rad[i] += 1
            for dy in range(-rad[i], rad[i] + 1):
                for dx in range(-rad[i], rad[i] + 1):
                    r = (dy**2 + dx**2)**.5
                    tot += 1
                    if rad[i] - 1 < r <= rad[i]:
                        j = y_c[i] + dy
                        k = x_c[i] + dx
                        if not 0 <= j < height or not 0<= k < width:    # hit the border
                            break
                        if img_c[j, k] < thr:
                            bg += 1
                            if float(bg) / tot > bg_th:
                                break
                else:
                    continue
                break
            else:
                continue
            break
    return rad


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# def area_estimate(img, y, x, rad_py, cutoff_ratio=0.5, lowest_cutoff=0):
#     """
#     Count area by flooding, based on radius estimate results.
#     :param lowest_cutoff:
#     :param img:
#     :param y:
#     :param x:
#     :param cutoff_ratio:
#     :return:
#     """
#     assert len(x) == len(y) == len(rad_py)
#     cdef:
#         int tot_cand = len(y), i, j, k
#         float cr = cutoff_ratio, thr, lc = lowest_cutoff
#         np.ndarray[np.uint8_t, ndim=2, cast=True] img_c = img
#         np.ndarray[np.int32_t, ndim=2] flood
#         vector[int] y_c = y, x_c = x, area = [1] * tot_cand, rad = rad_py
#         vector[pair[int, int]] dire = [(1, 0), (-1, 0), (0, 1), (0, -1)]
#         queue[pair[int, int]] que
#         pair[int, int] t, tt
#         int ys, xs, ye, se, height, width
#     floods = []
#     for i in range(tot_cand):
#         ys = max(0, y_c[i] - rad[i] * 2)
#         xs = max(0, x_c[i] - rad[i] * 2)
#         ye = min(img_c.shape[0], y_c[i] + rad[i] * 2)
#         xe = min(img_c.shape[1], x_c[i] + rad[i] * 2)
#         height = ye - ys
#         width = xe - xs
#         crop = img_c[ys:ye, xs:xe]
#         flood = np.zeros_like(crop, dtype=int)
#         t = y_c[i] - ys, x_c[i] - xs
#         cv2.circle(flood, (x_c[i] - xs, y_c[i] - ys), int(rad[i]), 1, -1)
#         que.push(t)
#         thr = crop[t.first, t.second] * cr
#         if lc > thr:
#             thr = lc
#         while not que.empty():
#             t = que.front()
#             for d in dire:
#                 tt = t.first + d.first, t.second + d.second
#                 if height > tt.first >= 0 and width > tt.second >= 0 and \
#                         crop[tt.first, tt.second] >= thr and flood[tt.first, tt.second] == 1:
#                     que.push(tt)
#                     area[i] += 1
#                     flood[tt.first, tt.second] = 255
#             que.pop()
#         flood[flood == 1] = 0
#         floods.append((ys, xs, flood))
#     return area, floods

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# def merge_centers(img, y, x, gap=.9, merge_dist=50):
#     """
#     Merge centers if there's no major change in the profile between them.
#     :param img:
#     :param y:
#     :param x:
#     :param gap:
#     :param merge_dist:
#     :return:
#     """
#     assert len(x) == len(y)
#     if not isinstance(merge_dist, list):
#         merge_dist = [merge_dist] * len(y)
#     cdef:
#         int tot_cand = len(y), i, j, sy, sx, vmax, vmin, t
#         np.ndarray[np.uint8_t, ndim=2, cast=True] img_c = img
#         np.ndarray[np.int64_t, ndim=2] data = np.transpose([y, x])
#         vector[vector[int]] conn = [tuple()] * tot_cand
#         bint flag
#         np.ndarray[np.int64_t, ndim=1] start_pos, end_pos, k, pos, tpos
#         vector[bint] flags = [False] * tot_cand
#         vector[int] ans_y, ans_x, current_node
#         queue[int] que
#     tree = KDTree(data)
#     query = []
#     for y, x, d in zip(y, x, merge_dist):
#         query.append(tree.query_radius([[y, x]], d)[0])
#     for i in range(len(query)):
#         start_pos = data[i]
#         for j in query[i]:
#             if j == i:
#                 continue
#             # check if already profiled
#             if j < i:
#                 flag = False
#                 for t in conn[j]:
#                     if t == i:
#                         flag = True
#                         break
#                 if flag:
#                     conn[i].push_back(j)
#                 continue
#             end_pos = data[j]
#             k = end_pos - start_pos
#             pos = start_pos.copy()
#             tpos = pos - start_pos
#             sy = 1 if k[0] > 0 else -1
#             sx = 1 if k[1] > 0 else -1
#             vmax = vmin = img_c[end_pos[0], end_pos[1]]
#             # profiling
#             while tpos[0] != k[0] and tpos[1] != k[1]:
#                 t = img_c[pos[0], pos[1]]
#                 if t > vmax:
#                     vmax = t
#                 if t < vmin:
#                     vmin = t
#                 if abs(tpos[1] * k[0]) > abs(tpos[0] * k[1]):
#                     pos[0] += sy
#                 else:
#                     pos[1] += sx
#                 tpos = pos - start_pos
#             if vmax - vmin < gap * vmax:
#                 conn[i].push_back(j)
#     # connect
#     for i in range(tot_cand):
#         if flags[i]:
#             continue
#         current_node = [i]
#         sy = 0
#         sx = 0
#         que.push(i)
#         flags[i] = True
#         while not que.empty():
#             for j in conn[que.front()]:
#                 if not flags[j]:
#                     current_node.push_back(j)
#                     que.push(j)
#                     flags[j] = True
#             que.pop()
#         for j in current_node:
#             sy += data[j, 0]
#             sx += data[j, 1]
#         ans_y.push_back(sy / current_node.size())
#         ans_x.push_back(sx / current_node.size())
#
#     return ans_y, ans_x
