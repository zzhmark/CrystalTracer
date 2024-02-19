import os

import cython
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.pair cimport pair
import pandas as pd
from skimage.morphology import dilation, disk


from findmaxima2d import find_maxima, find_local_maxima
import cv2
from scipy.ndimage import generate_binary_structure
from crystal_tracer.algorithm.gwdt import gwdt
import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour, disk_level_set
from skimage.filters import difference_of_gaussians, threshold_local
from skimage.feature._canny import _preprocess
from skimage.util import img_as_ubyte
import scipy.ndimage as ndi
import sys

class HidePrint:
    def __init__(self):
        self.origin = None
    def __enter__(self):
        sys.stdout.close()
        sys.stdout = self.origin

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.origin = sys.stdout
        sys.stdout = open(os.devnull, 'w')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef frame_detection(np.ndarray gfp, np.ndarray bf, thr_blk_sz=31, tolerance=10, cutoff_ratio=.1, bg_thr=.0001,
                      active_contour=True,
                      shift=(0, 0), dog_sigma=1., sobel_sigma=1., bf_weight=1., gfp_weight=.1, dilation_radius=2):
    """
    Detect and segment crystals in a single frame, using gfp and bf images. When only gfp is available, you can use it
    as bf as well. The bf image will only be used in active contour.

    :param gfp: fluorescent image
    :param bf: bright field image, if you don't have the bf image, use the gfp here
    :param tolerance: for tuning the sensitivity in maxima finding
    :param cutoff_ratio: the intensity below this ratio * center intensity will be regarded as background in radius estimation
    :param bg_thr: No. of background pixel exceeding this limit will cause radius estimation to stop
    :param thr_blk_sz: adaptive thresholding field size, applied on gfp image
    :param active_contour: whether to use active contour to get accurate prediction of area, otherwise only predict radius
    :param shift: the shift of gfp from the bf image, (x, y)
    :param dog_sigma: sigma for difference of gaussian on bf. the larger sigma will be 1.6x this value.
    :param sobel_sigma: sigma on gfp before sobel. makes the edge more smooth
    :param bf_weight: weight of the bf image
    :param gfp_weight: weight of gfp edges to add on bf
    :param dilation_radius: the radius to dilate the segmentation, 0 to turn off.
    :return: DataFrame, segmentation
    """

    # distance transform and find centers
    fgnd_img = (gfp - threshold_local(gfp, thr_blk_sz)).clip(0)
    fgnd_img = img_as_ubyte(fgnd_img / fgnd_img.max())
    # cv2.imwrite('../../data/fgnd.tif', fgnd_img)
    structure = generate_binary_structure(gfp.ndim, 10)
    with HidePrint():
        dt = gwdt(fgnd_img, structure)
        y, x, regs = find_maxima(dt, find_local_maxima(dt), tolerance)
    rad = radius_estimate(fgnd_img, y, x, cutoff_ratio=cutoff_ratio, bg_thr=bg_thr)
    cdef np.ndarray[np.float32_t, ndim=2] edges
    cdef:
        vector[float] area, new_y, new_x
        int i, win_rad, ys, xs, ye, xe, a
        np.ndarray[np.int64_t, ndim=1] x_, y_
        np.ndarray[np.int8_t, ndim=2] ls

    if active_contour:
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
        edges = (bf_weight * dog - magnitude * gfp_weight).astype(np.float32)

        # rad is for initiating contours
        seg = []
        for i, (yy, xx, rr) in enumerate(zip(y, x, rad)):
            win_rad = int(rr * 2)
            ys = max(0, yy - win_rad)
            xs = max(0, xx - win_rad)
            ye = min(bf.shape[0], yy + win_rad)
            xe = min(bf.shape[1], xx + win_rad)
            sub_img = edges[ys:ye, xs:xe].astype(float)
            ls = morphological_geodesic_active_contour(sub_img, 20, disk_level_set(sub_img.shape, radius=rr))
            if dilation_radius > 0:
                ls = dilation(ls, disk(dilation_radius))
            a = np.sum(ls > 0)
            if a > 0:
                area.push_back(a)
                seg.append(ls)
                y_, x_ = np.nonzero(ls)
                new_y.push_back(y_.mean() + ys)
                new_x.push_back(x_.mean() + xs)

        return pd.DataFrame({'y': new_y, 'x': new_x, 'area': area}), seg

    seg = area_estimate(fgnd_img, y, x, rad, cutoff_ratio=.1)

    return pd.DataFrame({'y': y, 'x': x, 'radius': rad}), seg


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def area_estimate(img, y, x, rad_py, cutoff_ratio=0.5, lowest_cutoff=0):
    """
    Count area by flooding, based on radius estimate results.
    :param lowest_cutoff:
    :param img:
    :param y:
    :param x:
    :param cutoff_ratio:
    :return:
    """
    assert len(x) == len(y) == len(rad_py)
    cdef:
        int tot_cand = len(y), i, j, k
        float cr = cutoff_ratio, thr, lc = lowest_cutoff
        np.ndarray[np.uint8_t, ndim=2, cast=True] img_c = img
        np.ndarray[np.int32_t, ndim=2] flood
        vector[int] y_c = y, x_c = x, area = [1] * tot_cand, rad = rad_py
        vector[pair[int, int]] dire = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        queue[pair[int, int]] que
        pair[int, int] t, tt
        int ys, xs, ye, se, height, width
    floods = []
    for i in range(tot_cand):
        ys = max(0, y_c[i] - rad[i] * 2)
        xs = max(0, x_c[i] - rad[i] * 2)
        ye = min(img_c.shape[0], y_c[i] + rad[i] * 2)
        xe = min(img_c.shape[1], x_c[i] + rad[i] * 2)
        height = ye - ys
        width = xe - xs
        crop = img_c[ys:ye, xs:xe]
        flood = np.zeros_like(crop, dtype=int)
        t = y_c[i] - ys, x_c[i] - xs
        cv2.circle(flood, (x_c[i] - xs, y_c[i] - ys), int(rad[i]), 1, -1)
        que.push(t)
        thr = crop[t.first, t.second] * cr
        if lc > thr:
            thr = lc
        while not que.empty():
            t = que.front()
            for d in dire:
                tt = t.first + d.first, t.second + d.second
                if height > tt.first >= 0 and width > tt.second >= 0 and \
                        crop[tt.first, tt.second] >= thr and flood[tt.first, tt.second] == 1:
                    que.push(tt)
                    area[i] += 1
                    flood[tt.first, tt.second] = 255
            que.pop()
        flood[flood == 1] = 0
        floods.append((ys, xs, flood))
    return area, floods


