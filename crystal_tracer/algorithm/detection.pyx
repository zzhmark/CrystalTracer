import cython
cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.pair cimport pair
import pandas as pd
from skimage.morphology import dilation, disk
from libc.math cimport sqrt
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef frame_detection(cnp.ndarray gfp, cnp.ndarray bf, int thr_blk_sz=31, int tolerance=10, float cutoff_ratio=.1,
                      float bg_thr=.0001, bint active_contour=True, tuple[int, int] shift=(0, 0), float dog_sigma=1.,
                      float sobel_sigma=1., float bf_weight=1., float gfp_weight=.1, int dilation_radius=2):
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
    :return: DataFrame, a list of masks
    """

    # distance transform and find centers
    fgnd_img = (gfp - threshold_local(gfp, thr_blk_sz)).clip(0)
    fgnd_img = img_as_ubyte(fgnd_img / fgnd_img.max())
    structure = generate_binary_structure(gfp.ndim, 10)
    dt = gwdt(fgnd_img, structure)
    y, x, regs = find_maxima(dt, find_local_maxima(dt), tolerance)
    rad = radius_estimate(fgnd_img, y, x, cutoff_ratio=cutoff_ratio, bg_thr=bg_thr)
    new_y, new_x, new_rad, area, ys_list, xs_list, masks = \
        contour_estimate(gfp, bf, y, x, rad, shift, dog_sigma,
                         sobel_sigma, bf_weight, gfp_weight, dilation_radius) if active_contour else \
            flood_estimate(fgnd_img, y, x, rad, cutoff_ratio=cutoff_ratio)
    return pd.DataFrame(
        {'y': new_y, 'x': new_x, 'radius': new_rad, 'area': area, 'y_start': ys_list, 'x_start': xs_list}), masks


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef contour_estimate(cnp.ndarray gfp, cnp.ndarray bf, const vector[int]& y, const vector[int]& x,
                       const vector[int]& rad, tuple[int, int] shift=(0, 0), float dog_sigma=1.,
                       float sobel_sigma=1., float bf_weight=1., float gfp_weight=.1, int dilation_radius=2):
    cdef:
        int i, win_rad, ys, xs, ye, xe, area, height = gfp.shape[0], width = gfp.shape[1]
        cnp.ndarray[cnp.float32_t, ndim=2] edges
        vector[float] new_y, new_x
        vector[int] new_rad, ys_list, xs_list, area_list
        cnp.ndarray[cnp.int64_t, ndim=1] x_, y_
        cnp.ndarray[cnp.uint8_t, ndim=2] ls
        list masks = []
    # loading the bf image and merge them and get an edge map
    # the gfp image provides the sobel magnitude
    # the bf image provides the edges
    dog = difference_of_gaussians(bf, dog_sigma)
    smoothed = _preprocess(gfp, None, sobel_sigma, 'constant', 0)[0]
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    magnitude = isobel * isobel + jsobel * jsobel
    np.sqrt(magnitude, out=magnitude)
    magnitude = cv2.warpAffine(magnitude, np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=float),
                               (gfp.shape[1], gfp.shape[0]))
    edges = (bf_weight * dog - magnitude * gfp_weight).astype(np.float32)

    for i in range(y.size()):
        win_rad = rad[i] * 2
        ys = max(0, y[i] - win_rad)
        xs = max(0, x[i] - win_rad)
        ye = min(height, y[i] + win_rad)
        xe = min(width, x[i] + win_rad)
        ls = morphological_geodesic_active_contour(
            edges[ys:ye, xs:xe], 20, disk_level_set((ye-ys, xe-xs), radius=rad[i])).astype(np.uint8)
        if dilation_radius > 0:
            ls = dilation(ls, disk(dilation_radius))
        area = np.sum(ls > 0)
        if area > 0:
            ys_list.push_back(ys)
            xs_list.push_back(xs)
            area_list.push_back(area)
            new_rad.push_back(rad[i])
            masks.append(ls)
            y_, x_ = np.nonzero(ls)
            new_y.push_back(y_.mean() + ys)
            new_x.push_back(x_.mean() + xs)
    return new_y, new_x, new_rad, area, ys_list, xs_list, masks


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef radius_estimate(cnp.ndarray[cnp.uint8_t, ndim=2] img, const vector[int]& y, const vector[int]& x,
                      float cutoff_ratio=0.5, float bg_thr=.001, float lowest_cutoff=0):
    """
    Estimate the radii of crystals

    :param img: the input fluorescent image
    :param y: y coords of the seeds
    :param x: x coords of the seeds
    :param cutoff_ratio: the threshold based on the chip intensity is determined by this rate
    :param bg_thr: the rate of bg pixel detected to stop the radius enlargement
    :param lowest_cutoff: the cutoff should be more than this
    :return: a list of radii for each seed
    """
    cdef:
        int i, j, k, width = img.shape[1], height = img.shape[0], tot_count, bg_count, dy, dx
        vector[int] rad

    for i in range(y.size()):
        rad.push_back(0)
        tot_count = bg_count = 0
        thr = max(img[y[i], x[i]] * cutoff_ratio, lowest_cutoff)
        while True:
            rad[i] += 1
            for dy in range(-rad[i], rad[i] + 1):
                for dx in range(-rad[i], rad[i] + 1):
                    r = sqrt(dy*dy + dx*dx)
                    tot_count += 1
                    if rad[i] - 1 < r <= rad[i]:
                        j = y[i] + dy
                        k = x[i] + dx
                        if not 0 <= j < height or not 0<= k < width:    # hit the border
                            break
                        if img[j, k] < thr:
                            bg_count += 1
                            if float(bg_count) / tot_count > bg_thr:
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
cpdef flood_estimate(cnp.ndarray[cnp.uint8_t, ndim=2] img, const vector[int]& y, const vector[int]& x,
                     const vector[int]& rad, float cutoff_ratio=0.5, float lowest_cutoff=0):
    """
    Measure crystal sizes by flooding, based on radius estimate results.
    :param img: input fluorescent image
    :param y: y coordinates of seeds
    :param x: x coordinates of seeds
    :param rad: radii of the candidate crystals
    :param cutoff_ratio: the ratio between the seed intensity and background
    :param lowest_cutoff: the cutoff should be more than this
    :return:
    """
    cdef:
        int i, j, k, ys, xs, ye, se, height, width, area
        float thr
        cnp.ndarray[cnp.int32_t, ndim=2] flood
        vector[float] new_y, new_x
        vector[int] area_list, ys_list, xs_list, new_rad
        vector[pair[int, int]] shift = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        queue[pair[int, int]] que
        pair[int, int] t, tt
        list masks = []

    for i in range(y.size()):
        ys = max(0, y[i] - rad[i] * 2)
        xs = max(0, x[i] - rad[i] * 2)
        ye = min(img.shape[0], y[i] + rad[i] * 2)
        xe = min(img.shape[1], x[i] + rad[i] * 2)
        height = ye - ys
        width = xe - xs
        crop = img[ys:ye, xs:xe]
        flood = np.zeros_like(crop, dtype=int)
        t = y[i] - ys, x[i] - xs
        cv2.circle(flood, (x[i] - xs, y[i] - ys), int(rad[i]), 1, -1)
        que.push(t)
        thr = crop[t.first, t.second] * cutoff_ratio
        if lowest_cutoff > thr:
            thr = lowest_cutoff
        area = 0
        while not que.empty():
            t = que.front()
            for d in shift:
                tt = t.first + d.first, t.second + d.second
                if height > tt.first >= 0 and width > tt.second >= 0 and \
                        crop[tt.first, tt.second] >= thr and flood[tt.first, tt.second] == 1:
                    que.push(tt)
                    area += 1
                    flood[tt.first, tt.second] = 255
            que.pop()
        flood[flood == 1] = 0
        masks.append(flood.astype(np.uint8))
        area_list.push_back(area)
        ys_list.push_back(ys)
        xs_list.push_back(xs)
        new_y.push_back(y[i])
        new_x.push_back(x[i])
        new_rad.push_back(rad[i])
    return new_y, new_x, new_rad, area, ys_list, xs_list, masks


