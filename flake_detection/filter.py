import math
from findmaxima2d import find_maxima, find_local_maxima
import cv2
from .find_circles import radius_estimate, area_estimate, merge_centers
from scipy.ndimage import generate_binary_structure
from gwdt import gwdt
import numpy as np


def circle_filter(img_path):
    img: np.array = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    m = np.quantile(img, 0.0001), np.quantile(img, 0.9999)
    img = img.clip(m[0], m[1])
    img = ((img - m[0]) / (m[1] - m[0]) * 255).astype(np.uint8)
    img = cv2.bilateralFilter(img, -1, 10, 10)

    # distance transform and find centers
    thr = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)[0]
    fgnd_img = img - thr
    fgnd_img[fgnd_img < 0] = 0
    structure = generate_binary_structure(img.ndim, 10)
    dt = gwdt(fgnd_img, structure)
    y, x, regs = find_maxima(dt, find_local_maxima(dt), 10)
    # rad = radius_estimate(img, y, x, lowest_cutoff=thr, cutoff_ratio=0.7)
    # y, x = merge_centers(img, y, x, 0.5, [i * 2 for i in rad])

    # detection
    rad = radius_estimate(img, y, x, lowest_cutoff=thr, cutoff_ratio=0.2, bg_thr=0.0001)
    area, flood = area_estimate(img, y, x, rad, lowest_cutoff=thr, cutoff_ratio=0.2)

    # # filter
    # tf = []
    # for i, yy, xx, r, a in zip(range(len(y)), y, x, rad, area):
    #     # area is too much bigger than radius infer
    #     if a > r ** 2 * math.pi * 2:
    #         tf.append(False)
    #         continue
    #     pix = np.argwhere(flood == i)
    #
    #     # area is not likely a circle
    #     # plot a radial distribution, it should be flat
    #     if a > 100:
    #         rr = r * 24
    #         z = [0] * rr
    #         max_dist = 0
    #         for ct in pix:
    #             dist = ((yy - ct[0]) ** 2 + (xx - ct[1]) ** 2) ** 0.5
    #             dist = min(round(dist), rr - 1)
    #             max_dist = max(dist, max_dist)
    #             z[dist] += 1
    #         for k in range(1, rr):
    #             z[k] /= 2 * math.pi * k
    #         s = np.quantile((abs(np.array(z[:max_dist + 1]) - 1)), 0.5)  # a bit lower radius range, for robustness
    #         if s > 0.5:
    #             tf.append(False)
    #             continue
    #     tf.append(True)

    return {'y': y, 'x': x, 'r': rad, 'area': area}, flood
