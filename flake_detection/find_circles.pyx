import cython
cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.pair cimport pair
from sklearn.neighbors import KDTree
import cv2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def radius_estimate(img_py, y_py, x_py, cutoff_ratio=0.5, bg_thr=.001, lowest_cutoff=0):
    """
    Estimate radius of d circles

    :param lowest_cutoff:
    :param bg_thr:
    :param img_py:
    :param y_py:
    :param x_py:
    :param cutoff_ratio:
    :return:
    """
    assert len(x_py) == len(y_py)
    cdef:
        int tot_cand = len(y_py), i, j, k, width = img_py.shape[1], height = img_py.shape[0], tot, bg, dy, dx
        float cr = cutoff_ratio, thr, r, bg_th = bg_thr, lc = lowest_cutoff
        np.ndarray[np.uint8_t, ndim=2, cast=True] img = img_py
        vector[int] y = y_py, x = x_py, rad = [0] * tot_cand
    for i in range(tot_cand):
        tot = bg = 0
        thr = img[y[i], x[i]] * cr
        if lc > thr:
            thr = lc
        while True:
            rad[i] += 1
            for dy in range(-rad[i], rad[i] + 1):
                for dx in range(-rad[i], rad[i] + 1):
                    r = (dy**2 + dx**2)**.5
                    tot += 1
                    if rad[i] - 1 < r <= rad[i]:
                        j = y[i] + dy
                        k = x[i] + dx
                        if not 0 <= j < height or not 0<= k < width:    # hit the border
                            break
                        if img[j, k] < thr:
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

def area_estimate(img_py, y_py, x_py, rad_py, cutoff_ratio=0.5, lowest_cutoff=0):
    """
    Count area by flooding, based on radius estimate results.
    :param lowest_cutoff:
    :param img_py:
    :param y_py:
    :param x_py:
    :param cutoff_ratio:
    :return:
    """
    assert len(x_py) == len(y_py) == len(rad_py)
    cdef:
        int tot_cand = len(y_py), i, j, k, width = img_py.shape[1], height = img_py.shape[0]
        float cr = cutoff_ratio, thr, lc = lowest_cutoff
        np.ndarray[np.uint8_t, ndim=2, cast=True] img = img_py
        np.ndarray[np.int32_t, ndim=2] disks = np.ones_like(img_py, dtype=int) * -1, visited = np.ones_like(img_py, dtype=int) * -1
        vector[int] y = y_py, x = x_py, area = [1] * tot_cand, rad = rad_py
        vector[pair[int, int]] dire = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        queue[pair[int, int]] que
        pair[int, int] t, tt

    # draw filled disks
    for i in np.argsort(rad_py):
        cv2.circle(disks, (x[i], y[i]), rad[i], i, -1)

    for i in range(tot_cand):
        t = y[i], x[i]
        que.push(t)
        thr = img[t.first, t.second] * cr
        if lc > thr:
            thr = lc
        visited[t.first, t.second] = i
        while not que.empty():
            t = que.front()
            for d in dire:
                tt = t.first + d.first, t.second + d.second
                if height > tt.first >= 0 and width > tt.second >= 0 and \
                        visited[tt.first, tt.second] == -1 and img[tt.first, tt.second] >= thr and \
                        (disks[tt.first, tt.second] == i or disks[tt.first, tt.second] == -1):
                    que.push(tt)
                    area[i] += 1
                    visited[tt.first, tt.second] = i
            que.pop()
    return area, visited

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def merge_centers(img_py, y_py, x_py, gap_py=0.9, merge_dist=50):
    """
    Merge centers if there's no major change in the profile between them.
    :param img_py:
    :param y_py:
    :param x_py:
    :param gap_py:
    :param merge_dist:
    :return:
    """
    assert len(x_py) == len(y_py)
    if not isinstance(merge_dist, list):
        merge_dist = [merge_dist] * len(y_py)
    cdef:
        int tot_cand = len(y_py), i, j, sy, sx, vmax, vmin, t
        float gap = gap_py
        np.ndarray[np.uint8_t, ndim=2, cast=True] img = img_py
        np.ndarray[np.int64_t, ndim=2] data = np.transpose([y_py, x_py])
        vector[vector[int]] conn = [tuple()] * tot_cand
        bint flag
        np.ndarray[np.int64_t, ndim=1] start_pos, end_pos, k, pos, tpos
        vector[bint] flags = [False] * tot_cand
        vector[int] ans_y, ans_x, current_node
        queue[int] que
    tree = KDTree(data)
    query = []
    for y, x, d in zip(y_py, x_py, merge_dist):
        query.append(tree.query_radius([[y, x]], d)[0])
    for i in range(len(query)):
        start_pos = data[i]
        for j in query[i]:
            if j == i:
                continue
            # check if already profiled
            if j < i:
                flag = False
                for t in conn[j]:
                    if t == i:
                        flag = True
                        break
                if flag:
                    conn[i].push_back(j)
                continue
            end_pos = data[j]
            k = end_pos - start_pos
            pos = start_pos.copy()
            tpos = pos - start_pos
            sy = 1 if k[0] > 0 else -1
            sx = 1 if k[1] > 0 else -1
            vmax = vmin = img[end_pos[0], end_pos[1]]
            # profiling
            while tpos[0] != k[0] and tpos[1] != k[1]:
                t = img[pos[0], pos[1]]
                if t > vmax:
                    vmax = t
                if t < vmin:
                    vmin = t
                if abs(tpos[1] * k[0]) > abs(tpos[0] * k[1]):
                    pos[0] += sy
                else:
                    pos[1] += sx
                tpos = pos - start_pos
            if vmax - vmin < gap * vmax:
                conn[i].push_back(j)
    # connect
    for i in range(tot_cand):
        if flags[i]:
            continue
        current_node = [i]
        sy = 0
        sx = 0
        que.push(i)
        flags[i] = True
        while not que.empty():
            for j in conn[que.front()]:
                if not flags[j]:
                    current_node.push_back(j)
                    que.push(j)
                    flags[j] = True
            que.pop()
        for j in current_node:
            sy += data[j, 0]
            sx += data[j, 1]
        ans_y.push_back(sy / current_node.size())
        ans_x.push_back(sx / current_node.size())

    return ans_y, ans_x

