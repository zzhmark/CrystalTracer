from sklearn.neighbors import KDTree
import numpy as np
from collections import OrderedDict
from multiprocessing import Pool
from functools import cmp_to_key


def cmp(a, b):
    a_dist, a_area, a_t = a[1:]
    b_dist, b_area, b_t = b[1:]
    return a_dist ** 2 - b_dist ** 2 + a_area - b_area + (a_t - b_t) ** 3


def trace_single(i, trees, frames, page_look_back, dist_thr):
    y = frames[0]['y'][i]
    x = frames[0]['x'][i]
    area = frames[0]['area'][i]
    chain = OrderedDict()  # storing results
    chain[0] = i
    for j in range(1, len(trees)):
        candid = []
        for k in range(page_look_back):
            if j + k >= len(trees):
                break
            ind, dist = trees[j + k].query_radius([[y, x]], dist_thr, return_distance=True)
            ind = ind[0]
            dist = dist[0]
            if len(ind) == 0:
                continue
            area_diff = [abs(frames[k + j]['area'][v] - area) for v in ind]
            candid.extend([*zip(ind, dist, area_diff, [k] * len(ind))])
        if len(candid) == 0:
            break
        candid.sort(key=cmp_to_key(cmp))
        fr = j + candid[0][3]
        ind = candid[0][0]
        chain[fr] = ind
        # update current circle
        y = frames[fr]['y'][ind]
        x = frames[fr]['x'][ind]
        area = frames[fr]['area'][ind]
    return chain


def backtrace(frames, page_look_back=3, dist_thr=30, nproc=8):
    frames.reverse()
    # kdtree for search
    trees = [KDTree([*zip(d['y'], d['x'])]) for d in frames]
    # start from the last frame
    arglist = []
    for i in range(len(frames[0])):
        arglist.append([i, trees, frames, page_look_back, dist_thr])
    with Pool(nproc) as p:
        res = p.starmap(trace_single, arglist)
    frames.reverse()
    return dict(zip(range(len(frames[0])), res))