import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from math import sqrt, atan2, pi
from scipy.optimize import linear_sum_assignment


class Predictor:
    def __init__(self, min_sampling_count=2, min_sampling_elapse=2):
        self.min_sampling_count = min_sampling_count
        self.min_sampling_elapse = min_sampling_elapse
        self._mod = None

    def fit(self, x, y):
        # fit time-area model
        # no enough points, return just average
        if len(x) < self.min_sampling_count or x[-1] - x[0] < self.min_sampling_elapse:
            # when there is not enough or time frame is too short
            self._mod = np.mean(y, axis=0)
        else:
            self._mod = LinearRegression()
            # increasing weight for new entries
            self._mod.fit(np.array(x).reshape(-1, 1), y, [x[0] - i + 1 for i in x])
        return self

    def predict(self, x):
        if type(self._mod) is LinearRegression:
            return self._mod.predict([[x]])[0]
        else:
            return self._mod


def angle_dist(center, other):
    vec = other - center
    dist = np.linalg.norm(vec, axis=-1)
    angle = np.array([atan2(v[0], v[1]) for v in vec])
    order = np.argsort(angle)
    return dist[order], angle[order]


def neighborhood_match(center, other, centers, others):
    dist, rad = angle_dist(center, other)
    scores = []
    for c, o in zip(centers, others):
        d, r = angle_dist(c, o)
        diff_r = abs(np.subtract.outer(rad, r))
        diff_r = np.clip(diff_r, None, 2 * pi - diff_r)
        diff_d = abs(np.subtract.outer(dist, d))
        diff = diff_d * diff_r
        choice = linear_sum_assignment(diff)
        scores.append(diff[choice].mean())
    return scores


def independent_match(tables: list[pd.DataFrame], area_normalizer=500., nn=5, time_gap_thr=5, min_sampling_count=5,
                      min_sampling_elapse=10,
                      max_area_overflow=.25, max_intensity_overflow=.25, callback=None):
    """
    Connect the crystals in each frame in and independent manner, starting from the last frame. It forms a track for
    each crystal in the last frame until it disappears in reverse time order. The output will be reversed back.

    :param tables: a list of dataframes of detected crystals
    :param area_normalizer: the distance threshold controller
    :param nn: number of nearest neighboring crystals considered for matching
    :param time_gap_thr: max time gap allowed in the track
    :param min_sampling_count: min No. of sampling points for area fitting
    :param min_sampling_elapse: min time elapse of sampling points for area fitting
    :param max_area_overflow: the max ratio of area difference
    :param max_intensity_overflow: the max ratio of intensity difference
    :param callback: a function to call in each iteration
    :return: identified tracks, a list of lists of tuples, (frame, index)
    """

    # init from the last frame
    # chains: the tracks. list of (frame, crystal_id)
    # pred_area: predicted area based on previous discovery
    nn += 1
    tracks = []
    pred_area = []
    pred_gray = []
    trees = [KDTree(t[['y', 'x']]) for t in tables]
    for ind, row in tables[-1].iterrows():
        tracks.append([(len(tables) - 1, ind)])
        # init as the start crystal size
        pred_area.append(Predictor(min_sampling_count, min_sampling_elapse).fit([0], [row['area']]))
        pred_gray.append(Predictor().fit([0], [row['intensity']]))

    callback()
    # start tracking from the one but last frame
    for i_frame in tqdm(range(len(tables) - 2 , -1, -1)):
        cur_pos = tables[i_frame][['y', 'x']].to_numpy()
        n = min(nn, len(cur_pos))
        cur_ind = trees[i_frame].query(cur_pos, n, dualtree=True, return_distance=False)

        # check for each crystal which track can be appended to
        for track, mod_area, mod_gray in zip(tracks, pred_area, pred_gray):

            # these tracks are terminated for big time gap
            if track[-1][0] - i_frame > time_gap_thr + 1:
                continue

            i_frame_last, i_crystal_last = track[-1]
            center = tables[i_frame_last].loc[i_crystal_last, ['y', 'x']].to_numpy()
            ref_area = mod_area.predict(i_frame)
            dist_thr = area_normalizer / sqrt(ref_area)
            i_crystals = trees[i_frame].query_radius([center], dist_thr)[0]

            # estimate the area for this frame
            scale = sqrt(area_normalizer / ref_area)
            i_crystals = i_crystals[np.abs(ref_area - tables[i_frame].loc[i_crystals, 'area'].to_numpy()) <
                                    max_area_overflow * scale * ref_area]
            ref_gray = mod_gray.predict(i_frame)
            i_crystals = i_crystals[np.abs(tables[i_frame].loc[i_crystals, 'intensity'].to_numpy() - ref_gray) <
                                    max_intensity_overflow * scale * ref_gray]

            if len(i_crystals) == 0:
                continue

            ind = trees[i_frame_last].query([center], n, return_distance=False)[0][1:]
            ans = neighborhood_match(center, tables[i_frame_last].loc[ind, ['y', 'x']].to_numpy(),
                                     tables[i_frame].loc[i_crystals, ['y', 'x']].to_numpy(),
                                     [tables[i_frame].loc[cur_ind[i][1:], ['y', 'x']].to_numpy() for i in i_crystals])

            # update track if time gap is met
            track.append((i_frame, i_crystals[np.argmin(ans)]))

            # update area prediction
            mod_area.fit([c[0] for c in track], [tables[c[0]].at[c[1], 'area'] for c in track])
            mod_gray.fit([c[0] for c in track], [tables[c[0]].at[c[1], 'intensity'] for c in track])

        if callback is not None:
            callback()

    for t in tracks:
        t.reverse()

    return tracks


def ratio_diff(a, b):
    t = a / b
    return (t + 1 / t) / 2


def linear_programming2(tables, area_normalizer=500., nn=10, time_gap_thr=10, min_sampling_count=5, min_sampling_elapse=10,
                       max_area_overflow=.25, max_intensity_overflow=.25, w_dist=1., w_area=1., w_intensity=1., w_local=1.,
                       callback=None):
    """
    Connect the crystals in each frame in and independent manner, starting from the last frame. It forms a track for
    each crystal in the last frame until it disappears in reverse time order. The output will be reversed back.

    :param tables: a list of dataframes of detected crystals
    :param area_normalizer: the distance threshold
    :param nn: number of nearest neighboring crystals considered for matching
    :param time_gap_thr: max time gap allowed in the track
    :param min_sampling_count: min No. of sampling points for area fitting
    :param min_sampling_elapse: min time elapse of sampling points for area fitting
    :param max_area_overflow: the max ratio of area difference
    :param max_intensity_overflow: the max ratio of intensity difference
    :return: identified tracks, a list of lists of tuples, (frame, index)
    """

    def cost(v1, v2):
        p1 = v1[:2]
        p2 = v2[:2]
        dist = np.linalg.norm(p1 - p2)
        area_diff = ratio_diff(v1[2] - v2[2])
        gray_diff = ratio_diff(v1[3] - v2[3])
        v1 = v1[4:].reshape(-1, 2)
        v2 = v2[4:].reshape(-1, 2)
        local_score = neighborhood_match(p1, v1, [p2], [v2])[0]
        return (w_dist * dist + w_local * local_score) * area_diff * gray_diff

    cost_func = np.vectorize(cost, signature='(n),(n)->()')

    nn += 1

    # init from the last frame
    # chains: the tracks. list of (frame, crystal_id)
    # pred_area: predicted area based on previous discovery
    tracks = []
    pred_area = []
    pred_gray = []
    trees = [KDTree(t[['y', 'x']]) for t in tables]
    for ind, row in tables[-1].iterrows():
        tracks.append([(len(tables) - 1, ind)])
        # init as the start crystal size
        pred_area.append(Predictor(min_sampling_count, min_sampling_elapse).fit([0], [row['area']]))
        pred_gray.append(Predictor().fit([0], [row['intensity']]))

    # start tracking from the one but last frame
    for i_frame in tqdm(range(len(tables) - 2 , -1, -1)):
        if callback is not None:
            callback()
        cur_features = tables[i_frame][['y', 'x', 'area', 'intensity']].to_numpy()
        n = min(nn, len(cur_features))
        cur_ind = trees[i_frame].query(cur_features[:, :2], n, dualtree=True, return_distance=False)
        cur_pos = np.array([cur_features[i[1:], :2] for i in cur_ind]).reshape(cur_ind.shape[0], -1)
        cur_features = np.concatenate([cur_features, cur_pos], axis=1)

        active = []
        pre_features = []
        for t, mod_area, mod_gray in zip(tracks, pred_area, pred_gray):
            i_frame_last, i_crystal_last = t[-1]
            if i_frame_last - i_frame > time_gap_thr + 1:
                continue
            active.append(t)
            feature = tables[i_frame_last].loc[i_crystal_last, ['y', 'x']].to_list()
            prev_ind = trees[i_frame_last].query([feature], n, return_distance=False)[0][1:]
            prev_pos = tables[i_frame_last].loc[prev_ind, ['y', 'x']].to_numpy().reshape(-1)
            feature.append(mod_area.predict(i_frame))
            feature.append(mod_gray.predict(i_frame))
            feature.extend(list(prev_pos))
            pre_features.append(feature)
        if len(active) == 0:
            continue
        pre_features = np.array(pre_features)

        # linear programming
        mat = cost_func(pre_features.reshape(-1, 1, pre_features.shape[1]),
                        cur_features.reshape(1, -1, cur_features.shape[1]))
        choice = linear_sum_assignment(mat)

        choice = dict(zip(*choice))

        for i, (t, mod_area, mod_gray, f1) in enumerate(zip(active, pred_area, pred_gray, pre_features)):
            # update tracks and models
            coord = f1[:2]
            ref_area = f1[2]
            ref_gray = f1[3]
            dist_thr = area_normalizer / sqrt(ref_area)
            if i not in choice:
                continue
            c = choice[i]
            f2 = cur_features[c]
            cur_coord = f2[:2]
            cur_area = f2[2]
            cur_gray = f2[3]
            scale = sqrt(area_normalizer / ref_area)
            if np.linalg.norm(coord - cur_coord) > dist_thr or \
                    abs(ref_area - cur_area) > max_area_overflow * scale * ref_area or \
                    abs(cur_gray - ref_gray) > max_intensity_overflow * scale * ref_gray:
                continue
            t.append((i_frame, c))
            mod_area.fit([c[0] for c in t], [tables[c[0]].at[c[1], 'area'] for c in t])
            mod_gray.fit([c[0] for c in t], [tables[c[0]].at[c[1], 'intensity'] for c in t])

    for t in tracks:
        t.reverse()

    callback()
    return tracks


def linear_programming(tables, area_normalizer=500., nn=10, time_gap_thr=10, min_sampling_count=5, min_sampling_elapse=10,
                       max_area_overflow=.25, max_intensity_overflow=.25, w_dist=1., w_area=1., w_intensity=1., w_local=1.,
                       callback=None):
    """
    Connect the crystals in each frame in and independent manner, starting from the last frame. It forms a track for
    each crystal in the last frame until it disappears in reverse time order. The output will be reversed back.

    :param tables: a list of dataframes of detected crystals
    :param area_normalizer: the distance threshold
    :param nn: number of nearest neighboring crystals considered for matching
    :param time_gap_thr: max time gap allowed in the track
    :param min_sampling_count: min No. of sampling points for area fitting
    :param min_sampling_elapse: min time elapse of sampling points for area fitting
    :param max_area_overflow: the max ratio of area difference
    :param max_intensity_overflow: the max ratio of intensity difference
    :return: identified tracks, a list of lists of tuples, (frame, index)
    """

    def cost(v1, v2):
        p1 = v1[:2]
        p2 = v2[:2]
        dist = np.linalg.norm(p1 - p2)
        # area_diff = ratio_diff(v1[2], v2[2])
        # gray_diff = ratio_diff(v1[3], v2[3])
        # v1 = v1[4:].reshape(-1, 2)
        # v2 = v2[4:].reshape(-1, 2)
        # local_score = neighborhood_match(p1, v1, [p2], [v2])[0]
        return dist ** 10

    cost_func = np.vectorize(cost, signature='(n),(n)->()')

    nn += 1

    # init from the last frame
    # chains: the tracks. list of (frame, crystal_id)
    # pred_area: predicted area based on previous discovery
    tracks = [[(len(tables) - 1, i)] for i in range(len(tables[-1]))]
    trees = [KDTree(t[['y', 'x']]) for t in tables]

    # start tracking from the 2nd last frame
    for i_frame in tqdm(range(len(tables) - 2 , -1, -1)):
        if callback is not None:
            callback()
        pre_features = tables[i_frame + 1][['y', 'x', 'area', 'intensity']].to_numpy()
        cur_features = tables[i_frame][['y', 'x', 'area', 'intensity']].to_numpy()
        # n = min(nn, len(cur_features), len(pre_features))
        # cur_ind = trees[i_frame].query(cur_features[:, :2], n, dualtree=True, return_distance=False)
        # pre_ind = trees[i_frame + 1].query(pre_features[:, :2], n, dualtree=True, return_distance=False)
        # cur_pos = np.array([cur_features[i[1:], :2] for i in cur_ind]).reshape(cur_ind.shape[0], -1)
        # pre_pos = np.array([pre_features[i[1:], :2] for i in pre_ind]).reshape(pre_ind.shape[0], -1)
        # cur_features = np.concatenate([cur_features, cur_pos], axis=1)
        # pre_features = np.concatenate([pre_features, pre_pos], axis=1)

        # linear programming
        mat = cost_func(pre_features.reshape(-1, 1, pre_features.shape[1]),
                        cur_features.reshape(1, -1, cur_features.shape[1]))
        choice = linear_sum_assignment(mat)

        choice = dict(zip(choice[0], choice[1]))

        for t in tracks:
            # update tracks and models
            i_frame_last, i_crystal_last = t[-1]
            if i_frame_last - i_frame > 1:
                continue
            if i_crystal_last not in choice:
                continue
            c = choice[i_crystal_last]
            # f1 = pre_features[i_crystal_last]
            # f2 = cur_features[c]
            #
            # coord = f1[:2]
            # ref_area = f1[2]
            # ref_gray = f1[3]
            # dist_thr = area_normalizer / sqrt(ref_area)
            # cur_coord = f2[:2]
            # cur_area = f2[2]
            # cur_gray = f2[3]
            # scale = sqrt(area_normalizer / ref_area)
            # if np.linalg.norm(coord - cur_coord) > dist_thr or \
            #         abs(ref_area - cur_area) > max_area_overflow * scale * ref_area or \
            #         abs(cur_gray - ref_gray) > max_intensity_overflow * scale * ref_gray:
            #     continue
            t.append((i_frame, c))

    for t in tracks:
        t.reverse()

    callback()
    return tracks