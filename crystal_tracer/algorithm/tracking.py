import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


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


def independent_match(tables, dist_thr=30, nn=5, time_gap_thr=10, min_sampling_count=10, min_sampling_elapse=100,
                      max_area_overflow=.2, max_area_diff=25., position_sampling_count=3, callback=None):
    """
    Connect the crystals in each frame in and independent manner, starting from the last frame. It forms a track for
    each crystal in the last frame until it disappears in reverse time order. The output will be reversed back.

    :param tables: a list of dataframes of detected crystals
    :param dist_thr: the radius range for match the crystal
    :param nn: number of nearest neighboring crystals considered for matching
    :param time_gap_thr: max time gap allowed in the track
    :param min_sampling_count: min No. of sampling points for area fitting
    :param min_sampling_elapse: min time elapse of sampling points for area fitting
    :param max_area_overflow: the max ratio of area difference
    :param max_area_diff: the max exact area difference in pixel
    :param position_sampling_count: the number of points for fitting crystal position
    :param callback: a function to call in each iteration
    :return: identified tracks, a list of lists of tuples, (frame, index)
    """

    # init from the last frame
    # chains: the tracks. list of (frame, crystal_id)
    # prev_pos: previous crystal position
    # pred_area: predicted area based on previous discovery
    tracks = []
    pred_area = []
    pred_pos = []
    for ind, row in tables[-1].iterrows():
        tracks.append([(len(tables) - 1, ind)])
        # init as the start crystal size
        pred_area.append(Predictor(min_sampling_count, min_sampling_elapse).fit([0], [row['area']]))
        pred_pos.append(Predictor().fit([0], [[row['y'], row['x']]]))

    callback()
    # start tracking from the one but last frame
    for i_frame in tqdm(range(len(tables) - 1 , -1, -1)):
        # the position of the expected position of each track in this frame predicted by a linear model
        expect_pos = np.array([p.predict(i_frame) for p in pred_pos])
        expect_tree = KDTree(expect_pos)
        # the assignment of detections to tracks has a priority
        # the detections with less competition is considered first
        # measured by the distance to their nearest neighbour
        cur_pos = tables[i_frame][['y', 'x']].to_numpy()
        cur_tree = KDTree(cur_pos)  # kdtree based on detections of current frame

        # based on previous track ends, find the candidate crystals within a radius
        candidate_crystals = cur_tree.query_radius(expect_pos, dist_thr)

        # the features and indices of features of the current pos
        # this is for reference to calculate the priority
        ref_features, ref_features_ind = expect_tree.query(expect_pos, nn, dualtree=True)

        # check for each crystal which track can be appended to
        for track, mod_area, mod_pos, i_crystals, ref_vec, ref_ind \
                in zip(tracks, pred_area, pred_pos, candidate_crystals, ref_features, ref_features_ind):
            # these tracks are terminated for big time gap
            if track[-1][0] - i_frame > time_gap_thr:
                continue

            # estimate the area for this frame
            ref_area = mod_area.predict(i_frame)
            i_crystals = i_crystals[np.abs(tables[i_frame].loc[i_crystals, 'area'].to_numpy() - ref_area) <
                                    min(max_area_overflow * ref_area, max_area_diff)]

            if len(i_crystals) == 0:
                continue

            cand_vec = [np.linalg.norm(cur_pos[i] - expect_pos[ref_ind], axis=1) for i in i_crystals]
            # most similar in neighborhood
            best_match = i_crystals[np.argmin(np.linalg.norm(cand_vec - ref_vec, axis=1))]

            # update track if time gap is met
            track.append((i_frame, best_match))

            # update area prediction
            mod_area.fit([c[0] for c in track], [tables[c[0]].at[c[1], 'area'] for c in track])
            mod_pos.fit([c[0] for c in track], [tables[c[0]].loc[c[1], ['y', 'x']].to_numpy()
                                                for c in track[-position_sampling_count:]])

        if callback is not None:
            callback()

    for t in tracks:
        t.reverse()

    return tracks
