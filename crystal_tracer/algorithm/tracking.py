import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


def independent_match(tables, dist_thr=20, nn=5, time_gap_thr=10, time_sampling_range=(10, 100),
                      max_area_overflow=.2, max_area_diff=25., callback=None):
    """
    Connect the crystals in each frame in and independent manner, starting from the last frame. It forms a track for
    each crystal in the last frame until it disappears in reverse time order. The output will be reversed back.

    :param tables: a list of dataframes of detected crystals
    :param dist_thr: the radius range for match the crystal
    :param nn: number of nearest neighboring crystals considered for matching
    :param time_gap_thr: max time gap allowed in the track
    :param time_sampling_range: area sampling window
    :param max_area_overflow: the max ratio of area difference
    :param max_area_diff: the max exact area difference in pixel
    :param callback: a function to call in each iteration
    :return: identified tracks, a list of lists of tuples, (frame, index)
    """
    # invert the frames
    tables = [*reversed(tables)]

    # init from the last frame
    # chains: the tracks. list of (frame, crystal_id)
    # prev_pos: previous crystal position
    # pred_area: predicted area based on previous discovery
    tracks = []
    prev_pos = []
    pred_area = []
    for ind, row in tables[0].iterrows():
        tracks.append([(0, ind)])
        prev_pos.append(row[['y', 'x']].tolist())
        pred_area.append(row['area'])  # init as the start crystal size

    # fit time-area model
    # no enough points, return just average
    def area_fit(chain):
        x = [c[0] for c in chain]  # time frame
        y = [tables[c[0]].at[c[1], 'area'] for c in chain]
        if len(x) < time_sampling_range[0] or x[-1] - x[0] < time_sampling_range[
            1]:  # when there is not enough or time frame is too short
            return np.mean(y)
        model = LinearRegression()
        model.fit(np.array(x).reshape((-1, 1)), y, [*range(1, len(y) + 1)])  # increasing weight for new entries
        return model

    def area_predict(mod, time):
        if type(mod) is LinearRegression:
            return mod.predict([[time]])[0]
        else:
            return mod

    callback()
    # start tracking from the one but last frame
    for i_frame in tqdm(range(1, len(tables))):
        # try mapping all current crystals onto the tracks

        track_tree = KDTree(prev_pos)  # kdtree based on previous tracks' end coordinates
        df: pd.DataFrame = tables[i_frame]  # dataframe of current detections

        # the assignment of detections to tracks has a priority
        # the detections with less competition is considered first
        # measured by the distance to their nearest neighbour
        cur_pos = df[['y', 'x']].to_numpy()
        cur_tree = KDTree(cur_pos)  # kdtree based on detections of current frame
        cur_dists = cur_tree.query(cur_pos, 2, dualtree=True)[0]  # used to sort the detections
        # the bigger the distance to neighbour, the more it's near the front of the queue
        cur_order = np.flip(np.argsort(cur_dists[:, 1].reshape(-1)))
        sorted_cur_pos = cur_pos[cur_order]

        # based on sorted current pos, find the candidate track ends within a radius
        candidate_tracks = track_tree.query_radius(sorted_cur_pos, dist_thr)
        # the features and indices of features of the current pos
        # this is for reference to calculate the priority
        ref_features, ref_features_ind = cur_tree.query(sorted_cur_pos, nn, dualtree=True)

        # enumerate through the detection queries in the sorted order
        settled = [False] * len(cur_pos)  # flag for whether each detection is assigned
        new_pos = [None] * len(prev_pos)  # temp for new connected detections

        for e, (i_track, ref_vec, ref_ind, a) in enumerate(zip(candidate_tracks,
                                                               ref_features,
                                                               ref_features_ind,
                                                               df.loc[cur_order, 'area'])):
            if len(i_track) == 0:
                continue

            def area_test(ind):
                area = area_predict(pred_area[ind], i_frame)
                return abs(a - area) < max(max_area_overflow * area, max_area_diff)

            # filter the tracks by the area prediction allowance
            i_track = i_track[[area_test(i) for i in i_track]]
            if len(i_track) == 0:  # this detection can't be matched to any track
                continue
            track_vec = [np.linalg.norm(prev_pos[i] - cur_pos[ref_ind], axis=1) for i in i_track]
            for i in i_track[np.argsort(np.linalg.norm(track_vec - ref_vec, axis=1))]:
                if new_pos[i] is None and i_frame - tracks[i][-1][0] < time_gap_thr:
                    new_pos[i] = cur_order[e]
                    break
            else:
                continue
            settled[cur_order[e]] = True

        # update pos, area
        for e, p in enumerate(new_pos):
            if p is None:
                continue
            tracks[e].append((i_frame, p))
            prev_pos[e] = cur_pos[p]
            pred_area[e] = area_fit(tracks[e])
        if callback is not None:
            callback()

    return tracks
