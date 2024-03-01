from crystal_tracer.visual.draw import draw_contour
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from crystal_tracer.img_utils import load_czi_slice
from skimage.util import img_as_ubyte


def make_video(track: list[tuple[int, int]], save_path: Path, czi_path: Path, table_paths: list[Path],
               mask_paths: list[Path], win_rad=30, frame_rate=10.):
    """
    generate a video for one traced track.

    :param track: a list of tuples containing (frame_index, crystal_id)
    :param save_path: the save path of the video
    :param czi_path: the CZI image path
    :param table_paths: the paths to the crystal statistics ineach time frame
    :param mask_paths: the paths to the segmentation of crystals
    :param win_rad: window radius of the video
    :param frame_rate: frame rate of the video
    """
    assert len(table_paths) == len(mask_paths)
    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (win_rad * 2, win_rad * 2))
    tot = len(table_paths)
    last = None
    for i, j in reversed(track):
        i = tot - i - 1
        img = load_czi_slice(czi_path, 0, i)
        height, width = img.shape
        y, x = pd.read_csv(table_paths[i]).loc[j, ['x', 'y']].values.astype(int).ravel()
        mask = np.load(mask_paths[i])[f'arr_{j}']
        ys, ye = max(y - win_rad, 0), min(height, y + win_rad)
        xs, xe = max(x - win_rad, 0), min(width, x + win_rad)
        img = img_as_ubyte(img[ys: ye, xs: xe])
        new_mask = np.zeros_like(img)
        y_, x_ = np.nonzero(mask)
        rr = mask.shape[0] // 2
        y_ += -rr + y - ys
        x_ += -rr + x - xs
        y_ = np.clip(y_, 0, height - 1)
        x_ = np.clip(x_, 0, width - 1)
        new_mask[(y_, x_)] = 1
        img = draw_contour(img, new_mask)
        for k in range(1 if last is None else i - last):
            writer.write(img)
        last = i
