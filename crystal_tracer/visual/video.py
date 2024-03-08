from crystal_tracer.visual.draw import draw_contour
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from crystal_tracer.img_utils import load_czi_slice, get_czi_shape
import platform


def make_video(track: list[tuple[int, int]], save_path: Path | None, czi_path: Path, table_paths: list[Path],
               mask_paths: list[Path], win_rad=30, frame_rate=25.):
    """
    generate a video for one traced track.

    :param track: a list of tuples containing (frame_index, crystal_id)
    :param save_path: the save path of the AVI video
    :param czi_path: the CZI image path
    :param table_paths: the paths to the crystal statistics in each time frame
    :param mask_paths: the paths to the segmentation of crystals
    :param win_rad: window radius of the video
    :param frame_rate: frame rate of the video
    """
    assert len(table_paths) == len(mask_paths)
    tag = 'DIVX' if platform.system() == 'Windows' else 'XVID'
    out_size = (win_rad * 2, win_rad * 2)
    writer = None if save_path is None else cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*tag), frame_rate, out_size)
    last = None
    tot, c, height, width = get_czi_shape(czi_path)
    out = []
    for i, j in track:
        ys_old, xs_old, intensity = pd.read_csv(table_paths[i]).loc[j, ['y_start', 'x_start', 'intensity']].values.astype(int).ravel()
        mask = np.load(mask_paths[i])[f'arr_{j}']
        size_y, size_x = mask.shape
        cty, ctx = ys_old + size_y // 2, xs_old + size_x // 2
        ys, ye = max(cty - win_rad, 0), min(height, cty + win_rad)
        xs, xe = max(ctx - win_rad, 0), min(width, ctx + win_rad)
        new_mask = np.zeros([ye - ys, xe - xs], dtype=np.uint8)
        y_, x_ = np.nonzero(mask)
        y_ += ys_old - ys
        x_ += xs_old - xs
        y_ = np.clip(y_, 0, height - 1)
        x_ = np.clip(x_, 0, width - 1)
        new_mask[(y_, x_)] = 1
        img = load_czi_slice(czi_path, 0, i)[ys: ye, xs: xe]
        img = img.clip(None, intensity * 2) / (intensity * 2)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img = draw_contour(img, new_mask)
        pad_width = (*out_size, 3) - np.array(img.shape)
        pad_width = np.stack((pad_width // 2, pad_width - pad_width // 2), axis=1)
        img = np.pad(img, pad_width)
        for k in range(1 if last is None else i - last):
            out.append(img)
            if writer is not None:
                writer.write(img)
        last = i
    return out
