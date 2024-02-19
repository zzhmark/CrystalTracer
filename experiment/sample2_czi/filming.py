import cv2
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from crystal_tracer.visual.draw import draw_circles
from tqdm import tqdm
from czifile import imread

out_dir = Path('../../data/case2')
flood_dir = out_dir / "flood"
csv_dir = out_dir / "detection"
vid_dir = out_dir / "video"



def main(args):
    stacks, frames, trace = args
    if len(trace) < 100:
        return
    frame_inds = [len(frames) - i - 1 for i, j in trace]
    chip_inds = [j for i, j in trace]
    frame_inds.reverse()
    chip_inds.reverse()
    save_path = vid_dir / f'{chip_inds[-1]}_range({frame_inds[0]},{frame_inds[-1]}).avi'
    x = [frames[i]['x'][j] for i, j in zip(frame_inds, chip_inds)]
    y = [frames[i]['y'][j] for i, j in zip(frame_inds, chip_inds)]
    rad = [frames[i]['radius'][j] for i, j in zip(frame_inds, chip_inds)]
    win_rad = 30
    img_seq = []
    ctx = int(x[0])
    cty = int(y[0])
    for i, (xx, yy, rr, fr, key) in enumerate(zip(x, y, rad, frame_inds, chip_inds)):
        # with open(flood_dir / f"{fr}.pickle", 'rb') as f:
        #     flood = pickle.load(f)
        img = stacks[0, fr, 0, ..., 0]
        ys = max(int(cty) - win_rad, 0)
        xs = max(int(ctx) - win_rad, 0)
        ye = min(img.shape[0], int(cty) + win_rad)
        xe = min(img.shape[1], int(ctx) + win_rad)
        img = img[ys:ye, xs:xe]
        # a = np.zeros_like(img, dtype=np.uint8)
        # y_, x_ = np.nonzero(flood[key])
        # rr = flood[key].shape[0] // 2
        # y_ += -rr + int(yy) - ys
        # x_ += -rr + int(xx) - xs
        # y_ = np.clip(y_, 0, img.shape[0] - 1)
        # x_ = np.clip(x_, 0, img.shape[1] - 1)
        # a[(y_, x_)] = 1
        img = (img / img.max() * 255).astype(np.uint8)
        # img = draw_contour(img, a)
        img = draw_circles(img, [yy - ys], [xx - xs], [rr])
        if i == len(frame_inds) - 1:
            img_seq.append(img)
        else:
            img_seq.extend([img] * (frame_inds[i + 1] - frame_inds[i]))
        cty = int(yy * .5 + cty * .5)
        ctx = int(xx * .5 + ctx * .5)

    vid = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'DIVX'), 10, (win_rad*2, win_rad*2))
    for img in img_seq:
        vid.write(img)


if __name__ == '__main__':
    in_folder = Path(r'D:\下载\Compressed')
    vid_dir.mkdir(exist_ok=True)
    frames = [pd.read_csv(csv_dir / f"{i}.csv") for i in range(len([*csv_dir.glob('*.csv')]))]
    stacks = imread(r"D:\下载\Mitosis_Transient_scene1.czi")
    with open('../../data/case2/traces.pickle', 'rb') as f:
        traces = pickle.load(f)
    arglist = [(stacks, frames, i) for i in traces]
    with Pool(10) as p:
        for res in tqdm(p.imap(main, arglist), total=len(arglist)): pass
