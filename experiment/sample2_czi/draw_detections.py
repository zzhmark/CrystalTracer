import pandas as pd
import pickle
from czifile import imread
from pathlib import Path
from crystal_tracer.visual.draw import draw_contour, draw_circles
import numpy as np
import cv2



out_dir = Path('../../data/case2')
flood_dir = out_dir / "flood"
csv_dir = out_dir / "detection"
draw_dir = out_dir / "draw"


def main(args):
    stack, slice = args
    gfp = stack[0, slice, 0, ..., 0]
    with open(flood_dir / f"{slice}.pickle", 'rb') as f:
        flood = pickle.load(f)
    frame = pd.read_csv(csv_dir / f"{slice}.csv")
    win_rad = 50
    for ind, row in frame.iterrows():
        y = row['y']
        x = row['x']
        ys = max(int(y) - win_rad, 0)
        xs = max(int(x) - win_rad, 0)
        ye = min(gfp.shape[0], int(y) + win_rad)
        xe = min(gfp.shape[1], int(x) + win_rad)
        img = gfp[ys:ye, xs:xe]
        a = np.zeros_like(img)
        y_, x_ = np.nonzero(flood[ind])
        rr = flood[ind].shape[0] // 2
        y_ += -rr + int(y) - ys
        x_ += -rr + int(x) - xs
        y_ = np.clip(y_, 0, img.shape[0] - 1)
        x_ = np.clip(x_, 0, img.shape[1] - 1)
        a[(y_, x_)] = 1
        img = (img / img.max() * 255).astype(np.uint8)
        img = draw_contour(img, a)
        cv2.imwrite(str(draw_dir /  f'{slice}-{ind}.tif'), img)


def main_circle(args):
    stack, slice = args
    gfp = stack[0, slice, 0, ..., 0]
    frame = pd.read_csv(csv_dir / f"{slice}_circle.csv")
    gfp = (gfp / gfp.max() * 255).astype('uint8')
    img = draw_circles(gfp, frame['y'], frame['x'], frame['radius'])
    cv2.imwrite(str(draw_dir /  f'{slice}.tif'), img)


if __name__ == '__main__':
    draw_dir.mkdir(exist_ok=True)
    stacks = imread(r"D:\下载\Mitosis_Transient_scene1.czi")
    main_circle((stacks, 100))
    # arglist = [(stacks, i) for i in range(stacks.shape[1])]
    # with Pool(12) as p:
    #     for res in tqdm(p.imap(main, arglist), total=len(arglist)): pass