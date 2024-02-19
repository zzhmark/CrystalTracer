import cv2
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
from multiprocessing import Pool
from itertools import repeat
from crystal_tracer.visual.draw import draw_contour
from tqdm import tqdm
from io import BytesIO
from PIL import Image


out_folder = Path('../../data/case1/vid')
flood_folder = Path('../../data/case1/flood')


def main(args):
    trace, zd, frames = args
    frame_inds = [len(frames) - i - 1 for i, j in trace]
    chip_inds = [j for i, j in trace]
    frame_inds.reverse()
    chip_inds.reverse()
    save_path = out_folder / f'{chip_inds[-1]}_range({frame_inds[0]},{frame_inds[-1]}).avi'
    x = [frames[i][1]['x'][j] for i, j in zip(frame_inds, chip_inds)]
    y = [frames[i][1]['y'][j] for i, j in zip(frame_inds, chip_inds)]
    fname = [zd[frames[i][0].replace('GFP', 'BF')] for i in frame_inds]
    flood_name = [Path(frames[i][0]).with_suffix('.pickle') for i in frame_inds]
    win_rad = 50
    img_seq = []
    ctx = int(x[0])
    cty = int(y[0])
    for i, (xx, yy, (zf, p), floods, key) in enumerate(zip(x, y, fname, flood_name, chip_inds)):
        p = str(p).replace('_GFP', '_BF')
        with open(zf, 'rb') as f:
            with zipfile.ZipFile(f) as z:
                with Image.open(BytesIO(zipfile.Path(z, p).read_bytes())) as im:
                    img = np.array(im)
        with open(flood_folder / floods, 'rb') as f:
            flood = pickle.load(f)
        ys = max(int(cty) - win_rad, 0)
        xs = max(int(ctx) - win_rad, 0)
        ye = min(img.shape[0], int(cty) + win_rad)
        xe = min(img.shape[1], int(ctx) + win_rad)
        img = img[ys:ye, xs:xe]
        a = np.zeros_like(img)
        y_, x_ = np.nonzero(flood[key])
        rr = flood[key].shape[0] // 2
        y_ += -rr + int(yy) - ys
        x_ += -rr + int(xx) - xs
        y_ = np.clip(y_, 0, img.shape[0] - 1)
        x_ = np.clip(x_, 0, img.shape[1] - 1)
        a[(y_, x_)] = 1
        img = (img / img.max() * 255).astype(np.uint8)
        img = draw_contour(img, a)
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
    out_folder.mkdir(exist_ok=True)
    zip_dict = {}
    for z in in_folder.glob('*.zip'):
        with zipfile.ZipFile(z) as zz:
            for tif in zz.namelist():
                zip_dict[Path(tif).name.split('_')[1]] = z, tif
    df_folder = Path('../../data/case1/detection')
    frames = [(p.with_suffix('.tif').name, pd.read_csv(p)) for p in df_folder.glob('*.csv')]
    with open('../../data/case1/traces.pickle', 'rb') as f:
        traces = pickle.load(f)
    with Pool(8) as p:
        for res in tqdm(p.imap(main, zip(traces, repeat(zip_dict, len(traces)), repeat(frames, len(traces)))),
                        total=len(traces)):
            pass
