import cv2
import pickle
import pandas as pd
import numpy as np
import math
from pathlib import Path
import zipfile
import tempfile
from multiprocessing import Pool


flood_folder = Path('../test_data/flood')


def draw_areas(img: np.ndarray, flood: np.ndarray, id, color):
    if len(img.shape) == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    out[flood == id] = color
    return out


def video_single(zd, trace, res, save_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        keys = list(trace.values())
        x = [res[len(res) - f - 1][1]['x'][j] for f, j in trace.items()]
        y = [res[len(res) - f - 1][1]['y'][j] for f, j in trace.items()]
        fname = [zd[res[len(res) - f - 1][0]] for f, j in trace.items()]
        flood_name = [Path(res[len(res) - f - 1][0]).with_suffix('.pickle') for f, j in trace.items()]
        fnum = [len(res) - f - 1 for f in trace.keys()]
        win_rad = 50
        ye = None
        xe = None
        x.reverse()
        y.reverse()
        fname.reverse()
        fnum.reverse()
        flood_name.reverse()
        keys.reverse()
        img_seq = []
        for i, (xx, yy, (zf, p), floods, key) in enumerate(zip(x, y, fname, flood_name, keys)):
            with zipfile.ZipFile(zf) as z:
                z.extract(p, tmpdir)
            img = cv2.imread(str(Path(tmpdir) / p), cv2.IMREAD_GRAYSCALE)
            with open(flood_folder / floods, 'rb') as f:
                flood = pickle.load(f)
            ys = max(int(yy) - win_rad, 0)
            xs = max(int(xx) - win_rad, 0)
            ye = min(img.shape[0], int(yy) + win_rad)
            xe = min(img.shape[1], int(xx) + win_rad)
            img = img[ys:ye, xs:xe]
            a = np.zeros_like(img)
            y_, x_ = np.nonzero(flood[key][2])
            y_ += flood[key][0] - ys
            x_ += flood[key][1] - xs
            a[(y_, x_)] = 1
            xx -= xs
            yy -= ys
            img = (img / img.max() * 255).astype(np.uint8)
            img = draw_areas(img, a, 1, [0, 0, 255])
            if i == len(fnum) - 1:
                img_seq.append(img)
            else:
                img_seq.extend([img] * (fnum[i + 1] - fnum[i]))

        vid = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'DIVX'), 20, (win_rad*2, win_rad*2))
        for img in img_seq:
            vid.write(img)


if __name__ == '__main__':
    in_folder = Path(r'D:\下载\Compressed')
    out_folder = Path('../test_data/vid')
    out_folder.mkdir(exist_ok=True)
    zip_dict = {}
    for z in in_folder.glob('*.zip'):
        with zipfile.ZipFile(z) as zz:
            for tif in zz.namelist():
                zip_dict[Path(tif).name.split('_')[1]] = z, tif
    df_folder = '../test_data/detection'
    res = []
    for p in Path(df_folder).glob('*.csv'):
        res.append((p.with_suffix('.tif').name, pd.read_csv(p)))
    with open('../test_data/traces.pickle', 'rb') as f:
        traces = pickle.load(f)
    arglist = []
    for k, t in traces.items():
        tt = list(t)
        tt = tt[-1] - tt[0]
        arglist.append([zip_dict, t, res, out_folder / f'final={k}_range={tt}.avi'])
    with Pool(8) as p:
        p.starmap(video_single, arglist)
