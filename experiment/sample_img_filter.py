import pickle
from pathlib import Path
import zipfile
import tempfile
from multiprocessing import Pool

import pandas as pd

from flake_detection.filter import circle_filter
import cv2
from flake_detection.draw import *

flood_folder = Path('../test_data/flood')
flood_folder.mkdir(exist_ok=True)
detection_folder = Path('../test_data/detection')
detection_folder.mkdir(exist_ok=True)


def extract_and_filter(zf, tmpd, name):
    with zipfile.ZipFile(zf) as z:
        z.extract(name, tmpd)
    p = Path(tmpd) / name
    res, flood = circle_filter(p)
    name = Path(name).name.split('_')[1]
    # img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    # out = draw_areas(img, flood)
    # cv2.imwrite(str(flood_folder / name), out)
    pd.DataFrame(res).to_csv((detection_folder / name).with_suffix('.csv'), index=False)
    return name, res


if __name__ == '__main__':
    in_folder = Path(r'D:\下载\Compressed')
    with tempfile.TemporaryDirectory() as tmpdir:
        arglist = []
        for z in in_folder.glob('*.zip'):
            with zipfile.ZipFile(z) as zz:
                for tif in zz.namelist():
                    if not tif.endswith('.tif'):
                        continue
                    if not Path(tif).name.split('_')[1].startswith('GFP'):
                        continue
                    arglist.append([z, tmpdir, tif])
        with Pool(12) as p:
            p.starmap(extract_and_filter, arglist)
