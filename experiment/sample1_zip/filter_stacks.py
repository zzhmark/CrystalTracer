import pickle
from pathlib import Path
import zipfile
import tempfile
import cv2
from multiprocessing import Pool

from tqdm import tqdm
from crystal_tracer.algorithm.detection import frame_detection

flood_folder = Path('../../data/case1/flood')
flood_folder.mkdir(exist_ok=True)
detection_folder = Path('../../data/case1/detection')
detection_folder.mkdir(exist_ok=True)


def main(args):
    tmpd, zf1, gfp_path, zf2, bf_path = args
    with zipfile.ZipFile(zf1) as z:
        z.extract(gfp_path, tmpd)
    with zipfile.ZipFile(zf2) as z:
        z.extract(bf_path, tmpd)
    p1 = Path(tmpd) / gfp_path
    p2 = Path(tmpd) / bf_path
    gfp = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
    bf = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
    frame, segment = frame_detection(gfp, bf)
    gfp_path = Path(gfp_path).name.split('_')[1]
    with open((flood_folder / gfp_path).with_suffix('.pickle'), 'wb') as f:
        pickle.dump(segment, f)
    frame.to_csv((detection_folder / gfp_path).with_suffix('.csv'), index=False)


if __name__ == '__main__':
    in_folder = Path(r'D:\下载\Compressed')
    with tempfile.TemporaryDirectory() as tmpdir:
        arglist = {}
        for z in in_folder.glob('*.zip'):
            with zipfile.ZipFile(z) as zz:
                for tif in zz.namelist():
                    if not tif.endswith('.tif'):
                        continue
                    if not Path(tif).name.split('_')[1].startswith('GFP'):
                        continue
                    arglist[tif] = [tmpdir, z, tif]
        for z in in_folder.glob('*.zip'):
            with zipfile.ZipFile(z) as zz:
                for tif in zz.namelist():
                    if not tif.endswith('.tif'):
                        continue
                    if not Path(tif).name.split('_')[1].startswith('BF'):
                        continue
                    t = tif.replace('_BF', '_GFP')
                    arglist[t].extend([z, tif])
        with Pool(12) as p:
            for res in tqdm(p.imap(main, arglist), total=len(arglist)):
                pass
