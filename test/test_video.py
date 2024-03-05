from crystal_tracer.visual.video import make_video
from pathlib import Path
import pickle

wkdir = Path('../data/case3')


if __name__ == '__main__':
    with open(wkdir / 'tracks.pkl', 'rb') as f:
        tracks = pickle.load(f)
    tabs = sorted((wkdir / 'detection').glob('*.csv'), key= lambda p: int(p.stem))
    masks = sorted((wkdir / 'detection').glob('*.npz'), key= lambda p: int(p.stem))
    make_video(tracks[77], Path('77.avi'), Path(r"D:\下载\FC1-01-Create Image Subset-01.czi"),
               tabs, masks, 50)