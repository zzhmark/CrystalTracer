import pickle
from pathlib import Path
from czifile import imread
from multiprocessing import Pool
from tqdm import tqdm
from crystal_tracer.algorithm.detection import frame_detection


out_dir = Path('../../data/case2')
flood_dir = out_dir / "flood"
csv_dir = out_dir / "detection"


def main(args):
    stack, slice = args
    gfp = stack[0, slice, 0, ..., 0]
    bf = stack[0, slice, 1, ..., 0]
    frame, seg = frame_detection(gfp, bf, thr_blk_sz=21, active_contour=False)
    with open(flood_dir / f"{slice}.pickle", 'wb') as f:
        pickle.dump(seg, f)
    frame.to_csv(csv_dir / f"{slice}.csv", index=False)


if __name__ == '__main__':
    flood_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)
    stacks = imread(r"D:\下载\Mitosis_Transient_scene1.czi")
    # main((stacks, 400))
    arglist = [(stacks, i) for i in range(stacks.shape[1])]
    with Pool(10) as p:
        for res in tqdm(p.imap(main, arglist), total=len(arglist)): pass
