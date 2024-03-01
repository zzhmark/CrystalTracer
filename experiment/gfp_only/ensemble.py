import pickle
import numpy as np
from pathlib import Path
from crystal_tracer.algorithm.detection import frame_detection
from crystal_tracer.algorithm.tracking import independent_match
import matplotlib.pyplot as plt
import cv2
from crystal_tracer.visual.draw import draw_contour


def video(stacks, frames, trace, flood_dir, out_dir):
    if len(trace) < 100:
        return
    frame_inds = [len(frames) - i - 1 for i, j in trace]
    chip_inds = [j for i, j in trace]
    frame_inds.reverse()
    chip_inds.reverse()
    save_path = out_dir / f'{chip_inds[-1]}_range({frame_inds[0]},{frame_inds[-1]}).avi'
    x = [frames[i]['x'][j] for i, j in zip(frame_inds, chip_inds)]
    y = [frames[i]['y'][j] for i, j in zip(frame_inds, chip_inds)]
    rad = [frames[i]['radius'][j] for i, j in zip(frame_inds, chip_inds)]
    win_rad = 30
    img_seq = []
    ctx = int(x[0])
    cty = int(y[0])
    for i, (xx, yy, rr, fr, key) in enumerate(zip(x, y, rad, frame_inds, chip_inds)):
        with open(flood_dir / f"{fr}.pkl", 'rb') as f:
            flood = pickle.load(f)
        img = stacks[0, fr, 0, ..., 0]
        ys = max(int(cty) - win_rad, 0)
        xs = max(int(ctx) - win_rad, 0)
        ye = min(img.shape[0], int(cty) + win_rad)
        xe = min(img.shape[1], int(ctx) + win_rad)
        img = img[ys:ye, xs:xe]
        a = np.zeros_like(img, dtype=np.uint8)
        y_, x_ = np.nonzero(flood[key])
        rr = flood[key].shape[0] // 2
        y_ += -rr + int(yy) - ys
        x_ += -rr + int(xx) - xs
        y_ = np.clip(y_, 0, img.shape[0] - 1)
        x_ = np.clip(x_, 0, img.shape[1] - 1)
        a[(y_, x_)] = 1
        img = (img / img.max() * 255).astype(np.uint8)
        img = draw_contour(img, a)
        # img = draw_circles(img, [yy - ys], [xx - xs], [rr])
        if i == len(frame_inds) - 1:
            img_seq.append(img)
        else:
            img_seq.extend([img] * (frame_inds[i + 1] - frame_inds[i]))
        cty = int(yy * .5 + cty * .5)
        ctx = int(xx * .5 + ctx * .5)

    vid = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'DIVX'), 10, (win_rad*2, win_rad*2))
    for img in img_seq:
        vid.write(img)


def plot(trace, frames, out_dir: Path):
    out_path = out_dir / f"{trace[0][1]}.png"
    x = []
    y = []
    if len(trace) < 10:
        return
    for fr, j in trace:
        x.append(len(frames) - fr - 1)
        y.append(frames[x[-1]].at[j, 'area'])

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Area')
    ax.set_title('Crystal Growth')
    ax.set_xlim(0, len(frames))
    ax.set_ylim(0, 300)
    fig.savefig(out_path)
    plt.close(fig)


def filter(img: np.ndarray, slice: int, outpath: Path):
    frame, seg = frame_detection(img, img, thr_blk_sz=21, active_contour=True, dilation_radius=0)
    with open(outpath / f"{slice}.pkl", 'wb') as f:
        pickle.dump(seg, f)
    frame.to_csv(outpath / f"{slice}.csv", index=False)
    return frame


if __name__ == '__main__':
    from czifile import imread
    from multiprocessing import Pool
    out_dir = Path('../../data/case3')
    imgs = [r"D:\下载\FC1-01-Create Image Subset-01.czi", r"D:\下载\FC1-01-Create Image Subset-03.czi",
            r"D:\下载\FC1-01-Create Image Subset-04.czi", r"D:\下载\FC1-01-Create Image Subset-05.czi",
            r"D:\下载\10x_I213-1-05-Create Image Subset-01.czi", r"D:\下载\10x_I213-1-05-Create Image Subset-03.czi",
            r"D:\下载\10x_I213-1-05-Create Image Subset-04.czi"]
    with Pool(10) as p:
        for img_path in imgs:
            stacks = imread(img_path)
            path = Path(img_path)
            outpath = out_dir / path.stem

            # filtering
            resp = []
            path = outpath / 'detection'
            path.mkdir(exist_ok=True, parents=True)
            for slice in range(stacks.shape[1]):
                resp.append(p.apply_async(filter, (stacks[0, slice, 0, ..., 0], slice, path)))

            # tracking
            frames = [r.get() for r in resp]
            track = independent_match(frames)
            with open(outpath / 'traces.pickle', 'wb') as f:
                pickle.dump(track, f)

            # plot
            resp = []
            path = outpath / 'plot'
            path.mkdir(exist_ok=True, parents=True)
            for t in track:
                resp.append(p.apply_async(plot, (t, frames, outpath)))
            resp = [r.get() for r in resp]

            # film
            resp = []
            path = outpath / 'video'
            path.mkdir(exist_ok=True, parents=True)
            for t in track:
                resp.append(p.apply_async(plot, (t, frames, outpath / 'detection', outpath)))
            resp = [r.get() for r in resp]