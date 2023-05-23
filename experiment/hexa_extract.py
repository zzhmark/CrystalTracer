import math
from findmaxima2d import find_maxima, find_local_maxima
from flake_detection.draw import *
from flake_detection.find_circles import *
from scipy.ndimage import generate_binary_structure
from gwdt import gwdt
import numpy as np
from pathlib import Path
import pandas as pd


if __name__ == '__main__':
    data_dir = Path('../test_data')
    for gfp in (data_dir / 'in').rglob('*_GFP*.tif'):
        img = cv2.imread(str(gfp), cv2.IMREAD_GRAYSCALE)
        m = np.quantile(img, 0.0001), np.quantile(img, 0.9999)
        img = img.clip(m[0], m[1])
        img = ((img - m[0]) / (m[1] - m[0]) * 255).astype(np.uint8)
        img = cv2.bilateralFilter(img, -1, 10, 10)
        # smoothing & scaling
        # img = cv2.GaussianBlur(img, (3, 3), 0)

        # distance transform and find centers
        thr = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)[0]
        fgnd_img = img - thr
        fgnd_img[fgnd_img < 0] = 0
        structure = generate_binary_structure(img.ndim, 10)
        dt = gwdt(fgnd_img, structure)
        y, x, regs = find_maxima(dt, find_local_maxima(dt), 10)

        # detection
        rad = radius_estimate(img, y, x, lowest_cutoff=thr, cutoff_ratio=0.2, bg_thr=0.0001)
        area, floods = area_estimate(img, y, x, rad, lowest_cutoff=thr, cutoff_ratio=0.2)
        # area_center = []
        # for i in range(len(y)):
        #     pix = np.argwhere(flood == i)
        #     ct = pix.mean(axis=0)
        #     area_center.append(ct)
        # y = [int(c[0]) for c in area_center]
        # x = [int(c[1]) for c in area_center]
        # rad = radius_estimate(img, y, x, lowest_cutoff=thr, cutoff_ratio=0.2)

        # filter
        # tf = []
        # for i, yy, xx, r, a in zip(range(len(y)), y, x, rad, area):
        #     # area is too big
        #     if a > r**2 * math.pi * 1.5:
        #         tf.append(False)
        #         continue
        #     pix = np.argwhere(flood==i)
        #     # area center is away from the circle
        #     ct = pix.mean(axis=0)
        #     dist = ((yy-ct[0])**2+(xx-ct[1])**2)**0.5
        #     if dist > r:
        #         tf.append(False)
        #         continue
        #     # area is not likely a circle
        #     # plot a radial distribution, it should be flat
        #     if a > 100:
        #         rr = r * 24
        #         z = [0] * rr
        #         max_dist = 0
        #         for ct in pix:
        #             dist = ((yy - ct[0]) ** 2 + (xx - ct[1]) ** 2) ** 0.5
        #             dist = min(round(dist), rr - 1)
        #             max_dist = max(dist, max_dist)
        #             z[dist] += 1
        #         for k in range(1, rr):
        #             z[k] /= 2 * math.pi * k
        #         s = np.quantile((abs(np.array(z[:max_dist+1]) - 1)), 0.5)    # a bit lower radius range, for robustness
        #         if s > 0.5:
        #             tf.append(False)
        #             continue
        #     tf.append(True)
        # y = [y[i] for i, f in enumerate(tf) if f]
        # x = [x[i] for i, f in enumerate(tf) if f]
        # area = [area[i] for i, f in enumerate(tf) if f]
        # rad = [rad[i] for i, f in enumerate(tf) if f]

        flood = np.ones_like(img) * -1
        for i, (ys, xs, f) in enumerate(floods):
            y_, x_ = np.nonzero(f)
            y_ += ys
            x_ += xs
            flood[(y_, x_)] = i

        out1 = draw_areas(img, flood)
        out2 = draw_circles(img, y, x, [int(r) for r in rad])
        path = data_dir / 'out' / Path(gfp).relative_to(data_dir / 'in')
        cv2.imwrite(str(path.with_name(path.name.split('.')[0] + '_area.tif')), out1)
        cv2.imwrite(str(path.with_name(path.name.split('.')[0] + '_circle.tif')), out2)

        pd.DataFrame({'y': y, 'x': x, 'radius': rad, 'area': area}).to_csv(path.with_suffix('.csv'))


