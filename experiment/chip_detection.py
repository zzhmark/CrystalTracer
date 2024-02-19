from crystal_tracer.visual.draw import *
from crystal_tracer.algorithm.detection import *
from scipy.ndimage import generate_binary_structure
from crystal_tracer.algorithm.gwdt import gwdt
import numpy as np
from pathlib import Path
import pandas as pd
from skimage.segmentation import morphological_geodesic_active_contour, disk_level_set
from skimage.filters import difference_of_gaussians
from skimage.feature._canny import _preprocess
import scipy.ndimage as ndi
from skimage.morphology import dilation, disk


if __name__ == '__main__':
    data_dir = Path('../data')
    for gfp in (data_dir / 'in').rglob('*_GFP*.tif'):
        img = cv2.imread(str(gfp), cv2.IMREAD_GRAYSCALE)
        m = np.quantile(img, 0.0001), np.quantile(img, 0.9999)
        img = img.clip(m[0], m[1])
        img = ((img - m[0]) / (m[1] - m[0]) * 255).astype(np.uint8)

        img3 = cv2.imread(str(gfp).replace('_GFP', '_BF'), cv2.IMREAD_GRAYSCALE)
        img2 = difference_of_gaussians(img3, 1)
        smoothed, eroded_mask = _preprocess(img, None, 1, 'constant', 0)
        jsobel = ndi.sobel(smoothed, axis=1)
        isobel = ndi.sobel(smoothed, axis=0)
        magnitude = isobel * isobel
        magnitude += jsobel * jsobel
        np.sqrt(magnitude, out=magnitude)
        img4 = cv2.warpAffine(magnitude, np.float32([[1, 0, 2], [0, 1, 0]]), (img.shape[1], img.shape[0]))
        img2 -= img4 * 0.1

        # distance transform and find centers
        thr = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)[0]
        fgnd_img = img - thr
        fgnd_img[fgnd_img < 0] = 0
        structure = generate_binary_structure(img.ndim, 10)
        dt = gwdt(fgnd_img, structure)
        y, x, regs = find_maxima(dt, find_local_maxima(dt), 10)

        # detection
        rad = radius_estimate(img, y, x, lowest_cutoff=thr, cutoff_ratio=0.5, bg_thr=0.0001)
        area = []
        # flood = np.ones_like(img, dtype=int) * -1
        flood = np.zeros_like(img)
        for i, (yy, xx, rr) in enumerate(zip(y, x, rad)):
            win_rad = int(rr * 2)
            ys = max(0, yy - win_rad)
            xs = max(0, xx - win_rad)
            ye = min(img.shape[0], yy + win_rad)
            xe = min(img.shape[1], xx + win_rad)
            sub_img = img2[ys:ye, xs:xe].astype(float)
            ls = morphological_geodesic_active_contour(sub_img, 20, disk_level_set(sub_img.shape, radius=rr))
            ls = dilation(ls, disk(2))
            area.append(np.sum(ls > 0))
            flood[ys:ye, xs:xe][ls.astype(bool)] = 1
            # flood[ys:ye, xs:xe][ls.astype(bool)] = i

        # area, floods = area_estimate(img, y, x, rad, lowest_cutoff=thr, cutoff_ratio=0.2)


        # flood = np.ones_like(img) * -1
        # for i, (ys, xs, f) in enumerate(floods):
        #     y_, x_ = np.nonzero(f)
        #     y_ += ys
        #     x_ += xs
        #     flood[(y_, x_)] = i

        # out1 = draw_areas(img3, flood)
        # out2 = draw_circles(img, y, x, [int(r) for r in rad])
        out3 = draw_contour(img3, flood)
        path = data_dir / 'out' / Path(gfp).relative_to(data_dir / 'in')
        # cv2.imwrite(str(path.with_name(path.name.split('.')[0] + '_area.tif')), out1)
        # cv2.imwrite(str(path.with_name(path.name.split('.')[0] + '_circle.tif')), out2)
        cv2.imwrite(str(path.with_name(path.name.split('.')[0] + '_contour.tif')), out3)
        # plt.figure()
        # plt.imshow(img3, cmap='gray')
        # plt.axis('off')
        # plt.contour(flood, [0.5], colors='r', linewidths=0.1)
        # plt.savefig(path.with_name(path.name.split('.')[0] + '_contour.tif'), dpi=500)
        # plt.close()

        pd.DataFrame({'y': y, 'x': x, 'area': area}).to_csv(path.with_suffix('.csv'))


