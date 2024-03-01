from crystal_tracer.visual.draw import draw_patches
import pandas as pd
import pickle
import numpy as np
from skimage.io import imsave

if __name__ == '__main__':
    with open('../data/case2/flood/100.pickle', 'rb') as f:
        p = pickle.load(f)
    y = [i[0] for i in p[1]]
    x = [i[1] for i in p[1]]
    m = [i[2].astype(np.uint8) for i in p[1]]
    img = np.zeros([2000, 2000], dtype=np.uint8)
    c = draw_patches(img, y, x, m)
    imsave('test.png', c)
