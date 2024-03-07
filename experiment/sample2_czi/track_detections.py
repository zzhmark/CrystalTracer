from crystal_tracer.algorithm.tracking import independent_match
import pickle
import pandas as pd
from pathlib import Path
import math

if __name__ == '__main__':
    df_folder = Path('../../data/case2/detection')
    frames = [pd.read_csv(df_folder / f"{i}.csv") for i in range(len([*df_folder.glob('*.csv')]))]
    for f in frames:
        f['area'] = math.pi * f['radius']**2
    res = independent_match(frames, area_normalizer=5, nn=10)
    with open('../../data/case2/traces.pickle', 'wb') as f:
        pickle.dump(res, f)