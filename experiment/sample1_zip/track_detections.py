from crystal_tracer.algorithm.tracking import independent_match
import pickle
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    df_folder = '../data/detection'
    frames = [pd.read_csv(p) for p in Path(df_folder).glob('*.csv')]
    res = independent_match(frames)
    with open('../../data/case1/traces.pickle', 'wb') as f:
        pickle.dump(res, f)