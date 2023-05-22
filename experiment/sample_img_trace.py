from flake_detection.trace import backtrace
import pickle
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    df_folder = '../test_data/detection'
    frames = []
    for p in Path(df_folder).glob('*.csv'):
        frames.append(pd.read_csv(p))
    res = backtrace(frames, 3, 30, 12)
    with open('../test_data/traces.pickle', 'wb') as f:
        pickle.dump(res, f)