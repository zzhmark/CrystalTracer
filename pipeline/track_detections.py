from tracer.tracking import federated_track
import pickle
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    df_folder = '../test_data/detection'
    frames = [pd.read_csv(p) for p in Path(df_folder).glob('*.csv')]
    res = federated_track(frames)
    with open('../test_data/traces.pickle', 'wb') as f:
        pickle.dump(res, f)