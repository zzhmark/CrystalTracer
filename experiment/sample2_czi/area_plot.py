import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm


plot_folder = Path('../../data/case2/plot')


def main(args):
    trace, frames = args
    out_path = plot_folder / f"{trace[0][1]}.png"
    x = []
    y = []
    if len(trace) < 10:
        return
    for fr, j in trace:
        x.append(len(frames) - fr - 1)
        y.append(frames[x[-1]].at[j, 'area'])
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('timestamp')
    plt.ylabel('flooeded area')
    plt.title('crystal growth')
    plt.xlim(0, len(frames))
    plt.ylim(0, 300)
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    df_folder = Path('../../data/case2/detection')
    plot_folder.mkdir(exist_ok=True)
    frames = [pd.read_csv(df_folder / f"{i}.csv") for i in range(len([*df_folder.glob('*.csv')]))]
    with open('../../data/case2/traces.pickle', 'rb') as f:
        traces = pickle.load(f)
    with Pool(12) as p:
        for res in tqdm(p.imap(main, zip(traces, repeat(frames, len(traces))), chunksize=10), total=len(traces)):
            pass
