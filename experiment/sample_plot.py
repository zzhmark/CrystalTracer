import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd
from multiprocessing import Pool


def plot(out_path, trace, dfs):
    x = []
    y = []
    for fr, j in trace.items():
        x.append(len(dfs) - fr - 1)
        y.append(dfs[x[-1]].at[j, 'area'])
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('timestamp')
    plt.ylabel('flooeded area')
    plt.title('crystal growth')
    plt.xlim(0, len(dfs))
    plt.ylim(0, 1000)
    plt.savefig(out_path)
    plt.close()



if __name__ == '__main__':
    df_folder = '../test_data/detection'
    plot_folder = Path('../test_data/plot')
    plot_folder.mkdir(exist_ok=True)
    res = []
    for p in Path(df_folder).glob('*.csv'):
        res.append(pd.read_csv(p))
    with open('../test_data/traces.pickle', 'rb') as f:
        traces = pickle.load(f)
    arglist = []
    for k, v in traces.items():
        arglist.append([plot_folder / f"{k}.png", v, res])
    with Pool(10) as p:
        p.starmap(plot, arglist)