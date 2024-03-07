import pandas as pd
from PySide6.QtCore import QThread, Signal
from crystal_tracer.img_utils import load_czi_slice
from crystal_tracer.algorithm.detection import frame_detection
from crystal_tracer.algorithm.tracking import independent_match, linear_programming
from crystal_tracer.visual.video import make_video
from crystal_tracer.visual.draw import draw_patches
import matplotlib.animation as animation
from skimage.util import img_as_ubyte
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import sys
import os
import asyncio
import aiofiles
import pickle
from io import StringIO
import matplotlib.pyplot as plt
from pathlib import Path


class HidePrint:
    def __init__(self):
        self.origin = None

    def __enter__(self):
        sys.stdout.close()
        sys.stdout = self.origin

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.origin = sys.stdout
        sys.stdout = open(os.devnull, 'w')


def detection_task(time: int, img_path: Path, gfp_channel: int, bf_channel: int, save_dir: Path, *args):
    gfp = load_czi_slice(img_path, gfp_channel, time)
    if gfp_channel == bf_channel:
        bf = gfp
    else:
        bf = load_czi_slice(img_path, bf_channel, time)
    with HidePrint():
        table, mask = frame_detection(gfp, bf, *args)
    np.savez(save_dir / f'{time}.npz', *mask)
    table.to_csv(save_dir / f'{time}.csv', index=False)


class DetectionTask(QThread):
    increment = Signal()

    def __init__(self, max_thread, n_page, *args):
        super().__init__()
        self.max_thread = max_thread
        self.n_page = n_page
        self.args = args

    def run(self):
        with Pool(self.max_thread) as pool:
            results = [pool.apply_async(detection_task, (i, *self.args)) for i in range(self.n_page)]
            for result in tqdm(results, total=self.n_page):
                result.wait()
                self.increment.emit()


class TrackingTask(QThread):
    increment = Signal()

    def __init__(self, table_paths, save_path, *args):
        super().__init__()
        self.table_paths = table_paths
        self.save_path = save_path
        self.args = args
        self.tables = None

    def _load(self):
        async def async_read_csv(path):
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
                return pd.read_csv(StringIO(content))

        async def load_all_tables():
            coroutines = [async_read_csv(path) for path in self.table_paths]
            return await asyncio.gather(*coroutines)

        self.tables = list(asyncio.run(load_all_tables()))

    def run(self):
        self._load()
        tracks = independent_match(self.tables, *self.args, callback=self.increment.emit)
        with open(self.save_path, 'wb') as f:
            pickle.dump(tracks, f)


class TrackingTask2(TrackingTask):
    def run(self):
        self._load()
        tracks = linear_programming(self.tables, *self.args, callback=self.increment.emit)
        with open(self.save_path, 'wb') as f:
            pickle.dump(tracks, f)


def plot_area(path, track, tables, max_area=500):
    x, y = [], []
    for i, j in track:
        x.append(i)
        y.append(tables[x[-1]].at[j, 'area'])
    pd.DataFrame({
        'time': x,
        'area': y
    }).to_csv(path.with_suffix('.csv'), index=False)

    time_interp = np.linspace(x[0], x[-1], x[-1] - x[0] + 1)
    area_interp = np.interp(time_interp, x, y)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    line, = ax.plot([], [])
    ax.set_xlabel('Time elapse')
    ax.set_ylabel('Crystal area')
    ax.set_xlim(0, len(tables))
    ax.set_xlim(0, len(tables))
    ax.set_ylim(0, max_area)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(time_interp[:i], area_interp[:i])
        return line,

    ani = animation.FuncAnimation(fig, animate, len(time_interp), init, interval=20, blit=True, repeat=False)
    ani.save(path.with_suffix('.mp4'), writer='ffmpeg')
    fig.savefig(path.with_suffix('.png'), dpi=300)
    plt.close(fig)


class RecordingTask(QThread):
    increment = Signal()

    def __init__(self, max_thread: int, tracks: list[list[tuple[int, int]]], save_dir: Path, img_path: Path,
                 table_paths: list[Path], mask_paths: list[Path], win_rad: int, frame_rate: float):
        super().__init__()
        self.max_thread = max_thread
        self.tracks = tracks
        self.save_dir = save_dir
        self.img_path = img_path
        self.table_paths = table_paths
        self.mask_paths = mask_paths
        self.win_rad = win_rad
        self.frame_rate = frame_rate

    def run(self):
        with Pool(self.max_thread) as pool:
            results = [pool.apply_async(make_video, (t, self.save_dir / f'{i}.avi', self.img_path,
                                                     self.table_paths, self.mask_paths, self.win_rad, self.frame_rate))
                       for i, t in enumerate(self.tracks)]

            async def async_read_csv(path):
                async with aiofiles.open(path, 'r') as f:
                    content = await f.read()
                    return pd.read_csv(StringIO(content))

            async def load_all_tables():
                coroutines = [async_read_csv(path) for path in self.table_paths]
                return await asyncio.gather(*coroutines)

            tables = asyncio.run(load_all_tables())

            for i, result in tqdm(enumerate(results), total=len(results)):
                plot_area(self.save_dir / f'{i}.mp4', self.tracks[i], tables)
                result.wait()
                self.increment.emit()


class PreviewRunner(QThread):
    def __init__(self, img_path, gfp_channel, bf_channel, page, *args):
        super().__init__()
        self.img_path = img_path
        self.channel = gfp_channel, bf_channel
        self.page = page
        self.args = args
        self.img = None

    def run(self):
        gfp = load_czi_slice(self.img_path, self.channel[0], self.page)
        if self.channel[0] == self.channel[1]:
            bf = gfp
        else:
            bf = load_czi_slice(self.img_path, self.channel[1], self.page)
        with HidePrint():
            table, masks = frame_detection(gfp, bf, *self.args)

        self.img = draw_patches(img_as_ubyte(gfp), table['y_start'].to_list(), table['x_start'].to_list(), masks)


class WalkRunner(QThread):
    def __init__(self, tracks: list[list[tuple[int, int]]], table_paths: list[Path]):
        super().__init__()
        self.tracks = tracks
        self.table_paths = table_paths
        self.timescale = None
        self.walks = None

    def run(self):

        async def async_read_csv(path):
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
                return pd.read_csv(StringIO(content))

        async def load_all_tables():
            coroutines = [async_read_csv(path) for path in self.table_paths]
            return await asyncio.gather(*coroutines)

        tables = asyncio.run(load_all_tables())

        self.walks = []
        tot = len(tables)
        self.timescale = np.linspace(0, tot - 1, tot)
        for t in self.tracks:
            xx = []
            yy = []
            time = []
            for i, j in t:
                y, x = tables[i].loc[j, ['y', 'x']].to_numpy().ravel()
                yy.append(y)
                xx.append(x)
                time.append(i)
            new_time = self.timescale[time[0]:time[-1] + 1]
            yy = np.interp(new_time, time, yy)
            xx = np.interp(new_time, time, xx)
            self.walks.append(np.array([xx, yy, new_time]))
