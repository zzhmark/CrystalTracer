import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QButtonGroup, QFileDialog, QMessageBox, QProgressDialog
from crystal_tracer.gui.ui_loader import loadUi
from PySide6.QtCore import Slot, QThread, Signal, QThreadPool, QRunnable
from czifile import CziFile
from pathlib import Path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import imageio
from crystal_tracer.algorithm.detection import frame_detection
from crystal_tracer.algorithm.tracking import independant_match


class CrystalTracerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # data
        self._img_path: Path = None
        self._tables: list[pd.DataFrame] = None
        self._masks: list[np.ndarray] = None
        self._tracks: list[list[tuple[int, int]]] = None
        self._plots: list[np.ndarray] = None
        self._videos: list[np.ndarray] = None

        # Load the UI file
        loadUi('main.ui', self)
        self.turn2detection: QButtonGroup
        self.panel_bouton_group = QButtonGroup(self)
        self.panel_bouton_group.setExclusive(True)
        self.panel_bouton_group.addButton(self.turn2detection)
        self.panel_bouton_group.addButton(self.turn2tracking)
        self.panel_bouton_group.addButton(self.turn2recording)

        # turn pages
        self.turn2detection.clicked.connect(self.on_turn_clicked)
        self.turn2tracking.clicked.connect(self.on_turn_clicked)
        self.turn2recording.clicked.connect(self.on_turn_clicked)

        # actions
        self.action_load_czi.triggered.connect(self.on_load_czi)
        self.action_load_detection.triggered.connect(self.on_load_detection)
        self.action_load_tracking.triggered.connect(self.on_load_tracking)
        self.action_load_recording.triggered.connect(self.on_load_recording)
        self.action_load_wkdir.triggered.connect(self.load_wkdir)
        self.action_set_wkdir.triggered.connect(self.set_wkdir)

        # browse buttons
        self.browse_save_detection.clicked.connect(self.set_save_dir_detection)
        self.browse_save_tracking.clicked.connect(self.set_save_path_tracking)
        self.browse_save_recording.clicked.connect(self.set_save_dir_recording)

        # process buttons
        self.run_detection.clicked.connect(self.task_detection)
        self.run_tracking.clicked.connect(self.task_tracking)
        self.run_recording.clicked.connect(self.task_recording)

        # thread limit
        n = QThread.idealThreadCount()
        self.nproc_detection.setMaximum(n)
        self.nproc_detection.setValue(n)
        self.nproc_recording.setMaximum(n)
        self.nproc_recording.setValue(n)

    @Slot()
    def task_detection(self):
        pool = QThreadPool()
        for i in range(self._czi.shape[1]):
            gfp = self._czi[0, i, 0, ..., 0]
            bf = self._czi[0, i, 1, ..., 0]
            resp.append(p.apply_async(filter, (self._czi[0, slice, 0, ..., 0], slice, path)))
            task = DetectionTask(gfp, bf, ...)
            self.thread_pool.start(task)

    @Slot()
    def task_tracking(self):
        pass

    @Slot()
    def task_recording(self):
        pass

    @Slot()
    def set_save_dir_detection(self):
        path = QFileDialog.getExistingDirectory(self, 'Select a directory for saving detection results')
        if path:
            self.save_dir_detection.setText(path)

    @Slot()
    def set_save_path_tracking(self):
        path = QFileDialog.getSaveFileName(self, 'Specify a file path to save tracks',
                                           filter='Python Pickle File (*.pkl)')[0]
        if path:
            self.save_path_tracking.setText(path)

    @Slot()
    def set_save_dir_recording(self):
        path = QFileDialog.getExistingDirectory(self, 'Select a directory for saving recording files')
        if path:
            self.save_path_recording.setText(path)

    @Slot()
    def load_wkdir(self):
        path = QFileDialog.getExistingDirectory(self, 'Load from working directory')
        if path:
            t1 = time.time()
            msg = ''
            path = Path(path)
            p = path / 'detection'
            if p.is_dir():
                self.load_czi(p, False)
            else:
                msg += f'Could not find detection directory {p}\n'
            p = path / 'tracks.pkl'
            if p.is_file():
                self.load_tracking(p, False)
            else:
                msg += f'Could not find track file {p}\n'
            p = path / 'recording'
            if p.is_dir():
                self.load_recording(p, False)
            else:
                msg += f'Could not find recording directory {p}.\n'
            t2 = time.time()
            msg += f'Loading took {t2 - t1:.2f} seconds.'
            print(msg)
            QMessageBox.information(self, 'Loading complete', msg)

    @Slot()
    def set_wkdir(self):
        path = QFileDialog.getExistingDirectory(self, 'Set working directory')
        if path:
            path = Path(path).absolute()
            self.save_dir_detection.setText(str(path / 'detection'))
            self.save_path_tracking.setText(str(path / 'tracks.pkl'))
            self.save_dir_recording.setText(str(path / 'recording'))
            QMessageBox.information(self, 'Loading from working directory complete',
                                    f'detection results save dir: {self.save_dir_detection.text()}\n'
                                    f'tracks save path: {self.save_path_tracking.text()}'
                                    f'recording save dir: {self.save_dir_recording.text()}')

    @Slot()
    def on_load_czi(self):
        path = QFileDialog.getOpenFileName(self, 'Load microscope image data', filter='Carl Zeiss Images (*.czi)')[0]
        if path:
            self._img_path = Path(path)
            self.load_czi(path)

    def load_czi(self, path, box=True):
        t1 = time.time()
        print('Loading CZI image', path)
        progress = QProgressDialog('Loading CZI image...', 'Cancel', 0, 0, self)
        progress.setCancelButton(None)
        progress.setModal(True)
        task = LoadCZITask(path)
        task.finished.connect(progress.close)
        task.start()
        progress.exec()
        t2 = time.time()
        msg = f'The CZI file has been loaded in {t2 - t1:.2f} seconds.'
        print(msg)
        if box:
            QMessageBox.information(self, 'Loading complete', msg)

    @Slot()
    def on_load_detection(self):
        path = QFileDialog.getExistingDirectory(self, 'Load detections results')
        if path:
            self.load_detection(path)

    def load_detection(self, path, box=True):
        t1 = time.time()
        print('Loading detection results', path)
        path = Path(path)
        df = sorted(path.glob('*.csv'))
        npy = sorted(path.glob('*.npy'))
        assert len(npy) == len(df), "No. of tables and No. of masks fail to match"
        assert len(npy) > 0, "Not find anything to load"
        self._tables = []
        self._masks = []
        progress = QProgressDialog('Loading detection results...', 'Cancel', 0, 100, self)
        progress.setValue(0)
        progress.show()
        progress.setModal(True)
        count = 0
        for p1, p2 in zip(df, npy):
            self._tables.append(pd.read_csv(p1))
            self._masks.append(np.load(p2))
            count += 1
            progress.setValue(100 * count // len(df))
            if progress.wasCanceled():
                self._tables = None
                self._masks = None
                return
            QApplication.processEvents()
        t2 = time.time()
        msg = f'The detection results have been loaded in {t2 - t1:.2f} seconds.'
        print(msg)
        if box:
            QMessageBox.information(self, 'Loading complete', msg)

    @Slot()
    def on_load_tracking(self):
        path = QFileDialog.getOpenFileName(self, 'Load tracking results', filter='Python Pickle File (*.pkl)')[0]
        if path:
            self.load_tracking(path)

    def load_tracking(self, path, box=True):
        t1 = time.time()
        print('Loading tracking results', path)
        with open(path, 'rb') as f:
            self._tracks = pickle.load(f)
        t2 = time.time()
        msg = f'The tracks have been loaded in {t2 - t1:.2f} seconds.'
        print(msg)
        if box:
            QMessageBox.information(self, 'Loading complete', msg)

    @Slot()
    def on_load_recording(self):
        path = QFileDialog.getExistingDirectory(self, 'Load recording files')
        if path:
            self.load_recording(path)

    def load_recording(self, path, box=True):
        t1 = time.time()
        print('Loading recording', path)
        path = Path(path)
        plots = sorted(path.glob('*.png'))
        videos = sorted(path.glob('*.avi'))
        assert len(plots) == len(videos), "No. of plots and No. of videos fail to match"
        assert len(plots) > 0, "Not find anything to load"
        progress = QProgressDialog('Loading recording files...', 'Cancel', 0, 100, self)
        progress.setValue(0)
        progress.show()
        progress.setModal(True)
        count = 0
        self._plots = []
        self._videos = []
        for p1, p2 in zip(plots, videos):
            self._plots.append(plt.imread(p1))
            frames = []
            reader = imageio.get_reader(p2)
            for frame in reader:
                frames.append(frame)
            self._videos.append(np.array(frames))
            count += 1
            progress.setValue(100 * count // len(plots))
            if progress.wasCanceled():
                self._plots = None
                self._videos = None
                return
            QApplication.processEvents()
        t2 = time.time()
        msg = f'The recording files have been loaded in {t2 - t1:.2f} seconds.'
        print(msg)
        if box:
            QMessageBox.information(self, 'Loading complete', msg)

    @Slot()
    def on_turn_clicked(self):
        button = self.sender()
        if button is self.turn2detection:
            index = 0
        elif button is self.turn2tracking:
            index = 1
        elif button is self.turn2recording:
            index = 2
        else:
            raise ValueError("Invalid button")
        self.stacked_widget.setCurrentIndex(index)


class LoadCZITask(QThread):
    finished = Signal()

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.img = None

    def run(self):
        self.img = imread(self.path)
        self.finished.emit()


class DetectionTask(QRunnable):
    def __init__(self, gfp, bf, **params):
        super().__init__()
        self._gfp = gfp
        self._bf = bf
        self.table: pd.DataFrame = None
        self.mask: np.ndarray = None
        self._params = params

    @Slot()
    def run(self):
        self.table, self.mask = frame_detection(self._gfp, self._bf, **self._params)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CrystalTracerApp()
    window.show()
    sys.exit(app.exec())
