import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QButtonGroup, QFileDialog, QMessageBox, QVBoxLayout, \
    QProgressDialog
from crystal_tracer.gui.ui_loader import loadUi
from PySide6.QtCore import Slot, QThread, Qt
from pathlib import Path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import pickle
from crystal_tracer.img_utils import load_czi_slice, get_czi_shape
from crystal_tracer.gui.workers import PreviewRunner, DetectionTask, TrackingTask, RecordingTask
import os


class CrystalTracerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # data
        self.task = None
        self._img_path: Path | None = None
        self._img_shape: tuple[int, int, int, int] | None = None
        self._table_paths: list[Path] | None = None
        self._mask_paths: list[Path] | None = None
        self._tracks: list[list[tuple[int, int]]] | None = None
        self._plot_paths: list[Path] | None = None
        self._video_paths: list[Path] | None = None

        # Load the UI file
        loadUi('main.ui', self)
        self.panel_bouton_group = QButtonGroup(self)
        self.panel_bouton_group.setExclusive(True)
        self.panel_bouton_group.addButton(self.turn2detection)
        self.panel_bouton_group.addButton(self.turn2tracking)
        self.panel_bouton_group.addButton(self.turn2recording)
        self.stacked_widget.setCurrentIndex(0)
        self.turn2detection.setChecked(True)

        # turn pages
        self.turn2detection.clicked.connect(self.on_turn_clicked)
        self.turn2tracking.clicked.connect(self.on_turn_clicked)
        self.turn2recording.clicked.connect(self.on_turn_clicked)

        # actions
        self.action_open_image.triggered.connect(self.on_import_image)
        self.action_open_detection.triggered.connect(self.on_open_detection)
        self.action_open_tracking.triggered.connect(self.on_open_tracking)
        self.action_open_recording.triggered.connect(self.on_open_recording)
        self.action_open_wkdir.triggered.connect(self.open_wkdir)
        self.action_open.triggered.connect(self.open_sth)
        self.addAction(self.action_open)
        self.action_set_wkdir.triggered.connect(self.init_wkdir)

        # browse buttons
        self.browse_save_detection.clicked.connect(self.set_save_dir_detection)
        self.browse_save_tracking.clicked.connect(self.set_save_path_tracking)
        self.browse_save_recording.clicked.connect(self.set_save_dir_recording)

        # process buttons
        self.run_detection.clicked.connect(self.task_detection)
        self.run_tracking.clicked.connect(self.task_tracking)
        self.run_recording.clicked.connect(self.task_recording)

        # scroll
        self.raw_scroll.valueChanged.connect(self.status_show_page)
        self.raw_scroll.sliderReleased.connect(self.detection_update_raw)
        self.raw_scroll.actionTriggered.connect(self.detection_update_raw_action)
        self.raw_scroll.setContextMenuPolicy(Qt.NoContextMenu)

        # thread limit
        n = QThread.idealThreadCount()
        self.nproc_detection.setMaximum(n)
        self.nproc_detection.setValue(n)
        self.nproc_recording.setMaximum(n)
        self.nproc_recording.setValue(n)

        # canvas
        fig = plt.figure()
        ax = self.ax_detection_input = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        vb = QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        self.canvas_detection_input = FigureCanvasQTAgg(fig)
        tb = NavigationToolbar2QT(self.canvas_detection_input)
        vb.addWidget(self.canvas_detection_input)
        vb.addWidget(tb)
        self.detection_input.setLayout(vb)

        fig = plt.figure()
        ax = self.ax_detection_output = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        vb = QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        self.canvas_detection_output = FigureCanvasQTAgg(fig)
        tb = NavigationToolbar2QT(self.canvas_detection_output)
        vb.addWidget(self.canvas_detection_output)
        vb.addWidget(tb)
        self.detection_output.setLayout(vb)

        self.fig_tracking_result = plt.figure()
        plt.tight_layout()
        vb = QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        vb.addWidget(FigureCanvasQTAgg(self.fig_tracking_result))
        self.tracking_result.setLayout(vb)

        self.fig_growth_plot = plt.figure()
        plt.tight_layout()
        vb = QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        vb.addWidget(FigureCanvasQTAgg(self.fig_growth_plot))
        self.growth_plot.setLayout(vb)

        self.fig_growth_video = plt.figure()
        plt.tight_layout()
        vb = QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        vb.addWidget(FigureCanvasQTAgg(self.fig_growth_video))
        self.growth_video.setLayout(vb)

        # misc
        self.radio_gfp.toggled.connect(self.detection_update_raw)
        self.preview_filter.clicked.connect(self.on_preview_detection)
        self.erase_image.clicked.connect(self.on_erase_image)
        self.erase_detection.clicked.connect(self.on_erase_detection)
        self.erase_tracks.clicked.connect(self.on_erase_tracks)
        self.erase_recording.clicked.connect(self.on_erase_recording)
        self.visualize_tracking.clicked.connect(self.on_visualize_tracking)

    def on_visualize_tracking(self):
        pass

    def on_erase_image(self):
        self.label_image.setText('')
        self._img_shape = None
        self._img_path = None

    def on_erase_detection(self):
        self.label_detection.setText('')
        self._table_paths = None
        self._mask_paths = None

    def on_erase_tracks(self):
        self.label_tracks.setText('')
        self._tracks = None

    def on_erase_recording(self):
        self.label_recording.setText('')
        self._plot_paths = None
        self._video_paths = None

    @Slot()
    def detection_update_raw(self):
        if self._img_path is None:
            return
        if self.radio_gfp.isChecked():
            channel = self.gfp_channel.value()
        else:
            channel = self.bf_channel.value()
        img = load_czi_slice(self._img_path, channel, self.raw_scroll.value())
        self.ax_detection_input.imshow(img, interpolation='none', cmap='gray')
        self.canvas_detection_input.draw()

    @Slot(int)
    def detection_update_raw_action(self, action):
        if action != 7:
            self.detection_update_raw()

    @Slot(int)
    def status_show_page(self, val):
        if self._img_path is None:
            self.statusBar().showMessage(f'No image imported.')
        else:
            self.statusBar().showMessage(f'Timestamp: {val}')

    @Slot()
    def on_preview_detection(self):
        if self._img_path is None:
            QMessageBox.critical(self, 'Error', 'No image imported.')
            return
        print('Perform preview filtering')
        progress = QProgressDialog('Filtering the image slice..', 'Cancel', 0, 0, self)
        progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        progress.setCancelButton(None)
        progress.setWindowTitle('Processing')
        progress.setModal(True)
        progress.show()
        task = PreviewRunner(self._img_path, self.gfp_channel.value(), self.bf_channel.value(), self.raw_scroll.value(),
                             self.block_size.value(), self.tolerance.value(), self.cutoff_ratio.value(),
                             self.bg_thr.value(), self.active_contour.isChecked(),
                             (self.shift_x.value(), self.shift_y.value()), self.dog_sigma.value(),
                             self.sobel_sigma.value(), self.bf_weight.value(), self.gfp_weight.value(),
                             self.dilation_radius.value())
        task.finished.connect(progress.close)
        progress.canceled.connect(task.quit)
        task.start()
        progress.exec()
        if task.img is not None:
            self.ax_detection_output.imshow(task.img, interpolation='none')
            self.canvas_detection_output.draw()

    @Slot()
    def task_detection(self):
        if self._img_path is None:
            QMessageBox.critical(self, 'Error', 'No image imported.')
            return
        if not self.save_dir_detection.text():
            QMessageBox.critical(self, 'Error', 'No save directory specified.')
            return
        print('Perform detection')
        self.setEnabled(False)
        self.detection_progress.setMaximum(self._img_shape[0])
        self.detection_progress.setValue(0)
        self.task = DetectionTask(self.nproc_detection.value(), self._img_shape[0], self._img_path,
                                  self.gfp_channel.value(), self.bf_channel.value(),
                                  self.raw_scroll.value(), self.save_dir_detection.text(),
                                  self.block_size.value(), self.tolerance.value(), self.cutoff_ratio.value(),
                                  self.bg_thr.value(),
                                  self.active_contour.isChecked(), (self.shift_x.value(), self.shift_y.value()),
                                  self.dog_sigma.value(),
                                  self.sobel_sigma.value(), self.bf_weight.value(), self.gfp_weight.value(),
                                  self.dilation_radius.value())
        self.task.increment.connect(self.increment_detection_progress)
        self.task.finished.connect(self.after_detection)
        self.task.start()

    @Slot()
    def open_sth(self):
        index = self.stacked_widget.currentIndex()
        if index == 0:
            self.on_import_image()
        elif index == 1:
            self.on_open_detection()
        elif index == 2:
            self.on_import_image()
            self.on_open_detection()
            self.on_open_tracking()
        else:
            raise Exception('Invalid index')

    @Slot()
    def after_detection(self):
        self.task = None
        self.resolve_detection(self.save_dir_detection.text())
        self.setEnabled(True)

    @Slot()
    def increment_detection_progress(self):
        self.detection_progress.setValue(self.detection_progress.value() + 1)

    @Slot()
    def task_tracking(self):
        if self._table_paths is None:
            QMessageBox.critical(self, 'Error', 'No detection results specified.')
            return
        if not self.save_path_tracking.text():
            QMessageBox.critical(self, 'Error', 'No save path specified.')
            return
        print('Perform detection')
        self.setEnabled(False)
        self.tracking_progress.setMaximum(len(self._table_paths))
        self.tracking_progress.setValue(0)
        self.task = TrackingTask(self._table_paths, self.save_path_tracking.text(),
                                 self.dist_thr.value(), self.nn.value(), self.time_gap.value(),
                                 (self.time_range_lower.value(), self.time_range_upper.value()),
                                 self.area_overflow.value(), self.area_diff.value())
        self.task.increment.connect(self.increment_tracking_progress)
        self.task.finished.connect(self.after_tracking)
        self.task.start()

    @Slot()
    def increment_tracking_progress(self):
        self.tracking_progress.setValue(self.tracking_progress.value() + 1)

    @Slot()
    def after_tracking(self):
        self.task = None
        self.load_tracking(self.save_path_tracking.text())
        self.setEnabled(True)

    @Slot()
    def task_recording(self):
        if self._img_path is None:
            QMessageBox.critical(self, 'Error', 'No image imported.')
            return
        if self._table_paths is None:
            QMessageBox.critical(self, 'Error', 'No detection results specified.')
            return
        if self._tracks is None:
            QMessageBox.critical(self, 'Error', 'No tracking results specified.')
            return
        if not self.save_dir_recording.text():
            QMessageBox.critical(self, 'Error', 'No save directory specified.')
            return
        print('Perform recording')
        self.setEnabled(False)
        self.recording_progress.setMaximum(len(self._table_paths))
        self.recording_progress.setValue(0)
        self.task = RecordingTask(self.nproc_recording.value(), self._tracks, self.save_dir_recording.text(),
                                 self._img_path,  self._table_paths, self._mask_paths, 30, 10.)
        self.task.increment.connect(self.increment_recording_progress)
        self.task.finished.connect(self.after_recording)
        self.task.start()

    @Slot()
    def increment_recording_progress(self):
        self.recording_progress.setValue(self.recording_progress.value() + 1)

    @Slot()
    def after_recording(self):
        self.task = None
        self.find_recording(self.save_dir_recording.text())
        self.setEnabled(True)


    @Slot()
    def set_save_dir_detection(self):
        path = self.save_dir_detection.text()
        path = QFileDialog.getExistingDirectory(self, 'Select a directory for saving detection results',
                                                path if path else os.path.expanduser('~'))
        if path:
            self.save_dir_detection.setText(path)

    @Slot()
    def set_save_path_tracking(self):
        path = self.save_path_tracking.text()
        path = QFileDialog.getSaveFileName(self, 'Specify a file path to save tracking results',
                                           path if path else os.path.expanduser('~'), 'Python Pickle File (*.pkl)')[0]
        if path:
            self.save_path_tracking.setText(path)

    @Slot()
    def set_save_dir_recording(self):
        path = self.save_path_recording.text()
        path = QFileDialog.getExistingDirectory(self, 'Select a directory for saving recording files',
                                                path if path else os.path.expanduser('~'))
        if path:
            self.save_path_recording.setText(path)

    @Slot()
    def open_wkdir(self):
        path = QFileDialog.getExistingDirectory(self, 'Load from working directory', os.path.expanduser('~'))
        if path:
            msg = ''
            path = Path(path)
            p = path / 'detection'
            self.save_dir_detection.setText(str(p))
            try:
                self.resolve_detection(p, False)
            except:
                msg = f'Detection results resolving failed.\n{p}'
                print(msg)
                QMessageBox.warning(self, 'Warning', msg)
            p = path / 'tracks.pkl'
            self.save_path_tracking.setText(str(p))
            try:
                self.load_tracking(p, False)
            except:
                msg = f'Tracking results loading failed.\n{p}'
                print(msg)
                QMessageBox.warning(self, 'Warning', msg)
            p = path / 'recording'
            self.save_dir_recording.setText(str(p))
            try:
                self.find_recording(p, False)
            except:
                msg = f'Recording file resolving failed.\n{p}'
                print(msg)
                QMessageBox.warning(self, 'Warning', msg)

    @Slot()
    def init_wkdir(self):
        path = QFileDialog.getExistingDirectory(self, 'Set working directory', os.path.expanduser('~'))
        if path:
            path = Path(path)
            p1 = path / 'detection'
            p1.mkdir(exist_ok=True, parents=True)
            p2 = path / 'recording'
            p2.mkdir(exist_ok=True, parents=True)
            self.save_dir_detection.setText(str(p1))
            self.save_path_tracking.setText(str(path / 'tracks.pkl'))
            self.save_dir_recording.setText(str(p2))
            QMessageBox.information(self, 'Success', 'Initializing working directory complete')

    @Slot()
    def on_import_image(self):
        path = QFileDialog.getOpenFileName(self, 'Import a raw image stack', os.path.expanduser('~'),
                                           filter='Carl Zeiss Images (*.czi)')[0]
        if path:
            self._img_path = Path(path)
            if self._img_path.suffix == '.czi':
                t, c, y, x = self._img_shape = get_czi_shape(self._img_path)
                self.gfp_channel.setMaximum(c - 1)
                self.gfp_channel.setValue(0)
                self.bf_channel.setMaximum(c - 1)
                self.bf_channel.setValue(1)
                self.raw_scroll.setMaximum(t - 1)
                msg = f'{self._img_path} (T: {t}, C: {c}, Y: {y}, X: {x})'
                print(msg)
                self.label_image.setText(msg)
            self.raw_scroll.setValue(0)
            self.detection_update_raw()

    @Slot()
    def on_open_detection(self):
        path = QFileDialog.getExistingDirectory(self, 'Load detections results', os.path.expanduser('~'))
        if path:
            self.resolve_detection(path)

    def resolve_detection(self, path, box=True):
        print('Loading detection results', path)
        path = Path(path)
        df = sorted(path.glob('*.csv'))
        npz = sorted(path.glob('*.npz'))
        assert len(df) == len(npz), "No. of tables and No. of masks fail to match"
        assert len(df) > 0, "Couldn't find anything to load"
        self._table_paths = df
        self._mask_paths = npz
        msg = f'{path} ({len(df)})'
        self.label_detection.setText(msg)
        msg = f'Found {len(df)} detection instances.'
        print(msg)
        if box:
            QMessageBox.information(self, 'Success', msg)

    @Slot()
    def on_open_tracking(self):
        path = QFileDialog.getOpenFileName(self, 'Load tracking results', os.path.expanduser('~'),
                                           'Python Pickle File (*.pkl)')[0]
        if path:
            self.load_tracking(path)

    def load_tracking(self, path, box=True):
        print('Loading tracking results', path)
        with open(path, 'rb') as f:
            self._tracks = pickle.load(f)
        msg = f'{path} ({len(self._tracks)})'
        self.label_tracks.setText(msg)
        msg = f'Tracking results loaded.'
        print(msg)
        if box:
            QMessageBox.information(self, 'Success', msg)

    @Slot()
    def on_open_recording(self):
        path = QFileDialog.getExistingDirectory(self, 'Load recording files', os.path.expanduser('~'))
        if path:
            self.find_recording(path)

    def find_recording(self, path, box=True):
        path = Path(path)
        plots = sorted(path.glob('*.png'))
        videos = sorted(path.glob('*.avi'))
        assert len(plots) == len(videos), "No. of plots and No. of videos fail to match"
        assert len(plots) > 0, "Couldn't find anything to load"
        self._plot_paths = plots
        self._video_paths = videos
        msg = f'{path} ({len(plots)})'
        self.label_recording.setText(msg)
        msg = f'Found {len(plots)} recording instances.'
        print(msg)
        if box:
            QMessageBox.information(self, 'Success', msg)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CrystalTracerApp()
    window.show()
    sys.exit(app.exec())
