import os
import sys
import pickle
import shutil
import configparser
from pathlib import Path
from PySide6.QtCore import Slot, QThread, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QButtonGroup, QFileDialog, QMessageBox, QProgressDialog
from crystal_tracer.gui.ui_loader import loadUi
from crystal_tracer.img_utils import load_czi_slice, get_czi_shape
from crystal_tracer.gui.workers import PreviewRunner, DetectionTask, TrackingTask, RecordingTask, WalkRunner
from crystal_tracer.gui.components import FigureComponent, AnimatedFigureComponent


class CrystalTracerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # data
        self._wkdir: Path | None = None
        self._config_path: Path| None = None
        self._img_path: Path | None = None
        self._tracks_path: Path | None = None
        self._table_paths: list[Path] | None = None
        self._mask_paths: list[Path] | None = None
        self._tracks: list[list[tuple[int, int]]] | None = None
        self._plot_paths: list[Path] | None = None
        self._video_paths: list[Path] | None = None
        self.task = None
        self.ani = None

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
        self.action_load_config.triggered.connect(self.load_config)
        self.action_save_config.triggered.connect(self.save_config)
        self.addAction(self.action_open)
        self.action_set_wkdir.triggered.connect(self.set_wkdir)

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
        self.detection_input.setLayout(FigureComponent())
        self.detection_output.setLayout(FigureComponent())
        self.tracking_result.setLayout(AnimatedFigureComponent())
        self.growth_plot.setLayout(FigureComponent())
        self.growth_video.setLayout(FigureComponent())

        # misc
        self.radio_gfp.toggled.connect(self.detection_update_raw)
        self.preview_filter.clicked.connect(self.on_preview_detection)
        self.visualize_tracks.clicked.connect(self.on_visualize_tracks)
        self.erase_image.clicked.connect(self.on_erase_image)
        self.erase_detection.clicked.connect(self.on_erase_detection)
        self.erase_tracks.clicked.connect(self.on_erase_tracks)
        self.erase_recording.clicked.connect(self.on_erase_recording)
        self.group_input.setEnabled(False)
        self.group_output.setEnabled(False)
        self.group_tracks.setEnabled(False)
        self.group_plot.setEnabled(False)
        self.group_video.setEnabled(False)
        self.play_bar.valueChanged.connect(self.slider_changed)

    @Slot()
    def load_config(self):
        if self._config_path is not None:
            path = self._config_path
        elif self._wkdir is not None:
            path = self._wkdir
            if self._img_path is not None:
                path = path / self._img_path.with_suffix('.ini').name
        elif self._img_path is not None:
            path = self._img_path.with_suffix('.ini')
        else:
            path = os.path.expanduser('~')
        path = QFileDialog.getOpenFileName(self, 'Load configuration', str(path), 'INI files (*.ini)')[0]
        if not path:
            return
        config = configparser.ConfigParser()
        config.read(path, encoding='utf-8')
        print('Load config')
        if config.has_section('paths.open'):
            paths = config['paths.open']
            if 'wkdir' in paths:
                self._wkdir = paths['wkdir']
            if 'image' in paths:
                self.import_image(paths['image'])
            if 'detection' in paths:
                self.resolve_detection(paths['detection'])
            if 'tracks' in paths:
                self.load_tracking(paths['tracks'])
            if 'recording' in paths:
                self.find_recording(paths['recording'])
        if config.has_section('paths.save'):
            paths = config['paths.save']
            if 'detection' in paths:
                self.save_dir_detection.setText(paths['detection'])
            if 'tracks' in paths:
                self.save_path_tracking.setText(paths['tracks'])
            if 'recording' in paths:
                self.save_dir_recording.setText(paths['recording'])
        if config.has_section('parameters.detection'):
            section = config['parameters.detection']
            self.gfp_channel.setValue(int(section['gfp_channel']))
            self.bf_channel.setValue(int(section['bf_channel']))
            self.block_size.setValue(int(section['block_size']))
            self.tolerance.setValue(int(section['tolerance']))
            self.cutoff_ratio.setValue(float(section['cutoff_ratio'])),
            self.bg_thr.setValue(float(section['bg_thr']))
            self.active_contour.setChecked(config.getboolean('parameters.detection', 'active_contour')),
            self.shift_x.setValue(int(section['shift_x']))
            self.shift_y.setValue(int(section['shift_y']))
            self.dog_sigma.setValue(float(section['dog_sigma']))
            self.sobel_sigma.setValue(float(section['sobel_sigma']))
            self.bf_weight.setValue(float(section['bf_weight']))
            self.gfp_weight.setValue(float(section['gfp_weight']))
            self.dilation_radius.setValue(int(section['dilation_radius']))
        if config.has_section('parameters.tracking'):
            section = config['parameters.tracking']
            self.dist_thr.setValue(float(section['dist_thr'])),
            self.nn.setValue(int(section['nn'])),
            self.time_gap.setValue(int(section['time_gap'])),
            self.min_sampling_count.setValue(int(section['min_sampling_count'])),
            self.min_sampling_elapse.setValue(int(section['min_sampling_elapse'])),
            self.area_overflow.setValue(float(section['area_overflow'])),
            self.area_diff.setValue(float(section['area_diff'])),
            self.pos_sampling_count.setValue(int(section['pos_sampling_count']))
        self._config_path = Path(path)

    @Slot()
    def save_config(self):
        if self._config_path is not None:
            path = self._config_path
        elif self._wkdir is not None:
            path = self._wkdir
            if self._img_path is not None:
                path = path / self._img_path.with_suffix('.ini').name
        elif self._img_path is not None:
            path = self._img_path.with_suffix('.ini')
        else:
            path = os.path.expanduser('~')
        path = QFileDialog.getSaveFileName(self, 'Save configuration', str(path), 'INI files (*.ini)')[0]
        if not path:
            return
        print('Save config')
        config = configparser.ConfigParser()
        paths = {}
        if self._wkdir is not None:
            paths['wkdir'] = str(self._wkdir)
        if self._img_path is not None:
            paths['image'] = str(self._img_path)
        if self._table_paths is not None:
            paths['detection'] = str(self._table_paths[0].parent)
        if self._tracks_path is not None:
            paths['tracks'] = str(self._tracks_path)
        if self._plot_paths is not None:
            paths['recording'] = str(self._plot_paths[0].parent)
        config['paths.open'] = paths
        paths = {}
        if self.save_dir_detection.text():
            paths['detection'] = self.save_dir_detection.text()
        if self.save_path_tracking.text():
            paths['tracks'] = self.save_path_tracking.text()
        if self.save_dir_recording.text():
            paths['recording'] = self.save_dir_recording.text()
        config['paths.save'] = paths
        config['parameters.detection'] = {
            'gfp_channel': self.gfp_channel.value(),
            'bf_channel': self.bf_channel.value(),
            'block_size': self.block_size.value(),
            'tolerance': self.tolerance.value(),
            'cutoff_ratio': self.cutoff_ratio.value(),
            'bg_thr': self.bg_thr.value(),
            'active_contour': self.active_contour.isChecked(),
            'shift_x': self.shift_x.value(),
            'shift_y': self.shift_y.value(),
            'dog_sigma': self.dog_sigma.value(),
            'sobel_sigma': self.sobel_sigma.value(),
            'bf_weight': self.bf_weight.value(),
            'gfp_weight': self.gfp_weight.value(),
            'dilation_radius': self.dilation_radius.value()
        }
        config['parameters.tracking'] = {
            'dist_thr': self.dist_thr.value(),
            'nn': self.nn.value(),
            'time_gap': self.time_gap.value(),
            'min_sampling_count': self.min_sampling_count.value(),
            'min_sampling_elapse': self.min_sampling_elapse.value(),
            'area_overflow': self.area_overflow.value(),
            'area_diff': self.area_diff.value(),
            'pos_sampling_count': self.pos_sampling_count.value()
        }
        with open(path, 'w', encoding='utf-8') as f:
            config.write(f)

    def on_visualize_tracks(self):
        if self._tracks is None:
            QMessageBox.critical(self, 'Error', 'No track imported.')
            return
        if self._table_paths is None:
            QMessageBox.critical(self, 'Error', 'No detection results imported.')
            return
        print('Perform track visualization')
        progress = QProgressDialog('Visualizing the tracks..', 'Cancel', 0, 0, self)
        progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        progress.setCancelButton(None)
        progress.setWindowTitle('Processing')
        progress.setModal(True)
        progress.show()
        task = WalkRunner(self._tracks, self._table_paths)
        task.finished.connect(progress.close)
        progress.canceled.connect(task.quit)
        task.start()
        progress.exec()
        if task.isFinished():
            self.play_bar.setMaximum(len(task.timescale) - 1)
            self.play_bar.setMinimum(0)
            self.play_bar.setValue(0)
            self.tracking_result.layout().new(task.timescale, task.walks)

    @Slot(int)
    def slider_changed(self, value):
        prop = self.tracking_result.layout()
        if prop.ani is None:
            self.statusBar().showMessage(f'No track visualized.')
            return
        else:
            self.statusBar().showMessage(f'Timestamp: {value}')
        prop.ani.frame_seq = prop.ani.new_frame_seq()
        prop.ani.event_source.stop()
        prop.update_plot(value)
        prop.canvas.draw_idle()

    def on_erase_image(self):
        self.label_image.setText('')
        self.group_input.setEnabled(False)
        self.group_output.setEnabled(False)
        self.detection_input.layout().reset()
        self.detection_output.layout().reset()
        self._img_path = None

    def on_erase_detection(self):
        self.label_detection.setText('')
        self._table_paths = None
        self._mask_paths = None

    def on_erase_tracks(self):
        self.label_tracks.setText('')
        self.group_tracks.setEnabled(False)
        self.tracing_result.layout().reset()
        self._tracks = None
        self._tracks_path = None

    def on_erase_recording(self):
        self.label_recording.setText('')
        self.group_plot.setEnabled(False)
        self.group_video.setEnabled(False)
        self.growth_plot.layout().reset()
        self.growth_video.layout().reset()
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
        self.detection_input.layout().ax.imshow(img, interpolation='none', cmap='gray')
        self.detection_input.layout().canvas.draw()

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
        if task.isFinished():
            self.detection_output.layout().ax.imshow(task.img, interpolation='none')
            self.detection_output.layout().canvas.draw()

    @Slot()
    def task_detection(self):
        if self._img_path is None:
            QMessageBox.critical(self, 'Error', 'No image imported.')
            return
        p = self.save_dir_detection.text()
        if not p:
            QMessageBox.critical(self, 'Error', 'No save directory specified.')
            return
        print('Perform detection')
        p = Path(p)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        self.setEnabled(False)
        t, c, y, x = get_czi_shape(self._img_path)
        self.detection_progress.setMaximum(t)
        self.detection_progress.setValue(0)
        self.task = DetectionTask(self.nproc_detection.value(), t, self._img_path,
                                  self.gfp_channel.value(), self.bf_channel.value(), p,
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
        print('Detection task done.')
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
        p = self.save_path_tracking.text()
        if not p:
            QMessageBox.critical(self, 'Error', 'No save path specified.')
            return
        print('Perform detection')
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.setEnabled(False)
        self.tracking_progress.setMaximum(len(self._table_paths))
        self.tracking_progress.setValue(0)
        self.task = TrackingTask(self._table_paths, p, self.dist_thr.value(), self.nn.value(),
                                 self.time_gap.value(), self.min_sampling_count.value(),
                                 self.min_sampling_elapse.value(), self.area_overflow.value(), self.area_diff.value(),
                                 self.pos_sampling_count.value())
        self.task.increment.connect(self.increment_tracking_progress)
        self.task.finished.connect(self.after_tracking)
        self.task.start()

    @Slot()
    def increment_tracking_progress(self):
        self.tracking_progress.setValue(self.tracking_progress.value() + 1)

    @Slot()
    def after_tracking(self):
        print('Tracking task done.')
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
        p = self.save_dir_recording.text()
        if not p:
            QMessageBox.critical(self, 'Error', 'No save directory specified.')
            return
        print('Perform recording')
        p = Path(p)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        self.setEnabled(False)
        self.recording_progress.setMaximum(len(self._tracks))
        self.recording_progress.setValue(0)
        self.task = RecordingTask(self.nproc_recording.value(), self._tracks, p,
                                  self._img_path, self._table_paths, self._mask_paths, 50, 25.)
        self.task.increment.connect(self.increment_recording_progress)
        self.task.finished.connect(self.after_recording)
        self.task.start()

    @Slot()
    def increment_recording_progress(self):
        self.recording_progress.setValue(self.recording_progress.value() + 1)

    @Slot()
    def after_recording(self):
        print('Recording task done.')
        self.task = None
        self.find_recording(self.save_dir_recording.text())
        self.setEnabled(True)

    def _init_detection_dir(self):
        path = self.save_dir_detection.text()
        if not path:
            if self._wkdir is not None:
                path = self._wkdir
            elif self._img_path is not None:
                path = self._img_path.parent / 'detection'
            elif self._config_path is not None:
                path = self._config_path.parent / 'detection'
            else:
                path = os.path.expanduser('~')
        return str(path)

    @Slot()
    def set_save_dir_detection(self):
        path = QFileDialog.getExistingDirectory(self, 'Select a directory for saving detection results',
                                                self._init_detection_dir())
        if path:
            self.save_dir_detection.setText(path)

    def _init_tracking_path(self):
        path = self.save_path_tracking.text()
        if not path:
            if self._wkdir is not None:
                path = self._wkdir
            elif self._img_path is not None:
                path = self._img_path.parent / 'tracks.pkl'
            elif self._config_path is not None:
                path = self._config_path.parent / 'tracks.pkl'
            else:
                path = os.path.expanduser('~')
        return str(path)

    @Slot()
    def set_save_path_tracking(self):
        path = QFileDialog.getSaveFileName(self, 'Specify a file path to save tracking results',
                                           self._init_tracking_path(), 'Python Pickle File (*.pkl)')[0]
        if path:
            self.save_path_tracking.setText(path)

    def _init_recording_dir(self):
        path = self.save_path_recording.text()
        if not path:
            if self._wkdir is not None:
                path = self._wkdir
            elif self._img_path is not None:
                path = self._img_path.parent / 'recording'
            elif self._config_path is not None:
                path = self._config_path.parent / 'recording'
            else:
                path = os.path.expanduser('~')
        return str(path)

    @Slot()
    def set_save_dir_recording(self):
        path = QFileDialog.getExistingDirectory(self, 'Select a directory for saving recording files',
                                                self._init_recording_dir())
        if path:
            self.save_path_recording.setText(path)

    @Slot()
    def open_wkdir(self):
        if self.set_wkdir():
            self.resolve_detection(self.save_dir_detection.text())
            self.load_tracking(self.save_path_tracking.text())
            self.find_recording(self.save_dir_recording.text())

    @Slot()
    def set_wkdir(self):
        if self._wkdir is not None:
            path = self._wkdir
        elif self._img_path is not None:
            path = self._img_path.parent
        elif self._config_path is not None:
            path = self._config_path.parent
        else:
            path = os.path.expanduser('~')
        path = QFileDialog.getExistingDirectory(self, 'Set working directory', str(path))
        if not path:
            return False
        path = Path(path)
        self._wkdir = path
        p1 = path / 'detection'
        p2 = path / 'recording'
        self.save_dir_detection.setText(str(p1))
        self.save_path_tracking.setText(str(path / 'tracks.pkl'))
        self.save_dir_recording.setText(str(p2))
        return True

    @Slot()
    def on_import_image(self):
        if self._img_path is not None:
            path = self._img_path
        elif self._wkdir is not None:
            path = self._wkdir
        elif self._config_path is not None:
            path = self._config_path.parent
        else:
            path = os.path.expanduser('~')
        path = QFileDialog.getOpenFileName(self, 'Import a raw image stack', str(path),
                                           filter='Carl Zeiss Images (*.czi)')[0]
        if path:
            self.import_image(path)

    def import_image(self, path):
        self._img_path = Path(path)
        if self._img_path.suffix == '.czi':
            t, c, y, x = get_czi_shape(self._img_path)
            self.gfp_channel.setMaximum(c - 1)
            self.gfp_channel.setValue(0)
            self.bf_channel.setMaximum(c - 1)
            self.bf_channel.setValue(1)
            self.raw_scroll.setMaximum(t - 1)
            msg = f'{self._img_path} (T: {t}, C: {c}, Y: {y}, X: {x})'
            print(msg)
            self.label_image.setText(msg)
        self.raw_scroll.setValue(0)
        self.group_input.setEnabled(True)
        self.group_output.setEnabled(True)
        self.detection_update_raw()

    @Slot()
    def on_open_detection(self):
        path = QFileDialog.getExistingDirectory(self, 'Load detections results', self._init_detection_dir())
        if path:
            self.resolve_detection(path)

    def resolve_detection(self, path):
        try:
            print('Loading detection results', path)
            path = Path(path)
            df = sorted(path.glob('*.csv'), key= lambda p: int(p.stem))
            npz = sorted(path.glob('*.npz'), key= lambda p: int(p.stem))
            assert len(df) == len(npz), "No. of tables and No. of masks fail to match"
            assert len(df) > 0, "Couldn't find anything to load"
            self._table_paths = df
            self._mask_paths = npz
            msg = f'{path} ({len(df)})'
            self.label_detection.setText(msg)
            msg = f'Found {len(df)} detection instances.'
            print(msg)
            QMessageBox.information(self, 'Success', msg)
        except:
            msg = f'Detection results resolving failed.\n{path}'
            print(msg)
            QMessageBox.warning(self, 'Warning', msg)

    @Slot()
    def on_open_tracking(self):
        path = QFileDialog.getOpenFileName(self, 'Load tracking results', self._init_tracking_path(),
                                           'Python Pickle File (*.pkl)')[0]
        if path:
            self.load_tracking(path)

    def load_tracking(self, path):
        try:
            print('Loading tracking results', path)
            with open(path, 'rb') as f:
                self._tracks = pickle.load(f)
            msg = f'{path} ({len(self._tracks)})'
            self._tracks_path = path
            self.label_tracks.setText(msg)
            self.group_tracks.setEnabled(True)
            msg = f'Tracking results loaded.'
            print(msg)
            QMessageBox.information(self, 'Success', msg)
        except:
            msg = f'Tracking results loading failed.\n{path}'
            print(msg)
            QMessageBox.warning(self, 'Warning', msg)

    @Slot()
    def on_open_recording(self):
        path = QFileDialog.getExistingDirectory(self, 'Load recording files', self._init_recording_dir())
        if path:
            self.find_recording(path)

    def find_recording(self, path):
        try:
            path = Path(path)
            plots = sorted(path.glob('*.mp4'), key= lambda p: int(p.stem))
            videos = sorted(path.glob('*.avi'), key= lambda p: int(p.stem))
            assert len(plots) == len(videos), "No. of plots and No. of videos fail to match"
            assert len(plots) > 0, "Couldn't find anything to load"
            self._plot_paths = plots
            self._video_paths = videos
            msg = f'{path} ({len(plots)})'
            self.label_recording.setText(msg)
            self.group_plot.setEnabled(True)
            self.group_video.setEnabled(True)
            msg = f'Found {len(plots)} recording instances.'
            print(msg)
            QMessageBox.information(self, 'Success', msg)
        except:
            msg = f'Recording file resolving failed.\n{path}'
            print(msg)
            QMessageBox.warning(self, 'Warning', msg)

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
