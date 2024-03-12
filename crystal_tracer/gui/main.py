import os
import sys
import pickle
import shutil
import configparser
from pathlib import Path
import numpy as np
from PySide6.QtCore import Slot, QThread, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QButtonGroup, QFileDialog, QMessageBox, QProgressDialog, \
    QInputDialog
from PySide6.QtGui import QStandardItemModel, QStandardItem
from crystal_tracer.gui.ui_loader import loadUi
from crystal_tracer.img_utils import load_czi_slice, get_czi_shape
from crystal_tracer.gui.workers import PreviewFilterTask, DetectionTask, TrackingTask, \
    RecordingTask, WalkTask, TrackingTask2, PreviewRecordingTask
from crystal_tracer.gui.components import FigureComponent, AnimatedLines3D, AnimatedLine2D, MultiPage


class CrystalTracerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # data
        self._wkdir: Path | None = None
        self._config_path: Path | None = None
        self._img_path: Path | None = None
        self._tracks_path: Path | None = None
        self._table_paths: list[Path] | None = None
        self._mask_paths: list[Path] | None = None
        self._tracks: list[list[tuple[int, int]]] | None = None
        self.task: QThread | None = None
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
        self.action_open_wkdir.triggered.connect(self.open_wkdir)
        self.action_open.triggered.connect(self.open_sth)
        self.action_load_config.triggered.connect(self.load_config)
        self.action_save_config.triggered.connect(self.save_config)
        self.addAction(self.action_open)
        self.action_set_wkdir.triggered.connect(self.set_wkdir)
        self.action_batch.triggered.connect(self.run_batch)

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
        self.tracking_result.setLayout(AnimatedLines3D())
        self.growth_plot.setLayout(AnimatedLine2D())
        self.growth_video.setLayout(MultiPage())

        # listview
        self.list_track_model = QStandardItemModel()
        self.list_track.setModel(self.list_track_model)
        self.list_track.clicked.connect(self.preview_recording)

        # misc
        self.radio_gfp.toggled.connect(self.detection_update_raw)
        self.preview_filter.clicked.connect(self.on_preview_detection)
        self.visualize_tracks.clicked.connect(self.on_visualize_tracks)
        self.erase_image.clicked.connect(self.on_erase_image)
        self.erase_detection.clicked.connect(self.on_erase_detection)
        self.erase_tracks.clicked.connect(self.on_erase_tracks)
        self.group_input.setEnabled(False)
        self.group_output.setEnabled(False)
        self.group_tracks.setEnabled(False)
        self.group_plot.setEnabled(False)
        self.group_video.setEnabled(False)
        self.play_bar.valueChanged.connect(self.slider_changed)
        self.play_bar_2.valueChanged.connect(self.slider_changed)

    @Slot(int)
    def preview_recording(self, index):
        if self._img_path is None:
            QMessageBox.critical(self, 'Error', 'No image imported.')
            return
        if self._table_paths is None:
            QMessageBox.critical(self, 'Error', 'No detection results imported.')
            return
        if self._tracks is None:
            QMessageBox.critical(self, 'Error', 'No tracking results specified.')
            return
        print('Perform recording preview')
        item = self.list_track_model.itemFromIndex(index)
        progress = QProgressDialog('Visualizing the selected track..', 'Cancel', 0, 0, self)
        progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowCloseButtonHint)
        progress.setCancelButton(None)
        progress.setWindowTitle('Processing')
        progress.setModal(True)
        progress.show()
        task = PreviewRecordingTask(self._tracks[item.data(Qt.UserRole)], self._img_path, self._table_paths,
                                    self._mask_paths, self.win_rad.value(), self.min_per_frame.value())
        task.finished.connect(progress.close)
        progress.canceled.connect(task.quit)
        task.start()
        progress.exec()
        if task.isFinished():
            self.group_plot.setEnabled(True)
            self.group_video.setEnabled(True)
            self.play_bar_2.setMaximum(len(task.x_data) - 1)
            self.play_bar_2.setMinimum(0)
            self.play_bar_2.setValue(0)
            self.growth_plot.layout().new(task.x_data, task.y_data)
            self.growth_video.layout().new(task.stack)

    def update_listview(self):
        if self._tracks is None:
            self.list_track_model.clear()
        else:
            elapse = [t[-1][0] - t[0][0] for t in self._tracks]
            for i in reversed(np.argsort(elapse)):
                i1, i2 = self._tracks[i][-1]
                item = QStandardItem(f'T_{i1}_C_{i2} ({elapse[i]})')
                item.setData(i , Qt.UserRole)
                self.list_track_model.appendRow(item)

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
        self._load_config(path)

    def _load_config(self, path):
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
            self.area_norm.setValue(float(section['area_norm'])),
            self.nn.setValue(int(section['nn'])),
            self.time_gap.setValue(int(section['time_gap'])),
            self.min_sampling_count.setValue(int(section['min_sampling_count'])),
            self.min_sampling_elapse.setValue(int(section['min_sampling_elapse'])),
            self.area_overflow.setValue(float(section['area_overflow'])),
            self.intensity_overflow.setValue(float(section['intensity_overflow']))
            self.use_contig.setChecked(config.getboolean('parameters.tracking', 'use_contig')),

        if config.has_section('parameters.recording'):
            section = config['parameters.recording']
            self.win_rad.setValue(int(section['win_rad'])),
            self.frame_rate.setValue(float(section['frame_rate'])),
            self.elapse_thr.setValue(int(section['elapse_thr'])),
            self.min_per_frame.setValue(float(section['min_per_frame']))

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
        self._save_config(path)

    def _save_config(self, path):
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
            'area_norm': self.area_norm.value(),
            'nn': self.nn.value(),
            'time_gap': self.time_gap.value(),
            'min_sampling_count': self.min_sampling_count.value(),
            'min_sampling_elapse': self.min_sampling_elapse.value(),
            'area_overflow': self.area_overflow.value(),
            'intensity_overflow': self.intensity_overflow.value(),
            'use_contig': self.use_contig.isChecked()
        }

        config['parameters.recording'] = {
            'win_rad': self.win_rad.value(),
            'frame_rate': self.frame_rate.value(),
            'elapse_thr': self.elapse_thr.value(),
            'min_per_frame': self.min_per_frame.value()
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
        task = WalkTask(self._tracks, self._table_paths)
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
        slider = self.sender()
        if slider is self.play_bar:
            prop = [self.tracking_result.layout()]
        elif slider is self.play_bar_2:
            prop = [self.growth_plot.layout(), self.growth_video.layout()]
        else:
            raise ValueError("Invalid slider")
        if prop[0].ani is None:
            self.statusBar().showMessage(f'No track visualized.')
            return
        else:
            self.statusBar().showMessage(f'Timestamp: {value}')
        for p in prop:
            p.update_plot(value)

    def on_erase_image(self):
        self.label_image.setText('')
        self.group_input.setEnabled(False)
        self.group_output.setEnabled(False)
        self.group_plot.setEnabled(False)
        self.group_video.setEnabled(False)
        self.detection_input.layout().reset()
        self.detection_output.layout().reset()
        self.growth_plot.layout().reset()
        self.growth_video.layout().reset()
        self.update_listview()
        self._img_path = None

    def on_erase_detection(self):
        self.label_detection.setText('')
        self.group_plot.setEnabled(False)
        self.group_video.setEnabled(False)
        self.growth_plot.layout().reset()
        self.growth_video.layout().reset()
        self.update_listview()
        self._table_paths = None
        self._mask_paths = None

    def on_erase_tracks(self):
        self.label_tracks.setText('')
        self.group_tracks.setEnabled(False)
        self.group_plot.setEnabled(False)
        self.group_video.setEnabled(False)
        self.tracing_result.layout().reset()
        self.growth_plot.layout().reset()
        self.growth_video.layout().reset()
        self.update_listview()
        self._tracks = None
        self._tracks_path = None

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
        task = PreviewFilterTask(self._img_path, self.gfp_channel.value(), self.bf_channel.value(), self.raw_scroll.value(),
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
    def task_detection(self, msgbox=True, modal=False):
        if self._img_path is None:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No image imported.')
            return False
        p = self.save_dir_detection.text()
        if not p:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No save directory specified.')
            return False
        print('Perform detection')
        p = Path(p)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        self.setEnabled(False)
        t, c, y, x = get_czi_shape(self._img_path)
        self.task = DetectionTask(self.nproc_detection.value(), t, self._img_path,
                                  self.gfp_channel.value(), self.bf_channel.value(), p,
                                  self.block_size.value(), self.tolerance.value(), self.cutoff_ratio.value(),
                                  self.bg_thr.value(),
                                  self.active_contour.isChecked(), (self.shift_x.value(), self.shift_y.value()),
                                  self.dog_sigma.value(),
                                  self.sobel_sigma.value(), self.bf_weight.value(), self.gfp_weight.value(),
                                  self.dilation_radius.value())
        if modal:
            progress = QProgressDialog('Performing detection..', 'Cancel', 0, 0, self)
            progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowCloseButtonHint)
            progress.setCancelButton(None)
            progress.setWindowTitle('Processing')
            progress.setModal(True)
            progress.show()
            self.task.finished.connect(progress.close)
            progress.canceled.connect(self.task.quit)
            self.task.start()
            progress.exec()
            self.setEnabled(True)
        else:
            self.detection_progress.setMaximum(t)
            self.detection_progress.setValue(0)
            self.task.increment.connect(self.increment_detection_progress)
            self.task.finished.connect(self.after_detection)
            self.task.start()
        return True

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
    def task_tracking(self, msgbox=True, modal=False):
        if self._table_paths is None:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No detection results specified.')
            return False
        p = self.save_path_tracking.text()
        if not p:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No save path specified.')
            return False
        print('Perform tracking')
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.tracking_progress.setMaximum(len(self._table_paths))
        self.tracking_progress.setValue(0)
        if self.track_algorithms.currentIndex() == 0:
            self.task = TrackingTask(self._table_paths, p, self.area_norm.value(), self.nn.value(),
                                     self.time_gap.value(), self.min_sampling_count.value(),
                                     self.min_sampling_elapse.value(), self.area_overflow.value(),
                                     self.intensity_overflow.value())
        else:
            self.task = TrackingTask2(self._table_paths, p, self.area_norm.value(), self.use_contig.value())
        if modal:
            progress = QProgressDialog('Performing tracking..', 'Cancel', 0, 0, self)
            progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowCloseButtonHint)
            progress.setCancelButton(None)
            progress.setWindowTitle('Processing')
            progress.setModal(True)
            progress.show()
            self.task.finished.connect(progress.close)
            progress.canceled.connect(self.task.quit)
            self.task.start()
            progress.exec()
        else:
            self.setEnabled(False)
            self.task.increment.connect(self.increment_tracking_progress)
            self.task.finished.connect(self.after_tracking)
            self.task.start()
        return True

    @Slot()
    def increment_tracking_progress(self):
        self.tracking_progress.setValue(self.tracking_progress.value() + 1)

    @Slot()
    def after_tracking(self):
        print('Tracking task done.')
        self.task = None
        self.load_tracking(self.save_path_tracking.text())
        self.on_visualize_tracks()
        self.setEnabled(True)

    @Slot()
    def task_recording(self, msgbox=True, modal=False):
        if self._img_path is None:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No image imported.')
            return False
        if self._table_paths is None:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No detection results specified.')
            return False
        if self._tracks is None:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No tracking results specified.')
            return False
        p = self.save_dir_recording.text()
        if not p:
            if msgbox:
                QMessageBox.critical(self, 'Error', 'No save directory specified.')
            return False
        print('Perform recording')
        p = Path(p)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        tracks = [t for t in self._tracks if len(t) > self.elapse_thr.value()]
        self.recording_progress.setMaximum(len(tracks))
        self.recording_progress.setValue(0)
        self.task = RecordingTask(self.nproc_recording.value(), tracks, p, self._img_path, self._table_paths,
                                  self._mask_paths, self.win_rad.value(), self.frame_rate.value(),
                                  self.min_per_frame.value())
        if modal:
            progress = QProgressDialog('Performing recording..', 'Cancel', 0, 0, self)
            progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowCloseButtonHint)
            progress.setCancelButton(None)
            progress.setWindowTitle('Processing')
            progress.setModal(True)
            progress.show()
            self.task.finished.connect(progress.close)
            progress.canceled.connect(self.task.quit)
            self.task.start()
            progress.exec()
        else:
            self.setEnabled(False)
            self.task.increment.connect(self.increment_recording_progress)
            self.task.finished.connect(self.after_recording)
            self.task.start()
        return True

    @Slot()
    def increment_recording_progress(self):
        self.recording_progress.setValue(self.recording_progress.value() + 1)

    @Slot()
    def after_recording(self):
        print('Recording task done.')
        self.task = None
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

    def _init_wkdir(self, path):
        path = Path(path)
        self._wkdir = path
        p1 = path / 'detection'
        p2 = path / 'recording'
        self.save_dir_detection.setText(str(p1))
        self.save_path_tracking.setText(str(path / 'tracks.pkl'))
        self.save_dir_recording.setText(str(p2))

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
        self._init_wkdir(path)
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
            df = sorted(path.glob('*.csv'), key=lambda p: int(p.stem))
            npz = sorted(path.glob('*.npz'), key=lambda p: int(p.stem))
            assert len(df) == len(npz), "No. of tables and No. of masks fail to match"
            assert len(df) > 0, "Couldn't find anything to load"
            self._table_paths = df
            self._mask_paths = npz
            msg = f'{path} ({len(df)})'
            self.label_detection.setText(msg)
            msg = f'Found {len(df)} detection instances.'
            print(msg)
            # QMessageBox.information(self, 'Success', msg)
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
            self.update_listview()
            msg = f'Tracking results loaded.'
            print(msg)
            # QMessageBox.information(self, 'Success', msg)
        except:
            msg = f'Tracking results loading failed.\n{path}'
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

    @Slot()
    def run_batch(self):
        if self._config_path is not None:
            path = self._config_path.parent
        elif self._wkdir is not None:
            path = self._wkdir
        elif self._img_path is not None:
            path = self._img_path.parent
        else:
            path = os.path.expanduser('~')
        paths, _ = QFileDialog.getOpenFileNames(self, 'Choose images', path,
                                                'Carl Zeiss Images (*.czi);;INI Files (*.ini)')
        if len(paths) == 0:
            return
        n, ok = QInputDialog.getInt(self, "Input", "Enter the number of processors to use:",
                                                 QThread.idealThreadCount(), 1, QThread.idealThreadCount())
        if not ok:
            return

        response = QMessageBox.question(self, "Continue on existing results?", "Created locations will be skipped.",
                                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

        if response == QMessageBox.Cancel:
            return

        self.nproc_detection.setValue(n)
        self.nproc_recording.setValue(n)

        for p in paths:
            p = Path(p)
            if p.suffix == '.czi':
                self.import_image(p)
                self._init_wkdir(p.parent / p.stem)
            elif p.suffix == '.ini':
                self._load_config(p)
            if response == QMessageBox.Yes and Path(self.save_dir_detection.text()).exists() or \
                    self.task_detection(msgbox=False, modal=True):
                self.resolve_detection(self.save_dir_detection.text())
                if response == QMessageBox.Yes and Path(self.save_path_tracking.text()).exists() or \
                        self.task_tracking(msgbox=False, modal=True):
                    self.load_tracking(self.save_path_tracking.text())
                    if response == QMessageBox.Yes and Path(self.save_dir_recording.text()).exists() or \
                            self.task_recording(msgbox=False, modal=True):
                        self._save_config(self._wkdir / 'config.ini')
                        continue
            print(f'{p} faild.')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CrystalTracerApp()
    window.show()
    with open("Ubuntu.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    sys.exit(app.exec())
