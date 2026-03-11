# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

import os
import sys
import traceback

# Plugin-local libraries (optional): qAttitude/lib
LIB = os.path.join(os.path.dirname(__file__), "lib")
if os.path.isdir(LIB) and LIB not in sys.path:
    sys.path.insert(0, LIB)

import numpy as np
import pandas as pd

from qgis.PyQt.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QSpinBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QRadioButton,
    QMessageBox,
    QPlainTextEdit,
    QButtonGroup,
)

from qgis.core import QgsProject, QgsVectorLayer, QgsMimeDataUtils, QgsMapLayer

from qgis.PyQt.QtCore import pyqtSignal, Qt

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mplstereonet
from mplstereonet import stereonet_math

from . import stereo_gis_plot
from .stereo_gis_analysis import (
    read_orientations_from_layer_selection,
    vmf_mean_axial,
    bingham_principal_axes_axial,
    kmedoids_axial,
    kmeans,
    lmn_to_trend_plunge,
    wrap360,
    dipdir2strike,
)


class LayerDropGroupBox(QGroupBox):
    layerDropped = pyqtSignal(QgsMapLayer)

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if QgsMimeDataUtils.isUriList(event.mimeData()):
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if QgsMimeDataUtils.isUriList(event.mimeData()):
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not QgsMimeDataUtils.isUriList(event.mimeData()):
            event.ignore()
            return

        uris = QgsMimeDataUtils.decodeUriList(event.mimeData())
        for uri in uris:
            layer_id_attr = getattr(uri, "layerId", None)
            layer_id = layer_id_attr() if callable(layer_id_attr) else layer_id_attr
            if not layer_id:
                continue

            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is not None and layer.type() == QgsMapLayer.VectorLayer:
                self.layerDropped.emit(layer)
                event.setDropAction(Qt.CopyAction)
                event.accept()
                return

        event.ignore()


class StereoGisDialog(QDialog):
    def __init__(self, iface, plugin):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.plugin = plugin

        self.analysis_layer = None
        self._layer_ids_by_index = []  # keeps combo index -> layer.id()

        self.setWindowTitle("qAttitude")
        self.resize(1100, 750)

        self._picked_medoid_indices = []
        self._picking_enabled = False
        self._last_projected = None

        self._init_ui()
        self._plot_empty()

    def showEvent(self, event):
        super().showEvent(event)
        # On first show, if no layer is set, try to use the active one
        if self.analysis_layer is None:
            active_layer = self.iface.activeLayer()
            if active_layer and active_layer.type() == QgsMapLayer.VectorLayer:
                self.set_analysis_layer(active_layer)

        self._populate_layers()
        self._select_layer_in_combo(self.analysis_layer)
        self._refresh_fields()
        self._refresh_field_controls()
        self._load_data_and_plot()

    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()
        right = QVBoxLayout()
        main.addLayout(left, 0)
        main.addLayout(right, 1)

        # Inputs
        g_in = LayerDropGroupBox("Input layer (all or selected features)", self)
        g_in.layerDropped.connect(self.on_layer_dropped)
        left.addWidget(g_in)
        grid = QGridLayout(g_in)

        grid.addWidget(QLabel("Layer:"), 0, 0)
        self.layer_combo = QComboBox()
        grid.addWidget(self.layer_combo, 0, 1, 1, 3)

        self.field1_label = QLabel("Dip field:")
        self.field2_label = QLabel("DipDir field:")
        self.field1_combo = QComboBox()
        self.field2_combo = QComboBox()
        grid.addWidget(self.field1_label, 1, 0)
        grid.addWidget(self.field1_combo, 1, 1, 1, 3)
        grid.addWidget(self.field2_label, 2, 0)
        grid.addWidget(self.field2_combo, 2, 1, 1, 3)

        self.data_planes = QRadioButton("Planes (dip/dipdir)")
        self.data_lines = QRadioButton("Lines (plunge/trend)")
        self.data_planes.setChecked(True)
        grid.addWidget(self.data_planes, 3, 0, 1, 2)
        grid.addWidget(self.data_lines, 3, 2, 1, 2)

        self.analysis_axial = QRadioButton("Axial/Bidirectional")
        self.analysis_polar = QRadioButton("Polar/Unidirectional")
        self.analysis_axial.setChecked(True)
        grid.addWidget(self.analysis_axial, 4, 0, 1, 2)
        grid.addWidget(self.analysis_polar, 4, 2, 1, 2)

        self.data_type_group = QButtonGroup(self)
        self.data_type_group.addButton(self.data_planes)
        self.data_type_group.addButton(self.data_lines)

        self.analysis_type_group = QButtonGroup(self)
        self.analysis_type_group.addButton(self.analysis_axial)
        self.analysis_type_group.addButton(self.analysis_polar)

        self.layer_combo.currentIndexChanged.connect(self.on_layer_combo_changed)
        self.data_planes.toggled.connect(self.on_data_type_changed)
        self.data_lines.toggled.connect(self.on_data_type_changed)
        self.field1_combo.currentIndexChanged.connect(self._load_data_and_plot)
        self.field2_combo.currentIndexChanged.connect(self._load_data_and_plot)
        self.analysis_axial.toggled.connect(self._load_data_and_plot)
        self.analysis_polar.toggled.connect(self._load_data_and_plot)

        # K-medoids
        g_km = QGroupBox("K-means clustering")
        left.addWidget(g_km)
        gridk = QGridLayout(g_km)

        gridk.addWidget(QLabel("Number of clusters:"), 0, 0)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 999999)
        self.k_spin.setValue(1)
        gridk.addWidget(self.k_spin, 0, 1)

        self.init_random = QRadioButton("Automatic seeds")
        self.init_pick = QRadioButton("Pick seeds on plot")
        self.init_random.setChecked(True)
        gridk.addWidget(self.init_random, 1, 0)
        gridk.addWidget(self.init_pick, 1, 1)

        gridk.addWidget(QLabel("Random seed:"), 2, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 10**9)
        self.seed_spin.setValue(0)
        gridk.addWidget(self.seed_spin, 2, 1)

        self.btn_pick = QPushButton("Pick seeds")
        self.btn_clear_picks = QPushButton("Clear picks")
        self.lbl_picks = QLabel("Picked: 0")
        gridk.addWidget(self.btn_pick, 3, 0)
        gridk.addWidget(self.btn_clear_picks, 3, 1)
        gridk.addWidget(self.lbl_picks, 4, 0, 1, 2)

        self.btn_pick.clicked.connect(self._toggle_picking)
        self.btn_clear_picks.clicked.connect(self._clear_picks)

        # Plot options and Parametric distribution fitting
        g_plot = QGroupBox("Plot options")
        left.addWidget(g_plot)
        gridp = QGridLayout(g_plot)

        self.chk_individual = QCheckBox("Plot data (mode for planes)")
        self.chk_individual.setChecked(True)
        gridp.addWidget(self.chk_individual, 0, 0)

        self.plane_mode_combo = QComboBox()
        self.plane_mode_combo.addItems(["Poles", "Great circles", "Both"])
        gridp.addWidget(self.plane_mode_combo, 0, 1)

        self.chk_contours = QCheckBox("Plot contours (levels)")
        self.chk_contours.setChecked(False)
        self.contour_levels = QSpinBox()
        self.contour_levels.setRange(3, 30)
        self.contour_levels.setValue(10)

        gridp.addWidget(self.chk_contours, 1, 0)
        gridp.addWidget(self.contour_levels, 1, 1)

        self.chk_plot_clusters = QCheckBox("Plot k-means clusters")
        self.chk_plot_clusters.setChecked(False)
        gridp.addWidget(self.chk_plot_clusters, 2, 0, 1, 2)

        self.chk_vmf = QCheckBox("Plot Von Mises-Fisher mean (red)")
        self.chk_vmf.setChecked(False)
        self.chk_kent = QCheckBox("Plot Kent mean (green)")
        self.chk_kent.setChecked(False)
        self.chk_bingham = QCheckBox("Plot Bingham β axis & girdle (blue)")
        self.chk_bingham.setChecked(False)

        gridp.addWidget(self.chk_vmf, 3, 0, 1, 2)
        gridp.addWidget(self.chk_kent, 4, 0, 1, 2)
        gridp.addWidget(self.chk_bingham, 5, 0, 1, 2)

        gridp.addWidget(QLabel("Point/GCs Color:"), 6, 0)
        self.point_color_combo = QComboBox()
        self.point_color_combo.addItems(
            [
                "black",
                "red",
                "blue",
                "green",
                "cyan",
                "magenta",
                "yellow",
                "purple",
                "brown",
                "lightgreen",
                "olive",
            ]
        )
        gridp.addWidget(self.point_color_combo, 6, 1)

        gridp.addWidget(QLabel("Contour Color Map:"), 7, 0)
        self.contour_cmap_combo = QComboBox()
        cmaps = [
            "Greys",
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
        ]
        self.contour_cmap_combo.addItems(cmaps)
        gridp.addWidget(self.contour_cmap_combo, 7, 1)

        self.chk_individual.stateChanged.connect(self._update_plot)
        self.plane_mode_combo.currentIndexChanged.connect(self._update_plot)
        self.chk_contours.stateChanged.connect(self._update_plot)
        self.contour_levels.valueChanged.connect(self._update_plot)
        self.chk_plot_clusters.stateChanged.connect(self._update_plot)
        self.k_spin.valueChanged.connect(self._update_plot)
        self.chk_vmf.stateChanged.connect(self._update_plot)
        self.chk_bingham.stateChanged.connect(self._update_plot)
        self.point_color_combo.currentIndexChanged.connect(self._update_plot)
        self.contour_cmap_combo.currentIndexChanged.connect(self._update_plot)

        # Saving
        g_save = QGroupBox("Save to files (off by default)")
        left.addWidget(g_save)
        grids = QGridLayout(g_save)

        self.chk_save = QCheckBox("Save outputs")
        self.chk_save.setChecked(False)
        grids.addWidget(self.chk_save, 0, 0, 1, 2)

        grids.addWidget(QLabel("Directory:"), 1, 0)
        self.out_dir = QLineEdit("")
        self.btn_browse = QPushButton("Browse…")
        grids.addWidget(self.out_dir, 1, 1)
        grids.addWidget(self.btn_browse, 1, 2)
        self.btn_browse.clicked.connect(self._browse_dir)

        left.addStretch(1)

        # Plot
        self.fig = Figure(figsize=(6, 6), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(500)
        right.addWidget(self.canvas, 5)

        self.ax = self.fig.add_subplot(111, projection="stereonet")
        self.ax.grid(True)
        self.ax.grid(kind="polar")
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        g_log = QGroupBox("Log")
        right.addWidget(g_log, 2)
        gridlog = QGridLayout(g_log)

        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Analysis log messages will appear here.")
        self.log_output.setMinimumHeight(180)
        gridlog.addWidget(self.log_output, 0, 0, 1, 2)

        self.btn_clear_log = QPushButton("Clear log")
        self.btn_clear_log.clicked.connect(self.clear_log)
        gridlog.addWidget(self.btn_clear_log, 1, 1)

    def on_layer_dropped(self, layer):
        self.set_analysis_layer(layer)
        self._populate_layers()
        self._select_layer_in_combo(layer)
        self._refresh_fields()
        self._refresh_field_controls()
        self._load_data_and_plot()

    def on_layer_combo_changed(self):
        layer = self._current_layer_from_combo()
        self.set_analysis_layer(layer)
        self._refresh_fields()
        self._refresh_field_controls()
        self._load_data_and_plot()

    def on_data_type_changed(self, checked):
        if not checked:
            return
        self._refresh_field_controls()
        self._load_data_and_plot()

    def _populate_layers(self):
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self._layer_ids_by_index = []

        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() != QgsMapLayer.VectorLayer:
                continue
            self.layer_combo.addItem(layer.name())
            self._layer_ids_by_index.append(layer.id())

        self.layer_combo.blockSignals(False)

    def _select_layer_in_combo(self, layer: QgsMapLayer) -> None:
        if layer is None:
            return
        try:
            idx = self._layer_ids_by_index.index(layer.id())
            self.layer_combo.blockSignals(True)
            self.layer_combo.setCurrentIndex(idx)
            self.layer_combo.blockSignals(False)
        except ValueError:
            pass

    def set_analysis_layer(self, layer: QgsMapLayer):
        if self.analysis_layer:
            try:
                self.analysis_layer.selectionChanged.disconnect(
                    self._load_data_and_plot
                )
            except TypeError:
                pass

        if layer and layer.type() == QgsMapLayer.VectorLayer:
            self.analysis_layer = layer
            self.analysis_layer.selectionChanged.connect(self._load_data_and_plot)
            self.append_log(f"Input layer set to: {layer.name()}")
        else:
            self.analysis_layer = None

    def _current_layer_from_combo(self) -> QgsMapLayer | None:
        if self.layer_combo.currentIndex() >= 0:
            idx = self.layer_combo.currentIndex()
            if 0 <= idx < len(self._layer_ids_by_index):
                layer_id = self._layer_ids_by_index[idx]
                return QgsProject.instance().mapLayer(layer_id)
        return None

    def _refresh_fields(self):
        layer = self.analysis_layer

        self.field1_combo.blockSignals(True)
        self.field2_combo.blockSignals(True)
        self.field1_combo.clear()
        self.field2_combo.clear()

        if not layer:
            self.field1_combo.blockSignals(False)
            self.field2_combo.blockSignals(False)
            return

        preferred1 = [
            "dip",
            "Dip",
            "DIP",
            "plunge",
            "Plunge",
            "PLUNGE",
            "inc",
            "Inc",
            "INC",
        ]
        preferred2 = [
            "dir",
            "Dir",
            "DIR",
            "dipdir",
            "DipDir",
            "Dipdir",
            "DIPDIR",
            "plunge",
            "Plunge",
            "PLUNGE",
            "imm",
            "Imm",
            "IMM",
        ]
        all_fields = [f.name() for f in layer.fields()]

        self.field1_combo.addItems(all_fields)
        self.field2_combo.addItems(all_fields)

        for col in preferred1:
            if col in all_fields:
                self.field1_combo.setCurrentIndex(all_fields.index(col))
                break
        for col in preferred2:
            if col in all_fields:
                self.field2_combo.setCurrentIndex(all_fields.index(col))
                break

        self.field1_combo.blockSignals(False)
        self.field2_combo.blockSignals(False)

    def _refresh_field_controls(self):
        is_planes = self.data_planes.isChecked()
        self.field1_label.setText("Dip field:" if is_planes else "Plunge field:")
        self.field2_label.setText("DipDir field:" if is_planes else "Trend field:")
        self.plane_mode_combo.setEnabled(is_planes)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory", "")
        if d:
            self.out_dir.setText(d)

    def _toggle_picking(self):
        if not self.init_pick.isChecked():
            QMessageBox.information(
                self, "Pick medoid seeds", "Select 'Pick seeds on plot' first."
            )
            return
        self._picking_enabled = not self._picking_enabled
        self.btn_pick.setText(
            "Picking: ON" if self._picking_enabled else "Pick medoid seeds"
        )

    def _clear_picks(self):
        self._picked_medoid_indices = []
        self.lbl_picks.setText("Picked: 0")

    def _on_plot_click(self, event):
        if (
            not self._picking_enabled
            or event.inaxes != self.ax
            or self._last_projected is None
        ):
            return

        k = int(self.k_spin.value())
        if len(self._picked_medoid_indices) >= k:
            return

        x, y = float(event.xdata), float(event.ydata)
        P = self._last_projected
        d2 = (P[:, 0] - x) ** 2 + (P[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        if idx in self._picked_medoid_indices:
            return

        self._picked_medoid_indices.append(idx)
        self.lbl_picks.setText(f"Picked: {len(self._picked_medoid_indices)} / {k}")
        if len(self._picked_medoid_indices) == k:
            self._picking_enabled = False
            self.btn_pick.setText("Pick medoid seeds")

    def _plot_empty(self):
        try:
            self.ax.clear()
            self.ax.grid(True, zorder=0, alpha=0.5)
            self.ax.grid(kind="polar", zorder=0, alpha=0.5)
            self.canvas.draw()
        except Exception:
            pass

    def append_log(self, message: str):
        self.log_output.appendPlainText(str(message))

    def clear_log(self):
        self.log_output.clear()

    def _load_data_and_plot(self):
        sender = self.sender()
        if isinstance(sender, QRadioButton) and not sender.isChecked():
            return

        layer = self.analysis_layer
        if (
            not layer
            or not self.field1_combo.currentText()
            or not self.field2_combo.currentText()
        ):
            self.plugin.data = pd.DataFrame()
            self._update_plot()
            return

        is_planes = self.data_planes.isChecked()
        field1 = self.field1_combo.currentText()
        field2 = self.field2_combo.currentText()
        analysis_type = "axial" if self.analysis_axial.isChecked() else "polar"

        try:
            self.plugin.data = read_orientations_from_layer_selection(
                layer,
                field1,
                field2,
                is_planes=is_planes,
                analysis_type=analysis_type,
                log=self.append_log,
            )
        except Exception as e:
            self.plugin.data = pd.DataFrame()
            tb = traceback.format_exc()
            self.append_log(f"ERROR: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self, "qAttitude", f"Error: {type(e).__name__}: {e}\n\n{tb}"
            )

        self._update_plot()

    def _update_plot(self):
        if self.plugin.data.empty:
            self._plot_empty()
            return

        is_planes = self.data_planes.isChecked()
        show_individual = self.chk_individual.isChecked()
        show_contours = self.chk_contours.isChecked()
        plane_mode = self.plane_mode_combo.currentIndex()
        plot_poles = not is_planes or plane_mode in (0, 2)
        plot_gcs = is_planes and plane_mode in (1, 2)

        try:
            self._last_projected = stereo_gis_plot.plot_base_stereonet(
                self.ax,
                self.plugin.data,
                plot_poles=show_individual and plot_poles,
                plot_gcs=show_individual and plot_gcs,
                show_contours=show_contours,
                contour_levels=self.contour_levels.value(),
                point_color=self.point_color_combo.currentText(),
                contour_cmap=self.contour_cmap_combo.currentText(),
            )

            if self.chk_plot_clusters.isChecked():
                self._plot_clusters()
            if self.chk_vmf.isChecked():
                self._plot_vmf()
            if self.chk_bingham.isChecked():
                self._plot_bingham()

            self.canvas.draw()
            self.append_log("Plot updated.")
        except Exception as e:
            tb = traceback.format_exc()
            self.append_log(f"ERROR: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self, "qAttitude", f"Error: {type(e).__name__}: {e}\n\n{tb}"
            )

    def _plot_clusters(self):
        k = int(self.k_spin.value())

        # Determine the number of original data points
        analysis_type = "axial" if self.analysis_axial.isChecked() else "polar"
        if analysis_type == "axial":
            n_orig = self.plugin.data["lower_hemi"].sum()
        else:
            n_orig = self.plugin.data.shape[0]

        if n_orig == 0:
            self.append_log("No data to cluster.")
            return

        if k > n_orig:
            self.append_log(
                f"Number of clusters ({k}) is greater than the number of data points ({n_orig}). "
                f"Reducing number of clusters to {n_orig}."
            )
            k = n_orig
            self.k_spin.blockSignals(True)
            self.k_spin.setValue(k)
            self.k_spin.blockSignals(False)

        data_with_clusters, means = kmeans(
            self.plugin.data.copy(),
            n_clusters=k,
            analysis_type=analysis_type,
            init="k-means++",
            random_state=self.seed_spin.value(),
            log=self.append_log,
        )

        is_planes = self.data_planes.isChecked()
        plane_mode = self.plane_mode_combo.currentIndex()
        plot_poles = not is_planes or plane_mode in (0, 2)
        plot_gcs = is_planes and plane_mode in (1, 2)

        stereo_gis_plot.plot_clusters(
            self.ax, data_with_clusters, means, plot_poles, plot_gcs
        )
        self.append_log("k-means clustering completed.")

    def _plot_vmf(self):
        vmf = vmf_mean_axial(self.plugin.data, log=self.append_log)
        m = vmf["mean_xyz"]
        if np.isfinite(m).all():
            tr, pl = stereo_gis_plot.plot_vmf(self.ax, m)
            self.append_log(
                f"VMF mean plotted at trend={tr:.2f}, plunge={pl:.2f} in red."
            )

    def _plot_bingham(self):
        b = bingham_principal_axes_axial(self.plugin.data, log=self.append_log)
        beta = b["beta_axis_xyz"]
        tr, pl = stereo_gis_plot.plot_bingham(self.ax, beta)
        self.append_log(
            f"Bingham beta axis plotted at trend={tr:.2f}, plunge={pl:.2f}."
        )