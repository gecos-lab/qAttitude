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
from sklearn.cluster import KMeans

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

from qgis.core import QgsProject, QgsVectorLayer, QgsMimeDataUtils, QgsMapLayer, QgsProcessingException

from qgis.PyQt.QtCore import pyqtSignal, Qt

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mplstereonet
from mplstereonet import stereonet_math


def _log(log, message: str) -> None:
    if log is not None:
        log(message)


def wrap360(deg):
    deg = deg % 360.0
    return deg


def deg2rad(deg):
    return deg * np.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / np.pi


def trend_plunge_to_lmn(trend, plunge):
    trend_rad = deg2rad(trend)
    plunge_rad = deg2rad(plunge)
    l = np.cos(plunge_rad) * np.cos(trend_rad)  # East
    m = np.cos(plunge_rad) * np.sin(trend_rad)  # North
    n = -np.sin(plunge_rad)  # Up (negative is down-plunge)
    return l, m, n


def lmn_to_trend_plunge(l, m, n):
    plunge_rad = np.arcsin(-n)
    trend_rad = np.arctan2(m, l)  # Corrected arctan2 order
    plunge = wrap360(rad2deg(plunge_rad))
    trend = wrap360(rad2deg(trend_rad))
    return trend, plunge


def dipdir_dip_to_pole_lmn(dipdir, dip):
    pole_trend = wrap360(dipdir + 180)
    pole_plunge = 90.0 - dip
    return trend_plunge_to_lmn(pole_trend, pole_plunge)


def dipdir2strike(dipdir):
    strike = wrap360(dipdir - 90.0)
    return strike


def strike2dipdir(strike):
    dipdir = wrap360(strike + 90.0)
    return dipdir


def read_orientations_from_layer_selection(
    layer,
    field1: str,
    field2: str,
    is_planes: bool = True,
    analysis_type: str = "axial",
    log=None,
) -> pd.DataFrame:
    """
    Reads orientations from the layer.
    Uses selected features if any are selected; otherwise uses all features.
    """
    # first read lists of data irrespective of plane vs. line
    idx1 = layer.fields().indexOf(field1)
    idx2 = layer.fields().indexOf(field2)
    if idx1 < 0 or idx2 < 0:
        raise QgsProcessingException("Selected field not found in layer.")

    use_selected = bool(layer.selectedFeatureCount())
    feats = layer.getSelectedFeatures() if use_selected else layer.getFeatures()

    _log(
        log,
        f"Reading orientations from {'selected features' if use_selected else 'all features'} "
        f"using fields '{field1}' and '{field2}'.",
    )

    in_list_1 = []
    in_list_2 = []
    invalid_count = 0

    for f in feats:
        a = f.attributes()
        v1 = a[idx1]
        v2 = a[idx2]
        if v1 is None or v2 is None:
            invalid_count += 1
            continue
        try:
            v1 = float(v1)
            v2 = float(v2)
        except (ValueError, TypeError):
            invalid_count += 1
            continue

        # Validate dip/plunge and dipdir/trend values
        if is_planes:  # Planes: dip, dipdir
            if not (0.0 <= v1 <= 90.0):
                invalid_count += 1
                continue
            if not (0.0 <= v2 <= 360.0):
                invalid_count += 1
                continue
        else:  # Lines: plunge, trend
            if not (0.0 <= v1 <= 90.0):
                invalid_count += 1
                continue
            if not (0.0 <= v2 <= 360.0):
                invalid_count += 1
                continue

        in_list_1.append(v1)
        in_list_2.append(v2)
    in_list_1 = np.array(in_list_1)
    in_list_2 = np.array(in_list_2)

    # now populate the dataframe according to plane vs. line
    data_lower = pd.DataFrame({})
    if is_planes:
        data_lower["dip"] = in_list_1
        data_lower["dipdir"] = in_list_2
        data_lower["strike"] = dipdir2strike(in_list_2)
        data_lower["plunge"] = 90.0 - in_list_1
        data_lower["trend"] = wrap360(in_list_2 + 180)
        l, m, n = dipdir_dip_to_pole_lmn(in_list_2, in_list_1)
    else:
        data_lower["plunge"] = in_list_1
        data_lower["trend"] = in_list_2
        l, m, n = trend_plunge_to_lmn(in_list_2, in_list_1)

    data_lower["l"] = l
    data_lower["m"] = m
    data_lower["n"] = n
    data_lower["cluster"] = None
    data_lower["lower_hemi"] = True

    # now for axial data duplicate the data to perform axially symmetric orientation analysis
    if analysis_type == "axial":
        data_upper = data_lower.copy()
        data_upper["l"] = -data_upper["l"]
        data_upper["m"] = -data_upper["m"]
        data_upper["n"] = -data_upper["n"]
        data_upper["lower_hemi"] = False
        data = pd.concat([data_lower, data_upper]).reset_index(drop=True)
    else:
        data = data_lower.reset_index(drop=True)

    _log(
        log,
        f"Orientation reading complete: valid={data_lower.shape[0]}, invalid/skipped={invalid_count}.",
    )
    if not data.empty:
        _log(
            log,
            data.iloc[0:10].to_string(),
        )
    return data


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


class qAttitudeDialog(QDialog):
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

        # K-means
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
            self.plugin.data = self.plugin.data.iloc[0:0]
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
            self.plugin.data = self.plugin.data.iloc[0:0]
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
            self.ax.clear()
            self.ax.grid(True, zorder=0, alpha=0.5)
            self.ax.grid(kind="polar", zorder=0, alpha=0.5)

            if not self.plugin.data.empty:
                query = "lower_hemi == True"
                if show_contours:
                    self.ax.density_contourf(
                        self.plugin.data.query(query)["plunge"],
                        self.plugin.data.query(query)["trend"],
                        measurement="lines",
                        cmap=self.contour_cmap_combo.currentText(),
                        alpha=0.6,
                        levels=int(self.contour_levels.value()),
                        zorder=1,
                    )
                if plot_gcs:
                    self.ax.plane(
                        self.plugin.data.query(query)["strike"].to_list(),
                        self.plugin.data.query(query)["dip"].to_list(),
                        color=self.point_color_combo.currentText(),
                        linewidth=1,
                        alpha=0.5,
                        zorder=2,
                    )
                if plot_poles:
                    self.ax.line(
                        self.plugin.data.query(query)["plunge"].to_list(),
                        self.plugin.data.query(query)["trend"].to_list(),
                        ".",
                        color=self.point_color_combo.currentText(),
                        markersize=4,
                        alpha=1,
                        zorder=3,
                    )
                try:
                    x, y = stereonet_math.line(
                        self.plugin.data.query(query)["plunge"].to_list(),
                        self.plugin.data.query(query)["trend"].to_list(),
                    )
                    self._last_projected = np.column_stack([x, y]).astype(float)
                except Exception:
                    self._last_projected = None

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
        analysis_type = "axial" if self.analysis_axial.isChecked() else "polar"
        n_clusters = k
        if analysis_type == "axial":
            n_clusters = k * 2

        n = len(self.plugin.data)
        if n == 0:
            self.append_log("No data to cluster.")
            return

        if k > n:
            self.append_log(
                f"Number of clusters ({k}) is greater than the number of data points ({n}). "
                f"Reducing number of clusters to {n}."
            )
            k = n
            n_clusters = k * 2 if analysis_type == "axial" else k
            self.k_spin.blockSignals(True)
            self.k_spin.setValue(k)
            self.k_spin.blockSignals(False)

        vectors = self.plugin.data[["l", "m", "n"]].values
        kmeans_model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init="auto",
            max_iter=100,
            tol=0.0001,
            verbose=0,
            random_state=self.seed_spin.value(),
            copy_x=True,
            algorithm="lloyd",
        ).fit(vectors)

        self.plugin.data["cluster"] = kmeans_model.labels_

        means = pd.DataFrame(kmeans_model.cluster_centers_, columns=["l", "m", "n"])
        means["cluster"] = np.arange(means.shape[0])
        means["trend"], means["plunge"] = lmn_to_trend_plunge(
            means["l"], means["m"], means["n"]
        )
        means["lower_hemi"] = means["n"] <= 0
        means["dip"] = 90.0 - means["plunge"]
        dipdir = wrap360(means["trend"] - 180)
        means["strike"] = dipdir2strike(dipdir)

        if analysis_type == "axial":
            means = means.loc[means["lower_hemi"] == True].reset_index(drop=True)

        is_planes = self.data_planes.isChecked()
        plane_mode = self.plane_mode_combo.currentIndex()
        plot_poles = not is_planes or plane_mode in (0, 2)
        plot_gcs = is_planes and plane_mode in (1, 2)

        cmap = plt.get_cmap("tab10")
        for cluster in means["cluster"].to_list():
            query = "lower_hemi == True and cluster == " + str(cluster)
            color = cmap(cluster % 10)

            if plot_poles:
                self.ax.line(
                    self.plugin.data.query(query)["plunge"].to_list(),
                    self.plugin.data.query(query)["trend"].to_list(),
                    ".",
                    color=color,
                    markersize=6,
                    alpha=0.9,
                    zorder=4,
                )
                self.ax.line(
                    means.query(f"cluster == {cluster}")["plunge"].to_list(),
                    means.query(f"cluster == {cluster}")["trend"].to_list(),
                    marker="*",
                    color=color,
                    markersize=14,
                    markeredgecolor="k",
                    zorder=5,
                )

            if plot_gcs:
                self.ax.plane(
                    self.plugin.data.query(query)["strike"].to_list(),
                    self.plugin.data.query(query)["dip"].to_list(),
                    color=color,
                    linewidth=1.0,
                    alpha=0.45,
                    zorder=4,
                )
                self.ax.plane(
                    means.query(f"cluster == {cluster}")["strike"].to_list(),
                    means.query(f"cluster == {cluster}")["dip"].to_list(),
                    color=color,
                    linewidth=4.0,
                    alpha=1,
                    zorder=5,
                )
        self.append_log("k-means clustering completed.")

    def _plot_vmf(self):
        V = self.plugin.data[["l", "m", "n"]].values
        S = V.sum(axis=0)
        S_norm = float(np.linalg.norm(S))
        if S_norm == 0.0:
            return

        mean_xyz = S / S_norm
        tr, pl = lmn_to_trend_plunge(mean_xyz[0], mean_xyz[1], mean_xyz[2])
        self.ax.line(pl, tr, "r*", markersize=12, zorder=5)
        self.append_log(
            f"VMF mean plotted at trend={tr:.2f}, plunge={pl:.2f} in red."
        )

    def _plot_bingham(self):
        V = self.plugin.data[["l", "m", "n"]].values
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        T = (V.T @ V) / V.shape[0]
        evals, evecs = np.linalg.eigh(T)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        beta = evecs[:, 0]
        beta = beta / np.linalg.norm(beta)

        tr, pl = lmn_to_trend_plunge(beta[0], beta[1], beta[2])
        self.ax.line(pl, tr, "D", color="#1f77b4", markersize=7, zorder=5)

        dipdir = wrap360(tr + 180)
        dip = 90.0 - pl
        strike = dipdir2strike(dipdir)
        self.ax.plane(strike, dip, color="#1f77b4", linewidth=1.4, alpha=0.85, zorder=5)
        self.append_log(
            f"Bingham beta axis plotted at trend={tr:.2f}, plunge={pl:.2f}."
        )
