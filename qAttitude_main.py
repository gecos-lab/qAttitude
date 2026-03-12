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

pd.options.display.expand_frame_repr = True
pd.options.display.max_colwidth = 8
pd.options.display.max_rows = 20
pd.options.display.precision = 3

from sklearn.cluster import KMeans
import sphstat.singlesample
from mpmath.libmp.backend import sage_utils
import sphstat.singlesample as ss_singlesample
import sphstat.utils as ss_utils

from qgis.PyQt.QtWidgets import (
    QWidget,
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
    QScrollArea,
)

from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsMimeDataUtils,
    QgsMapLayer,
    QgsProcessingException,
)

from qgis.PyQt.QtCore import pyqtSignal, Qt, QEvent

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mplstereonet
from mplstereonet import stereonet_math


# ================= useful functions =================


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
    pole_trend = wrap360(np.asarray(dipdir) + 180)
    pole_plunge = 90.0 - dip
    return trend_plunge_to_lmn(pole_trend, pole_plunge)


def dipdir2strike(dipdir):
    strike = wrap360(np.asarray(dipdir) - 90.0)
    return strike


def strike2dipdir(strike):
    dipdir = wrap360(np.asarray(strike) + 90.0)
    return dipdir


def trend2dipdir(trend):
    dipdir = wrap360(np.asarray(trend) + 180)
    return dipdir


def dipdir2trend(dipdir):
    trend = wrap360(np.asarray(dipdir) + 180)
    return trend


def trend2strike(trend):
    strike = dipdir2strike(trend2dipdir(np.asarray(trend)))
    return strike


def strike2trend(strike):
    trend = dipdir2trend(strike2dipdir(np.asarray(strike)))
    return trend


def dip2plunge(dip):
    plunge = 90.0 - np.asarray(dip)
    return plunge


def plunge2dip(plunge):
    dip = 90.0 - np.asarray(plunge)
    return dip


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
    data_lower["low_hemi"] = True

    # now for axial data duplicate the data to perform axially symmetric orientation analysis
    if analysis_type == "axial":
        data_upper = data_lower.copy()
        data_upper["l"] = -data_upper["l"]
        data_upper["m"] = -data_upper["m"]
        data_upper["n"] = -data_upper["n"]
        data_upper["low_hemi"] = False
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


# ================= useful classes =================


class LayerDropGroupBox(QGroupBox):
    """Class used to manage drag and drop of layer into dialog."""

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


# ================= main class =================


class qAttitudeDialog(QWidget):
    """Main class creating the dialog and managing analysis and plotting with various methods."""

    def __init__(self, iface, plugin):
        super().__init__()
        self.iface = iface
        self.plugin = plugin

        self.analysis_layer = None
        self._layer_ids_by_index = []  # keeps combo index -> layer.id()

        self._picked_medoid_indices = []
        self._picking_enabled = False
        self._last_projected = None

        # data dataframe storing all data dynamically selected in the layer
        data_columns = [
            "dip",
            "dipdir",
            "strike",
            "plunge",
            "trend",
            "l",
            "m",
            "n",
            "cluster",
            "low_hemi",
        ]
        self.data = pd.DataFrame(columns=data_columns)

        # means dataframe storing mean orientations and other parameters for each cluster created on the fly
        means_columns = [
            "cluster",
            "low_hemi",
            "n_data",
            "k_tr",
            "k_pl",
            "vmf_tr",
            "vmf_pl",
            "vmf_K",
            "vmf_t_a",
            "vmf_ck_l",
            "vmf_ck_h",
            "kent_tr",
            "kent_pl",
            "kent_K",
            "kent_b",
            "kent_ts1",
            "kent_ts2",
            "bg_e1_tr",
            "bg_e1_pl",
            "bg_e1_mg",
            "bg_e2_tr",
            "bg_e2_pl",
            "bg_e2_mg",
            "bg_e3_tr",
            "bg_e3_pl",
            "bg_e3_mg",
        ]
        self.means = pd.DataFrame(columns=means_columns)

        self._init_ui()
        self._plot_empty()

    # ================= GUI methods =================

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel and (
            isinstance(source, QComboBox) or isinstance(source, QSpinBox)
        ):
            return True
        return super().eventFilter(source, event)

    def showEvent(self, event):
        """Reimplements showEvent() that runs when opening the plugin."""
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
        self._load_data_calc_and_plot()

    def _init_ui(self):
        """Creates all GUI objects."""
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        # Plot canvas
        self.fig = Figure(figsize=(6, 6), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(300)
        main.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection="stereonet")
        self.ax.grid(True)
        self.ax.grid(kind="polar")
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        # Scroll area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main.addWidget(scroll_area)

        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        controls_layout = QVBoxLayout(scroll_content)

        # Inputs
        g_in = LayerDropGroupBox("Input layer (all or selected features)", self)
        g_in.layerDropped.connect(self.on_layer_dropped)
        controls_layout.addWidget(g_in)
        grid = QGridLayout(g_in)

        grid.addWidget(QLabel("Layer:"), 0, 0)
        self.layer_combo = QComboBox()
        self.layer_combo.installEventFilter(self)
        grid.addWidget(self.layer_combo, 0, 1, 1, 3)

        self.field1_label = QLabel("Dip field:")
        self.field2_label = QLabel("DipDir field:")
        self.field1_combo = QComboBox()
        self.field1_combo.installEventFilter(self)
        self.field2_combo = QComboBox()
        self.field2_combo.installEventFilter(self)
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
        self.field1_combo.currentIndexChanged.connect(self._load_data_calc_and_plot)
        self.field2_combo.currentIndexChanged.connect(self._load_data_calc_and_plot)
        self.analysis_axial.toggled.connect(self._load_data_calc_and_plot)
        self.analysis_polar.toggled.connect(self._load_data_calc_and_plot)

        # K-means
        g_km = QGroupBox("K-means clustering")
        controls_layout.addWidget(g_km)
        gridk = QGridLayout(g_km)

        gridk.addWidget(QLabel("Number of clusters:"), 0, 0)
        self.k_spin = QSpinBox()
        self.k_spin.installEventFilter(self)
        self.k_spin.setRange(1, 20)
        self.k_spin.setValue(1)
        gridk.addWidget(self.k_spin, 0, 1)

        self.init_random = QRadioButton("Automatic seeds")
        self.init_pick = QRadioButton("Pick seeds on plot")
        self.init_random.setChecked(True)
        gridk.addWidget(self.init_random, 1, 0)
        gridk.addWidget(self.init_pick, 1, 1)

        gridk.addWidget(QLabel("Random seed:"), 2, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.installEventFilter(self)
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
        # create plot options group box
        g_plot = QGroupBox("Plot options")
        controls_layout.addWidget(g_plot)
        gridp = QGridLayout(g_plot)

        # create widgets
        self.chk_individual = QCheckBox("Plot data (mode for planes)")
        self.chk_individual.setChecked(True)

        self.plane_mode_combo = QComboBox()
        self.plane_mode_combo.installEventFilter(self)
        self.plane_mode_combo.addItems(["Poles", "Great circles", "Both"])

        self.chk_contours = QCheckBox("Plot contours (levels)")
        self.chk_contours.setChecked(False)

        self.contour_levels = QSpinBox()
        self.contour_levels.installEventFilter(self)
        self.contour_levels.setRange(3, 30)
        self.contour_levels.setValue(10)

        self.chk_plot_clusters = QCheckBox("Plot k-means clusters")
        self.chk_plot_clusters.setChecked(False)

        self.chk_plot_kmeans_poles = QCheckBox("Plot k-means center poles")
        self.chk_plot_kmeans_poles.setChecked(False)

        self.chk_plot_kmeans_gcs = QCheckBox("Plot k-means center great circles")
        self.chk_plot_kmeans_gcs.setChecked(False)

        self.chk_vmf_pl = QCheckBox("Plot Von Mises-Fisher (Kent) mean poles")
        self.chk_vmf_pl.setChecked(False)

        self.chk_vmf_gcs = QCheckBox("Plot Von Mises-Fisher (Kent) mean great circles")
        self.chk_vmf_gcs.setChecked(False)

        self.chk_bingham_1 = QCheckBox("Plot Bingham major axis as poles")
        self.chk_bingham_1.setChecked(False)

        self.chk_bingham_2 = QCheckBox("Plot Bingham intermediate axis as poles")
        self.chk_bingham_2.setChecked(False)

        self.chk_bingham_3 = QCheckBox("Plot Bingham minor axis as poles")
        self.chk_bingham_3.setChecked(False)

        self.chk_bingham_gcs = QCheckBox("Plot great circles ┴ to Bingham minor axis")
        self.chk_bingham_gcs.setChecked(False)

        self.point_color_combo = QComboBox()
        self.point_color_combo.installEventFilter(self)
        self.point_color_combo.addItems(
            [
                "black",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "yellow",
                "purple",
                "brown",
                "lightgreen",
                "olive",
            ]
        )

        self.contour_cmap_combo = QComboBox()
        self.contour_cmap_combo.installEventFilter(self)
        cmaps = [
            "Greys",
            "Reds",
            "Greens",
            "Blues",
            "Purples",
            "Oranges",
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

        # add widgets to grid layout
        gridp.addWidget(self.chk_individual, 0, 0)
        gridp.addWidget(self.plane_mode_combo, 0, 1)
        gridp.addWidget(self.chk_contours, 1, 0)
        gridp.addWidget(self.contour_levels, 1, 1)
        gridp.addWidget(self.chk_plot_clusters, 2, 0, 1, 2)
        gridp.addWidget(self.chk_plot_kmeans_poles, 3, 0, 1, 2)
        gridp.addWidget(self.chk_plot_kmeans_gcs, 4, 0, 1, 2)
        gridp.addWidget(self.chk_vmf_pl, 5, 0, 1, 2)
        gridp.addWidget(self.chk_vmf_gcs, 6, 0, 1, 2)
        gridp.addWidget(self.chk_bingham_1, 7, 0, 1, 2)
        gridp.addWidget(self.chk_bingham_2, 8, 0, 1, 2)
        gridp.addWidget(self.chk_bingham_3, 9, 0, 1, 2)
        gridp.addWidget(self.chk_bingham_gcs, 10, 0, 1, 2)
        gridp.addWidget(QLabel("Point/GCs Color:"), 11, 0)
        gridp.addWidget(self.point_color_combo, 11, 1)
        gridp.addWidget(QLabel("Contour Color Map:"), 12, 0)
        gridp.addWidget(self.contour_cmap_combo, 12, 1)

        # connect signals
        self.k_spin.valueChanged.connect(self._calc_clusters_and_plot)
        self.chk_individual.stateChanged.connect(self._update_plot)
        self.plane_mode_combo.currentIndexChanged.connect(self._update_plot)
        self.chk_contours.stateChanged.connect(self._update_plot)
        self.contour_levels.valueChanged.connect(self._update_plot)
        self.chk_plot_clusters.stateChanged.connect(self._update_plot)
        self.chk_plot_kmeans_poles.stateChanged.connect(self._update_plot)
        self.chk_plot_kmeans_gcs.stateChanged.connect(self._update_plot)
        self.chk_vmf_pl.stateChanged.connect(self._update_plot)
        self.chk_vmf_gcs.stateChanged.connect(self._update_plot)
        self.chk_bingham_1.stateChanged.connect(self._update_plot)
        self.chk_bingham_2.stateChanged.connect(self._update_plot)
        self.chk_bingham_3.stateChanged.connect(self._update_plot)
        self.chk_bingham_gcs.stateChanged.connect(self._update_plot)
        self.point_color_combo.currentIndexChanged.connect(self._update_plot)
        self.contour_cmap_combo.currentIndexChanged.connect(self._update_plot)

        # Saving
        g_save = QGroupBox("Save to files (off by default)")
        controls_layout.addWidget(g_save)
        grids = QGridLayout(g_save)

        self.chk_save = QCheckBox("Save outputs")
        self.chk_save.setChecked(False)

        self.out_dir = QLineEdit("")

        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse_dir)

        grids.addWidget(self.chk_save, 0, 0, 1, 2)
        grids.addWidget(QLabel("Directory:"), 1, 0)
        grids.addWidget(self.out_dir, 1, 1)
        grids.addWidget(self.btn_browse, 1, 2)

        # Log window
        g_log = QGroupBox("Log")
        controls_layout.addWidget(g_log)
        gridlog = QGridLayout(g_log)

        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Analysis log messages will appear here.")
        self.log_output.setMinimumHeight(180)
        gridlog.addWidget(self.log_output, 0, 0, 1, 2)

        self.btn_clear_log = QPushButton("Clear log")
        self.btn_clear_log.clicked.connect(self.clear_log)
        gridlog.addWidget(self.btn_clear_log, 1, 1)

        controls_layout.addStretch(1)

    def on_layer_dropped(self, layer):
        """Runs when a layer is dropped on the plugin window."""
        self.set_analysis_layer(layer)
        self._populate_layers()
        self._select_layer_in_combo(layer)
        self._refresh_fields()
        self._refresh_field_controls()
        self._load_data_calc_and_plot()

    def on_layer_combo_changed(self):
        """Runs when the layer combo value is changed."""
        layer = self._current_layer_from_combo()
        self.set_analysis_layer(layer)
        self._refresh_fields()
        self._refresh_field_controls()
        self._load_data_calc_and_plot()

    def on_data_type_changed(self, checked):
        """Runs when the data type buttons value is changed."""
        if not checked:
            return
        self._refresh_field_controls()
        self._load_data_calc_and_plot()

    def _populate_layers(self):
        """Used to clear the layer combo from previous list and populate it with a new one."""
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
        """Used to select the layer while blocking signals so other parte of the GUI are not messed up."""
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
        """Used to set the layer to be analyzed at startup, when a layer is dropped, or after the
        combo is changed. Signals are disconnected and then reconnected to avoid loops.
        """
        if self.analysis_layer:
            try:
                self.analysis_layer.selectionChanged.disconnect(
                    self._load_data_calc_and_plot
                )
            except TypeError:
                pass

        if layer and layer.type() == QgsMapLayer.VectorLayer:
            self.analysis_layer = layer
            self.analysis_layer.selectionChanged.connect(self._load_data_calc_and_plot)
            self.append_log(f"Input layer set to: {layer.name()}")
        else:
            self.analysis_layer = None

    def _current_layer_from_combo(self) -> QgsMapLayer | None:
        """Get the current layer from combo."""
        if self.layer_combo.currentIndex() >= 0:
            idx = self.layer_combo.currentIndex()
            if 0 <= idx < len(self._layer_ids_by_index):
                layer_id = self._layer_ids_by_index[idx]
                return QgsProject.instance().mapLayer(layer_id)
        return None

    def _refresh_fields(self):
        """Used to refresh all fields at startup or when the layer is changed."""
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
        """Used to change controls depending on planes vs lines."""
        is_planes = self.data_planes.isChecked()
        self.field1_label.setText("Dip field:" if is_planes else "Plunge field:")
        self.field2_label.setText("DipDir field:" if is_planes else "Trend field:")
        self.plane_mode_combo.setEnabled(is_planes)
        self.chk_plot_kmeans_gcs.setEnabled(is_planes)
        self.chk_vmf_gcs.setEnabled(is_planes)

    def _browse_dir(self):
        """Used to browse directory for saving."""
        d = QFileDialog.getExistingDirectory(self, "Select output directory", "")
        if d:
            self.out_dir.setText(d)

    def _toggle_picking(self):
        """Used to toggle picking of kmedoids seeds."""
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
        """Used to clear kmedoids seeds."""
        self._picked_medoid_indices = []
        self.lbl_picks.setText("Picked: 0")

    def _on_plot_click(self, event):
        """Used for picking of kmedoids seeds."""
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

    def append_log(self, message: str):
        self.log_output.appendPlainText(str(message))

    def clear_log(self):
        self.log_output.clear()

    # ================= PROCESSING AND ANALYSIS methods =================

    def _load_data_calc_and_plot(self):
        """Main method to load data, calculate means, depending on the number of
        clusters and axial/polar, and finally plot."""
        sender = self.sender()
        if isinstance(sender, QRadioButton) and not sender.isChecked():
            return

        layer = self.analysis_layer
        if (
            not layer
            or not self.field1_combo.currentText()
            or not self.field2_combo.currentText()
        ):
            self.data = self.data.iloc[0:0]
            self.means = self.means.iloc[0:0]
            self._update_plot()
            return

        is_planes = self.data_planes.isChecked()
        field1 = self.field1_combo.currentText()
        field2 = self.field2_combo.currentText()
        analysis_type = "axial" if self.analysis_axial.isChecked() else "polar"

        try:
            self.data = read_orientations_from_layer_selection(
                layer,
                field1,
                field2,
                is_planes=is_planes,
                analysis_type=analysis_type,
                log=self.append_log,
            )
            # when data are properly loaded, run analysis and plot
            self._calc_clusters_and_plot()
        except Exception as e:
            self.data = self.data.iloc[0:0]
            self.means = self.means.iloc[0:0]
            tb = traceback.format_exc()
            self.append_log(f"ERROR: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self, "qAttitude", f"Error: {type(e).__name__}: {e}\n\n{tb}"
            )

    def _calc_clusters_and_plot(self):
        """Used to calculate kmeans clusters when their number is changed ot seeds are redefined, and finally plot."""
        try:
            k = int(self.k_spin.value())
            analysis_type = "axial" if self.analysis_axial.isChecked() else "polar"
            n_clusters = k
            if analysis_type == "axial":
                n_clusters = k * 2

            n = len(self.data)
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

            vectors = self.data[["l", "m", "n"]].values
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

            # assign cluster labels to data
            self.data["cluster"] = kmeans_model.labels_

            # clean and polulate the means dataframe
            self.means = self.means.iloc[0:0]
            self.means["cluster"] = np.arange(kmeans_model.cluster_centers_.shape[0])
            self.means["low_hemi"] = kmeans_model.cluster_centers_[:, 2] <= 0
            self.means["n_data"] = self.data.groupby("cluster").size()
            self.means["k_tr"], self.means["k_pl"] = lmn_to_trend_plunge(
                kmeans_model.cluster_centers_[:, 0],
                kmeans_model.cluster_centers_[:, 1],
                kmeans_model.cluster_centers_[:, 2],
            )

            if analysis_type == "axial":
                self.means = self.means.loc[self.means["low_hemi"] == True].reset_index(
                    drop=True
                )

            # calculate means for each cluster using the samplecart dictionary in the format used by ss
            for cluster in self.means["cluster"]:
                samplecart = dict()
                samplecart["points"] = self.data.loc[
                    self.data["cluster"] == cluster, ["l", "m", "n"]
                ].to_numpy()
                samplecart["type"] = "cart"
                samplecart["n"] = self.data.loc[self.data["cluster"] == cluster].shape[
                    0
                ]

                # Fisher distribution parameters
                alpha = 0.05
                fisher_dist = ss_singlesample.fisherparams(
                    samplecart=samplecart, alpha=alpha
                )
                th, ph = fisher_dist["mdir"]
                lat, lon = ss_utils.poltoll(th, ph)
                self.means.loc[self.means["cluster"] == cluster, "vmf_pl"] = (
                    lat * 180 / np.pi
                )
                self.means.loc[self.means["cluster"] == cluster, "vmf_tr"] = (
                    lon * 180 / np.pi
                )
                self.means.loc[self.means["cluster"] == cluster, "vmf_K"] = fisher_dist[
                    "kappa"
                ]
                self.means.loc[self.means["cluster"] == cluster, "vmf_t_a"] = (
                    fisher_dist["thetalpha"] * 180 / np.pi
                )
                (
                    self.means.loc[self.means["cluster"] == cluster, "vmf_ck_l"],
                    self.means.loc[self.means["cluster"] == cluster, "vmf_ck_h"],
                ) = fisher_dist["cikappa"]

                # Kent distribution parameters
                axes, kappahat, betahat = ss_singlesample.kentparams(
                    samplecart=samplecart
                )
                th, ph = ss_utils.cart2sph(axes[0])
                th = th % (np.pi)
                ph = ph % (2 * np.pi)
                lat, lon = ss_utils.poltoll(th, ph)
                self.means.loc[self.means["cluster"] == cluster, "kent_pl"] = (
                    lat * 180 / np.pi
                )
                self.means.loc[self.means["cluster"] == cluster, "kent_tr"] = (
                    lon * 180 / np.pi
                )
                self.means.loc[self.means["cluster"] == cluster, "kent_K"] = kappahat
                self.means.loc[self.means["cluster"] == cluster, "kent_b"] = betahat

                # Kent elliptical confidence cone for the mean direction
                # cconept could be used to plot ellipse on stereoplot, but must be converted to plunge/trend
                cconept, ths1, ths2 = ss_singlesample.kentmeanccone(
                    samplecart=samplecart
                )
                self.means.loc[self.means["cluster"] == cluster, "kent_ts1"] = (
                    ths1 * 180 / np.pi
                )
                self.means.loc[self.means["cluster"] == cluster, "kent_ts2"] = (
                    ths2 * 180 / np.pi
                )

                # Bingham distribution parameters
                T = samplecart["points"].T @ samplecart["points"] / samplecart["n"]
                eigenvalues, eigenvectors = np.linalg.eig(T)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                self.means.loc[self.means["cluster"] == cluster, "bg_e1_mg"] = (
                    eigenvalues[0]
                )
                self.means.loc[self.means["cluster"] == cluster, "bg_e2_mg"] = (
                    eigenvalues[1]
                )
                self.means.loc[self.means["cluster"] == cluster, "bg_e3_mg"] = (
                    eigenvalues[2]
                )
                e1 = eigenvectors[:, 0]
                e2 = eigenvectors[:, 1]
                e3 = eigenvectors[:, 2]
                if e1[2] > 0:
                    e1 = -e1
                if e2[2] > 0:
                    e2 = -e2
                if e3[2] > 0:
                    e3 = -e3
                (
                    self.means.loc[self.means["cluster"] == cluster, "bg_e1_tr"],
                    self.means.loc[self.means["cluster"] == cluster, "bg_e1_pl"],
                ) = lmn_to_trend_plunge(e1[0], e1[1], e1[2])
                (
                    self.means.loc[self.means["cluster"] == cluster, "bg_e2_tr"],
                    self.means.loc[self.means["cluster"] == cluster, "bg_e2_pl"],
                ) = lmn_to_trend_plunge(e2[0], e2[1], e2[2])
                (
                    self.means.loc[self.means["cluster"] == cluster, "bg_e3_tr"],
                    self.means.loc[self.means["cluster"] == cluster, "bg_e3_pl"],
                ) = lmn_to_trend_plunge(e3[0], e3[1], e3[2])

            self.append_log(f"means dataframe:\n{self.means.to_string()}")

            # tests

            # # Is uniform [True] test:
            # uniform_test = ss_singlesample.isuniform(sample=samplecart, alpha=alpha)
            # this_stats['Uniform test statistic'] = uniform_test['teststat']
            # this_stats['Uniform critical range'] = uniform_test['crange']
            # this_stats['Is uniform test'] = uniform_test['testresult']
            #
            # # Is Fisher [True] test
            # fisher_test = ss_singlesample.isfisher(samplecart=samplecart, alpha=alpha, plotflag=False)
            # this_stats['Colatitute test statistic'] = fisher_test['colatitute']['stat']
            # this_stats['Colatitute critical range'] = fisher_test['colatitute']['crange']
            # this_stats['Is colatitute exponential'] = fisher_test['colatitute']['H0']
            # this_stats['Longitude test statistic'] = fisher_test['longitude']['stat']
            # this_stats['Longitude critical range'] = fisher_test['longitude']['crange']
            # this_stats['Is longitude uniform'] = fisher_test['longitude']['H0']
            # this_stats['Two-variable test statistic'] = fisher_test['twovariable']['stat']
            # this_stats['Two-variable critical range'] = fisher_test['twovariable']['crange']
            # this_stats['Is two-variable normal'] = fisher_test['twovariable']['H0']
            # this_stats['Is Fisher test'] = fisher_test['H0']
            #
            # # Is Fisher [True] vs. Kent [False] test
            # fisher_kent_test = ss_singlesample.isfishervskent(samplecart=samplecart, alpha=alpha)
            # this_stats['Fisher vs. Kent test statistic'] = fisher_kent_test['K']
            # this_stats['Fisher vs. Kent critical value'] = fisher_kent_test['cval']
            # this_stats['Fisher vs. Kent p-value'] = fisher_kent_test['p']
            # this_stats['Is Fisher vs. Kent test'] = fisher_kent_test['testresult']

        except Exception as e:
            self.means = self.means.iloc[0:0]
            tb = traceback.format_exc()
            self.append_log(
                "ERROR: more than one data point needs to be selected to calculate average values."
            )
            # QMessageBox.critical(
            #     self, "qAttitude", "ERROR: more than one data point needs to be selected to calculate average values."
            # )

        # finally update plot
        self._update_plot()

    # ================= PLOTTING methods =================

    def _plot_empty(self):
        """Used to create an empty plot at startup or when no data are available."""
        try:
            self.ax.clear()
            self.ax.grid(True, zorder=0, alpha=0.5)
            self.ax.grid(kind="polar", zorder=0, alpha=0.5)
            self.canvas.draw()
        except Exception:
            pass

    def _update_plot(self):
        """Used to update plot depending on GUI state. First the base plot is created, then overlays are added."""
        if self.data.empty:
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

            if not self.data.empty:
                query = "low_hemi == True"
                if show_contours:
                    self.ax.density_contourf(
                        self.data.query(query)["plunge"],
                        self.data.query(query)["trend"],
                        measurement="lines",
                        cmap=self.contour_cmap_combo.currentText(),
                        alpha=0.6,
                        levels=int(self.contour_levels.value()),
                        zorder=1,
                    )
                if show_individual:
                    if plot_gcs:
                        self.ax.plane(
                            self.data.query(query)["strike"].to_list(),
                            self.data.query(query)["dip"].to_list(),
                            color=self.point_color_combo.currentText(),
                            linewidth=1,
                            alpha=0.5,
                            zorder=2,
                        )
                    if plot_poles:
                        self.ax.line(
                            self.data.query(query)["plunge"].to_list(),
                            self.data.query(query)["trend"].to_list(),
                            ".",
                            color=self.point_color_combo.currentText(),
                            markersize=4,
                            alpha=1,
                            zorder=3,
                        )
                try:
                    x, y = stereonet_math.line(
                        self.data.query(query)["plunge"].to_list(),
                        self.data.query(query)["trend"].to_list(),
                    )
                    self._last_projected = np.column_stack([x, y]).astype(float)
                except Exception:
                    self._last_projected = None

            if self.chk_plot_clusters.isChecked():
                self._plot_clusters()
            if self.chk_plot_kmeans_poles.isChecked():
                self._plot_kmeans_poles()
            if self.chk_plot_kmeans_gcs.isChecked():
                self._plot_kmeans_gcs()
            if self.chk_vmf_pl.isChecked():
                self._plot_vmf_pl()
            if self.chk_vmf_gcs.isChecked():
                self._plot_vmf_gcs()
            if self.chk_bingham_1.isChecked():
                self._plot_bingham_1()
            if self.chk_bingham_2.isChecked():
                self._plot_bingham_2()
            if self.chk_bingham_3.isChecked():
                self._plot_bingham_3()
            if self.chk_bingham_gcs.isChecked():
                self._plot_bingham_gcs()

            self.canvas.draw()
            self.append_log("Plot updated.")
        except Exception as e:
            tb = traceback.format_exc()
            self.append_log(f"ERROR: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self, "qAttitude", f"Error: {type(e).__name__}: {e}\n\n{tb}"
            )

    def _plot_clusters(self):
        """Used to plot clusters."""
        is_planes = self.data_planes.isChecked()
        plane_mode = self.plane_mode_combo.currentIndex()
        plot_poles = not is_planes or plane_mode in (0, 2)
        plot_gcs = is_planes and plane_mode in (1, 2)
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"].to_list():
            query = "low_hemi == True and cluster == " + str(cluster)
            color = cmap(cluster % 10)

            if plot_poles:
                self.ax.line(
                    self.data.query(query)["plunge"].to_list(),
                    self.data.query(query)["trend"].to_list(),
                    ".",
                    color=color,
                    markersize=6,
                    alpha=0.9,
                    zorder=4,
                )

            if plot_gcs:
                self.ax.plane(
                    self.data.query(query)["strike"].to_list(),
                    self.data.query(query)["dip"].to_list(),
                    color=color,
                    linewidth=1.0,
                    alpha=0.45,
                    zorder=4,
                )

        self.append_log("k-means clusters plotted.")

    def _plot_kmeans_poles(self):
        """Used to plot k-means cluster poles."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"].to_list():
            color = cmap(cluster % 10)

            self.ax.line(
                self.means.query(f"cluster == {cluster}")["k_pl"].to_list(),
                self.means.query(f"cluster == {cluster}")["k_tr"].to_list(),
                marker="o",
                color=color,
                markersize=10,
                markeredgecolor="k",
                zorder=6,
            )
        self.append_log("k-means poles plotted.")

    def _plot_kmeans_gcs(self):
        """Used to plot k-means cluster great circles."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"].to_list():
            color = cmap(cluster % 10)

            self.ax.plane(
                trend2strike(
                    self.means.query(f"cluster == {cluster}")["k_tr"].to_list()
                ),
                plunge2dip(self.means.query(f"cluster == {cluster}")["k_pl"].to_list()),
                ls="-",
                color=color,
                linewidth=2.0,
                alpha=1,
                zorder=5,
            )
        self.append_log("k-means great circles plotted.")

    def _plot_vmf_pl(self):
        """Used to plot Von Mises-Fisher mean."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"]:
            color = cmap(cluster % 10)
            self.ax.line(
                self.means.loc[self.means["cluster"] == cluster, "vmf_pl"].to_list(),
                self.means.loc[self.means["cluster"] == cluster, "vmf_tr"].to_list(),
                marker="*",
                color=color,
                markersize=12,
                markeredgecolor="k",
                zorder=6,
            )
        self.append_log(f"VMF mean poles plotted.")

    def _plot_vmf_gcs(self):
        """Used to plot Von Mises-Fisher mean."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"]:
            color = cmap(cluster % 10)
            self.ax.plane(
                trend2strike(
                    self.means.loc[self.means["cluster"] == cluster, "vmf_tr"].to_list()
                ),
                plunge2dip(
                    self.means.loc[self.means["cluster"] == cluster, "vmf_pl"].to_list()
                ),
                ls="-",
                color=color,
                linewidth=2.0,
                alpha=1,
                zorder=5,
            )
        self.append_log(f"VMF mean great circles plotted.")

    def _plot_bingham_1(self):
        """Used to plot Bingham mean."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"]:
            color = cmap(cluster % 10)
            self.ax.line(
                self.means.loc[self.means["cluster"] == cluster, "bg_e1_pl"].to_list(),
                self.means.loc[self.means["cluster"] == cluster, "bg_e1_tr"].to_list(),
                marker="H",
                color=color,
                markersize=10,
                markeredgecolor="k",
                zorder=6,
            )
        self.append_log(f"Bingham main axis plotted.")

    def _plot_bingham_2(self):
        """Used to plot Bingham mean."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"]:
            color = cmap(cluster % 10)
            self.ax.line(
                self.means.loc[self.means["cluster"] == cluster, "bg_e2_pl"].to_list(),
                self.means.loc[self.means["cluster"] == cluster, "bg_e2_tr"].to_list(),
                marker="s",
                color=color,
                markersize=10,
                markeredgecolor="k",
                zorder=6,
            )
        self.append_log(f"Bingham main axis plotted.")

    def _plot_bingham_3(self):
        """Used to plot Bingham mean."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"]:
            color = cmap(cluster % 10)
            self.ax.line(
                self.means.loc[self.means["cluster"] == cluster, "bg_e3_pl"].to_list(),
                self.means.loc[self.means["cluster"] == cluster, "bg_e3_tr"].to_list(),
                marker="^",
                color=color,
                markersize=10,
                markeredgecolor="k",
                zorder=6,
            )
        self.append_log(f"Bingham main axis plotted.")

    def _plot_bingham_gcs(self):
        """Used to plot Bingham mean."""
        cmap = plt.get_cmap("tab10")

        for cluster in self.means["cluster"]:
            color = cmap(cluster % 10)
            self.ax.plane(
                trend2strike(
                    self.means.loc[
                        self.means["cluster"] == cluster, "bg_e3_tr"
                    ].to_list()
                ),
                plunge2dip(
                    self.means.loc[
                        self.means["cluster"] == cluster, "bg_e3_pl"
                    ].to_list()
                ),
                ls="-",
                color=color,
                linewidth=2.0,
                alpha=1,
                zorder=5,
            )
        self.append_log(f"Bingham main axis plotted.")
