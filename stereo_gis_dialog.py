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

from .stereo_gis_analysis import (
    read_orientations_from_layer_selection,
    vmf_mean_axial,
    bingham_principal_axes_axial,
    kmedoids_pam_axial,
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
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface = iface

        self.analysis_layer = None
        self._layer_ids_by_index = []  # keeps combo index -> layer.id()

        self.setWindowTitle("qAttitude")
        self.resize(1100, 750)

        self._picked_medoid_indices = []
        self._picking_enabled = False
        self._last_vectors_xyz = None
        self._last_projected = None

        self._init_ui()

        # Initialize the plot area immediately (empty stereonet)
        self._plot_empty("Drop/select a vector layer to begin")

    def showEvent(self, event):
        super().showEvent(event)
        self._populate_layers()

    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()
        right = QVBoxLayout()
        main.addLayout(left, 0)
        main.addLayout(right, 1)

        # Inputs
        g_in = LayerDropGroupBox("Input layer (all or selected features)", self)
        g_in.layerDropped.connect(self.set_analysis_layer)
        left.addWidget(g_in)
        grid = QGridLayout(g_in)

        grid.addWidget(QLabel("Layer:"), 0, 0)
        self.layer_combo = QComboBox()
        grid.addWidget(self.layer_combo, 0, 1, 1, 3)

        grid.addWidget(QLabel("Data:"), 1, 0)
        self.data_combo = QComboBox()
        self.data_combo.addItems(["Planes (dip/dipdir)", "Lines (plunge/trend)"])
        grid.addWidget(self.data_combo, 1, 1, 1, 3)

        self.field1_label = QLabel("Dip field:")
        self.field2_label = QLabel("DipDir field:")
        self.field1_combo = QComboBox()
        self.field2_combo = QComboBox()
        grid.addWidget(self.field1_label, 2, 0)
        grid.addWidget(self.field1_combo, 2, 1, 1, 3)
        grid.addWidget(self.field2_label, 3, 0)
        grid.addWidget(self.field2_combo, 3, 1, 1, 3)

        self.layer_combo.currentIndexChanged.connect(self._refresh_fields)
        self.data_combo.currentIndexChanged.connect(self._refresh_field_controls)

        # K-medoids
        g_km = QGroupBox("K-medoids clustering")
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
        gridk.addWidget(self.init_random, 1, 0, 1, 2)
        gridk.addWidget(self.init_pick, 2, 0, 1, 2)

        gridk.addWidget(QLabel("Random seed:"), 3, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 10 ** 9)
        self.seed_spin.setValue(0)
        gridk.addWidget(self.seed_spin, 3, 1)

        self.btn_pick = QPushButton("Pick seeds")
        self.btn_clear_picks = QPushButton("Clear picks")
        self.lbl_picks = QLabel("Picked: 0")
        gridk.addWidget(self.btn_pick, 4, 0)
        gridk.addWidget(self.btn_clear_picks, 4, 1)
        gridk.addWidget(self.lbl_picks, 5, 0, 1, 2)

        self.btn_pick.clicked.connect(self._toggle_picking)
        self.btn_clear_picks.clicked.connect(self._clear_picks)

        # Plot options and Parametric distribution fitting
        g_plot = QGroupBox("Plot options")
        left.addWidget(g_plot)
        gridp = QGridLayout(g_plot)

        self.chk_individual = QCheckBox("Plot individual data")
        self.chk_individual.setChecked(True)
        self.chk_contours = QCheckBox("Plot contours (points)")
        self.chk_contours.setChecked(False)
        self.contour_levels = QSpinBox()
        self.contour_levels.setRange(3, 30)
        self.contour_levels.setValue(10)

        gridp.addWidget(self.chk_individual, 0, 0, 1, 2)
        gridp.addWidget(self.chk_contours, 1, 0, 1, 1)
        gridp.addWidget(QLabel("Levels:"), 1, 1)
        gridp.addWidget(self.contour_levels, 1, 2)

        self.plane_mode_combo = QComboBox()
        self.plane_mode_combo.addItems(["Poles", "Great circles", "Both"])
        gridp.addWidget(QLabel("Planes mode:"), 2, 0)
        gridp.addWidget(self.plane_mode_combo, 2, 1, 1, 2)

        # Parametric distribution fitting (within Plot options)
        gridp.addWidget(QLabel("Parametric distribution fitting:"), 3, 0, 1, 2)

        self.chk_vmf = QCheckBox("Plot Von Mises-Fisher mean (red)")
        self.chk_vmf.setChecked(True)
        self.chk_bingham = QCheckBox("Plot Bingham β axis & girdle (blue)")
        self.chk_bingham.setChecked(True)

        gridp.addWidget(self.chk_vmf, 4, 0, 1, 2)
        gridp.addWidget(self.chk_bingham, 5, 0, 1, 2)

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

        # Run
        self.btn_run = QPushButton("Run analysis")
        left.addWidget(self.btn_run)
        self.btn_run.clicked.connect(self._run_analysis)

        left.addStretch(1)

        # Plot
        self.fig = Figure(figsize=(6, 6), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(500)
        right.addWidget(self.canvas, 5)

        self.ax = self.fig.add_subplot(111, projection="stereonet")
        self.ax.grid(True)
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

        self._refresh_field_controls()
        self._populate_layers()

    def _populate_layers(self):
        current_layer = self._current_layer()

        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self._layer_ids_by_index = []

        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() != QgsMapLayer.VectorLayer:
                continue
            self.layer_combo.addItem(layer.name())
            self._layer_ids_by_index.append(layer.id())

        self.layer_combo.blockSignals(False)

        if current_layer is not None:
            self._select_layer_in_combo(current_layer)
        elif self.analysis_layer is not None:
            self._select_layer_in_combo(self.analysis_layer)

        self._refresh_fields()

    def _select_layer_in_combo(self, layer: QgsMapLayer) -> None:
        """
        Select the given layer in the combo if present.
        """
        if layer is None:
            return
        try:
            idx = self._layer_ids_by_index.index(layer.id())
        except ValueError:
            return
        self.layer_combo.setCurrentIndex(idx)

    def set_analysis_layer(self, layer: QgsMapLayer):
        """
        Called when a layer is dropped onto the Input group box.
        Make drag&drop update the same selection used by _current_layer().
        """
        if layer is None:
            return
        if layer.type() != QgsMapLayer.VectorLayer:
            QMessageBox.warning(self, "qAttitude", "Please drop a vector layer.")
            return

        self.analysis_layer = layer
        self._populate_layers()
        self._select_layer_in_combo(layer)
        self._refresh_fields()
        self.append_log(f"Input layer set to: {layer.name()}")

    def _current_layer(self) -> QgsMapLayer | None:
        """
        Return the currently selected layer.
        Primary source: the UI selector (combo).
        Secondary source: self.analysis_layer (if combo not available).
        """
        if self.layer_combo.currentIndex() >= 0:
            idx = self.layer_combo.currentIndex()
            if 0 <= idx < len(self._layer_ids_by_index):
                layer_id = self._layer_ids_by_index[idx]
                lyr = QgsProject.instance().mapLayer(layer_id)
                if lyr is not None:
                    return lyr

        return self.analysis_layer

    def _refresh_fields(self):
        layer = self._current_layer()
        self.field1_combo.clear()
        self.field2_combo.clear()
        if not layer:
            return
        for f in layer.fields():
            self.field1_combo.addItem(f.name())
            self.field2_combo.addItem(f.name())

    def _refresh_field_controls(self):
        is_planes = self.data_combo.currentIndex() == 0
        if is_planes:
            self.field1_label.setText("Dip field:")
            self.field2_label.setText("DipDir field:")
            self.plane_mode_combo.setEnabled(True)
        else:
            self.field1_label.setText("Plunge field:")
            self.field2_label.setText("Trend field:")
            self.plane_mode_combo.setEnabled(False)

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
        if not self._picking_enabled:
            return
        if event.inaxes != self.ax:
            return
        if self._last_projected is None:
            QMessageBox.information(
                self,
                "Pick medoid seeds",
                "Run analysis first (to compute point projection), then pick medoid seeds.",
            )
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

    def _log_section(self, title: str, lines: list[str]) -> None:
        self.append_log(title)
        for line in lines:
            self.append_log(f"  {line}")

    def _plot_empty(self, title: str = "No data to plot") -> None:
        """
        Clears the plot and shows an empty stereonet.
        Safe to call any time (e.g., when no valid features/values exist).
        """
        try:
            self.fig.clear()
            self.ax = self.fig.add_subplot(111, projection="stereonet")
            self.ax.grid(True)
            # self.ax.set_title(title)
            self.canvas.draw()
        except Exception:
            # As a last resort, avoid crashing the plugin due to plotting issues
            pass

    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(str(message))

    def clear_log(self) -> None:
        self.log_output.clear()

    def _run_analysis(self):
        layer = self._current_layer()
        if layer is None:
            QMessageBox.critical(self, "qAttitude", "No vector layer selected.")
            return

        is_planes = self.data_combo.currentIndex() == 0
        field1 = self.field1_combo.currentText()
        field2 = self.field2_combo.currentText()

        try:
            self.append_log("=" * 60)
            self.append_log("Starting analysis")
            self.append_log(f"Layer: {layer.name()}")
            self.append_log(
                f"Mode: {'Planes (dip/dipdir)' if is_planes else 'Lines (plunge/trend)'}"
            )
            self.append_log(f"Fields: {field1}, {field2}")

            data = read_orientations_from_layer_selection(
                layer, is_planes, field1, field2, log=self.append_log
            )

            vectors_xyz = data["vectors_xyz"]
            n = int(vectors_xyz.shape[0])
            self.append_log(f"Valid orientations loaded: {n}")

            if n == 0:
                QMessageBox.warning(
                    self,
                    "qAttitude",
                    "No valid orientation values in the layer/selection.",
                )
                self._plot_empty()
                return

            trends = data["trends_deg"]
            plunges = data["plunges_deg"]

            # clear plot
            self.fig.clear()
            self.ax = self.fig.add_subplot(111, projection="stereonet")
            self.ax.grid(True)

            show_individual = self.chk_individual.isChecked()
            show_contours = self.chk_contours.isChecked()

            plane_mode = self.plane_mode_combo.currentIndex()  # 0 poles,1 gcs,2 both
            plot_poles = (not is_planes) or (plane_mode in (0, 2))
            plot_gcs = is_planes and (plane_mode in (1, 2))

            self.append_log(
                f"Plot options: individual={show_individual}, contours={show_contours}, "
                f"plot_poles={plot_poles}, plot_great_circles={plot_gcs}"
            )

            # base plot
            if show_individual and plot_poles:
                self.ax.line(plunges, trends, "k.", markersize=4, alpha=0.85)
                self.append_log(f"{n} poles plotted.")

            if (
                show_individual
                and plot_gcs
                and data.get("strikes_deg") is not None
                and data.get("dips_deg") is not None
            ):
                self.ax.plane(
                    data["strikes_deg"],
                    data["dips_deg"],
                    color="0.4",
                    linewidth=1,
                    alpha=1,
                )
                self.append_log(f"{n} great circles plotted.")

            if show_contours:
                self.ax.density_contourf(
                    plunges,
                    trends,
                    measurement="lines",
                    cmap="Greys",
                    alpha=0.6,
                    levels=int(self.contour_levels.value()),
                )
                self.append_log(
                    f"Contours plotted with {int(self.contour_levels.value())} levels."
                )

            # overlays
            if self.chk_vmf.isChecked():
                self.append_log("Computing Von Mises-Fisher mean...")
                vmf = vmf_mean_axial(vectors_xyz, log=self.append_log)
                m = vmf["mean_xyz"]
                if np.isfinite(m).all():
                    tr, pl = lmn_to_trend_plunge(m)
                    self.ax.line(pl, tr, "r*", markersize=12)
                    self.append_log(
                        f"VMF mean plotted at trend={tr:.2f}, plunge={pl:.2f} in red."
                    )

            if self.chk_bingham.isChecked():
                self.append_log("Computing Bingham principal axes...")
                b = bingham_principal_axes_axial(vectors_xyz, log=self.append_log)
                beta = b["beta_axis_xyz"]
                tr, pl = lmn_to_trend_plunge(beta)
                self.ax.line(pl, tr, "D", color="#1f77b4", markersize=7)

                dipdir = wrap360(tr + 180)
                dip = 90.0 - pl
                strike = dipdir2strike(dipdir)
                self.ax.plane(strike, dip, color="#1f77b4", linewidth=1.4, alpha=0.85)
                self.append_log(
                    f"Bingham beta axis plotted at trend={tr:.2f}, plunge={pl:.2f}."
                )

            # k-medoids (always performed)
            cluster_summary = []
            k = int(self.k_spin.value())
            self.append_log(f"Running k-medoids with k={k}.")

            if k > n:
                QMessageBox.warning(
                    self,
                    "qAttitude",
                    f"k={k} cannot exceed number of observations n={n}.",
                )
                self.append_log(f"Invalid k: {k} > {n}.")
                return

            if self.init_pick.isChecked():
                if len(self._picked_medoid_indices) != k:
                    QMessageBox.warning(
                        self,
                        "qAttitude",
                        f"Pick exactly k={k} medoids on plot (picked {len(self._picked_medoid_indices)}).",
                    )
                    self.append_log(
                        f"Invalid picked medoids count: expected {k}, got {len(self._picked_medoid_indices)}."
                    )
                    return
                init_medoids = np.array(self._picked_medoid_indices, dtype=int)
                self.append_log(
                    f"Using manually picked medoids: {init_medoids.tolist()}"
                )
            else:
                rng = np.random.default_rng(int(self.seed_spin.value()))
                init_medoids = rng.choice(n, size=k, replace=False)
                self.append_log(
                    f"Using automatic initial medoids: {init_medoids.tolist()}"
                )

            labels, medoids = kmedoids_pam_axial(
                vectors_xyz,
                k=k,
                maxiter=100,
                init_medoids=init_medoids,
                log=self.append_log,
            )

            cmap = plt.get_cmap("tab10")
            for ci in range(k):
                idx = np.where(labels == ci)[0]
                color = cmap(ci % 10)

                if plot_poles:
                    if is_planes:
                        self.ax.pole(
                            trends[idx],
                            plunges[idx],
                            ".",
                            color=color,
                            markersize=6,
                            alpha=0.9,
                        )
                    else:
                        self.ax.line(
                            plunges[idx],
                            trends[idx],
                            ".",
                            color=color,
                            markersize=6,
                            alpha=0.9,
                        )

                if (
                    is_planes
                    and plot_gcs
                    and data.get("strikes_deg") is not None
                    and data.get("dips_deg") is not None
                ):
                    self.ax.plane(
                        data["strikes_deg"][idx],
                        data["dips_deg"][idx],
                        color=color,
                        linewidth=1.0,
                        alpha=0.45,
                    )

            for ci, mi in enumerate(medoids):
                color = cmap(ci % 10)
                if is_planes:
                    self.ax.pole(
                        [trends[mi]],
                        [plunges[mi]],
                        marker="*",
                        color=color,
                        markersize=14,
                        markeredgecolor="k",
                    )
                else:
                    self.ax.line(
                        [plunges[mi]],
                        [trends[mi]],
                        marker="*",
                        color=color,
                        markersize=14,
                        markeredgecolor="k",
                    )

            for ci in range(k):
                idx = np.where(labels == ci)[0]
                if idx.size == 0:
                    continue
                vmf_c = vmf_mean_axial(vectors_xyz[idx], log=self.append_log)
                mean_xyz = vmf_c["mean_xyz"]
                m_tr, m_pl = (
                    lmn_to_trend_plunge(mean_xyz)
                    if np.isfinite(mean_xyz).all()
                    else (float("nan"), float("nan"))
                )
                cluster_summary.append(
                    (
                        ci,
                        int(idx.size),
                        int(medoids[ci]),
                        f"{m_tr:.2f}",
                        f"{m_pl:.2f}",
                        f"{vmf_c['Rbar']:.3f}",
                        f"{vmf_c['kappa']:.3g}",
                    )
                )

            self.append_log("k-medoids clustering completed.")

            # self.ax.set_title(f"{'Planes' if is_planes else 'Lines'} (n={n})")
            self.canvas.draw()
            self.append_log("Plot updated.")

            # cache projected XY for picking
            try:
                x, y = stereonet_math.line(plunges, trends)
                self._last_projected = np.column_stack([x, y]).astype(float)
                self.append_log("Projected plot coordinates cached for medoid picking.")
            except Exception:
                self._last_projected = None
                self.append_log("Could not cache projected plot coordinates.")

            self._last_vectors_xyz = vectors_xyz

            if cluster_summary:
                self._log_section(
                    "Clustering summary:",
                    [
                        (
                            f"Cluster {cluster_id}: n={count}, medoid index={medoid_index}, "
                            f"VMF mean trend={mean_trend}, plunge={mean_plunge}, "
                            f"R̄={rbar}, κ≈{kappa}"
                        )
                        for (
                        cluster_id,
                        count,
                        medoid_index,
                        mean_trend,
                        mean_plunge,
                        rbar,
                        kappa,
                    ) in cluster_summary
                    ],
                )
            else:
                self.append_log("Clustering summary: no non-empty clusters to report.")

            self.append_log("Hypothesis tests summary: not implemented yet.")

            # save only on request
            if self.chk_save.isChecked():
                out_dir = self.out_dir.text().strip()
                if not out_dir or not os.path.isdir(out_dir):
                    QMessageBox.warning(
                        self,
                        "qAttitude",
                        "Save enabled, but output directory is invalid.",
                    )
                    self.append_log("Save requested, but output directory is invalid.")
                    return

                out_path = os.path.join(out_dir, "stereonet.png")
                self.fig.savefig(
                    out_path,
                    dpi=200,
                    bbox_inches="tight",
                )
                self.append_log(f"Figure saved to: {out_path}")

                log_path = os.path.join(out_dir, "stereonet_log.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(self.log_output.toPlainText())
                    if not self.log_output.toPlainText().endswith("\n"):
                        f.write("\n")
                self.append_log(f"Log saved to: {log_path}")

            self.append_log("Analysis completed successfully.")

        except Exception as e:
            tb = traceback.format_exc()
            self.append_log(f"ERROR: {type(e).__name__}: {e}")
            QMessageBox.critical(
                self, "qAttitude", f"Error: {type(e).__name__}: {e}\n\n{tb}"
            )
