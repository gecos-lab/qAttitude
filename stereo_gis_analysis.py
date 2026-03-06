# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

import math
import numpy as np
from qgis.core import QgsProcessingException


def _log(log, message: str) -> None:
    if log is not None:
        log(message)


def wrap360(deg: float) -> float:
    deg = deg % 360.0
    return deg


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def trend_plunge_to_xyz(trend_deg: float, plunge_deg: float) -> np.ndarray:
    trend_rad = deg2rad(trend_deg)
    plunge_rad = deg2rad(plunge_deg)
    x = np.cos(plunge_rad) * np.cos(trend_rad)  # East
    y = np.cos(plunge_rad) * np.sin(trend_rad)  # North
    z = -np.sin(plunge_rad)  # Up (negative is down-plunge)
    v = np.array([x, y, z], dtype=float)
    return v


def xyz_to_trend_plunge(v: np.ndarray) -> tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    plunge_rad = np.arcsin(-z)
    trend_rad = np.arctan2(x, y)
    plunge_deg = wrap360(rad2deg(plunge_rad))
    trend_deg = wrap360(rad2deg(trend_rad))
    return trend_deg, plunge_deg


def dipdir_dip_to_pole_xyz(dipdir_deg: float, dip_deg: float) -> np.ndarray:
    pole_trend_deg = wrap360(dipdir_deg + 180)
    pole_plunge_deg = 90.0 - dip_deg
    return trend_plunge_to_xyz(pole_trend_deg, pole_plunge_deg)


def dipdir2strike(dipdir_deg: float) -> float:
    strike_deg = wrap360(dipdir_deg - 90.0)
    return strike_deg


def strike2dipdir(strike_deg: float) -> float:
    dipdir_deg = wrap360(strike_deg + 90.0)
    return dipdir_deg


def mirror_to_other_hemisphere(v: np.ndarray) -> np.ndarray:
    return -v


def vmf_mean_axial(vectors_xyz: np.ndarray, log=None) -> dict:
    # _________________________________________________________________
    if vectors_xyz.shape[0] == 0:
        raise QgsProcessingException("No vectors for VMF.")

    _log(log, f"VMF: input vectors shape = {vectors_xyz.shape}")

    vectors_both = np.vstack([vectors_xyz, mirror_to_other_hemisphere(vectors_xyz)])

    V = np.array([mirror_to_other_hemisphere(v) for v in vectors_xyz], dtype=float)
    S = V.sum(axis=0)
    S_norm = float(np.linalg.norm(S))
    if S_norm == 0.0:
        return {
            "mean_xyz": np.array([np.nan, np.nan, np.nan]),
            "Rbar": 0.0,
            "kappa": float("nan"),
        }

    mean_xyz = S / S_norm
    n = V.shape[0]
    Rbar = S_norm / n

    denom = max(1e-12, 1.0 - Rbar * Rbar)
    kappa = (Rbar * (3.0 - Rbar * Rbar)) / denom

    _log(log, f"VMF: Rbar = {Rbar:.6f}, kappa ≈ {kappa:.6g}")

    return {"mean_xyz": mean_xyz, "Rbar": Rbar, "kappa": kappa}


def bingham_principal_axes_axial(vectors_xyz: np.ndarray, log=None) -> dict:
    # _________________________________________________________________
    if vectors_xyz.shape[0] == 0:
        raise QgsProcessingException("No vectors for Bingham summary.")

    _log(log, f"Bingham: input vectors shape = {vectors_xyz.shape}")

    V = np.array([mirror_to_other_hemisphere(v) for v in vectors_xyz], dtype=float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    T = (V.T @ V) / V.shape[0]
    evals, evecs = np.linalg.eigh(T)  # ascending
    idx = np.argsort(evals)[::-1]  # descending
    evals = evals[idx]
    evecs = evecs[:, idx]

    beta = evecs[:, 0]
    beta = beta / np.linalg.norm(beta)
    beta = mirror_to_other_hemisphere(beta)

    _log(log, f"Bingham: eigenvalues = {np.array2string(evals, precision=6)}")

    return {"axes_xyz": evecs, "evals": evals, "beta_axis_xyz": beta}


def axial_angular_distance(u: np.ndarray, v: np.ndarray) -> float:
    dot = abs(np.dot(u, v))
    angle_rad = np.arccos(dot)
    angle_deg = rad2deg(angle_rad)
    return angle_deg


def kmedoids_pam_axial(
    vectors_xyz: np.ndarray,
    k: int,
    maxiter: int = 100,
    init_medoids: np.ndarray | None = None,
    log=None,
):
    # _________________________________________________________________
    n = vectors_xyz.shape[0]
    if n == 0:
        raise QgsProcessingException("No vectors to cluster.")
    if not (1 <= k <= n):
        raise QgsProcessingException(f"k must be in [1, {n}]")

    _log(log, f"k-medoids: n={n}, k={k}, maxiter={maxiter}")

    V = np.array([v / np.linalg.norm(v) for v in vectors_xyz], dtype=float)

    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = axial_angular_distance(V[i], V[j])
            D[i, j] = d
            D[j, i] = d

    if init_medoids is None:
        medoids = np.arange(k, dtype=int)
        _log(log, f"k-medoids: default initial medoids = {medoids.tolist()}")
    else:
        medoids = np.array(init_medoids, dtype=int)
        if medoids.size != k:
            raise QgsProcessingException("init_medoids must have length k")
        _log(log, f"k-medoids: provided initial medoids = {medoids.tolist()}")

    labels = np.zeros(n, dtype=int)

    for iteration in range(int(maxiter)):
        dist_to_m = D[:, medoids]
        labels_new = np.argmin(dist_to_m, axis=1)

        medoids_new = medoids.copy()
        for ci in range(k):
            idx = np.where(labels_new == ci)[0]
            if idx.size == 0:
                continue
            intra = D[np.ix_(idx, idx)]
            costs = intra.sum(axis=1)
            medoids_new[ci] = int(idx[np.argmin(costs)])

        if np.array_equal(medoids_new, medoids) and np.array_equal(labels_new, labels):
            labels = labels_new
            medoids = medoids_new
            _log(log, f"k-medoids: converged at iteration {iteration + 1}")
            break

        labels = labels_new
        medoids = medoids_new
    else:
        _log(log, "k-medoids: reached maximum iterations without early convergence")

    _log(log, f"k-medoids: final medoids = {medoids.tolist()}")

    return labels, medoids


def read_orientations_from_layer_selection(
    layer, is_planes: bool, field1: str, field2: str, log=None
) -> dict:
    """
    Reads orientations from the layer.
    Uses selected features if any are selected; otherwise uses all features.

    Planes: field1=dip,    field2=dipdir -> vectors are POLES.
    Lines:  field1=plunge, field2=trend  -> vectors are lines.
    """
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

    vectors = []
    strikes = []
    dips = []
    dipdirs = []
    trends = []
    plunges = []

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
        except Exception:
            invalid_count += 1
            continue
        if not (0.0 <= v1 <= 90.0):
            invalid_count += 1
            continue
        if not (0.0 <= v2 <= 360.0):
            invalid_count += 1
            continue

        if is_planes:
            dip = v1
            dipdir = v2
            pole_trend = wrap360(dipdir + 180)
            pole_plunge = 90.0 - dip
            pole_xyz = dipdir_dip_to_pole_xyz(dipdir, dip)

            vectors.append(pole_xyz)
            dips.append(dip)
            dipdirs.append(dipdir)
            strikes.append(dipdir2strike(dipdir))
            trends.append(pole_trend)
            plunges.append(pole_plunge)
        else:
            plunge = v1
            trend = v2
            line_xyz = trend_plunge_to_xyz(trend, plunge)
            vectors.append(line_xyz)
            trends.append(trend)
            plunges.append(plunge)

    vectors_xyz = np.asarray(vectors, dtype=float)

    _log(
        log,
        f"Orientation reading complete: valid={vectors_xyz.shape[0]}, invalid/skipped={invalid_count}.",
    )

    return {
        "vectors_xyz": vectors_xyz,
        "strikes_deg": np.asarray(strikes, dtype=float) if strikes else None,
        "dipdirs_deg": np.asarray(dipdirs, dtype=float) if dipdirs else None,
        "dips_deg": np.asarray(dips, dtype=float) if dips else None,
        "trends_deg": np.asarray(trends, dtype=float),
        "plunges_deg": np.asarray(plunges, dtype=float),
    }
