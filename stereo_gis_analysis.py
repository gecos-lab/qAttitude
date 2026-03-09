# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

import numpy as np
import pandas as pd
from kmedoids import KMedoids
from qgis.core import QgsProcessingException


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
    trend_rad = np.arctan2(l, m)
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


def mirror_to_other_hemisphere(l, m, n):
    return -l, -m, -n


def vmf_mean_axial(vector_xyz: np.ndarray, log=None) -> dict:
    # _________________________________________________________________
    if vector_xyz.shape[0] == 0:
        raise QgsProcessingException("No vectors for VMF.")

    _log(log, f"VMF: input vectors shape = {vector_xyz.shape}")

    V = np.array([mirror_to_other_hemisphere(v) for v in vector_xyz], dtype=float)
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


def bingham_principal_axes_axial(vector_xyz: np.ndarray, log=None) -> dict:
    # _________________________________________________________________
    if vector_xyz.shape[0] == 0:
        raise QgsProcessingException("No vectors for Bingham summary.")

    _log(log, f"Bingham: input vectors shape = {vector_xyz.shape}")

    V = np.array([mirror_to_other_hemisphere(v) for v in vector_xyz], dtype=float)
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
    vector_xyz: np.ndarray,
    k: int,
    maxiter: int = 100,
    init_medoids: np.ndarray | None = None,
    log=None,
):

    #     # NEW _________________________________________________________________

    n = vector_xyz.shape[0]
    if n == 0:
        raise QgsProcessingException("No vectors to cluster.")
    if not (1 <= k <= n):
        raise QgsProcessingException(f"k must be in [1, {n}]")

    _log(log, f"k-medoids: n={n}, k={k}, maxiter={maxiter}")

    vectors_both = np.vstack([vector_xyz, mirror_to_other_hemisphere(vector_xyz)])

    init = 'random'  # Supported inits are 'random', 'first' and 'build'.
    _log(log, "1")
    kmedoids = KMedoids(n_clusters=k, init='random', metric=axial_angular_distance).fit(vectors_both)

    labels = kmedoids.labels_[:n]
    medoids = kmedoids.medoid_indices_

    _log(log, f"k-medoids: final medoids = {medoids.tolist()}")

    # OLD _________________________________________________________________

    # n = vector_xyz.shape[0]
    # if n == 0:
    #     raise QgsProcessingException("No vectors to cluster.")
    # if not (1 <= k <= n):
    #     raise QgsProcessingException(f"k must be in [1, {n}]")
    #
    # _log(log, f"k-medoids: n={n}, k={k}, maxiter={maxiter}")
    #
    # V = np.array([v / np.linalg.norm(v) for v in vector_xyz], dtype=float)
    #
    # D = np.zeros((n, n), dtype=float)
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         d = axial_angular_distance(V[i], V[j])
    #         D[i, j] = d
    #         D[j, i] = d
    #
    # if init_medoids is None:
    #     medoids = np.arange(k, dtype=int)
    #     _log(log, f"k-medoids: default initial medoids = {medoids.tolist()}")
    # else:
    #     medoids = np.array(init_medoids, dtype=int)
    #     if medoids.size != k:
    #         raise QgsProcessingException("init_medoids must have length k")
    #     _log(log, f"k-medoids: provided initial medoids = {medoids.tolist()}")
    #
    # labels = np.zeros(n, dtype=int)
    #
    # for iteration in range(int(maxiter)):
    #     dist_to_m = D[:, medoids]
    #     labels_new = np.argmin(dist_to_m, axis=1)
    #
    #     medoids_new = medoids.copy()
    #     for ci in range(k):
    #         idx = np.where(labels_new == ci)[0]
    #         if idx.size == 0:
    #             continue
    #         intra = D[np.ix_(idx, idx)]
    #         costs = intra.sum(axis=1)
    #         medoids_new[ci] = int(idx[np.argmin(costs)])
    #
    #     if np.array_equal(medoids_new, medoids) and np.array_equal(labels_new, labels):
    #         labels = labels_new
    #         medoids = medoids_new
    #         _log(log, f"k-medoids: converged at iteration {iteration + 1}")
    #         break
    #
    #     labels = labels_new
    #     medoids = medoids_new
    # else:
    #     _log(log, "k-medoids: reached maximum iterations without early convergence")
    #
    # _log(log, f"k-medoids: final medoids = {medoids.tolist()}")

    return labels, medoids


def read_orientations_from_layer_selection(
    layer, is_planes: bool, field1: str, field2: str, log=None
) -> dict:
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
        except Exception:
            invalid_count += 1
            continue
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
        data_lower['dip'] = in_list_1
        data_lower['dipdir'] = in_list_2
        data_lower['strike'] = dipdir2strike(in_list_2)
        data_lower['plunge'] = 90.0 - in_list_1
        data_lower['trend'] = wrap360(in_list_2 + 180)
        l, m, n = dipdir_dip_to_pole_lmn(in_list_2, in_list_1)
        data_lower['l'] = l
        data_lower['m'] = m
        data_lower['n'] = n
        data_lower['label'] = 0
        data_lower['lower_hemi'] = True
    else:
        data_lower['plunge'] = in_list_1
        data_lower['trend'] = in_list_2
        l, m, n = trend_plunge_to_lmn(in_list_2, in_list_1)
        data_lower['l'] = l
        data_lower['m'] = m
        data_lower['n'] = n
        data_lower['label'] = 0
        data_lower['lower_hemi'] = True

    # now duplicate the data to perform axially symmetric orientation analysis
    data_upper = data_lower.copy()
    data_upper['l'] = -data_upper['l']
    data_upper['m'] = -data_upper['m']
    data_upper['n'] = -data_upper['n']
    data_upper['lower_hemi'] = False

    data_both = pd.concat([data_lower, data_upper])

    _log(
        log,
        f"Orientation reading complete: valid={data_lower.shape[0]}, invalid/skipped={invalid_count}.",
    )
    _log(
        log,
        data_both.head(),
    )
    return data_both
