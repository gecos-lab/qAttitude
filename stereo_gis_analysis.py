# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

import numpy as np
import pandas as pd
from kmedoids import KMedoids
from sklearn.cluster import KMeans
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


def mirror_to_other_hemisphere(l, m, n):
    return -l, -m, -n


def vmf_mean_axial(vector_xyz: np.ndarray, log=None) -> dict:
    # _________________________________________________________________
    if vector_xyz.shape[0] == 0:
        raise QgsProcessingException("No vectors for VMF.")

    _log(log, f"VMF: input vectors shape = {vector_xyz.shape}")

    V = vector_xyz[["l", "m", "n"]].values
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

    V = vector_xyz[["l", "m", "n"]].values
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    T = (V.T @ V) / V.shape[0]
    evals, evecs = np.linalg.eigh(T)  # ascending
    idx = np.argsort(evals)[::-1]  # descending
    evals = evals[idx]
    evecs = evecs[:, idx]

    beta = evecs[:, 0]
    beta = beta / np.linalg.norm(beta)

    _log(log, f"Bingham: eigenvalues = {np.array2string(evals, precision=6)}")

    return {"axes_xyz": evecs, "evals": evals, "beta_axis_xyz": beta}


def axial_angular_distance(u: np.ndarray, v: np.ndarray) -> float:
    dot = abs(np.dot(u, v))
    angle_rad = np.arccos(dot)
    return angle_rad


def kmedoids_axial(
    data,
    k: int,
    maxiter: int = 100,
    init_medoids: np.ndarray | None = None,
    log=None,
):
    """
    Calculate k-medoids clustering for axial orientation analysis, using the kmedoids library
    with sklearn-compatible API.
    https://github.com/kno10/python-kmedoids
    https://python-kmedoids.readthedocs.io/en/latest/
    Parameters:
        n_clusters (int) – The number of clusters to form (maximum number of clusters if method=”dynmsc”)

        metric (string, default: 'precomputed') – It is recommended to use ‘precomputed’, in particular when
        experimenting with different n_clusters. If you have sklearn installed, you may pass any metric
        supported by sklearn.metrics.pairwise_distances.

        metric_params (dict, default=None) – Additional keyword arguments for the metric function.

        method (string, "fasterpam" (default), "fastpam1", "pam", "alternate", "fastermsc", "fastmsc",
        "pamsil" or "pammedsil") – Which algorithm to use

        init (string, "random" (default), "first" or "build") – initialization method

        max_iter (int) – Specify the maximum number of iterations when fitting

        random_state (int, RandomState instance or None) – random seed if no medoids are given

    Variables:
        cluster_centers – None for ‘precomputed’

        medoid_indices – The indices of the medoid rows in X

        labels – Labels of each point

        inertia – Sum of distances of samples to their closest cluster center
    """
    # initialize
    n = data.shape[0]
    if n == 0:
        raise QgsProcessingException("No vectors to cluster.")
    if not (1 <= k <= n):
        raise QgsProcessingException(f"k must be in [1, {n}]")
    _log(log, f"k-medoids: n={n}, k={k}, maxiter={maxiter}")
    init = "random"  # Supported inits are 'random', 'first' and 'build'.

    # run clustering
    vectors = data[["l", "m", "n"]].values
    kmedoids = KMedoids(n_clusters=k * 2, init=init, metric=axial_angular_distance).fit(
        vectors
    )

    # write results in dataframe
    data["cluster"] = kmedoids.labels_

    # show results
    _log(log, f"medoids indices: {kmedoids.medoid_indices_}")
    _log(log, f"medoids: {kmedoids.cluster_centers_}")
    _log(log, f"data['cluster'].unique(): {data['cluster'].unique()}")

    medoids = pd.DataFrame(kmedoids.cluster_centers_, columns=["l", "m", "n"])
    medoids["data_index"] = kmedoids.medoid_indices_
    # Uses .iloc[row_positions, column_positions] to select by position
    # medoids['data_index'] contains the row positions (0, 1, 2, ...)
    # data.columns.get_indexer([...]) gets the column positions
    # .values converts to numpy array and assigns to medoids
    medoids[["plunge", "trend", "lower_hemi", "cluster"]] = data.iloc[
        medoids["data_index"],
        data.columns.get_indexer(["plunge", "trend", "lower_hemi", "cluster"]),
    ].values

    _log(log, f"all medoids: {medoids.to_string()}")

    # keep clusters with medoid pointing downwards only
    medoids = medoids.loc[medoids["lower_hemi"] == True].reset_index(drop=True)
    _log(log, f"n kept medoids - downwards only: {medoids.shape[0]}")
    _log(log, f"kept medoids - lower hemisphere: {medoids.to_string()}")

    return data, medoids


def kmeans(
    data,
    nn_clusters=1,
    init="k-means++",
    max_iter=100,
    random_state=None,
    log=None,
):
    """
    Calculate k-medoids clustering for axial orientation analysis, using the scikit-learn library.
    https://github.com/kno10/python-kmedoids
    https://python-kmedoids.readthedocs.io/en/latest/
    Method for initialization:
        ‘k-means++’ : selects initial cluster centroids using sampling based on an empirical probability
        distribution of the points’ contribution to the overall inertia. This technique speeds up convergence.
        The algorithm implemented is “greedy k-means++”. It differs from the vanilla k-means++ by making
        several trials at each sampling step and choosing the best centroid among them.

        ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.

    Variables:
        cluster_centers – None for ‘precomputed’

        medoid_indices – The indices of the medoid rows in X

        labels – Labels of each point

        inertia – Sum of distances of samples to their closest cluster center
    """
    # initialize
    n = data.shape[0]
    if n == 0:
        raise QgsProcessingException("No vectors to cluster.")
    if not (1 <= nn_clusters <= n):
        raise QgsProcessingException(f"k must be in [1, {n}]")
    _log(log, f"k-means: n={n}, k={nn_clusters}, maxiter={max_iter}")

    # run clustering
    vectors = data[["l", "m", "n"]].values
    kmeans = KMeans(
        n_clusters=nn_clusters * 2,
        init=init,
        n_init="auto",
        max_iter=max_iter,
        tol=0.0001,
        verbose=0,
        random_state=random_state,
        copy_x=True,
        algorithm="lloyd",
    ).fit(vectors)

    # write results in dataframe
    data["cluster"] = kmeans.labels_

    # show results
    _log(log, f"means: {kmeans.cluster_centers_}")
    _log(log, f"data['cluster'].unique(): {data['cluster'].unique()}")

    means = pd.DataFrame(kmeans.cluster_centers_, columns=["l", "m", "n"])
    means["cluster"] = np.arange(means.shape[0])
    means["trend"], means["plunge"] = lmn_to_trend_plunge(
        means["l"], means["m"], means["n"]
    )
    means["lower_hemi"] = means["n"] <= 0
    
    # Add strike and dip for plotting great circles
    means['dip'] = 90.0 - means['plunge']
    dipdir = wrap360(means['trend'] - 180)
    means['strike'] = dipdir2strike(dipdir)

    _log(log, f"all means: {means.to_string()}")

    # keep clusters with medoid pointing downwards only
    means = means.loc[means["lower_hemi"] == True].reset_index(drop=True)
    _log(log, f"n kept means - downwards only: {means.shape[0]}")
    _log(log, f"kept means - lower hemisphere: {means.to_string()}")

    return data, means


def read_orientations_from_layer_selection(
    layer, is_planes: bool, field1: str, field2: str, log=None
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

    # now duplicate the data to perform axially symmetric orientation analysis
    data_upper = data_lower.copy()
    data_upper["l"] = -data_upper["l"]
    data_upper["m"] = -data_upper["m"]
    data_upper["n"] = -data_upper["n"]
    data_upper["lower_hemi"] = False

    data = pd.concat([data_lower, data_upper]).reset_index(drop=True)

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
