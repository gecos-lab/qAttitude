# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

import matplotlib.pyplot as plt
import mplstereonet
from mplstereonet import stereonet_math
import numpy as np


def plot_base_stereonet(
    ax,
    data,
    plot_poles=True,
    plot_gcs=False,
    show_contours=False,
    contour_levels=10,
    point_color="k",
    contour_cmap="Greys",
):
    """
    Plots the main stereonet data.
    """
    ax.clear()
    ax.grid(True, zorder=0, alpha=0.5)
    ax.grid(kind="polar", zorder=0, alpha=0.5)

    if data.empty:
        return

    query = "lower_hemi == True"

    # Plot contours
    if show_contours:
        ax.density_contourf(
            data.query(query)["plunge"],
            data.query(query)["trend"],
            measurement="lines",
            cmap=contour_cmap,
            alpha=0.6,
            levels=int(contour_levels),
            zorder=1,
        )

    # Plot great circles
    if plot_gcs:
        ax.plane(
            data.query(query)["strike"].to_list(),
            data.query(query)["dip"].to_list(),
            color=point_color,
            linewidth=1,
            alpha=0.5,
            zorder=2,
        )

    # Plot individual data points
    if plot_poles:
        ax.line(
            data.query(query)["plunge"].to_list(),
            data.query(query)["trend"].to_list(),
            ".",
            color=point_color,
            markersize=4,
            alpha=1,
            zorder=3,
        )

    # Cache projected XY for picking
    try:
        x, y = stereonet_math.line(
            data.query(query)["plunge"].to_list(), data.query(query)["trend"].to_list()
        )
        return np.column_stack([x, y]).astype(float)
    except Exception:
        return None


def plot_clusters(ax, data_with_clusters, means, plot_poles, plot_gcs):
    cmap = plt.get_cmap("tab10")
    for cluster in means["cluster"].to_list():
        query = "lower_hemi == True and cluster == " + str(cluster)
        color = cmap(cluster % 10)

        if plot_poles:
            ax.line(
                data_with_clusters.query(query)["plunge"].to_list(),
                data_with_clusters.query(query)["trend"].to_list(),
                ".",
                color=color,
                markersize=6,
                alpha=0.9,
                zorder=4,
            )
            ax.line(
                means.query(f"cluster == {cluster}")["plunge"].to_list(),
                means.query(f"cluster == {cluster}")["trend"].to_list(),
                marker="*",
                color=color,
                markersize=14,
                markeredgecolor="k",
                zorder=5,
            )

        if plot_gcs:
            ax.plane(
                data_with_clusters.query(query)["strike"].to_list(),
                data_with_clusters.query(query)["dip"].to_list(),
                color=color,
                linewidth=1.0,
                alpha=0.45,
                zorder=4,
            )
            ax.plane(
                means.query(f"cluster == {cluster}")["strike"].to_list(),
                means.query(f"cluster == {cluster}")["dip"].to_list(),
                color=color,
                linewidth=4.0,
                alpha=1,
                zorder=5,
            )


def plot_vmf(ax, m):
    tr, pl = lmn_to_trend_plunge(m[0], m[1], m[2])
    ax.line(pl, tr, "r*", markersize=12, zorder=5)
    return tr, pl


def plot_bingham(ax, beta):
    tr, pl = lmn_to_trend_plunge(beta[0], beta[1], beta[2])
    ax.line(pl, tr, "D", color="#1f77b4", markersize=7, zorder=5)

    dipdir = wrap360(tr + 180)
    dip = 90.0 - pl
    strike = dipdir2strike(dipdir)
    ax.plane(strike, dip, color="#1f77b4", linewidth=1.4, alpha=0.85, zorder=5)
    return tr, pl
