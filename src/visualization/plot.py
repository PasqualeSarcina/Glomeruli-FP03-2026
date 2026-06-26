from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


ProjectionMode = Literal["2d", "3d"]


def plot_clustering_on_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    mode: ProjectionMode = "2d",
    title: str = "Clustering su UMAP",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    save_path: str | Path | None = None,
    show: bool = True,
    point_size: float = 20.0,
    alpha: float = 0.85,
    include_legend: bool = True,
) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Calculate UMAP and plot clustering labels on 2D projections.

    mode="2d" fits a 2-component UMAP and plots components 1-2.
    mode="3d" fits a 3-component UMAP and plots components 1-2 and 2-3.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same length.")
    if mode not in {"2d", "3d"}:
        raise ValueError("mode must be either '2d' or '3d'.")
    if not 2 <= n_neighbors < embeddings.shape[0]:
        raise ValueError("n_neighbors must be between 2 and n_samples - 1.")
    if not 0.0 <= min_dist <= 0.8:
        raise ValueError("min_dist must be between 0.0 and 0.8.")

    n_components = 2 if mode == "2d" else 3
    coordinates = _fit_umap(
        embeddings,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    projections = [(0, 1)] if mode == "2d" else [(0, 1), (1, 2)]
    figure, axes_array = plt.subplots(
        1,
        len(projections),
        figsize=(6 * len(projections), 5),
        squeeze=False,
    )
    axes = tuple(axes_array.ravel())

    unique_labels = sorted(np.unique(labels), key=lambda label: (label == -1, label))
    color_by_label = _build_label_colors(unique_labels)

    for axis, (x_component, y_component) in zip(axes, projections):
        for label in unique_labels:
            mask = labels == label
            axis.scatter(
                coordinates[mask, x_component],
                coordinates[mask, y_component],
                s=point_size,
                alpha=alpha,
                c=[color_by_label[label]],
                label=_format_label(label),
                edgecolors="none",
            )

        axis.set_xlabel(f"UMAP {x_component + 1}")
        axis.set_ylabel(f"UMAP {y_component + 1}")
        axis.set_title(f"Componenti {x_component + 1}-{y_component + 1}")
        axis.grid(True, alpha=0.2)

    figure.suptitle(title)
    if include_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color_by_label[label],
                markeredgecolor="none",
                markersize=7,
                label=_format_label(label),
            )
            for label in unique_labels
        ]
        axes[-1].legend(
            handles=handles,
            title="Cluster",
            loc="best",
            fontsize="small",
        )

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(Path(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return figure, axes


def _fit_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def _build_label_colors(labels: list[int | np.integer]) -> dict[int | np.integer, tuple]:
    cmap = plt.get_cmap("tab20")
    cluster_labels = [label for label in labels if label != -1]
    colors = {
        label: cmap(index % cmap.N)
        for index, label in enumerate(cluster_labels)
    }
    if -1 in labels:
        colors[-1] = (0.55, 0.55, 0.55, 0.75)
    return colors


def _format_label(label: int | np.integer) -> str:
    return "Noise (-1)" if label == -1 else f"Cluster {label}"


globals()["plot_clustering_on-umap"] = plot_clustering_on_umap
