from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image, UnidentifiedImageError


def show_clustering_samples(
    labels: Sequence[int],
    probabilities: Sequence[float] | np.ndarray | None = None,
    image_paths: Sequence[str | Path] | None = None,
    x: int = 5,
    include_noise: bool = False,
    base_dir: str | Path | None = None,
    title: str = "Samples per cluster",
    image_size: float = 2.2,
    show: bool = True,
    save_path: str | Path | None = None,
    score_name: str = "p",
) -> tuple[Figure, np.ndarray]:
    """
    Show the first x images per cluster.

    If probabilities is provided, samples are sorted by descending score.

    probabilities can be:
    - None: no sorting by probability/score
    - 1D array: HDBSCAN probabilities, local purity, confidence score, etc.
    - 2D array: GMM membership probabilities
    """

    labels_array = np.asarray(labels)

    if image_paths is None:
        raise ValueError("image_paths must not be None.")

    image_paths = list(image_paths)

    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array.")

    if len(image_paths) != labels_array.shape[0]:
        raise ValueError("image_paths and labels must have the same length.")

    if x < 1:
        raise ValueError("x must be at least 1.")

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    has_scores = probabilities is not None

    assigned_scores = _assigned_cluster_probabilities(
        probabilities,
        labels_array,
    )

    cluster_labels = _cluster_labels(labels_array, include_noise)

    if not cluster_labels:
        raise ValueError("No clusters to display.")

    figure, axes = plt.subplots(
        len(cluster_labels),
        x,
        figsize=(image_size * x, image_size * len(cluster_labels)),
        squeeze=False,
    )

    for row, cluster_label in enumerate(cluster_labels):

        cluster_indices = np.flatnonzero(labels_array == cluster_label)

        if has_scores:
            sorted_indices = cluster_indices[
                np.argsort(
                    -assigned_scores[cluster_indices],
                    kind="stable",
                )
            ]
        else:
            sorted_indices = cluster_indices

        selected_indices = sorted_indices[:x]

        for column in range(x):

            axis = axes[row, column]
            axis.axis("off")

            if column == 0:
                axis.set_ylabel(
                    _format_cluster_label(cluster_label),
                    rotation=0,
                    labelpad=45,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

            if column >= len(selected_indices):
                continue

            sample_index = int(selected_indices[column])
            image_path = _resolve_path(
                image_paths[sample_index],
                base_dir,
            )

            _show_image(axis, image_path)

            if has_scores:
                score = float(assigned_scores[sample_index])
                axis.set_title(
                    f"idx={sample_index}\n{score_name}={score:.3f}",
                    fontsize=9,
                )
            else:
                axis.set_title(
                    f"idx={sample_index}",
                    fontsize=9,
                )

    figure.suptitle(title)
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(Path(save_path), dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return figure, axes


def _assigned_cluster_probabilities(
    probabilities: Sequence[float] | np.ndarray | None,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Restituisce uno score per ogni sample.

    Se probabilities è None, restituisce un array di 1.
    Questo permette di usare la funzione anche con Leiden.
    """

    if probabilities is None:
        return np.ones(labels.shape[0], dtype=float)

    probabilities_array = np.asarray(probabilities, dtype=float)

    if probabilities_array.ndim == 1:

        if probabilities_array.shape[0] != labels.shape[0]:
            raise ValueError(
                "probabilities and labels must have the same length."
            )

        return probabilities_array

    if probabilities_array.ndim == 2:

        if probabilities_array.shape[0] != labels.shape[0]:
            raise ValueError(
                "probabilities and labels must have the same length."
            )

        if np.any((labels >= probabilities_array.shape[1]) & (labels != -1)):
            raise ValueError(
                "2D probabilities require non-noise labels to be valid "
                "column indices."
            )

        assigned = np.zeros(labels.shape[0], dtype=float)

        non_noise_mask = labels != -1

        row_indices = np.arange(labels.shape[0])[non_noise_mask]
        column_indices = labels[non_noise_mask].astype(int)

        assigned[non_noise_mask] = probabilities_array[
            row_indices,
            column_indices,
        ]

        return assigned

    raise ValueError("probabilities must be None, a 1D array or a 2D array.")


def _cluster_labels(labels: np.ndarray, include_noise: bool) -> list[int]:
    unique_labels = [int(label) for label in np.unique(labels)]
    if not include_noise:
        unique_labels = [label for label in unique_labels if label != -1]
    return sorted(unique_labels, key=lambda label: (label == -1, label))


def _resolve_path(path: str | Path, base_dir: str | Path | None) -> Path:
    image_path = Path(path)
    if base_dir is not None and not image_path.is_absolute():
        image_path = Path(base_dir) / image_path
    return image_path


def _show_image(axis: Axes, image_path: Path) -> None:
    if not image_path.exists():
        axis.text(0.5, 0.5, "missing", ha="center", va="center")
        return

    try:
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
    except (OSError, UnidentifiedImageError):
        axis.text(0.5, 0.5, "unreadable", ha="center", va="center")


def _format_cluster_label(label: int) -> str:
    return "Noise" if label == -1 else f"Cluster {label}"


__all__ = ["show_clustering_samples"]
