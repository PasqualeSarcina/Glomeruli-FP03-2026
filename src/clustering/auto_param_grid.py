from __future__ import annotations

from itertools import product

import numpy as np


def _make_set(values, min_value=None, max_value=None):
    """
    Converte una lista di candidati in una lista di interi unici,
    filtrando i valori fuori dal range [min_value, max_value].

    Restituisce una lista, non un set Python, per preservare l'ordine
    deterministico dei candidati.
    """

    clean_values = []

    for value in values:
        value = int(round(float(value)))

        if min_value is not None and value < min_value:
            continue

        if max_value is not None and value > max_value:
            continue

        if value not in clean_values:
            clean_values.append(value)

    return clean_values


def _deterministic_grid_sample(param_grid, max_size):
    if max_size is None or len(param_grid) <= max_size:
        return param_grid

    max_size = int(max_size)

    if max_size < 1:
        raise ValueError("max_auto_param_combinations deve essere >= 1.")

    selected_indices = np.linspace(
        0,
        len(param_grid) - 1,
        num=max_size,
        dtype=int,
    )

    selected_indices = sorted(set(int(index) for index in selected_indices))

    return [param_grid[index] for index in selected_indices]


def make_auto_param_grid(
    n_samples,
    n_features,
    max_param_combinations=240,
):
    """
    Crea una griglia UMAP + HDBSCAN guidata dalla dimensione del dataset.

    La griglia non contiene un target implicito sul numero di cluster: include
    impostazioni locali/fini e globali/conservative, cosi' la selezione puo'
    scegliere anche soluzioni con 1-2 cluster quando sono quelle piu' stabili.
    """

    n_samples = int(n_samples)
    n_features = int(n_features)

    if n_samples < 3:
        raise ValueError("Servono almeno 3 campioni per UMAP + HDBSCAN.")

    if n_features < 1:
        raise ValueError("n_features deve essere >= 1.")

    sqrt_samples = np.sqrt(n_samples)

    max_components = min(n_features, n_samples - 2, 50)

    n_components_values = _make_set(
        [10, 15, 20, 30, 40, 50],
        min_value=1,
        max_value=max_components,
    )

    n_neighbors_values = _make_set(
        [
            round(sqrt_samples),
            round(2.0 * sqrt_samples),
            round(0.05 * n_samples),
            round(0.10 * n_samples),
            round(0.075 * n_samples),
            #round(0.15 * n_samples),
            #round(0.20 * n_samples),
        ],
        min_value=2,
        max_value=min(n_samples - 1, round(0.20 * n_samples), 200),
    )

    min_cluster_size_values = _make_set(
        [
            round(0.02 * n_samples),
            round(0.035 * n_samples),
            round(0.05 * n_samples),
            round(0.075 * n_samples),
           # round(0.10 * n_samples),
        ],
        min_value=5,
        max_value=min(n_samples - 1, round(0.30 * n_samples)),
    )

    param_grid = []

    for n_components, n_neighbors, min_cluster_size in product(
        n_components_values,
        n_neighbors_values,
        min_cluster_size_values,
    ):
        min_samples_values = _make_set(
            [
                5,
                10,
                round(0.25 * min_cluster_size),
                round(0.50 * min_cluster_size),
            ],
            min_value=5,
            max_value=min(min_cluster_size, n_samples - 1),
        )

        for min_samples in min_samples_values:
            param_grid.append({
                "n_components": int(n_components),
                "n_neighbors": int(n_neighbors),
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
            })

    return _deterministic_grid_sample(
        param_grid,
        max_size=max_param_combinations,
    )


_make_auto_param_grid = make_auto_param_grid


__all__ = [
    "make_auto_param_grid",
    "_make_auto_param_grid",
]