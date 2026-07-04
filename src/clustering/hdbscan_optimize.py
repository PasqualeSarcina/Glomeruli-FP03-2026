from __future__ import annotations

from itertools import combinations
from typing import Iterable

import hdbscan as hdbscan_lib
import numpy as np
import pandas as pd
import umap

from hdbscan.validity import validity_index
from sklearn.metrics import adjusted_rand_score


def _validate_n_components_values(n_components_values, n_features):
    values = list(n_components_values)

    if len(values) == 0:
        raise ValueError("n_components_values non puo' essere vuoto.")

    validated_values = []

    for value in values:
        n_components = int(value)

        if n_components < 1:
            raise ValueError("Ogni n_components deve essere >= 1.")

        if n_components > n_features:
            raise ValueError(
                "Ogni n_components deve essere minore o uguale al numero di "
                "componenti di ogni embedding."
            )

        if n_components not in validated_values:
            validated_values.append(n_components)

    return validated_values


def _count_clusters_and_noise(labels):
    labels = np.asarray(labels)
    cluster_ids = [
        label
        for label in np.unique(labels)
        if label != -1
    ]

    return int(len(cluster_ids)), float(np.mean(labels == -1))


def _check_cluster_constraints(
    n_clusters,
    noise_ratio,
    min_clusters,
    max_clusters,
    max_noise,
):
    if n_clusters < min_clusters:
        return False, "too_few_clusters"

    if n_clusters > max_clusters:
        return False, "too_many_clusters"

    if noise_ratio > max_noise:
        return False, "too_much_noise"

    return True, "ok"


def _compute_dbcv(X_clustered, labels):
    X_clustered = np.asarray(X_clustered, dtype=np.float64)
    X_clustered = np.ascontiguousarray(X_clustered)
    labels = np.asarray(labels)

    n_clusters, _ = _count_clusters_and_noise(labels)

    if n_clusters < 2:
        return np.nan

    return float(
        validity_index(
            X_clustered,
            labels,
            metric="euclidean",
        )
    )


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.mean(values))


def _safe_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.std(values))


def _mean_pairwise_ari(labels_by_run):
    if len(labels_by_run) < 2:
        return np.nan, []

    ari_values = []

    for labels_a, labels_b in combinations(labels_by_run, 2):
        ari_values.append(
            float(
                adjusted_rand_score(
                    labels_a,
                    labels_b,
                )
            )
        )

    return _safe_mean(ari_values), ari_values


def _fit_umap_hdbscan(
    embeddings,
    n_components,
    n_neighbors,
    min_cluster_size,
    min_samples,
):
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=0.0,
        n_components=int(n_components),
        metric="cosine",
    )

    X_umap = reducer.fit_transform(embeddings)
    X_umap = np.asarray(X_umap, dtype=np.float64)
    X_umap = np.ascontiguousarray(X_umap)

    clusterer = hdbscan_lib.HDBSCAN(
        metric="euclidean",
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
    )

    labels = clusterer.fit_predict(X_umap)

    return {
        "umap_embedding": X_umap,
        "umap_model": reducer,
        "hdbscan_model": clusterer,
        "labels": labels,
    }


def optimize_umap_hdbscan_n_components(
    embeddings,
    n_components_values: Iterable[int],
    n_runs=10,
    min_clusters=6,
    max_clusters=12,
    max_noise=0.33,
):
    """
    Ottimizza n_components di UMAP per una pipeline UMAP + HDBSCAN.

    L'input deve essere una matrice di embedding gia' normalizzati con
    L2 + PCA95%, con una riga per ogni glomerulo.

    Parametri fissati:
    - UMAP: min_dist=0.0, metric="cosine"
    - HDBSCAN: metric="euclidean"

    Per ogni valore di n_components vengono eseguite n_runs senza impostare
    random_state in UMAP. Il DBCV viene calcolato per ogni run sullo spazio
    UMAP usato da HDBSCAN. L'ARI medio e' la media degli ARI pairwise tra le
    label prodotte dalle run valide dello stesso n_components.

    Una run e' valida se rispetta:
    - min_clusters <= numero cluster HDBSCAN <= max_clusters
    - noise_ratio <= max_noise

    La selezione finale ordina per mean_dbcv decrescente e, a parita', per
    mean_ari decrescente, usando solo le run valide.

    Returns
    -------
    output : dict
        Dizionario con:
        - best_n_components
        - best_mean_dbcv
        - best_mean_ari
        - best_params
        - results
        - run_details
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    embeddings = np.ascontiguousarray(embeddings)

    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1]
    n_components_values = _validate_n_components_values(
        n_components_values=n_components_values,
        n_features=n_features,
    )

    n_runs = int(n_runs)

    if n_runs < 2:
        raise ValueError("n_runs deve essere >= 2 per calcolare l'ARI.")

    min_clusters = int(min_clusters)
    max_clusters = int(max_clusters)
    max_noise = float(max_noise)

    if min_clusters < 1:
        raise ValueError("min_clusters deve essere >= 1.")

    if max_clusters < min_clusters:
        raise ValueError("max_clusters deve essere >= min_clusters.")

    if max_noise < 0.0 or max_noise > 1.0:
        raise ValueError("max_noise deve essere compreso tra 0.0 e 1.0.")

    n_neighbors = int(
        np.clip(
            round(1.5 * np.sqrt(n_samples)),
            15,
            100,
        )
    )

    if n_neighbors >= n_samples:
        raise ValueError(
            "n_neighbors calcolato con la formula richiesta e' maggiore o "
            "uguale al numero di glomeruli."
        )

    min_cluster_size = int(max(10, round(0.03 * n_samples)))
    min_samples = int(
        np.clip(
            round(0.5 * min_cluster_size),
            5,
            30,
        )
    )

    if min_cluster_size > n_samples:
        raise ValueError(
            "min_cluster_size calcolato con la formula richiesta e' maggiore "
            "del numero di glomeruli."
        )

    results = []
    run_details = []

    for n_components in n_components_values:
        labels_by_run = []
        valid_labels_by_run = []
        dbcv_values = []
        valid_dbcv_values = []
        n_cluster_values = []
        valid_n_cluster_values = []
        noise_ratio_values = []
        valid_noise_ratio_values = []
        successful_runs = 0
        valid_runs = 0

        for run_index in range(n_runs):
            run_row = {
                "n_components": int(n_components),
                "run_index": int(run_index),
                "n_neighbors": int(n_neighbors),
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "dbcv": np.nan,
                "n_clusters": np.nan,
                "noise_ratio": np.nan,
                "success": False,
                "valid_for_selection": False,
                "constraint_reason": None,
                "error": None,
            }

            try:
                fit_result = _fit_umap_hdbscan(
                    embeddings=embeddings,
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )

                labels = fit_result["labels"]
                X_umap = fit_result["umap_embedding"]

                n_clusters, noise_ratio = _count_clusters_and_noise(labels)
                valid_for_selection, constraint_reason = _check_cluster_constraints(
                    n_clusters=n_clusters,
                    noise_ratio=noise_ratio,
                    min_clusters=min_clusters,
                    max_clusters=max_clusters,
                    max_noise=max_noise,
                )

                try:
                    dbcv = _compute_dbcv(
                        X_clustered=X_umap,
                        labels=labels,
                    )
                except Exception as error:
                    dbcv = np.nan
                    run_row["error"] = f"dbcv_error: {error}"

                labels_by_run.append(labels)
                dbcv_values.append(dbcv)
                n_cluster_values.append(n_clusters)
                noise_ratio_values.append(noise_ratio)
                successful_runs += 1

                if valid_for_selection:
                    valid_labels_by_run.append(labels)
                    valid_dbcv_values.append(dbcv)
                    valid_n_cluster_values.append(n_clusters)
                    valid_noise_ratio_values.append(noise_ratio)
                    valid_runs += 1

                run_row.update({
                    "dbcv": dbcv,
                    "n_clusters": int(n_clusters),
                    "noise_ratio": float(noise_ratio),
                    "success": True,
                    "valid_for_selection": bool(valid_for_selection),
                    "constraint_reason": constraint_reason,
                })

            except Exception as error:
                run_row["error"] = str(error)

            run_details.append(run_row)

        mean_ari, ari_values = _mean_pairwise_ari(valid_labels_by_run)
        mean_ari_all, ari_values_all = _mean_pairwise_ari(labels_by_run)

        results.append({
            "n_components": int(n_components),
            "n_runs": int(n_runs),
            "successful_runs": int(successful_runs),
            "valid_runs": int(valid_runs),
            "valid_run_ratio": (
                float(valid_runs / successful_runs)
                if successful_runs > 0
                else 0.0
            ),
            "n_neighbors": int(n_neighbors),
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "min_clusters": int(min_clusters),
            "max_clusters": int(max_clusters),
            "max_noise": float(max_noise),
            "mean_dbcv": _safe_mean(valid_dbcv_values),
            "std_dbcv": _safe_std(valid_dbcv_values),
            "mean_ari": mean_ari,
            "std_ari": _safe_std(ari_values),
            "mean_n_clusters": _safe_mean(valid_n_cluster_values),
            "mean_noise_ratio": _safe_mean(valid_noise_ratio_values),
            "mean_dbcv_all": _safe_mean(dbcv_values),
            "std_dbcv_all": _safe_std(dbcv_values),
            "mean_ari_all": mean_ari_all,
            "std_ari_all": _safe_std(ari_values_all),
            "mean_n_clusters_all": _safe_mean(n_cluster_values),
            "mean_noise_ratio_all": _safe_mean(noise_ratio_values),
        })

    results_df = pd.DataFrame(results)
    run_details_df = pd.DataFrame(run_details)

    finite_selection_mask = (
        np.isfinite(results_df["mean_dbcv"].to_numpy(dtype=float))
        & np.isfinite(results_df["mean_ari"].to_numpy(dtype=float))
    )

    selectable_results = results_df.loc[finite_selection_mask].copy()

    if selectable_results.empty:
        return {
            "best_n_components": None,
            "best_mean_dbcv": np.nan,
            "best_mean_ari": np.nan,
            "best_params": {
                "umap": {
                    "n_neighbors": int(n_neighbors),
                    "min_dist": 0.0,
                    "metric": "cosine",
                    "n_components": None,
                },
                "hdbscan": {
                    "metric": "euclidean",
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                },
                "constraints": {
                    "min_clusters": int(min_clusters),
                    "max_clusters": int(max_clusters),
                    "max_noise": float(max_noise),
                },
            },
            "results": results_df,
            "run_details": run_details_df,
        }

    selectable_results = selectable_results.sort_values(
        by=["mean_dbcv", "mean_ari"],
        ascending=[False, False],
    )

    best_row = selectable_results.iloc[0]
    best_n_components = int(best_row["n_components"])

    best_params = {
        "umap": {
            "n_neighbors": int(n_neighbors),
            "min_dist": 0.0,
            "metric": "cosine",
            "n_components": best_n_components,
        },
        "hdbscan": {
            "metric": "euclidean",
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
        },
        "constraints": {
            "min_clusters": int(min_clusters),
            "max_clusters": int(max_clusters),
            "max_noise": float(max_noise),
        },
    }

    return {
        "best_n_components": best_n_components,
        "best_mean_dbcv": float(best_row["mean_dbcv"]),
        "best_mean_ari": float(best_row["mean_ari"]),
        "best_params": best_params,
        "results": results_df.sort_values(
            by=["mean_dbcv", "mean_ari"],
            ascending=[False, False],
        ).reset_index(drop=True),
        "run_details": run_details_df,
    }


optimize_hdbscan_umap_n_components = optimize_umap_hdbscan_n_components


__all__ = [
    "optimize_umap_hdbscan_n_components",
    "optimize_hdbscan_umap_n_components",
]
