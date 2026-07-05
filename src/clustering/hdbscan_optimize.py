from __future__ import annotations

from collections import Counter
from itertools import combinations

import hdbscan as hdbscan_lib
import numpy as np
import pandas as pd
import umap

from clustering.auto_param_grid import make_auto_param_grid
from hdbscan.validity import validity_index
from sklearn.metrics import adjusted_rand_score


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

    if max_clusters is not None and n_clusters > max_clusters:
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


def _safe_mode_int(values):
    values = [
        int(value)
        for value in values
        if np.isfinite(float(value))
    ]

    if len(values) == 0:
        return np.nan

    return int(Counter(values).most_common(1)[0][0])


def _validate_metric_weight(name, value):
    weight = float(value)

    if not np.isfinite(weight):
        raise ValueError(f"{name} deve essere un numero finito.")

    if weight < 0.0:
        raise ValueError(f"{name} deve essere >= 0.0.")

    return weight


def _weighted_metric_sum(weighted_values):
    score = 0.0

    for weight, value in weighted_values:
        if weight == 0.0:
            continue

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        score += weight * value

    return float(score)


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
    min_dist,
    min_cluster_size,
    min_samples,
    cluster_selection_method
):
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        n_components=int(n_components),
        metric="cosine",
    )

    X_umap = reducer.fit_transform(embeddings)
    X_umap = np.asarray(X_umap, dtype=np.float64)
    X_umap = np.ascontiguousarray(X_umap)

    clusterer = hdbscan_lib.HDBSCAN(
        metric="euclidean",
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_method = cluster_selection_method
    )

    labels = clusterer.fit_predict(X_umap)

    return {
        "umap_embedding": X_umap,
        "umap_model": reducer,
        "hdbscan_model": clusterer,
        "labels": labels,
    }


def optimize_umap_hdbscan_auto(
    embeddings,
    n_runs=10,
    min_clusters=1,
    max_clusters=None,
    max_noise=0.50,
    min_valid_run_ratio=0.60,
    min_mean_ari=0.50,
    min_mean_dbcv=None,
    dbcv_weight=0.35,
    ari_weight=0.45,
    valid_run_ratio_weight=0.15,
    noise_weight=0.05,
    cluster_selection_method="eom",
    max_auto_param_combinations=240,
):
    """
    Ottimizza automaticamente una pipeline UMAP + HDBSCAN senza label.

    L'input deve essere una matrice di embedding gia' preprocessati,
    normalizzati/ridotti, con una riga per ogni glomerulo.

    Parametri fissati della pipeline:
    - UMAP: metric="cosine"
    - HDBSCAN: metric="euclidean"

    Viene sempre generata una griglia automatica da n_samples e n_features su:
    - n_components
    - n_neighbors
    - min_dist
    - min_cluster_size
    - min_samples

    Per ogni combinazione vengono eseguite n_runs senza impostare random_state
    in UMAP. La stabilita' e' misurata con ARI pairwise tra le label delle run
    valide della stessa combinazione. Il DBCV viene usato solo quando e'
    definito, quindi per soluzioni a 1 cluster non diventa un vincolo
    artificiale contro il collasso in pochi cluster.

    Una run e' valida se rispetta:
    - numero cluster HDBSCAN >= min_clusters
    - numero cluster HDBSCAN <= max_clusters, se max_clusters non e' None
    - noise_ratio <= max_noise

    Una combinazione entra nella selezione finale solo se:
    - valid_run_ratio >= min_valid_run_ratio
    - mean_ari >= min_mean_ari
    - mean_dbcv >= min_mean_dbcv, se min_mean_dbcv non e' None e la
      soluzione ha almeno 2 cluster

    La selezione finale ordina per combined_score decrescente, calcolato come:
    dbcv_weight * dbcv_for_score
    + ari_weight * mean_ari
    + valid_run_ratio_weight * valid_run_ratio
    + noise_weight * (1 - mean_noise_ratio).

    dbcv_for_score coincide con mean_dbcv quando e' finito; per soluzioni
    valide a un solo cluster usa 0.0, valore neutro nella scala DBCV [-1, 1].
    Non viene aggiunto nessun premio o vincolo sul numero assoluto di cluster.

    Returns
    -------
    output : dict
        Dizionario con:
        - best_n_components
        - best_n_neighbors
        - best_min_dist
        - best_min_cluster_size
        - best_min_samples
        - best_n_clusters
        - best_combined_score
        - best_mean_dbcv
        - best_mean_ari
        - best_params
        - results
        - run_details
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    embeddings = np.ascontiguousarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError("embeddings deve essere una matrice 2D.")

    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1]

    if n_samples < 3:
        raise ValueError("Servono almeno 3 glomeruli per UMAP + HDBSCAN.")

    if n_features < 1:
        raise ValueError("Ogni embedding deve avere almeno una feature.")

    param_grid = make_auto_param_grid(
        n_samples=n_samples,
        n_features=n_features,
        max_param_combinations=max_auto_param_combinations,
    )

    n_runs = int(n_runs)

    if n_runs < 2:
        raise ValueError("n_runs deve essere >= 2 per calcolare l'ARI.")

    min_clusters = int(min_clusters)
    max_clusters = None if max_clusters is None else int(max_clusters)
    max_noise = float(max_noise)
    min_valid_run_ratio = float(min_valid_run_ratio)
    min_mean_ari = float(min_mean_ari)
    min_mean_dbcv = (
        None
        if min_mean_dbcv is None
        else float(min_mean_dbcv)
    )
    dbcv_weight = _validate_metric_weight("dbcv_weight", dbcv_weight)
    ari_weight = _validate_metric_weight("ari_weight", ari_weight)
    valid_run_ratio_weight = _validate_metric_weight(
        "valid_run_ratio_weight",
        valid_run_ratio_weight,
    )
    noise_weight = _validate_metric_weight("noise_weight", noise_weight)

    if min_clusters < 1:
        raise ValueError("min_clusters deve essere >= 1.")

    if max_clusters is not None and max_clusters < min_clusters:
        raise ValueError("max_clusters deve essere >= min_clusters.")

    if max_noise < 0.0 or max_noise > 1.0:
        raise ValueError("max_noise deve essere compreso tra 0.0 e 1.0.")

    if min_valid_run_ratio < 0.0 or min_valid_run_ratio > 1.0:
        raise ValueError(
            "min_valid_run_ratio deve essere compreso tra 0.0 e 1.0."
        )

    if min_mean_ari < -1.0 or min_mean_ari > 1.0:
        raise ValueError("min_mean_ari deve essere compreso tra -1.0 e 1.0.")

    if min_mean_dbcv is not None and (
        not np.isfinite(min_mean_dbcv)
        or min_mean_dbcv < -1.0
        or min_mean_dbcv > 1.0
    ):
        raise ValueError(
            "min_mean_dbcv deve essere None oppure compreso tra -1.0 e 1.0."
        )

    if (
        dbcv_weight == 0.0
        and ari_weight == 0.0
        and valid_run_ratio_weight == 0.0
        and noise_weight == 0.0
    ):
        raise ValueError(
            "Almeno un peso della metrica combinata deve essere > 0.0."
        )

    results = []
    run_details = []

    for param_index, params in enumerate(param_grid):
        n_components = int(params["n_components"])
        n_neighbors = int(params["n_neighbors"])
        min_dist = float(params["min_dist"])
        min_cluster_size = int(params["min_cluster_size"])
        min_samples = params["min_samples"]
        min_samples_for_fit = None if min_samples is None else int(min_samples)

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
                "param_index": int(param_index),
                "n_components": int(n_components),
                "run_index": int(run_index),
                "n_neighbors": int(n_neighbors),
                "min_dist": float(min_dist),
                "min_cluster_size": int(min_cluster_size),
                "min_samples": min_samples_for_fit,
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
                    min_dist=min_dist,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples_for_fit,
                    cluster_selection_method=cluster_selection_method,
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
        valid_run_ratio = (
            float(valid_runs / successful_runs)
            if successful_runs > 0
            else 0.0
        )
        mean_dbcv = _safe_mean(valid_dbcv_values)
        std_dbcv = _safe_std(valid_dbcv_values)
        std_ari = _safe_std(ari_values)
        mean_n_clusters = _safe_mean(valid_n_cluster_values)
        mode_n_clusters = _safe_mode_int(valid_n_cluster_values)
        std_n_clusters = _safe_std(valid_n_cluster_values)
        mean_noise_ratio = _safe_mean(valid_noise_ratio_values)
        mean_dbcv_all = _safe_mean(dbcv_values)
        std_dbcv_all = _safe_std(dbcv_values)
        std_ari_all = _safe_std(ari_values_all)
        mean_n_clusters_all = _safe_mean(n_cluster_values)
        mode_n_clusters_all = _safe_mode_int(n_cluster_values)
        std_n_clusters_all = _safe_std(n_cluster_values)
        mean_noise_ratio_all = _safe_mean(noise_ratio_values)
        dbcv_for_score = mean_dbcv

        if (
            not np.isfinite(dbcv_for_score)
            and np.isfinite(mean_n_clusters)
            and mean_n_clusters < 2.0
        ):
            dbcv_for_score = 0.0

        noise_score = (
            1.0 - mean_noise_ratio
            if np.isfinite(mean_noise_ratio)
            else np.nan
        )
        combined_score = _weighted_metric_sum(
            [
                (dbcv_weight, dbcv_for_score),
                (ari_weight, mean_ari),
                (valid_run_ratio_weight, valid_run_ratio),
                (noise_weight, noise_score),
            ]
        )

        results.append({
            "param_index": int(param_index),
            "grid_source": "auto",
            "n_components": int(n_components),
            "n_runs": int(n_runs),
            "successful_runs": int(successful_runs),
            "valid_runs": int(valid_runs),
            "valid_run_ratio": valid_run_ratio,
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "min_cluster_size": int(min_cluster_size),
            "min_samples": min_samples_for_fit,
            "min_clusters": int(min_clusters),
            "max_clusters": max_clusters,
            "max_noise": float(max_noise),
            "min_valid_run_ratio": float(min_valid_run_ratio),
            "min_mean_ari": float(min_mean_ari),
            "min_mean_dbcv": min_mean_dbcv,
            "dbcv_weight": float(dbcv_weight),
            "ari_weight": float(ari_weight),
            "valid_run_ratio_weight": float(valid_run_ratio_weight),
            "noise_weight": float(noise_weight),
            "mean_dbcv": mean_dbcv,
            "dbcv_for_score": dbcv_for_score,
            "std_dbcv": std_dbcv,
            "mean_ari": mean_ari,
            "std_ari": std_ari,
            "mean_n_clusters": mean_n_clusters,
            "mode_n_clusters": mode_n_clusters,
            "std_n_clusters": std_n_clusters,
            "mean_noise_ratio": mean_noise_ratio,
            "noise_score": noise_score,
            "combined_score": combined_score,
            "mean_dbcv_all": mean_dbcv_all,
            "std_dbcv_all": std_dbcv_all,
            "mean_ari_all": mean_ari_all,
            "std_ari_all": std_ari_all,
            "mean_n_clusters_all": mean_n_clusters_all,
            "mode_n_clusters_all": mode_n_clusters_all,
            "std_n_clusters_all": std_n_clusters_all,
            "mean_noise_ratio_all": mean_noise_ratio_all,
        })

    results_df = pd.DataFrame(results)
    run_details_df = pd.DataFrame(run_details)

    finite_selection_mask = (
        np.isfinite(results_df["mean_ari"].to_numpy(dtype=float))
        & np.isfinite(results_df["combined_score"].to_numpy(dtype=float))
    )
    single_cluster_selection_mask = (
        np.isfinite(results_df["mean_n_clusters"].to_numpy(dtype=float))
        & (results_df["mean_n_clusters"].to_numpy(dtype=float) < 2.0)
    )
    dbcv_selection_mask = (
        True
        if min_mean_dbcv is None
        else (
            single_cluster_selection_mask
            | (
                np.isfinite(results_df["mean_dbcv"].to_numpy(dtype=float))
                & (
                    results_df["mean_dbcv"].to_numpy(dtype=float)
                    >= min_mean_dbcv
                )
            )
        )
    )

    results_df["valid_for_selection"] = (
        finite_selection_mask
        & dbcv_selection_mask
        & (
            results_df["valid_run_ratio"].to_numpy(dtype=float)
            >= min_valid_run_ratio
        )
        & (results_df["mean_ari"].to_numpy(dtype=float) >= min_mean_ari)
    )

    selectable_results = results_df.loc[
        results_df["valid_for_selection"]
    ].copy()

    if selectable_results.empty:
        return {
            "best_n_components": None,
            "best_n_neighbors": None,
            "best_min_dist": None,
            "best_min_cluster_size": None,
            "best_min_samples": None,
            "best_n_clusters": None,
            "best_combined_score": np.nan,
            "best_mean_dbcv": np.nan,
            "best_mean_ari": np.nan,
            "used_auto_param_grid": True,
            "n_param_combinations": int(len(param_grid)),
            "best_params": {
                "umap": {
                    "n_neighbors": None,
                    "min_dist": None,
                    "metric": "cosine",
                    "n_components": None,
                },
                "hdbscan": {
                    "metric": "euclidean",
                    "min_cluster_size": None,
                    "min_samples": None,
                    "cluster_selection_method":None
                },
                "constraints": {
                    "min_clusters": int(min_clusters),
                    "max_clusters": max_clusters,
                    "max_noise": float(max_noise),
                    "min_valid_run_ratio": float(min_valid_run_ratio),
                    "min_mean_ari": float(min_mean_ari),
                    "min_mean_dbcv": min_mean_dbcv,
                },
                "selection": {
                    "metric": "combined_score",
                    "dbcv_weight": float(dbcv_weight),
                    "ari_weight": float(ari_weight),
                    "valid_run_ratio_weight": float(valid_run_ratio_weight),
                    "noise_weight": float(noise_weight),
                    "single_cluster_dbcv_for_score": 0.0,
                },
            },
            "results": results_df,
            "run_details": run_details_df,
        }

    selectable_results = selectable_results.sort_values(
        by=[
            "combined_score",
            "mean_ari",
            "dbcv_for_score",
            "valid_run_ratio",
            "noise_score",
            "std_n_clusters",
        ],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    )

    best_row = selectable_results.iloc[0]
    best_n_components = int(best_row["n_components"])
    best_n_neighbors = int(best_row["n_neighbors"])
    best_min_dist = float(best_row["min_dist"])
    best_min_cluster_size = int(best_row["min_cluster_size"])
    best_min_samples = best_row["min_samples"]
    best_min_samples = (
        None
        if pd.isna(best_min_samples)
        else int(best_min_samples)
    )
    best_n_clusters = best_row["mode_n_clusters"]
    best_n_clusters = (
        None
        if pd.isna(best_n_clusters)
        else int(best_n_clusters)
    )

    best_params = {
        "umap": {
            "n_neighbors": best_n_neighbors,
            "min_dist": best_min_dist,
            "metric": "cosine",
            "n_components": best_n_components,
        },
        "hdbscan": {
            "metric": "euclidean",
            "min_cluster_size": best_min_cluster_size,
            "min_samples": best_min_samples,
            "cluster_selection_method": cluster_selection_method
        },
        "constraints": {
            "min_clusters": int(min_clusters),
            "max_clusters": max_clusters,
            "max_noise": float(max_noise),
            "min_valid_run_ratio": float(min_valid_run_ratio),
            "min_mean_ari": float(min_mean_ari),
            "min_mean_dbcv": min_mean_dbcv,
        },
        "selection": {
            "metric": "combined_score",
            "dbcv_weight": float(dbcv_weight),
            "ari_weight": float(ari_weight),
            "valid_run_ratio_weight": float(valid_run_ratio_weight),
            "noise_weight": float(noise_weight),
            "single_cluster_dbcv_for_score": 0.0,
        },
    }

    return {
        "best_n_components": best_n_components,
        "best_n_neighbors": best_n_neighbors,
        "best_min_dist": best_min_dist,
        "best_min_cluster_size": best_min_cluster_size,
        "best_min_samples": best_min_samples,
        "best_n_clusters": best_n_clusters,
        "best_combined_score": float(best_row["combined_score"]),
        "best_mean_dbcv": float(best_row["mean_dbcv"]),
        "best_mean_ari": float(best_row["mean_ari"]),
        "used_auto_param_grid": True,
        "n_param_combinations": int(len(param_grid)),
        "best_params": best_params,
        "results": results_df.sort_values(
            by=[
                "valid_for_selection",
                "combined_score",
                "mean_ari",
                "dbcv_for_score",
                "valid_run_ratio",
                "noise_score",
                "std_n_clusters",
            ],
            ascending=[False, False, False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True),
        "run_details": run_details_df,
    }


optimize_umap_hdbscan_n_components = optimize_umap_hdbscan_auto
optimize_hdbscan_umap_n_components = optimize_umap_hdbscan_auto


__all__ = [
    "optimize_umap_hdbscan_auto",
    "optimize_umap_hdbscan_n_components",
    "optimize_hdbscan_umap_n_components",
]
