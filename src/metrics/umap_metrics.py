import numpy as np
import pandas as pd

from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


def continuity(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k: int
) -> float:
    """
    Calcola la continuity.

    Misura se i veri vicini nello spazio sorgente vengono preservati
    nello spazio embedded.

    Valori:
    - vicino a 1 = buona preservazione
    - vicino a 0 = cattiva preservazione
    """

    n_samples = X.shape[0]


    # rank_embedded[i, j] = rank del punto j rispetto al punto i
    # nello spazio embedded.
    rank_embedded = np.empty(
        (n_samples, n_samples),
        dtype=np.int32
    )

    rank_embedded.fill(-1)

    for i in range(n_samples):
        rank_embedded[i, i] = 0

        # embedded_order esclude il punto stesso.
        # Quindi il primo vicino ha rank 1.
        rank_embedded[i, X_embedded[i]] = np.arange(1, n_samples)

    penalty = 0.0

    for i in range(n_samples):
        source_neighbors = set(X[i, :k])
        embedded_neighbors = set(X_embedded[i, :k])

        # Vicini che esistevano nello spazio sorgente
        # ma sono stati persi nello spazio embedded.
        missing_neighbors = source_neighbors - embedded_neighbors

        for j in missing_neighbors:
            penalty += rank_embedded[i, j] - k

    normalizer = n_samples * k * (2 * n_samples - 3 * k - 1)

    continuity = 1.0 - (2.0 / normalizer) * penalty

    return float(continuity)


def qnn(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k: int
) -> float:
    """
    Calcola QNN(k), cioè k-nearest neighbor preservation.

    Misura quanti vicini dello spazio sorgente sono preservati
    nello spazio embedded.

    Valori:
    - 0 = nessun vicino preservato
    - 1 = tutti i vicini preservati
    """

    n_samples = X.shape[0]

    scores = []

    for i in range(n_samples):
        source_neighbors = set(X[i, :k])
        embedded_neighbors = set(X_embedded[i, :k])

        overlap = source_neighbors.intersection(embedded_neighbors)

        scores.append(len(overlap) / k)

    return float(np.mean(scores))


def lcmc(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k: int
) -> float:
    """
    Calcola LCMC(k).

    LCMC corregge QNN rispetto alla preservazione attesa per caso.

    Valori più alti indicano migliore qualità locale.
    """

    n_samples = X.shape[0]

    qnn_value = qnn(X, X_embedded, k)

    random_baseline = k / (n_samples - 1)

    result = qnn_value - random_baseline

    return float(result)


def auc_lcmc(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k_max: int
) -> float:
    """
    Calcola AUC_LCMC come media dei valori LCMC(k)
    da k = 1 a k = k_max.
    """

    lcmc_values = []

    for k in range(1, k_max + 1):
        value = lcmc(X, X_embedded, k)
        lcmc_values.append(value)

    return float(np.mean(lcmc_values))


def evaluate_umap_embedding(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k_values=(5, 10, 15),
    source_metric="euclidean",
    embedded_metric="euclidean",
) -> dict:
    """
    Valuta una proiezione UMAP usando le metriche definite in questo modulo.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dati nello spazio sorgente.

    X_embedded : array-like, shape (n_samples, n_components)
        Coordinate prodotte da UMAP.

    k_values : iterable of int
        Valori di k per trustworthiness, continuity, QNN e LCMC.

    source_metric : str, default="euclidean"
        Metrica per calcolare i vicini nello spazio sorgente.

    embedded_metric : str, default="euclidean"
        Metrica per calcolare i vicini nello spazio UMAP.
    """

    X = np.asarray(X)
    X_embedded = np.asarray(X_embedded)

    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D.")
    if X_embedded.ndim != 2:
        raise ValueError("X_embedded deve essere un array 2D.")
    if X.shape[0] != X_embedded.shape[0]:
        raise ValueError(
            "X e X_embedded devono avere lo stesso numero di campioni."
        )

    k_values = _validate_k_values(k_values, X.shape[0])
    source_neighbors = _nearest_neighbor_indices(X, metric=source_metric)
    embedded_neighbors = _nearest_neighbor_indices(
        X_embedded,
        metric=embedded_metric,
    )

    return _evaluate_umap_embedding_from_neighbors(
        X=X,
        X_embedded=X_embedded,
        source_neighbors=source_neighbors,
        embedded_neighbors=embedded_neighbors,
        k_values=k_values,
        source_metric=source_metric,
    )


def optimize_umap_parameters(
    X,
    umap_n_neighbors_values=(15, 30, 40, 50),
    umap_min_dist_values=(0.0, 0.01, 0.1, 0.3),
    umap_n_components_values=(2, 3, 5, 10, 15),
    umap_metric_values=("euclidean",),
    k_values=(5, 10, 15),
    random_state=42,
    init="spectral",
    spread=1.0,
    n_epochs=None,
    embedded_metric="euclidean",
    metric_weights=None,
):
    """
    Ottimizza UMAP tramite grid search sulle metriche di qualita' dell'embedding.

    La funzione prova le combinazioni dei parametri UMAP, calcola
    trustworthiness, continuity, QNN, LCMC e AUC_LCMC, poi sceglie la
    configurazione con il miglior ranking medio pesato.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dati prima di UMAP, per esempio embedding normalizzati o X_pca.

    metric_weights : dict or None
        Pesi opzionali per le colonne aggregate:
        "trustworthiness_mean", "continuity_mean", "qnn_mean", "lcmc_mean",
        "auc_lcmc". Se None usa pesi bilanciati verso la qualita' locale.

    Returns
    -------
    output : dict
        Dizionario con:
        - best_params
        - best_metrics
        - best_embedding
        - best_reducer
        - results
    """

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D.")
    if X.shape[0] < 3:
        raise ValueError("Servono almeno 3 campioni per ottimizzare UMAP.")

    k_values = _validate_k_values(k_values, X.shape[0])
    metric_weights = _metric_weights(metric_weights)
    source_neighbors_cache = {}

    results = []
    config_counter = 0

    for umap_metric in umap_metric_values:
        source_neighbors_cache[umap_metric] = _nearest_neighbor_indices(
            X,
            metric=umap_metric,
        )

        for umap_n_neighbors in umap_n_neighbors_values:
            if not 2 <= umap_n_neighbors < X.shape[0]:
                continue

            for umap_min_dist in umap_min_dist_values:
                if umap_min_dist < 0 or umap_min_dist > spread:
                    continue

                for umap_n_components in umap_n_components_values:
                    if not 1 <= umap_n_components < X.shape[0]:
                        continue

                    config_counter += 1
                    config_key = f"config_{config_counter}"

                    row = {
                        "config_key": config_key,
                        "umap_n_neighbors": int(umap_n_neighbors),
                        "umap_min_dist": float(umap_min_dist),
                        "umap_n_components": int(umap_n_components),
                        "umap_metric": umap_metric,
                        "umap_init": init,
                        "umap_spread": float(spread),
                        "umap_n_epochs": n_epochs,
                    }

                    try:
                        reducer = UMAP(
                            n_neighbors=umap_n_neighbors,
                            min_dist=umap_min_dist,
                            n_components=umap_n_components,
                            metric=umap_metric,
                            random_state=random_state,
                            init=init,
                            spread=spread,
                            n_epochs=n_epochs,
                        )

                        X_umap = reducer.fit_transform(X)
                        embedded_neighbors = _nearest_neighbor_indices(
                            X_umap,
                            metric=embedded_metric,
                        )
                        metrics = _evaluate_umap_embedding_from_neighbors(
                            X=X,
                            X_embedded=X_umap,
                            source_neighbors=source_neighbors_cache[umap_metric],
                            embedded_neighbors=embedded_neighbors,
                            k_values=k_values,
                            source_metric=umap_metric,
                        )

                        valid_solution = all(
                            np.isfinite(metrics[metric_name])
                            for metric_name in metric_weights
                        )

                        row.update(metrics)
                        row.update(
                            {
                                "valid_solution": valid_solution,
                                "error": None,
                            }
                        )

                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                        row.update(
                            {
                                "valid_solution": False,
                                "error": str(error),
                            }
                        )

                    results.append(row)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return {
            "best_params": None,
            "best_metrics": None,
            "best_embedding": None,
            "best_reducer": None,
            "results": results_df,
        }

    valid_df = results_df[results_df["valid_solution"]].copy()

    if len(valid_df) == 0:
        print("Nessuna configurazione UMAP valida trovata.")
        print("Prova ad ampliare la griglia o a ridurre i valori di k.")

        return {
            "best_params": None,
            "best_metrics": None,
            "best_embedding": None,
            "best_reducer": None,
            "results": results_df,
        }

    rank_columns = []
    final_rank_score = 0.0

    for metric_name, weight in metric_weights.items():
        rank_column = f"rank_{metric_name}"
        valid_df[rank_column] = valid_df[metric_name].rank(ascending=False)
        rank_columns.append(rank_column)
        final_rank_score = final_rank_score + weight * valid_df[rank_column]

    valid_df["final_rank_score"] = final_rank_score

    for col in rank_columns + ["final_rank_score"]:
        results_df[col] = np.nan
        results_df.loc[valid_df.index, col] = valid_df[col]

    results_df = results_df.sort_values(
        by=["valid_solution", "final_rank_score"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_params = {
        "umap": {
            "n_neighbors": int(best_row["umap_n_neighbors"]),
            "min_dist": float(best_row["umap_min_dist"]),
            "n_components": int(best_row["umap_n_components"]),
            "metric": best_row["umap_metric"],
            "random_state": random_state,
            "init": best_row["umap_init"],
            "spread": float(best_row["umap_spread"]),
            "n_epochs": best_row["umap_n_epochs"],
        },
        "evaluation": {
            "k_values": list(k_values),
            "embedded_metric": embedded_metric,
            "metric_weights": metric_weights,
        },
    }

    best_reducer = UMAP(
        n_neighbors=best_params["umap"]["n_neighbors"],
        min_dist=best_params["umap"]["min_dist"],
        n_components=best_params["umap"]["n_components"],
        metric=best_params["umap"]["metric"],
        random_state=random_state,
        init=best_params["umap"]["init"],
        spread=best_params["umap"]["spread"],
        n_epochs=best_params["umap"]["n_epochs"],
    )
    best_embedding = best_reducer.fit_transform(X)

    best_metrics = evaluate_umap_embedding(
        X=X,
        X_embedded=best_embedding,
        k_values=k_values,
        source_metric=best_params["umap"]["metric"],
        embedded_metric=embedded_metric,
    )
    best_metrics["final_rank_score"] = float(best_row["final_rank_score"])

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_embedding": best_embedding,
        "best_reducer": best_reducer,
        "results": results_df,
    }


def _evaluate_umap_embedding_from_neighbors(
    X,
    X_embedded,
    source_neighbors,
    embedded_neighbors,
    k_values,
    source_metric,
):
    metrics = {}

    for k in k_values:
        metrics[f"trustworthiness_k_{k}"] = float(
            trustworthiness(
                X,
                X_embedded,
                n_neighbors=k,
                metric=source_metric,
            )
        )
        metrics[f"continuity_k_{k}"] = continuity(
            source_neighbors,
            embedded_neighbors,
            k,
        )
        metrics[f"qnn_k_{k}"] = qnn(
            source_neighbors,
            embedded_neighbors,
            k,
        )
        metrics[f"lcmc_k_{k}"] = lcmc(
            source_neighbors,
            embedded_neighbors,
            k,
        )

    metrics["trustworthiness_mean"] = float(
        np.mean([metrics[f"trustworthiness_k_{k}"] for k in k_values])
    )
    metrics["continuity_mean"] = float(
        np.mean([metrics[f"continuity_k_{k}"] for k in k_values])
    )
    metrics["qnn_mean"] = float(
        np.mean([metrics[f"qnn_k_{k}"] for k in k_values])
    )
    metrics["lcmc_mean"] = float(
        np.mean([metrics[f"lcmc_k_{k}"] for k in k_values])
    )
    metrics["auc_lcmc"] = auc_lcmc(
        source_neighbors,
        embedded_neighbors,
        max(k_values),
    )

    return metrics


def _nearest_neighbor_indices(X, metric="euclidean") -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D.")

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Servono almeno 2 campioni per calcolare i vicini.")

    neighbors = NearestNeighbors(
        n_neighbors=n_samples,
        metric=metric,
    )
    neighbors.fit(X)
    indices = neighbors.kneighbors(X, return_distance=False)

    ordered_neighbors = []
    for sample_index, sample_neighbors in enumerate(indices):
        filtered_neighbors = sample_neighbors[sample_neighbors != sample_index]
        ordered_neighbors.append(filtered_neighbors[: n_samples - 1])

    return np.asarray(ordered_neighbors, dtype=np.int32)


def _validate_k_values(k_values, n_samples: int) -> tuple[int, ...]:
    if isinstance(k_values, int):
        k_values = (k_values,)

    validated = tuple(sorted({int(k) for k in k_values}))
    if len(validated) == 0:
        raise ValueError("k_values non puo' essere vuoto.")

    max_allowed_k = (n_samples - 1) // 2
    for k in validated:
        if k < 1:
            raise ValueError("Ogni valore di k deve essere almeno 1.")
        if k > max_allowed_k:
            raise ValueError(
                "Ogni valore di k deve essere minore di n_samples / 2 "
                "per calcolare trustworthiness."
            )

    return validated


def _metric_weights(metric_weights) -> dict:
    default_weights = {
        "trustworthiness_mean": 0.30,
        "continuity_mean": 0.30,
        "qnn_mean": 0.20,
        "lcmc_mean": 0.10,
        "auc_lcmc": 0.10,
    }

    if metric_weights is None:
        return default_weights

    unknown_metrics = set(metric_weights) - set(default_weights)
    if unknown_metrics:
        raise ValueError(
            "metric_weights contiene metriche non supportate: "
            f"{sorted(unknown_metrics)}"
        )

    weights = default_weights | metric_weights
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("La somma dei pesi deve essere positiva.")

    return {
        metric_name: float(weight / total_weight)
        for metric_name, weight in weights.items()
    }


optimize_umap = optimize_umap_parameters


__all__ = [
    "auc_lcmc",
    "continuity",
    "evaluate_umap_embedding",
    "lcmc",
    "optimize_umap",
    "optimize_umap_parameters",
    "qnn",
]
