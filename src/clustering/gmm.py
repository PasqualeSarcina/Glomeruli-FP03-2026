from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def evaluate_gmm_clustered_data(
    X,
    labels,
    probabilities=None,
    silhouette_metric="euclidean",
):
    """
    Valuta un clustering già eseguito con Gaussian Mixture Model.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dati usati per il clustering.

    labels : array-like, shape (n_samples,)
        Label prodotte dal GMM.

    probabilities : array-like, shape (n_samples, n_components), optional
        Probabilità posteriori prodotte da GaussianMixture.predict_proba.

    silhouette_metric : str, default="euclidean"
        Metrica usata per la silhouette.

    Returns
    -------
    metrics : dict
        Dizionario con metriche interne e statistiche sui cluster.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D.")
    if labels.ndim != 1:
        raise ValueError("labels deve essere un array 1D.")
    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X ha {X.shape[0]} righe, ma labels ha lunghezza {labels.shape[0]}."
        )

    n_samples = len(labels)
    unique_clusters = np.unique(labels)
    n_clusters = int(len(unique_clusters))

    cluster_sizes = {
        str(cluster): int(np.sum(labels == cluster))
        for cluster in unique_clusters
    }

    if n_clusters > 0:
        size_values = np.array(list(cluster_sizes.values()))
        min_cluster_size = int(size_values.min())
        max_cluster_size = int(size_values.max())
        mean_cluster_size = float(size_values.mean())
        median_cluster_size = float(np.median(size_values))
        largest_cluster_ratio = float(max_cluster_size / n_samples)
    else:
        min_cluster_size = None
        max_cluster_size = None
        mean_cluster_size = None
        median_cluster_size = None
        largest_cluster_ratio = None

    if n_clusters >= 2 and n_samples > n_clusters:
        silhouette = float(
            silhouette_score(
                X,
                labels,
                metric=silhouette_metric,
            )
        )
        davies_bouldin = float(davies_bouldin_score(X, labels))
        calinski_harabasz = float(calinski_harabasz_score(X, labels))
        valid_internal_metrics = True
    else:
        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan
        valid_internal_metrics = False

    probability_metrics = _probability_metrics(probabilities, labels)

    metrics = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "mean_cluster_size": mean_cluster_size,
        "median_cluster_size": median_cluster_size,
        "largest_cluster_ratio": largest_cluster_ratio,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "valid_internal_metrics": valid_internal_metrics,
        "cluster_sizes": cluster_sizes,
    }
    metrics.update(probability_metrics)

    return metrics


def _safe_float(value):
    """
    Converte un valore in float.
    Se il valore non è convertibile o è None, ritorna np.nan.
    """

    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _normalized_rank(values, ascending=True):
    """
    Calcola un rank normalizzato tra 0 e 1.

    0 = migliore
    1 = peggiore
    """

    values = pd.Series(values)

    ranks = values.rank(
        ascending=ascending,
        method="average",
    )

    if len(values) <= 1:
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
        )

    normalized = (ranks - 1) / (len(values) - 1)

    return normalized


def optimize_gmm_bic_parameters(
    X,
    gmm_n_components_values=(2, 3, 4, 5, 6, 7, 8, 9, 10),
    gmm_covariance_type_values=("full", "tied", "diag", "spherical"),
    min_clusters=2,
    max_clusters=None,
    min_cluster_size=1,
    random_state=42,
    n_init=10,
    max_iter=500,
    reg_covar=1e-6,
    reg_covar_values=None,
    init_params="kmeans",
    init_params_values=None,
    selection_metric="multiobjective",
    multiobjective_weights=None,
    silhouette_metric="euclidean",
):
    """
    Ottimizza un Gaussian Mixture Model provando diverse configurazioni.

    La funzione può selezionare il modello migliore usando una singola metrica
    oppure una selezione multi-obiettivo.

    Metriche supportate per selection_metric:
    - "bic"                  -> più basso è meglio
    - "aic"                  -> più basso è meglio
    - "icl"                  -> più basso è meglio
    - "silhouette"           -> più alto è meglio
    - "davies_bouldin"       -> più basso è meglio
    - "calinski_harabasz"    -> più alto è meglio
    - "mean_max_probability" -> più alto è meglio
    - "median_max_probability" -> più alto è meglio
    - "low_confidence_ratio_60" -> più basso è meglio
    - "low_confidence_ratio_70" -> più basso è meglio
    - "multiobjective"       -> combina BIC, silhouette, Davies-Bouldin
                                 e Calinski-Harabasz tramite rank normalizzati

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dati da clusterizzare.

    gmm_n_components_values : iterable of int
        Numero di componenti Gaussiane da provare.

    gmm_covariance_type_values : iterable of str
        Tipi di covarianza da provare: "full", "tied", "diag", "spherical".

    min_clusters : int, default=2
        Numero minimo di cluster non vuoti richiesto.

    max_clusters : int or None, default=None
        Numero massimo di cluster non vuoti accettato.

    min_cluster_size : int, default=1
        Dimensione minima richiesta per ogni cluster non vuoto.

    random_state : int, default=42
        Seed per la riproducibilità.

    n_init : int, default=10
        Numero di inizializzazioni GMM.

    max_iter : int, default=500
        Numero massimo di iterazioni.

    reg_covar : float, default=1e-6
        Valore usato se reg_covar_values è None.

    reg_covar_values : iterable of float or None
        Lista di valori reg_covar da provare.

    init_params : str, default="kmeans"
        Metodo usato se init_params_values è None.

    init_params_values : iterable of str or None
        Lista di metodi di inizializzazione da provare.

    selection_metric : str, default="multiobjective"
        Criterio usato per scegliere il modello migliore.

    multiobjective_weights : dict or None
        Pesi usati quando selection_metric="multiobjective".
        Se None, usa pesi consigliati per GMM:
        - bic: 0.40
        - silhouette: 0.25
        - davies_bouldin: 0.20
        - calinski_harabasz: 0.15

    silhouette_metric : str, default="euclidean"
        Metrica usata per calcolare la silhouette.

    Returns
    -------
    output : dict
        Dizionario con:
        - best_params
        - best_metrics
        - best_labels
        - best_probabilities
        - best_model
        - best_config_key
        - results
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D.")

    if X.shape[0] == 0:
        raise ValueError("X non puo' essere vuoto.")

    allowed_selection_metrics = (
        "bic",
        "aic",
        "icl",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "mean_max_probability",
        "median_max_probability",
        "low_confidence_ratio_60",
        "low_confidence_ratio_70",
        "multiobjective",
    )

    if selection_metric not in allowed_selection_metrics:
        raise ValueError(
            "selection_metric deve essere uno tra: "
            f"{allowed_selection_metrics}"
        )

    if reg_covar_values is None:
        reg_covar_values = (reg_covar,)

    if init_params_values is None:
        init_params_values = (init_params,)

    if multiobjective_weights is None:
        multiobjective_weights = {
            "bic": 0.40,
            "silhouette": 0.25,
            "davies_bouldin": 0.20,
            "calinski_harabasz": 0.15,
        }

    ranking_directions = {
        # True = più basso è meglio
        # False = più alto è meglio
        "bic": True,
        "aic": True,
        "icl": True,
        "davies_bouldin": True,
        "low_confidence_ratio_60": True,
        "low_confidence_ratio_70": True,

        "silhouette": False,
        "calinski_harabasz": False,
        "mean_max_probability": False,
        "median_max_probability": False,
    }

    if selection_metric == "multiobjective":
        invalid_weight_metrics = [
            metric
            for metric in multiobjective_weights.keys()
            if metric not in ranking_directions
        ]

        if invalid_weight_metrics:
            raise ValueError(
                "multiobjective_weights contiene metriche non supportate: "
                f"{invalid_weight_metrics}"
            )

        weight_sum = sum(multiobjective_weights.values())

        if weight_sum <= 0:
            raise ValueError(
                "La somma dei pesi in multiobjective_weights deve essere > 0."
            )

        # Normalizzo i pesi per sicurezza.
        multiobjective_weights = {
            metric: weight / weight_sum
            for metric, weight in multiobjective_weights.items()
        }

    results = []

    # Salvo i modelli fuori dal DataFrame per evitare colonne object pesanti.
    fitted_objects = {}

    config_counter = 0

    for gmm_n_components in gmm_n_components_values:

        if gmm_n_components < 1:
            continue

        if gmm_n_components > X.shape[0]:
            continue

        for gmm_covariance_type in gmm_covariance_type_values:

            for current_reg_covar in reg_covar_values:

                for current_init_params in init_params_values:

                    config_counter += 1
                    config_key = f"config_{config_counter}"

                    row = {
                        "config_key": config_key,
                        "gmm_n_components": int(gmm_n_components),
                        "gmm_covariance_type": gmm_covariance_type,
                        "gmm_n_init": int(n_init),
                        "gmm_max_iter": int(max_iter),
                        "gmm_reg_covar": float(current_reg_covar),
                        "gmm_init_params": current_init_params,
                        "selection_metric": selection_metric,
                    }

                    try:
                        model = GaussianMixture(
                            n_components=gmm_n_components,
                            covariance_type=gmm_covariance_type,
                            random_state=random_state,
                            n_init=n_init,
                            max_iter=max_iter,
                            reg_covar=current_reg_covar,
                            init_params=current_init_params,
                        )

                        model.fit(X)

                        labels = model.predict(X)
                        probabilities = model.predict_proba(X)

                        bic = float(model.bic(X))
                        aic = float(model.aic(X))

                        eps = 1e-12

                        entropy = float(
                            -np.sum(
                                probabilities
                                * np.log(probabilities + eps)
                            )
                        )

                        icl = float(bic + 2.0 * entropy)

                        max_probabilities = probabilities.max(axis=1)

                        mean_max_probability = float(
                            np.mean(max_probabilities)
                        )

                        median_max_probability = float(
                            np.median(max_probabilities)
                        )

                        low_confidence_ratio_60 = float(
                            np.mean(max_probabilities < 0.60)
                        )

                        low_confidence_ratio_70 = float(
                            np.mean(max_probabilities < 0.70)
                        )

                        metrics = evaluate_gmm_clustered_data(
                            X=X,
                            labels=labels,
                            probabilities=probabilities,
                            silhouette_metric=silhouette_metric,
                        )

                        silhouette = _safe_float(
                            metrics.get("silhouette", np.nan)
                        )

                        davies_bouldin = _safe_float(
                            metrics.get("davies_bouldin", np.nan)
                        )

                        calinski_harabasz = _safe_float(
                            metrics.get("calinski_harabasz", np.nan)
                        )

                        n_clusters_found = int(metrics["n_clusters"])
                        current_min_cluster_size = int(
                            metrics["min_cluster_size"]
                        )

                        valid_solution = (
                            bool(model.converged_)
                            and np.isfinite(bic)
                            and np.isfinite(aic)
                            and np.isfinite(icl)
                            and n_clusters_found >= min_clusters
                            and (
                                max_clusters is None
                                or n_clusters_found <= max_clusters
                            )
                            and current_min_cluster_size >= min_cluster_size
                        )

                        row.update(metrics)

                        row.update(
                            {
                                "bic": bic,
                                "aic": aic,
                                "icl": icl,
                                "entropy": entropy,
                                "lower_bound": float(model.lower_bound_),
                                "converged": bool(model.converged_),
                                "n_iter": int(model.n_iter_),
                                "silhouette": silhouette,
                                "davies_bouldin": davies_bouldin,
                                "calinski_harabasz": calinski_harabasz,
                                "mean_max_probability": mean_max_probability,
                                "median_max_probability": median_max_probability,
                                "low_confidence_ratio_60": low_confidence_ratio_60,
                                "low_confidence_ratio_70": low_confidence_ratio_70,
                                "valid_solution": valid_solution,
                                "error": None,
                            }
                        )

                        fitted_objects[config_key] = {
                            "model": model,
                            "labels": labels,
                            "probabilities": probabilities,
                            "metrics": metrics,
                        }

                    except (ValueError, np.linalg.LinAlgError) as error:
                        row.update(
                            {
                                "bic": np.nan,
                                "aic": np.nan,
                                "icl": np.nan,
                                "entropy": np.nan,
                                "lower_bound": np.nan,
                                "converged": False,
                                "n_iter": np.nan,
                                "silhouette": np.nan,
                                "davies_bouldin": np.nan,
                                "calinski_harabasz": np.nan,
                                "mean_max_probability": np.nan,
                                "median_max_probability": np.nan,
                                "low_confidence_ratio_60": np.nan,
                                "low_confidence_ratio_70": np.nan,
                                "valid_solution": False,
                                "error": str(error),
                            }
                        )

                    results.append(row)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Nessuna configurazione GMM valutata.")

        return {
            "best_params": None,
            "best_metrics": None,
            "best_labels": None,
            "best_probabilities": None,
            "best_model": None,
            "best_config_key": None,
            "results": results_df,
        }

    valid_mask = results_df["valid_solution"].astype(bool)

    # ------------------------------------------------------------
    # Ranking di tutte le metriche principali
    # ------------------------------------------------------------
    for metric_name, ascending in ranking_directions.items():

        rank_column = f"rank_{metric_name}"
        rank_norm_column = f"rank_norm_{metric_name}"

        results_df[rank_column] = np.nan
        results_df[rank_norm_column] = np.nan

        if metric_name in results_df.columns:

            metric_mask = (
                valid_mask
                & np.isfinite(results_df[metric_name])
            )

            if metric_mask.sum() > 0:
                results_df.loc[metric_mask, rank_column] = results_df.loc[
                    metric_mask,
                    metric_name,
                ].rank(ascending=ascending)

                results_df.loc[
                    metric_mask,
                    rank_norm_column,
                ] = _normalized_rank(
                    results_df.loc[metric_mask, metric_name],
                    ascending=ascending,
                )

    # ------------------------------------------------------------
    # Calcolo selection_score
    # ------------------------------------------------------------
    results_df["selection_score"] = np.nan
    results_df["multiobjective_score"] = np.nan

    if selection_metric == "multiobjective":

        required_metrics = list(multiobjective_weights.keys())

        multiobjective_mask = valid_mask.copy()

        for metric_name in required_metrics:
            multiobjective_mask = (
                multiobjective_mask
                & np.isfinite(results_df[metric_name])
                & np.isfinite(results_df[f"rank_norm_{metric_name}"])
            )

        if multiobjective_mask.sum() > 0:

            results_df.loc[
                multiobjective_mask,
                "multiobjective_score",
            ] = 0.0

            for metric_name, weight in multiobjective_weights.items():

                rank_norm_column = f"rank_norm_{metric_name}"

                results_df.loc[
                    multiobjective_mask,
                    "multiobjective_score",
                ] += (
                    weight
                    * results_df.loc[multiobjective_mask, rank_norm_column]
                )

            results_df.loc[
                multiobjective_mask,
                "selection_score",
            ] = results_df.loc[
                multiobjective_mask,
                "multiobjective_score",
            ]

    else:

        metric_name = selection_metric

        if metric_name not in results_df.columns:
            raise ValueError(
                f"La metrica '{metric_name}' non è presente in results_df."
            )

        metric_mask = (
            valid_mask
            & np.isfinite(results_df[metric_name])
        )

        if metric_mask.sum() > 0:

            # Se la metrica è da massimizzare, uso il segno meno.
            if ranking_directions[metric_name] is False:
                results_df.loc[
                    metric_mask,
                    "selection_score",
                ] = -results_df.loc[metric_mask, metric_name]

            else:
                results_df.loc[
                    metric_mask,
                    "selection_score",
                ] = results_df.loc[metric_mask, metric_name]

    selection_mask = (
        valid_mask
        & np.isfinite(results_df["selection_score"])
    )

    results_df["rank_selection"] = np.nan

    if selection_mask.sum() > 0:
        results_df.loc[
            selection_mask,
            "rank_selection",
        ] = results_df.loc[
            selection_mask,
            "selection_score",
        ].rank(ascending=True)

    results_df = results_df.sort_values(
        by=["valid_solution", "selection_score"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Selezione del best model
    # ------------------------------------------------------------
    final_selection_mask = (
        results_df["valid_solution"].astype(bool)
        & np.isfinite(results_df["selection_score"])
    )

    if final_selection_mask.sum() == 0:
        print("Nessuna configurazione GMM valida trovata.")
        print("Prova ad ampliare la griglia o a ridurre min_cluster_size.")

        return {
            "best_params": None,
            "best_metrics": None,
            "best_labels": None,
            "best_probabilities": None,
            "best_model": None,
            "best_config_key": None,
            "results": results_df,
        }

    best_row = results_df.loc[final_selection_mask].iloc[0]
    best_config_key = best_row["config_key"]

    best_object = fitted_objects[best_config_key]

    best_model = best_object["model"]
    best_labels = best_object["labels"]
    best_probabilities = best_object["probabilities"]

    best_params = {
        "gmm": {
            "n_components": int(best_row["gmm_n_components"]),
            "covariance_type": best_row["gmm_covariance_type"],
            "random_state": random_state,
            "n_init": int(best_row["gmm_n_init"]),
            "max_iter": int(best_row["gmm_max_iter"]),
            "reg_covar": float(best_row["gmm_reg_covar"]),
            "init_params": best_row["gmm_init_params"],
            "selection_metric": selection_metric,
        }
    }

    if selection_metric == "multiobjective":
        best_params["gmm"]["multiobjective_weights"] = (
            multiobjective_weights
        )

    best_metrics = best_object["metrics"].copy()

    metric_columns_to_copy = [
        "bic",
        "aic",
        "icl",
        "entropy",
        "selection_score",
        "multiobjective_score",
        "lower_bound",
        "converged",
        "n_iter",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "mean_max_probability",
        "median_max_probability",
        "low_confidence_ratio_60",
        "low_confidence_ratio_70",
        "rank_selection",
        "rank_bic",
        "rank_silhouette",
        "rank_davies_bouldin",
        "rank_calinski_harabasz",
        "rank_norm_bic",
        "rank_norm_silhouette",
        "rank_norm_davies_bouldin",
        "rank_norm_calinski_harabasz",
    ]

    for column in metric_columns_to_copy:
        if column in best_row.index:
            best_metrics[column] = best_row[column]

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_labels": best_labels,
        "best_probabilities": best_probabilities,
        "best_model": best_model,
        "best_config_key": best_config_key,
        "results": results_df,
    }


def cluster_gmm(
    X,
    n_components,
    covariance_type="full",
    random_state=42,
    n_init=10,
    max_iter=500,
    reg_covar=1e-6,
    init_params="kmeans",
):
    """
    Esegue un singolo clustering GMM senza ricerca BIC.

    Returns
    -------
    output : dict
        Dizionario con model, labels, probabilities e metrics.
    """

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D.")

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=reg_covar,
        init_params=init_params,
    )
    model.fit(X)

    labels = model.predict(X)
    probabilities = model.predict_proba(X)
    metrics = evaluate_gmm_clustered_data(
        X=X,
        labels=labels,
        probabilities=probabilities,
        silhouette_metric="euclidean",
    )
    metrics.update(
        {
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X)),
            "lower_bound": float(model.lower_bound_),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
        }
    )

    return {
        "model": model,
        "labels": labels,
        "probabilities": probabilities,
        "metrics": metrics,
    }


def _probability_metrics(probabilities, labels):
    if probabilities is None:
        return {
            "mean_assigned_probability": None,
            "min_assigned_probability": None,
            "mean_probability_entropy": None,
        }

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)

    if probabilities.ndim != 2:
        raise ValueError("probabilities deve essere un array 2D.")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            "probabilities e labels devono avere lo stesso numero di righe."
        )
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError(
            "Le label devono corrispondere alle colonne di probabilities."
        )

    row_indices = np.arange(labels.shape[0])
    assigned_probabilities = probabilities[row_indices, labels.astype(int)]

    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    if probabilities.shape[1] > 1:
        entropy = entropy / np.log(probabilities.shape[1])

    return {
        "mean_assigned_probability": float(np.mean(assigned_probabilities)),
        "min_assigned_probability": float(np.min(assigned_probabilities)),
        "mean_probability_entropy": float(np.mean(entropy)),
    }


__all__ = [
    "cluster_gmm",
    "evaluate_gmm_clustered_data",
    "optimize_gmm_bic_parameters",
]
