import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def optimize_gmm_bic(
    X,
    n_components_values=range(2, 21),
    covariance_types=("full", "tied", "diag", "spherical"),
    reg_covar_values=(1e-6, 1e-5, 1e-4, 1e-3),
    n_init=20,
    random_state=42,
    max_iter=500,

    # Nuovi parametri
    min_cluster_size="auto",
    min_cluster_size_absolute=5,
    min_cluster_fraction=0.01,
    small_cluster_penalty_weight=1.0,
    hard_min_cluster_size=False,
):
    #TODO: inserire penalizzazione silhouette bassa / cluster grandi
    """
    Ottimizza un GaussianMixture usando BIC come criterio principale,
    con penalità opzionale per cluster troppo piccoli.

    BIC più basso = modello migliore.

    La selezione finale usa:

        selection_score = bic_per_sample + small_cluster_penalty_weight * small_cluster_penalty

    dove:
        bic_per_sample = BIC / n_samples

    e small_cluster_penalty penalizza i cluster con dimensione inferiore
    alla soglia adattiva min_cluster_size.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Embedding su cui applicare GMM, per esempio UMAP 2D o UMAP 10D.

    min_cluster_size : "auto" oppure int
        Se "auto", viene calcolato in modo scalabile:
            max(min_cluster_size_absolute, sqrt(n_samples), min_cluster_fraction * n_samples)

    min_cluster_size_absolute : int
        Dimensione minima assoluta.

    min_cluster_fraction : float
        Percentuale minima del dataset. Per esempio 0.01 = 1%.

    small_cluster_penalty_weight : float
        Peso della penalità. Valori consigliati:
            0.5  = penalità leggera
            1.0  = penalità media
            2.0  = penalità forte

    hard_min_cluster_size : bool
        Se True, le soluzioni con cluster sotto soglia non possono essere selezionate.
        Se False, vengono solo penalizzate.

    Returns
    -------
    output : dict
        Dizionario con:
        - best_model
        - best_params
        - best_labels
        - best_probabilities
        - best_metrics
        - results
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D.")

    n_samples = X.shape[0]

    # Soglia scalabile per cluster piccoli
    if min_cluster_size == "auto":
        resolved_min_cluster_size = int(
            max(
                min_cluster_size_absolute,
                np.ceil(np.sqrt(n_samples)),
                np.ceil(min_cluster_fraction * n_samples),
            )
        )
    elif isinstance(min_cluster_size, int):
        resolved_min_cluster_size = min_cluster_size
    else:
        raise ValueError("min_cluster_size deve essere 'auto' oppure un intero.")

    def compute_cluster_size_penalty(labels, n_components, min_size):
        """
        Calcola penalità per cluster più piccoli della soglia.

        La penalità cresce con il deficit relativo:
            deficit = min_size - cluster_size

        Penalità:
            somma dei deficit relativi al quadrato
        """

        counts = np.bincount(labels, minlength=n_components)

        deficits = np.maximum(0, min_size - counts)

        relative_deficits = deficits / max(min_size, 1)

        small_cluster_penalty = float(np.sum(relative_deficits ** 2))

        n_small_clusters = int(np.sum(counts < min_size))

        small_cluster_sample_count = int(np.sum(counts[counts < min_size]))

        small_cluster_sample_ratio = float(small_cluster_sample_count / len(labels))

        return {
            "counts": counts,
            "min_cluster_size_found": int(counts.min()),
            "max_cluster_size_found": int(counts.max()),
            "n_small_clusters": n_small_clusters,
            "small_cluster_sample_count": small_cluster_sample_count,
            "small_cluster_sample_ratio": small_cluster_sample_ratio,
            "small_cluster_penalty": small_cluster_penalty,
            "valid_size_solution": bool(n_small_clusters == 0),
        }

    results = []
    best_model = None
    best_selection_score = np.inf

    for n_components in n_components_values:
        if n_components < 1 or n_components >= X.shape[0]:
            continue

        for covariance_type in covariance_types:
            for reg_covar in reg_covar_values:

                row = {
                    "n_components": int(n_components),
                    "covariance_type": covariance_type,
                    "reg_covar": float(reg_covar),
                    "n_init": int(n_init),
                    "min_cluster_size_threshold": int(resolved_min_cluster_size),
                    "valid_solution": False,
                    "valid_size_solution": False,
                    "valid_for_selection": False,
                    "bic": np.nan,
                    "aic": np.nan,
                    "bic_per_sample": np.nan,
                    "selection_score": np.nan,
                    "small_cluster_penalty": np.nan,
                    "n_small_clusters": np.nan,
                    "small_cluster_sample_count": np.nan,
                    "small_cluster_sample_ratio": np.nan,
                    "min_cluster_size_found": np.nan,
                    "max_cluster_size_found": np.nan,
                    "converged": False,
                    "n_iter": np.nan,
                    "error": None,
                }

                try:
                    model = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        reg_covar=reg_covar,
                        n_init=n_init,
                        random_state=random_state,
                        max_iter=max_iter,
                    )

                    model.fit(X)

                    labels = model.predict(X)

                    bic = model.bic(X)
                    aic = model.aic(X)
                    bic_per_sample = bic / n_samples

                    size_info = compute_cluster_size_penalty(
                        labels=labels,
                        n_components=n_components,
                        min_size=resolved_min_cluster_size,
                    )

                    small_cluster_penalty = size_info["small_cluster_penalty"]

                    selection_score = (
                        bic_per_sample
                        + small_cluster_penalty_weight * small_cluster_penalty
                    )

                    valid_for_selection = True

                    if hard_min_cluster_size and not size_info["valid_size_solution"]:
                        valid_for_selection = False

                    row.update({
                        "valid_solution": True,
                        "valid_size_solution": size_info["valid_size_solution"],
                        "valid_for_selection": valid_for_selection,
                        "bic": float(bic),
                        "aic": float(aic),
                        "bic_per_sample": float(bic_per_sample),
                        "selection_score": float(selection_score),
                        "small_cluster_penalty": float(size_info["small_cluster_penalty"]),
                        "n_small_clusters": int(size_info["n_small_clusters"]),
                        "small_cluster_sample_count": int(size_info["small_cluster_sample_count"]),
                        "small_cluster_sample_ratio": float(size_info["small_cluster_sample_ratio"]),
                        "min_cluster_size_found": int(size_info["min_cluster_size_found"]),
                        "max_cluster_size_found": int(size_info["max_cluster_size_found"]),
                        "converged": bool(model.converged_),
                        "n_iter": int(model.n_iter_),
                    })

                    if (
                        valid_for_selection
                        and np.isfinite(selection_score)
                        and selection_score < best_selection_score
                    ):
                        best_selection_score = selection_score
                        best_model = model

                except Exception as error:
                    row["error"] = str(error)

                results.append(row)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["valid_for_selection", "selection_score", "bic"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    if best_model is None:
        return {
            "best_model": None,
            "best_params": None,
            "best_labels": None,
            "best_probabilities": None,
            "best_metrics": None,
            "results": results_df,
        }

    best_labels = best_model.predict(X)
    best_probabilities = best_model.predict_proba(X)

    best_size_info = compute_cluster_size_penalty(
        labels=best_labels,
        n_components=best_model.n_components,
        min_size=resolved_min_cluster_size,
    )

    best_bic = best_model.bic(X)
    best_aic = best_model.aic(X)
    best_bic_per_sample = best_bic / n_samples

    best_selection_score = (
        best_bic_per_sample
        + small_cluster_penalty_weight * best_size_info["small_cluster_penalty"]
    )

    best_params = {
        "n_components": best_model.n_components,
        "covariance_type": best_model.covariance_type,
        "reg_covar": best_model.reg_covar,
        "n_init": n_init,
        "random_state": random_state,
        "max_iter": max_iter,
        "min_cluster_size_threshold": resolved_min_cluster_size,
        "small_cluster_penalty_weight": small_cluster_penalty_weight,
        "hard_min_cluster_size": hard_min_cluster_size,
    }

    max_probabilities = np.max(best_probabilities, axis=1)

    best_metrics = {
        "bic": float(best_bic),
        "aic": float(best_aic),
        "bic_per_sample": float(best_bic_per_sample),
        "selection_score": float(best_selection_score),
        "converged": bool(best_model.converged_),
        "n_iter": int(best_model.n_iter_),
        "mean_max_probability": float(max_probabilities.mean()),
        "median_max_probability": float(np.median(max_probabilities)),
        "min_max_probability": float(max_probabilities.min()),
        "min_cluster_size_found": int(best_size_info["min_cluster_size_found"]),
        "max_cluster_size_found": int(best_size_info["max_cluster_size_found"]),
        "n_small_clusters": int(best_size_info["n_small_clusters"]),
        "small_cluster_sample_count": int(best_size_info["small_cluster_sample_count"]),
        "small_cluster_sample_ratio": float(best_size_info["small_cluster_sample_ratio"]),
        "small_cluster_penalty": float(best_size_info["small_cluster_penalty"]),
        "valid_size_solution": bool(best_size_info["valid_size_solution"]),
    }

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_labels": best_labels,
        "best_probabilities": best_probabilities,
        "best_metrics": best_metrics,
        "results": results_df,
    }



def evaluate_gmm_clustered_data(
    X,
    labels,
    probabilities=None,
    silhouette_metric="euclidean",
):
    """
    Valuta un clustering prodotto da GaussianMixture.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Spazio su cui è stato applicato GMM, per esempio UMAP 10D.

    labels : array-like, shape (n_samples,)
        Label hard assegnate dal GMM tramite model.predict(X).

    probabilities : array-like, shape (n_samples, n_components), optional
        Probabilità di appartenenza restituite da model.predict_proba(X).

    silhouette_metric : str
        Metrica usata per la silhouette. Di solito "euclidean".

    Returns
    -------
    metrics : dict
        Dizionario con metriche globali del clustering.
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

    n_samples = X.shape[0]
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)

    metrics = {
        "n_samples": int(n_samples),
        "n_features": int(X.shape[1]),
        "n_clusters": int(n_clusters),
        "min_cluster_size": int(np.min(counts)),
        "max_cluster_size": int(np.max(counts)),
        "mean_cluster_size": float(np.mean(counts)),
        "median_cluster_size": float(np.median(counts)),
    }

    # Metriche geometriche globali.
    # Richiedono almeno 2 cluster e meno cluster del numero di campioni.
    if 2 <= n_clusters < n_samples:
        try:
            metrics["silhouette"] = float(
                silhouette_score(
                    X,
                    labels,
                    metric=silhouette_metric,
                )
            )
        except Exception:
            metrics["silhouette"] = np.nan

        try:
            metrics["davies_bouldin"] = float(
                davies_bouldin_score(X, labels)
            )
        except Exception:
            metrics["davies_bouldin"] = np.nan

        try:
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(X, labels)
            )
        except Exception:
            metrics["calinski_harabasz"] = np.nan

    else:
        metrics["silhouette"] = np.nan
        metrics["davies_bouldin"] = np.nan
        metrics["calinski_harabasz"] = np.nan

    # Metriche probabilistiche specifiche del GMM.
    if probabilities is not None:
        probabilities = np.asarray(probabilities)

        if probabilities.ndim != 2:
            raise ValueError("probabilities deve essere un array 2D.")

        if probabilities.shape[0] != n_samples:
            raise ValueError(
                f"probabilities ha {probabilities.shape[0]} righe, "
                f"ma X ha {n_samples} righe."
            )

        max_probabilities = np.max(probabilities, axis=1)

        metrics["mean_max_probability"] = float(np.mean(max_probabilities))
        metrics["median_max_probability"] = float(np.median(max_probabilities))
        metrics["min_max_probability"] = float(np.min(max_probabilities))
        metrics["max_max_probability"] = float(np.max(max_probabilities))

        metrics["fraction_probability_below_0_50"] = float(
            np.mean(max_probabilities < 0.50)
        )
        metrics["fraction_probability_below_0_60"] = float(
            np.mean(max_probabilities < 0.60)
        )
        metrics["fraction_probability_below_0_70"] = float(
            np.mean(max_probabilities < 0.70)
        )
        metrics["fraction_probability_below_0_80"] = float(
            np.mean(max_probabilities < 0.80)
        )

        # Entropia normalizzata delle probabilità.
        # 0 = assegnazione molto sicura
        # 1 = assegnazione molto incerta
        eps = 1e-12
        n_components = probabilities.shape[1]

        entropy = -np.sum(
            probabilities * np.log(probabilities + eps),
            axis=1,
        )

        if n_components > 1:
            normalized_entropy = entropy / np.log(n_components)
        else:
            normalized_entropy = entropy

        metrics["mean_normalized_entropy"] = float(
            np.mean(normalized_entropy)
        )
        metrics["median_normalized_entropy"] = float(
            np.median(normalized_entropy)
        )
        metrics["max_normalized_entropy"] = float(
            np.max(normalized_entropy)
        )

    else:
        metrics["mean_max_probability"] = np.nan
        metrics["median_max_probability"] = np.nan
        metrics["min_max_probability"] = np.nan
        metrics["max_max_probability"] = np.nan
        metrics["fraction_probability_below_0_50"] = np.nan
        metrics["fraction_probability_below_0_60"] = np.nan
        metrics["fraction_probability_below_0_70"] = np.nan
        metrics["fraction_probability_below_0_80"] = np.nan
        metrics["mean_normalized_entropy"] = np.nan
        metrics["median_normalized_entropy"] = np.nan
        metrics["max_normalized_entropy"] = np.nan

    return metrics

