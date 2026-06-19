import numpy as np
import pandas as pd

from umap import UMAP
from sklearn.cluster import HDBSCAN 

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

def evaluate_clustered_data(
    X,
    labels,
    noise_label=-1,
    ignore_noise=True,
    silhouette_metric="euclidean",
):
    """
    Valuta un clustering già eseguito.

    Pensata soprattutto per HDBSCAN, dove i punti noise hanno label -1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Dati già clusterizzati.
        Esempio: emb_umap.

    labels : array-like, shape (n_samples,)
        Label già prodotte dal clustering.
        Esempio: labels = hdbscan_model.fit_predict(emb_umap)

    noise_label : int, default=-1
        Label usata per indicare il noise.

    ignore_noise : bool, default=True
        Se True, esclude i punti noise dal calcolo delle metriche interne.

    silhouette_metric : str, default="euclidean"
        Metrica usata per la silhouette.

    Returns
    -------
    metrics : dict
        Dizionario con le metriche del clustering.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X ha {X.shape[0]} righe, ma labels ha lunghezza {labels.shape[0]}."
        )

    n_samples = len(labels)

    # -------------------------
    # Noise
    # -------------------------
    noise_mask = labels == noise_label
    non_noise_mask = labels != noise_label

    n_noise = int(np.sum(noise_mask))
    n_non_noise = int(np.sum(non_noise_mask))
    noise_ratio = float(n_noise / n_samples)

    # -------------------------
    # Cluster, escluso noise
    # -------------------------
    cluster_labels = labels[non_noise_mask]
    unique_clusters = np.unique(cluster_labels)
    n_clusters = int(len(unique_clusters))

    # -------------------------
    # Dimensioni cluster
    # -------------------------
    cluster_sizes = {}

    for cluster in unique_clusters:
        cluster_sizes[str(cluster)] = int(np.sum(cluster_labels == cluster))

    if n_clusters > 0:
        size_values = np.array(list(cluster_sizes.values()))

        min_cluster_size = int(size_values.min())
        max_cluster_size = int(size_values.max())
        mean_cluster_size = float(size_values.mean())
        median_cluster_size = float(np.median(size_values))
        largest_cluster_ratio = float(max_cluster_size / n_non_noise)
    else:
        min_cluster_size = None
        max_cluster_size = None
        mean_cluster_size = None
        median_cluster_size = None
        largest_cluster_ratio = None

    # -------------------------
    # Dati usati per le metriche
    # -------------------------
    if ignore_noise:
        X_eval = X[non_noise_mask]
        labels_eval = labels[non_noise_mask]
    else:
        X_eval = X
        labels_eval = labels

    n_eval_samples = len(labels_eval)
    n_eval_clusters = len(np.unique(labels_eval))

    # -------------------------
    # Metriche interne
    # -------------------------
    if n_eval_clusters >= 2 and n_eval_samples > n_eval_clusters:
        silhouette = float(
            silhouette_score(
                X_eval,
                labels_eval,
                metric=silhouette_metric,
            )
        )

        davies_bouldin = float(
            davies_bouldin_score(
                X_eval,
                labels_eval,
            )
        )

        calinski_harabasz = float(
            calinski_harabasz_score(
                X_eval,
                labels_eval,
            )
        )

        valid_internal_metrics = True

    else:
        silhouette = None
        davies_bouldin = None
        calinski_harabasz = None
        valid_internal_metrics = False

    metrics = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_non_noise": n_non_noise,
        "noise_ratio": noise_ratio,
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

    return metrics


def optimize_umap_hdbscan_parameters(
    X,
    umap_n_neighbors_values=(15, 30, 40, 50),
    umap_min_dist_values=(0.0, 0.01, 0.1, 0.3),
    umap_n_components_values=(2, 3, 5, 10, 15, 30),
    umap_metric_values=("euclidean",),
    hdbscan_min_cluster_sizes=(10, 15, 20, 25, 30, 40),
    hdbscan_min_samples_values=(5, 10, 15, 20),
    hdbscan_cluster_selection_methods=("eom",),
    min_clusters=2,
    max_clusters=10,
    max_noise_ratio=0.60,
    random_state=42,
):
    """
    Ottimizza UMAP + HDBSCAN tramite grid search.

    Parameters
    ----------
    X : array-like
        Dati prima di UMAP.
        Nel tuo caso, di solito: X_pca.

    Returns
    -------
    output : dict
        Dizionario con:
        - best_params
        - best_metrics
        - best_labels
        - best_embedding
        - results
    """

    X = np.asarray(X)

    results = []

    best_score = np.inf
    best_params = None
    best_metrics = None
    best_labels = None
    best_embedding = None

    config_counter = 0

    for umap_n_neighbors in umap_n_neighbors_values:
        for umap_min_dist in umap_min_dist_values:
            for umap_n_components in umap_n_components_values:
                for umap_metric in umap_metric_values:

                    # -------------------------
                    # UMAP
                    # -------------------------
                    reducer = UMAP(
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        n_components=umap_n_components,
                        metric=umap_metric,
                        random_state=random_state,
                    )

                    X_umap = reducer.fit_transform(X)

                    for hdbscan_min_cluster_size in hdbscan_min_cluster_sizes:
                        for hdbscan_min_samples in hdbscan_min_samples_values:
                            for hdbscan_method in hdbscan_cluster_selection_methods:

                                if hdbscan_min_samples > hdbscan_min_cluster_size:
                                    continue

                                config_counter += 1

                                config_key = f"config_{config_counter}"

                                # -------------------------
                                # HDBSCAN
                                # -------------------------
                                clusterer = HDBSCAN(
                                    min_cluster_size=hdbscan_min_cluster_size,
                                    min_samples=hdbscan_min_samples,
                                    metric="euclidean",
                                    cluster_selection_method=hdbscan_method,
                                    allow_single_cluster=False,
                                    n_jobs=-1,
                                )

                                labels = clusterer.fit_predict(X_umap)

                                # -------------------------
                                # Metriche
                                # -------------------------
                                metrics = evaluate_clustered_data(
                                    X=X_umap,
                                    labels=labels,
                                    noise_label=-1,
                                    ignore_noise=True,
                                    silhouette_metric="euclidean",
                                )

                                valid_solution = (
                                    metrics["valid_internal_metrics"] is True
                                    and metrics["n_clusters"] >= min_clusters
                                    and metrics["n_clusters"] <= max_clusters
                                    and metrics["noise_ratio"] <= max_noise_ratio
                                )

                                row = {
                                    "config_key": config_key,

                                    "umap_n_neighbors": umap_n_neighbors,
                                    "umap_min_dist": umap_min_dist,
                                    "umap_n_components": umap_n_components,
                                    "umap_metric": umap_metric,

                                    "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
                                    "hdbscan_min_samples": hdbscan_min_samples,
                                    "hdbscan_cluster_selection_method": hdbscan_method,

                                    "valid_solution": valid_solution,
                                }

                                for key, value in metrics.items():
                                    row[key] = value

                                results.append(row)

    results_df = pd.DataFrame(results)

    valid_df = results_df[results_df["valid_solution"]].copy()

    if len(valid_df) == 0:
        print("Nessuna configurazione valida trovata.")
        print("Prova ad aumentare max_noise_ratio oppure ad ampliare la griglia.")

        return {
            "best_params": None,
            "best_metrics": None,
            "best_labels": None,
            "best_embedding": None,
            "results": results_df,
        }

    # -------------------------
    # Ranking delle metriche
    # -------------------------
    valid_df["rank_silhouette"] = valid_df["silhouette"].rank(
        ascending=False
    )

    valid_df["rank_davies_bouldin"] = valid_df["davies_bouldin"].rank(
        ascending=True
    )

    valid_df["rank_calinski_harabasz"] = valid_df["calinski_harabasz"].rank(
        ascending=False
    )

    valid_df["rank_noise"] = valid_df["noise_ratio"].rank(
        ascending=True
    )

    valid_df["rank_largest_cluster"] = valid_df["largest_cluster_ratio"].rank(
        ascending=True
    )

    # Più basso = meglio
    valid_df["final_rank_score"] = (
        0.35 * valid_df["rank_silhouette"]
        + 0.25 * valid_df["rank_davies_bouldin"]
        + 0.25 * valid_df["rank_calinski_harabasz"]
        + 0.10 * valid_df["rank_noise"]
        + 0.05 * valid_df["rank_largest_cluster"]
    )

    # Reinserisco i ranking nel dataframe completo
    rank_columns = [
        "rank_silhouette",
        "rank_davies_bouldin",
        "rank_calinski_harabasz",
        "rank_noise",
        "rank_largest_cluster",
        "final_rank_score",
    ]

    for col in rank_columns:
        results_df[col] = np.nan
        results_df.loc[valid_df.index, col] = valid_df[col]

    results_df = results_df.sort_values(
        by=["valid_solution", "final_rank_score"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # -------------------------
    # Migliore configurazione
    # -------------------------
    best_row = results_df.iloc[0]

    best_params = {
        "umap": {
            "n_neighbors": int(best_row["umap_n_neighbors"]),
            "min_dist": float(best_row["umap_min_dist"]),
            "n_components": int(best_row["umap_n_components"]),
            "metric": best_row["umap_metric"],
        },
        "hdbscan": {
            "min_cluster_size": int(best_row["hdbscan_min_cluster_size"]),
            "min_samples": int(best_row["hdbscan_min_samples"]),
            "cluster_selection_method": best_row["hdbscan_cluster_selection_method"],
            "metric": "euclidean",
        },
    }

    # Ricostruisco il miglior embedding e le migliori label
    best_reducer = UMAP(
        n_neighbors=best_params["umap"]["n_neighbors"],
        min_dist=best_params["umap"]["min_dist"],
        n_components=best_params["umap"]["n_components"],
        metric=best_params["umap"]["metric"],
        random_state=random_state,
    )

    best_embedding = best_reducer.fit_transform(X)

    best_clusterer = HDBSCAN(
        min_cluster_size=best_params["hdbscan"]["min_cluster_size"],
        min_samples=best_params["hdbscan"]["min_samples"],
        metric="euclidean",
        cluster_selection_method=best_params["hdbscan"]["cluster_selection_method"],
        allow_single_cluster=False,
        n_jobs=-1,
    )

    best_labels = best_clusterer.fit_predict(best_embedding)

    best_metrics = evaluate_clustered_data(
        X=best_embedding,
        labels=best_labels,
        noise_label=-1,
        ignore_noise=True,
        silhouette_metric="euclidean",
    )

    output = {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_labels": best_labels,
        "best_embedding": best_embedding,
        "results": results_df,
    }

    return output