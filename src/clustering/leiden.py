import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

import igraph as ig
import leidenalg as la
import optuna


def build_knn_adjacency(
    X,
    n_neighbors=30,
    metric="euclidean",
    weighted=True,
    symmetrize=True,
):
    """
    Costruisce una matrice di adiacenza kNN a partire dagli embeddings.

    Parameters
    ----------
    X : np.ndarray
        Embeddings.

    n_neighbors : int
        Numero di vicini del grafo kNN.

    metric : str
        Metrica usata per il kNN.

    weighted : bool
        Se True, gli archi vengono pesati con una funzione gaussiana
        della distanza.
        Se False, il grafo è binario.

    symmetrize : bool
        Se True, rende il grafo non diretto.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        Matrice di adiacenza sparse.
    """

    X = np.asarray(X)

    if weighted:
        adjacency = kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="distance",
            metric=metric,
            include_self=False,
        )

        distances = adjacency.data

        if len(distances) == 0:
            raise ValueError("Il grafo kNN non contiene archi.")

        sigma = np.median(distances)

        if sigma <= 0:
            sigma = np.mean(distances)

        if sigma <= 0:
            sigma = 1.0

        adjacency.data = np.exp(
            -((adjacency.data ** 2) / (2 * sigma ** 2))
        )

    else:
        adjacency = kneighbors_graph(
            X,
            n_neighbors=n_neighbors,
            mode="connectivity",
            metric=metric,
            include_self=False,
        )

    if symmetrize:
        adjacency = adjacency.maximum(adjacency.T)

    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    return adjacency.tocsr()



def adjacency_to_igraph(adjacency):
    """
    Converte una matrice di adiacenza scipy sparse in un grafo igraph.
    """

    if not sparse.issparse(adjacency):
        adjacency = sparse.csr_matrix(adjacency)

    adjacency = adjacency.tocsr()

    # Tengo solo triangolare superiore per evitare duplicati
    adjacency_upper = sparse.triu(adjacency, k=1).tocsr()

    sources, targets = adjacency_upper.nonzero()
    weights = adjacency_upper.data.astype(float)

    edges = list(zip(sources.tolist(), targets.tolist()))

    graph = ig.Graph(
        n=adjacency.shape[0],
        edges=edges,
        directed=False,
    )

    graph.es["weight"] = weights.tolist()

    return graph


def run_leiden_clustering(
    X,
    n_neighbors=30,
    metric="euclidean",
    weighted=True,
    resolution=1.0,
    partition_type="RBConfiguration",
    n_iterations=2,
    random_state=42,
):
    """
    Esegue Leiden clustering usando leidenalg + igraph.

    Parameters
    ----------
    X : np.ndarray
        Embeddings.

    n_neighbors : int
        Numero di vicini per il grafo kNN.

    metric : str
        Metrica del grafo kNN.

    weighted : bool
        Se True usa archi pesati.

    resolution : float
        Parametro di resolution di Leiden.

    partition_type : str
        Tipo di partizione:
        - "RBConfiguration"
        - "CPM"
        - "Modularity"

    n_iterations : int
        Numero di iterazioni Leiden.

    random_state : int
        Seed.

    Returns
    -------
    output : dict
        Dizionario con labels, graph, partition, adjacency e parametri.
    """

    adjacency = build_knn_adjacency(
        X=X,
        n_neighbors=n_neighbors,
        metric=metric,
        weighted=weighted,
        symmetrize=True,
    )

    graph = adjacency_to_igraph(adjacency)

    partition_classes = {
        "RBConfiguration": la.RBConfigurationVertexPartition,
        "CPM": la.CPMVertexPartition,
        "Modularity": la.ModularityVertexPartition,
    }

    if partition_type not in partition_classes:
        raise ValueError(
            f"partition_type deve essere uno tra: "
            f"{list(partition_classes.keys())}"
        )

    partition_class = partition_classes[partition_type]

    leiden_kwargs = {
        "weights": "weight",
        "seed": random_state,
        "n_iterations": n_iterations,
    }

    if partition_type in ["RBConfiguration", "CPM"]:
        leiden_kwargs["resolution_parameter"] = resolution

    partition = la.find_partition(
        graph,
        partition_class,
        **leiden_kwargs,
    )

    labels = np.asarray(partition.membership)

    try:
        quality = float(partition.quality())
    except Exception:
        quality = np.nan

    return {
        "labels": labels,
        "graph": graph,
        "adjacency": adjacency,
        "partition": partition,
        "quality": quality,
        "params": {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "weighted": weighted,
            "resolution": resolution,
            "partition_type": partition_type,
            "n_iterations": n_iterations,
            "random_state": random_state,
        },
    }


def evaluate_leiden_clustered_data(
    X,
    labels,
    quality=np.nan,
    silhouette_metric="euclidean",
):
    """
    Valuta un clustering Leiden con metriche interne.

    Parameters
    ----------
    X : np.ndarray
        Spazio su cui valutare il clustering.

    labels : array-like
        Label dei cluster.

    quality : float
        Valore della funzione obiettivo Leiden, se disponibile.

    silhouette_metric : str
        Metrica per silhouette_score.

    Returns
    -------
    metrics : dict
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    unique_labels, counts = np.unique(labels, return_counts=True)

    n_clusters = len(unique_labels)
    n_samples = len(labels)

    cluster_sizes = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts)
    }

    min_cluster_size = int(counts.min())
    max_cluster_size = int(counts.max())

    if n_clusters >= 2 and n_samples > n_clusters:
        silhouette = float(
            silhouette_score(
                X,
                labels,
                metric=silhouette_metric,
            )
        )

        davies_bouldin = float(
            davies_bouldin_score(
                X,
                labels,
            )
        )

        calinski_harabasz = float(
            calinski_harabasz_score(
                X,
                labels,
            )
        )

        valid_internal_metrics = True

    else:
        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan
        valid_internal_metrics = False

    metrics = {
        "n_samples": int(n_samples),
        "n_clusters": int(n_clusters),
        "cluster_sizes": cluster_sizes,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "quality": float(quality),
        "valid_internal_metrics": valid_internal_metrics,
    }

    return metrics


def _normalized_rank(values, ascending=True):
    """
    Rank normalizzato tra 0 e 1.

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

    return (ranks - 1) / (len(values) - 1)


def select_best_leiden_result(
    results_df,
    weights=None,
):
    """
    Seleziona il miglior risultato Leiden usando rank multi-obiettivo.

    Metriche:
    - silhouette: alta è meglio
    - davies_bouldin: basso è meglio
    - calinski_harabasz: alto è meglio
    - quality: alta è meglio
    """

    if weights is None:
        weights = {
            "silhouette": 0.40,
            "davies_bouldin": 0.30,
            "calinski_harabasz": 0.20,
            "quality": 0.10,
        }

    directions = {
        "silhouette": False,
        "davies_bouldin": True,
        "calinski_harabasz": False,
        "quality": False,
    }

    valid_results = results_df[
        results_df["valid_solution"].astype(bool)
    ].copy()

    for metric in weights.keys():
        valid_results = valid_results[
            np.isfinite(valid_results[metric])
        ]

    if len(valid_results) == 0:
        raise ValueError(
            "Nessuna soluzione valida disponibile per la selezione."
        )

    valid_results["selection_score"] = 0.0

    for metric, weight in weights.items():

        rank_column = f"rank_norm_{metric}"

        valid_results[rank_column] = _normalized_rank(
            valid_results[metric],
            ascending=directions[metric],
        )

        valid_results["selection_score"] += (
            weight * valid_results[rank_column]
        )

    valid_results = valid_results.sort_values(
        by="selection_score",
        ascending=True,
    ).reset_index(drop=True)

    valid_results["rank_selection"] = np.arange(
        1,
        len(valid_results) + 1,
    )

    return valid_results

def optimize_leiden_parameters_optuna(
    X,
    leiden_n_neighbors_values=(15, 20, 30, 40, 50),
    leiden_resolution_values=(0.4, 0.6, 0.8, 1.0, 1.2, 1.5),
    leiden_metric_values=("euclidean",),
    leiden_weighted_values=(True,),
    leiden_partition_type_values=("RBConfiguration",),
    min_clusters=4,
    max_clusters=10,
    min_cluster_size=10,
    random_state=42,
    n_trials=50,
    n_iterations=2,
    primary_metric="silhouette",
    multiobjective_weights=None,
    study_name="leiden_optimization",
):
    """
    Ottimizza Leiden usando Optuna.

    Nota:
    Optuna viene usato per esplorare lo spazio dei parametri.
    La scelta finale viene fatta con selezione multi-obiettivo
    basata su rank normalizzati.

    Parameters
    ----------
    X : np.ndarray
        Embeddings.

    leiden_n_neighbors_values : iterable
        Valori di n_neighbors del grafo kNN.

    leiden_resolution_values : iterable
        Valori di resolution Leiden.

    leiden_metric_values : iterable
        Metriche kNN.

    leiden_weighted_values : iterable
        True/False per grafo pesato.

    leiden_partition_type_values : iterable
        Tipi di partizione Leiden.

    min_clusters : int
        Numero minimo di cluster.

    max_clusters : int
        Numero massimo di cluster.

    min_cluster_size : int
        Dimensione minima di ogni cluster.

    random_state : int
        Seed base.

    n_trials : int
        Numero di trial Optuna.

    n_iterations : int
        Iterazioni Leiden.

    primary_metric : str
        Metrica usata da Optuna durante la ricerca.
        La selezione finale resta multi-obiettivo.

    multiobjective_weights : dict or None
        Pesi per la selezione finale.

    study_name : str
        Nome dello studio Optuna.

    Returns
    -------
    output : dict
        Dizionario con best_params, best_metrics, best_labels, results.
    """

    X = np.asarray(X)

    maximize_metrics = {
        "silhouette",
        "calinski_harabasz",
        "quality",
    }

    minimize_metrics = {
        "davies_bouldin",
    }

    if primary_metric not in maximize_metrics | minimize_metrics:
        raise ValueError(
            f"primary_metric deve essere uno tra: "
            f"{maximize_metrics | minimize_metrics}"
        )

    records = []
    objects_by_trial = {}

    def objective(trial):

        n_neighbors = trial.suggest_categorical(
            "n_neighbors",
            list(leiden_n_neighbors_values),
        )

        resolution = trial.suggest_categorical(
            "resolution",
            list(leiden_resolution_values),
        )

        metric = trial.suggest_categorical(
            "metric",
            list(leiden_metric_values),
        )

        weighted = trial.suggest_categorical(
            "weighted",
            list(leiden_weighted_values),
        )

        partition_type = trial.suggest_categorical(
            "partition_type",
            list(leiden_partition_type_values),
        )

        trial_seed = random_state + trial.number

        row = {
            "trial_number": int(trial.number),
            "n_neighbors": n_neighbors,
            "resolution": resolution,
            "metric": metric,
            "weighted": weighted,
            "partition_type": partition_type,
            "n_iterations": n_iterations,
            "random_state": trial_seed,
        }

        try:
            clustering = run_leiden_clustering(
                X=X,
                n_neighbors=n_neighbors,
                metric=metric,
                weighted=weighted,
                resolution=resolution,
                partition_type=partition_type,
                n_iterations=n_iterations,
                random_state=trial_seed,
            )

            labels = clustering["labels"]

            metrics = evaluate_leiden_clustered_data(
                X=X,
                labels=labels,
                quality=clustering["quality"],
                silhouette_metric="euclidean",
            )

            valid_solution = (
                metrics["n_clusters"] >= min_clusters
                and metrics["n_clusters"] <= max_clusters
                and metrics["min_cluster_size"] >= min_cluster_size
                and metrics["valid_internal_metrics"]
                and np.isfinite(metrics[primary_metric])
            )

            row.update(metrics)
            row["valid_solution"] = bool(valid_solution)
            row["error"] = None

            objects_by_trial[trial.number] = {
                "labels": labels,
                "clustering": clustering,
            }

            records.append(row)

            if not valid_solution:
                return -1e9

            primary_value = metrics[primary_metric]

            if primary_metric in maximize_metrics:
                return float(primary_value)

            if primary_metric in minimize_metrics:
                return float(-primary_value)

        except Exception as error:

            row.update(
                {
                    "n_clusters": np.nan,
                    "min_cluster_size": np.nan,
                    "max_cluster_size": np.nan,
                    "silhouette": np.nan,
                    "davies_bouldin": np.nan,
                    "calinski_harabasz": np.nan,
                    "quality": np.nan,
                    "valid_internal_metrics": False,
                    "valid_solution": False,
                    "error": str(error),
                }
            )

            records.append(row)

            return -1e9

    sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    results_df = pd.DataFrame(records)

    if len(results_df) == 0:
        raise ValueError("Nessun trial eseguito correttamente.")

    selected_results = select_best_leiden_result(
        results_df,
        weights=multiobjective_weights,
    )

    best_row = selected_results.iloc[0]
    best_trial_number = int(best_row["trial_number"])

    best_object = objects_by_trial[best_trial_number]
    best_labels = best_object["labels"]
    best_clustering = best_object["clustering"]

    best_params = {
        "leiden": {
            "n_neighbors": int(best_row["n_neighbors"]),
            "resolution": float(best_row["resolution"]),
            "metric": best_row["metric"],
            "weighted": bool(best_row["weighted"]),
            "partition_type": best_row["partition_type"],
            "n_iterations": int(best_row["n_iterations"]),
            "random_state": int(best_row["random_state"]),
            "selection": "multiobjective_rank",
        }
    }

    if multiobjective_weights is not None:
        best_params["leiden"]["multiobjective_weights"] = (
            multiobjective_weights
        )

    best_metrics = best_row.to_dict()

    results_df = results_df.sort_values(
        by=["valid_solution"],
        ascending=[False],
    ).reset_index(drop=True)

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_labels": best_labels,
        "best_clustering": best_clustering,
        "best_graph": best_clustering["graph"],
        "best_adjacency": best_clustering["adjacency"],
        "study": study,
        "results": results_df,
        "selected_results": selected_results,
    }