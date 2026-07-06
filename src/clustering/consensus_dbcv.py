import numpy as np
import pandas as pd
from hdbscan.validity import validity_index


def _relabel_contiguous(labels):
    """
    Converte labels arbitrarie in labels contigue 0, 1, 2, ...
    mantenendo -1 come rumore.

    Esempio:
    [-1, 3, 3, 8, 8] -> [-1, 0, 0, 1, 1]
    """
    labels = np.asarray(labels)
    new_labels = np.full(labels.shape, -1, dtype=int)

    valid_old_labels = sorted(set(labels) - {-1})

    old_to_new = {
        old_label: new_label
        for new_label, old_label in enumerate(valid_old_labels)
    }

    new_to_old = {
        new_label: old_label
        for old_label, new_label in old_to_new.items()
    }

    for old_label, new_label in old_to_new.items():
        new_labels[labels == old_label] = new_label

    return new_labels, new_to_old


def dbcv_per_cluster_consensus(
    consensus_labels,
    umap_embeddings,
    metric="euclidean",
    remove_noise=True
):
    """
    Calcola il DBCV per cluster delle labels finali del consensus
    su uno o più embedding UMAP.

    Parameters
    ----------
    consensus_labels : array-like, shape (n_samples,)
        Labels finali del consensus clustering.
        Il rumore deve essere indicato con -1.

    umap_embeddings : list[np.ndarray] oppure np.ndarray
        Lista degli embedding UMAP usati nelle run.
        Esempio:
        [umap_run1, umap_run2, ..., umap_run30]

        Può anche essere un singolo embedding UMAP.

    metric : str, default="euclidean"
        Metrica usata da DBCV.

    remove_noise : bool, default=True
        Se True, rimuove i punti con label -1 prima del calcolo.
        Consigliato per il DBCV per cluster.

    Returns
    -------
    df_summary : pd.DataFrame
        DataFrame con una riga per cluster.
    """

    labels_original = np.asarray(consensus_labels)

    if isinstance(umap_embeddings, np.ndarray):
        if umap_embeddings.ndim == 2:
            umap_embeddings = [umap_embeddings]
        else:
            raise ValueError(
                "umap_embeddings deve essere un array 2D oppure una lista di array 2D."
            )

    labels, new_to_old = _relabel_contiguous(labels_original)

    valid_clusters = sorted(set(labels) - {-1})
    n_clusters = len(valid_clusters)
    noise_size = int(np.sum(labels == -1))

    if n_clusters < 2:
        raise ValueError("DBCV richiede almeno 2 cluster non-noise.")

    rows = []

    for run_idx, emb in enumerate(umap_embeddings, start=1):
        X = np.ascontiguousarray(emb, dtype=np.float64)

        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Run {run_idx}: embedding con {X.shape[0]} campioni, "
                f"ma labels con {labels.shape[0]} campioni."
            )

        if remove_noise:
            mask = labels != -1
            X_eval = X[mask]
            labels_eval = np.ascontiguousarray(labels[mask], dtype=np.int64)
        else:
            X_eval = X
            labels_eval = np.ascontiguousarray(labels, dtype=np.int64)

        global_dbcv, cluster_scores = validity_index(
            X_eval,
            labels_eval,
            metric=metric,
            per_cluster_scores=True
        )

        cluster_scores = np.asarray(cluster_scores)

        for relabelled_cluster in valid_clusters:
            original_cluster = new_to_old[relabelled_cluster]

            rows.append({
                "run": run_idx,
                "cluster": original_cluster,
                "cluster_relabelled": relabelled_cluster,
                "size": int(np.sum(labels == relabelled_cluster)),
                "global_dbcv": float(global_dbcv),
                "dbcv": float(cluster_scores[relabelled_cluster]),
            })

    df_detail = pd.DataFrame(rows)

    df_summary = (
        df_detail
        .groupby(["cluster", "size"], as_index=False)
        .agg(
            dbcv_mean=("dbcv", "mean"),
            dbcv_std=("dbcv", "std"),
            dbcv_min=("dbcv", "min"),
            dbcv_max=("dbcv", "max"),
            dbcv_median=("dbcv", "median"),
            n_runs_valid=("dbcv", lambda x: x.notna().sum()),
            global_dbcv_mean=("global_dbcv", "mean"),
            global_dbcv_std=("global_dbcv", "std"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    noise_row = pd.DataFrame([{
        "cluster": -1,
        "size": noise_size,
        "dbcv_mean": np.nan,
        "dbcv_std": np.nan,
        "dbcv_min": np.nan,
        "dbcv_max": np.nan,
        "dbcv_median": np.nan,
        "n_runs_valid": 0,
        "global_dbcv_mean": np.nan,
        "global_dbcv_std": np.nan,
    }])

    df_summary = pd.concat([noise_row, df_summary], ignore_index=True)

    return df_summary