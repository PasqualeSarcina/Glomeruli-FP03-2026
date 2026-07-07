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
    cluster_sizes = {
        relabelled_cluster: int(np.sum(labels == relabelled_cluster))
        for relabelled_cluster in valid_clusters
    }
    evaluable_clusters = [
        relabelled_cluster
        for relabelled_cluster in valid_clusters
        if cluster_sizes[relabelled_cluster] >= 2
    ]

    if n_clusters < 2:
        rows = []

        # eventuale riga rumore
        if noise_size > 0:
            rows.append({
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
            })

        # eventuale unico cluster non-noise
        for relabelled_cluster in valid_clusters:
            original_cluster = new_to_old[relabelled_cluster]

            rows.append({
                "cluster": original_cluster,
                "size": int(np.sum(labels == relabelled_cluster)),
                "dbcv_mean": np.nan,
                "dbcv_std": np.nan,
                "dbcv_min": np.nan,
                "dbcv_max": np.nan,
                "dbcv_median": np.nan,
                "n_runs_valid": 0,
                "global_dbcv_mean": np.nan,
                "global_dbcv_std": np.nan,
            })

        return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)

    rows = []

    for run_idx, emb in enumerate(umap_embeddings, start=1):
        X = np.ascontiguousarray(emb, dtype=np.float64)

        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Run {run_idx}: embedding con {X.shape[0]} campioni, "
                f"ma labels con {labels.shape[0]} campioni."
            )

        # hdbscan.validity_index fails on singleton clusters because their
        # internal MST is empty. Keep them in the output with NaN DBCV.
        if len(evaluable_clusters) >= 2:
            score_index_by_cluster = {
                relabelled_cluster: score_index
                for score_index, relabelled_cluster in enumerate(evaluable_clusters)
            }

            if remove_noise:
                mask = np.isin(labels, evaluable_clusters)
            else:
                mask = np.ones(labels.shape, dtype=bool)

            X_eval = X[mask]
            labels_eval_source = labels[mask]
            labels_eval = np.full(labels_eval_source.shape, -1, dtype=np.int64)

            for relabelled_cluster, score_index in score_index_by_cluster.items():
                labels_eval[labels_eval_source == relabelled_cluster] = score_index

            labels_eval = np.ascontiguousarray(labels_eval, dtype=np.int64)

            global_dbcv, cluster_scores = validity_index(
                X_eval,
                labels_eval,
                metric=metric,
                per_cluster_scores=True
            )

            cluster_scores = np.asarray(cluster_scores, dtype=np.float64)
        else:
            score_index_by_cluster = {}
            global_dbcv = np.nan
            cluster_scores = np.array([], dtype=np.float64)

        for relabelled_cluster in valid_clusters:
            original_cluster = new_to_old[relabelled_cluster]
            score_index = score_index_by_cluster.get(relabelled_cluster)
            dbcv = (
                float(cluster_scores[score_index])
                if score_index is not None and score_index < len(cluster_scores)
                else np.nan
            )

            rows.append({
                "run": run_idx,
                "cluster": original_cluster,
                "cluster_relabelled": relabelled_cluster,
                "size": cluster_sizes[relabelled_cluster],
                "global_dbcv": float(global_dbcv),
                "dbcv": dbcv,
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

    if noise_size > 0:
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
