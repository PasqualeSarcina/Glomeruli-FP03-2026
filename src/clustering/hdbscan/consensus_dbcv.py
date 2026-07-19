import numpy as np
from hdbscan.validity import validity_index


def dbcv_cluster_consensus(
    consensus_labels,
    umap_embeddings,
    metric="euclidean",
    remove_noise=True,
):
    """
    Average per-cluster DBCV scores across one or more UMAP embeddings.

    Returns
    -------
    dict
        Mapping from cluster label to mean DBCV score. Noise is excluded.
    """

    labels = np.asarray(consensus_labels)

    # Accept either one embedding or a sequence of embeddings.
    if isinstance(umap_embeddings, np.ndarray):
        if umap_embeddings.ndim == 2:
            umap_embeddings = [umap_embeddings]
        else:
            raise ValueError(
                "umap_embeddings must be a 2D array or a sequence of 2D arrays."
            )

    clusters = sorted(set(labels) - {-1})

    dbcv_values = {c: [] for c in clusters}

    # DBCV requires at least two clusters with at least two samples each.
    evaluable_clusters = [
        c for c in clusters
        if np.sum(labels == c) >= 2
    ]

    if len(evaluable_clusters) < 2:
        return {c: np.nan for c in clusters}

    for emb in umap_embeddings:
        X = np.asarray(emb, dtype=np.float64)

        if X.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embedding has {X.shape[0]} samples, but labels has "
                f"{labels.shape[0]}."
            )

        if remove_noise:
            mask = np.isin(labels, evaluable_clusters)
        else:
            mask = labels != -1

        X_eval = np.ascontiguousarray(X[mask], dtype=np.float64)
        labels_eval_source = labels[mask]

        # validity_index expects consecutive cluster labels.
        cluster_to_new = {
            old_c: new_c
            for new_c, old_c in enumerate(evaluable_clusters)
        }

        labels_eval = np.array(
            [cluster_to_new[c] for c in labels_eval_source],
            dtype=np.int64,
        )

        labels_eval = np.ascontiguousarray(labels_eval, dtype=np.int64)

        _, cluster_scores = validity_index(
            X_eval,
            labels_eval,
            metric=metric,
            per_cluster_scores=True,
        )

        for old_c, new_c in cluster_to_new.items():
            dbcv_values[old_c].append(float(cluster_scores[new_c]))

    dbcv_dict = {}

    for c in clusters:
        values = dbcv_values[c]

        if len(values) == 0:
            dbcv_dict[c] = np.nan
        else:
            dbcv_dict[c] = float(np.mean(values))

    return dbcv_dict
