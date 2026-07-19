import numpy as np


def mean_intracluster_cosine(
    X: np.ndarray,
    labels: np.ndarray,
    noise_label: int | None = -1,
    min_cluster_size: int = 2,
):
    """
    Compute the pair-weighted mean cosine similarity within clusters.

    Parameters
    ----------
    X : np.ndarray
        L2-normalized sample embeddings.

    labels : np.ndarray
        Cluster label for each sample.

    noise_label : int or None
        Label to exclude as noise, or ``None`` to include every label.

    min_cluster_size : int
        Ignore clusters smaller than this value.

    Returns
    -------
    float
        Weighted mean similarity, or NaN if no cluster can be evaluated.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    unique_labels = set(labels)

    if noise_label is not None:
        unique_labels = unique_labels - {noise_label}

    cluster_scores = []
    cluster_weights = []

    for c in sorted(unique_labels):
        idx = np.where(labels == c)[0]
        n = len(idx)

        if n < min_cluster_size:
            continue

        Xc = X[idx]

        # Derive the pairwise sum from the cluster's vector sum.
        s = Xc.sum(axis=0)

        total_sim_including_diag = np.dot(s, s)

        total_sim_excluding_diag = total_sim_including_diag - n

        n_pairs = n * (n - 1)

        mean_cos = total_sim_excluding_diag / n_pairs

        cluster_scores.append(mean_cos)
        cluster_weights.append(n_pairs)

    if len(cluster_scores) == 0:
        return np.nan

    cluster_scores = np.asarray(cluster_scores)
    cluster_weights = np.asarray(cluster_weights)

    return float(np.average(cluster_scores, weights=cluster_weights))
