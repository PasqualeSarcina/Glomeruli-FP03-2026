import numpy as np


def mean_intracluster_cosine(
    X: np.ndarray,
    labels: np.ndarray,
    noise_label: int | None = -1,
    min_cluster_size: int = 2,
):
    """
    Calcola la cosine similarity media intra-cluster.

    X deve essere già L2-normalizzato.
    labels può venire da HDBSCAN, Leiden, KMeans, GMM, consensus, ecc.

    Parameters
    ----------
    X : np.ndarray
        Embedding pre-UMAP, preferibilmente L2-normalizzato.

    labels : np.ndarray
        Labels del clustering.

    noise_label : int or None
        Label del rumore.
        - HDBSCAN: noise_label=-1
        - Leiden: noise_label=None oppure -1 se non hai noise

    min_cluster_size : int
        Cluster più piccoli di questa soglia vengono ignorati.

    Returns
    -------
    float
        Cosine similarity media intra-cluster, pesata per numero di coppie.
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

        # Somma vettoriale degli embedding del cluster
        s = Xc.sum(axis=0)

        # Somma di tutte le cosine similarity incluse le diagonali
        total_sim_including_diag = np.dot(s, s)

        # Rimuovo le self-similarity = 1
        total_sim_excluding_diag = total_sim_including_diag - n

        # Numero di coppie ordinate i != j
        n_pairs = n * (n - 1)

        mean_cos = total_sim_excluding_diag / n_pairs

        cluster_scores.append(mean_cos)
        cluster_weights.append(n_pairs)

    if len(cluster_scores) == 0:
        return np.nan

    cluster_scores = np.asarray(cluster_scores)
    cluster_weights = np.asarray(cluster_weights)

    return float(np.average(cluster_scores, weights=cluster_weights))