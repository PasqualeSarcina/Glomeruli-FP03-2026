import numpy as np


def compute_leiden_clustering_score(
    modularity,
    cosine_sim,
    labels,
    bad_cluster_weight,
    cosine_weight,
):
    """
    Compute a Leiden consensus clustering score.

    Parameters
    ----------
    modularity : dict
        Mapping cluster_id -> mean modularity contribution,
        returned by modularity_cluster_consensus().

    cosine_sim : float
        Mean intra-cluster cosine similarity, preferably computed
        on the pre-UMAP L2-normalized embedding.

    labels : array-like
        Consensus Leiden labels.

    bad_cluster_weight : float
        Weight applied to negative modularity contributions.

    cosine_weight : float
        Weight applied to intra-cluster cosine similarity.

    Returns
    -------
    float
        Final clustering score.
    """

    labels = np.asarray(labels)

    modularity_values = []

    for cluster_id, cluster_modularity in modularity.items():
        cluster_size = np.sum(labels == cluster_id)

        if (
            cluster_id != -1
            and cluster_size > 0
            and np.isfinite(cluster_modularity)
        ):
            modularity_values.append(float(cluster_modularity))

    if not modularity_values:
        return np.nan

    modularity_values = np.asarray(
        modularity_values,
        dtype=float,
    )

    # I contributi dei cluster devono essere sommati:
    # la loro somma corrisponde alla modularità globale.
    global_modularity = np.sum(modularity_values)

    negative_mask = modularity_values < 0

    # Non si applicano nuovamente i pesi delle dimensioni:
    # il contributo di modularità incorpora già il volume del cluster.
    bad_cluster_penalty = (
        np.sum(np.abs(modularity_values[negative_mask]))
        if np.any(negative_mask)
        else 0.0
    )

    score = (
        global_modularity
        - bad_cluster_weight * bad_cluster_penalty
        + cosine_weight * cosine_sim
    )

    return float(score)