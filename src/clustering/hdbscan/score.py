import numpy as np


def compute_clustering_score(
    dbcv,
    cosine_sim,
    labels,
    bad_cluster_weight,
    cosine_weight,
    noise_weight,
    noise_tolerance=0.5,
):
    labels = np.asarray(labels)

    dbcv_values = []
    cluster_sizes = []

    for cluster_id, cluster_dbcv in dbcv.items():
        cluster_size = np.sum(labels == cluster_id)

        if (
            cluster_id != -1
            and cluster_size > 0
            and np.isfinite(cluster_dbcv)
        ):
            dbcv_values.append(float(cluster_dbcv))
            cluster_sizes.append(cluster_size)

    if not dbcv_values:
        return np.nan

    dbcv_values = np.asarray(dbcv_values)
    cluster_sizes = np.asarray(cluster_sizes)

    mean_dbcv = np.average(
        dbcv_values,
        weights=cluster_sizes
    )

    negative_mask = dbcv_values < 0

    bad_cluster_penalty = (
        np.average(
            np.abs(dbcv_values[negative_mask]),
            weights=cluster_sizes[negative_mask]
        )
        if np.any(negative_mask)
        else 0.0
    )

    noise_fraction = np.mean(labels == -1)
    noise_penalty = max(0.0, noise_fraction - noise_tolerance)

    score = (
        mean_dbcv
        - bad_cluster_weight * bad_cluster_penalty
        + cosine_weight * cosine_sim
        - noise_weight * noise_penalty
    )

    return float(score)