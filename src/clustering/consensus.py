import numpy as np
from collections import Counter
from sklearn.cluster import AgglomerativeClustering


def consensus_clustering(
        label_runs,
        n_clusters="median",
        noise_label=-1,
        max_noise_frequency=0.50,
        min_consensus_strength=0.30,
        min_final_cluster_size=5,
        allow_final_noise=True,
):
    """
    Build consensus labels from multiple clustering runs.

    Parameters
    ----------
    label_runs : array-like, shape (n_runs, n_samples)
        Cluster labels from each run.

    n_clusters : int, "median", or "mode"
        Final cluster count or a rule for deriving it from the runs.

    noise_label : int or None
        Label used for noise. Use ``None`` if noise is not defined.

    max_noise_frequency : float
        Maximum fraction of runs in which a sample may be noise.

    min_consensus_strength : float
        Minimum mean co-association within the final cluster.

    min_final_cluster_size : int
        Minimum size retained when final noise is allowed.

    allow_final_noise : bool
        Mark weak samples and small clusters as ``-1`` when true.

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Final consensus labels.
    """

    label_runs = np.asarray(label_runs)

    if label_runs.ndim != 2:
        raise ValueError("label_runs must have shape (n_runs, n_samples).")

    n_runs, n_samples = label_runs.shape

    if n_runs < 2:
        raise ValueError("Consensus clustering requires at least two runs.")

    # Count clusters in each run.
    ks = []

    for labels in label_runs:
        labels = np.asarray(labels)

        unique_labels = set(labels)

        if noise_label is not None:
            unique_labels = unique_labels - {noise_label}

        ks.append(len(unique_labels))

    ks_valid = [k for k in ks if k > 0]

    if len(ks_valid) == 0:
        return np.full(n_samples, -1, dtype=int)

    if n_clusters == "median":
        n_clusters_final = int(round(np.median(ks_valid)))

    elif n_clusters == "mode":
        n_clusters_final = Counter(ks_valid).most_common(1)[0][0]

    elif isinstance(n_clusters, int):
        n_clusters_final = int(n_clusters)

    else:
        raise ValueError("n_clusters must be an int, 'median', or 'mode'.")

    n_clusters_final = max(1, n_clusters_final)

    # Build the co-association matrix.
    same_cluster = np.zeros((n_samples, n_samples), dtype=np.float32)
    co_observed = np.zeros((n_samples, n_samples), dtype=np.float32)

    for labels in label_runs:
        labels = np.asarray(labels)

        if noise_label is None:
            valid_mask = np.ones(n_samples, dtype=bool)
        else:
            valid_mask = labels != noise_label

        valid_idx = np.where(valid_mask)[0]

        co_observed[np.ix_(valid_idx, valid_idx)] += 1.0

        unique_labels = sorted(set(labels[valid_mask]))

        for c in unique_labels:
            idx = np.where(labels == c)[0]
            same_cluster[np.ix_(idx, idx)] += 1.0

    consensus = np.divide(
        same_cluster,
        co_observed,
        out=np.zeros_like(same_cluster),
        where=co_observed > 0,
    )

    np.fill_diagonal(consensus, 1.0)

    # Keep samples that are not too frequently marked as noise.
    if noise_label is None:
        noise_frequency = np.zeros(n_samples, dtype=np.float32)
    else:
        noise_frequency = np.mean(label_runs == noise_label, axis=0)

    eligible = noise_frequency <= max_noise_frequency
    eligible_idx = np.where(eligible)[0]

    if len(eligible_idx) < max(2, min_final_cluster_size):
        return np.full(n_samples, -1, dtype=int)

    max_reasonable_k = max(
        1,
        len(eligible_idx) // max(1, min_final_cluster_size),
    )

    n_clusters_used = min(n_clusters_final, max_reasonable_k)

    # Cluster eligible samples using consensus distance.
    consensus_sub = consensus[np.ix_(eligible_idx, eligible_idx)]

    distance_sub = 1.0 - consensus_sub
    distance_sub = np.clip(distance_sub, 0.0, 1.0)
    np.fill_diagonal(distance_sub, 0.0)

    model = AgglomerativeClustering(
        n_clusters=n_clusters_used,
        metric="precomputed",
        linkage="average",
    )

    labels_sub = model.fit_predict(distance_sub)

    labels_raw = np.full(n_samples, -1, dtype=int)
    labels_raw[eligible_idx] = labels_sub

    # Compute each sample's mean within-cluster consensus strength.
    consensus_strength = np.zeros(n_samples, dtype=np.float32)

    for c in sorted(set(labels_raw) - {-1}):
        idx = np.where(labels_raw == c)[0]

        if len(idx) <= 1:
            consensus_strength[idx] = 0.0
            continue

        block = consensus[np.ix_(idx, idx)].copy()
        np.fill_diagonal(block, np.nan)

        consensus_strength[idx] = np.nanmean(block, axis=1)

    labels_filtered = labels_raw.copy()

    # Remove weak samples and undersized clusters.
    if allow_final_noise:
        weak = consensus_strength < min_consensus_strength
        labels_filtered[weak] = -1

        for c in sorted(set(labels_filtered) - {-1}):
            idx = np.where(labels_filtered == c)[0]

            if len(idx) < min_final_cluster_size:
                labels_filtered[idx] = -1

    # Relabel final clusters with consecutive integers.
    labels_final = np.full(n_samples, -1, dtype=int)

    final_clusters = sorted(set(labels_filtered) - {-1})

    for new_c, old_c in enumerate(final_clusters):
        labels_final[labels_filtered == old_c] = new_c

    return labels_final
