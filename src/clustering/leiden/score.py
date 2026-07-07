import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy


def local_structure_per_cluster_consensus(
    consensus_labels,
    X_ref,
    k=10,
    metric="cosine",
    ignore_noise=True,
):
    """
    Valuta quanto i cluster finali del consensus rispettano
    la struttura locale dello spazio embedding di riferimento.

    Pensata per:
        Leiden + consensus clustering

    Parameters
    ----------
    consensus_labels : array-like, shape (n_samples,)
        Labels finali del consensus clustering.
        I label -1, se presenti, indicano punti instabili/ambigui.

    X_ref : array-like, shape (n_samples, n_features)
        Embedding di riferimento su cui valutare la coerenza locale.
        Nel tuo caso consigliato:
            pca_95_embedding_l2

    k : int
        Numero di vicini da considerare.

    metric : str
        Metrica per i kNN.
        Consigliato:
            "cosine" su pca_95_embedding_l2
            "euclidean" su UMAP embedding

    ignore_noise : bool
        Se True, ignora i punti con label -1.

    Returns
    -------
    pd.DataFrame
        Riga globale + una riga per cluster.
    """

    labels = np.asarray(consensus_labels)
    X_ref = np.asarray(X_ref)

    n_samples = len(labels)

    if X_ref.shape[0] != n_samples:
        raise ValueError(
            "X_ref e consensus_labels devono avere lo stesso numero di campioni."
        )

    valid_mask = labels != -1 if ignore_noise else np.ones(n_samples, dtype=bool)
    valid_clusters = sorted(set(labels[valid_mask]))

    if len(valid_clusters) == 0:
        return pd.DataFrame()

    kk = min(k, n_samples - 1)

    nn = NearestNeighbors(
        n_neighbors=kk + 1,
        metric=metric,
    )
    nn.fit(X_ref)

    distances, indices = nn.kneighbors(X_ref)

    # Per pesare gli archi se usiamo distanza euclidea
    all_distances = distances[:, 1:].ravel()
    positive_distances = all_distances[all_distances > 0]

    if len(positive_distances) > 0:
        sigma = np.median(positive_distances)
    else:
        sigma = 1.0

    sigma = max(float(sigma), 1e-12)

    per_point_rows = []

    for i in range(n_samples):
        if ignore_noise and labels[i] == -1:
            continue

        own_label = labels[i]

        neigh_idx = indices[i, 1:]
        neigh_dist = distances[i, 1:]

        neighbor_labels = []
        same_cluster_flags = []
        edge_cut_flags = []
        weights = []

        for j, dist in zip(neigh_idx, neigh_dist):
            if ignore_noise and labels[j] == -1:
                continue

            neigh_label = labels[j]

            same_cluster = own_label == neigh_label

            neighbor_labels.append(neigh_label)
            same_cluster_flags.append(same_cluster)
            edge_cut_flags.append(not same_cluster)

            if metric == "cosine":
                # sklearn cosine distance = 1 - cosine similarity
                weight = max(0.0, 1.0 - float(dist))
            else:
                # peso decrescente con la distanza
                weight = float(np.exp(-float(dist) / sigma))

            weights.append(weight)

        if len(neighbor_labels) == 0:
            continue

        same_cluster_flags = np.asarray(same_cluster_flags, dtype=float)
        edge_cut_flags = np.asarray(edge_cut_flags, dtype=float)
        weights = np.asarray(weights, dtype=float)

        local_agreement = same_cluster_flags.mean()
        edge_cut_ratio = edge_cut_flags.mean()

        if weights.sum() > 0:
            weighted_edge_cut_ratio = np.sum(weights * edge_cut_flags) / np.sum(weights)
        else:
            weighted_edge_cut_ratio = np.nan

        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        probs = counts / counts.sum()

        knn_entropy = entropy(probs)

        if len(valid_clusters) > 1:
            normalized_entropy = knn_entropy / np.log(len(valid_clusters))
        else:
            normalized_entropy = 0.0

        normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))

        per_point_rows.append({
            "sample_index": i,
            "cluster": own_label,
            "local_agreement": float(local_agreement),
            "edge_cut_ratio": float(edge_cut_ratio),
            "weighted_edge_cut_ratio": float(weighted_edge_cut_ratio),
            "knn_entropy": float(knn_entropy),
            "normalized_entropy": float(normalized_entropy),
        })

    per_point_df = pd.DataFrame(per_point_rows)

    if per_point_df.empty:
        return pd.DataFrame()

    rows = []

    # ------------------------------------------------------------
    # Riga globale
    # ------------------------------------------------------------
    rows.append({
        "cluster": "global",
        "n_samples": int(len(per_point_df)),
        "local_agreement_mean": float(per_point_df["local_agreement"].mean()),
        "local_agreement_median": float(per_point_df["local_agreement"].median()),
        "edge_cut_ratio_mean": float(per_point_df["edge_cut_ratio"].mean()),
        "weighted_edge_cut_ratio_mean": float(per_point_df["weighted_edge_cut_ratio"].mean()),
        "knn_entropy_mean": float(per_point_df["knn_entropy"].mean()),
        "normalized_entropy_mean": float(per_point_df["normalized_entropy"].mean()),
    })

    # ------------------------------------------------------------
    # Righe per cluster
    # ------------------------------------------------------------
    for cluster_id in valid_clusters:
        df_c = per_point_df[per_point_df["cluster"] == cluster_id]

        rows.append({
            "cluster": int(cluster_id),
            "n_samples": int(len(df_c)),
            "local_agreement_mean": float(df_c["local_agreement"].mean()),
            "local_agreement_median": float(df_c["local_agreement"].median()),
            "edge_cut_ratio_mean": float(df_c["edge_cut_ratio"].mean()),
            "weighted_edge_cut_ratio_mean": float(df_c["weighted_edge_cut_ratio"].mean()),
            "knn_entropy_mean": float(df_c["knn_entropy"].mean()),
            "normalized_entropy_mean": float(df_c["normalized_entropy"].mean()),
        })

    return pd.DataFrame(rows)


def consensus_param_metrics_leiden(
    local_consensus_df,
    consensus_labels,
    probabilities=None,
    min_cluster_size=None,
):
    labels = np.asarray(consensus_labels)
    n = len(labels)

    valid_mask = labels != -1
    valid_labels = sorted(set(labels[valid_mask]))

    n_clusters = len(valid_labels)
    unstable_ratio = np.mean(labels == -1)

    if n_clusters == 0:
        return {
            "n_clusters": 0,
            "unstable_ratio": float(unstable_ratio),
            "dominance": np.nan,
            "effective_n_clusters": np.nan,
            "balance_score": np.nan,
            "min_cluster_size": np.nan,
            "micro_cluster_ratio": np.nan,
            "mean_consensus_probability": np.nan,
            "local_agreement": np.nan,
            "weighted_edge_cut": np.nan,
            "normalized_knn_label_entropy": np.nan,
        }

    cluster_sizes = np.array([
        np.sum(labels == c)
        for c in valid_labels
    ])

    valid_n = cluster_sizes.sum()
    cluster_props = cluster_sizes / valid_n

    dominance = cluster_props.max()

    effective_n_clusters = 1.0 / np.sum(cluster_props ** 2)
    balance_score = effective_n_clusters / n_clusters

    if min_cluster_size is None:
        min_cluster_size = max(5, round(0.02 * n))

    micro_cluster_mask = cluster_sizes < min_cluster_size
    micro_cluster_ratio = cluster_sizes[micro_cluster_mask].sum() / valid_n

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        mean_consensus_probability = float(np.mean(probabilities[valid_mask]))
    else:
        mean_consensus_probability = np.nan

    global_row = local_consensus_df[
        local_consensus_df["cluster"] == "global"
    ].iloc[0]

    return {
        "n_clusters": int(n_clusters),
        "unstable_ratio": float(unstable_ratio),
        "dominance": float(dominance),
        "effective_n_clusters": float(effective_n_clusters),
        "balance_score": float(balance_score),
        "min_cluster_size": int(cluster_sizes.min()),
        "micro_cluster_ratio": float(micro_cluster_ratio),
        "mean_consensus_probability": mean_consensus_probability,
        "local_agreement": float(global_row["local_agreement_mean"]),
        "weighted_edge_cut": float(global_row["weighted_edge_cut_ratio_mean"]),
        "normalized_knn_label_entropy": float(global_row["normalized_entropy_mean"]),
    }


def consensus_param_score_leiden(
    metrics,
    min_clusters=4,
    max_clusters=8,
    max_unstable=0.30,
    max_dominance=0.70,
    penalty_unstable=1.0,
    penalty_dominance=0.75,
    penalty_micro_cluster=0.50,
):
    n_clusters = metrics["n_clusters"]

    if n_clusters < min_clusters or n_clusters > max_clusters:
        return -np.inf

    local_agreement = metrics["local_agreement"]
    weighted_edge_cut = metrics["weighted_edge_cut"]
    normalized_entropy = metrics["normalized_knn_label_entropy"]
    mean_consensus_probability = metrics["mean_consensus_probability"]
    balance_score = metrics["balance_score"]

    required_values = [
        local_agreement,
        weighted_edge_cut,
        normalized_entropy,
        balance_score,
    ]

    if not all(np.isfinite(v) for v in required_values):
        return -np.inf

    if not np.isfinite(mean_consensus_probability):
        mean_consensus_probability = 0.0

    base_score = (
        0.30 * local_agreement
        + 0.30 * (1.0 - weighted_edge_cut)
        + 0.15 * (1.0 - normalized_entropy)
        + 0.15 * mean_consensus_probability
        + 0.10 * balance_score
    )

    unstable_excess = max(0.0, metrics["unstable_ratio"] - max_unstable)
    dominance_excess = max(0.0, metrics["dominance"] - max_dominance)
    micro_cluster_ratio = metrics["micro_cluster_ratio"]

    score = (
        base_score
        - penalty_unstable * unstable_excess
        - penalty_dominance * dominance_excess
        - penalty_micro_cluster * micro_cluster_ratio
    )

    return float(score)