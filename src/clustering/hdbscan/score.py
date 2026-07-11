import numpy as np


def consensus_param_score(
    dbcv_df,
    consensus_labels,
    max_noise,
    penalty_std=0.25,
    penalty_bad_cluster=0.50,
):
    labels = np.asarray(consensus_labels)

    n_clusters = len(set(labels) - {-1})
    noise_ratio = np.mean(labels == -1)

    if n_clusters < 2:
        return -np.inf

    # vincolo duro sul rumore
    if noise_ratio > max_noise:
        return -np.inf

    non_noise_df = dbcv_df[dbcv_df["cluster"] != -1]

    if non_noise_df.empty:
        return -np.inf

    global_dbcv_mean = non_noise_df["global_dbcv_mean"].iloc[0]
    global_dbcv_std = non_noise_df["global_dbcv_std"].iloc[0]
    min_cluster_dbcv_median = non_noise_df["dbcv_median"].min()

    if not np.isfinite(global_dbcv_mean) or not np.isfinite(min_cluster_dbcv_median):
        return -np.inf

    if not np.isfinite(global_dbcv_std):
        global_dbcv_std = 0.0

    bad_cluster_penalty = max(0.0, -min_cluster_dbcv_median)

    score = (
        global_dbcv_mean
        - penalty_std * global_dbcv_std
        - penalty_bad_cluster * bad_cluster_penalty
    )

    return float(score)

def consensus_param_metrics(dbcv_df, consensus_labels):
    labels = np.asarray(consensus_labels)

    n_clusters = len(set(labels) - {-1})
    noise_ratio = np.mean(labels == -1)

    non_noise_df = dbcv_df[dbcv_df["cluster"] != -1]

    if non_noise_df.empty:
        return {
            "n_clusters": n_clusters,
            "noise_ratio": float(noise_ratio),
            "global_dbcv_mean": np.nan,
            "global_dbcv_std": np.nan,
            "min_cluster_dbcv_median": np.nan,
            "mean_cluster_dbcv_median": np.nan,
        }

    return {
        "n_clusters": n_clusters,
        "noise_ratio": float(noise_ratio),
        "global_dbcv_mean": float(non_noise_df["global_dbcv_mean"].iloc[0]),
        "global_dbcv_std": float(non_noise_df["global_dbcv_std"].iloc[0]),
        "min_cluster_dbcv_median": float(non_noise_df["dbcv_median"].min()),
        "mean_cluster_dbcv_median": float(non_noise_df["dbcv_median"].mean()),
    }

