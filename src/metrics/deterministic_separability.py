"""
Deterministic separability probes for comparing backbones.

UMAP and HDBSCAN-on-UMAP are stochastic, so part of the variance observed
between backbones would come from the seed rather than from embedding quality.
Ward linkage and PCA are seed-free, so the comparison here is reproducible.
These are probes, not the project's final clustering pipeline.
"""


import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def effective_dimensionality(
    X: np.ndarray,
    variance_targets: tuple[float, ...] = (0.90, 0.95, 0.99),
    random_state: int = 42,
) -> dict:
    """
    Effective dimensionality via PCA: components needed for each variance
    threshold, in absolute terms and as a fraction of the original dimension.
    Also returns the participation ratio PR = (sum L)^2 / sum(L^2), which is 1
    when all variance sits on one component and n_features when it is uniform.
    """

    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    max_components = min(n_samples, n_features)

    pca = PCA(n_components=max_components, svd_solver="full", random_state=random_state)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    result = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
    }

    for target in variance_targets:
        n_components_needed = int(np.searchsorted(cumulative, target) + 1)
        n_components_needed = min(n_components_needed, max_components)

        key = f"{int(target * 100)}"
        result[f"n_components_var{key}"] = n_components_needed
        result[f"frac_dims_var{key}"] = float(n_components_needed / n_features)

    eigenvalues = pca.explained_variance_
    participation_ratio = float(
        (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    )

    result["participation_ratio"] = participation_ratio
    result["participation_ratio_frac"] = float(participation_ratio / n_features)

    return result


def ward_clustering_curve(
    X: np.ndarray,
    k_values: tuple[int, ...] = tuple(range(2, 16)),
    max_samples_for_metrics: int | None = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Internal clustering metrics with Ward linkage for each k in k_values.
    Clustering runs on all points; the O(n^2) metrics may be evaluated on a
    reproducible subsample. Returns one row per k.
    """

    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    rng = np.random.default_rng(random_state)

    if max_samples_for_metrics is not None and n_samples > max_samples_for_metrics:
        metric_idx = np.sort(
            rng.choice(n_samples, size=max_samples_for_metrics, replace=False)
        )
    else:
        metric_idx = np.arange(n_samples)

    rows = []

    for k in k_values:
        if k < 2 or k >= n_samples:
            continue

        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)

        counts = np.bincount(labels, minlength=k)

        labels_eval = labels[metric_idx]
        X_eval = X[metric_idx]

        row = {
            "k": int(k),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
        }

        if len(np.unique(labels_eval)) < 2:
            row.update(
                {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
            )
        else:
            try:
                row["silhouette"] = float(
                    silhouette_score(X_eval, labels_eval, metric="euclidean")
                )
            except Exception:
                row["silhouette"] = np.nan

            try:
                row["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels_eval))
            except Exception:
                row["davies_bouldin"] = np.nan

            try:
                row["calinski_harabasz"] = float(
                    calinski_harabasz_score(X_eval, labels_eval)
                )
            except Exception:
                row["calinski_harabasz"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_ward_curve(curve_df: pd.DataFrame) -> dict:
    """
    Collapse the Ward curve into scalars: the peak of each metric and the k at
    which it occurs, plus the mean silhouette so that a single lucky k does not
    dominate the comparison. Davies-Bouldin is lower-is-better.
    """

    if curve_df.empty:
        return {
            "best_silhouette": np.nan,
            "best_k_silhouette": None,
            "mean_silhouette": np.nan,
            "best_calinski_harabasz": np.nan,
            "best_k_calinski_harabasz": None,
            "best_davies_bouldin": np.nan,
            "best_k_davies_bouldin": None,
        }

    result = {}

    if curve_df["silhouette"].notna().any():
        idx_best_sil = curve_df["silhouette"].idxmax()
        result["best_silhouette"] = float(curve_df.loc[idx_best_sil, "silhouette"])
        result["best_k_silhouette"] = int(curve_df.loc[idx_best_sil, "k"])
        result["mean_silhouette"] = float(curve_df["silhouette"].mean(skipna=True))
    else:
        result["best_silhouette"] = np.nan
        result["best_k_silhouette"] = None
        result["mean_silhouette"] = np.nan

    if curve_df["calinski_harabasz"].notna().any():
        idx_best_ch = curve_df["calinski_harabasz"].idxmax()
        result["best_calinski_harabasz"] = float(
            curve_df.loc[idx_best_ch, "calinski_harabasz"]
        )
        result["best_k_calinski_harabasz"] = int(curve_df.loc[idx_best_ch, "k"])
    else:
        result["best_calinski_harabasz"] = np.nan
        result["best_k_calinski_harabasz"] = None

    if curve_df["davies_bouldin"].notna().any():
        idx_best_db = curve_df["davies_bouldin"].idxmin()
        result["best_davies_bouldin"] = float(curve_df.loc[idx_best_db, "davies_bouldin"])
        result["best_k_davies_bouldin"] = int(curve_df.loc[idx_best_db, "k"])
    else:
        result["best_davies_bouldin"] = np.nan
        result["best_k_davies_bouldin"] = None

    return result


def pca_reduce(
    X: np.ndarray,
    variance_target: float = 0.95,
    max_components: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    PCA reduction keeping variance_target of the variance.

    Separability is measured in the same kind of reduced space the real
    pipeline uses: across thousands of dimensions euclidean distances flatten
    out and the silhouette comes out artificially low. Returns (X_reduced, info).
    """

    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    upper = min(n_samples, n_features)
    if max_components is not None:
        upper = min(upper, max_components)

    pca = PCA(n_components=variance_target, svd_solver="full", random_state=random_state)
    X_reduced = pca.fit_transform(X)

    n_used = int(pca.n_components_)
    if max_components is not None and n_used > max_components:
        # recompute, truncated to the largest requested number of components
        pca = PCA(n_components=max_components, svd_solver="full", random_state=random_state)
        X_reduced = pca.fit_transform(X)
        n_used = int(pca.n_components_)

    info = {
        "pca_n_components": n_used,
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "pca_variance_target": float(variance_target),
    }

    return X_reduced.astype(np.float64), info


def hdbscan_clustering_metrics(
    X: np.ndarray,
    min_cluster_sizes: tuple[int, ...] = (10, 20, 40),
    min_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Density-based probe with HDBSCAN over a range of min_cluster_size.

    Unlike Ward it finds the number of clusters on its own, allows noise and
    handles non-spherical shapes. sklearn's HDBSCAN is seed-free and runs here
    directly on the PCA space, so the probe stays reproducible.

    One row per min_cluster_size. The silhouette is computed on clustered points
    only (NaN below 2 clusters), and the fraction of points in the largest
    cluster catches a collapse into a single group.
    """

    from sklearn.cluster import HDBSCAN

    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    rows = []
    for mcs in min_cluster_sizes:
        if mcs >= n_samples:
            continue

        model = HDBSCAN(
            min_cluster_size=int(mcs),
            min_samples=min_samples,
            copy=True,
        )
        labels = model.fit_predict(X)

        noise_mask = labels == -1
        n_noise = int(np.sum(noise_mask))
        cluster_labels = labels[~noise_mask]
        unique_clusters = np.unique(cluster_labels)
        n_clusters = int(len(unique_clusters))

        row = {
            "min_cluster_size": int(mcs),
            "n_clusters": n_clusters,
            "noise_fraction": float(n_noise / n_samples),
        }

        if n_clusters >= 1:
            counts = np.array([np.sum(cluster_labels == c) for c in unique_clusters])
            row["largest_cluster_fraction"] = float(counts.max() / n_samples)
        else:
            row["largest_cluster_fraction"] = np.nan

        # silhouette on clustered points only, and only with >= 2 clusters
        if n_clusters >= 2:
            try:
                row["silhouette_no_noise"] = float(
                    silhouette_score(X[~noise_mask], cluster_labels, metric="euclidean")
                )
            except Exception:
                row["silhouette_no_noise"] = np.nan
            # DBCV on the full set; noise is excluded inside the function
            try:
                row["dbcv"] = density_based_clustering_validation(X, labels, noise_label=-1)
            except Exception:
                row["dbcv"] = np.nan
        else:
            row["silhouette_no_noise"] = np.nan
            row["dbcv"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_hdbscan_metrics(hdbscan_df: pd.DataFrame) -> dict:
    """
    Collapse the HDBSCAN table into scalars. The best configuration is the one
    with the highest silhouette among those finding at least 2 clusters without
    collapsing into noise; the minimum observed noise fraction is also reported.
    """

    if hdbscan_df.empty:
        return {
            "hdbscan_best_silhouette": np.nan,
            "hdbscan_best_min_cluster_size": None,
            "hdbscan_best_n_clusters": None,
            "hdbscan_best_noise_fraction": np.nan,
            "hdbscan_min_noise_fraction": np.nan,
        }

    valid = hdbscan_df[hdbscan_df["silhouette_no_noise"].notna()]

    result = {
        "hdbscan_min_noise_fraction": float(hdbscan_df["noise_fraction"].min()),
    }

    if not valid.empty:
        idx = valid["silhouette_no_noise"].idxmax()
        result["hdbscan_best_silhouette"] = float(valid.loc[idx, "silhouette_no_noise"])
        result["hdbscan_best_min_cluster_size"] = int(valid.loc[idx, "min_cluster_size"])
        result["hdbscan_best_n_clusters"] = int(valid.loc[idx, "n_clusters"])
        result["hdbscan_best_noise_fraction"] = float(valid.loc[idx, "noise_fraction"])
    else:
        result["hdbscan_best_silhouette"] = np.nan
        result["hdbscan_best_min_cluster_size"] = None
        result["hdbscan_best_n_clusters"] = None
        result["hdbscan_best_noise_fraction"] = np.nan

    # DBCV reported separately: it is the most appropriate index for HDBSCAN
    if "dbcv" in hdbscan_df.columns and hdbscan_df["dbcv"].notna().any():
        idx_dbcv = hdbscan_df["dbcv"].idxmax()
        result["hdbscan_best_dbcv"] = float(hdbscan_df.loc[idx_dbcv, "dbcv"])
        result["hdbscan_best_dbcv_n_clusters"] = int(hdbscan_df.loc[idx_dbcv, "n_clusters"])
        result["hdbscan_best_dbcv_min_cluster_size"] = int(
            hdbscan_df.loc[idx_dbcv, "min_cluster_size"]
        )
    else:
        result["hdbscan_best_dbcv"] = np.nan
        result["hdbscan_best_dbcv_n_clusters"] = None
        result["hdbscan_best_dbcv_min_cluster_size"] = None

    return result


def density_based_clustering_validation(
    X: np.ndarray,
    labels: np.ndarray,
    noise_label: int = -1,
) -> float:
    """
    DBCV, Density-Based Clustering Validation (Moulavi et al., SDM 2014).

    Validation index built for density-based clustering: unlike the silhouette
    it handles arbitrary cluster shapes and accounts for internal density.
    Ranges from -1 to +1. Points labelled noise_label are excluded.

    Direct implementation of the paper (mutual reachability -> per-cluster MST
    -> density sparseness/separation), O(sum_i n_i^2). Returns the weighted mean
    over clusters, or NaN below 2 valid clusters.
    """

    from scipy.spatial.distance import cdist
    from scipy.sparse.csgraph import minimum_spanning_tree

    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    core_mask = labels != noise_label
    X_core = X[core_mask]
    labels_core = labels[core_mask]

    unique_labels = np.unique(labels_core)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return float("nan")

    n_total = X_core.shape[0]
    n_features = X_core.shape[1]

    # all-points-core-distance: apts_i = (mean_j 1/d(i,j)^dim)^(-1/dim), j != i
    all_core_dist = {}
    intra_index = {}
    for lab in unique_labels:
        idx = np.where(labels_core == lab)[0]
        intra_index[lab] = idx
        pts = X_core[idx]
        n_i = len(idx)
        if n_i <= 1:
            continue
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)

        # two numerical guards: coincident points (D=0) would give inf and are
        # dropped, and an all-zero sum of inverses would divide by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = D ** (-n_features)
        has_exact_duplicate = np.any(~np.isfinite(inv), axis=1)
        inv[~np.isfinite(inv)] = 0.0

        inv_sum = inv.sum(axis=1)
        mean_inv = inv_sum / (n_i - 1)

        core = np.zeros(n_i, dtype=np.float64)
        # exact duplicates mean infinite local density -> core = 0
        valid = (mean_inv > 0) & (~has_exact_duplicate)
        with np.errstate(divide="ignore", invalid="ignore"):
            core[valid] = mean_inv[valid] ** (-1.0 / n_features)
        # isolated points fall back to the largest finite distance in the cluster
        isolated = (mean_inv == 0) & (~has_exact_duplicate)
        if np.any(isolated):
            finite_D = D.copy()
            finite_D[~np.isfinite(finite_D)] = 0.0
            core[isolated] = finite_D.max()

        for local_j, global_i in enumerate(idx):
            all_core_dist[global_i] = float(core[local_j])

    def mutual_reachability(i, j, dij):
        return max(all_core_dist.get(i, 0.0), all_core_dist.get(j, 0.0), dij)

    # internal density sparseness (DSC) via the MST of the mutual-reach graph
    dsc = {}
    for lab in unique_labels:
        idx = intra_index[lab]
        n_i = len(idx)
        if n_i <= 1:
            dsc[lab] = 0.0
            continue
        pts = X_core[idx]
        D = cdist(pts, pts)
        M = np.zeros_like(D)
        for a in range(n_i):
            for b in range(a + 1, n_i):
                mr = mutual_reachability(idx[a], idx[b], D[a, b])
                M[a, b] = mr
                M[b, a] = mr
        mst = minimum_spanning_tree(M).toarray()
        # density sparseness = widest edge of the MST
        dsc[lab] = float(mst.max()) if mst.size else 0.0

    # density separation between clusters (DSPC)
    def cluster_separation(lab_a, lab_b):
        ia, ib = intra_index[lab_a], intra_index[lab_b]
        D = cdist(X_core[ia], X_core[ib])
        best = np.inf
        for a in range(len(ia)):
            for b in range(len(ib)):
                mr = mutual_reachability(ia[a], ib[b], D[a, b])
                if mr < best:
                    best = mr
        return best

    # per-cluster validity index and weighted mean
    total_score = 0.0
    for lab in unique_labels:
        min_sep = np.inf
        for other in unique_labels:
            if other == lab:
                continue
            sep = cluster_separation(lab, other)
            if sep < min_sep:
                min_sep = sep
        denom = max(min_sep, dsc[lab])
        v = 0.0 if denom == 0 else (min_sep - dsc[lab]) / denom
        weight = len(intra_index[lab]) / n_total
        total_score += weight * v

    return float(total_score)