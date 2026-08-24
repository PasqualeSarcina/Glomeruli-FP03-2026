import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _compute_knn_indices(
    X: np.ndarray,
    k_max: int,
) -> np.ndarray:
    """Indices of the k_max nearest neighbours of every sample, shape (n_samples, k_max)."""
    X = np.asarray(X, dtype=np.float32)

    n_samples = X.shape[0]

    if k_max >= n_samples:
        raise ValueError(
            f"k_max={k_max} must be smaller than n_samples={n_samples}."
        )

    X = normalize(X, norm="l2", axis=1)

    nn = NearestNeighbors(
        n_neighbors=k_max + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nn.fit(X)

    distances, indices = nn.kneighbors(X, return_distance=True)

    knn_indices = np.empty((n_samples, k_max), dtype=np.int64)

    for i in range(n_samples):
        # drop the sample itself from its own neighbours
        valid = indices[i] != i
        row = indices[i][valid]

        if len(row) < k_max:
            raise RuntimeError(
                f"Not enough valid neighbours for sample {i}."
            )

        knn_indices[i] = row[:k_max]

    return knn_indices


def mutual_nearest_neighbor_ratio(knn_indices: np.ndarray) -> float:
    """Mutual nearest neighbour ratio; knn_indices has shape (n_samples, k)."""
    n_samples, k = knn_indices.shape

    neighbor_sets = [set(knn_indices[i]) for i in range(n_samples)]

    mutual_count = 0

    for i in range(n_samples):
        for j in knn_indices[i]:
            if i in neighbor_sets[j]:
                mutual_count += 1

    total_edges = n_samples * k

    return float(mutual_count / total_edges)


def hubness_statistics(knn_indices: np.ndarray):
    """Skewness of the k-occurrence distribution and its maximum (hubness)."""
    n_samples, k = knn_indices.shape

    hubness_counts = np.bincount(
        knn_indices.ravel(),
        minlength=n_samples,
    )

    hubness_counts_float = hubness_counts.astype(np.float64)
    mean = hubness_counts_float.mean()
    std = hubness_counts_float.std(ddof=0)
    hubness_skew = 0.0 if std == 0 else float(
        np.mean(((hubness_counts_float - mean) / std) ** 3)
    )

    return hubness_skew, int(hubness_counts.max())


def evaluate_embedding_backbone(
    X: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
    hubness_k: int = 10,
    #metric: str = "cosine",
    #normalize_l2: bool = True,
) -> pd.DataFrame:
    """
    Nearest-neighbour metrics for one backbone: MNN@k for each k in ks, plus
    hubness skew and max hubness at hubness_k. Returns a single-row DataFrame.
    """
    k_max = max(max(ks), hubness_k)

    knn_indices = _compute_knn_indices(X, k_max)

    results = {}

    for k in ks:
        knn_k = knn_indices[:, :k]
        results[f"MNN@{k}"] = mutual_nearest_neighbor_ratio(knn_k)

    hubness_k_indices = knn_indices[:, :hubness_k]

    results["hubness skew"], results["max hubness"] = hubness_statistics(hubness_k_indices)

    return pd.DataFrame([results])
