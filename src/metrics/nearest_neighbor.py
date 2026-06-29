import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _skewness(x: np.ndarray) -> float:
    """
    Calcola la skewness senza dipendere da scipy.
    """
    x = np.asarray(x, dtype=np.float64)

    mean = x.mean()
    std = x.std(ddof=0)

    if std == 0:
        return 0.0

    return float(np.mean(((x - mean) / std) ** 3))


def _compute_knn_indices(
    X: np.ndarray,
    k_max: int,
    #metric: str = "cosine",
    #normalize_l2: bool = True,
) -> np.ndarray:
    """
    Calcola gli indici dei k-nearest-neighbor per ogni sample.

    Ritorna:
        knn_indices: array di shape (n_samples, k_max)
    """
    X = np.asarray(X, dtype=np.float32)

    n_samples = X.shape[0]

    if k_max >= n_samples:
        raise ValueError(
            f"k_max={k_max} non può essere >= n_samples={n_samples}."
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
        # rimuove il sample stesso dai suoi vicini
        valid = indices[i] != i
        row = indices[i][valid]

        if len(row) < k_max:
            raise RuntimeError(
                f"Non ci sono abbastanza vicini validi per il sample {i}."
            )

        knn_indices[i] = row[:k_max]

    return knn_indices


def mutual_nearest_neighbor_ratio(knn_indices: np.ndarray) -> float:
    """
    Calcola il Mutual Nearest Neighbor ratio.

    knn_indices deve avere shape (n_samples, k).
    """
    n_samples, k = knn_indices.shape

    neighbor_sets = [set(knn_indices[i]) for i in range(n_samples)]

    mutual_count = 0

    for i in range(n_samples):
        for j in knn_indices[i]:
            if i in neighbor_sets[j]:
                mutual_count += 1

    total_edges = n_samples * k

    return float(mutual_count / total_edges)


def hubness_statistics(knn_indices: np.ndarray) -> dict:
    """
    Calcola statistiche di hubness.

    knn_indices deve avere shape (n_samples, k).
    """
    n_samples, k = knn_indices.shape

    hubness_counts = np.bincount(
        knn_indices.ravel(),
        minlength=n_samples,
    )

    return {
        "hubness skew": _skewness(hubness_counts),
        "max hubness": int(hubness_counts.max()),
        "mean hubness": float(hubness_counts.mean()),
        "std hubness": float(hubness_counts.std(ddof=0)),
        "p95 hubness": float(np.percentile(hubness_counts, 95)),
        "p99 hubness": float(np.percentile(hubness_counts, 99)),
    }


def evaluate_embedding_backbone(
    X: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
    hubness_k: int = 10,
    #metric: str = "cosine",
    #normalize_l2: bool = True,
) -> pd.DataFrame:
    """
    Calcola le metriche nearest-neighbor principali per una singola backbone.

    Parametri
    ---------
    X:
        Embedding della backbone, shape (n_samples, n_features).

    ks:
        Valori di k per MNN@k.

    hubness_k:
        Valore di k usato per calcolare hubness skew e max hubness.

    metric:
        Distanza usata per i nearest-neighbor. Default: "cosine".

    normalize_l2:
        Se True, applica normalizzazione L2 agli embedding prima del calcolo.
        Consigliato con cosine.

    Ritorna
    -------
    DataFrame con una riga e metriche numeriche.
    """
    k_max = max(max(ks), hubness_k)

    knn_indices = _compute_knn_indices(
        X=X,
        k_max=k_max,
        #metric=metric,
        #normalize_l2=normalize_l2,
    )

    results = {}

    for k in ks:
        knn_k = knn_indices[:, :k]
        results[f"MNN@{k}"] = mutual_nearest_neighbor_ratio(knn_k)

    hubness_k_indices = knn_indices[:, :hubness_k]
    hub_stats = hubness_statistics(hubness_k_indices)

    results["hubness skew"] = hub_stats["hubness skew"]
    results["max hubness"] = hub_stats["max hubness"]

    return pd.DataFrame([results])