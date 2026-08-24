"""
Continuous-gradation metrics: how well an embedding lays glomeruli along a
smooth severity axis, rather than splitting them into discrete clusters.
All three metrics are deterministic and use the morphological severity proxy
from morphology_descriptors.py as reference.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def distance_morphology_correlation(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    max_pairs: int = 200000,
    random_state: int = 42,
) -> float:
    """
    Spearman correlation between embedding distances and absolute differences
    in morphology score, over a sample of pairs. Towards 1 = the embedding
    reflects the continuous severity gradient.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    morphology_score = np.asarray(morphology_score, dtype=np.float64)
    n = embeddings.shape[0]

    rng = np.random.default_rng(random_state)

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        emb_dist = pdist(embeddings, metric="euclidean")
        morph_diff = pdist(morphology_score.reshape(-1, 1), metric="cityblock")
    else:
        # sample random pairs instead of all of them
        i_idx = rng.integers(0, n, size=max_pairs)
        j_idx = rng.integers(0, n, size=max_pairs)
        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]
        emb_dist = np.linalg.norm(embeddings[i_idx] - embeddings[j_idx], axis=1)
        morph_diff = np.abs(morphology_score[i_idx] - morphology_score[j_idx])

    rho, _ = spearmanr(emb_dist, morph_diff)
    return float(rho)


def morphology_neighborhood_consistency(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    k: int = 15,
) -> float:
    """
    1 - (local / global) mean absolute deviation of the morphology score over
    the k nearest neighbours. ~1 = neighbours share a very similar morphology,
    ~0 = no better than random pairs, <0 = neighbours are more different than average.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    morphology_score = np.asarray(morphology_score, dtype=np.float64)
    n = embeddings.shape[0]

    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]  # drop the point itself

    global_mad = float(np.mean(np.abs(morphology_score - np.mean(morphology_score))))
    if global_mad == 0:
        return float("nan")

    local_mads = []
    for i in range(n):
        neighbor_scores = morphology_score[indices[i]]
        local_mads.append(np.mean(np.abs(neighbor_scores - morphology_score[i])))

    mean_local_mad = float(np.mean(local_mads))
    return float(1.0 - mean_local_mad / global_mad)


def morphology_gradient_smoothness(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    k: int = 15,
) -> float:
    """
    Moran's I of the morphology score over the k-nearest-neighbour graph.
    ~1 = smooth gradient, ~0 = no spatial autocorrelation, <0 = anti-correlation.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    z = np.asarray(morphology_score, dtype=np.float64)
    n = embeddings.shape[0]

    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]

    z_centered = z - np.mean(z)
    denom = np.sum(z_centered ** 2)
    if denom == 0:
        return float("nan")

    # binary weights: 1 for each of the k neighbours
    numerator = 0.0
    w_total = 0.0
    for i in range(n):
        for j in indices[i]:
            numerator += z_centered[i] * z_centered[j]
            w_total += 1.0

    morans_i = (n / w_total) * (numerator / denom)
    return float(morans_i)


def evaluate_gradation(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    k: int = 15,
    random_state: int = 42,
) -> dict:
    """All gradation metrics for one backbone, as a dict ready for a table row."""

    return {
        "grad_distance_morph_corr": distance_morphology_correlation(
            embeddings, morphology_score, random_state=random_state
        ),
        "grad_neighborhood_consistency": morphology_neighborhood_consistency(
            embeddings, morphology_score, k=k
        ),
        "grad_gradient_smoothness_moran": morphology_gradient_smoothness(
            embeddings, morphology_score, k=k
        ),
    }
