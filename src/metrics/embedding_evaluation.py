from typing import List, Any

import numpy as np


def find_exact_duplicate(
        embeddings: np.ndarray
) -> List[tuple[int, int]] | None:
    _, first_indices, inverse, counts = np.unique(
        embeddings,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    duplicate_pairs: list[tuple[int, int]] = []

    duplicated_group_ids = np.flatnonzero(counts > 1)

    for group_id in duplicated_group_ids:
        original_index = int(first_indices[group_id])
        group_indices = np.flatnonzero(inverse == group_id)

        for index in group_indices:
            index = int(index)

            if index != original_index:
                duplicate_pairs.append((original_index, index))

    duplicate_pairs.sort(key=lambda pair: (pair[0], pair[1]))

    if len(duplicate_pairs) == 0:
        return None

    return duplicate_pairs


def evaluate_embedding_norms(
        embeddings: np.ndarray,
        near_zero_eps: float = 1e-8,
        iqr_multiplier: float = 3.0,
) -> dict[str, Any]:
    embeddings = np.asarray(embeddings)

    finite_rows_mask = np.all(np.isfinite(embeddings), axis=1)

    norms = np.full(embeddings.shape[0], np.nan, dtype=np.float64)
    norms[finite_rows_mask] = np.linalg.norm(
        embeddings[finite_rows_mask],
        axis=1,
    )

    finite_norms = norms[np.isfinite(norms)]

    if finite_norms.size == 0:
        return {
            "summary": None,
            "near_zero_indices": [],
            "low_outlier_indices": [],
            "high_outlier_indices": [],
            "norms": norms,
        }

    q1 = float(np.percentile(finite_norms, 25))
    q3 = float(np.percentile(finite_norms, 75))
    iqr = q3 - q1

    low_threshold = max(0.0, q1 - iqr_multiplier * iqr)
    high_threshold = q3 + iqr_multiplier * iqr

    near_zero_mask = norms <= near_zero_eps
    low_outlier_mask = norms < low_threshold
    high_outlier_mask = norms > high_threshold

    return {
        "near_zero_indices": np.flatnonzero(near_zero_mask).astype(int).tolist(),
        "low_outlier_indices": np.flatnonzero(low_outlier_mask).astype(int).tolist(),
        "high_outlier_indices": np.flatnonzero(high_outlier_mask).astype(int).tolist(),
        "norms": norms,
    }
