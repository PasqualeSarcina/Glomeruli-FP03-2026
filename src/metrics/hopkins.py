import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_hopkins_dataframe(
    X: np.ndarray,
    n_runs: int = 100,
    n_samples: int | None = None,
    sample_fraction: float = 0.1,
    random_state: int = 42,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Hopkins statistic over n_runs resamplings of PCA-reduced embeddings.
    ~0.5 = random, >0.7 = clustering tendency, >0.8 = strong clustering tendency.
    """

    X = np.asarray(X, dtype=np.float64)

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")

    n, d = X.shape

    if n_samples is None:
        n_samples_eff = int(np.ceil(sample_fraction * n))
    else:
        n_samples_eff = int(n_samples)

    n_samples_eff = max(1, min(n_samples_eff, n - 1))

    rng_master = np.random.default_rng(random_state)
    seeds = rng_master.integers(0, 1_000_000, size=n_runs)

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    nn = NearestNeighbors(
        n_neighbors=2,
        metric=metric,
    )
    nn.fit(X)

    hopkins_values = []

    for seed in seeds:
        rng = np.random.default_rng(int(seed))

        real_indices = rng.choice(
            n,
            size=n_samples_eff,
            replace=False,
        )
        real_points = X[real_indices]

        # uniform points inside the bounding box of the PCA space
        random_points = rng.uniform(
            low=mins,
            high=maxs,
            size=(n_samples_eff, d),
        )

        # the first neighbour of a real point is itself, so take the second
        real_distances, _ = nn.kneighbors(real_points, n_neighbors=2)
        w = real_distances[:, 1]

        random_distances, _ = nn.kneighbors(random_points, n_neighbors=1)
        u = random_distances[:, 0]

        denominator = np.sum(u) + np.sum(w)

        if denominator == 0:
            hopkins = np.nan
        else:
            hopkins = np.sum(u) / denominator

        hopkins_values.append(hopkins)

    hopkins_values = np.asarray(hopkins_values, dtype=np.float64)

    result = pd.DataFrame(
        {
            "n_points": [n],
            "n_features": [d],
            "n_runs": [n_runs],
            "n_samples_per_run": [n_samples_eff],
            "sample_fraction": [sample_fraction],
            "metric": [metric],
            "hopkins_mean": [np.nanmean(hopkins_values)],
            "hopkins_std": [np.nanstd(hopkins_values, ddof=1)],
            "hopkins_min": [np.nanmin(hopkins_values)],
            "hopkins_max": [np.nanmax(hopkins_values)],
            "hopkins_median": [np.nanmedian(hopkins_values)],
        }
    )

    return result