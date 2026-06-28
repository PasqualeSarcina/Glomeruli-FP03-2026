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
    Calcola la statistica di Hopkins su embedding già PCA.

    Convenzione:
    - Hopkins ~ 0.5  -> distribuzione circa casuale
    - Hopkins > 0.7  -> possibile tendenza al clustering
    - Hopkins > 0.8  -> forte tendenza al clustering

    Parameters
    ----------
    X:
        Array di shape (n_samples, n_features), già PCA.
    n_runs:
        Numero di ripetizioni con seed diversi.
    n_samples:
        Numero di punti da campionare a ogni run.
        Se None, viene calcolato da sample_fraction.
    sample_fraction:
        Frazione dei punti da campionare se n_samples è None.
    random_state:
        Seed iniziale per riproducibilità.
    metric:
        Metrica usata da NearestNeighbors.
        Per PCA usare normalmente "euclidean".

    Returns
    -------
    pd.DataFrame
        DataFrame con statistiche riassuntive di Hopkins.
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X deve essere un array 2D di shape (n_samples, n_features).")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contiene NaN o infiniti.")

    n, d = X.shape

    if n < 3:
        raise ValueError("Servono almeno 3 campioni per calcolare Hopkins.")

    if n_runs < 1:
        raise ValueError("n_runs deve essere almeno 1.")

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

        # Campiona punti reali
        real_indices = rng.choice(
            n,
            size=n_samples_eff,
            replace=False,
        )
        real_points = X[real_indices]

        # Genera punti casuali uniformi nel bounding box dello spazio PCA
        random_points = rng.uniform(
            low=mins,
            high=maxs,
            size=(n_samples_eff, d),
        )

        # Distanza dei punti reali dal loro nearest neighbor reale più vicino
        # Il primo vicino è il punto stesso, quindi si prende il secondo
        real_distances, _ = nn.kneighbors(real_points, n_neighbors=2)
        w = real_distances[:, 1]

        # Distanza dei punti casuali dal nearest neighbor reale più vicino
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