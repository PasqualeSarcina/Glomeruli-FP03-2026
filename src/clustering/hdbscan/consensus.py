import numpy as np
from sklearn.cluster import AgglomerativeClustering


def consensus_clustering_from_labels_hdbscan(
    label_runs,
    n_clusters="median",
    ignore_noise=True,
    max_noise_frequency=0.50,
    min_consensus_strength=0.50,
):
    """
    Consensus clustering a partire da una lista di labels.

    Parameters
    ----------
    label_runs : list[np.ndarray] or np.ndarray
        Lista di labels prodotte da più run.
        Shape: (n_runs, n_samples)

    n_clusters : int or "median"
        Numero di cluster finali.
        Se "median", usa la mediana del numero di cluster non-noise
        ottenuti nelle run.

    ignore_noise : bool
        Se True, le coppie in cui almeno un punto è -1 non vengono
        considerate nel calcolo della co-associazione.

    max_noise_frequency : float
        Se un punto è noise in almeno questa frazione di run,
        viene marcato come -1 nel consenso finale.

    min_consensus_strength : float
        Se un punto ha bassa forza media con il proprio cluster finale,
        viene marcato come -1.

    Returns
    -------
    labels_final : np.ndarray
        Labels finali del consensus clustering.

    probabilities : np.ndarray
        Stabilità/probabilità di consenso per ogni punto.
        Valori in [0, 1]. Per i punti finali -1 vale 0.
    """

    label_runs = np.asarray(label_runs)

    if label_runs.ndim != 2:
        raise ValueError(
            "label_runs deve avere shape (n_runs, n_samples). "
            "Esempio: np.array([labels_run1, labels_run2, ...])"
        )

    n_runs, n_samples = label_runs.shape

    if n_runs < 2:
        raise ValueError("Servono almeno 2 run per fare consensus clustering.")

    # ------------------------------------------------------------
    # 1. Numero di cluster finale
    # ------------------------------------------------------------
    if n_clusters == "median":
        ks = []

        for labels in label_runs:
            k = len(set(labels)) - (1 if -1 in labels else 0)
            if k > 0:
                ks.append(k)

        if len(ks) == 0:
            raise ValueError("Nessuna run contiene cluster validi non-noise.")

        n_clusters_final = int(round(np.median(ks)))
        n_clusters_final = max(n_clusters_final, 1)

    elif isinstance(n_clusters, int):
        n_clusters_final = n_clusters

    else:
        raise ValueError("n_clusters deve essere un intero oppure 'median'.")

    if n_clusters_final >= n_samples:
        raise ValueError("n_clusters deve essere minore del numero di campioni.")

    # ------------------------------------------------------------
    # 2. Matrice di consenso
    # ------------------------------------------------------------
    same_cluster = np.zeros((n_samples, n_samples), dtype=np.float32)
    co_observed = np.zeros((n_samples, n_samples), dtype=np.float32)

    for labels in label_runs:
        labels = np.asarray(labels)

        if ignore_noise:
            valid_mask = labels != -1
        else:
            valid_mask = np.ones(n_samples, dtype=bool)

        valid_idx = np.where(valid_mask)[0]

        # Denominatore: coppie osservate
        co_observed[np.ix_(valid_idx, valid_idx)] += 1

        # Numeratore: coppie nello stesso cluster
        for c in np.unique(labels[valid_mask]):
            if c == -1:
                continue

            idx = np.where(labels == c)[0]
            same_cluster[np.ix_(idx, idx)] += 1

    consensus = np.divide(
        same_cluster,
        co_observed,
        out=np.zeros_like(same_cluster),
        where=co_observed > 0,
    )

    np.fill_diagonal(consensus, 1.0)

    # Frequenza con cui ogni punto è stato noise
    noise_frequency = np.mean(label_runs == -1, axis=0)

    # ------------------------------------------------------------
    # 3. Clustering finale sulla distanza 1 - consensus
    # ------------------------------------------------------------
    distance = 1.0 - consensus
    np.fill_diagonal(distance, 0.0)

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters_final,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters_final,
            affinity="precomputed",
            linkage="average",
        )

    labels_raw = model.fit_predict(distance)

    # ------------------------------------------------------------
    # 4. Consensus strength per punto
    # ------------------------------------------------------------
    consensus_strength = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        same_final_cluster = np.where(labels_raw == labels_raw[i])[0]
        same_final_cluster = same_final_cluster[same_final_cluster != i]

        if len(same_final_cluster) == 0:
            consensus_strength[i] = 0.0
        else:
            consensus_strength[i] = consensus[i, same_final_cluster].mean()

    # ------------------------------------------------------------
    # 5. Filtro noise finale
    # ------------------------------------------------------------
    unstable = (
        (noise_frequency >= max_noise_frequency)
        | (consensus_strength < min_consensus_strength)
    )

    labels_final = labels_raw.copy()
    labels_final[unstable] = -1

    # ------------------------------------------------------------
    # 6. Reindex cluster in 0, 1, 2, ...
    # ------------------------------------------------------------
    non_noise = labels_final != -1
    unique_clusters = np.unique(labels_final[non_noise])

    relabel_map = {old: new for new, old in enumerate(unique_clusters)}

    labels_reindexed = np.full_like(labels_final, fill_value=-1)

    for old, new in relabel_map.items():
        labels_reindexed[labels_final == old] = new

    labels_final = labels_reindexed

    # ------------------------------------------------------------
    # 7. Probabilità/stabilità finale
    # ------------------------------------------------------------
    probabilities = consensus_strength * (1.0 - noise_frequency)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    probabilities[labels_final == -1] = 0.0

    return labels_final, probabilities