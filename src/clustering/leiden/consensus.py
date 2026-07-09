import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _n_clusters_no_noise(labels):
    labels = np.asarray(labels)
    return len(set(labels) - {-1})


def _relabel_contiguous(labels):
    labels = np.asarray(labels)
    new_labels = np.full(labels.shape, -1, dtype=int)

    valid_labels = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(valid_labels)}

    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels, mapping


def _agglomerative_precomputed(distance, n_clusters):
    """
    Compatibile sia con vecchie che nuove versioni di sklearn.
    """
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )

    return model.fit_predict(distance)


def consensus_clustering_from_labels_leiden(
    label_runs,
    n_clusters="median",
    min_consensus_strength=0.70,
    mark_unstable_as_noise=False,
    strength_stat="mean",
    return_diagnostics=False,
):
    """
    Consensus clustering per Leiden.

    Parameters
    ----------
    label_runs : list/array, shape = (n_runs, n_samples)
        Lista delle labels ottenute da più run Leiden.

    n_clusters : "median", "mode" oppure int
        Numero di cluster finali da usare nell'agglomerative clustering
        sulla consensus matrix.

    min_consensus_strength : float
        Soglia sotto cui un punto è considerato instabile.

    mark_unstable_as_noise : bool
        Se False, tutti i punti restano assegnati a un cluster.
        Se True, i punti instabili vengono marcati come -1.

    strength_stat : "mean" oppure "median"
        Come calcolare la forza di consenso interna al cluster finale.

    return_diagnostics : bool
        Se True, restituisce anche un dizionario diagnostico.

    Returns
    -------
    labels_final : np.ndarray
        Labels finali del consensus.

    probabilities : np.ndarray
        Stabilità/probabilità di assegnazione di ciascun punto.

    diagnostics : dict, opzionale
        Informazioni aggiuntive sul consensus.
    """

    label_runs = np.asarray(label_runs)

    if label_runs.ndim != 2:
        raise ValueError("label_runs deve avere shape (n_runs, n_samples).")

    n_runs, n_samples = label_runs.shape

    if n_runs < 2:
        raise ValueError("Servono almeno 2 run per fare consensus clustering.")

    if strength_stat not in {"mean", "median"}:
        raise ValueError("strength_stat deve essere 'mean' oppure 'median'.")

    # 1. Numero di cluster finale
    ks = np.array([
        _n_clusters_no_noise(labels)
        for labels in label_runs
    ])

    ks = ks[ks > 0]

    if len(ks) == 0:
        raise ValueError("Nessuna run contiene cluster validi.")

    if n_clusters == "median":
        n_clusters_final = int(round(np.median(ks)))

    elif n_clusters == "mode":
        values, counts = np.unique(ks, return_counts=True)
        n_clusters_final = int(values[np.argmax(counts)])

    elif isinstance(n_clusters, int):
        n_clusters_final = n_clusters

    else:
        raise ValueError("n_clusters deve essere 'median', 'mode' oppure un intero.")

    n_clusters_final = max(1, n_clusters_final)

    if n_clusters_final >= n_samples:
        raise ValueError("n_clusters_final deve essere minore di n_samples.")

    # 2. Consensus matrix: quante volte due punti stanno nello stesso cluster
    same_cluster = np.zeros((n_samples, n_samples), dtype=np.float32)

    for labels in label_runs:
        labels = np.asarray(labels)

        for c in np.unique(labels):
            if c == -1:
                continue

            idx = np.where(labels == c)[0]
            same_cluster[np.ix_(idx, idx)] += 1

    consensus = same_cluster / float(n_runs)
    np.fill_diagonal(consensus, 1.0)

    # 3. Clustering finale sulla matrice di distanza consensus
    distance = 1.0 - consensus
    np.fill_diagonal(distance, 0.0)

    labels_raw = _agglomerative_precomputed(
        distance=distance,
        n_clusters=n_clusters_final,
    )

    # 4. Consensus strength per punto
    consensus_strength = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        same_final_cluster = np.where(labels_raw == labels_raw[i])[0]
        same_final_cluster = same_final_cluster[same_final_cluster != i]

        if len(same_final_cluster) == 0:
            consensus_strength[i] = 0.0
        else:
            values = consensus[i, same_final_cluster]

            if strength_stat == "mean":
                consensus_strength[i] = np.mean(values)
            else:
                consensus_strength[i] = np.median(values)

    unstable = consensus_strength < min_consensus_strength

    # 5. Labels finali
    labels_final = labels_raw.copy()

    if mark_unstable_as_noise:
        labels_final[unstable] = -1

    labels_final, relabel_map = _relabel_contiguous(labels_final)

    # 6. Probabilità/stabilità
    probabilities = np.clip(consensus_strength, 0.0, 1.0)

    if mark_unstable_as_noise:
        probabilities[labels_final == -1] = 0.0

    if return_diagnostics:
        diagnostics = {
            "consensus_matrix": consensus,
            "same_cluster": same_cluster,
            "labels_raw_before_filter": labels_raw,
            "labels_final": labels_final,
            "probabilities": probabilities,
            "consensus_strength": consensus_strength,
            "unstable": unstable,
            "unstable_ratio": float(np.mean(unstable)),
            "n_unstable": int(np.sum(unstable)),
            "n_clusters_requested": n_clusters,
            "n_clusters_final": n_clusters_final,
            "ks_per_run": ks,
            "n_runs": n_runs,
            "relabel_map": relabel_map,
        }

        return labels_final, probabilities, diagnostics

    return labels_final, probabilities