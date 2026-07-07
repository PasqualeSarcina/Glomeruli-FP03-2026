import numpy as np
from sklearn.cluster import AgglomerativeClustering


def consensus_clustering_from_labels_leiden(
    label_runs,
    n_clusters="median",
    min_consensus_strength=0.50,
    mark_unstable_as_noise=True,
    return_diagnostics=False,
):
    """
    Consensus clustering specifico per Leiden.

    Leiden assegna normalmente ogni punto a un cluster.
    Quindi:
    - non esiste noise nativo -1
    - eventuali -1 finali indicano punti instabili/ambigui
      secondo il consensus clustering

    Parameters
    ----------
    label_runs : list[np.ndarray] or np.ndarray
        Lista/array di labels prodotti da più run Leiden.
        Shape attesa: (n_runs, n_samples)

    n_clusters : int or "median"
        Numero di cluster finali.
        Se "median", usa la mediana del numero di cluster trovati nelle run.

    min_consensus_strength : float
        Soglia minima di stabilità di consenso.
        Se un punto ha consensus_strength sotto questa soglia,
        viene marcato come -1 se mark_unstable_as_noise=True.

    mark_unstable_as_noise : bool
        Se True, i punti instabili vengono marcati come -1.
        Nel caso Leiden, -1 significa "instabile/ambiguo",
        non rumore density-based.

    return_diagnostics : bool
        Se True, restituisce anche un dizionario diagnostico.

    Returns
    -------
    labels_final : np.ndarray
        Labels finali del consensus clustering.

    probabilities : np.ndarray
        Stabilità di consenso per ogni punto.
        Valori in [0, 1].
        Per i punti finali -1 vale 0 se mark_unstable_as_noise=True.

    diagnostics : dict, optional
        Restituito solo se return_diagnostics=True.
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

    if np.any(label_runs == -1):
        raise ValueError(
            "Questa funzione è pensata per labels Leiden raw, quindi senza -1. "
            "Se hai -1 nei label_runs, stai probabilmente passando labels già filtrati."
        )

    # ------------------------------------------------------------
    # 1. Numero di cluster finale
    # ------------------------------------------------------------
    if n_clusters == "median":
        ks = []

        for labels in label_runs:
            k = len(np.unique(labels))
            ks.append(k)

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

    for labels in label_runs:
        labels = np.asarray(labels)

        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            same_cluster[np.ix_(idx, idx)] += 1

    consensus = same_cluster / float(n_runs)

    np.fill_diagonal(consensus, 1.0)

    # ------------------------------------------------------------
    # 3. Clustering finale su distanza 1 - consensus
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
    # 5. Marcatura punti instabili
    # ------------------------------------------------------------
    unstable = consensus_strength < min_consensus_strength

    labels_final = labels_raw.copy()

    if mark_unstable_as_noise:
        labels_final[unstable] = -1

    # ------------------------------------------------------------
    # 6. Reindex cluster finali in 0, 1, 2, ...
    # ------------------------------------------------------------
    non_noise = labels_final != -1
    unique_clusters = np.unique(labels_final[non_noise])

    relabel_map = {
        old_label: new_label
        for new_label, old_label in enumerate(unique_clusters)
    }

    labels_reindexed = np.full_like(labels_final, fill_value=-1)

    for old_label, new_label in relabel_map.items():
        labels_reindexed[labels_final == old_label] = new_label

    labels_final = labels_reindexed

    # ------------------------------------------------------------
    # 7. Probabilità/stabilità finale
    # ------------------------------------------------------------
    probabilities = np.clip(consensus_strength.copy(), 0.0, 1.0)

    if mark_unstable_as_noise:
        probabilities[labels_final == -1] = 0.0

    if return_diagnostics:
        diagnostics = {
            "consensus_matrix": consensus,
            "distance_matrix": distance,
            "consensus_strength": consensus_strength,
            "unstable": unstable,
            "labels_raw_before_filter": labels_raw,
            "n_clusters_requested": n_clusters_final,
            "n_runs": n_runs,
            "n_samples": n_samples,
            "mean_consensus_strength": float(np.mean(consensus_strength)),
            "median_consensus_strength": float(np.median(consensus_strength)),
            "unstable_ratio": float(np.mean(unstable)),
        }

        return labels_final, probabilities, diagnostics

    return labels_final, probabilities