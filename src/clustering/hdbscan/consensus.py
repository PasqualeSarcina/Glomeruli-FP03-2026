import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import Counter


def _n_non_noise_clusters(labels):
    labels = np.asarray(labels)
    return len(set(labels) - {-1})


def _relabel_contiguous(labels):
    labels = np.asarray(labels)
    new_labels = np.full(labels.shape, -1, dtype=int)

    valid_old = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(valid_old)}

    for old, new in mapping.items():
        new_labels[labels == old] = new

    return new_labels, mapping


def _agglomerative_precomputed(distance, n_clusters):
    """
    Compatibile con versioni sklearn vecchie e nuove.
    """
    try:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance)
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        ).fit_predict(distance)


def consensus_clustering_from_labels_hdbscan(
    label_runs,
    n_clusters="median",
    ignore_noise=True,
    max_noise_frequency=0.50,
    min_consensus_strength=0.30,
    min_final_cluster_size=5,
    strength_stat="mean",
    return_diagnostics=False,
):
    """
    Consensus clustering robusto per label prodotte da più run HDBSCAN.

    Parameters
    ----------
    label_runs : list[np.ndarray] or np.ndarray
        Lista/array di labels, shape (n_runs, n_samples).

    n_clusters : int, "median", or "mode"
        Numero di cluster finali da usare sul consensus.
        - "median": mediana dei cluster non-noise nelle run.
        - "mode": numero di cluster più frequente nelle run.
        - int: numero fissato manualmente.

    ignore_noise : bool
        Se True, le coppie in cui almeno un punto è -1 non entrano nel
        denominatore della co-associazione.

    max_noise_frequency : float
        Punto scartato se è noise in più di questa frazione di run.

    min_consensus_strength : float
        Punto scartato se la sua co-associazione media/mediana col proprio
        cluster finale è sotto questa soglia.

    min_final_cluster_size : int
        Cluster finali più piccoli di questa soglia vengono rimarcati come noise.

    strength_stat : {"mean", "median"}
        Come calcolare la stabilità del punto rispetto al proprio cluster finale.

    return_diagnostics : bool
        Se True, restituisce anche un dizionario diagnostico.

    Returns
    -------
    labels_final : np.ndarray
        Label finali, con -1 per noise.

    probabilities : np.ndarray
        Stabilità finale per ogni punto, in [0, 1].
    """

    label_runs = np.asarray(label_runs)

    if label_runs.ndim != 2:
        raise ValueError("label_runs deve avere shape (n_runs, n_samples).")

    n_runs, n_samples = label_runs.shape

    if n_runs < 2:
        raise ValueError("Servono almeno 2 run per fare consensus clustering.")

    if strength_stat not in {"mean", "median"}:
        raise ValueError("strength_stat deve essere 'mean' oppure 'median'.")

    # ------------------------------------------------------------
    # 1. Numero di cluster richiesto
    # ------------------------------------------------------------
    ks = [_n_non_noise_clusters(labels) for labels in label_runs]
    ks_valid = [k for k in ks if k > 0]

    if len(ks_valid) == 0:
        raise ValueError("Nessuna run contiene cluster validi non-noise.")

    if n_clusters == "median":
        n_clusters_final = int(round(np.median(ks_valid)))
    elif n_clusters == "mode":
        n_clusters_final = Counter(ks_valid).most_common(1)[0][0]
    elif isinstance(n_clusters, int):
        n_clusters_final = n_clusters
    else:
        raise ValueError("n_clusters deve essere int, 'median' oppure 'mode'.")

    n_clusters_final = max(1, int(n_clusters_final))

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

        # Denominatore: coppie osservate insieme come non-noise
        co_observed[np.ix_(valid_idx, valid_idx)] += 1

        # Numeratore: coppie finite nello stesso cluster
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

    # ------------------------------------------------------------
    # 3. Frequenza noise per punto
    # ------------------------------------------------------------
    noise_frequency = np.mean(label_runs == -1, axis=0)

    # Punto eleggibile se non è troppo spesso noise
    eligible = noise_frequency <= max_noise_frequency
    eligible_idx = np.where(eligible)[0]

    labels_raw = np.full(n_samples, -1, dtype=int)
    labels_filtered = np.full(n_samples, -1, dtype=int)
    consensus_strength = np.zeros(n_samples, dtype=np.float32)

    # Se non ci sono abbastanza punti eleggibili, restituisco tutto noise
    if len(eligible_idx) < max(2, min_final_cluster_size):
        labels_final = np.full(n_samples, -1, dtype=int)
        probabilities = np.zeros(n_samples, dtype=np.float32)

        if return_diagnostics:
            diagnostics = {
                "consensus_matrix": consensus,
                "co_observed": co_observed,
                "same_cluster": same_cluster,
                "noise_frequency": noise_frequency,
                "eligible": eligible,
                "consensus_strength": consensus_strength,
                "labels_raw_before_filter": labels_raw,
                "labels_filtered_before_reindex": labels_filtered,
                "n_clusters_requested_initial": n_clusters_final,
                "n_clusters_used": 0,
                "ks_per_run": ks,
                "n_runs": n_runs,
            }
            return labels_final, probabilities, diagnostics

        return labels_final, probabilities

    # Non ha senso chiedere più cluster di quanti siano compatibili
    # con la dimensione minima finale.
    max_reasonable_k = max(1, len(eligible_idx) // max(1, min_final_cluster_size))
    n_clusters_used = min(n_clusters_final, max_reasonable_k)

    # ------------------------------------------------------------
    # 4. Clustering finale solo sui punti eleggibili
    # ------------------------------------------------------------
    consensus_sub = consensus[np.ix_(eligible_idx, eligible_idx)]

    distance_sub = 1.0 - consensus_sub
    distance_sub = np.clip(distance_sub, 0.0, 1.0)
    np.fill_diagonal(distance_sub, 0.0)

    raw_sub = _agglomerative_precomputed(
        distance_sub,
        n_clusters=n_clusters_used,
    )

    labels_raw[eligible_idx] = raw_sub

    # ------------------------------------------------------------
    # 5. Consensus strength per punto
    # ------------------------------------------------------------
    for c in sorted(set(labels_raw) - {-1}):
        idx = np.where(labels_raw == c)[0]

        if len(idx) <= 1:
            consensus_strength[idx] = 0.0
            continue

        block = consensus[np.ix_(idx, idx)].copy()
        np.fill_diagonal(block, np.nan)

        if strength_stat == "mean":
            consensus_strength[idx] = np.nanmean(block, axis=1)
        elif strength_stat == "median":
            consensus_strength[idx] = np.nanmedian(block, axis=1)

    # ------------------------------------------------------------
    # 6. Filtro per punti deboli
    # ------------------------------------------------------------
    labels_filtered = labels_raw.copy()

    weak = consensus_strength < min_consensus_strength
    labels_filtered[weak] = -1

    # ------------------------------------------------------------
    # 7. Filtro cluster finali troppo piccoli
    # ------------------------------------------------------------
    for c in sorted(set(labels_filtered) - {-1}):
        idx = np.where(labels_filtered == c)[0]

        if len(idx) < min_final_cluster_size:
            labels_filtered[idx] = -1

    # ------------------------------------------------------------
    # 8. Reindex finale
    # ------------------------------------------------------------
    labels_final, relabel_map = _relabel_contiguous(labels_filtered)

    # ------------------------------------------------------------
    # 9. Probabilità/stabilità finale
    # ------------------------------------------------------------
    probabilities = consensus_strength * (1.0 - noise_frequency)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities[labels_final == -1] = 0.0

    if return_diagnostics:
        diagnostics = {
            "consensus_matrix": consensus,
            "co_observed": co_observed,
            "same_cluster": same_cluster,
            "noise_frequency": noise_frequency,
            "eligible": eligible,
            "consensus_strength": consensus_strength,
            "weak": weak,
            "labels_raw_before_filter": labels_raw,
            "labels_filtered_before_reindex": labels_filtered,
            "labels_final": labels_final,
            "probabilities": probabilities,
            "relabel_map": relabel_map,
            "n_clusters_requested_initial": n_clusters_final,
            "n_clusters_used": n_clusters_used,
            "ks_per_run": ks,
            "n_runs": n_runs,
        }

        return labels_final, probabilities, diagnostics

    return labels_final, probabilities