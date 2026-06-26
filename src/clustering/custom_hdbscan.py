from __future__ import annotations

import itertools
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

import optuna
import umap
import hdbscan as hdbscan_lib

from hdbscan.validity import validity_index
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness


# ==================================================
# FUNZIONI DI SUPPORTO GENERALI
# ==================================================

def count_clusters_and_noise(labels):
    """
    Conta il numero di cluster escludendo il rumore di HDBSCAN.

    In HDBSCAN il rumore ha label -1.
    """

    labels = np.asarray(labels)

    cluster_ids = [
        label
        for label in np.unique(labels)
        if label != -1
    ]

    n_clusters = len(cluster_ids)
    noise_ratio = float(np.mean(labels == -1))

    return n_clusters, noise_ratio


def check_seed_validity(
    n_clusters,
    noise_ratio,
    dbcv,
    min_clusters,
    max_clusters,
    max_noise_ratio,
):
    """
    Controlla se una singola run seed-based rispetta i vincoli pratici.
    """

    if n_clusters < min_clusters:
        return False, "too_few_clusters"

    if n_clusters > max_clusters:
        return False, "too_many_clusters"

    if noise_ratio > max_noise_ratio:
        return False, "too_much_noise"

    if np.isnan(dbcv):
        return False, "dbcv_not_available"

    return True, "ok"


def compute_dbcv(X_clustered, labels):
    """
    Calcola DBCV sullo stesso spazio usato da HDBSCAN.

    Se HDBSCAN lavora su UMAP, DBCV deve essere calcolato
    sull'embedding UMAP.
    """

    X_clustered = np.asarray(X_clustered, dtype=np.float64)
    X_clustered = np.ascontiguousarray(X_clustered)

    labels = np.asarray(labels)

    n_clusters, _ = count_clusters_and_noise(labels)

    if n_clusters < 2:
        return np.nan

    score = validity_index(
        X_clustered,
        labels,
        metric="euclidean",
    )

    return float(score)


def fit_umap_hdbscan(
    X,
    umap_params,
    hdbscan_params,
    seed,
):
    """
    Fitta UMAP + HDBSCAN per un singolo seed e restituisce embedding,
    modello UMAP, modello HDBSCAN, label e membership strengths.
    """

    umap_model = umap.UMAP(
        n_neighbors=umap_params["n_neighbors"],
        min_dist=umap_params["min_dist"],
        n_components=umap_params["n_components"],
        metric=umap_params["metric"],
        random_state=int(seed),
    )

    X_umap = umap_model.fit_transform(X)
    X_umap = np.asarray(X_umap, dtype=np.float64)
    X_umap = np.ascontiguousarray(X_umap)

    hdbscan_model = hdbscan_lib.HDBSCAN(
        min_cluster_size=int(hdbscan_params["min_cluster_size"]),
        min_samples=int(hdbscan_params["min_samples"]),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    labels = hdbscan_model.fit_predict(X_umap)
    membership_strengths = hdbscan_model.probabilities_

    return {
        "umap_embedding": X_umap,
        "umap_model": umap_model,
        "hdbscan_model": hdbscan_model,
        "labels": labels,
        "membership_strengths": membership_strengths,
    }


# ==================================================
# STABILITA' TRA SEED
# ==================================================

def pairwise_clustered_only_metrics(labels_a, labels_b):
    """
    Confronta due partizioni HDBSCAN separando cluster e rumore.

    Metrica principale:
    - ARI clustered-only: calcolato solo sui punti che NON sono rumore
      in entrambe le run.

    Metrica di controllo:
    - AMI clustered-only: stessa maschera dell'ARI.

    Rumore:
    - noise_agreement: accordo binario clusterizzato/rumore.
    - noise_jaccard: Jaccard tra gli insiemi dei punti rumore.

    Nota: l'etichetta -1 non viene trattata come cluster reale nell'ARI.
    """

    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    if labels_a.shape != labels_b.shape:
        raise ValueError(
            "Le due label array devono avere la stessa shape."
        )

    noise_a = labels_a == -1
    noise_b = labels_b == -1

    common_clustered_mask = (~noise_a) & (~noise_b)
    common_clustered_count = int(np.sum(common_clustered_mask))
    common_clustered_ratio = float(np.mean(common_clustered_mask))

    if common_clustered_count < 2:
        ari = np.nan
        ami = np.nan
    else:
        ari = adjusted_rand_score(
            labels_a[common_clustered_mask],
            labels_b[common_clustered_mask],
        )
        ami = adjusted_mutual_info_score(
            labels_a[common_clustered_mask],
            labels_b[common_clustered_mask],
        )
        ari = float(ari)
        ami = float(ami)

    noise_agreement = float(np.mean(noise_a == noise_b))

    noise_intersection = int(np.sum(noise_a & noise_b))
    noise_union = int(np.sum(noise_a | noise_b))

    if noise_union == 0:
        noise_jaccard = np.nan
    else:
        noise_jaccard = float(noise_intersection / noise_union)

    return {
        "ari": ari,
        "ami": ami,
        "common_clustered_count": common_clustered_count,
        "common_clustered_ratio": common_clustered_ratio,
        "noise_agreement": noise_agreement,
        "noise_jaccard": noise_jaccard,
    }


def _empty_stability_summary():
    return {
        "n_runs": 0,
        "seeds": [],
        "mean_ari": np.nan,
        "mean_ami": np.nan,
        "mean_common_clustered_ratio": np.nan,
        "mean_noise_agreement": np.nan,
        "mean_noise_jaccard": np.nan,
        "ari_matrix": [],
        "ami_matrix": [],
        "common_clustered_ratio_matrix": [],
        "noise_agreement_matrix": [],
        "noise_jaccard_matrix": [],
        "medoid_seed": None,
        "medoid_score": np.nan,
        "ranked_seeds_by_medoid_score": [],
    }


def pairwise_seed_stability_summary(seed_records):
    """
    Calcola la stabilità pairwise tra seed.

    Questa funzione va chiamata preferibilmente sui soli seed validi
    quando si vuole ottenere la stabilità principale della configurazione.

    Per ogni coppia di seed calcola:
    - ARI clustered-only
    - AMI clustered-only
    - common_clustered_ratio
    - noise_agreement
    - noise_jaccard

    Inoltre calcola il seed medoide, cioè il seed con ARI medio più alto
    verso gli altri seed considerati.
    """

    usable_records = [
        record
        for record in seed_records
        if record.get("labels") is not None
    ]

    n_runs = len(usable_records)

    if n_runs == 0:
        return _empty_stability_summary()

    seeds = [int(record["seed"]) for record in usable_records]

    ari_matrix = np.full((n_runs, n_runs), np.nan, dtype=np.float64)
    ami_matrix = np.full((n_runs, n_runs), np.nan, dtype=np.float64)
    common_matrix = np.full((n_runs, n_runs), np.nan, dtype=np.float64)
    noise_agreement_matrix = np.full((n_runs, n_runs), np.nan, dtype=np.float64)
    noise_jaccard_matrix = np.full((n_runs, n_runs), np.nan, dtype=np.float64)

    for i in range(n_runs):
        ari_matrix[i, i] = 1.0
        ami_matrix[i, i] = 1.0
        noise_agreement_matrix[i, i] = 1.0
        noise_jaccard_matrix[i, i] = 1.0
        common_matrix[i, i] = float(
            np.mean(np.asarray(usable_records[i]["labels"]) != -1)
        )

    for i, j in itertools.combinations(range(n_runs), 2):
        metrics = pairwise_clustered_only_metrics(
            usable_records[i]["labels"],
            usable_records[j]["labels"],
        )

        ari_matrix[i, j] = metrics["ari"]
        ari_matrix[j, i] = metrics["ari"]

        ami_matrix[i, j] = metrics["ami"]
        ami_matrix[j, i] = metrics["ami"]

        common_matrix[i, j] = metrics["common_clustered_ratio"]
        common_matrix[j, i] = metrics["common_clustered_ratio"]

        noise_agreement_matrix[i, j] = metrics["noise_agreement"]
        noise_agreement_matrix[j, i] = metrics["noise_agreement"]

        noise_jaccard_matrix[i, j] = metrics["noise_jaccard"]
        noise_jaccard_matrix[j, i] = metrics["noise_jaccard"]

    if n_runs < 2:
        upper_mask = np.zeros((n_runs, n_runs), dtype=bool)
    else:
        upper_mask = np.triu(np.ones((n_runs, n_runs), dtype=bool), k=1)

    def upper_nanmean(matrix):
        values = matrix[upper_mask]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return np.nan
        return float(np.mean(values))

    mean_ari = upper_nanmean(ari_matrix)
    mean_ami = upper_nanmean(ami_matrix)
    mean_common_clustered_ratio = upper_nanmean(common_matrix)
    mean_noise_agreement = upper_nanmean(noise_agreement_matrix)
    mean_noise_jaccard = upper_nanmean(noise_jaccard_matrix)

    # Medoide: seed con ARI medio più alto verso gli altri seed considerati.
    ranked_seeds = []

    for i, seed in enumerate(seeds):
        row = np.delete(ari_matrix[i], i)
        row = row[~np.isnan(row)]

        if len(row) == 0:
            score = np.nan
        else:
            score = float(np.mean(row))

        ranked_seeds.append({
            "seed": seed,
            "medoid_score": score,
        })

    ranked_seeds = sorted(
        ranked_seeds,
        key=lambda item: (
            -np.inf
            if np.isnan(item["medoid_score"])
            else item["medoid_score"]
        ),
        reverse=True,
    )

    if len(ranked_seeds) == 0 or np.isnan(ranked_seeds[0]["medoid_score"]):
        medoid_seed = None
        medoid_score = np.nan
    else:
        medoid_seed = int(ranked_seeds[0]["seed"])
        medoid_score = float(ranked_seeds[0]["medoid_score"])

    return {
        "n_runs": n_runs,
        "seeds": seeds,
        "mean_ari": mean_ari,
        "mean_ami": mean_ami,
        "mean_common_clustered_ratio": mean_common_clustered_ratio,
        "mean_noise_agreement": mean_noise_agreement,
        "mean_noise_jaccard": mean_noise_jaccard,
        "ari_matrix": ari_matrix.tolist(),
        "ami_matrix": ami_matrix.tolist(),
        "common_clustered_ratio_matrix": common_matrix.tolist(),
        "noise_agreement_matrix": noise_agreement_matrix.tolist(),
        "noise_jaccard_matrix": noise_jaccard_matrix.tolist(),
        "medoid_seed": medoid_seed,
        "medoid_score": medoid_score,
        "ranked_seeds_by_medoid_score": ranked_seeds,
    }


def modal_cluster_count_summary(seed_details, only_valid=True):
    """
    Calcola il numero di cluster modale e la sua frequenza.

    Se only_valid=True, considera solo le run valide.
    """

    if only_valid:
        rows = [row for row in seed_details if row.get("valid", False)]
    else:
        rows = list(seed_details)

    cluster_counts = [
        int(row["n_clusters"])
        for row in rows
        if not pd.isna(row.get("n_clusters", np.nan))
    ]

    if len(cluster_counts) == 0:
        return {
            "modal_n_clusters": None,
            "modal_n_clusters_ratio": np.nan,
            "cluster_count_distribution": {},
        }

    counter = Counter(cluster_counts)
    modal_n_clusters, modal_count = counter.most_common(1)[0]

    return {
        "modal_n_clusters": int(modal_n_clusters),
        "modal_n_clusters_ratio": float(modal_count / len(cluster_counts)),
        "cluster_count_distribution": {
            int(key): int(value)
            for key, value in sorted(counter.items())
        },
    }


# ==================================================
# RECOVERY PER-CLUSTER TIPO HENNIG
# ==================================================

def cluster_jaccard_recovery(
    reference_labels,
    comparison_records,
    exclude_noise=True,
):
    """
    Calcola la recovery per-cluster tramite Jaccard medio.

    Per ogni cluster della partizione di riferimento:
    - cerca, in ogni altra run, il cluster con massimo overlap Jaccard;
    - calcola la media dei migliori Jaccard;
    - classifica il cluster come:
        solid      se mean_jaccard >= 0.75
        borderline se 0.50 <= mean_jaccard < 0.75
        dissolved  se mean_jaccard < 0.50

    Nota metodologica:
    se comparison_records sono seed UMAP, questa è una recovery seed-based.
    La versione più rigorosa alla Hennig usa bootstrap/perturbazioni dei dati.
    """

    reference_labels = np.asarray(reference_labels)

    reference_cluster_ids = sorted(np.unique(reference_labels).tolist())

    if exclude_noise:
        reference_cluster_ids = [
            cluster_id
            for cluster_id in reference_cluster_ids
            if cluster_id != -1
        ]

    rows = []

    for ref_cluster_id in reference_cluster_ids:
        ref_mask = reference_labels == ref_cluster_id
        ref_size = int(np.sum(ref_mask))

        best_jaccards = []
        best_matches = []

        for record in comparison_records:
            comp_labels = np.asarray(record["labels"])
            comp_cluster_ids = sorted(np.unique(comp_labels).tolist())

            if exclude_noise:
                comp_cluster_ids = [
                    cluster_id
                    for cluster_id in comp_cluster_ids
                    if cluster_id != -1
                ]

            best_jaccard = 0.0
            best_cluster_id = None

            for comp_cluster_id in comp_cluster_ids:
                comp_mask = comp_labels == comp_cluster_id

                intersection = int(np.sum(ref_mask & comp_mask))
                union = int(np.sum(ref_mask | comp_mask))

                if union == 0:
                    jaccard = 0.0
                else:
                    jaccard = intersection / union

                if jaccard > best_jaccard:
                    best_jaccard = float(jaccard)
                    best_cluster_id = int(comp_cluster_id)

            best_jaccards.append(best_jaccard)
            best_matches.append({
                "seed": int(record["seed"]),
                "best_matching_cluster": best_cluster_id,
                "jaccard": float(best_jaccard),
            })

        if len(best_jaccards) == 0:
            mean_jaccard = np.nan
            std_jaccard = np.nan
            min_jaccard = np.nan
            max_jaccard = np.nan
            status = "not_available"
        else:
            mean_jaccard = float(np.mean(best_jaccards))
            std_jaccard = float(np.std(best_jaccards))
            min_jaccard = float(np.min(best_jaccards))
            max_jaccard = float(np.max(best_jaccards))

            if mean_jaccard >= 0.75:
                status = "solid"
            elif mean_jaccard >= 0.50:
                status = "borderline"
            else:
                status = "dissolved"

        rows.append({
            "cluster": int(ref_cluster_id),
            "cluster_size": ref_size,
            "mean_jaccard_recovery": mean_jaccard,
            "std_jaccard_recovery": std_jaccard,
            "min_jaccard_recovery": min_jaccard,
            "max_jaccard_recovery": max_jaccard,
            "status": status,
            "best_matches_by_seed": best_matches,
        })

    recovery_df = pd.DataFrame(rows)

    if len(recovery_df) > 0:
        recovery_df = recovery_df.sort_values(
            by="mean_jaccard_recovery",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return recovery_df


# ==================================================
# CONTINUITY
# ==================================================

def continuity_score(
    X_high,
    X_low,
    n_neighbors=15,
    high_metric="euclidean",
):
    """
    Calcola la continuity tra lo spazio originale e lo spazio ridotto.

    Trustworthiness:
    controlla se nello spazio ridotto compaiono falsi vicini.

    Continuity:
    controlla se lo spazio ridotto perde vicini che erano presenti
    nello spazio originale.

    Valori vicini a 1 sono migliori.
    """

    X_high = np.asarray(X_high, dtype=np.float64)
    X_low = np.asarray(X_low, dtype=np.float64)

    n_samples = X_high.shape[0]

    if n_neighbors >= n_samples:
        raise ValueError(
            "n_neighbors deve essere minore del numero di campioni."
        )

    dist_high = pairwise_distances(
        X_high,
        metric=high_metric,
    )

    dist_low = pairwise_distances(
        X_low,
        metric="euclidean",
    )

    high_order = np.argsort(dist_high, axis=1)[:, 1:]
    low_order = np.argsort(dist_low, axis=1)[:, 1:]

    high_neighbors = high_order[:, :n_neighbors]
    low_neighbors = low_order[:, :n_neighbors]

    low_ranks = np.empty((n_samples, n_samples), dtype=np.int64)

    for i in range(n_samples):
        low_ranks[i, low_order[i]] = np.arange(1, n_samples)
        low_ranks[i, i] = 0

    penalty = 0.0

    for i in range(n_samples):
        low_neighbor_set = set(low_neighbors[i])

        for neighbor in high_neighbors[i]:
            if neighbor not in low_neighbor_set:
                penalty += low_ranks[i, neighbor] - n_neighbors

    normalizer = (
        n_samples
        * n_neighbors
        * (2 * n_samples - 3 * n_neighbors - 1)
    )

    continuity = 1.0 - (2.0 / normalizer) * penalty

    return float(continuity)


# ==================================================
# VALUTAZIONE DI UNA CONFIGURAZIONE
# ==================================================

def evaluate_umap_hdbscan_config(
    X,
    umap_params,
    hdbscan_params,
    seeds,
    min_clusters,
    max_clusters,
    max_noise_ratio,
    return_seed_records=False,
):
    """
    Valuta una configurazione UMAP + HDBSCAN su più seed.

    Scelta metodologica:
    - valid_seed_ratio misura quante run rispettano i vincoli;
    - mean_ari_valid misura la stabilità SOLO tra seed validi;
    - mean_ari_all è mantenuto come diagnostica;
    - il rumore viene valutato separatamente;
    - il seed medoide viene calcolato dentro il sottoinsieme dei seed validi.
    """

    seed_records = []
    seed_details = []

    for seed in seeds:
        try:
            fit_result = fit_umap_hdbscan(
                X=X,
                umap_params=umap_params,
                hdbscan_params=hdbscan_params,
                seed=seed,
            )

            labels = fit_result["labels"]
            X_umap = fit_result["umap_embedding"]

            n_clusters, noise_ratio = count_clusters_and_noise(labels)
            dbcv = compute_dbcv(X_clustered=X_umap, labels=labels)

            seed_valid, seed_reason = check_seed_validity(
                n_clusters=n_clusters,
                noise_ratio=noise_ratio,
                dbcv=dbcv,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                max_noise_ratio=max_noise_ratio,
            )

            record = {
                "seed": int(seed),
                "labels": labels,
                "dbcv": dbcv,
                "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
                "valid": seed_valid,
                "reason": seed_reason,
            }

            detail = {
                "seed": int(seed),
                "dbcv": dbcv,
                "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
                "valid": seed_valid,
                "reason": seed_reason,
            }

            seed_records.append(record)
            seed_details.append(detail)

        except Exception as error:
            seed_details.append({
                "seed": int(seed),
                "dbcv": np.nan,
                "n_clusters": np.nan,
                "noise_ratio": np.nan,
                "valid": False,
                "reason": f"error: {error}",
            })

    valid_seed_records = [
        record
        for record in seed_records
        if record["valid"]
    ]

    valid_flags = np.asarray(
        [detail["valid"] for detail in seed_details],
        dtype=bool,
    )

    valid_seed_ratio = float(np.mean(valid_flags)) if len(valid_flags) > 0 else 0.0

    valid_dbcv_values = [
        record["dbcv"]
        for record in valid_seed_records
        if not np.isnan(record["dbcv"])
    ]

    if len(valid_dbcv_values) > 0:
        mean_dbcv = float(np.mean(valid_dbcv_values))
        std_dbcv = float(np.std(valid_dbcv_values))
    else:
        mean_dbcv = np.nan
        std_dbcv = np.nan

    valid_n_clusters = [
        record["n_clusters"]
        for record in valid_seed_records
    ]

    if len(valid_n_clusters) > 0:
        mean_n_clusters = float(np.mean(valid_n_clusters))
    else:
        mean_n_clusters = np.nan

    valid_noise_ratios = [
        record["noise_ratio"]
        for record in valid_seed_records
    ]

    if len(valid_noise_ratios) > 0:
        mean_noise_ratio = float(np.mean(valid_noise_ratios))
    else:
        mean_noise_ratio = np.nan

    stability_valid = pairwise_seed_stability_summary(valid_seed_records)
    stability_all = pairwise_seed_stability_summary(seed_records)

    modal_valid = modal_cluster_count_summary(seed_details, only_valid=True)
    modal_all = modal_cluster_count_summary(seed_details, only_valid=False)

    result = {
        "mean_dbcv": mean_dbcv,
        "std_dbcv": std_dbcv,

        # Metriche principali: SOLO seed validi.
        "mean_ari_valid": stability_valid["mean_ari"],
        "mean_ami_valid": stability_valid["mean_ami"],
        "mean_common_clustered_ratio_valid": stability_valid[
            "mean_common_clustered_ratio"
        ],
        "mean_noise_agreement_valid": stability_valid[
            "mean_noise_agreement"
        ],
        "mean_noise_jaccard_valid": stability_valid[
            "mean_noise_jaccard"
        ],

        # Diagnostica: tutte le run riuscite, anche non valide.
        "mean_ari_all": stability_all["mean_ari"],
        "mean_ami_all": stability_all["mean_ami"],
        "mean_common_clustered_ratio_all": stability_all[
            "mean_common_clustered_ratio"
        ],
        "mean_noise_agreement_all": stability_all[
            "mean_noise_agreement"
        ],
        "mean_noise_jaccard_all": stability_all[
            "mean_noise_jaccard"
        ],

        # Alias: per compatibilità, mean_ari indica la metrica principale.
        "mean_ari": stability_valid["mean_ari"],
        "mean_ami": stability_valid["mean_ami"],

        "mean_n_clusters": mean_n_clusters,
        "mean_noise_ratio": mean_noise_ratio,
        "valid_seed_ratio": valid_seed_ratio,
        "n_valid_seeds": int(len(valid_seed_records)),
        "n_total_seeds": int(len(seeds)),

        "modal_n_clusters_valid": modal_valid["modal_n_clusters"],
        "modal_n_clusters_ratio_valid": modal_valid[
            "modal_n_clusters_ratio"
        ],
        "cluster_count_distribution_valid": modal_valid[
            "cluster_count_distribution"
        ],

        "modal_n_clusters_all": modal_all["modal_n_clusters"],
        "modal_n_clusters_ratio_all": modal_all[
            "modal_n_clusters_ratio"
        ],
        "cluster_count_distribution_all": modal_all[
            "cluster_count_distribution"
        ],

        "stability_valid": stability_valid,
        "stability_all": stability_all,
        "seed_details": seed_details,
    }

    if return_seed_records:
        result["seed_records"] = seed_records
        result["valid_seed_records"] = valid_seed_records

    return result


# ==================================================
# OTTIMIZZAZIONE OPTUNA
# ==================================================

def optimize_umap_hdbscan_dbcv_optuna(
    X,
    n_trials=50,

    seeds=(11, 22, 33, 44, 55),

    umap_n_neighbors_values=(15, 25, 40, 60, 80),
    umap_n_components_values=(2, 5, 7, 10, 15, 20),

    hdbscan_min_cluster_size_values=(10, 15, 20, 25, 30, 40, 50),
    hdbscan_min_samples_values=(3, 5, 10, 15),

    min_clusters=6,
    max_clusters=12,
    max_noise_ratio=0.33,
    min_valid_seed_ratio=0.80,
    min_modal_cluster_ratio=0.70,
    min_common_clustered_ratio=0.20,

    dbcv_weight=0.70,
    stability_weight=0.30,

    final_random_state=42,
    optuna_random_state=42,

    trustworthiness_n_neighbors=15,
    compute_continuity=True,
):
    """
    Ottimizza UMAP + HDBSCAN usando Optuna.

    Metodo implementato:

    1. Ogni configurazione viene valutata su più seed UMAP.

    2. Prima si misura la riproducibilità pratica:
       valid_seed_ratio = quota di seed che rispettano:
       - min_clusters <= n_clusters <= max_clusters
       - noise_ratio <= max_noise_ratio
       - DBCV disponibile

    3. La stabilità principale viene calcolata SOLO tra i seed validi:
       - mean_ari_valid clustered-only
       - mean_ami_valid clustered-only
       - noise_agreement separato

    4. I seed non validi non entrano nell'ARI principale,
       ma penalizzano la configurazione tramite valid_seed_ratio.

    5. Il seed finale viene scelto come medoide tra i soli seed validi.

    6. Dopo il fit finale viene calcolata la Jaccard recovery per-cluster
       in stile Hennig, in versione seed-based.
    """

    # --------------------------------------------------
    # Controlli su X
    # --------------------------------------------------

    if hasattr(X, "embedding_"):
        raise TypeError(
            "Hai passato un modello UMAP, non una matrice numerica. "
            "Usa umap_model.embedding_ oppure umap_model.fit_transform(...)."
        )

    X = np.asarray(X, dtype=np.float64)
    X = np.ascontiguousarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"X deve essere una matrice 2D, ma ha shape {X.shape}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contiene valori NaN o infiniti."
        )

    invalid_score = -10.0

    # --------------------------------------------------
    # Objective Optuna
    # --------------------------------------------------

    def objective(trial):
        umap_params = {
            "n_neighbors": trial.suggest_categorical(
                "umap_n_neighbors",
                list(umap_n_neighbors_values),
            ),
            "min_dist": 0.0,
            "n_components": trial.suggest_categorical(
                "umap_n_components",
                list(umap_n_components_values),
            ),
            "metric": "euclidean",
        }

        min_cluster_size = trial.suggest_categorical(
            "hdbscan_min_cluster_size",
            list(hdbscan_min_cluster_size_values),
        )

        min_samples = trial.suggest_categorical(
            "hdbscan_min_samples",
            list(hdbscan_min_samples_values),
        )

        if min_samples > min_cluster_size:
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "min_samples > min_cluster_size")
            return invalid_score

        hdbscan_params = {
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
        }

        try:
            metrics = evaluate_umap_hdbscan_config(
                X=X,
                umap_params=umap_params,
                hdbscan_params=hdbscan_params,
                seeds=seeds,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                max_noise_ratio=max_noise_ratio,
                return_seed_records=False,
            )

        except Exception as error:
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", f"evaluation_error: {error}")
            return invalid_score

        mean_dbcv = metrics["mean_dbcv"]
        std_dbcv = metrics["std_dbcv"]
        mean_ari_valid = metrics["mean_ari_valid"]
        mean_ami_valid = metrics["mean_ami_valid"]
        mean_common_clustered_ratio_valid = metrics[
            "mean_common_clustered_ratio_valid"
        ]
        mean_noise_agreement_valid = metrics[
            "mean_noise_agreement_valid"
        ]
        mean_noise_jaccard_valid = metrics[
            "mean_noise_jaccard_valid"
        ]
        valid_seed_ratio = metrics["valid_seed_ratio"]
        modal_n_clusters_ratio_valid = metrics[
            "modal_n_clusters_ratio_valid"
        ]

        # --------------------------------------------------
        # Salvataggio attributi Optuna
        # --------------------------------------------------

        attrs_to_save = {
            "mean_dbcv": mean_dbcv,
            "std_dbcv": std_dbcv,

            "mean_ari_valid": mean_ari_valid,
            "mean_ami_valid": mean_ami_valid,
            "mean_common_clustered_ratio_valid": (
                mean_common_clustered_ratio_valid
            ),
            "mean_noise_agreement_valid": mean_noise_agreement_valid,
            "mean_noise_jaccard_valid": mean_noise_jaccard_valid,

            "mean_ari_all": metrics["mean_ari_all"],
            "mean_ami_all": metrics["mean_ami_all"],
            "mean_common_clustered_ratio_all": metrics[
                "mean_common_clustered_ratio_all"
            ],
            "mean_noise_agreement_all": metrics[
                "mean_noise_agreement_all"
            ],
            "mean_noise_jaccard_all": metrics[
                "mean_noise_jaccard_all"
            ],

            "mean_n_clusters": metrics["mean_n_clusters"],
            "mean_noise_ratio": metrics["mean_noise_ratio"],
            "valid_seed_ratio": valid_seed_ratio,
            "n_valid_seeds": metrics["n_valid_seeds"],
            "n_total_seeds": metrics["n_total_seeds"],

            "modal_n_clusters_valid": metrics[
                "modal_n_clusters_valid"
            ],
            "modal_n_clusters_ratio_valid": (
                modal_n_clusters_ratio_valid
            ),
            "cluster_count_distribution_valid": metrics[
                "cluster_count_distribution_valid"
            ],

            "modal_n_clusters_all": metrics["modal_n_clusters_all"],
            "modal_n_clusters_ratio_all": metrics[
                "modal_n_clusters_ratio_all"
            ],
            "cluster_count_distribution_all": metrics[
                "cluster_count_distribution_all"
            ],

            "medoid_seed_valid": metrics["stability_valid"][
                "medoid_seed"
            ],
            "medoid_score_valid": metrics["stability_valid"][
                "medoid_score"
            ],

            "seed_details": metrics["seed_details"],
        }

        for key, value in attrs_to_save.items():
            trial.set_user_attr(key, value)

        # --------------------------------------------------
        # Controlli di validità della configurazione
        # --------------------------------------------------

        if valid_seed_ratio < min_valid_seed_ratio:
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "too_few_valid_seeds")
            return invalid_score

        if np.isnan(mean_dbcv):
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "mean_dbcv_not_available")
            return invalid_score

        if np.isnan(mean_ari_valid):
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "mean_ari_valid_not_available")
            return invalid_score

        if np.isnan(mean_common_clustered_ratio_valid):
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr(
                "reason",
                "mean_common_clustered_ratio_not_available",
            )
            return invalid_score

        if mean_common_clustered_ratio_valid < min_common_clustered_ratio:
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "too_few_common_clustered_points")
            return invalid_score

        if np.isnan(modal_n_clusters_ratio_valid):
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "modal_cluster_ratio_not_available")
            return invalid_score

        if modal_n_clusters_ratio_valid < min_modal_cluster_ratio:
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "weak_modal_cluster_count")
            return invalid_score

        # --------------------------------------------------
        # Score composito
        # --------------------------------------------------

        selection_score = (
            dbcv_weight * mean_dbcv
            + stability_weight * mean_ari_valid
        )

        trial.set_user_attr("config_valid", True)
        trial.set_user_attr("reason", "ok")
        trial.set_user_attr("selection_score", float(selection_score))

        return float(selection_score)

    # --------------------------------------------------
    # Creazione e avvio studio Optuna
    # --------------------------------------------------

    sampler = optuna.samplers.TPESampler(
        seed=optuna_random_state,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    trials_df = study.trials_dataframe()
    best_trial = study.best_trial

    # --------------------------------------------------
    # Controllo: nessuna configurazione valida
    # --------------------------------------------------

    if not best_trial.user_attrs.get("config_valid", False):
        return {
            "study": study,
            "trials": trials_df,

            "best_params": None,
            "best_selection_score": None,
            "best_mean_dbcv": None,
            "best_std_dbcv": None,
            "best_mean_ari": None,
            "best_mean_ari_valid": None,
            "best_mean_ari_all": None,
            "best_valid_seed_ratio": None,
            "best_mean_n_clusters": None,
            "best_mean_noise_ratio": None,

            "best_labels": None,
            "best_membership_strengths": None,
            "best_probabilities": None,

            "best_umap_embedding": None,
            "best_umap_model": None,
            "best_hdbscan_model": None,
            "best_model": None,

            "final_dbcv": None,
            "final_n_clusters": None,
            "final_noise_ratio": None,

            "selected_final_seed": None,
            "selected_final_seed_reason": None,

            "cluster_recovery": None,
            "sanity_checks": None,

            "message": (
                "Nessuna configurazione valida trovata. "
                "Controlla la tabella trials per capire il motivo."
            ),
        }

    best_params = study.best_params

    best_umap_params = {
        "n_neighbors": best_params["umap_n_neighbors"],
        "min_dist": 0.0,
        "n_components": best_params["umap_n_components"],
        "metric": "euclidean",
    }

    best_hdbscan_params = {
        "min_cluster_size": best_params["hdbscan_min_cluster_size"],
        "min_samples": best_params["hdbscan_min_samples"],
    }

    # --------------------------------------------------
    # Rivalutazione della migliore configurazione
    # con salvataggio delle label per seed
    # --------------------------------------------------

    best_config_metrics = evaluate_umap_hdbscan_config(
        X=X,
        umap_params=best_umap_params,
        hdbscan_params=best_hdbscan_params,
        seeds=seeds,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        max_noise_ratio=max_noise_ratio,
        return_seed_records=True,
    )

    valid_seed_records = best_config_metrics["valid_seed_records"]
    stability_valid = best_config_metrics["stability_valid"]

    ranked_valid_seeds = stability_valid["ranked_seeds_by_medoid_score"]
    candidate_final_seeds = [
        item["seed"]
        for item in ranked_valid_seeds
        if not np.isnan(item["medoid_score"])
    ]

    if len(candidate_final_seeds) == 0:
        candidate_final_seeds = [
            int(record["seed"])
            for record in valid_seed_records
        ]

    if len(candidate_final_seeds) == 0:
        candidate_final_seeds = [int(final_random_state)]

    # --------------------------------------------------
    # Fit finale con seed medoide valido
    # --------------------------------------------------

    final_fit_result = None
    selected_final_seed = None
    selected_final_seed_reason = None

    for candidate_seed in candidate_final_seeds:
        fit_result = fit_umap_hdbscan(
            X=X,
            umap_params=best_umap_params,
            hdbscan_params=best_hdbscan_params,
            seed=candidate_seed,
        )

        labels = fit_result["labels"]
        X_umap = fit_result["umap_embedding"]
        n_clusters, noise_ratio = count_clusters_and_noise(labels)
        dbcv = compute_dbcv(X_clustered=X_umap, labels=labels)

        seed_valid, seed_reason = check_seed_validity(
            n_clusters=n_clusters,
            noise_ratio=noise_ratio,
            dbcv=dbcv,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            max_noise_ratio=max_noise_ratio,
        )

        if seed_valid:
            final_fit_result = fit_result
            selected_final_seed = int(candidate_seed)
            selected_final_seed_reason = "valid_medoid_or_next_valid_medoid_rank"
            break

    if final_fit_result is None:
        # Fallback raro: se qualcosa non è riproducibile, usa final_random_state
        # ma segnala chiaramente il problema nel messaggio finale.
        final_fit_result = fit_umap_hdbscan(
            X=X,
            umap_params=best_umap_params,
            hdbscan_params=best_hdbscan_params,
            seed=final_random_state,
        )
        selected_final_seed = int(final_random_state)
        selected_final_seed_reason = "fallback_final_random_state_not_guaranteed_valid"

    final_umap_model = final_fit_result["umap_model"]
    best_umap_embedding = final_fit_result["umap_embedding"]
    final_hdbscan_model = final_fit_result["hdbscan_model"]
    best_labels = final_fit_result["labels"]
    best_membership_strengths = final_fit_result["membership_strengths"]

    final_n_clusters, final_noise_ratio = count_clusters_and_noise(
        best_labels
    )

    final_dbcv = compute_dbcv(
        X_clustered=best_umap_embedding,
        labels=best_labels,
    )

    # --------------------------------------------------
    # Recovery per-cluster tipo Hennig, seed-based
    # --------------------------------------------------

    comparison_records = [
        record
        for record in valid_seed_records
        if int(record["seed"]) != int(selected_final_seed)
    ]

    cluster_recovery = cluster_jaccard_recovery(
        reference_labels=best_labels,
        comparison_records=comparison_records,
        exclude_noise=True,
    )

    # --------------------------------------------------
    # Sanity checks finali: trustworthiness e continuity
    # --------------------------------------------------

    final_trustworthiness = trustworthiness(
        X,
        best_umap_embedding,
        n_neighbors=trustworthiness_n_neighbors,
        metric="euclidean",
    )

    if compute_continuity:
        final_continuity = continuity_score(
            X_high=X,
            X_low=best_umap_embedding,
            n_neighbors=trustworthiness_n_neighbors,
            high_metric="euclidean",
        )
    else:
        final_continuity = None

    sanity_checks = {
        "trustworthiness": float(final_trustworthiness),
        "continuity": (
            float(final_continuity)
            if final_continuity is not None
            else None
        ),
        "n_neighbors": trustworthiness_n_neighbors,
        "note": (
            "Trustworthiness e continuity sono calcolate solo dopo la scelta "
            "della configurazione. Non partecipano all'ottimizzazione."
        ),
    }

    # --------------------------------------------------
    # Dizionario finale dei parametri
    # --------------------------------------------------

    final_best_params = {
        "umap_n_neighbors": best_params["umap_n_neighbors"],
        "umap_min_dist": 0.0,
        "umap_n_components": best_params["umap_n_components"],
        "umap_metric": "euclidean",

        "hdbscan_min_cluster_size": best_params[
            "hdbscan_min_cluster_size"
        ],
        "hdbscan_min_samples": best_params["hdbscan_min_samples"],
        "hdbscan_metric": "euclidean",
        "cluster_selection_method": "eom",

        "selected_final_seed": selected_final_seed,
        "selected_final_seed_reason": selected_final_seed_reason,
        "optuna_random_state": optuna_random_state,
    }

    return {
        "study": study,
        "trials": trials_df,

        "best_params": final_best_params,
        "best_selection_score": float(study.best_value),

        "best_mean_dbcv": best_config_metrics["mean_dbcv"],
        "best_std_dbcv": best_config_metrics["std_dbcv"],

        # Metriche principali: seed validi.
        "best_mean_ari": best_config_metrics["mean_ari_valid"],
        "best_mean_ari_valid": best_config_metrics["mean_ari_valid"],
        "best_mean_ami_valid": best_config_metrics["mean_ami_valid"],
        "best_mean_common_clustered_ratio_valid": (
            best_config_metrics["mean_common_clustered_ratio_valid"]
        ),
        "best_mean_noise_agreement_valid": (
            best_config_metrics["mean_noise_agreement_valid"]
        ),
        "best_mean_noise_jaccard_valid": (
            best_config_metrics["mean_noise_jaccard_valid"]
        ),

        # Diagnostica: tutte le run riuscite.
        "best_mean_ari_all": best_config_metrics["mean_ari_all"],
        "best_mean_ami_all": best_config_metrics["mean_ami_all"],
        "best_mean_common_clustered_ratio_all": (
            best_config_metrics["mean_common_clustered_ratio_all"]
        ),
        "best_mean_noise_agreement_all": (
            best_config_metrics["mean_noise_agreement_all"]
        ),
        "best_mean_noise_jaccard_all": (
            best_config_metrics["mean_noise_jaccard_all"]
        ),

        "best_valid_seed_ratio": best_config_metrics["valid_seed_ratio"],
        "best_n_valid_seeds": best_config_metrics["n_valid_seeds"],
        "best_n_total_seeds": best_config_metrics["n_total_seeds"],

        "best_mean_n_clusters": best_config_metrics["mean_n_clusters"],
        "best_mean_noise_ratio": best_config_metrics["mean_noise_ratio"],

        "best_modal_n_clusters_valid": best_config_metrics[
            "modal_n_clusters_valid"
        ],
        "best_modal_n_clusters_ratio_valid": best_config_metrics[
            "modal_n_clusters_ratio_valid"
        ],
        "best_cluster_count_distribution_valid": best_config_metrics[
            "cluster_count_distribution_valid"
        ],

        "best_stability_valid": best_config_metrics["stability_valid"],
        "best_stability_all": best_config_metrics["stability_all"],
        "best_seed_details": best_config_metrics["seed_details"],

        "best_medoid_seed": stability_valid["medoid_seed"],
        "best_medoid_score": stability_valid["medoid_score"],
        "best_ranked_valid_seeds_by_medoid_score": (
            stability_valid["ranked_seeds_by_medoid_score"]
        ),
        "selected_final_seed": selected_final_seed,
        "selected_final_seed_reason": selected_final_seed_reason,

        "best_labels": best_labels,
        "best_membership_strengths": best_membership_strengths,

        # Alias utile se nel vecchio codice usavi ancora "probabilities".
        "best_probabilities": best_membership_strengths,

        "best_umap_embedding": best_umap_embedding,
        "best_umap_model": final_umap_model,
        "best_hdbscan_model": final_hdbscan_model,

        # Alias utile se nel vecchio codice usavi "best_model".
        "best_model": final_hdbscan_model,

        "final_dbcv": final_dbcv,
        "final_n_clusters": final_n_clusters,
        "final_noise_ratio": final_noise_ratio,

        "cluster_recovery": cluster_recovery,
        "sanity_checks": sanity_checks,

        "message": (
            "Ottimizzazione Optuna completata con stabilità calcolata "
            "sui soli seed validi, ARI/AMI clustered-only, seed finale "
            "medoide valido e recovery Jaccard per-cluster."
        ),
    }
