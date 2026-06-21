from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

import optuna
import umap
import hdbscan as hdbscan_lib

from hdbscan.validity import validity_index
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness


# ==================================================
# FUNZIONI DI SUPPORTO
# ==================================================

def mean_pairwise_ari(label_list):
    """
    Calcola l'ARI medio tra tutte le coppie di label.

    Serve per misurare quanto il clustering è stabile
    al variare del seed di UMAP.
    """

    if len(label_list) < 2:
        return np.nan

    ari_values = []

    for labels_a, labels_b in itertools.combinations(label_list, 2):
        ari = adjusted_rand_score(labels_a, labels_b)
        ari_values.append(ari)

    return float(np.mean(ari_values))


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

    # Distanze nello spazio originale
    dist_high = pairwise_distances(
        X_high,
        metric=high_metric,
    )

    # Distanze nello spazio ridotto
    dist_low = pairwise_distances(
        X_low,
        metric="euclidean",
    )

    # Ordine dei vicini, escludendo il punto stesso
    high_order = np.argsort(dist_high, axis=1)[:, 1:]
    low_order = np.argsort(dist_low, axis=1)[:, 1:]

    high_neighbors = high_order[:, :n_neighbors]
    low_neighbors = low_order[:, :n_neighbors]

    # Matrice dei rank nello spazio ridotto
    # low_ranks[i, j] = posizione/rank del punto j rispetto al punto i
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


def count_clusters_and_noise(labels):
    """
    Conta il numero di cluster escludendo il noise di HDBSCAN.

    In HDBSCAN il noise ha label -1.
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


def compute_dbcv(X_clustered, labels):
    """
    Calcola DBCV sullo stesso spazio usato da HDBSCAN.

    Se HDBSCAN lavora su UMAP, allora DBCV deve essere calcolato
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
):
    """
    Valuta una singola configurazione UMAP + HDBSCAN su più seed.

    Per ogni seed:
    - fitta UMAP
    - fitta HDBSCAN
    - calcola DBCV
    - controlla numero cluster
    - controlla noise ratio

    Alla fine restituisce:
    - DBCV medio
    - deviazione standard del DBCV
    - ARI medio tra seed
    - numero medio di cluster
    - noise medio
    - percentuale di seed validi
    """

    labels_per_seed = []

    dbcv_values = []
    n_clusters_values = []
    noise_ratio_values = []
    valid_flags = []

    seed_details = []

    for seed in seeds:

        seed_valid = True
        seed_reason = "ok"

        try:
            # ------------------------------
            # UMAP
            # ------------------------------

            umap_model = umap.UMAP(
                n_neighbors=umap_params["n_neighbors"],
                min_dist=umap_params["min_dist"],
                n_components=umap_params["n_components"],
                metric=umap_params["metric"],
                random_state=seed,
            )

            X_umap = umap_model.fit_transform(X)
            X_umap = np.asarray(X_umap, dtype=np.float64)
            X_umap = np.ascontiguousarray(X_umap)

            # ------------------------------
            # HDBSCAN
            # ------------------------------

            hdbscan_model = hdbscan_lib.HDBSCAN(
                min_cluster_size=hdbscan_params["min_cluster_size"],
                min_samples=hdbscan_params["min_samples"],
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )

            labels = hdbscan_model.fit_predict(X_umap)

            n_clusters, noise_ratio = count_clusters_and_noise(labels)

            # ------------------------------
            # DBCV
            # ------------------------------

            dbcv = compute_dbcv(
                X_clustered=X_umap,
                labels=labels,
            )

            # ------------------------------
            # Vincoli
            # ------------------------------

            if n_clusters < min_clusters:
                seed_valid = False
                seed_reason = "too_few_clusters"

            elif n_clusters > max_clusters:
                seed_valid = False
                seed_reason = "too_many_clusters"

            elif noise_ratio > max_noise_ratio:
                seed_valid = False
                seed_reason = "too_much_noise"

            elif np.isnan(dbcv):
                seed_valid = False
                seed_reason = "dbcv_not_available"

            labels_per_seed.append(labels)
            dbcv_values.append(dbcv)
            n_clusters_values.append(n_clusters)
            noise_ratio_values.append(noise_ratio)
            valid_flags.append(seed_valid)

            seed_details.append({
                "seed": seed,
                "dbcv": dbcv,
                "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
                "valid": seed_valid,
                "reason": seed_reason,
            })

        except Exception as error:
            valid_flags.append(False)

            seed_details.append({
                "seed": seed,
                "dbcv": np.nan,
                "n_clusters": np.nan,
                "noise_ratio": np.nan,
                "valid": False,
                "reason": f"error: {error}",
            })

    valid_flags = np.asarray(valid_flags, dtype=bool)

    # DBCV solo sui seed validi
    valid_dbcv_values = [
        dbcv
        for dbcv, is_valid in zip(dbcv_values, valid_flags)
        if is_valid and not np.isnan(dbcv)
    ]

    if len(valid_dbcv_values) > 0:
        mean_dbcv = float(np.mean(valid_dbcv_values))
        std_dbcv = float(np.std(valid_dbcv_values))
    else:
        mean_dbcv = np.nan
        std_dbcv = np.nan

    # ARI calcolato sui seed che hanno prodotto label
    mean_ari = mean_pairwise_ari(labels_per_seed)

    if len(n_clusters_values) > 0:
        mean_n_clusters = float(np.mean(n_clusters_values))
    else:
        mean_n_clusters = np.nan

    if len(noise_ratio_values) > 0:
        mean_noise_ratio = float(np.mean(noise_ratio_values))
    else:
        mean_noise_ratio = np.nan

    valid_seed_ratio = float(np.mean(valid_flags))

    return {
        "mean_dbcv": mean_dbcv,
        "std_dbcv": std_dbcv,
        "mean_ari": mean_ari,
        "mean_n_clusters": mean_n_clusters,
        "mean_noise_ratio": mean_noise_ratio,
        "valid_seed_ratio": valid_seed_ratio,
        "seed_details": seed_details,
    }


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

    dbcv_weight=0.70,
    stability_weight=0.30,

    final_random_state=42,
    optuna_random_state=42,

    trustworthiness_n_neighbors=15,
    compute_continuity=True,
):
    """
    Ottimizza UMAP + HDBSCAN usando Optuna.

    La funzione NON sceglie la configurazione migliore usando un solo seed UMAP.

    Ogni configurazione proposta da Optuna viene valutata su più seed.

    Criterio di selezione:

        selection_score =
            dbcv_weight * mean_dbcv
            +
            stability_weight * mean_ari

    Dove:
    - mean_dbcv misura la qualità density-based media
    - mean_ari misura la stabilità delle label tra seed diversi

    Dopo la selezione:
    - la pipeline viene rifittata con final_random_state
    - vengono calcolati trustworthiness e continuity come controlli finali
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

        # ------------------------------
        # Parametri UMAP
        # ------------------------------

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
            "metric": "euclidean"
        }

        # ------------------------------
        # Parametri HDBSCAN
        # ------------------------------

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

        # ------------------------------
        # Valutazione della configurazione
        # ------------------------------

        try:
            metrics = evaluate_umap_hdbscan_config(
                X=X,
                umap_params=umap_params,
                hdbscan_params=hdbscan_params,
                seeds=seeds,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                max_noise_ratio=max_noise_ratio,
            )

        except Exception as error:
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", f"evaluation_error: {error}")
            return invalid_score

        mean_dbcv = metrics["mean_dbcv"]
        std_dbcv = metrics["std_dbcv"]
        mean_ari = metrics["mean_ari"]
        mean_n_clusters = metrics["mean_n_clusters"]
        mean_noise_ratio = metrics["mean_noise_ratio"]
        valid_seed_ratio = metrics["valid_seed_ratio"]

        # ------------------------------
        # Salvataggio attributi Optuna
        # ------------------------------

        trial.set_user_attr("mean_dbcv", mean_dbcv)
        trial.set_user_attr("std_dbcv", std_dbcv)
        trial.set_user_attr("mean_ari", mean_ari)
        trial.set_user_attr("mean_n_clusters", mean_n_clusters)
        trial.set_user_attr("mean_noise_ratio", mean_noise_ratio)
        trial.set_user_attr("valid_seed_ratio", valid_seed_ratio)
        trial.set_user_attr("seed_details", metrics["seed_details"])

        # ------------------------------
        # Controlli di validità
        # ------------------------------

        if valid_seed_ratio < min_valid_seed_ratio:
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "too_few_valid_seeds")
            return invalid_score

        if np.isnan(mean_dbcv):
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "mean_dbcv_not_available")
            return invalid_score

        if np.isnan(mean_ari):
            trial.set_user_attr("config_valid", False)
            trial.set_user_attr("reason", "mean_ari_not_available")
            return invalid_score

        # ------------------------------
        # Score composito
        # ------------------------------

        selection_score = (
                dbcv_weight * mean_dbcv
                + stability_weight * mean_ari
        )

        trial.set_user_attr("config_valid", True)
        trial.set_user_attr("reason", "ok")
        trial.set_user_attr("selection_score", selection_score)

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

            "sanity_checks": None,

            "message": (
                "Nessuna configurazione valida trovata. "
                "Controlla la tabella trials per capire il motivo."
            ),
        }

    best_params = study.best_params

    # --------------------------------------------------
    # Fit finale riproducibile
    # --------------------------------------------------

    final_umap_model = umap.UMAP(
        n_neighbors=best_params["umap_n_neighbors"],
        min_dist=0.0,
        n_components=best_params["umap_n_components"],
        metric="euclidean",
        random_state=final_random_state,
    )

    best_umap_embedding = final_umap_model.fit_transform(X)
    best_umap_embedding = np.asarray(best_umap_embedding, dtype=np.float64)
    best_umap_embedding = np.ascontiguousarray(best_umap_embedding)

    final_hdbscan_model = hdbscan_lib.HDBSCAN(
        min_cluster_size=best_params["hdbscan_min_cluster_size"],
        min_samples=best_params["hdbscan_min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    best_labels = final_hdbscan_model.fit_predict(best_umap_embedding)
    best_membership_strengths = final_hdbscan_model.probabilities_

    final_n_clusters, final_noise_ratio = count_clusters_and_noise(
        best_labels
    )

    final_dbcv = compute_dbcv(
        X_clustered=best_umap_embedding,
        labels=best_labels,
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
            "Trustworthiness e continuity sono calcol   ate solo dopo la scelta "
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

        "hdbscan_min_cluster_size": best_params["hdbscan_min_cluster_size"],
        "hdbscan_min_samples": best_params["hdbscan_min_samples"],
        "hdbscan_metric": "euclidean",
        "cluster_selection_method": "eom",

        "final_random_state": final_random_state,
        "optuna_random_state": optuna_random_state,
    }

    return {
        "study": study,
        "trials": trials_df,

        "best_params": final_best_params,
        "best_selection_score": float(study.best_value),

        "best_mean_dbcv": best_trial.user_attrs.get("mean_dbcv"),
        "best_std_dbcv": best_trial.user_attrs.get("std_dbcv"),
        "best_mean_ari": best_trial.user_attrs.get("mean_ari"),
        "best_valid_seed_ratio": best_trial.user_attrs.get("valid_seed_ratio"),
        "best_mean_n_clusters": best_trial.user_attrs.get("mean_n_clusters"),
        "best_mean_noise_ratio": best_trial.user_attrs.get("mean_noise_ratio"),

        "best_labels": best_labels,
        "best_membership_strengths": best_membership_strengths,

        # Alias utile se nel tuo vecchio codice usavi ancora "probabilities"
        "best_probabilities": best_membership_strengths,

        "best_umap_embedding": best_umap_embedding,
        "best_umap_model": final_umap_model,
        "best_hdbscan_model": final_hdbscan_model,

        # Alias utile se nel vecchio codice usavi "best_model"
        "best_model": final_hdbscan_model,

        "final_dbcv": final_dbcv,
        "final_n_clusters": final_n_clusters,
        "final_noise_ratio": final_noise_ratio,

        "sanity_checks": sanity_checks,

        "message": "Ottimizzazione Optuna completata correttamente.",
    }