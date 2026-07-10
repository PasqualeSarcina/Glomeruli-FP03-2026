import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.embedding_evaluation import find_exact_duplicate, evaluate_embedding_norms
from src.metrics.hopkins import compute_hopkins_dataframe
from src.metrics.nearest_neighbor import evaluate_embedding_backbone
from src.metrics.deterministic_separability import (
    effective_dimensionality,
    ward_clustering_curve,
    summarize_ward_curve,
    hdbscan_clustering_metrics,
    summarize_hdbscan_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confronta backbone diverse tramite metriche intrinseche e "
            "deterministiche (nessun UMAP/HDBSCAN)."
        )
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glomeruli" / "embeddings",
        help="Directory con i file .npy prodotti da extract_glomeruli_embeds.py.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="Glob pattern per selezionare i file di embedding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "backbone_evaluation",
        help="Directory dove salvare i risultati.",
    )
    parser.add_argument(
        "--ward-k-min", type=int, default=2, help="k minimo per la curva Ward."
    )
    parser.add_argument(
        "--ward-k-max", type=int, default=15, help="k massimo per la curva Ward."
    )
    parser.add_argument(
        "--hopkins-n-runs",
        type=int,
        default=50,
        help="Numero di ripetizioni per la statistica di Hopkins.",
    )
    parser.add_argument(
        "--metric-sample-size",
        type=int,
        default=3000,
        help=(
            "Numero massimo di campioni usati per le metriche O(n^2) "
            "(silhouette, Ward). 0 disabilita il sottocampionamento."
        ),
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed per il sottocampionamento."
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["standard", "l2", "none"],
        default="standard",
        help=(
            "Preprocessing applicato agli embedding prima di PCA/Hopkins/Ward. "
            "'standard' = StandardScaler (per-feature, media 0 var 1). "
            "'l2' = normalizzazione L2 per-campione (coerente con distanza coseno / "
            "il preprocessing usato nella pipeline di clustering). "
            "'none' = nessuna normalizzazione. Default: standard."
        ),
    )
    parser.add_argument(
        "--pca-n-components",
        type=int,
        default=10,
        help=(
            "Numero di componenti PCA (dimensionalita' fissa bassa) usato prima "
            "della curva Ward 'realistica'. Riproduce la condizione della pipeline "
            "reale, che clusterizza dopo riduzione a poche dimensioni. "
            "Usa 0 per DISATTIVARE la PCA e valutare sullo spazio pieno "
            "(utile come confronto per giustificare l'uso della PCA). Default: 10."
        ),
    )
    parser.add_argument(
        "--hdbscan-min-cluster-sizes",
        type=int,
        nargs="+",
        default=[10, 20, 40],
        help=(
            "Valori di min_cluster_size provati dalla sonda HDBSCAN "
            "(density-based, deterministica). Default: 10 20 40."
        ),
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    X = np.load(path)
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"{path} deve contenere un array 2D, trovato shape={X.shape}.")

    finite_mask = np.all(np.isfinite(X), axis=1)
    n_removed = int(np.sum(~finite_mask))
    if n_removed > 0:
        print(f"  [WARN] {path.name}: rimuovo {n_removed} righe con NaN/Inf.")
        X = X[finite_mask]

    return X.astype(np.float32, copy=False)


def subsample(X: np.ndarray, max_samples: int, random_state: int) -> np.ndarray:
    n_samples = X.shape[0]
    if max_samples <= 0 or n_samples <= max_samples:
        return X
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(n_samples, size=max_samples, replace=False))
    return X[idx]


def evaluate_single_backbone(
    name: str,
    X_raw: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    print(f"\n=== {name} === (n_samples={X_raw.shape[0]}, n_features={X_raw.shape[1]})")

    row: dict = {"backbone": name, "n_samples": int(X_raw.shape[0]), "n_features": int(X_raw.shape[1])}

    # ---- 1. Sanity check --------------------------------------------------
    duplicates = find_exact_duplicate(X_raw)
    n_duplicates = 0 if duplicates is None else len(duplicates)
    row["n_exact_duplicates"] = n_duplicates

    norm_info = evaluate_embedding_norms(X_raw)
    row["n_near_zero_norm"] = len(norm_info["near_zero_indices"])
    row["n_norm_outliers"] = len(norm_info["low_outlier_indices"]) + len(
        norm_info["high_outlier_indices"]
    )

    # Normalizzazione prima di PCA/Hopkins/Ward. La scelta influenza le
    # metriche basate su distanza, quindi va allineata al preprocessing
    # usato nella pipeline di clustering reale:
    #   - "standard": per-feature (media 0, var 1). Evita che feature con
    #     scala grande dominino le distanze.
    #   - "l2": per-campione, porta ogni vettore su una sfera unitaria.
    #     Coerente con la distanza coseno tipica degli embedding CNN.
    #   - "none": nessuna normalizzazione.
    normalization = getattr(args, "normalization", "standard")
    if normalization == "standard":
        X_std = StandardScaler().fit_transform(X_raw).astype(np.float64)
    elif normalization == "l2":
        X_std = normalize(X_raw.astype(np.float64), norm="l2", axis=1)
    else:  # "none"
        X_std = X_raw.astype(np.float64)
    row["normalization"] = normalization

    # ---- 2. Dimensionalita' effettiva --------------------------------------
    dim_info = effective_dimensionality(X_std)
    row.update({f"dim_{k}": v for k, v in dim_info.items() if k not in ("n_samples", "n_features")})

    # Sottocampionamento comune per le metriche O(n^2)/O(n log n) pesanti.
    X_eval = subsample(X_std, args.metric_sample_size, args.random_state)

    # ---- 3. Hopkins ---------------------------------------------------------
    hopkins_df = compute_hopkins_dataframe(
        X_eval,
        n_runs=args.hopkins_n_runs,
        random_state=args.random_state,
    )
    row["hopkins_mean"] = float(hopkins_df["hopkins_mean"].iloc[0])
    row["hopkins_std"] = float(hopkins_df["hopkins_std"].iloc[0])

    # ---- 4. Mutual Nearest Neighbor + hubness -------------------------------
    try:
        nn_df = evaluate_embedding_backbone(X_eval)
        for col in nn_df.columns:
            row[f"nn_{col.replace('@', '_at_').replace(' ', '_')}"] = float(nn_df[col].iloc[0])
    except Exception as error:
        print(f"  [WARN] metriche nearest-neighbor fallite: {error}")

    # ---- 5a. Curva Ward sullo spazio pieno (standardizzato) -----------------
    # Valore assoluto tipicamente basso in alta dimensione (maledizione
    # della dimensionalita'): utile solo per il confronto relativo grezzo.
    k_values = tuple(range(args.ward_k_min, args.ward_k_max + 1))
    ward_curve_full = ward_clustering_curve(
        X_eval,
        k_values=k_values,
        max_samples_for_metrics=None,  # X_eval e' gia' sottocampionato sopra
        random_state=args.random_state,
    )
    ward_summary_full = summarize_ward_curve(ward_curve_full)
    row.update({f"ward_{k}": v for k, v in ward_summary_full.items()})

    # ---- 5b. Curva Ward dopo PCA a dimensionalita' FISSA bassa --------------
    # E' la misura piu' rappresentativa: la pipeline reale clusterizza dopo
    # UMAP (2-10 dim), non sullo spazio grezzo. Una PCA a poche componenti
    # riproduce quella condizione restando deterministica. Una PCA a
    # "varianza 95%" invece, su embedding rumorosi, mantiene comunque
    # centinaia di componenti e non risolve la maledizione della
    # dimensionalita', quindi qui usiamo un numero di componenti fisso.
    #
    # Caso speciale: --pca-n-components 0 disattiva la PCA e fa girare
    # Ward/HDBSCAN direttamente sullo spazio (normalizzato) pieno. Serve come
    # confronto per mostrare l'effetto della riduzione dimensionale: in alta
    # dimensione la silhouette collassa (maledizione della dimensionalita'),
    # ed e' proprio la giustificazione dell'uso della PCA.
    if args.pca_n_components and args.pca_n_components > 0:
        n_pca = min(args.pca_n_components, X_eval.shape[1], X_eval.shape[0] - 1)
        pca = PCA(n_components=n_pca, svd_solver="full", random_state=args.random_state)
        X_pca = pca.fit_transform(X_eval).astype(np.float64)
        row["pca_n_components"] = int(n_pca)
        row["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
    else:
        # Niente PCA: si usa lo spazio pieno (gia' normalizzato).
        X_pca = X_eval
        row["pca_n_components"] = 0  # 0 = nessuna riduzione
        row["pca_explained_variance"] = 1.0

    ward_curve_pca = ward_clustering_curve(
        X_pca,
        k_values=k_values,
        max_samples_for_metrics=None,
        random_state=args.random_state,
    )
    ward_summary_pca = summarize_ward_curve(ward_curve_pca)
    row.update({f"wardpca_{k}": v for k, v in ward_summary_pca.items()})

    # ---- 5c. Sonda HDBSCAN deterministica (density-based) sullo spazio PCA --
    # Metodo complementare a Ward: non fissa il numero di cluster, li scopre
    # da solo, e puo' marcare punti come rumore. Piu' vicino nello spirito
    # a HDBSCAN della pipeline reale, ma resta deterministico (gira su PCA,
    # non su UMAP). Utile con dataset sbilanciati: puo' isolare glomeruli
    # rari invece di appiattire tutto in 2 gruppi come tende a fare Ward.
    try:
        hdbscan_df = hdbscan_clustering_metrics(
            X_pca,
            min_cluster_sizes=tuple(args.hdbscan_min_cluster_sizes),
            random_state=args.random_state,
        )
        hdbscan_summary = summarize_hdbscan_metrics(hdbscan_df)
        row.update(hdbscan_summary)
    except Exception as error:
        print(f"  [WARN] HDBSCAN fallito: {error}")

    return row, ward_curve_full, ward_curve_pca


def sort_by_metric(
    details_df: pd.DataFrame,
    sort_metric: str = "wardpca_best_silhouette",
    ascending_is_better: bool = False,
) -> pd.DataFrame:
    """
    Ordina le backbone in base a UNA metrica standard esplicita, senza
    costruire punteggi aggregati con pesi arbitrari.

    Scelta metodologica: non si combinano piu' metriche in un unico
    "final_score" (i pesi sarebbero arbitrari e non difendibili). Il CSV
    riporta tutte le metriche standard separatamente (silhouette,
    Davies-Bouldin, Calinski-Harabasz, Hopkins, DBCV, ...) e l'ordinamento
    e' solo una comodita' di lettura basata su una metrica dichiarata.
    La conclusione va tratta guardando l'insieme delle metriche, non un
    singolo punteggio sintetico.

    Default: ordina per silhouette di Ward dopo PCA (piu' alta = meglio).
    """

    if sort_metric not in details_df.columns:
        return details_df.reset_index(drop=True)

    ordered = details_df.sort_values(
        sort_metric, ascending=ascending_is_better, na_position="last"
    ).reset_index(drop=True)
    return ordered


def main() -> None:
    args = parse_args()

    embeddings_dir = args.embeddings_dir.resolve()
    output_dir = args.output_dir.resolve()
    ward_curves_dir = output_dir / "ward_curves"

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Directory embeddings non trovata: {embeddings_dir}")

    ward_curves_pca_dir = output_dir / "ward_curves_pca"
    output_dir.mkdir(parents=True, exist_ok=True)
    ward_curves_dir.mkdir(parents=True, exist_ok=True)
    ward_curves_pca_dir.mkdir(parents=True, exist_ok=True)

    embedding_files = sorted(embeddings_dir.glob(args.pattern))
    if len(embedding_files) == 0:
        raise FileNotFoundError(
            f"Nessun file trovato in {embeddings_dir} con pattern {args.pattern!r}."
        )

    all_rows = []

    for embedding_path in embedding_files:
        name = embedding_path.stem
        X_raw = load_embeddings(embedding_path)

        row, ward_curve_full, ward_curve_pca = evaluate_single_backbone(name, X_raw, args)
        all_rows.append(row)

        ward_curve_full.to_csv(ward_curves_dir / f"{name}.csv", index=False)
        ward_curve_pca.to_csv(ward_curves_pca_dir / f"{name}.csv", index=False)

    details_df = pd.DataFrame(all_rows)
    details_path = output_dir / "backbone_metrics_detail.csv"
    details_df.to_csv(details_path, index=False)

    # Ordinamento per una metrica standard dichiarata (silhouette Ward dopo
    # PCA), senza punteggi aggregati arbitrari. Il file resta lo stesso set
    # di metriche standard, solo riordinato per comodita' di lettura.
    ordered_df = sort_by_metric(details_df, sort_metric="wardpca_best_silhouette")
    ordered_path = output_dir / "backbone_metrics_sorted.csv"
    ordered_df.to_csv(ordered_path, index=False)

    print("\n" + "=" * 90)
    print("METRICHE STANDARD PER BACKBONE (ordinate per silhouette Ward dopo PCA)")
    print("Nessun punteggio aggregato: la conclusione va tratta dall'insieme delle metriche.")
    print("=" * 90)
    display_cols = [
        "backbone",
        "wardpca_best_silhouette", "wardpca_best_k_silhouette", "wardpca_best_davies_bouldin",
        "hdbscan_best_dbcv", "hdbscan_best_dbcv_n_clusters",
        "hdbscan_best_silhouette", "hdbscan_min_noise_fraction",
        "hopkins_mean", "pca_n_components",
    ]
    display_cols = [c for c in display_cols if c in ordered_df.columns]
    print(ordered_df[display_cols].to_string(index=False))
    print("\nLegenda metriche (tutte standard, citabili):")
    print("  wardpca_best_silhouette      Silhouette (Rousseeuw 1987), Ward dopo PCA. Piu' alto = meglio.")
    print("  wardpca_best_davies_bouldin  Davies-Bouldin (1979). Piu' basso = meglio.")
    print("  hdbscan_best_dbcv            DBCV (Moulavi 2014), validazione density-based. Piu' alto = meglio.")
    print("  hdbscan_best_silhouette      Silhouette dei cluster HDBSCAN (esclude il rumore).")
    print("  hdbscan_min_noise_fraction   Frazione minima di punti marcati rumore da HDBSCAN.")
    print("  hopkins_mean                 Statistica di Hopkins (clusterabilita'). >0.7 = struttura.")

    print(f"\nDettaglio completo:      {details_path}")
    print(f"Metriche ordinate:       {ordered_path}")
    print(f"Curve Ward spazio pieno: {ward_curves_dir}")
    print(f"Curve Ward dopo PCA:     {ward_curves_pca_dir}")


if __name__ == "__main__":
    main()