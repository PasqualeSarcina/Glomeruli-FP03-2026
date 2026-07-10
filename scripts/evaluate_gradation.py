"""
Valuta quanto gli embedding di ciascuna backbone catturano una GRADAZIONE
CONTINUA della gravita' morfologica dei glomeruli, invece della separazione
in cluster discreti.

Complementare a evaluate_backbones.py: quello misura la separabilita' in
cluster (silhouette, DBCV), questo misura l'allineamento a un gradiente
morfologico continuo. Utile quando la variabile di interesse (necrotizzazione)
e' un continuum piuttosto che categorie nette.

Passi:
  1. estrae un proxy morfologico continuo della gravita' dai crop dei
     glomeruli (usando le maschere), tramite descrittori H&E + texture,
     ridotti a un asse con PCA;
  2. per ogni file di embedding .npy, calcola le metriche di gradazione
     (correlazione distanza-morfologia, consistenza del vicinato, smoothness
     del gradiente / indice di Moran);
  3. salva una tabella comparativa.

Uso:
    python scripts/evaluate_gradation.py \\
        --embeddings-dir data/glomeruli/embeddings \\
        --crops-dir data/glomeruli/crops \\
        --masks-dir data/glomeruli/masks
"""

from __future__ import annotations

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

from src.metrics.morphology_descriptors import build_morphology_matrix, morphology_severity_axis
from src.metrics.continuous_gradation import evaluate_gradation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valuta la gradazione continua (non il clustering) degli embedding."
    )
    parser.add_argument("--embeddings-dir", type=Path, default=PROJECT_ROOT / "data" / "glomeruli" / "embeddings")
    parser.add_argument("--crops-dir", type=Path, default=PROJECT_ROOT / "data" / "glomeruli" / "crops")
    parser.add_argument("--masks-dir", type=Path, default=PROJECT_ROOT / "data" / "glomeruli" / "masks")
    parser.add_argument("--pattern", type=str, default="*.npy")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "gradation_evaluation")
    parser.add_argument("--normalization", choices=["standard", "l2", "none"], default="l2")
    parser.add_argument("--pca-n-components", type=int, default=10, help="0 = niente PCA.")
    parser.add_argument("--k", type=int, default=15, help="Numero di vicini per le metriche locali.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def preprocess(X: np.ndarray, normalization: str, pca_n: int, random_state: int) -> np.ndarray:
    if normalization == "standard":
        X = StandardScaler().fit_transform(X)
    elif normalization == "l2":
        X = normalize(X, norm="l2", axis=1)
    X = X.astype(np.float64)
    if pca_n and pca_n > 0:
        n = min(pca_n, X.shape[1], X.shape[0] - 1)
        X = PCA(n_components=n, random_state=random_state).fit_transform(X)
    return X


def main() -> None:
    args = parse_args()

    crops_dir = args.crops_dir.resolve()
    masks_dir = args.masks_dir.resolve()
    embeddings_dir = args.embeddings_dir.resolve()

    if not crops_dir.is_dir():
        raise NotADirectoryError(f"Cartella crops non trovata: {crops_dir}")

    image_paths = sorted(p for p in crops_dir.iterdir() if p.suffix.lower() == ".png")
    if not image_paths:
        raise FileNotFoundError(f"Nessun crop .png in {crops_dir}")

    # maschere allineate per nome (crop X.png -> maschera X_mask.png se esiste)
    mask_paths = None
    if masks_dir.is_dir():
        mask_lookup = {p.stem.replace("_mask", ""): p for p in masks_dir.iterdir() if p.suffix.lower() == ".png"}
        mask_paths = [mask_lookup.get(img.stem) for img in image_paths]
        if any(m is None for m in mask_paths):
            n_missing = sum(m is None for m in mask_paths)
            print(f"  [WARN] {n_missing} maschere mancanti: per quei crop uso l'intera patch.")

    print(f"Estrazione descrittori morfologici da {len(image_paths)} crop...")
    morph_matrix, feature_names = build_morphology_matrix(image_paths, mask_paths)
    print(f"  descrittori: {feature_names}")

    morphology_score = morphology_severity_axis(morph_matrix, random_state=args.random_state)
    n_glomeruli = len(morphology_score)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # salvo il proxy morfologico per ispezione/riuso
    pd.DataFrame({"crop": [p.name for p in image_paths], "morphology_score": morphology_score}).to_csv(
        output_dir / "morphology_scores.csv", index=False
    )

    embedding_files = sorted(embeddings_dir.glob(args.pattern))
    if not embedding_files:
        raise FileNotFoundError(f"Nessun embedding in {embeddings_dir} con pattern {args.pattern!r}")

    rows = []
    for emb_path in embedding_files:
        X = np.load(emb_path)
        if X.ndim != 2:
            print(f"  [SKIP] {emb_path.name}: shape inattesa {X.shape}")
            continue
        if X.shape[0] != n_glomeruli:
            print(f"  [SKIP] {emb_path.name}: {X.shape[0]} righe != {n_glomeruli} crop. Disallineato.")
            continue

        X_proc = preprocess(X.astype(np.float64), args.normalization, args.pca_n_components, args.random_state)
        metrics = evaluate_gradation(X_proc, morphology_score, k=args.k, random_state=args.random_state)
        metrics["backbone"] = emb_path.stem
        metrics["n_samples"] = int(X.shape[0])
        rows.append(metrics)
        print(f"  {emb_path.stem}: dist-corr={metrics['grad_distance_morph_corr']:.3f}  "
              f"neigh-consist={metrics['grad_neighborhood_consistency']:.3f}  "
              f"moran={metrics['grad_gradient_smoothness_moran']:.3f}")

    results = pd.DataFrame(rows)
    cols = ["backbone", "n_samples", "grad_distance_morph_corr",
            "grad_neighborhood_consistency", "grad_gradient_smoothness_moran"]
    results = results[[c for c in cols if c in results.columns]]
    results = results.sort_values("grad_distance_morph_corr", ascending=False).reset_index(drop=True)

    out_path = output_dir / "gradation_metrics.csv"
    results.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("GRADAZIONE CONTINUA — ordinata per correlazione distanza-morfologia")
    print("=" * 80)
    print(results.to_string(index=False))
    print("\nLegenda (tutte piu' alto = meglio, rispettano la gradazione morfologica):")
    print("  grad_distance_morph_corr        Spearman(distanza embedding, differenza morfologica)")
    print("  grad_neighborhood_consistency   quanto i vicini nell'embedding sono simili morfologicamente")
    print("  grad_gradient_smoothness_moran  indice di Moran: gradiente morfologico liscio nello spazio")
    print(f"\nProxy morfologico salvato in: {output_dir / 'morphology_scores.csv'}")
    print(f"Risultati:                    {out_path}")


if __name__ == "__main__":
    main()
