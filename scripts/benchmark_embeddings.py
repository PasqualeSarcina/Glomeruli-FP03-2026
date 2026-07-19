import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: umap-learn. Install it with: pip install umap-learn"
    ) from exc

try:
    import hdbscan
    from hdbscan.validity import validity_index as hdbscan_validity_index
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: hdbscan. Install it with: pip install hdbscan"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "glomeruli" / "embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "embedding_benchmark"


@dataclass(frozen=True)
class PreprocessResult:
    name: str
    X: np.ndarray
    metadata: dict


# -----------------------------------------------------------------------------
# CLI utilities
# -----------------------------------------------------------------------------


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_pca_variants(value: str) -> list[str]:
    variants = []
    for item in parse_csv_strings(value):
        lowered = item.lower()
        if lowered in {"none", "no", "raw"}:
            variants.append("none")
        elif lowered.startswith("pca"):
            variants.append(lowered)
        else:
            # Accept values such as 0.95 and convert them to pca0.95.
            float(item)
            variants.append(f"pca{item}")
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark glomeruli embeddings produced by different backbones. "
            "The script compares embeddings before fixing the final clustering "
            "pipeline by testing PCA/no-PCA, UMAP stability and HDBSCAN quality."
        )
    )

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory containing .npy embedding files and optional .csv image-path files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="Glob pattern used to select embedding files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where benchmark CSV files, labels and plots are saved.",
    )
    parser.add_argument(
        "--pca-variants",
        type=parse_pca_variants,
        default=parse_pca_variants("none,pca0.95"),
        help=(
            "Comma-separated preprocessing variants. Examples: none,pca0.95 "
            "or none,pca0.90,pca0.95,pca0.99."
        ),
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=parse_csv_ints,
        default=parse_csv_ints("15,30"),
        help="Comma-separated UMAP n_neighbors values.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=parse_csv_floats,
        default=parse_csv_floats("0.0,0.1"),
        help="Comma-separated UMAP min_dist values.",
    )
    parser.add_argument(
        "--umap-n-components",
        type=parse_csv_ints,
        default=parse_csv_ints("2,10"),
        help="Comma-separated UMAP n_components values.",
    )
    parser.add_argument(
        "--umap-metrics",
        type=parse_csv_strings,
        default=parse_csv_strings("euclidean,cosine"),
        help="Comma-separated UMAP metrics. Useful values: euclidean,cosine.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=parse_csv_ints,
        default=parse_csv_ints("10,30"),
        help="Comma-separated HDBSCAN min_cluster_size values.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=str,
        default="auto",
        help=(
            "Comma-separated HDBSCAN min_samples values. Use 'auto' to set "
            "min_samples = min_cluster_size."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=parse_csv_ints,
        default=parse_csv_ints("0,1,2,3,4"),
        help="Comma-separated random seeds for UMAP/HDBSCAN stability analysis.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum acceptable number of non-noise clusters.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=15,
        help="Maximum acceptable number of non-noise clusters.",
    )
    parser.add_argument(
        "--max-noise-ratio",
        type=float,
        default=0.50,
        help="Maximum acceptable fraction of HDBSCAN noise points.",
    )
    parser.add_argument(
        "--metric-sample-size",
        type=int,
        default=1500,
        help=(
            "Maximum number of samples used for trustworthiness and internal metrics. "
            "Use 0 to disable subsampling."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save UMAP scatter plots for the best configuration of each embedding file.",
    )
    parser.add_argument(
        "--save-all-labels",
        action="store_true",
        help="Save label CSV files for every valid run. By default only the best labels are saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already contains previous outputs.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Loading and preprocessing
# -----------------------------------------------------------------------------


def load_embeddings(path: Path) -> np.ndarray:
    X = np.load(path)
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"{path} must contain a 2D array, found shape={X.shape}.")

    if not np.isfinite(X).all():
        finite_mask = np.isfinite(X).all(axis=1)
        removed = int(np.sum(~finite_mask))
        if removed == X.shape[0]:
            raise ValueError(f"{path} contains no fully finite rows.")
        print(f"[WARN] {path.name}: removing {removed} rows with NaN/Inf values.")
        X = X[finite_mask]

    return X.astype(np.float32, copy=False)


def embedding_basic_stats(X: np.ndarray) -> dict:
    feature_std = np.std(X, axis=0)
    row_norm = np.linalg.norm(X, axis=1)

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_std_min": float(np.min(feature_std)),
        "feature_std_mean": float(np.mean(feature_std)),
        "feature_std_median": float(np.median(feature_std)),
        "feature_std_max": float(np.max(feature_std)),
        "near_constant_feature_ratio": float(np.mean(feature_std < 1e-8)),
        "row_norm_mean": float(np.mean(row_norm)),
        "row_norm_std": float(np.std(row_norm)),
    }


def preprocess_embeddings(
    X_raw: np.ndarray,
    pca_variants: Iterable[str],
) -> list[PreprocessResult]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    results: list[PreprocessResult] = []

    for variant in pca_variants:
        if variant == "none":
            results.append(
                PreprocessResult(
                    name="standardized",
                    X=X_scaled.astype(np.float32),
                    metadata={
                        "preprocess": "standardized",
                        "pca_requested": None,
                        "pca_n_components": None,
                        "pca_explained_variance_ratio_sum": None,
                    },
                )
            )
            continue

        if not variant.startswith("pca"):
            raise ValueError(f"Unsupported PCA variant: {variant}")

        pca_target = float(variant.replace("pca", ""))
        if not 0.0 < pca_target <= 1.0:
            raise ValueError(f"PCA target must be in (0, 1], got {pca_target}.")

        pca = PCA(n_components=pca_target, svd_solver="full", random_state=0)
        X_pca = pca.fit_transform(X_scaled)

        results.append(
            PreprocessResult(
                name=f"pca_{pca_target:g}",
                X=X_pca.astype(np.float32),
                metadata={
                    "preprocess": f"pca_{pca_target:g}",
                    "pca_requested": pca_target,
                    "pca_n_components": int(pca.n_components_),
                    "pca_explained_variance_ratio_sum": float(
                        np.sum(pca.explained_variance_ratio_)
                    ),
                },
            )
        )

    return results


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def count_clusters_and_noise(labels: np.ndarray) -> tuple[int, float]:
    labels = np.asarray(labels)
    cluster_ids = [label for label in np.unique(labels) if label != -1]
    n_clusters = len(cluster_ids)
    noise_ratio = float(np.mean(labels == -1))
    return n_clusters, noise_ratio


def cluster_size_summary(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    cluster_ids = [label for label in np.unique(labels) if label != -1]
    sizes = [int(np.sum(labels == cluster_id)) for cluster_id in cluster_ids]

    if len(sizes) == 0:
        return {
            "min_cluster_size_found": np.nan,
            "median_cluster_size_found": np.nan,
            "max_cluster_size_found": np.nan,
            "cluster_size_cv": np.nan,
        }

    sizes_array = np.asarray(sizes, dtype=float)
    return {
        "min_cluster_size_found": int(np.min(sizes_array)),
        "median_cluster_size_found": float(np.median(sizes_array)),
        "max_cluster_size_found": int(np.max(sizes_array)),
        "cluster_size_cv": float(np.std(sizes_array) / max(np.mean(sizes_array), 1e-12)),
    }


def sample_indices(n_samples: int, max_samples: int, seed: int = 0) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or n_samples <= max_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def safe_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> dict:
    """Compute internal clustering metrics on non-noise points only."""
    labels = np.asarray(labels)
    non_noise = labels != -1
    n_clusters, _ = count_clusters_and_noise(labels)

    output = {
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "dbcv": np.nan,
    }

    if n_clusters < 2 or int(np.sum(non_noise)) <= n_clusters:
        return output

    idx_non_noise = np.where(non_noise)[0]
    selected_local = sample_indices(len(idx_non_noise), max_samples=max_samples, seed=seed)
    selected = idx_non_noise[selected_local]

    X_eval = np.ascontiguousarray(X[selected], dtype=np.float64)
    labels_eval = labels[selected]

    if len(np.unique(labels_eval)) < 2 or X_eval.shape[0] <= len(np.unique(labels_eval)):
        return output

    try:
        output["silhouette"] = float(silhouette_score(X_eval, labels_eval, metric="euclidean"))
    except Exception:
        pass

    try:
        output["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels_eval))
    except Exception:
        pass

    try:
        output["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, labels_eval))
    except Exception:
        pass

    try:
        # DBCV can include noise labels. Use the same UMAP space used by HDBSCAN.
        # For speed, it is computed on the same selected subset.
        output["dbcv"] = float(hdbscan_validity_index(X_eval, labels_eval, metric="euclidean"))
    except Exception:
        pass

    return output


def nearest_neighbor_overlap(X_high: np.ndarray, X_low: np.ndarray, k: int = 15) -> float:
    """Mean overlap of kNN sets between the source space and the UMAP space."""
    n_samples = X_high.shape[0]
    if n_samples <= k + 1:
        return np.nan

    high_neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_high)
    low_neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_low)

    high_idx = high_neighbors.kneighbors(X_high, return_distance=False)[:, 1:]
    low_idx = low_neighbors.kneighbors(X_low, return_distance=False)[:, 1:]

    overlaps = []
    for a, b in zip(high_idx, low_idx):
        overlaps.append(len(set(a).intersection(set(b))) / k)

    return float(np.mean(overlaps))


def safe_umap_quality_metrics(
    X_source: np.ndarray,
    X_umap: np.ndarray,
    max_samples: int,
    seed: int,
    k: int = 15,
) -> dict:
    idx = sample_indices(X_source.shape[0], max_samples=max_samples, seed=seed)
    X_source_eval = X_source[idx]
    X_umap_eval = X_umap[idx]

    output = {
        "trustworthiness_k15": np.nan,
        "knn_overlap_k15": np.nan,
    }

    if X_source_eval.shape[0] <= k + 1:
        return output

    try:
        output["trustworthiness_k15"] = float(
            trustworthiness(
                X_source_eval,
                X_umap_eval,
                n_neighbors=k,
                metric="euclidean",
            )
        )
    except Exception:
        pass

    try:
        output["knn_overlap_k15"] = nearest_neighbor_overlap(
            X_source_eval,
            X_umap_eval,
            k=k,
        )
    except Exception:
        pass

    return output


def validity_reason(
    n_clusters: int,
    noise_ratio: float,
    dbcv: float,
    min_clusters: int,
    max_clusters: int,
    max_noise_ratio: float,
) -> tuple[bool, str]:
    if n_clusters < min_clusters:
        return False, "too_few_clusters"
    if n_clusters > max_clusters:
        return False, "too_many_clusters"
    if noise_ratio > max_noise_ratio:
        return False, "too_much_noise"
    if not np.isfinite(dbcv):
        return False, "dbcv_not_available"
    return True, "ok"


# -----------------------------------------------------------------------------
# Benchmark core
# -----------------------------------------------------------------------------


def resolve_min_samples_values(raw_values: str, min_cluster_size: int) -> list[int]:
    values = []
    for item in parse_csv_strings(raw_values):
        if item.lower() == "auto":
            values.append(int(min_cluster_size))
        else:
            values.append(int(item))
    return sorted(set(values))


def run_single_configuration(
    X_source: np.ndarray,
    preprocess_name: str,
    embedding_name: str,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_n_components: int,
    umap_metric: str,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    seed: int,
    metric_sample_size: int,
    constraints: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    reducer = umap.UMAP(
        n_neighbors=int(umap_n_neighbors),
        min_dist=float(umap_min_dist),
        n_components=int(umap_n_components),
        metric=umap_metric,
        random_state=int(seed),
    )
    X_umap = reducer.fit_transform(X_source)
    X_umap = np.ascontiguousarray(X_umap, dtype=np.float64)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(hdbscan_min_cluster_size),
        min_samples=int(hdbscan_min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(X_umap)

    n_clusters, noise_ratio = count_clusters_and_noise(labels)
    sizes = cluster_size_summary(labels)
    internal = safe_internal_metrics(
        X=X_umap,
        labels=labels,
        max_samples=metric_sample_size,
        seed=seed,
    )
    umap_quality = safe_umap_quality_metrics(
        X_source=X_source,
        X_umap=X_umap,
        max_samples=metric_sample_size,
        seed=seed,
        k=15,
    )

    valid, reason = validity_reason(
        n_clusters=n_clusters,
        noise_ratio=noise_ratio,
        dbcv=internal["dbcv"],
        min_clusters=constraints["min_clusters"],
        max_clusters=constraints["max_clusters"],
        max_noise_ratio=constraints["max_noise_ratio"],
    )

    row = {
        "embedding_name": embedding_name,
        "preprocess": preprocess_name,
        "seed": int(seed),
        "umap_n_neighbors": int(umap_n_neighbors),
        "umap_min_dist": float(umap_min_dist),
        "umap_n_components": int(umap_n_components),
        "umap_metric": umap_metric,
        "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
        "hdbscan_min_samples": int(hdbscan_min_samples),
        "n_clusters": int(n_clusters),
        "noise_ratio": float(noise_ratio),
        "valid": bool(valid),
        "validity_reason": reason,
    }
    row.update(sizes)
    row.update(internal)
    row.update(umap_quality)

    return row, labels, X_umap


def config_columns() -> list[str]:
    return [
        "embedding_name",
        "preprocess",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_n_components",
        "umap_metric",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
    ]


def summarize_configurations(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()

    group_cols = config_columns()
    rows = []

    metric_cols = [
        "valid",
        "dbcv",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "noise_ratio",
        "n_clusters",
        "min_cluster_size_found",
        "median_cluster_size_found",
        "max_cluster_size_found",
        "cluster_size_cv",
        "trustworthiness_k15",
        "knn_overlap_k15",
    ]

    for key, group in details.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_runs"] = int(len(group))
        row["valid_runs"] = int(group["valid"].sum())
        row["valid_run_ratio"] = float(group["valid"].mean())

        for col in metric_cols:
            if col == "valid":
                continue
            row[f"mean_{col}"] = float(group[col].mean(skipna=True))
            row[f"std_{col}"] = float(group[col].std(skipna=True))

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = add_stability_to_summary(details, summary)
    summary = add_ranking_score(summary)
    return summary.sort_values("final_score", ascending=False).reset_index(drop=True)


def add_stability_to_summary(details: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    # The labels themselves are not inside details. They are joined later through a
    # compact key in the caller when possible. This fallback keeps the CSV usable
    # even if only aggregate metrics are available.
    if "mean_pairwise_ari" not in summary.columns:
        summary["mean_pairwise_ari"] = np.nan
    return summary


def compute_pairwise_ari(label_records: list[dict]) -> dict:
    valid_records = [record for record in label_records if record.get("valid")]
    if len(valid_records) < 2:
        return {
            "mean_pairwise_ari": np.nan,
            "std_pairwise_ari": np.nan,
            "n_pairwise_ari": 0,
            "medoid_seed": valid_records[0]["seed"] if len(valid_records) == 1 else None,
            "medoid_ari": np.nan,
        }

    pairwise = []
    seed_to_scores: dict[int, list[float]] = {int(record["seed"]): [] for record in valid_records}

    for a, b in itertools.combinations(valid_records, 2):
        labels_a = np.asarray(a["labels"])
        labels_b = np.asarray(b["labels"])
        common = (labels_a != -1) & (labels_b != -1)
        if int(np.sum(common)) < 2:
            continue
        value = float(adjusted_rand_score(labels_a[common], labels_b[common]))
        pairwise.append(value)
        seed_to_scores[int(a["seed"])].append(value)
        seed_to_scores[int(b["seed"])].append(value)

    if len(pairwise) == 0:
        return {
            "mean_pairwise_ari": np.nan,
            "std_pairwise_ari": np.nan,
            "n_pairwise_ari": 0,
            "medoid_seed": None,
            "medoid_ari": np.nan,
        }

    medoid_seed = None
    medoid_ari = -np.inf
    for seed, values in seed_to_scores.items():
        if len(values) == 0:
            continue
        score = float(np.mean(values))
        if score > medoid_ari:
            medoid_seed = seed
            medoid_ari = score

    return {
        "mean_pairwise_ari": float(np.mean(pairwise)),
        "std_pairwise_ari": float(np.std(pairwise)),
        "n_pairwise_ari": int(len(pairwise)),
        "medoid_seed": medoid_seed,
        "medoid_ari": float(medoid_ari) if np.isfinite(medoid_ari) else np.nan,
    }


def add_ranking_score(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    ranked = summary.copy()

    # Metrics are intentionally simple and interpretable:
    # - high valid_run_ratio: configuration respects cluster/noise constraints often;
    # - high mean_dbcv: density-based cluster quality for HDBSCAN;
    # - low mean_noise_ratio: useful because a high DBCV with too much noise is not useful;
    # - high stability: avoids selecting one lucky UMAP seed;
    # - high trustworthiness: UMAP preserves local neighbors.
    score_terms = []

    def percentile_rank(column: str, ascending: bool) -> pd.Series:
        values = ranked[column]
        if values.notna().sum() == 0:
            return pd.Series(0.0, index=ranked.index)
        return values.rank(pct=True, ascending=ascending).fillna(0.0)

    score_terms.append(0.30 * percentile_rank("valid_run_ratio", ascending=True))
    score_terms.append(0.25 * percentile_rank("mean_dbcv", ascending=True))
    score_terms.append(0.15 * percentile_rank("mean_noise_ratio", ascending=False))
    score_terms.append(0.15 * percentile_rank("mean_pairwise_ari", ascending=True))
    score_terms.append(0.10 * percentile_rank("mean_trustworthiness_k15", ascending=True))
    score_terms.append(0.05 * percentile_rank("mean_silhouette", ascending=True))

    ranked["final_score"] = sum(score_terms)
    return ranked


# -----------------------------------------------------------------------------
# Saving outputs
# -----------------------------------------------------------------------------


def find_sidecar_csv(embedding_path: Path) -> Path | None:
    candidate = embedding_path.with_suffix(".csv")
    if candidate.exists():
        return candidate
    return None


def load_image_table(embedding_path: Path, n_rows: int) -> pd.DataFrame:
    csv_path = find_sidecar_csv(embedding_path)
    if csv_path is None:
        return pd.DataFrame({"index": np.arange(n_rows)})

    table = pd.read_csv(csv_path)
    if "index" not in table.columns:
        table.insert(0, "index", np.arange(len(table)))
    if len(table) != n_rows:
        print(
            f"[WARN] {embedding_path.name}: sidecar CSV has {len(table)} rows, "
            f"but embeddings have {n_rows}. Using positional truncation/alignment."
        )
        table = table.iloc[:n_rows].copy()
        table["index"] = np.arange(len(table))
    return table


def save_labels(
    output_dir: Path,
    embedding_path: Path,
    image_table: pd.DataFrame,
    preprocess_name: str,
    config: dict,
    seed: int,
    labels: np.ndarray,
    X_umap: np.ndarray,
    suffix: str,
) -> Path:
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = sanitize_name(embedding_path.stem)
    filename = f"{stem}__{sanitize_name(preprocess_name)}__seed{seed}__{suffix}.csv"
    out_path = labels_dir / filename

    table = image_table.copy().reset_index(drop=True)
    table["cluster_label"] = labels
    table["is_noise"] = labels == -1

    # Save first two UMAP coordinates for easy plotting/inspection.
    for dim in range(min(2, X_umap.shape[1])):
        table[f"umap_{dim + 1}"] = X_umap[:, dim]

    for key, value in config.items():
        table[key] = value

    table.to_csv(out_path, index=False)
    return out_path


def save_plot(
    output_dir: Path,
    embedding_path: Path,
    preprocess_name: str,
    seed: int,
    labels: np.ndarray,
    X_umap: np.ndarray,
) -> Path | None:
    if plt is None or X_umap.shape[1] < 2:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=8, alpha=0.8)
    ax.set_title(f"{embedding_path.stem} | {preprocess_name} | seed={seed}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.colorbar(scatter, ax=ax, label="cluster label")
    fig.tight_layout()

    out_path = plots_dir / f"{sanitize_name(embedding_path.stem)}__{sanitize_name(preprocess_name)}__seed{seed}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def sanitize_name(value: str) -> str:
    allowed = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    embeddings_dir = args.embeddings_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_dir.glob("*.csv"):
            path.unlink()

    embedding_files = sorted(embeddings_dir.glob(args.pattern))
    if len(embedding_files) == 0:
        raise FileNotFoundError(
            f"No embedding files found in {embeddings_dir} with pattern {args.pattern!r}."
        )

    constraints = {
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "max_noise_ratio": args.max_noise_ratio,
    }

    all_detail_rows = []
    all_config_summary_rows = []
    all_embedding_stats = []

    for embedding_path in embedding_files:
        embedding_name = embedding_path.stem
        print(f"\n=== Benchmarking {embedding_path.name} ===")

        X_raw = load_embeddings(embedding_path)
        image_table = load_image_table(embedding_path, n_rows=X_raw.shape[0])

        stats = {
            "embedding_name": embedding_name,
            "embedding_file": str(embedding_path),
        }
        stats.update(embedding_basic_stats(X_raw))
        all_embedding_stats.append(stats)

        preprocess_results = preprocess_embeddings(X_raw, args.pca_variants)

        # Store only label arrays for stability. The best run is recomputed at the end
        # to avoid keeping every UMAP embedding in memory.
        label_records_by_config: dict[tuple, list[dict]] = {}

        for prep in preprocess_results:
            X = prep.X
            print(f"  - Variant: {prep.name}, shape={X.shape}")

            for (
                umap_n_neighbors,
                umap_min_dist,
                umap_n_components,
                umap_metric,
                hdbscan_min_cluster_size,
            ) in itertools.product(
                args.umap_n_neighbors,
                args.umap_min_dist,
                args.umap_n_components,
                args.umap_metrics,
                args.hdbscan_min_cluster_size,
            ):
                if umap_n_neighbors >= X.shape[0]:
                    continue
                if umap_n_components >= X.shape[0]:
                    continue

                for hdbscan_min_samples in resolve_min_samples_values(
                    args.hdbscan_min_samples,
                    min_cluster_size=hdbscan_min_cluster_size,
                ):
                    for seed in args.seeds:
                        config_key = (
                            embedding_name,
                            prep.name,
                            int(umap_n_neighbors),
                            float(umap_min_dist),
                            int(umap_n_components),
                            umap_metric,
                            int(hdbscan_min_cluster_size),
                            int(hdbscan_min_samples),
                        )

                        try:
                            row, labels, X_umap = run_single_configuration(
                                X_source=X,
                                preprocess_name=prep.name,
                                embedding_name=embedding_name,
                                umap_n_neighbors=umap_n_neighbors,
                                umap_min_dist=umap_min_dist,
                                umap_n_components=umap_n_components,
                                umap_metric=umap_metric,
                                hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                                hdbscan_min_samples=hdbscan_min_samples,
                                seed=seed,
                                metric_sample_size=args.metric_sample_size,
                                constraints=constraints,
                            )
                            row.update(prep.metadata)
                            row["error"] = None

                            label_records_by_config.setdefault(config_key, []).append(
                                {
                                    "seed": int(seed),
                                    "valid": bool(row["valid"]),
                                    "labels": labels,
                                }
                            )
                            if args.save_all_labels and row["valid"]:
                                save_labels(
                                    output_dir=output_dir,
                                    embedding_path=embedding_path,
                                    image_table=image_table,
                                    preprocess_name=prep.name,
                                    config={
                                        "umap_n_neighbors": umap_n_neighbors,
                                        "umap_min_dist": umap_min_dist,
                                        "umap_n_components": umap_n_components,
                                        "umap_metric": umap_metric,
                                        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
                                        "hdbscan_min_samples": hdbscan_min_samples,
                                    },
                                    seed=seed,
                                    labels=labels,
                                    X_umap=X_umap,
                                    suffix="labels",
                                )

                        except Exception as error:
                            row = {
                                "embedding_name": embedding_name,
                                "preprocess": prep.name,
                                "seed": int(seed),
                                "umap_n_neighbors": int(umap_n_neighbors),
                                "umap_min_dist": float(umap_min_dist),
                                "umap_n_components": int(umap_n_components),
                                "umap_metric": umap_metric,
                                "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
                                "hdbscan_min_samples": int(hdbscan_min_samples),
                                "valid": False,
                                "validity_reason": "error",
                                "error": str(error),
                            }
                            row.update(prep.metadata)

                        all_detail_rows.append(row)

        detail_df_embedding = pd.DataFrame(
            [row for row in all_detail_rows if row["embedding_name"] == embedding_name]
        )
        summary_embedding = summarize_configurations(detail_df_embedding)

        # Add actual stability metrics now that labels are available.
        if not summary_embedding.empty:
            stability_rows = []
            for _, summary_row in summary_embedding.iterrows():
                config_key = tuple(summary_row[col] for col in config_columns())
                # Convert numpy scalar formatting to the exact tuple type used above.
                config_key = (
                    str(config_key[0]),
                    str(config_key[1]),
                    int(config_key[2]),
                    float(config_key[3]),
                    int(config_key[4]),
                    str(config_key[5]),
                    int(config_key[6]),
                    int(config_key[7]),
                )
                stability_rows.append(compute_pairwise_ari(label_records_by_config.get(config_key, [])))

            stability_df = pd.DataFrame(stability_rows)
            for col in stability_df.columns:
                summary_embedding[col] = stability_df[col].values
            summary_embedding = add_ranking_score(summary_embedding)
            summary_embedding = summary_embedding.sort_values("final_score", ascending=False).reset_index(drop=True)

            all_config_summary_rows.append(summary_embedding)

            # Save labels and plot for the best valid configuration.
            best_valid = summary_embedding[summary_embedding["valid_runs"] > 0]
            if len(best_valid) > 0:
                best = best_valid.iloc[0]
                medoid_seed = best.get("medoid_seed")
                if pd.isna(medoid_seed) or medoid_seed is None:
                    medoid_seed = int(args.seeds[0])
                medoid_seed = int(medoid_seed)

                best_key = (
                    str(best["embedding_name"]),
                    str(best["preprocess"]),
                    int(best["umap_n_neighbors"]),
                    float(best["umap_min_dist"]),
                    int(best["umap_n_components"]),
                    str(best["umap_metric"]),
                    int(best["hdbscan_min_cluster_size"]),
                    int(best["hdbscan_min_samples"]),
                    medoid_seed,
                )

                # Recompute only the best medoid run to save labels and optional plot.
                prep_lookup = {prep.name: prep for prep in preprocess_results}
                best_prep_name = str(best["preprocess"])
                if best_prep_name in prep_lookup:
                    best_prep = prep_lookup[best_prep_name]
                    best_row, labels, X_umap = run_single_configuration(
                        X_source=best_prep.X,
                        preprocess_name=best_prep.name,
                        embedding_name=embedding_name,
                        umap_n_neighbors=int(best["umap_n_neighbors"]),
                        umap_min_dist=float(best["umap_min_dist"]),
                        umap_n_components=int(best["umap_n_components"]),
                        umap_metric=str(best["umap_metric"]),
                        hdbscan_min_cluster_size=int(best["hdbscan_min_cluster_size"]),
                        hdbscan_min_samples=int(best["hdbscan_min_samples"]),
                        seed=medoid_seed,
                        metric_sample_size=args.metric_sample_size,
                        constraints=constraints,
                    )
                    config_payload = {
                        "umap_n_neighbors": int(best["umap_n_neighbors"]),
                        "umap_min_dist": float(best["umap_min_dist"]),
                        "umap_n_components": int(best["umap_n_components"]),
                        "umap_metric": str(best["umap_metric"]),
                        "hdbscan_min_cluster_size": int(best["hdbscan_min_cluster_size"]),
                        "hdbscan_min_samples": int(best["hdbscan_min_samples"]),
                        "final_score": float(best["final_score"]),
                        "valid_run_ratio": float(best["valid_run_ratio"]),
                        "mean_dbcv": float(best["mean_dbcv"]),
                        "mean_noise_ratio": float(best["mean_noise_ratio"]),
                        "mean_pairwise_ari": (
                            None
                            if pd.isna(best["mean_pairwise_ari"])
                            else float(best["mean_pairwise_ari"])
                        ),
                    }
                    save_labels(
                        output_dir=output_dir,
                        embedding_path=embedding_path,
                        image_table=image_table,
                        preprocess_name=best_prep_name,
                        config=config_payload,
                        seed=medoid_seed,
                        labels=labels,
                        X_umap=X_umap,
                        suffix="best_labels",
                    )
                    if args.plot:
                        save_plot(
                            output_dir=output_dir,
                            embedding_path=embedding_path,
                            preprocess_name=best_prep_name,
                            seed=medoid_seed,
                            labels=labels,
                            X_umap=X_umap,
                        )

    details_df = pd.DataFrame(all_detail_rows)
    config_summary_df = (
        pd.concat(all_config_summary_rows, ignore_index=True)
        if len(all_config_summary_rows) > 0
        else pd.DataFrame()
    )
    embedding_stats_df = pd.DataFrame(all_embedding_stats)

    details_path = output_dir / "run_details.csv"
    config_summary_path = output_dir / "config_summary.csv"
    embedding_stats_path = output_dir / "embedding_stats.csv"
    best_by_embedding_path = output_dir / "best_by_embedding.csv"

    details_df.to_csv(details_path, index=False)
    config_summary_df.to_csv(config_summary_path, index=False)
    embedding_stats_df.to_csv(embedding_stats_path, index=False)

    if not config_summary_df.empty:
        best_by_embedding = (
            config_summary_df.sort_values("final_score", ascending=False)
            .groupby(["embedding_name", "preprocess"], as_index=False)
            .head(1)
            .sort_values("final_score", ascending=False)
        )
        best_by_embedding.to_csv(best_by_embedding_path, index=False)

    write_json(
        output_dir / "benchmark_config.json",
        {
            "embeddings_dir": str(embeddings_dir),
            "pattern": args.pattern,
            "pca_variants": args.pca_variants,
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_n_components": args.umap_n_components,
            "umap_metrics": args.umap_metrics,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "seeds": args.seeds,
            "constraints": constraints,
            "metric_sample_size": args.metric_sample_size,
        },
    )

    print("\nDone.")
    print(f"Run details:       {details_path}")
    print(f"Config summary:    {config_summary_path}")
    print(f"Embedding stats:   {embedding_stats_path}")
    if not config_summary_df.empty:
        print(f"Best by embedding: {best_by_embedding_path}")
        print(f"Best labels dir:   {output_dir / 'labels'}")


if __name__ == "__main__":
    main()
