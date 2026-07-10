import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostics for unsupervised glomeruli clusters without ground truth."
    )
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--path-column",
        default="auto",
        help="Column containing image paths. Use 'auto' to infer it.",
    )
    parser.add_argument("--cluster-column", default="cluster_label")
    parser.add_argument("--index-column", default="index")

    parser.add_argument(
        "--slide-column",
        default=None,
        help="Existing column for slide ID. If omitted, slide IDs are inferred from paths.",
    )
    parser.add_argument(
        "--patient-column",
        default=None,
        help="Existing column for patient ID, if available.",
    )
    parser.add_argument(
        "--batch-column",
        default=None,
        help="Existing column for stain/batch/site ID, if available.",
    )
    parser.add_argument(
        "--slide-regex",
        default=None,
        help=(
            "Optional regex used to extract slide_id from image path. "
            "Use a named group (?P<slide>...) or the first group."
        ),
    )
    parser.add_argument(
        "--patient-regex",
        default=None,
        help=(
            "Optional regex used to extract patient_id from image path. "
            "Use a named group (?P<patient>...) or the first group."
        ),
    )

    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Optional directory containing masks with matching filenames.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Metric for nearest-neighbor consistency.",
    )
    parser.add_argument("--k-values", type=parse_csv_ints, default=parse_csv_ints("5,10,20"))
    parser.add_argument("--max-quality-images", type=int, default=0, help="0 = process all images.")

    parser.add_argument(
        "--compare-labels-csv",
        type=Path,
        default=None,
        help="Optional labels CSV from another backbone for cross-backbone agreement.",
    )
    parser.add_argument("--compare-name", default="comparison")

    parser.add_argument(
        "--stability-labels-glob",
        default=None,
        help=(
            "Optional glob over multiple label CSVs from different seeds for the same embedding/config. "
            "Requires labels saved for multiple seeds."
        ),
    )

    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for file in path.glob("*.csv"):
            file.unlink()
        for file in path.glob("*.png"):
            file.unlink()
        for file in path.glob("*.json"):
            file.unlink()


def infer_path_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "image_path",
        "path",
        "filepath",
        "file_path",
        "crop_path",
        "filename",
        "file",
        "img_path",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: choose the first object column that appears to contain image filenames.
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20).tolist()
            if any(Path(x).suffix.lower() in IMAGE_EXTS for x in sample):
                return col
    return None


def resolve_image_path(raw_value: object, image_root: Path) -> Path | None:
    if pd.isna(raw_value):
        return None
    raw = str(raw_value)
    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(image_root / p)
    candidates.append(image_root / p.name)

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else None


def extract_with_regex(text: str, pattern: str | None, named_group: str) -> str | None:
    if not pattern:
        return None
    match = re.search(pattern, text)
    if not match:
        return None
    if named_group in match.groupdict():
        return str(match.group(named_group))
    if match.groups():
        return str(match.group(1))
    return str(match.group(0))


def infer_slide_id(path_value: object, slide_regex: str | None) -> str:
    if pd.isna(path_value):
        return "unknown"
    text = str(path_value)
    from_regex = extract_with_regex(text, slide_regex, "slide")
    if from_regex:
        return from_regex

    p = Path(text)
    parts = list(p.parts)
    # Prefer parent folder if the path has folders and parent is not generic.
    if len(parts) >= 2:
        parent = p.parent.name
        if parent.lower() not in {"crops", "images", "img", "data", "glomeruli"}:
            return parent

    stem = p.stem
    # Conservative filename heuristic: take the prefix before common glomerulus/crop tokens.
    for token in ["_glomer", "_crop", "_patch", "-glomer", "-crop", "-patch"]:
        if token in stem.lower():
            idx = stem.lower().find(token)
            return stem[:idx]
    # If nothing better exists, use the first two underscore-separated fields.
    bits = stem.split("_")
    if len(bits) >= 2:
        return "_".join(bits[:2])
    return stem


def infer_patient_id(path_value: object, patient_regex: str | None) -> str:
    if pd.isna(path_value):
        return "unknown"
    text = str(path_value)
    from_regex = extract_with_regex(text, patient_regex, "patient")
    if from_regex:
        return from_regex
    # Without metadata, patient ID cannot be known reliably.
    return "unknown"


def normalized_entropy(counts: Iterable[int]) -> float:
    arr = np.asarray(list(counts), dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.nan
    probs = arr[arr > 0] / total
    if len(probs) <= 1:
        return 0.0
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(len(arr)))


def plot_heatmap(table: pd.DataFrame, title: str, out_path: Path) -> None:
    if plt is None or table.empty:
        return
    fig_w = max(7, min(22, 0.55 * len(table.columns) + 3))
    fig_h = max(4, min(18, 0.5 * len(table.index) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    values = table.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(table.columns.name or "column")
    ax.set_ylabel(table.index.name or "row")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(table.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Loading and metadata enrichment
# -----------------------------------------------------------------------------


def load_labels(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(args.labels_csv)
    if args.cluster_column not in df.columns:
        raise ValueError(f"Missing cluster column: {args.cluster_column}")

    path_col = args.path_column
    if path_col == "auto":
        path_col = infer_path_column(df)
    if path_col is None or path_col not in df.columns:
        raise ValueError(
            "Could not infer image path column. Pass --path-column explicitly. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[args.cluster_column] = df[args.cluster_column].astype(int)
    df["resolved_image_path"] = [
        str(resolve_image_path(x, args.image_root)) for x in df[path_col]
    ]

    if args.slide_column and args.slide_column in df.columns:
        df["slide_id"] = df[args.slide_column].astype(str)
    else:
        df["slide_id"] = [infer_slide_id(x, args.slide_regex) for x in df[path_col]]

    if args.patient_column and args.patient_column in df.columns:
        df["patient_id"] = df[args.patient_column].astype(str)
    else:
        df["patient_id"] = [infer_patient_id(x, args.patient_regex) for x in df[path_col]]

    if args.batch_column and args.batch_column in df.columns:
        df["batch_id"] = df[args.batch_column].astype(str)
    else:
        df["batch_id"] = "unknown"

    return df, path_col


# -----------------------------------------------------------------------------
# 1. Cluster vs slide/patient/batch
# -----------------------------------------------------------------------------


def composition_diagnostics(
    df: pd.DataFrame,
    cluster_col: str,
    group_col: str,
    output_dir: Path,
    write_plots: bool,
) -> None:
    if group_col not in df.columns:
        return
    sub = df[[cluster_col, group_col]].copy()
    sub[group_col] = sub[group_col].fillna("unknown").astype(str)

    counts = pd.crosstab(sub[cluster_col], sub[group_col])
    counts.index.name = "cluster_label"
    counts.columns.name = group_col
    row_percent = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    col_percent = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1) * 100.0

    counts.to_csv(output_dir / f"cluster_by_{group_col}_counts.csv")
    row_percent.to_csv(output_dir / f"cluster_by_{group_col}_row_percent.csv")
    col_percent.to_csv(output_dir / f"cluster_by_{group_col}_column_percent.csv")

    rows = []
    for cluster, row in counts.iterrows():
        total = int(row.sum())
        top_group = str(row.idxmax()) if total > 0 else "unknown"
        top_count = int(row.max()) if total > 0 else 0
        rows.append(
            {
                "cluster_label": cluster,
                "n_samples": total,
                f"top_{group_col}": top_group,
                f"top_{group_col}_count": top_count,
                f"top_{group_col}_share": top_count / total if total else np.nan,
                f"{group_col}_normalized_entropy": normalized_entropy(row.values),
                f"n_unique_{group_col}": int((row > 0).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / f"cluster_{group_col}_purity_entropy.csv", index=False)

    if write_plots:
        # Use percentages for readability.
        plot_heatmap(
            row_percent,
            f"Cluster × {group_col} row percent",
            output_dir / f"heatmap_cluster_by_{group_col}_row_percent.png",
        )


# -----------------------------------------------------------------------------
# 2. Image quality / crop diagnostics
# -----------------------------------------------------------------------------


def find_mask_for_image(image_path, mask_root):
    if mask_root is None:
        return None

    image_path = Path(image_path)
    mask_root = Path(mask_root)

    candidates = [
        mask_root / image_path.name,
        mask_root / f"{image_path.stem}_mask{image_path.suffix}",
        mask_root / f"{image_path.stem}.png",
        mask_root / f"{image_path.stem}_mask.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def image_quality_features(image_path_raw: str, mask_root: Path | None) -> dict:
    p = Path(image_path_raw)
    out = {
        "image_exists": p.exists(),
        "width": np.nan,
        "height": np.nan,
        "aspect_ratio": np.nan,
        "brightness_mean": np.nan,
        "contrast_std": np.nan,
        "r_mean": np.nan,
        "g_mean": np.nan,
        "b_mean": np.nan,
        "white_ratio": np.nan,
        "dark_ratio": np.nan,
        "foreground_ratio_mask": np.nan,
        "mask_area_px": np.nan,
        "mask_touches_border": np.nan,
    }
    if not p.exists():
        return out

    try:
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        h, w = arr.shape[:2]
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        out.update(
            {
                "width": int(w),
                "height": int(h),
                "aspect_ratio": float(w / max(h, 1)),
                "brightness_mean": float(gray.mean()),
                "contrast_std": float(gray.std()),
                "r_mean": float(arr[:, :, 0].mean()),
                "g_mean": float(arr[:, :, 1].mean()),
                "b_mean": float(arr[:, :, 2].mean()),
                "white_ratio": float(np.mean(np.all(arr > 240, axis=2))),
                "dark_ratio": float(np.mean(np.all(arr < 20, axis=2))),
            }
        )

        mask_path = find_mask_for_image(p, mask_root)
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L").resize((w, h))
            m = np.asarray(mask) > 0
            out["foreground_ratio_mask"] = float(m.mean())
            out["mask_area_px"] = int(m.sum())
            out["mask_touches_border"] = bool(
                m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any()
            )
    except Exception:
        pass
    return out


def quality_diagnostics(
    df: pd.DataFrame,
    cluster_col: str,
    mask_root: Path | None,
    max_images: int,
    output_dir: Path,
    write_plots: bool,
) -> pd.DataFrame:
    work = df.copy()
    if max_images and max_images > 0 and len(work) > max_images:
        work = work.sample(max_images, random_state=0).sort_index()

    features = []
    for path in work["resolved_image_path"]:
        features.append(image_quality_features(path, mask_root))
    feat_df = pd.DataFrame(features, index=work.index)
    out_df = pd.concat([work.reset_index(drop=False), feat_df.reset_index(drop=True)], axis=1)
    out_df.to_csv(output_dir / "image_quality_features.csv", index=False)

    numeric_cols = [
        "width",
        "height",
        "aspect_ratio",
        "brightness_mean",
        "contrast_std",
        "r_mean",
        "g_mean",
        "b_mean",
        "white_ratio",
        "dark_ratio",
        "foreground_ratio_mask",
        "mask_area_px",
    ]
    available = [c for c in numeric_cols if c in out_df.columns and out_df[c].notna().any()]
    summary = out_df.groupby(cluster_col)[available].agg(["count", "mean", "std", "median", "min", "max"])
    summary.to_csv(output_dir / "image_quality_by_cluster.csv")

    if "mask_touches_border" in out_df.columns and out_df["mask_touches_border"].notna().any():
        border_summary = out_df.groupby(cluster_col)["mask_touches_border"].mean().reset_index()
        border_summary.rename(columns={"mask_touches_border": "mask_touches_border_ratio"}, inplace=True)
        border_summary.to_csv(output_dir / "mask_border_touch_by_cluster.csv", index=False)

    if write_plots and plt is not None:
        for col in ["brightness_mean", "contrast_std", "white_ratio", "foreground_ratio_mask"]:
            if col not in out_df.columns or out_df[col].notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            groups = []
            labels = []
            for cluster, group in out_df.groupby(cluster_col):
                values = group[col].dropna().to_numpy()
                if len(values):
                    groups.append(values)
                    labels.append(str(cluster))
            if groups:
                ax.boxplot(groups, labels=labels, showfliers=False)
                ax.set_title(f"{col} by cluster")
                ax.set_xlabel("cluster_label")
                ax.set_ylabel(col)
                fig.tight_layout()
                fig.savefig(output_dir / f"boxplot_{col}_by_cluster.png", dpi=180)
                plt.close(fig)

    return out_df


# -----------------------------------------------------------------------------
# 3. Nearest-neighbor same-cluster consistency
# -----------------------------------------------------------------------------


def nearest_neighbor_consistency(
    embeddings_path: Path,
    df: pd.DataFrame,
    cluster_col: str,
    metric: str,
    k_values: list[int],
    output_dir: Path,
) -> None:
    X = np.load(embeddings_path)
    X = np.asarray(X)
    if X.shape[0] != len(df):
        raise ValueError(f"Embeddings rows ({X.shape[0]}) != labels rows ({len(df)}).")

    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric=metric)
    nn.fit(X)
    neighbors = nn.kneighbors(X, return_distance=False)[:, 1:]

    labels = df[cluster_col].to_numpy()
    rows = []
    per_sample = df[[cluster_col, "slide_id", "patient_id", "batch_id"]].copy()
    for k in k_values:
        same = np.array([np.mean(labels[neighbors[i, :k]] == labels[i]) for i in range(len(labels))])
        same_non_noise = np.array(
            [
                np.mean(labels[neighbors[i, :k]][labels[neighbors[i, :k]] != -1] == labels[i])
                if np.any(labels[neighbors[i, :k]] != -1)
                else np.nan
                for i in range(len(labels))
            ]
        )
        per_sample[f"same_cluster_ratio@{k}"] = same
        per_sample[f"same_cluster_ratio_non_noise_neighbors@{k}"] = same_non_noise

        for cluster, idx in df.groupby(cluster_col).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            rows.append(
                {
                    "cluster_label": cluster,
                    "k": k,
                    "n_samples": int(len(idx)),
                    "mean_same_cluster_ratio": float(np.nanmean(same[idx])),
                    "median_same_cluster_ratio": float(np.nanmedian(same[idx])),
                    "std_same_cluster_ratio": float(np.nanstd(same[idx])),
                }
            )
        rows.append(
            {
                "cluster_label": "ALL",
                "k": k,
                "n_samples": int(len(labels)),
                "mean_same_cluster_ratio": float(np.nanmean(same)),
                "median_same_cluster_ratio": float(np.nanmedian(same)),
                "std_same_cluster_ratio": float(np.nanstd(same)),
            }
        )

    per_sample.to_csv(output_dir / "nearest_neighbor_same_cluster_per_sample.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "nearest_neighbor_same_cluster_summary.csv", index=False)


# -----------------------------------------------------------------------------
# 4. Core vs borderline samples
# -----------------------------------------------------------------------------


def core_borderline_diagnostics(df: pd.DataFrame, cluster_col: str, output_dir: Path) -> None:
    if not {"umap_1", "umap_2"}.issubset(df.columns):
        return
    work = df.copy()
    work["cluster_role"] = "noise"
    work["distance_to_cluster_centroid"] = np.nan
    work["cluster_distance_percentile"] = np.nan

    for cluster, group in work[work[cluster_col] != -1].groupby(cluster_col):
        idx = group.index
        coords = group[["umap_1", "umap_2"]].to_numpy(dtype=float)
        centroid = coords.mean(axis=0)
        dist = np.linalg.norm(coords - centroid, axis=1)
        if len(dist) == 1:
            pct = np.array([0.0])
        else:
            pct = pd.Series(dist).rank(pct=True).to_numpy()
        roles = np.full(len(dist), "middle", dtype=object)
        roles[pct <= 0.25] = "core"
        roles[pct >= 0.75] = "borderline"
        work.loc[idx, "distance_to_cluster_centroid"] = dist
        work.loc[idx, "cluster_distance_percentile"] = pct
        work.loc[idx, "cluster_role"] = roles

    work.to_csv(output_dir / "core_borderline_samples.csv", index=False)
    summary = pd.crosstab(work[cluster_col], work["cluster_role"])
    summary.to_csv(output_dir / "core_borderline_summary.csv")


# -----------------------------------------------------------------------------
# 5. Cross-backbone agreement
# -----------------------------------------------------------------------------


def align_two_label_tables(
    main_df: pd.DataFrame,
    other_df: pd.DataFrame,
    main_path_col: str,
    other_path_col: str | None,
    index_col: str,
) -> pd.DataFrame:
    left = main_df.copy()
    right = other_df.copy()

    if index_col in left.columns and index_col in right.columns:
        return left[[index_col, "cluster_label"]].merge(
            right[[index_col, "cluster_label"]], on=index_col, suffixes=("_main", "_other")
        )

    if other_path_col is not None and main_path_col in left.columns and other_path_col in right.columns:
        left["_basename"] = left[main_path_col].astype(str).map(lambda x: Path(x).name)
        right["_basename"] = right[other_path_col].astype(str).map(lambda x: Path(x).name)
        return left[["_basename", "cluster_label"]].merge(
            right[["_basename", "cluster_label"]], on="_basename", suffixes=("_main", "_other")
        )

    n = min(len(left), len(right))
    return pd.DataFrame(
        {
            "position": np.arange(n),
            "cluster_label_main": left["cluster_label"].iloc[:n].to_numpy(),
            "cluster_label_other": right["cluster_label"].iloc[:n].to_numpy(),
        }
    )


def cross_backbone_agreement(
    main_df: pd.DataFrame,
    other_csv: Path,
    main_path_col: str,
    index_col: str,
    output_dir: Path,
    compare_name: str,
    write_plots: bool,
) -> None:
    other = pd.read_csv(other_csv)
    if "cluster_label" not in other.columns:
        raise ValueError(f"Comparison file has no cluster_label column: {other_csv}")
    other_path_col = infer_path_column(other)

    aligned = align_two_label_tables(main_df, other, main_path_col, other_path_col, index_col)
    if "cluster_label_main" not in aligned.columns:
        # Occurs when merge suffixes are not applied because selected names changed.
        label_cols = [c for c in aligned.columns if "cluster_label" in c]
        if len(label_cols) >= 2:
            aligned = aligned.rename(columns={label_cols[0]: "cluster_label_main", label_cols[1]: "cluster_label_other"})
    aligned.to_csv(output_dir / f"cross_backbone_aligned_{compare_name}.csv", index=False)

    ct = pd.crosstab(aligned["cluster_label_main"], aligned["cluster_label_other"])
    ct.index.name = "main_cluster_label"
    ct.columns.name = f"{compare_name}_cluster_label"
    ct.to_csv(output_dir / f"cross_backbone_{compare_name}_counts.csv")
    row_pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    row_pct.to_csv(output_dir / f"cross_backbone_{compare_name}_row_percent.csv")

    main_labels = aligned["cluster_label_main"].to_numpy()
    other_labels = aligned["cluster_label_other"].to_numpy()
    non_noise = (main_labels != -1) & (other_labels != -1)
    metrics = {
        "n_aligned": int(len(aligned)),
        "ari_all": float(adjusted_rand_score(main_labels, other_labels)),
        "nmi_all": float(normalized_mutual_info_score(main_labels, other_labels)),
        "ari_non_noise_both": float(adjusted_rand_score(main_labels[non_noise], other_labels[non_noise]))
        if int(non_noise.sum()) >= 2
        else np.nan,
        "nmi_non_noise_both": float(normalized_mutual_info_score(main_labels[non_noise], other_labels[non_noise]))
        if int(non_noise.sum()) >= 2
        else np.nan,
        "n_non_noise_both": int(non_noise.sum()),
    }
    with open(output_dir / f"cross_backbone_{compare_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if write_plots:
        plot_heatmap(
            row_pct,
            f"Main clusters × {compare_name} clusters row percent",
            output_dir / f"heatmap_cross_backbone_{compare_name}_row_percent.png",
        )


# -----------------------------------------------------------------------------
# 6. Stability across labels from multiple seeds
# -----------------------------------------------------------------------------


def map_labels_to_reference(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_clusters = [c for c in np.unique(reference) if c != -1]
    tgt_clusters = [c for c in np.unique(target) if c != -1]
    mapped = np.full_like(target, fill_value=-1)
    if not ref_clusters or not tgt_clusters:
        return mapped

    # Greedy maximum-overlap mapping. It is enough for diagnostic use and avoids extra deps.
    overlaps = []
    for r in ref_clusters:
        for t in tgt_clusters:
            overlaps.append((int(np.sum((reference == r) & (target == t))), r, t))
    overlaps.sort(reverse=True)
    used_r, used_t = set(), set()
    mapping = {}
    for overlap, r, t in overlaps:
        if r in used_r or t in used_t:
            continue
        mapping[t] = r
        used_r.add(r)
        used_t.add(t)

    for t, r in mapping.items():
        mapped[target == t] = r
    return mapped


def stability_diagnostics(labels_glob: str, output_dir: Path) -> None:
    paths = sorted(Path().glob(labels_glob)) if not any(ch in labels_glob for ch in ["/", "\\"]) else sorted(Path(p) for p in __import__("glob").glob(labels_glob))
    if len(paths) < 2:
        return

    label_arrays = []
    names = []
    for p in paths:
        df = pd.read_csv(p)
        if "cluster_label" not in df.columns:
            continue
        label_arrays.append(df["cluster_label"].to_numpy(dtype=int))
        names.append(p.stem)
    if len(label_arrays) < 2:
        return

    n = min(len(x) for x in label_arrays)
    label_arrays = [x[:n] for x in label_arrays]
    ref = label_arrays[0]
    mapped = [ref] + [map_labels_to_reference(ref, arr) for arr in label_arrays[1:]]

    pair_rows = []
    for i in range(len(label_arrays)):
        for j in range(i + 1, len(label_arrays)):
            pair_rows.append(
                {
                    "run_a": names[i],
                    "run_b": names[j],
                    "ari": float(adjusted_rand_score(label_arrays[i], label_arrays[j])),
                    "nmi": float(normalized_mutual_info_score(label_arrays[i], label_arrays[j])),
                }
            )
    pd.DataFrame(pair_rows).to_csv(output_dir / "stability_pairwise_ari_nmi.csv", index=False)

    mapped_arr = np.vstack(mapped).T
    per_sample_stability = np.mean(mapped_arr == mapped_arr[:, [0]], axis=1)
    per_sample = pd.DataFrame(
        {
            "index": np.arange(n),
            "reference_cluster_label": ref[:n],
            "sample_stability_vs_reference": per_sample_stability,
        }
    )
    per_sample.to_csv(output_dir / "stability_per_sample.csv", index=False)

    cluster_summary = per_sample.groupby("reference_cluster_label")["sample_stability_vs_reference"].agg(
        ["count", "mean", "std", "median", "min", "max"]
    )
    cluster_summary.to_csv(output_dir / "stability_by_reference_cluster.csv")


# -----------------------------------------------------------------------------
# Noise summary
# -----------------------------------------------------------------------------


def noise_summary(df: pd.DataFrame, cluster_col: str, output_dir: Path) -> None:
    rows = []
    total = len(df)
    for label, group in df.groupby(cluster_col):
        rows.append(
            {
                "cluster_label": label,
                "n_samples": int(len(group)),
                "share_total": float(len(group) / total) if total else np.nan,
                "is_noise": bool(label == -1),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "cluster_size_and_noise_summary.csv", index=False)

    noise = df[df[cluster_col] == -1]
    if len(noise) > 0:
        for col in ["slide_id", "patient_id", "batch_id"]:
            counts = noise[col].value_counts(dropna=False).reset_index()
            counts.columns = [col, "noise_count"]
            counts["noise_share"] = counts["noise_count"] / len(noise)
            counts.to_csv(output_dir / f"noise_by_{col}.csv", index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, args.overwrite)

    df, path_col = load_labels(args)
    df.to_csv(args.output_dir / "labels_with_inferred_metadata.csv", index=False)

    # 1. Cluster vs slide/patient/batch.
    for group_col in ["slide_id", "patient_id", "batch_id"]:
        composition_diagnostics(
            df=df,
            cluster_col=args.cluster_column,
            group_col=group_col,
            output_dir=args.output_dir,
            write_plots=args.write_plots,
        )

    # 2. Quality/crop checks.
    quality_diagnostics(
        df=df,
        cluster_col=args.cluster_column,
        mask_root=args.mask_root,
        max_images=args.max_quality_images,
        output_dir=args.output_dir,
        write_plots=args.write_plots,
    )

    # 3. Quantitative nearest-neighbor consistency.
    nearest_neighbor_consistency(
        embeddings_path=args.embeddings,
        df=df,
        cluster_col=args.cluster_column,
        metric=args.metric,
        k_values=args.k_values,
        output_dir=args.output_dir,
    )

    # 5. Core vs borderline based on UMAP centroid distance.
    core_borderline_diagnostics(df=df, cluster_col=args.cluster_column, output_dir=args.output_dir)

    # 8. Cross-backbone agreement.
    if args.compare_labels_csv is not None:
        cross_backbone_agreement(
            main_df=df,
            other_csv=args.compare_labels_csv,
            main_path_col=path_col,
            index_col=args.index_column,
            output_dir=args.output_dir,
            compare_name=args.compare_name,
            write_plots=args.write_plots,
        )

    # 4. Optional stability, only if all seed labels were saved.
    if args.stability_labels_glob:
        stability_diagnostics(args.stability_labels_glob, args.output_dir)

    # 9. Noise diagnostics.
    noise_summary(df=df, cluster_col=args.cluster_column, output_dir=args.output_dir)

    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with open(args.output_dir / "cluster_diagnostics_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Done.")
    print(f"Outputs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
