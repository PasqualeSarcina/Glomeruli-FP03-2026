import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PATH_COLUMNS = [
    "image_path",
    "path",
    "filepath",
    "file_path",
    "crop_path",
    "img_path",
    "filename",
    "file",
]


@dataclass(frozen=True)
class QuerySelection:
    query_indices: list[int]
    strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create query + nearest-neighbor image grids for any glomeruli "
            "embedding file. Useful for qualitative evaluation of backbone spaces."
        )
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to a .npy file containing embeddings with shape (n_images, n_features).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help=(
            "CSV containing image paths. If omitted, the script tries to use a sidecar "
            "CSV with the same name as the .npy file. It can also be a best_labels CSV."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional root folder used to resolve relative image paths or filenames. "
            "Example: data/glomeruli/crops."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve relative paths. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where nearest-neighbor grids and CSV summaries are saved.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for nearest-neighbor search. Default: cosine.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "standardize", "pca0.95", "pca0.99"],
        default="none",
        help=(
            "Optional preprocessing applied before neighbor search. Usually keep 'none' "
            "if the .npy file is already the final embedding space."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of nearest neighbors to show for each query, excluding the query itself.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=20,
        help="Number of query images to inspect, unless per-cluster selection overrides it.",
    )
    parser.add_argument(
        "--query-strategy",
        choices=["random", "central", "per_cluster", "from_indices"],
        default="random",
        help=(
            "How to choose query images. 'central' selects points near the global embedding "
            "centroid. 'per_cluster' needs a cluster_label column in metadata-csv."
        ),
    )
    parser.add_argument(
        "--queries-per-cluster",
        type=int,
        default=5,
        help="Number of query images per cluster when --query-strategy per_cluster is used.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include cluster_label=-1 when using --query-strategy per_cluster.",
    )
    parser.add_argument(
        "--query-indices",
        type=str,
        default="",
        help="Comma-separated query indices used when --query-strategy from_indices is selected.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for query sampling.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=160,
        help="Square size in pixels for each image tile.",
    )
    parser.add_argument(
        "--label-height",
        type=int,
        default=24,
        help="Height in pixels reserved for tile labels.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=6,
        help="Padding in pixels between tiles.",
    )
    parser.add_argument(
        "--max-missing-images",
        type=int,
        default=20,
        help="Maximum number of missing image warnings printed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. Existing files are otherwise left untouched.",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    X = np.load(path)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, found shape={X.shape}.")
    if not np.isfinite(X).all():
        finite_mask = np.isfinite(X).all(axis=1)
        removed = int(np.sum(~finite_mask))
        if removed > 0:
            raise ValueError(
                f"Embedding contains {removed} rows with NaN/Inf. Clean it before using this script."
            )
    return X.astype(np.float32, copy=False)


def resolve_metadata_csv(embeddings_path: Path, metadata_csv: Path | None) -> Path:
    if metadata_csv is not None:
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
        return metadata_csv
    sidecar = embeddings_path.with_suffix(".csv")
    if sidecar.exists():
        return sidecar
    raise FileNotFoundError(
        "No metadata CSV provided and no sidecar CSV found. Pass --metadata-csv explicitly."
    )


def find_path_column(table: pd.DataFrame) -> str | None:
    lower_to_original = {col.lower(): col for col in table.columns}
    for candidate in PATH_COLUMNS:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    # Fallback: first column that looks path-like.
    for col in table.columns:
        values = table[col].dropna().astype(str).head(20).tolist()
        if any(Path(value).suffix.lower() in IMAGE_EXTENSIONS for value in values):
            return col
    return None


def build_image_index(image_root: Path | None, project_root: Path) -> dict[str, Path]:
    if image_root is None:
        return {}
    root = resolve_path(image_root, project_root)
    if not root.exists():
        print(f"[WARN] image_root does not exist: {root}")
        return {}
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name] = path
            index[path.stem] = path
    return index


def resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def resolve_image_paths(
    table: pd.DataFrame,
    n_rows: int,
    project_root: Path,
    image_root: Path | None,
    max_missing_warnings: int,
) -> list[Path | None]:
    if len(table) != n_rows:
        print(
            f"[WARN] Metadata rows ({len(table)}) != embedding rows ({n_rows}). "
            "Using positional alignment after truncation."
        )
        table = table.iloc[:n_rows].copy()

    path_col = find_path_column(table)
    if path_col is None:
        raise ValueError(
            "Could not find an image path column in metadata CSV. Expected one of: "
            + ", ".join(PATH_COLUMNS)
        )

    image_index = build_image_index(image_root, project_root)
    resolved: list[Path | None] = []
    missing_count = 0

    for raw_value in table[path_col].astype(str).tolist():
        raw_path = Path(raw_value)
        candidates: list[Path] = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append((project_root / raw_path).resolve())
            if image_root is not None:
                candidates.append((resolve_path(image_root, project_root) / raw_path).resolve())
                candidates.append((resolve_path(image_root, project_root) / raw_path.name).resolve())

        indexed = image_index.get(raw_path.name) or image_index.get(raw_path.stem)
        if indexed is not None:
            candidates.append(indexed)

        found = next((candidate for candidate in candidates if candidate.exists()), None)
        if found is None:
            missing_count += 1
            if missing_count <= max_missing_warnings:
                print(f"[WARN] Missing image for metadata value: {raw_value}")
        resolved.append(found)

    if missing_count > max_missing_warnings:
        print(f"[WARN] ...and {missing_count - max_missing_warnings} more missing images.")
    print(f"Resolved {n_rows - missing_count}/{n_rows} image paths.")
    return resolved


def preprocess_embeddings(X: np.ndarray, preprocess: str) -> tuple[np.ndarray, dict]:
    metadata: dict = {"preprocess": preprocess}
    if preprocess == "none":
        return X.astype(np.float32, copy=False), metadata
    if preprocess == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(X).astype(np.float32), metadata
    if preprocess.startswith("pca"):
        target = float(preprocess.replace("pca", ""))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=target, svd_solver="full", random_state=0)
        X_pca = pca.fit_transform(X_scaled).astype(np.float32)
        metadata["pca_n_components"] = int(pca.n_components_)
        metadata["pca_explained_variance"] = float(np.sum(pca.explained_variance_ratio_))
        return X_pca, metadata
    raise ValueError(f"Unsupported preprocess value: {preprocess}")


def fit_neighbors(X: np.ndarray, metric: str, k: int) -> tuple[NearestNeighbors, np.ndarray]:
    if metric == "cosine":
        X_search = normalize(X, norm="l2", axis=1)
        nn_metric = "cosine"
    else:
        X_search = X
        nn_metric = "euclidean"
    n_neighbors = min(k + 1, X_search.shape[0])
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=nn_metric)
    model.fit(X_search)
    return model, X_search


def select_queries(
    X: np.ndarray,
    table: pd.DataFrame,
    strategy: str,
    n_queries: int,
    seed: int,
    query_indices: str,
    queries_per_cluster: int,
    include_noise: bool,
) -> QuerySelection:
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    if strategy == "from_indices":
        indices = [int(item.strip()) for item in query_indices.split(",") if item.strip()]
        indices = [idx for idx in indices if 0 <= idx < n]
        if not indices:
            raise ValueError("No valid indices provided with --query-indices.")
        return QuerySelection(indices, strategy)

    valid_indices = np.arange(n)

    if strategy == "random":
        count = min(n_queries, n)
        return QuerySelection(sorted(rng.choice(valid_indices, size=count, replace=False).tolist()), strategy)

    if strategy == "central":
        X_eval = normalize(X, norm="l2", axis=1) if X.shape[0] > 0 else X
        centroid = np.mean(X_eval, axis=0, keepdims=True)
        distances = np.linalg.norm(X_eval - centroid, axis=1)
        indices = np.argsort(distances)[: min(n_queries, n)].tolist()
        return QuerySelection(indices, strategy)

    if strategy == "per_cluster":
        if "cluster_label" not in table.columns:
            raise ValueError("--query-strategy per_cluster requires a cluster_label column in metadata-csv.")
        indices: list[int] = []
        labels = table["cluster_label"].to_numpy()
        cluster_ids = sorted(set(labels.tolist()))
        if not include_noise:
            cluster_ids = [label for label in cluster_ids if int(label) != -1]

        for cluster_id in cluster_ids:
            cluster_idx = np.where(labels == cluster_id)[0]
            if len(cluster_idx) == 0:
                continue
            # Use points nearest to the cluster centroid to get representative queries.
            X_cluster = X[cluster_idx]
            centroid = np.mean(X_cluster, axis=0, keepdims=True)
            distances = np.linalg.norm(X_cluster - centroid, axis=1)
            selected_local = np.argsort(distances)[: min(queries_per_cluster, len(cluster_idx))]
            indices.extend(cluster_idx[selected_local].tolist())
        return QuerySelection(indices, strategy)

    raise ValueError(f"Unsupported query strategy: {strategy}")


def load_tile(path: Path | None, tile_size: int) -> Image.Image:
    if path is None or not path.exists():
        image = Image.new("RGB", (tile_size, tile_size), color=(235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.text((10, tile_size // 2 - 8), "missing", fill=(80, 80, 80))
        return image
    try:
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (tile_size, tile_size), method=Image.Resampling.LANCZOS)
        return image
    except Exception:
        image = Image.new("RGB", (tile_size, tile_size), color=(235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.text((10, tile_size // 2 - 8), "error", fill=(120, 0, 0))
        return image


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def make_neighbor_grid(
    query_index: int,
    neighbor_indices: Iterable[int],
    distances: Iterable[float],
    image_paths: list[Path | None],
    table: pd.DataFrame,
    tile_size: int,
    label_height: int,
    padding: int,
    metric: str,
) -> Image.Image:
    neighbor_indices = list(neighbor_indices)
    distances = list(distances)
    all_indices = [query_index] + neighbor_indices
    all_distances = [0.0] + distances
    n_tiles = len(all_indices)

    width = n_tiles * tile_size + (n_tiles + 1) * padding
    height = tile_size + label_height + 2 * padding
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for pos, (idx, dist) in enumerate(zip(all_indices, all_distances)):
        x = padding + pos * (tile_size + padding)
        y = padding + label_height
        tile = load_tile(image_paths[idx], tile_size)
        canvas.paste(tile, (x, y))

        prefix = "Q" if pos == 0 else f"NN{pos}"
        cluster_text = ""
        if "cluster_label" in table.columns:
            cluster_text = f" c={table.iloc[idx]['cluster_label']}"
        label = f"{prefix} i={idx}{cluster_text} d={dist:.3f}"
        draw_label(draw, (x, padding), label)

        if pos == 0:
            # Highlight query tile with a dark border.
            draw.rectangle([x, y, x + tile_size - 1, y + tile_size - 1], outline=(0, 0, 0), width=3)

    return canvas


def save_query_grid(
    output_dir: Path,
    embedding_stem: str,
    query_index: int,
    grid: Image.Image,
    overwrite: bool,
) -> Path:
    grids_dir = output_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    out_path = grids_dir / f"{embedding_stem}__query_{query_index:04d}.png"
    if out_path.exists() and not overwrite:
        return out_path
    grid.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = resolve_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = resolve_path(args.embeddings, project_root)
    metadata_csv = resolve_metadata_csv(embeddings_path, args.metadata_csv)

    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata:   {metadata_csv}")
    print(f"Output:     {output_dir}")

    X_raw = load_embeddings(embeddings_path)
    table = pd.read_csv(metadata_csv)
    if len(table) != X_raw.shape[0]:
        table = table.iloc[: X_raw.shape[0]].copy().reset_index(drop=True)
    else:
        table = table.reset_index(drop=True)

    image_paths = resolve_image_paths(
        table=table,
        n_rows=X_raw.shape[0],
        project_root=project_root,
        image_root=args.image_root,
        max_missing_warnings=args.max_missing_images,
    )

    X, prep_metadata = preprocess_embeddings(X_raw, args.preprocess)
    nn_model, X_search = fit_neighbors(X, metric=args.metric, k=args.k)

    selection = select_queries(
        X=X_search,
        table=table,
        strategy=args.query_strategy,
        n_queries=args.n_queries,
        seed=args.seed,
        query_indices=args.query_indices,
        queries_per_cluster=args.queries_per_cluster,
        include_noise=args.include_noise,
    )
    print(f"Selected {len(selection.query_indices)} query images using strategy={selection.strategy}.")

    distances_all, indices_all = nn_model.kneighbors(X_search[selection.query_indices], return_distance=True)

    rows = []
    for query_pos, query_index in enumerate(selection.query_indices):
        indices = indices_all[query_pos].tolist()
        distances = distances_all[query_pos].tolist()

        # Remove the query itself if present, then keep k neighbors.
        neighbors = [(idx, dist) for idx, dist in zip(indices, distances) if idx != query_index]
        neighbors = neighbors[: args.k]
        neighbor_indices = [idx for idx, _ in neighbors]
        neighbor_distances = [dist for _, dist in neighbors]

        grid = make_neighbor_grid(
            query_index=query_index,
            neighbor_indices=neighbor_indices,
            distances=neighbor_distances,
            image_paths=image_paths,
            table=table,
            tile_size=args.tile_size,
            label_height=args.label_height,
            padding=args.padding,
            metric=args.metric,
        )
        grid_path = save_query_grid(
            output_dir=output_dir,
            embedding_stem=embeddings_path.stem,
            query_index=query_index,
            grid=grid,
            overwrite=args.overwrite,
        )

        query_row = table.iloc[query_index].to_dict()
        query_cluster = query_row.get("cluster_label", np.nan)
        for rank, (neighbor_index, distance) in enumerate(neighbors, start=1):
            neighbor_row = table.iloc[neighbor_index].to_dict()
            rows.append(
                {
                    "embedding_name": embeddings_path.stem,
                    "metric": args.metric,
                    "preprocess": args.preprocess,
                    "query_strategy": selection.strategy,
                    "query_index": int(query_index),
                    "query_cluster_label": query_cluster,
                    "query_image_path": str(image_paths[query_index]) if image_paths[query_index] else None,
                    "neighbor_rank": int(rank),
                    "neighbor_index": int(neighbor_index),
                    "neighbor_cluster_label": neighbor_row.get("cluster_label", np.nan),
                    "neighbor_image_path": str(image_paths[neighbor_index]) if image_paths[neighbor_index] else None,
                    "distance": float(distance),
                    "grid_path": str(grid_path),
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"{embeddings_path.stem}__nearest_neighbors_summary.csv"
    summary.to_csv(summary_path, index=False)

    config = {
        "embeddings": str(embeddings_path),
        "metadata_csv": str(metadata_csv),
        "image_root": str(args.image_root) if args.image_root else None,
        "metric": args.metric,
        "preprocess": args.preprocess,
        "preprocess_metadata": prep_metadata,
        "k": args.k,
        "query_strategy": args.query_strategy,
        "n_queries": args.n_queries,
        "queries_per_cluster": args.queries_per_cluster,
        "include_noise": args.include_noise,
        "seed": args.seed,
    }
    config_path = output_dir / f"{embeddings_path.stem}__nearest_neighbors_config.json"
    pd.Series(config, dtype="object").to_json(config_path, indent=2)

    print("Done.")
    print(f"Grids:   {output_dir / 'grids'}")
    print(f"Summary: {summary_path}")
    print(f"Config:  {config_path}")


if __name__ == "__main__":
    main()
