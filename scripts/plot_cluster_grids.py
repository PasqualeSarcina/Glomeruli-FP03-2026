import argparse
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save one contact-sheet grid per cluster from a labels CSV."
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="CSV produced by benchmark_embeddings.py, e.g. *_best_labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where grids are saved. Default: results/cluster_grids/<labels_csv_stem>.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root used to resolve relative image paths like data/glomeruli/crops/xxx.png. "
            "Default: current working directory."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing the crop images. If set, unresolved paths are matched "
            "by filename inside this directory."
        ),
    )
    parser.add_argument(
        "--image-col",
        type=str,
        default=None,
        help="Image path column. If omitted, the script tries image_path, path, filepath, file, filename.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="cluster_label",
        help="Cluster label column. Default: cluster_label.",
    )
    parser.add_argument(
        "--max-per-cluster",
        type=int,
        default=36,
        help="Maximum number of images shown for each cluster/noise grid. Default: 36.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Number of columns in each grid. Default: 6.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=160,
        help="Square size of each image tile in pixels. Default: 160.",
    )
    parser.add_argument(
        "--selection",
        choices=["random", "centroid", "first"],
        default="centroid",
        help=(
            "How to choose images when a cluster has more samples than max-per-cluster. "
            "centroid uses umap_1/umap_2 if present, otherwise falls back to random."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --selection random. Default: 0.",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Also create a grid for HDBSCAN noise points, label -1.",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Do not draw a title on top of each grid.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Save cluster_grid_summary.csv with counts and output paths.",
    )
    return parser.parse_args()


def infer_image_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Column {requested!r} not found. Available columns: {list(df.columns)}")
        return requested

    candidates = ["image_path", "path", "filepath", "file_path", "file", "filename"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not infer image path column. Use --image-col. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_path_string(value: object) -> str:
    # CSVs generated on Windows often contain backslashes. Converting them makes
    # the script portable on Linux/macOS while still working on Windows.
    return str(value).strip().replace("\\", "/")


def resolve_image_path(raw_path: object, project_root: Path, image_root: Path | None) -> Path | None:
    normalized = normalize_path_string(raw_path)
    candidate = Path(normalized)

    tried: list[Path] = []

    if candidate.is_absolute():
        tried.append(candidate)
    else:
        tried.append(project_root / candidate)
        tried.append(Path.cwd() / candidate)

    if image_root is not None:
        image_root = image_root.resolve()
        tried.append(image_root / candidate.name)
        # One-level recursive fallback by filename. This is useful when the CSV
        # stores data/glomeruli/crops/name.png but the user passes --image-root.
        if image_root.exists():
            matches = list(image_root.rglob(candidate.name))
            tried.extend(matches[:3])

    for path in tried:
        if path.exists() and path.is_file():
            return path.resolve()

    return None


def select_rows(group: pd.DataFrame, max_items: int, selection: str, seed: int) -> pd.DataFrame:
    if len(group) <= max_items:
        return group.copy()

    if selection == "first":
        return group.head(max_items).copy()

    if selection == "random":
        return group.sample(n=max_items, random_state=seed).copy()

    if selection == "centroid" and {"umap_1", "umap_2"}.issubset(group.columns):
        coords = group[["umap_1", "umap_2"]].to_numpy(dtype=float)
        centroid = coords.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(coords - centroid, axis=1)
        selected_idx = np.argsort(distances)[:max_items]
        return group.iloc[selected_idx].copy()

    # Fallback if centroid was requested but UMAP coordinates are unavailable.
    return group.sample(n=max_items, random_state=seed).copy()


def load_tile(path: Path, tile_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (tile_size, tile_size))

    canvas = Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
    left = (tile_size - image.width) // 2
    top = (tile_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def get_font(size: int = 16) -> ImageFont.ImageFont:
    # Use default font for maximum portability.
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def make_grid(
    image_paths: Iterable[Path],
    title: str,
    out_path: Path,
    cols: int,
    tile_size: int,
    draw_title: bool,
) -> None:
    image_paths = list(image_paths)
    if len(image_paths) == 0:
        return

    rows = math.ceil(len(image_paths) / cols)
    padding = 8
    title_height = 34 if draw_title else 0

    width = cols * tile_size + (cols + 1) * padding
    height = rows * tile_size + (rows + 1) * padding + title_height

    grid = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(grid)

    if draw_title:
        font = get_font(18)
        draw.text((padding, padding), title, fill=(0, 0, 0), font=font)

    y_offset = title_height + padding

    for i, path in enumerate(image_paths):
        row = i // cols
        col = i % cols
        x = padding + col * (tile_size + padding)
        y = y_offset + row * (tile_size + padding)
        try:
            tile = load_tile(path, tile_size=tile_size)
        except Exception:
            tile = Image.new("RGB", (tile_size, tile_size), color=(230, 230, 230))
            d = ImageDraw.Draw(tile)
            d.text((8, 8), "load\nerror", fill=(0, 0, 0), font=get_font(14))
        grid.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> None:
    args = parse_args()

    labels_csv = args.labels_csv.resolve()
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    project_root = args.project_root.resolve() if args.project_root else Path.cwd().resolve()
    image_root = args.image_root.resolve() if args.image_root else None

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "results" / "cluster_grids" / labels_csv.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)
    image_col = infer_image_column(df, args.image_col)

    if args.label_col not in df.columns:
        raise ValueError(f"Column {args.label_col!r} not found. Available columns: {list(df.columns)}")

    df = df.copy()
    df[args.label_col] = df[args.label_col].astype(int)

    # Resolve image paths once and keep only existing images.
    resolved_paths = []
    missing = 0
    for value in df[image_col]:
        path = resolve_image_path(value, project_root=project_root, image_root=image_root)
        if path is None:
            missing += 1
            resolved_paths.append(None)
        else:
            resolved_paths.append(path)

    df["resolved_image_path"] = resolved_paths
    df_existing = df[df["resolved_image_path"].notna()].copy()

    if len(df_existing) == 0:
        raise FileNotFoundError(
            "No image paths could be resolved. Try passing --project-root or --image-root. "
            f"Example raw path from CSV: {df[image_col].iloc[0]!r}"
        )

    print(f"Loaded labels: {len(df)} rows")
    print(f"Resolved images: {len(df_existing)}")
    if missing:
        print(f"[WARN] Missing images: {missing}")

    labels = sorted(df_existing[args.label_col].unique())
    if not args.include_noise:
        labels = [label for label in labels if label != -1]

    summary_rows = []

    for label in labels:
        group = df_existing[df_existing[args.label_col] == label].copy()
        selected = select_rows(
            group,
            max_items=args.max_per_cluster,
            selection=args.selection,
            seed=args.seed,
        )

        label_name = "noise" if label == -1 else f"cluster_{label}"
        out_path = output_dir / f"{label_name}_grid.png"
        title = f"{labels_csv.stem} | {label_name} | shown {len(selected)}/{len(group)}"

        make_grid(
            image_paths=selected["resolved_image_path"].tolist(),
            title=title,
            out_path=out_path,
            cols=args.cols,
            tile_size=args.tile_size,
            draw_title=not args.no_titles,
        )

        print(f"Saved {label_name}: {out_path}")
        summary_rows.append(
            {
                "label": int(label),
                "name": label_name,
                "n_total": int(len(group)),
                "n_shown": int(len(selected)),
                "grid_path": str(out_path),
            }
        )

    if args.write_summary:
        summary_path = output_dir / "cluster_grid_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
