"""
Preprocessing figure for the report: WSI -> tissue mask -> retained windows -> patches.

Run on the cluster (needs openslide + the slides). Reuses the cached tissue mask
in data/tissue_masks/<slide>.npy when present, so it does not recompute the
entropy masking just to draw a picture.

    python scripts/make_preprocessing_figure.py ~/slides/RECHERCHE-004.svs \
        --out figure_preprocessing.png

Panel (d) shows real extracted patches with their rasterised glomerulus masks,
picked among the windows that actually contain an annotated glomerulus.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import openslide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extract_masks import (
    extract_tissue_mask, parse_xml_annotations, create_patch_seg_mask,
)

WINDOW = 2000          # sliding window at level 0, as in preprocess_data.py
MIN_TISSUE = 0.05      # a window is kept when >=5% of it is tissue
GLOM_RGB = np.array([0.85, 0.20, 0.20])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slide", type=Path)
    ap.add_argument("--xml", type=Path, default=None,
                    help="Annotation XML. Defaults to the slide path with .xml.")
    ap.add_argument("--level", type=int, default=2, help="Level used for the tissue mask.")
    ap.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "tissue_masks")
    ap.add_argument("--n-patches", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("figure_preprocessing.png"))
    args = ap.parse_args()

    xml_path = args.xml or args.slide.with_suffix(".xml")
    slide = openslide.OpenSlide(args.slide)
    w0, h0 = slide.level_dimensions[0]

    # (a) overview thumbnail
    thumb = np.array(slide.get_thumbnail((900, 900)).convert("RGB")) / 255.0

    # (b) tissue mask — reuse the cached one if the pipeline already produced it
    cached = args.cache / f"{args.slide.stem}.npy"
    if cached.exists():
        tissue = np.load(cached)
        print(f"tissue mask: reused {cached}")
    else:
        print("tissue mask: computing (no cache found)")
        tissue = extract_tissue_mask(args.slide, level=args.level)
    tissue = tissue.astype(bool)
    th, tw = tissue.shape

    # (c) which sliding windows survive the tissue test
    sx, sy = w0 / tw, h0 / th          # level-0 pixels per mask pixel
    kept, dropped = [], []
    for y0 in range(0, h0 - WINDOW + 1, WINDOW):
        for x0 in range(0, w0 - WINDOW + 1, WINDOW):
            my0, my1 = int(y0 / sy), int((y0 + WINDOW) / sy)
            mx0, mx1 = int(x0 / sx), int((x0 + WINDOW) / sx)
            sub = tissue[my0:my1, mx0:mx1]
            frac = sub.mean() if sub.size else 0.0
            (kept if frac >= MIN_TISSUE else dropped).append((x0, y0))
    print(f"windows: {len(kept)} kept / {len(kept) + len(dropped)} total")

    # (d) real patches containing an annotated glomerulus
    polygons = parse_xml_annotations(xml_path) if xml_path.exists() else []
    print(f"annotations: {len(polygons)} polygons")
    # Score every candidate window first, then take the best ones. Scanning in
    # raster order and stopping at the first hits gives glomeruli clipped by the
    # patch border, which illustrate the pipeline poorly.
    scored = []
    for x0, y0 in kept:
        m = np.array(create_patch_seg_mask(polygons, (x0, y0), WINDOW))
        area = int(m.sum())        # int(): negating an unsigned sum below would overflow
        if area < 0.02 * WINDOW * WINDOW:          # want a clearly visible glomerulus
            continue
        touches = (m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())
        scored.append((touches, -area, x0, y0))    # whole glomeruli first, then largest
    scored.sort()

    patches = []
    for touches, negarea, x0, y0 in scored[: args.n_patches]:
        m = np.array(create_patch_seg_mask(polygons, (x0, y0), WINDOW))
        img = np.array(slide.read_region((x0, y0), 0, (WINDOW, WINDOW)).convert("RGB")) / 255.0
        img = img[::5, ::5]                        # downsample: this is a figure, not data
        patches.append((img, m[::5, ::5].astype(bool)))
        print(f"patch ({x0},{y0}) glom_px={-negarea} clipped={touches}")

    # ---------------- layout: 3 panels on top, patch strip below ----------------
    # Row heights must follow the content aspect, otherwise the wide-and-short
    # slide thumbnails float in a tall axes box and leave a large empty band.
    ncols = max(3, len(patches))
    top_aspect = thumb.shape[0] / thumb.shape[1]      # panels in row 0 are this tall per unit width
    fig_w = 7.0
    col_w = fig_w / ncols
    fig_h = col_w * top_aspect + col_w + 0.9          # + room for the titles
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = fig.add_gridspec(2, ncols, height_ratios=[top_aspect, 1.0],
                          hspace=0.25, wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(thumb); ax.set_title("(a) Whole Slide Image", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tissue, cmap="gray"); ax.set_title("(b) Entropy tissue mask", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(thumb)
    fx, fy = thumb.shape[1] / w0, thumb.shape[0] / h0
    for x0, y0 in kept:
        ax.add_patch(mpatches.Rectangle((x0 * fx, y0 * fy), WINDOW * fx, WINDOW * fy,
                                        fill=False, ec="#1F77B4", lw=0.25))
    ax.set_title(f"(c) Retained windows ($\\geq${int(MIN_TISSUE * 100)}% tissue)", fontsize=8)

    for i, (img, m) in enumerate(patches):
        ax = fig.add_subplot(gs[1, i])
        over = img.copy()
        over[m] = 0.55 * over[m] + 0.45 * GLOM_RGB
        ax.imshow(over)
        if i == 0:
            ax.set_title("(d) Extracted patches with the rasterised glomerulus mask",
                         fontsize=8, loc="left")

    for a in fig.axes:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#BBBBBB"); s.set_linewidth(0.5)

    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
