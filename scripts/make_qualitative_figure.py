"""
Qualitative segmentation figure for the report: input patch | ground truth | prediction.

Run on the cluster (needs TF + the trained model + the patch dataset), then copy
the PNG back into the report's figure/ folder.

    python scripts/make_qualitative_figure.py \
        models/run/best_model.keras data/dataset \
        --out figure_qualitative.png

Picks rows spanning the quality range rather than the best ones: the per-patch
IoU is computed for every candidate and the script samples across its
distribution (a strong, a median and a weak case), so the figure is honest about
the failure modes instead of showcasing cherry-picked successes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_SIZE = (384, 384)
GLOM_RGB = np.array([0.85, 0.20, 0.20])  # overlay tint for the glomerulus class


def load_pair(img_path: Path, mask_path: Path):
    img = tf.image.decode_png(tf.io.read_file(str(img_path)), channels=3)
    img = tf.image.resize(img, INPUT_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.image.decode_png(tf.io.read_file(str(mask_path)), channels=1)
    mask = tf.image.resize(mask, INPUT_SIZE, method="nearest")
    return img.numpy(), mask.numpy()[..., 0].astype(np.uint8)


def patch_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    inter = np.logical_and(gt == 1, pred == 1).sum()
    union = np.logical_or(gt == 1, pred == 1).sum()
    return float(inter) / float(union) if union else float("nan")


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img.copy()
    sel = mask == 1
    out[sel] = (1 - alpha) * out[sel] + alpha * GLOM_RGB
    return np.clip(out, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="Trained .keras model.")
    ap.add_argument("dataset", type=Path, help="Dataset root with test/img and test/mask.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("figure_qualitative.png"))
    ap.add_argument("--rows", type=int, default=3, help="Number of example rows.")
    ap.add_argument("--max-scan", type=int, default=120,
                    help="How many glomerulus-bearing patches to score before picking.")
    ap.add_argument("--worst-percentile", type=float, default=0.85,
                    help="Where the weakest row is taken from, as a fraction of the "
                         "ranked patches (0.85 = the 85th percentile of badness, not the "
                         "single worst). 1.0 restores the absolute worst case.")
    ap.add_argument("--min-glom-pixels", type=int, default=8000,
                    help="Skip patches whose ground-truth glomerulus is smaller than this "
                         "(384x384 = 147k px, a whole glomerulus is ~15%%). Low values let "
                         "slivers clipped by the patch border through, and those score IoU~0 "
                         "for reasons that have nothing to do with the model.")
    args = ap.parse_args()

    img_dir = args.dataset / args.split / "img"
    mask_dir = args.dataset / args.split / "mask"
    if not img_dir.is_dir():
        sys.exit(f"missing {img_dir}")

    model = keras.models.load_model(args.model, compile=False)

    # only patches that actually contain a glomerulus are informative here
    candidates = []
    for p in sorted(img_dir.glob("*.png")):
        if "_reinhard_" in p.name:
            continue
        m = mask_dir / p.name
        if m.exists():
            candidates.append((p, m))
    if not candidates:
        sys.exit("no image/mask pairs found")

    # Stride across the whole list instead of taking the first max-scan entries:
    # the patches are sorted by filename, so the head of the list is a single
    # slide and the figure would illustrate only that one.
    step = max(1, len(candidates) // args.max_scan)
    scan = candidates[::step][: args.max_scan]

    scored = []
    for img_path, mask_path in scan:
        img, gt = load_pair(img_path, mask_path)
        if gt.sum() < args.min_glom_pixels:   # border slivers score IoU~0 spuriously
            continue
        probs = model(img[None, ...], training=False)
        pred = np.argmax(probs.numpy()[0], axis=-1).astype(np.uint8)
        scored.append((patch_iou(gt, pred), img_path.name, img, gt, pred))

    if not scored:
        sys.exit("no patch had enough glomerulus pixels to score")

    scored.sort(key=lambda r: r[0], reverse=True)
    # Span the distribution instead of showing the best rows, but stop at the
    # --worst-percentile rather than at the absolute minimum: the very worst patch
    # is usually a degenerate case (a mis-annotated or clipped mask) that says
    # nothing about how the model actually behaves.
    lo = int(round(args.worst_percentile * (len(scored) - 1)))
    idx = np.linspace(0, lo, num=min(args.rows, lo + 1)).astype(int)
    chosen = [scored[i] for i in idx]

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(7.0, 2.35 * n), dpi=300)
    axes = np.atleast_2d(axes)
    titles = ("Input patch", "Ground truth", "Prediction")

    for r, (iou, name, img, gt, pred) in enumerate(chosen):
        for c, panel in enumerate((img, overlay(img, gt), overlay(img, pred))):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#BBBBBB"); s.set_linewidth(0.5)
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        slide = name.split("_")[0].replace("RECHERCHE-", "")
        axes[r, 0].set_ylabel(f"slide {slide}\nIoU {iou:.2f}", fontsize=8)
        print(f"row {r}: {name}  IoU={iou:.3f}")

    fig.tight_layout(pad=0.4)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"\nsaved {args.out}  (scored {len(scored)} patches)")


if __name__ == "__main__":
    main()
