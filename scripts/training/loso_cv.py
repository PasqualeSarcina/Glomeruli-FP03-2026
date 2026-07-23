"""
Leave-one-slide-out cross-validation for the reference SegNet-VGG19 recipe.

Motivation: the fixed split gives a single 2-slide test and, worse, a 1-slide
validation set that has repeatedly ranked models the opposite way from the test
set (Runs 4-8). LOSO removes both problems: for each of the 9 slides we hold it
out as the test fold, train on the other 8, and evaluate on the held-out slide.
Averaging over the 9 folds gives an honest generalization estimate with a
per-slide breakdown (which slides are hard).

Design choices that make this clean:
  * NO validation set and NO EarlyStopping. We train a FIXED budget (the Run 7
    schedule: 10 frozen epochs + 20 fine-tune) and evaluate the FINAL model on
    the held-out slide. Selecting an epoch by a held-in val set would reintroduce
    exactly the small-val-set unreliability this experiment exists to avoid, and
    selecting by the test slide would be leakage. Fixed budget sidesteps both.
  * Recipe = Run 7 (the best model): VGG19 ImageNet encoder, two-phase training,
    on-the-fly D4 + horizontal flip + HED stain jitter augmentation.
  * The static Reinhard-augmented patches (*_reinhard_*.png) are DROPPED: they
    exist only for the original-train slides (uneven per slide) and a held-out
    slide used as a Reinhard target would leak. The on-the-fly stain jitter
    replaces them.
  * keras.backend.clear_session() between folds — 9 fresh models in one process
    would otherwise accumulate graph state / metric-name suffixes and memory.

Usage:
    python scripts/training/loso_cv.py "$SCRATCH_FLASH/glomeruli-dataset" \
        --output-dir "$SIM/models/loso"
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import SegmentationDataset
from src.segmentation.segnet import (
    build_segnet_vgg19, build_segnet_resnet50, compile_segnet,
    freeze_encoder, unfreeze_encoder, freeze_resnet_encoder, unfreeze_resnet_encoder,
    tta_predict,
)

SLIDE_RE = re.compile(r"(RECHERCHE-\d+)")


def gather_pairs_by_slide(dataset_root: Path) -> dict:
    """
    Pool every BASE patch (no *_reinhard_* variants) across the on-disk
    train/validation/test folders and group by slide of origin, read from the
    filename prefix (e.g. RECHERCHE-003_100015_44007.png -> RECHERCHE-003).
    """
    pairs = defaultdict(list)
    for split in ("train", "validation", "test"):
        img_dir = dataset_root / split / "img"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.png")):
            if "_reinhard_" in img.name:
                continue
            mask = dataset_root / split / "mask" / img.name
            if not mask.exists():
                continue
            m = SLIDE_RE.match(img.name)
            if m:
                pairs[m.group(1)].append((str(img), str(mask)))
    return dict(pairs)


def evaluate_iou(model, test_ds, tta=False) -> tuple:
    """Per-class IoU (background, glomerulus) + mean, via one accumulated 2x2 CM."""
    cm = tf.zeros((2, 2), dtype=tf.int64)
    for images, masks in test_ds:
        probs = tta_predict(model, images) if tta else model(images, training=False)
        preds = tf.argmax(probs, axis=-1)
        y_true = tf.cast(tf.squeeze(masks, axis=-1), tf.int64)
        cm += tf.math.confusion_matrix(
            tf.reshape(y_true, [-1]), tf.reshape(tf.cast(preds, tf.int64), [-1]),
            num_classes=2, dtype=tf.int64,
        )
    cm = cm.numpy().astype(np.float64)
    inter = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - inter
    iou = inter / np.maximum(union, 1e-12)
    return float(iou[0]), float(iou[1]), float(np.mean(iou))


def train_one_fold(train_pairs, test_pairs, args) -> tuple:
    train_ds = SegmentationDataset.from_pairs(
        [p[0] for p in train_pairs], [p[1] for p in train_pairs],
        batch_size=args.batch_size, shuffle=True, augment=True,
        flip_horizontal=True, stain_jitter=args.stain_jitter,
        copy_paste=args.copy_paste,
    ).build()
    test_ds = SegmentationDataset.from_pairs(
        [p[0] for p in test_pairs], [p[1] for p in test_pairs],
        batch_size=args.batch_size, shuffle=False, augment=False,
    ).build()

    if args.encoder == "resnet50":
        model = build_segnet_resnet50(encoder_weights_path=str(args.encoder_weights)
                                      if args.encoder_weights else None)
        freeze, unfreeze = freeze_resnet_encoder, unfreeze_resnet_encoder
    else:
        model = build_segnet_vgg19()
        freeze, unfreeze = freeze_encoder, unfreeze_encoder
    miou = keras.metrics.MeanIoU(num_classes=2, sparse_y_pred=False)

    # Phase 1: frozen encoder warm-up (no validation, no callbacks).
    freeze(model)
    compile_segnet(model, initial_lr=args.phase_1_lr, miou_metric=miou)
    model.fit(train_ds, epochs=args.phase_1_epochs, verbose=2)

    # Phase 2: full fine-tune.
    unfreeze(model)
    compile_segnet(model, initial_lr=args.phase_2_lr, miou_metric=miou)
    model.fit(train_ds, epochs=args.phase_1_epochs + args.phase_2_epochs,
              initial_epoch=args.phase_1_epochs, verbose=2)

    return evaluate_iou(model, test_ds, tta=args.tta)


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-slide-out CV for SegNet-VGG19.")
    ap.add_argument("dataset_path", type=Path, help="Root with train/ validation/ test/ subfolders.")
    ap.add_argument("--encoder", choices=["vgg19", "resnet50"], default="vgg19",
                    help="Encoder backbone. 'resnet50' loads converted pathology weights.")
    ap.add_argument("--encoder-weights", type=Path, default=None,
                    help="Converted Keras encoder weights (.weights.h5) for --encoder resnet50 "
                         "(see scripts/encoders/convert_swav_resnet50.py / setup_and_convert.sh).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--phase-1-epochs", type=int, default=10)
    ap.add_argument("--phase-1-lr", type=float, default=0.01)
    ap.add_argument("--phase-2-epochs", type=int, default=20)
    ap.add_argument("--phase-2-lr", type=float, default=0.001)
    ap.add_argument("--stain-jitter", type=float, default=0.05)
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="Copy-paste augmentation probability (e.g. 0.5). 0 disables.")
    ap.add_argument("--tta", action="store_true",
                    help="Test-time augmentation (8x D4) when evaluating each fold.")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "loso")
    args = ap.parse_args()
    print(f"Config: encoder={args.encoder}  weights={args.encoder_weights}  "
          f"copy_paste={args.copy_paste}  tta={args.tta}  stain_jitter={args.stain_jitter}")

    print(f"TensorFlow {tf.__version__}, GPUs: {tf.config.list_physical_devices('GPU')}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs_by_slide = gather_pairs_by_slide(args.dataset_path)
    slides = sorted(pairs_by_slide)
    total = sum(len(v) for v in pairs_by_slide.values())
    print(f"\n{len(slides)} slides, {total} base patches (Reinhard variants dropped):")
    for s in slides:
        print(f"  {s}: {len(pairs_by_slide[s])}")

    results = {}
    for i, held in enumerate(slides, start=1):
        test_pairs = pairs_by_slide[held]
        train_pairs = [p for s in slides if s != held for p in pairs_by_slide[s]]
        print(f"\n===== Fold {i}/{len(slides)} — hold out {held} "
              f"(train {len(train_pairs)} / test {len(test_pairs)}) =====")
        bg, gl, mn = train_one_fold(train_pairs, test_pairs, args)
        results[held] = {"n_test": len(test_pairs), "iou_background": bg,
                         "iou_glomerulus": gl, "mean_iou": mn}
        print(f"[{held}] glomerulus IoU {gl:.4f} | mean IoU {mn:.4f}")
        keras.backend.clear_session()

    glom = np.array([r["iou_glomerulus"] for r in results.values()])
    mean = np.array([r["mean_iou"] for r in results.values()])
    summary = {
        "per_slide": results,
        "glomerulus_iou_mean": float(glom.mean()), "glomerulus_iou_std": float(glom.std()),
        "mean_iou_mean": float(mean.mean()), "mean_iou_std": float(mean.std()),
    }
    with (args.output_dir / "loso_results.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ LOSO summary ================")
    print(f"{'slide':<16}{'n_test':>8}{'glom IoU':>11}{'mean IoU':>11}")
    for s in slides:
        r = results[s]
        print(f"{s:<16}{r['n_test']:>8}{r['iou_glomerulus']:>11.4f}{r['mean_iou']:>11.4f}")
    print("-" * 46)
    print(f"{'MEAN +/- STD':<16}{'':>8}{glom.mean():>7.4f}±{glom.std():.4f} {mean.mean():>6.4f}±{mean.std():.4f}")
    print(f"\nReference (fixed split): Run 7 test glomerulus 0.7914 / mean 0.8910")
    print(f"Saved: {args.output_dir / 'loso_results.json'}")


if __name__ == "__main__":
    main()
