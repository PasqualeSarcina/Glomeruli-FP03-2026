"""
Evaluate a trained SegNet-VGG19 model on a dataset split (inference only).

Computes per-class IoU (background, glomerulus) and mean IoU over the whole
split via a single accumulated 2x2 confusion matrix. No training, no weights
are modified — safe to run on a checkpoint.

The model is loaded with compile=False: we only need forward passes, so there
is no need to supply custom_objects for combined_loss/dice_loss, and the IoU is
computed here directly (consistent with the val_mean_io_u tracked in training,
which also uses argmax over the softmax output with num_classes=2).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import SegmentationDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a SegNet model on a dataset split (inference only)."
    )
    p.add_argument("model_path", type=Path, help="Path to a .keras model file.")
    p.add_argument(
        "dataset_split", type=Path,
        help="Path to the split directory (must contain img/ and mask/ sub-folders).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--label", type=str, default=None,
                   help="Optional name for this model in the printed/JSON output.")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Optional path to write the metrics as JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = keras.models.load_model(str(args.model_path), compile=False)

    test_ds = SegmentationDataset(
        args.dataset_split,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    ).build()

    # Accumulate one 2x2 confusion matrix across the entire split.
    cm = tf.zeros((2, 2), dtype=tf.int64)
    n_batches = 0
    for images, masks in test_ds:
        probs = model(images, training=False)                   # (B, H, W, 2)
        preds = tf.argmax(probs, axis=-1)                       # (B, H, W)
        y_true = tf.cast(tf.squeeze(masks, axis=-1), tf.int64)  # (B, H, W)
        cm += tf.math.confusion_matrix(
            tf.reshape(y_true, [-1]),
            tf.reshape(tf.cast(preds, tf.int64), [-1]),
            num_classes=2,
            dtype=tf.int64,
        )
        n_batches += 1

    cm = cm.numpy().astype(np.float64)
    # IoU per class = TP / (TP + FP + FN) = diag / (row_sum + col_sum - diag)
    intersection = np.diag(cm)
    row_sum = cm.sum(axis=1)   # ground-truth pixels per class
    col_sum = cm.sum(axis=0)   # predicted pixels per class
    union = row_sum + col_sum - intersection
    iou = intersection / np.maximum(union, 1e-12)
    mean_iou = float(np.mean(iou))

    label = args.label or args.model_path.name
    print(f"\n=== Test evaluation: {label} ===")
    print(f"Batches evaluated: {n_batches}")
    print(f"Confusion matrix [rows=true, cols=pred]:\n{cm.astype(np.int64)}")
    print(f"IoU background (class 0): {iou[0]:.4f}")
    print(f"IoU glomerulus (class 1): {iou[1]:.4f}")
    print(f"Mean IoU:                 {mean_iou:.4f}")

    if args.output_json is not None:
        metrics = {
            "label": label,
            "model_path": str(args.model_path),
            "split": str(args.dataset_split),
            "iou_background": float(iou[0]),
            "iou_glomerulus": float(iou[1]),
            "mean_iou": mean_iou,
            "confusion_matrix": cm.astype(np.int64).tolist(),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
