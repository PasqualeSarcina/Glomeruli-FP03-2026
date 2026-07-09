import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import SegmentationDataset
from src.segmentation.unet import build_unet, compile_unet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U-Net training for glomeruli segmentation."
    )

    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset root. Must contain train/ and validation/ folders.",
    )

    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument(
        "--loss-fn",
        choices=("combined", "crossentropy"),
        default="combined",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )

    parser.add_argument(
        "--base-filters",
        type=int,
        default=64,
        choices=(32, 64),
        help="Number of filters in the first U-Net encoder block.",
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        help="Dropout at U-Net bottleneck. 0.0 disables dropout.",
    )

    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Add random horizontal flip to training augmentation.",
    )

    parser.add_argument(
        "--brightness-delta",
        type=float,
        default=0.0,
        help="Max delta for random brightness jitter. 0.0 disables.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "unet",
        help="Where to save checkpoints, logs, and history.",
    )

    return parser.parse_args()


def build_datasets(args: argparse.Namespace):
    train_ds = SegmentationDataset(
        args.dataset_path / "train",
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        flip_horizontal=args.flip_horizontal,
        brightness_delta=args.brightness_delta,
    ).build()

    val_ds = SegmentationDataset(
        args.dataset_path / "validation",
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    ).build()

    return train_ds, val_ds


def make_callbacks(
    output_dir: Path,
    log_name: str,
    early_stopping: bool = True,
):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_mean_io_u",
            mode="max",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(output_dir / log_name),
            append=True,
        ),
    ]

    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_mean_io_u",
                mode="max",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )
        )

    return callbacks


def main() -> None:
    args = parse_args()

    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(args)

    model = build_unet(
        input_shape=(384, 384, 3),
        num_classes=2,
        base_filters=args.base_filters,
        dropout_rate=args.dropout_rate,
    )

    miou = keras.metrics.MeanIoU(
        num_classes=2,
        sparse_y_pred=False,
        name="mean_io_u",
    )

    model = compile_unet(
        model,
        initial_lr=args.lr,
        loss_fn=args.loss_fn,
        miou_metric=miou,
    )

    model.summary()

    print(
        f"\n=== Training U-Net: lr={args.lr}, "
        f"epochs={args.epochs}, "
        f"base_filters={args.base_filters}, "
        f"dropout={args.dropout_rate} ==="
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=make_callbacks(
            args.output_dir,
            "training_log.csv",
            early_stopping=True,
        ),
    )

    model.save(str(args.output_dir / "final_model.keras"))

    with (args.output_dir / "history.json").open("w") as f:
        json.dump(history.history, f, indent=2)

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()