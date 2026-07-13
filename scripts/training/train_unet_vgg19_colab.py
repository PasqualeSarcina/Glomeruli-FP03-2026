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
from src.segmentation.unet_vgg19 import (
    build_unet_vgg19,
    compile_unet_vgg19,
    freeze_encoder,
    unfreeze_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-phase U-Net-VGG19 training for Colab: frozen encoder "
            "warm-up followed by full fine-tuning."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Dataset root containing train/ and validation/ subfolders.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--loss-fn",
        choices=("combined", "crossentropy"),
        default="combined",
    )
    parser.add_argument("--phase-1-epochs", type=int, default=10)
    parser.add_argument("--phase-1-lr", type=float, default=0.01)
    parser.add_argument("--phase-2-epochs", type=int, default=20)
    parser.add_argument("--phase-2-lr", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--brightness-delta", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "unet_vgg19",
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
    best_so_far: float = -float("inf"),
    early_stopping: bool = False,
):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_mean_io_u",
            mode="max",
            verbose=1,
            initial_value_threshold=best_so_far,
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

    model = build_unet_vgg19(dropout_rate=args.dropout_rate)

    # Reuse one metric instance so its logged name stays val_mean_io_u
    # after recompiling for phase 2.
    miou = keras.metrics.MeanIoU(
        num_classes=2,
        sparse_y_pred=False,
    )

    print(
        f"\n=== Phase 1: frozen encoder, lr={args.phase_1_lr}, "
        f"{args.phase_1_epochs} epochs ==="
    )
    freeze_encoder(model)
    model = compile_unet_vgg19(
        model,
        initial_lr=args.phase_1_lr,
        loss_fn=args.loss_fn,
        miou_metric=miou,
    )
    model.summary()

    history_phase_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.phase_1_epochs,
        callbacks=make_callbacks(
            args.output_dir,
            "training_log.csv",
        ),
    )

    phase_1_best = max(
        history_phase_1.history.get(
            "val_mean_io_u",
            [-float("inf")],
        )
    )

    print(
        f"\n=== Phase 2: unfrozen, lr={args.phase_2_lr}, "
        f"{args.phase_2_epochs} epochs ==="
    )
    unfreeze_encoder(model)
    model = compile_unet_vgg19(
        model,
        initial_lr=args.phase_2_lr,
        loss_fn=args.loss_fn,
        miou_metric=miou,
    )

    history_phase_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.phase_1_epochs + args.phase_2_epochs,
        initial_epoch=args.phase_1_epochs,
        callbacks=make_callbacks(
            args.output_dir,
            "training_log.csv",
            best_so_far=phase_1_best,
            early_stopping=True,
        ),
    )

    model.save(str(args.output_dir / "final_model.keras"))

    combined_history = {
        key: (
            history_phase_1.history.get(key, [])
            + history_phase_2.history.get(key, [])
        )
        for key in (
            set(history_phase_1.history)
            | set(history_phase_2.history)
        )
    }

    with (args.output_dir / "history.json").open("w") as file:
        json.dump(combined_history, file, indent=2)

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
