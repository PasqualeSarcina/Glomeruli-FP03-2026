import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import SegmentationDataset
from src.segmentation.segnet import (
    build_segnet_vgg19,
    compile_segnet,
    freeze_encoder,
    unfreeze_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-phase SegNet-VGG19 training (frozen encoder warm-up, then full fine-tune)."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset root (must contain train/ and validation/ sub-folders).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loss-fn", choices=("combined", "crossentropy"), default="combined")

    parser.add_argument("--phase-1-epochs", type=int, default=10,
                        help="Frozen-encoder warm-up epochs.")
    parser.add_argument("--phase-1-lr", type=float, default=0.01)

    parser.add_argument("--phase-2-epochs", type=int, default=20,
                        help="Full fine-tune epochs (encoder + decoder).")
    parser.add_argument("--phase-2-lr", type=float, default=0.001)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Where to save checkpoints, logs, and history.",
    )
    return parser.parse_args()


def build_datasets(args: argparse.Namespace):
    train_ds = SegmentationDataset(
        args.dataset_path / "train",
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
    ).build()

    val_ds = SegmentationDataset(
        args.dataset_path / "validation",
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    ).build()

    return train_ds, val_ds


def make_callbacks(output_dir: Path, log_name: str, best_so_far: float = -float("inf")):
    return [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_mean_io_u",
            mode="max",
            verbose=1,
            initial_value_threshold=best_so_far,
        ),
        keras.callbacks.CSVLogger(str(output_dir / log_name), append=True),
    ]


def main() -> None:
    args = parse_args()

    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(args)

    model = build_segnet_vgg19()

    # ----- Phase 1: frozen encoder warm-up -----
    print(f"\n=== Phase 1: frozen encoder, lr={args.phase_1_lr}, {args.phase_1_epochs} epochs ===")
    freeze_encoder(model)
    model = compile_segnet(model, initial_lr=args.phase_1_lr, loss_fn=args.loss_fn)
    model.summary()

    history_phase_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.phase_1_epochs,
        callbacks=make_callbacks(args.output_dir, "training_log.csv"),
    )

    # ----- Phase 2: unfrozen full fine-tune -----
    # Carry over phase-1's best val IoU so phase-2's ModelCheckpoint doesn't
    # overwrite a better phase-1 model with a worse phase-2 first epoch.
    phase_1_best = max(history_phase_1.history.get("val_mean_io_u", [-float("inf")]))

    print(f"\n=== Phase 2: unfrozen, lr={args.phase_2_lr}, {args.phase_2_epochs} epochs ===")
    unfreeze_encoder(model)
    model = compile_segnet(model, initial_lr=args.phase_2_lr, loss_fn=args.loss_fn)

    history_phase_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.phase_1_epochs + args.phase_2_epochs,
        initial_epoch=args.phase_1_epochs,
        callbacks=make_callbacks(args.output_dir, "training_log.csv", best_so_far=phase_1_best),
    )

    model.save(str(args.output_dir / "final_model.keras"))

    combined_history = {
        key: history_phase_1.history.get(key, []) + history_phase_2.history.get(key, [])
        for key in set(history_phase_1.history) | set(history_phase_2.history)
    }
    with (args.output_dir / "history.json").open("w") as f:
        json.dump(combined_history, f, indent=2)

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
