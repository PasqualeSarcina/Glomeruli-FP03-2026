import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import SegmentationDataset
from src.segmentation.segnet import build_segnet_vgg19, compile_segnet, lr_step_decay
import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SegNet-VGG19 for glomerulus segmentation."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset root (must contain train/, validation/ sub-folders).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. Default: 10.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size. Default: 8.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial SGD learning rate. Default: 0.1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory where model checkpoints and logs are saved. Default: models/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

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

    model = build_segnet_vgg19()
    model = compile_segnet(model, initial_lr=args.lr)
    model.summary()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(args.output_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_mean_io_u",
            mode="max",
            verbose=1,
        ),
        keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1),
        keras.callbacks.CSVLogger(str(args.output_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(str(args.output_dir / "final_model.keras"))

    with (args.output_dir / "history.json").open("w") as f:
        json.dump(history.history, f, indent=2)

    print(f"Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
