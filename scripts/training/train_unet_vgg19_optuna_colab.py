"""
Optuna hyperparameter search for U-Net-VGG19 glomeruli segmentation.

Mirrors Federico's SegNet-VGG19 Optuna search:
- phase2_lr: log-uniform [1e-5, 1e-3]
- dropout_rate: uniform [0.0, 0.4]
- phase1_epochs: integer [5, 15]
- flip_horizontal: categorical [True, False]
- brightness_delta: uniform [0.0, 0.3]
- maximum 20 epochs per trial
- TPE sampler, seed 42
- MedianPruner with 3 startup trials and 5 warm-up steps
- combined loss
- phase-1 learning rate fixed at 0.01

This version uses U-Net-VGG19 and is suitable for Google Colab.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import keras
import optuna
import tensorflow as tf
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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

MAX_TOTAL_EPOCHS = 20
PHASE_1_LR = 0.01


class _PruningCallback(keras.callbacks.Callback):
    """Report validation MeanIoU to Optuna and stop pruned trials."""

    def __init__(self, trial: optuna.Trial) -> None:
        super().__init__()
        self.trial = trial
        self.global_step = 0
        self.pruned = False

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        val_iou = (logs or {}).get("val_mean_io_u")

        if val_iou is not None:
            self.trial.report(float(val_iou), step=self.global_step)

        self.global_step += 1

        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True


def _build_datasets(
    args: argparse.Namespace,
    flip_horizontal: bool,
    brightness_delta: float,
):
    train_ds = SegmentationDataset(
        args.dataset_path / "train",
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        flip_horizontal=flip_horizontal,
        brightness_delta=brightness_delta,
    ).build()

    val_ds = SegmentationDataset(
        args.dataset_path / "validation",
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    ).build()

    return train_ds, val_ds


def _objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    keras.backend.clear_session()
    gc.collect()

    phase2_lr = trial.suggest_float("phase2_lr", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    phase1_epochs = trial.suggest_int("phase1_epochs", 5, 15)
    flip_horizontal = trial.suggest_categorical(
        "flip_horizontal",
        [True, False],
    )
    brightness_delta = trial.suggest_float(
        "brightness_delta",
        0.0,
        0.3,
    )

    phase2_epochs = max(5, MAX_TOTAL_EPOCHS - phase1_epochs)

    print(f"\n--- Trial {trial.number} ---")
    print(
        f"phase2_lr={phase2_lr:.2e} "
        f"dropout={dropout_rate:.2f} "
        f"phase1_epochs={phase1_epochs} "
        f"phase2_epochs={phase2_epochs} "
        f"flip_horizontal={flip_horizontal} "
        f"brightness_delta={brightness_delta:.2f}"
    )

    train_ds, val_ds = _build_datasets(
        args,
        flip_horizontal=flip_horizontal,
        brightness_delta=brightness_delta,
    )

    trial_dir = args.output_dir / f"trial-{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    with (trial_dir / "sampled_params.json").open("w") as file:
        json.dump(trial.params, file, indent=2)

    model = build_unet_vgg19(dropout_rate=dropout_rate)

    # Reuse the metric instance across recompilation so the metric name remains
    # "mean_io_u" / "val_mean_io_u", matching Federico's training.
    miou = keras.metrics.MeanIoU(
        num_classes=2,
        sparse_y_pred=False,
    )

    pruning_callback = _PruningCallback(trial)
    csv_logger = keras.callbacks.CSVLogger(
        str(trial_dir / "training_log.csv"),
        append=True,
    )

    # Phase 1: frozen VGG19 encoder.
    freeze_encoder(model)
    compile_unet_vgg19(
        model,
        initial_lr=PHASE_1_LR,
        loss_fn="combined",
        miou_metric=miou,
    )

    checkpoint_phase1 = keras.callbacks.ModelCheckpoint(
        str(trial_dir / "best_model.keras"),
        monitor="val_mean_io_u",
        mode="max",
        save_best_only=True,
        verbose=0,
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=[
            csv_logger,
            checkpoint_phase1,
            pruning_callback,
        ],
        verbose=2,
    )

    if pruning_callback.pruned:
        raise optuna.TrialPruned()

    phase1_best = max(
        history_phase1.history.get(
            "val_mean_io_u",
            [float("-inf")],
        )
    )

    # Phase 2: full fine-tuning.
    unfreeze_encoder(model)
    compile_unet_vgg19(
        model,
        initial_lr=phase2_lr,
        loss_fn="combined",
        miou_metric=miou,
    )

    checkpoint_phase2 = keras.callbacks.ModelCheckpoint(
        str(trial_dir / "best_model.keras"),
        monitor="val_mean_io_u",
        mode="max",
        save_best_only=True,
        verbose=0,
        initial_value_threshold=phase1_best,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_mean_io_u",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs + phase2_epochs,
        initial_epoch=phase1_epochs,
        callbacks=[
            csv_logger,
            checkpoint_phase2,
            early_stopping,
            pruning_callback,
        ],
        verbose=2,
    )

    if pruning_callback.pruned:
        raise optuna.TrialPruned()

    best_val_iou = max(
        max(
            history_phase1.history.get(
                "val_mean_io_u",
                [float("-inf")],
            )
        ),
        max(
            history_phase2.history.get(
                "val_mean_io_u",
                [float("-inf")],
            )
        ),
    )

    result = {
        **trial.params,
        "phase1_lr": PHASE_1_LR,
        "phase2_epochs": phase2_epochs,
        "best_val_iou": best_val_iou,
    }

    with (trial_dir / "params.json").open("w") as file:
        json.dump(result, file, indent=2)

    print(
        f"Trial {trial.number} finished — "
        f"best val_mean_io_u: {best_val_iou:.4f}"
    )

    del model, train_ds, val_ds
    keras.backend.clear_session()
    gc.collect()

    return best_val_iou


def _save_study_results(
    study: optuna.Study,
    output_dir: Path,
) -> None:
    summary = {
        "trial_number": study.best_trial.number,
        "best_val_iou": study.best_value,
        **study.best_params,
    }

    with (output_dir / "best_params.json").open("w") as file:
        json.dump(summary, file, indent=2)

    trials = study.trials_dataframe()
    trials.to_csv(output_dir / "all_trials.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna search for U-Net-VGG19 on Google Colab."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Dataset root containing train/ and validation/ folders.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL. Example: "
            "sqlite:////content/drive/MyDrive/unet_optuna/optuna.db"
        ),
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="unet_vgg19_optuna",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "unet_vgg19_optuna",
    )
    args = parser.parse_args()

    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Default to a persistent local SQLite study inside output-dir.
    # Put output-dir on Google Drive to preserve it across Colab disconnects.
    if args.storage is None:
        database_path = (args.output_dir / "optuna.db").resolve()
        args.storage = f"sqlite:///{database_path}"

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=5,
        ),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: _objective(trial, args),
        n_trials=args.n_trials,
        catch=(tf.errors.ResourceExhaustedError,),
        gc_after_trial=True,
    )

    print("\n=== Search complete ===")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation IoU: {study.best_value:.4f}")
    print(
        "Best parameters:\n"
        + json.dumps(study.best_params, indent=2)
    )

    _save_study_results(study, args.output_dir)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
