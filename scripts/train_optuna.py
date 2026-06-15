"""
Optuna hyperparameter search for SegNet-VGG19 glomeruli segmentation.

Tuned params:
  phase2_lr       log-uniform [1e-5, 1e-3]
  dropout_rate    uniform     [0.0, 0.4]
  phase1_epochs   int         [5, 15]
  flip_horizontal categorical [True, False]
  brightness_delta uniform    [0.0, 0.3]

Each trial caps at MAX_TOTAL_EPOCHS (phase1 + phase2).
Bad trials are pruned early via MedianPruner.
The study is backed by a SQLite DB so it survives job restarts.
"""

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf
import keras
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

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

MAX_TOTAL_EPOCHS = 20  # hard cap per trial (phase1 + phase2)
PHASE_1_LR = 0.01      # warm-up LR kept fixed across trials


class _PruningCallback(keras.callbacks.Callback):
    """Reports val_mean_io_u to Optuna each epoch; gracefully stops pruned trials."""

    def __init__(self, trial: optuna.Trial) -> None:
        super().__init__()
        self.trial = trial
        self._global_step = 0
        self.pruned = False

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        val_iou = (logs or {}).get("val_mean_io_u")
        if val_iou is not None:
            self.trial.report(float(val_iou), step=self._global_step)
        self._global_step += 1
        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True


def _objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    phase2_lr        = trial.suggest_float("phase2_lr", 1e-5, 1e-3, log=True)
    dropout_rate     = trial.suggest_float("dropout_rate", 0.0, 0.4)
    phase1_epochs    = trial.suggest_int("phase1_epochs", 5, 15)
    flip_horizontal  = trial.suggest_categorical("flip_horizontal", [True, False])
    brightness_delta = trial.suggest_float("brightness_delta", 0.0, 0.3)

    phase2_max = max(5, MAX_TOTAL_EPOCHS - phase1_epochs)

    print(f"\n--- Trial {trial.number} ---")
    print(f"  phase2_lr={phase2_lr:.2e}  dropout={dropout_rate:.2f}  "
          f"phase1_ep={phase1_epochs}  flip_h={flip_horizontal}  "
          f"brightness={brightness_delta:.2f}")

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

    trial_dir = args.output_dir / f"trial-{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    model = build_segnet_vgg19(dropout_rate=dropout_rate)
    miou = keras.metrics.MeanIoU(num_classes=2, sparse_y_pred=False)
    pruning_cb = _PruningCallback(trial)
    csv_log = keras.callbacks.CSVLogger(str(trial_dir / "training_log.csv"), append=True)

    # ----- Phase 1: frozen encoder -----
    freeze_encoder(model)
    compile_segnet(model, initial_lr=PHASE_1_LR, loss_fn="combined", miou_metric=miou)

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(trial_dir / "best_model.keras"),
        monitor="val_mean_io_u", mode="max", save_best_only=True, verbose=0,
    )

    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=[csv_log, checkpoint, pruning_cb],
        verbose=2,
    )

    if pruning_cb.pruned:
        raise optuna.TrialPruned()

    phase1_best = max(h1.history.get("val_mean_io_u", [float("-inf")]))

    # ----- Phase 2: full fine-tune -----
    unfreeze_encoder(model)
    compile_segnet(model, initial_lr=phase2_lr, loss_fn="combined", miou_metric=miou)

    checkpoint2 = keras.callbacks.ModelCheckpoint(
        str(trial_dir / "best_model.keras"),
        monitor="val_mean_io_u", mode="max", save_best_only=True, verbose=0,
        initial_value_threshold=phase1_best,
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_mean_io_u", mode="max",
        patience=5, restore_best_weights=True, verbose=1,
    )

    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1_epochs + phase2_max,
        initial_epoch=phase1_epochs,
        callbacks=[csv_log, checkpoint2, early_stop, pruning_cb],
        verbose=2,
    )

    if pruning_cb.pruned:
        raise optuna.TrialPruned()

    best_val_iou = max(
        max(h1.history.get("val_mean_io_u", [float("-inf")])),
        max(h2.history.get("val_mean_io_u", [float("-inf")])),
    )

    with (trial_dir / "params.json").open("w") as f:
        json.dump({**trial.params, "best_val_iou": best_val_iou}, f, indent=2)

    print(f"Trial {trial.number} finished — best val_mean_io_u: {best_val_iou:.4f}")
    return best_val_iou


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for SegNet-VGG19."
    )
    parser.add_argument("dataset_path", type=Path,
                        help="Dataset root with train/ and validation/ sub-folders.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db. Enables crash recovery.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "models" / "optuna",
    )
    args = parser.parse_args()

    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        study_name="segnet_run5",
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: _objective(trial, args),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    print("\n=== Search complete ===")
    print(f"Best trial : #{study.best_trial.number}")
    print(f"Best val IoU: {study.best_value:.4f}")
    print(f"Best params : {json.dumps(study.best_params, indent=2)}")

    summary = {
        "trial_number": study.best_trial.number,
        "best_val_iou": study.best_value,
        **study.best_params,
    }
    out = args.output_dir / "best_params.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
