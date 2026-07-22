#!/bin/bash
#SBATCH --job-name=segnet_run9_imagenet
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s348829@studenti.polito.it
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---- adjust these if needed ----
REPO=$HOME/Glomeruli-FP03-2026
CONDA_ENV=glomeruli
DATASET=$SCRATCH_FLASH/glomeruli-dataset          # pre-extracted patches (train/ validation/ test/)
ENCODER_WEIGHTS=$REPO/models/encoders/imagenet_resnet50.weights.h5   # from setup_and_convert.sh imagenet
RUN8_MODEL=$REPO/models/hpc-run8-1811510/best_model.keras           # SwAV ResNet50 (same arch, diff pretraining)
RUN7_MODEL=$REPO/models/hpc-run7-1797796/best_model.keras           # VGG19 reference
# --------------------------------

# Run 9 is the clean control for Run 8: identical ResNet50-v1.5 architecture,
# only the encoder pretraining differs (ImageNet here vs SwAV pathology in
# Run 8). Comparing Run 9 vs Run 8 isolates "pretraining domain"; comparing
# both vs Run 7 (VGG19) shows the overall picture.

# Fail fast, before spending any GPU allocation, if the one-time conversion
# hasn't been run yet (login node: `bash scripts/encoders/setup_and_convert.sh imagenet`).
if [ ! -f "$ENCODER_WEIGHTS" ]; then
    echo "ERROR: $ENCODER_WEIGHTS not found."
    echo "Run this on the LOGIN node first: bash scripts/encoders/setup_and_convert.sh imagenet"
    exit 1
fi

SIM=$SCRATCH_FLASH/glomeruli-segnet-$SLURM_JOB_ID
mkdir -p "$SIM"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Sim dir: $SIM"
echo "Dataset: $DATASET"
echo "Encoder weights: $ENCODER_WEIGHTS"
nvidia-smi || true

# Stage code on scratch (skip venv, git, previous models). The converted
# encoder weights and baseline models live under $REPO/models/ (visible from
# compute nodes via $HOME), referenced directly rather than staged.
rsync -a \
    --exclude='.venv' --exclude='.git' --exclude='models' --exclude='__pycache__' \
    "$REPO"/ "$SIM"/

cd "$SIM"

# Activate conda env (no shell init in .bashrc, must source explicitly)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# tensorflow[and-cuda] ships CUDA libs as pip wheels under site-packages/nvidia/*/lib/
# but TF doesn't auto-extend LD_LIBRARY_PATH — do it here so libcudart/cudnn are found.
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

RESULTS=$REPO/models/hpc-run9-$SLURM_JOB_ID
mkdir -p "$RESULTS"

# ---- Run 9: Run 7 recipe (D4 + hflip + stain jitter) with an ImageNet-
# pretrained ResNet50-v1.5 encoder. Identical to Run 8 EXCEPT the encoder
# weights (ImageNet vs SwAV) — isolates the pretraining-domain variable.
python scripts/training/train_hpc.py "$DATASET" \
    --encoder resnet50_swav \
    --encoder-weights "$ENCODER_WEIGHTS" \
    --batch-size 8 \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --flip-horizontal \
    --stain-jitter 0.05 \
    --output-dir "$SIM/models"

# Persist training results IMMEDIATELY, before evaluation.
rsync -a "$SIM/models/" "$RESULTS/"
echo "Training results saved to $RESULTS"

# ---- Three-way test-set evaluation (inference only; defensive) ----
# Run 9 (new: ImageNet ResNet50)
python scripts/evaluation/evaluate.py "$SIM/models/best_model.keras" "$DATASET/test" \
    --batch-size 8 --label "run9-$SLURM_JOB_ID" \
    --output-json "$SIM/models/test_metrics_run9.json" \
    || echo "WARN: Run 9 test evaluation failed"

# Run 8 (SwAV ResNet50 — the direct comparison, same architecture)
if [ -f "$RUN8_MODEL" ]; then
    python scripts/evaluation/evaluate.py "$RUN8_MODEL" "$DATASET/test" \
        --batch-size 8 --label "run8-1811510" \
        --output-json "$SIM/models/test_metrics_run8.json" \
        || echo "WARN: Run 8 evaluation failed"
else
    echo "WARN: Run 8 model not found at $RUN8_MODEL"
fi

# Run 7 (VGG19 reference)
if [ -f "$RUN7_MODEL" ]; then
    python scripts/evaluation/evaluate.py "$RUN7_MODEL" "$DATASET/test" \
        --batch-size 8 --label "run7-1797796" \
        --output-json "$SIM/models/test_metrics_run7.json" \
        || echo "WARN: Run 7 evaluation failed"
else
    echo "WARN: Run 7 model not found at $RUN7_MODEL"
fi

# Persist again to capture the test-metric JSONs alongside the model.
rsync -a "$SIM/models/" "$RESULTS/"
echo "Final results (incl. test metrics) saved to $RESULTS"
