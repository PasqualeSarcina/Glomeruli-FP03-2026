#!/bin/bash
#SBATCH --job-name=segnet_run8_swav
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
ENCODER_WEIGHTS=$REPO/models/encoders/swav_resnet50.weights.h5   # from scripts/encoders/setup_and_convert.sh
RUN7_MODEL=$REPO/models/hpc-run7-1797796/best_model.keras        # current best, to also score on test
# --------------------------------

# Fail fast, before spending any GPU allocation, if the one-time conversion
# step hasn't been run yet. This must happen on the login node beforehand
# (see scripts/encoders/setup_and_convert.sh) — compute nodes typically
# can't reach GitHub to download the checkpoint.
if [ ! -f "$ENCODER_WEIGHTS" ]; then
    echo "ERROR: $ENCODER_WEIGHTS not found."
    echo "Run this on the LOGIN node first: bash scripts/encoders/setup_and_convert.sh"
    exit 1
fi

SIM=$SCRATCH_FLASH/glomeruli-segnet-$SLURM_JOB_ID
mkdir -p "$SIM"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Sim dir: $SIM"
echo "Dataset: $DATASET"
echo "Encoder weights: $ENCODER_WEIGHTS"
nvidia-smi || true

# Stage code on scratch (skip venv, git, previous models). Note the
# converted encoder weights live under $REPO/models/encoders/, not on
# scratch — same pattern as RUN4_MODEL in train_run6.sbatch/train_run7.sbatch,
# referenced directly from $REPO since $HOME is visible from compute nodes too.
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

RESULTS=$REPO/models/hpc-run8-$SLURM_JOB_ID
mkdir -p "$RESULTS"

# ---- Run 8: Run 7 recipe (D4 rotations + hflip + HED stain jitter), only
# the encoder changes: pathology-pretrained ResNet50 (SwAV, Lunit) instead
# of ImageNet-pretrained VGG19. Same decoder, same loss, same two-phase
# schedule, same batch size — isolates the encoder-pretraining variable
# exactly like Run 7 isolated the augmentation variable against Run 4.
python scripts/training/train_hpc.py "$DATASET" \
    --encoder resnet50_swav \
    --encoder-weights "$ENCODER_WEIGHTS" \
    --batch-size 8 \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --flip-horizontal \
    --stain-jitter 0.05 \
    --output-dir "$SIM/models"

# Persist training results IMMEDIATELY, before evaluation, so an eval hiccup
# can never cost us the (expensive) trained model.
rsync -a "$SIM/models/" "$RESULTS/"
echo "Training results saved to $RESULTS"

# ---- Test-set evaluation (inference only; defensive so it can't abort the job) ----
# New Run 8 model
python scripts/evaluation/evaluate.py "$SIM/models/best_model.keras" "$DATASET/test" \
    --batch-size 8 --label "run8-$SLURM_JOB_ID" \
    --output-json "$SIM/models/test_metrics_run8.json" \
    || echo "WARN: Run 8 test evaluation failed"

# Run 7 baseline model (current best, lives under $HOME, not staged to scratch)
if [ -f "$RUN7_MODEL" ]; then
    python scripts/evaluation/evaluate.py "$RUN7_MODEL" "$DATASET/test" \
        --batch-size 8 --label "run7-1797796" \
        --output-json "$SIM/models/test_metrics_run7.json" \
        || echo "WARN: Run 7 baseline evaluation failed"
else
    echo "WARN: Run 7 model not found at $RUN7_MODEL — skipping baseline eval"
fi

# Persist again to capture the test-metric JSONs alongside the model.
rsync -a "$SIM/models/" "$RESULTS/"
echo "Final results (incl. test metrics) saved to $RESULTS"
