#!/bin/bash

#SBATCH --job-name=unet_optuna
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-unet-optuna-%j.out
#SBATCH --error=slurm-unet-optuna-%j.err

set -euo pipefail

# Change these if your paths or environment name differ.
REPO="$HOME/Glomeruli-FP03-2026"
CONDA_ENV="glomeruli"
DATASET="$SCRATCH_FLASH/glomeruli-dataset"

# Permanent Optuna results.
RESULTS="$REPO/models/unet-vgg19-optuna"

# Temporary copy of the source code.
SIM="$SCRATCH_FLASH/unet-optuna-$SLURM_JOB_ID"

mkdir -p "$SIM"
mkdir -p "$RESULTS"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO"
echo "Dataset: $DATASET"
echo "Results: $RESULTS"
echo "Temporary directory: $SIM"

if [[ ! -d "$DATASET/train" ]]; then
    echo "ERROR: Missing $DATASET/train"
    exit 1
fi

if [[ ! -d "$DATASET/validation" ]]; then
    echo "ERROR: Missing $DATASET/validation"
    exit 1
fi

nvidia-smi || true

# Copy only the source code to fast storage.
rsync -a \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='models' \
    --exclude='__pycache__' \
    "$REPO/" "$SIM/"

cd "$SIM"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "Environment: $CONDA_DEFAULT_ENV"

# Make pip-installed NVIDIA libraries visible to TensorFlow.
PYTHON_VERSION="$(
    python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"

NVIDIA_DIR="$CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/nvidia"

if [[ -d "$NVIDIA_DIR" ]]; then
    NVIDIA_LIBRARY_PATH="$(
        find "$NVIDIA_DIR" -name lib -type d -print0 |
        xargs -0 -r printf '%s:'
    )"

    export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}${LD_LIBRARY_PATH:-}"
fi

python -c "
import tensorflow as tf
import optuna

print('TensorFlow:', tf.__version__)
print('Optuna:', optuna.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))

if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('TensorFlow cannot see the allocated GPU.')
"

# Keep the Optuna database in permanent storage.
STORAGE="sqlite:///$RESULTS/optuna.db"

python scripts/training/train_unet_vgg19_optuna_colab.py "$DATASET" \
    --batch-size 4 \
    --n-trials 15 \
    --study-name "unet_vgg19_optuna" \
    --storage "$STORAGE" \
    --output-dir "$RESULTS"

echo "Optuna search completed."
echo "Results saved to: $RESULTS"