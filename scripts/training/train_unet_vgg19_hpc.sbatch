#!/bin/bash

#SBATCH --job-name=unet_vgg19
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-unet-%j.out
#SBATCH --error=slurm-unet-%j.err

set -euo pipefail

# ============================================================
# CHANGE THESE THREE VALUES IF NECESSARY
# ============================================================

REPO="$HOME/Glomeruli-FP03-2026"
CONDA_ENV="glomeruli"
DATASET="$SCRATCH_FLASH/glomeruli-dataset"

# ============================================================

# Temporary job directory on fast scratch storage.
SIM="$SCRATCH_FLASH/glomeruli-unet-$SLURM_JOB_ID"

mkdir -p "$SIM"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO"
echo "Dataset: $DATASET"
echo "Working directory: $SIM"
echo "=========================================="

# Confirm that the dataset exists.
if [[ ! -d "$DATASET/train" ]]; then
    echo "ERROR: $DATASET/train does not exist."
    exit 1
fi

if [[ ! -d "$DATASET/validation" ]]; then
    echo "ERROR: $DATASET/validation does not exist."
    exit 1
fi

# Display allocated GPU information.
nvidia-smi || true

# Copy the repository to fast scratch storage.
rsync -a \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='models' \
    --exclude='__pycache__' \
    "$REPO/" "$SIM/"

cd "$SIM"

# Activate Conda.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# TensorFlow CUDA libraries installed through pip can be located here.
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

# Verify that TensorFlow can see the GPU.
python -c "
import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('Available GPUs:', tf.config.list_physical_devices('GPU'))

if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('TensorFlow cannot see the allocated GPU.')
"

# Run U-Net VGG19 training.
python scripts/training/train_unet_vgg19_colab.py "$DATASET" \
    --batch-size 4 \
    --loss-fn combined \
    --phase-1-epochs 10 \
    --phase-1-lr 0.01 \
    --phase-2-epochs 20 \
    --phase-2-lr 0.001 \
    --flip-horizontal \
    --output-dir "$SIM/models/unet_vgg19"

# Copy results from temporary scratch back to the repository.
RESULTS="$REPO/models/unet-vgg19-hpc-$SLURM_JOB_ID"

mkdir -p "$RESULTS"

rsync -a \
    "$SIM/models/unet_vgg19/" \
    "$RESULTS/"

echo "=========================================="
echo "Training completed."
echo "Results saved to:"
echo "$RESULTS"
echo "=========================================="