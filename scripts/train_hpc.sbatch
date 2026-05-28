#!/bin/bash
#SBATCH --job-name=segnet_train
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=REPLACE_WITH_YOUR_EMAIL@studenti.polito.it
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---- adjust these if needed ----
REPO=$HOME/Glomeruli-FP03-2026
CONDA_ENV=glomeruli
DATASET=$SCRATCH_FLASH/glomeruli-dataset   # pre-extracted patches (train/ validation/ test/)
# --------------------------------

SIM=$SCRATCH_FLASH/glomeruli-segnet-$SLURM_JOB_ID
mkdir -p "$SIM"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Sim dir: $SIM"
echo "Dataset: $DATASET"
nvidia-smi || true

# Stage code on scratch (skip venv, git, previous models)
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

python scripts/train_hpc.py "$DATASET" \
    --batch-size 8 \
    --loss-fn combined \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --output-dir "$SIM/models"

# Persist results back to $HOME (scratch is auto-cleaned after 30 days)
RESULTS=$REPO/models/hpc-run-$SLURM_JOB_ID
mkdir -p "$RESULTS"
rsync -a "$SIM/models/" "$RESULTS/"
echo "Results saved to $RESULTS"
