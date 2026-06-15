#!/bin/bash
#SBATCH --job-name=segnet_optuna
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s348829@studenti.polito.it
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---- adjust these if needed ----
REPO=$HOME/Glomeruli-FP03-2026
CONDA_ENV=glomeruli
DATASET=$SCRATCH_FLASH/glomeruli-dataset   # pre-extracted patches (train/ validation/ test/)
N_TRIALS=15
# --------------------------------

SIM=$SCRATCH_FLASH/glomeruli-optuna-$SLURM_JOB_ID
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

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

# Install optuna if not already present (fast no-op if cached)
pip install optuna --quiet

python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

DB_PATH="$SIM/optuna_run5.db"

python scripts/train_optuna.py "$DATASET" \
    --batch-size 8 \
    --n-trials "$N_TRIALS" \
    --storage "sqlite:///$DB_PATH" \
    --output-dir "$SIM/models/optuna"

# Persist results back to $HOME (scratch is auto-cleaned after 30 days)
RESULTS=$REPO/models/optuna-run5-$SLURM_JOB_ID
mkdir -p "$RESULTS"
rsync -a "$SIM/models/optuna/" "$RESULTS/"
cp "$DB_PATH" "$RESULTS/optuna_run5.db"
echo "Results saved to $RESULTS"
