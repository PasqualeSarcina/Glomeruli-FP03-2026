#!/bin/bash
#SBATCH --job-name=segnet_loso
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

REPO=$HOME/Glomeruli-FP03-2026
CONDA_ENV=glomeruli
DATASET=$SCRATCH_FLASH/glomeruli-dataset

# Leave-one-slide-out CV of the Run 7 recipe (VGG19 ImageNet + D4 + stain jitter),
# fixed-budget training (no validation / no EarlyStopping), 9 folds. ~40 min/fold
# on A40 -> ~6h total, fits the 12h walltime.

SIM=$SCRATCH_FLASH/glomeruli-segnet-$SLURM_JOB_ID
mkdir -p "$SIM"
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

rsync -a --exclude='.venv' --exclude='.git' --exclude='models' --exclude='__pycache__' "$REPO"/ "$SIM"/
cd "$SIM"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

RESULTS=$REPO/models/loso-$SLURM_JOB_ID
mkdir -p "$RESULTS"

python scripts/training/loso_cv.py "$DATASET" \
    --batch-size 8 \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --stain-jitter 0.05 \
    --output-dir "$SIM/models/loso"

rsync -a "$SIM/models/loso/" "$RESULTS/"
echo "LOSO results saved to $RESULTS"
