#!/bin/bash
#SBATCH --job-name=segnet_run6
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
RUN4_MODEL=$REPO/models/hpc-run-1742197/best_model.keras   # baseline to also score on test
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

RESULTS=$REPO/models/hpc-run6-$SLURM_JOB_ID
mkdir -p "$RESULTS"

# ---- Run 6: Run 4 recipe + regularization (dropout + brightness jitter) ----
# Only difference vs Run 4 (1742197) is the two regularization knobs below,
# so any change in test IoU is attributable to regularization.
python scripts/training/train_hpc.py "$DATASET" \
    --batch-size 8 \
    --loss-fn combined \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --dropout-rate 0.07 \
    --brightness-delta 0.243 \
    --output-dir "$SIM/models"

# Persist training results IMMEDIATELY, before evaluation, so an eval hiccup
# can never cost us the (expensive) trained model.
rsync -a "$SIM/models/" "$RESULTS/"
echo "Training results saved to $RESULTS"

# ---- Test-set evaluation (inference only; defensive so it can't abort the job) ----
# New Run 6 model
python scripts/evaluation/evaluate.py "$SIM/models/best_model.keras" "$DATASET/test" \
    --batch-size 8 --label "run6-$SLURM_JOB_ID" \
    --output-json "$SIM/models/test_metrics_run6.json" \
    || echo "WARN: Run 6 test evaluation failed"

# Run 4 baseline model (lives under $HOME, not staged to scratch)
if [ -f "$RUN4_MODEL" ]; then
    python scripts/evaluation/evaluate.py "$RUN4_MODEL" "$DATASET/test" \
        --batch-size 8 --label "run4-1742197" \
        --output-json "$SIM/models/test_metrics_run4.json" \
        || echo "WARN: Run 4 baseline evaluation failed"
else
    echo "WARN: Run 4 model not found at $RUN4_MODEL — skipping baseline eval"
fi

# Persist again to capture the test-metric JSONs alongside the model.
rsync -a "$SIM/models/" "$RESULTS/"
echo "Final results (incl. test metrics) saved to $RESULTS"
