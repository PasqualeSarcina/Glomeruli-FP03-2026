#!/bin/bash
#SBATCH --job-name=segnet_run11_bt
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
DATASET=$SCRATCH_FLASH/glomeruli-dataset
ENCODER_WEIGHTS=$REPO/models/encoders/barlowtwins_resnet50.weights.h5   # from setup_and_convert.sh barlowtwins
RUN9_MODEL=$REPO/models/hpc-run9-1815511/best_model.keras               # ImageNet ResNet50 (same arch — direct control)
RUN7_MODEL=$REPO/models/hpc-run7-1797796/best_model.keras               # VGG19 reference
# --------------------------------

# Run 11 = Lunit Barlow Twins pathology ResNet50 encoder. Same as Run 8/9/10
# except the encoder weights — third pathology SSL method (feature
# decorrelation) in the same ResNet50 family.

if [ ! -f "$ENCODER_WEIGHTS" ]; then
    echo "ERROR: $ENCODER_WEIGHTS not found."
    echo "Run this on the LOGIN node first: bash scripts/encoders/setup_and_convert.sh barlowtwins"
    exit 1
fi

SIM=$SCRATCH_FLASH/glomeruli-segnet-$SLURM_JOB_ID
mkdir -p "$SIM"
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Encoder weights: $ENCODER_WEIGHTS"
nvidia-smi || true

rsync -a --exclude='.venv' --exclude='.git' --exclude='models' --exclude='__pycache__' "$REPO"/ "$SIM"/
cd "$SIM"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

RESULTS=$REPO/models/hpc-run11-$SLURM_JOB_ID
mkdir -p "$RESULTS"

python scripts/training/train_hpc.py "$DATASET" \
    --encoder resnet50_swav \
    --encoder-weights "$ENCODER_WEIGHTS" \
    --batch-size 8 \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --flip-horizontal \
    --stain-jitter 0.05 \
    --output-dir "$SIM/models"

rsync -a "$SIM/models/" "$RESULTS/"
echo "Training results saved to $RESULTS"

# ---- Three-way test eval: Run 11 vs Run 9 (ImageNet, same arch) vs Run 7 (reference) ----
python scripts/evaluation/evaluate.py "$SIM/models/best_model.keras" "$DATASET/test" \
    --batch-size 8 --label "run11-bt-$SLURM_JOB_ID" \
    --output-json "$SIM/models/test_metrics_run11.json" || echo "WARN: Run 11 eval failed"

if [ -f "$RUN9_MODEL" ]; then
    python scripts/evaluation/evaluate.py "$RUN9_MODEL" "$DATASET/test" \
        --batch-size 8 --label "run9-imagenet" \
        --output-json "$SIM/models/test_metrics_run9.json" || echo "WARN: Run 9 eval failed"
fi
if [ -f "$RUN7_MODEL" ]; then
    python scripts/evaluation/evaluate.py "$RUN7_MODEL" "$DATASET/test" \
        --batch-size 8 --label "run7-vgg19" \
        --output-json "$SIM/models/test_metrics_run7.json" || echo "WARN: Run 7 eval failed"
fi

rsync -a "$SIM/models/" "$RESULTS/"
echo "Final results (incl. test metrics) saved to $RESULTS"
