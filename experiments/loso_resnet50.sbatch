#!/bin/bash
#SBATCH --job-name=loso_rn50
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s348829@studenti.polito.it
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# =============================================================================
# TensorFlow LOSO with a ResNet50 encoder + CONVERTED pathology weights.
# The TF-native counterpart of the PyTorch DeepLabV3+/Lunit experiment (exam
# requires TF). Architecture here is SegNet-ResNet50 (TF has no DeepLabV3+),
# so numbers are NOT directly comparable to the torch DeepLabV3+ run — they ARE
# comparable to the TF SegNet-VGG19/ImageNet LOSO (0.702 ± 0.094).
#
# PREREQUISITE (run once on the LOGIN node — it needs torch+internet, GPU nodes
# have neither):
#   bash scripts/setup_and_convert.sh barlowtwins   # -> models/encoders/barlowtwins_resnet50.weights.h5
#   (or 'imagenet' for the ResNet50 ImageNet control)
#
# Pick the encoder weights with WEIGHTS=... :
#   Barlow Twins (default):  sbatch experiments/loso_resnet50.sbatch
#   ImageNet control:        sbatch --export=ALL,WEIGHTS=$HOME/Glomeruli-FP03-2026/models/encoders/imagenet_resnet50.weights.h5 experiments/loso_resnet50.sbatch
# =============================================================================
set -euo pipefail

REPO=$HOME/Glomeruli-FP03-2026
SLIDES_DIR=${SLIDES_DIR:-$HOME/slides}
DATA=${DATA:-$REPO/data/dataset}
WEIGHTS=${WEIGHTS:-$REPO/models/encoders/barlowtwins_resnet50.weights.h5}

cd "$REPO"
nvidia-smi || true
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate glomeruli

# preprocessing if the patch dataset is missing
if [ ! -d "$DATA/train/img" ] || [ -z "$(ls -A "$DATA/train/img" 2>/dev/null)" ]; then
    echo "===== preprocessing patches ====="
    python scripts/preprocess_data.py "$SLIDES_DIR"
fi

# converted encoder weights must exist (produced offline by setup_and_convert.sh)
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: converted encoder weights not found: $WEIGHTS" >&2
    echo "       Run first on the LOGIN node:" >&2
    echo "       bash scripts/setup_and_convert.sh barlowtwins" >&2
    exit 1
fi

# TF needs its bundled CUDA libs on LD_LIBRARY_PATH
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
python -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"

echo "===== LOSO: SegNet-ResNet50, encoder weights = $WEIGHTS ====="
python scripts/loso_cv.py "$DATA" \
    --encoder resnet50 --encoder-weights "$WEIGHTS" \
    --stain-jitter 0.05 --tta \
    --output-dir "$REPO/models/loso-rn50-$SLURM_JOB_ID"

echo "done -> $REPO/models/loso-rn50-$SLURM_JOB_ID/loso_results.json"
