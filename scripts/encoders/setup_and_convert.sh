#!/bin/bash
# Run this on the LOGIN node (not via sbatch) before submitting a resnet50
# training job. Reason: it needs internet (download the pretrained weights,
# pip-install torch/tensorflow) and GPU compute nodes on this cluster have
# none — only the login node does. Same login-vs-compute split as
# roadmap-hpc-training.md's "Phase 0 prep" vs "Phase 1 HPC session".
#
# Usage (arg = pretraining source, default swav):
#   bash scripts/encoders/setup_and_convert.sh swav         # Lunit SwAV pathology (Run 8)
#   bash scripts/encoders/setup_and_convert.sh imagenet     # torchvision ImageNet control (Run 9)
#   bash scripts/encoders/setup_and_convert.sh mocov2       # Lunit MoCoV2 pathology (Run 10)
#   bash scripts/encoders/setup_and_convert.sh barlowtwins  # Lunit Barlow Twins pathology (Run 11)
#
# What this does:
#   1. Creates a throwaway Python venv, separate from the `glomeruli` conda
#      env used for GPU training — it never touches TensorFlow's CUDA setup,
#      so it can't break anything Run 1-8 depend on.
#   2. Installs CPU-only torch/torchvision + CPU tensorflow (just to build the
#      Keras graph and copy weights; no training happens here).
#   3. Runs convert_swav_resnet50.py for the chosen --source, which loads the
#      pretrained torch weights, transplants them into the hand-built Keras
#      ResNet50-v1.5, and VERIFIES the conversion numerically before saving.
#
# Safe to re-run: every step is idempotent (skips work already done).

set -euo pipefail

SOURCE="${1:-swav}"   # swav | mocov2 | barlowtwins (Lunit pathology) | imagenet (control)
case "$SOURCE" in
    swav|mocov2|barlowtwins|retccl|imagenet) OUTPUT_NAME="${SOURCE}_resnet50.weights.h5" ;;
    *) echo "Unknown source '$SOURCE' — use swav | mocov2 | barlowtwins | retccl | imagenet."; exit 1 ;;
esac

REPO="$HOME/Glomeruli-FP03-2026"
VENV="$HOME/venvs/convert-swav"
OUTPUT="$REPO/models/encoders/$OUTPUT_NAME"

cd "$REPO"

if [ ! -d "$VENV" ]; then
    echo "Creating conversion venv at $VENV"
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

echo "Installing conversion dependencies (CPU-only, no CUDA needed)..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --quiet tensorflow numpy

echo ""
echo "Running conversion (source=$SOURCE)..."
# NOTE: do NOT pass --checkpoint-cache here. The convert script derives a
# per-source cache path (models/encoders/<source>_rn50.torch) so each source
# downloads ITS OWN checkpoint. A fixed cache path would make every source
# reuse whatever is already cached (this caused mocov2/barlowtwins/retccl to
# silently convert the SwAV weights). For imagenet the flag is unused anyway.
python scripts/encoders/convert_swav_resnet50.py \
    --source "$SOURCE" \
    --output "$OUTPUT"

deactivate

echo ""
echo "Done. Encoder weights ready at $OUTPUT"
