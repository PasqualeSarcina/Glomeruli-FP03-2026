#!/bin/bash
#SBATCH --job-name=segnet_eval
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s348829@studenti.polito.it
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

REPO=$HOME/Glomeruli-FP03-2026
CONDA_ENV=glomeruli
DATASET=$SCRATCH_FLASH/glomeruli-dataset

RUN4_MODEL=$REPO/models/hpc-run-1742197/best_model.keras
RUN6_MODEL=$REPO/models/hpc-run6-1746763/best_model.keras
RESULTS=$REPO/models/eval-run4-run6

mkdir -p "$RESULTS"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

python "$REPO/scripts/evaluate.py" "$RUN4_MODEL" "$DATASET/test" \
    --batch-size 8 --label "run4-1742197" \
    --output-json "$RESULTS/test_metrics_run4.json"

python "$REPO/scripts/evaluate.py" "$RUN6_MODEL" "$DATASET/test" \
    --batch-size 8 --label "run6-1746763" \
    --output-json "$RESULTS/test_metrics_run6.json"

echo "Evaluation complete. Results in $RESULTS"
