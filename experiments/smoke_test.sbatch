#!/bin/bash
#SBATCH --job-name=fp03_smoke
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# =============================================================================
# FP03 — SMOKE TEST (does the delivered repo actually run?)
#
# Runs every stage of full_pipeline.sbatch at the smallest size that still
# executes real code: 2 slides, 1+1 epochs, 1 backbone. It answers "does the
# merged repo import and run end to end", NOT "how good is the model" — the
# reported numbers here are meaningless and must never enter the report, whose
# figures come from the 9-fold LOSO runs.
#
# Deliberately does NOT use `set -e`: every stage runs even if an earlier one
# fails, so one submission reports ALL breakages instead of only the first.
# Exit code is non-zero if any stage failed, and the summary lists which.
#
#   sbatch experiments/smoke_test.sbatch
#   sbatch --export=ALL,SLIDES_DIR=/path/to/slides experiments/smoke_test.sbatch
# =============================================================================

set -uo pipefail

REPO=$HOME/Glomeruli-FP03-2026
CONDA_ENV=glomeruli
SLIDES_DIR=${SLIDES_DIR:-$HOME/slides}
N_SLIDES=${N_SLIDES:-2}
BACKBONE=${BACKBONE:-mobilenet}      # the one selected in the report; also the fastest

WORK=${SCRATCH_FLASH:-/tmp}/fp03-smoke-$SLURM_JOB_ID
SUBSET=$WORK/slides
RESULTS=$REPO/models/smoke-$SLURM_JOB_ID

FAILED=()
run_stage() {           # run_stage <name> <command...>
    local name="$1"; shift
    echo ""; echo "===== $name ====="
    # 2>&1: keep each stage's traceback next to its own header in one file.
    if "$@" 2>&1; then
        echo "-- OK: $name"
    else
        # stdout, not stderr: the marker must land in the same stream as the
        # stage output that caused it, otherwise cause and symptom end up in
        # two different files and are impossible to correlate.
        echo "-- FAIL: $name (continuing, so the rest is still tested)"
        FAILED+=("$name")
    fi
}

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

# --------------------------------------------------------------- slide subset --
# Preprocessing every slide would dominate the runtime; N_SLIDES is enough to
# exercise the same code paths (multi-slide split included).
mkdir -p "$SUBSET"
n=0
for svs in "$SLIDES_DIR"/*.svs; do
    [ -e "$svs" ] || { echo "ERROR: no .svs in $SLIDES_DIR" >&2; exit 1; }
    stem=$(basename "$svs" .svs)
    ln -sf "$svs" "$SUBSET/$stem.svs"
    [ -f "$SLIDES_DIR/$stem.xml" ] && ln -sf "$SLIDES_DIR/$stem.xml" "$SUBSET/$stem.xml"
    n=$((n + 1)); [ "$n" -ge "$N_SLIDES" ] && break
done
echo "Smoke subset: $n slide(s) in $SUBSET"

# ----------------------------------------------------------------- stage repo --
mkdir -p "$WORK/repo"
rsync -a --exclude='.venv' --exclude='.git' --exclude='models' \
      --exclude='__pycache__' --exclude='data/dataset' --exclude='data/glomeruli' \
      "$REPO"/ "$WORK/repo"/
cd "$WORK/repo"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

# igraph + leidenalg are needed by src/clustering/leiden/fit.py. They were missing
# from this list (and from requirements.txt) until job 1891711, which is why
# Leiden had failed here and in the earlier full-pipeline run 1843596.
python -m pip install --quiet --upgrade-strategy only-if-needed \
    hdbscan umap-learn scikit-learn pandas matplotlib tqdm igraph leidenalg "numpy~=2.4.6" \
    || echo "WARN: clustering deps did not resolve; those stages will fail below."

run_stage "import check" python -c "
import tensorflow as tf, keras
from src.segmentation.segnet import build_segnet_vgg19, build_segnet_resnet50, combined_loss
from src.segmentation.dataset import SegmentationDataset
from src.data.extract_masks import extract_tissue_mask, parse_xml_annotations
print('TF', tf.__version__, '| GPU', tf.config.list_physical_devices('GPU'))
print('project imports OK')
"

mkdir -p "$RESULTS"
DATASET="$WORK/repo/data/dataset"
GLOM="$WORK/repo/data/glomeruli"

run_stage "Stage 1 preprocessing" \
    python scripts/preprocess_data.py "$SUBSET" --annotations-path "$SUBSET"

for split in train validation test; do
    echo "  $split: $(find "$DATASET/$split/img" -name '*.png' 2>/dev/null | wc -l) patches"
done

run_stage "Stage 2 training (1+1 epochs)" \
    python scripts/train_segmentation.py "$DATASET" \
        --batch-size 8 --phase-1-epochs 1 --phase-2-epochs 1 \
        --flip-horizontal --stain-jitter 0.05 \
        --output-dir "$RESULTS/segmentation"

run_stage "Stage 2 evaluation" \
    python scripts/evaluate_segmentation.py \
        "$RESULTS/segmentation/best_model.keras" "$DATASET/test" \
        --tta --label "SMOKE (meaningless numbers)" \
        --output-json "$RESULTS/segmentation/test_metrics.json"

run_stage "Bridge crop extraction" \
    python scripts/extract_glomeruli_from_annotations.py "$SUBSET" --annotations-path "$SUBSET"
echo "  crops: $(find "$GLOM/crops" -name '*.png' 2>/dev/null | wc -l)"

run_stage "Stage 3 embeddings ($BACKBONE)" \
    python scripts/extract_glomeruli_embeds.py "$BACKBONE" "$GLOM"

STEM=$(ls "$GLOM"/embeddings/*.npy 2>/dev/null | head -1)
STEM="${STEM%.npy}"
if [ -n "$STEM" ]; then
    run_stage "Stage 3 HDBSCAN" python scripts/cluster_hdbscan.py "$STEM"
    run_stage "Stage 3 Leiden"  python scripts/cluster_leiden.py  "$STEM"
else
    echo "-- FAIL: no embeddings produced, clustering not reachable" >&2
    FAILED+=("Stage 3 clustering (no embeddings)")
fi

# LOSO needs >=2 slides; 1 fold per slide, 1+1 epochs — just proves it runs.
run_stage "LOSO harness" \
    python scripts/loso_cv.py "$DATASET" \
        --phase-1-epochs 1 --phase-2-epochs 1 \
        --output-dir "$RESULTS/loso"

echo ""; echo "================ SMOKE TEST SUMMARY ================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL STAGES PASSED — the repo runs end to end."
    echo "(Metrics above are from 1 epoch on $n slides: meaningless by design.)"
    exit 0
fi
echo "${#FAILED[@]} stage(s) FAILED:"
printf '  - %s\n' "${FAILED[@]}"
exit 1
