#!/bin/bash
#SBATCH --job-name=fp03_full_pipeline
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

# =============================================================================
# FP03 — FULL END-TO-END PIPELINE
#
# Runs every stage of the project in order, on real data:
#   Stage 1  Preprocessing        preprocess_data.py              (needs .svs)
#   Stage 2  Segmentation train   train_segmentation.py           -> best_model.keras
#            Segmentation eval    evaluate_segmentation.py        -> test IoU (+TTA)
#   Bridge   Glomeruli crops      extract_glomeruli_from_annotations.py (needs .svs)
#   Stage 3  Embeddings           extract_glomeruli_embeds.py     (8 backbones)
#            Clustering           cluster_hdbscan.py / cluster_leiden.py
#
# The backbone and clustering evaluations are NOT part of this job: they live in
# notebooks/backbone_evaluation.ipynb and notebooks/clustering_evaluation.ipynb,
# and are run locally on the embeddings and clustering runs this job produces.
#
# This is a PRODUCTION run (30 training epochs, all 8 backbones, full grids),
# not a smoke test. Expect many hours; walltime is set to 24h.
#
# PREREQUISITES (checked in the preflight below; the job aborts early, before
# consuming GPU hours, if a hard one is missing):
#   * SLIDES_DIR must contain the raw .svs slides AND their .xml annotations
#     (Stage 1 and the crop extraction both read the slides directly).
#   * data/checkpoints/KimiaNetKerasWeights.h5 is needed ONLY for the kimianet
#     backbone; if absent, that one backbone is skipped (soft) and the rest run.
#   * The DINOv2/v3 backbones download their presets from the internet via
#     keras-hub on first use (compute nodes have network access — confirmed in
#     earlier runs).
# =============================================================================

set -euo pipefail

# ------------------------------------------------------------------ config ---
REPO=$HOME/Glomeruli-FP03-2026
CONDA_ENV=glomeruli

# ---> SET THIS: directory holding the raw .svs slides + .xml annotations. <---
#      There is no confirmed on-cluster location yet, so this is a placeholder.
#      If annotations live elsewhere, also set SLIDES_ANNOTATIONS.
SLIDES_DIR=${SLIDES_DIR:-$HOME/slides}
SLIDES_ANNOTATIONS=${SLIDES_ANNOTATIONS:-$SLIDES_DIR}

# Backbones to run for the embedding/clustering stage. Trim this list to make
# the run shorter. dinov2/dinov3 additionally need a --mode (set below).
BACKBONES=(nasnet densenet169 densenet201 mobilenet xception kimianet dinov2 dinov3)

SIM=$SCRATCH_FLASH/fp03-pipeline-$SLURM_JOB_ID
RESULTS=$REPO/models/pipeline-$SLURM_JOB_ID   # pulled back to $HOME at the end

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Slides dir: $SLIDES_DIR"
nvidia-smi || true

# ---------------------------------------------------------------- preflight --
# Fail fast, before staging or touching the GPU, if the root input is missing.
shopt -s nullglob
svs_files=("$SLIDES_DIR"/*.svs)
shopt -u nullglob
if [ ${#svs_files[@]} -eq 0 ]; then
    echo "ERROR: no .svs slides found in '$SLIDES_DIR'." >&2
    echo "       Set SLIDES_DIR to the real slides location and resubmit, e.g." >&2
    echo "       sbatch --export=ALL,SLIDES_DIR=/path/to/slides experiments/full_pipeline.sbatch" >&2
    echo "       (The whole pipeline starts from the .svs — Stage 1 and the crop" >&2
    echo "        extraction cannot run without them.)" >&2
    exit 1
fi
echo "Preflight OK: ${#svs_files[@]} .svs slide(s) found."

# --------------------------------------------------------------- stage repo --
mkdir -p "$SIM"
rsync -a --exclude='.venv' --exclude='.git' --exclude='models' \
      --exclude='__pycache__' --exclude='data/dataset' --exclude='data/glomeruli' \
      "$REPO"/ "$SIM"/
cd "$SIM"

# ----------------------------------------------------------------- conda env --
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# The base 'glomeruli' env has TF 2.21 (GPU) + numpy/scipy/skimage/openslide.
# Stage 3 only needs the clustering deps below — install just those.
#
# Two dependency traps this guards against (both would otherwise waste a run):
#  1. keras-hub (needed ONLY by DINOv2/v3) pulls tensorflow-text, which caps
#     tensorflow to <2.21 → conflicts with this env's TF 2.21 GPU build. That
#     conflict aborted job 1841816. So keras-hub is NOT installed; the 2 DINO
#     backbones are skipped (their import fails and hits the WARN guard below),
#     the 6 CNN backbones + all clustering still run.
#  2. umap-learn pulls numba, whose numpy ceiling can be below this env's numpy
#     2.4. To stop pip from DOWNGRADING numpy (which would break TF and kill
#     even the segmentation), we pin numpy to its current 2.4 line. If the
#     clustering deps can't resolve under that pin, pip installs nothing and we
#     fall through (|| echo) with TF intact — segmentation is never sacrificed
#     to a Stage-3 dependency problem.
# Validate the resolution WITHOUT a queue slot first (login node, env active):
#     pip install --dry-run hdbscan umap-learn scikit-learn pandas matplotlib tqdm "numpy~=2.4.6"
#  3. igraph + leidenalg are imported by src/clustering/leiden/fit.py. They were
#     absent from this list, which is why Leiden failed on every backbone in run
#     1843596 while HDBSCAN succeeded.
python -m pip install --quiet --upgrade-strategy only-if-needed \
    hdbscan umap-learn scikit-learn pandas matplotlib tqdm igraph leidenalg "numpy~=2.4.6" \
    || echo "WARN: Stage-3 clustering deps did not resolve — TF/segmentation unaffected; embedding/clustering stages may be skipped."

# TF needs its bundled CUDA libs on LD_LIBRARY_PATH (they aren't auto-discovered).
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$(find "$NVIDIA_DIR" -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

mkdir -p "$RESULTS"

# =============================================================================
# STAGE 1 — Preprocessing: .svs -> patch dataset (data/dataset/{train,val,test})
# =============================================================================
echo ""; echo "===== STAGE 1: preprocessing ====="
python scripts/preprocess_data.py "$SLIDES_DIR" \
    --annotations-path "$SLIDES_ANNOTATIONS"

DATASET="$SIM/data/dataset"
for split in train validation test; do
    n=$(find "$DATASET/$split/img" -name '*.png' 2>/dev/null | wc -l)
    echo "  $split: $n patches"
done

# =============================================================================
# STAGE 2 — Segmentation: two-phase training (Run 7 recipe) + test-set eval
# =============================================================================
echo ""; echo "===== STAGE 2: segmentation training ====="
python scripts/train_segmentation.py "$DATASET" \
    --batch-size 8 \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --flip-horizontal --stain-jitter 0.05 \
    --output-dir "$RESULTS/segmentation"

echo ""; echo "===== STAGE 2: segmentation evaluation (with TTA) ====="
python scripts/evaluate_segmentation.py \
    "$RESULTS/segmentation/best_model.keras" "$DATASET/test" \
    --tta --label "full-pipeline SegNet-VGG19" \
    --output-json "$RESULTS/segmentation/test_metrics.json"

# =============================================================================
# BRIDGE — Extract individual glomerulus crops from the annotations
#          (-> data/glomeruli/{crops,masks}) for the unsupervised stage.
# =============================================================================
echo ""; echo "===== BRIDGE: glomeruli crop extraction ====="
python scripts/extract_glomeruli_from_annotations.py "$SLIDES_DIR" \
    --annotations-path "$SLIDES_ANNOTATIONS"

GLOM="$SIM/data/glomeruli"
n_crops=$(find "$GLOM/crops" -name '*.png' 2>/dev/null | wc -l)
echo "  extracted $n_crops glomerulus crops"

# =============================================================================
# STAGE 3 — Embeddings (per backbone) + clustering + backbone comparison
# =============================================================================
echo ""; echo "===== STAGE 3: embedding extraction ====="
KIMIA_WEIGHTS="$SIM/data/checkpoints/KimiaNetKerasWeights.h5"
for bb in "${BACKBONES[@]}"; do
    echo "--- backbone: $bb ---"
    case "$bb" in
        kimianet)
            if [ ! -f "$KIMIA_WEIGHTS" ]; then
                echo "  SKIP kimianet: weights not found at $KIMIA_WEIGHTS"
                continue
            fi
            python scripts/extract_glomeruli_embeds.py kimianet "$GLOM" || echo "  WARN: kimianet failed, continuing"
            ;;
        dinov2)
            # input-size must be a multiple of 14 for DINOv2 (224 = 16*14).
            python scripts/extract_glomeruli_embeds.py dinov2 "$GLOM" \
                --mode cls --backbone-size base --input-size 224 || echo "  WARN: dinov2 failed, continuing"
            ;;
        dinov3)
            # input-size must be a multiple of the patch size (base -> 16; 224 ok).
            python scripts/extract_glomeruli_embeds.py dinov3 "$GLOM" \
                --mode cls --backbone-size base --input-size 224 || echo "  WARN: dinov3 failed, continuing"
            ;;
        *)
            # nasnet / densenet169 / densenet201 / mobilenet / xception:
            # no --mode, default input size recommended.
            python scripts/extract_glomeruli_embeds.py "$bb" "$GLOM" || echo "  WARN: $bb failed, continuing"
            ;;
    esac
done

EMB="$GLOM/embeddings"
echo ""; echo "  embeddings produced:"; ls -1 "$EMB"/*.npy 2>/dev/null || echo "  (none)"

echo ""; echo "===== STAGE 3: clustering (HDBSCAN + Leiden per embedding) ====="
# Drive clustering off whatever embeddings were actually produced, so a skipped
# backbone doesn't break the loop. cluster_* take the path stem (no extension).
shopt -s nullglob
for npy in "$EMB"/*.npy; do
    stem="${npy%.npy}"
    echo "--- clustering $(basename "$stem") ---"
    python scripts/cluster_hdbscan.py "$stem" || echo "  WARN: hdbscan failed on $(basename "$stem")"
    python scripts/cluster_leiden.py  "$stem" || echo "  WARN: leiden failed on $(basename "$stem")"
done
shopt -u nullglob

# =============================================================================
# Collect all generated artifacts back to $HOME (scratch is volatile).
# =============================================================================
echo ""; echo "===== collecting results to $RESULTS ====="
mkdir -p "$RESULTS/glomeruli"
rsync -a "$EMB/" "$RESULTS/glomeruli/embeddings/" 2>/dev/null || true
# cluster_hdbscan/leiden write to "<repo>/results/clustering/<timestamp>_<method>/".
rsync -a "$SIM/results/clustering/" "$RESULTS/clustering/" 2>/dev/null || true

echo "Full pipeline complete. Artifacts in $RESULTS"
