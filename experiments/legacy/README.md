# Legacy experiment configs

Archived SLURM scripts for the historical segmentation runs (Run 6–12, LOSO, and
the Run 4/6 eval). Kept for **provenance** — they document exactly what each run
launched — but are **not maintained** and are superseded by
`experiments/full_pipeline.sbatch` (full pipeline) and `experiments/train_segmentation.sbatch`
(generic training launcher).

Paths and flags inside these files reflect the layout at the time they ran and
are stale: the scripts have since moved to a flat `scripts/`, and the ResNet50
encoder choice was renamed from `--encoder resnet50_swav` to `--encoder resnet50`
once the converter grew beyond the SwAV weights.

Removed entirely (not archived): the Optuna HPO branch (`train_optuna.py` +
`train_run5_optuna.sbatch`) — dead code once the Run 7 recipe was fixed; recoverable
from git history if ever needed. Run 5 itself is documented in the wiki `project-progress`.
