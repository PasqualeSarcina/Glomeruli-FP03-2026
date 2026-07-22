# Legacy experiment configs

Archived SLURM scripts for the historical segmentation runs (Run 6–12, LOSO, and
the Run 4/6 eval). Kept for **provenance** — they document exactly what each run
launched — but are **not maintained** and are superseded by
`scripts/test_full_pipeline.sbatch` (full pipeline) and `scripts/training/train_hpc.sbatch`
(generic training launcher).

Paths inside these files reflect the layout at the time they ran and may be stale.

Removed entirely (not archived): the Optuna HPO branch (`train_optuna.py` +
`train_run5_optuna.sbatch`) — dead code once the Run 7 recipe was fixed; recoverable
from git history if ever needed. Run 5 itself is documented in the wiki `project-progress`.
