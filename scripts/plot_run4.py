import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CSV_PATH = Path(__file__).parent.parent / "models" / "hpc-run-1742197" / "training_log.csv"
OUT_DIR   = Path(__file__).parent.parent / "models" / "hpc-run-1742197"

PHASE_SPLIT = 10   # first epoch of Phase 2 (fine-tuning)

def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def main():
    rows  = load_csv(CSV_PATH)
    epochs       = [r["epoch"] for r in rows]
    train_loss   = [r["loss"] for r in rows]
    val_loss     = [r["val_loss"] for r in rows]
    train_acc    = [r["accuracy"] for r in rows]
    val_acc      = [r["val_accuracy"] for r in rows]
    train_iou    = [r["mean_io_u"] for r in rows]
    val_iou      = [r["val_mean_io_u"] for r in rows]

    best_epoch = int(epochs[val_iou.index(max(val_iou))])
    best_iou   = max(val_iou)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SegNet-VGG19 — Run 4 (HPC Legion, JOBID 1742197)", fontsize=13, fontweight="bold")

    specs = [
        (axes[0], train_loss,  val_loss,  "Loss (BCE + Dice)",   "Loss"),
        (axes[1], train_acc,   val_acc,   "Accuracy",            "Accuracy"),
        (axes[2], train_iou,   val_iou,   "Mean IoU",            "IoU"),
    ]

    for ax, train_vals, val_vals, title, ylabel in specs:
        ax.plot(epochs, train_vals, label="Train",      color="#2196F3", linewidth=1.8)
        ax.plot(epochs, val_vals,   label="Validation", color="#F44336", linewidth=1.8)
        ax.axvline(x=PHASE_SPLIT - 0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # shade phases
        ax.axvspan(-0.5, PHASE_SPLIT - 0.5, alpha=0.04, color="blue")
        ax.axvspan(PHASE_SPLIT - 0.5, max(epochs) + 0.5, alpha=0.04, color="orange")

    # mark best val IoU on the IoU plot
    ax_iou = axes[2]
    ax_iou.scatter([best_epoch], [best_iou], color="#4CAF50", zorder=5, s=80)
    ax_iou.annotate(
        f"best {best_iou:.4f}\n(ep {best_epoch})",
        xy=(best_epoch, best_iou),
        xytext=(best_epoch + 1.2, best_iou - 0.025),
        fontsize=8,
        color="#4CAF50",
        arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1),
    )

    # phase legend
    p1_patch = mpatches.Patch(color="blue",   alpha=0.15, label="Phase 1 — frozen encoder")
    p2_patch = mpatches.Patch(color="orange", alpha=0.15, label="Phase 2 — fine-tuning")
    fig.legend(handles=[p1_patch, p2_patch], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = OUT_DIR / "run4_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
