import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones.nasnet import NASNet
from src.backbones.densenet169 import DenseNet169
from src.backbones.densenet201 import DenseNet201
from src.backbones.mobilenet import MobileNet
from src.backbones.xception import Xception
from src.metrics.augmentation_invariance import compute_augmentation_robustness_metrics


BACKBONE_REGISTRY = {
    "densenet169": {
        "build": lambda input_size: DenseNet169(input_size or 224),
        "mask_mode": "optional",  # fix: prima "required", vedi nasnet.py/densenet169.py aggiornati
    },
    "densenet201": {
        "build": lambda input_size: DenseNet201(input_size or 224),
        "mask_mode": "optional",
    },
    "mobilenet": {
        "build": lambda input_size: MobileNet(input_size or 224),
        "mask_mode": "optional",
    },
    "xception": {
        "build": lambda input_size: Xception(input_size or 299),
        "mask_mode": "optional",
    },
    "nasnet": {
        "build": lambda input_size: NASNet(input_size or 331),
        "mask_mode": "optional",  # fix: prima "none" (pooling="avg" incorporato), ora pooling=None + mask manuale
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valuta l'invarianza alle augmentation (rotazioni/flip) di una backbone."
    )

    parser.add_argument(
        "glomeruli_dir",
        type=Path,
        help="Directory con le sottocartelle crops/ e masks/ (output di extract_glomeruli_from_annotations.py).",
    )
    parser.add_argument(
        "backbone",
        type=str,
        choices=sorted(BACKBONE_REGISTRY.keys()),
        help="Backbone da valutare.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Input size custom. Se omesso, usa il default raccomandato per la backbone scelta.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=200,
        help="Numero di crop campionati casualmente da valutare. 0 = tutti (costoso).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--augmentations",
        type=str,
        default="rot90,rot180,rot270,flip_h,flip_v",
        help="Lista separata da virgole tra: rot90,rot180,rot270,flip_h,flip_v.",
    )
    parser.add_argument(
        "--retrieval-ks",
        type=str,
        default="5,10,20",
        help="Valori di k per self-retrieval@k, separati da virgola.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "backbone_evaluation" / "augmentation_invariance",
    )

    return parser.parse_args()


def build_backbone(args: argparse.Namespace):
    config = BACKBONE_REGISTRY[args.backbone]
    model = config["build"](args.input_size)
    return model, config["mask_mode"]


def backbone_run_name(args: argparse.Namespace) -> str:
    return args.backbone


def main() -> None:
    args = parse_args()

    crops_dir = args.glomeruli_dir / "crops"
    masks_dir = args.glomeruli_dir / "masks"

    if not crops_dir.is_dir():
        raise NotADirectoryError(f"Directory crops non trovata: {crops_dir}")

    image_paths = sorted(
        p for p in crops_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(f"Nessun crop .png trovato in {crops_dir}")

    model, mask_mode = build_backbone(args)

    mask_paths = None
    masks_available = masks_dir.is_dir()

    if mask_mode == "required":
        if not masks_available:
            raise NotADirectoryError(
                f"La backbone '{args.backbone}' richiede la maschera, ma {masks_dir} non esiste."
            )
        mask_paths = sorted(
            p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
        )
        if len(mask_paths) != len(image_paths):
            raise ValueError(
                f"Numero di crop ({len(image_paths)}) e maschere ({len(mask_paths)}) non combacia."
            )
    elif mask_mode == "optional":
        if masks_available:
            mask_paths = sorted(
                p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
            )
            if len(mask_paths) != len(image_paths):
                print(
                    f"  [WARN] Numero di crop ({len(image_paths)}) e maschere "
                    f"({len(mask_paths)}) non combacia: procedo senza maschera."
                )
                mask_paths = None
        else:
            print(f"  [INFO] {masks_dir} non trovata: procedo senza maschera (fallback ad average pooling).")
    else:  # "none"
        mask_paths = None

    def embed_fn(image, mask):
        return model(image, mask)

    subset_size = args.subset_size if args.subset_size and args.subset_size > 0 else None
    augmentations = tuple(a.strip() for a in args.augmentations.split(",") if a.strip())
    retrieval_ks = tuple(int(k.strip()) for k in args.retrieval_ks.split(",") if k.strip())

    print(
        f"Valutazione invarianza per '{backbone_run_name(args)}' "
        f"su {subset_size or len(image_paths)} crop "
        f"(mask_mode='{mask_mode}', maschera {'usata' if mask_paths is not None else 'non usata'})..."
    )

    results = compute_augmentation_robustness_metrics(
        image_paths=image_paths,
        mask_paths=mask_paths,
        embed_fn=embed_fn,
        augmentations=augmentations,
        retrieval_ks=retrieval_ks,
        subset_size=subset_size,
        random_state=args.random_state,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = backbone_run_name(args)
    overall_path = output_dir / f"{run_name}_overall.csv"
    by_aug_path = output_dir / f"{run_name}_by_augmentation.csv"

    results["overall"].insert(0, "backbone", run_name)
    results["by_augmentation"].insert(0, "backbone", run_name)

    results["overall"].to_csv(overall_path, index=False)
    results["by_augmentation"].to_csv(by_aug_path, index=False)

    print("\n=== Risultati aggregati (tutte le augmentation) ===")
    print(results["overall"].to_string(index=False))
    print("\n=== Risultati per singola augmentation ===")
    print(results["by_augmentation"].to_string(index=False))

    print(f"\nSalvato in: {overall_path}")
    print(f"Salvato in: {by_aug_path}")


if __name__ == "__main__":
    main()
