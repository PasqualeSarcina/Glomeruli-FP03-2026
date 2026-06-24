import argparse
import csv
from itertools import repeat
from pathlib import Path
import sys
from typing import get_args

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backbones.nasnet import NASNet
from src.backbones.dinov2 import DinoV2ModelName, DinoV2
from src.backbones.dinov3 import DinoV3ModelName, DinoV3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract glomeruli embeds from previously processed SVS slides."
                    "To process glomeruli crops, execute extract_glomeruli_from_annotations.py"
    )

    subparsers = parser.add_subparsers(
        dest="backbone",
        required=True
    )

    dinov2_parser = subparsers.add_parser(
        "dinov2",
        description="Extract glomeruli embeds using DINOv2 backbone."
    )
    dinov2_parser.add_argument(
        "--mode",
        choices=["cls", "patch", "both"],
        required=True,
    )
    dinov2_parser.add_argument(
        "--backbone-size",
        type=str,
        choices=get_args(DinoV2ModelName),
        default="base",
        help="DINOv2 model name. Default: base.",
    )

    dinov3_parser = subparsers.add_parser(
        "dinov3",
        description="Extract glomeruli embeds using DINOv3 backbone."
    )
    dinov3_parser.add_argument(
        "--backbone-size",
        type=str,
        choices=get_args(DinoV3ModelName),
        default="base",
        help="DINOv3 model name. Default: base",
    )
    dinov3_parser.add_argument(
        "--mode",
        choices=["cls", "patch"],
        required=True,
    )

    nasnet_parser = subparsers.add_parser(
        "nasnet",
        description="Extract glomeruli embeds using NASNet."
    )

    parser.add_argument(
        "glomeruli_dir",
        type=Path,
        help="Directory containing glomeruli crops (and masks).",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        help="Input image size. For DINOv2, the input size must be a multiple of 14."
             " For DINOv3, the input size must be a multiple of the patch size of the selected model"
             " (small: 16, base: 16, large: 14, huge_plus: 14, 7b: 14)."
             "For NASNet is recommended to not set a different input dimension",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.backbone

    crops_dir = args.glomeruli_dir / "crops"
    if not crops_dir.is_dir():
        raise NotADirectoryError(f"Crops directory not found: {crops_dir}")

    image_paths = sorted(
        path for path in crops_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    masks_dir = args.glomeruli_dir / "masks"
    masks_paths = sorted(
        path for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


    match model_name:
        case "dinov2":
            model = DinoV2(
                args.backbone_size,
                args.input_size,
                args.mode
            )

        case "dinov3":
            model = DinoV3(
                args.backbone_size,
                args.input_size
            )

        case "nasnet":
            model = NASNet(
                args.input_size or 331
            )

        case _:
            raise ValueError(f"Invalid backbone: {model_name}")

    output_dir = PROJECT_ROOT / "data" / "glomeruli" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = [model_name]
    if getattr(args, "backbone_size", None) is not None:
        parts.append(args.backbone_size)
    if model_name in ("dinov2", "dinov3"):
        parts.append(args.mode)
    if model_name == "dinov2":
        #parts.append("masked")
        pass
    else:
        parts.append(str(Path(crops_dir).stem))
    parts.append("embeddings")
    filename_stem = "_".join(parts)

    np_embeddings = output_dir / f"{filename_stem}.npy"
    print(f"Saving embeddings to {np_embeddings}")
    csv_file = output_dir / f"{filename_stem}.csv"
    print(f"Saving image paths to {csv_file}")

    if np_embeddings.exists():
        np_embeddings.unlink()
    #np_memmap = np.lib.format.open_memmap(
    #    np_embeddings,
    #    mode="w+",
    #    dtype="float32",
    #    shape=(len(image_paths), model.hidden_dim)
    #)

    embeddings = np.empty(
        shape=(len(image_paths), model.hidden_dim),
        dtype="float32"
    )

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_path"])

        image_mask_pairs = zip(
            image_paths,
            masks_paths if masks_paths is not None else repeat(None),
        )
        progress = tqdm(
            image_mask_pairs,
            total=len(image_paths),
            desc=f"Extracting embeddings with {model_name}",
        )

        for i, (image_path, mask_path) in enumerate(progress):
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")

            embeddings[i] = model(image, mask)

            writer.writerow([i, str(image_path)])
    #np_memmap.flush()
    np.save(np_embeddings, embeddings)

if __name__ == "__main__":
    main()
