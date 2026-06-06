import argparse
import csv
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.backbones.nasnet import NASNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        choices=["cls", "patch"],
        required=True,
    )
    dinov2_parser.add_argument(
        "--backbone-size",
        type=str,
        choices=DinoV2ModelName,
        default="large",
        help="DINOv2 model name. Default: large.",
    )

    dinov3_parser = subparsers.add_parser(
        "dinov3",
        description="Extract glomeruli embeds using DINOv3 backbone."
    )
    dinov3_parser.add_argument(
        "--backbone-size",
        type=str,
        choices=DinoV3ModelName,
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
        "crops_dir",
        type=Path,
        help="Directory containing .png glomeruli crops.",
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
    match model_name:
        case "dinov2":
            model = DinoV2(
                args.backbone_size,
                args.input_size
            )

        case "dinov3":
            model = DinoV3(
                args.backbone_size,
                args.input_size
            )

        case "nasnet":
            model = NASNet(
                args.input_size if args.input_size else 331
            )

        case _:
            raise ValueError(f"Invalid backbone: {model_name}")

    crops_dir = args.crops_dir
    assert crops_dir.exists()
    output_dir = PROJECT_ROOT / "data" / "glomeruli" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = [model_name, args.backbone_size]

    if model_name in ("dinov2", "dinov3"):
        parts.append(args.mode)
    parts.append("embeddings")
    filename = "_".join(parts) + ".npy"

    np_embeddings = output_dir / filename
    csv_file = output_dir / filename

    image_paths = sorted(path for path in crops_dir.iterdir())

    np_memmap = np.lib.format.open_memmap(
        np_embeddings,
        mode="w+",
        dtype="float32",
        shape=(len(image_paths), model.backbone.hidden_dim)
    )
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_path"])

        for i, image_path in enumerate(tqdm(image_paths, desc=f"Extracting embeddings with {model_name}_{args.backbone_size}")):
            image = Image.open(image_path).convert("RGB")

            embed = model(image, args.mode)
            embed = np.squeeze(embed)

            if args.mode == "patch":
                embed = embed.mean(axis=0)

            np_memmap[i] = embed

            writer.writerow([i, str(image_path)])
    np_memmap.flush()

if __name__ == "__main__":
    main()