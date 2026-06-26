import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.crop_glomeruli import crop_glomeruli
from src.data.extract_masks import parse_xml_annotations

target_mean = np.array([68.0, 10.0, 6.0], dtype=np.float32)
target_std  = np.array([16.0, 7.0, 6.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract glomeruli from SVS slides based on annotations."
    )
    parser.add_argument(
        "slides_path",
        type=Path,
        help="Directory containing .svs slide files.",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=None,
        help="Directory containing .xml annotation files. Defaults to slides_path.",
    )
    parser.add_argument(
        "--images-size",
        type=int,
        default=1024,
        help="Size of the square images of glomeruli after resizing. Default: 1024.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Margin around glomeruli annotations. Default: 0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = PROJECT_ROOT / "data" / "glomeruli"
    output_path.mkdir(parents=True, exist_ok=True)

    slides_path = args.slides_path
    annotations_path = args.annotations_path or slides_path

    if not slides_path.is_dir():
        raise NotADirectoryError(f"Slides directory not found: {slides_path}")

    if not annotations_path.is_dir():
        raise NotADirectoryError(f"Annotations directory not found: {annotations_path}")

    slide_paths = sorted(slides_path.glob("*.svs"))

    if not slide_paths:
        raise FileNotFoundError(f"No .svs slides found in: {slides_path}")

    for slide_path in tqdm(slide_paths, desc="Extracting glomeruli"):
        slide_name = slide_path.stem
        xml_path = annotations_path / f"{slide_name}.xml"

        if not xml_path.exists():
            print(f"Skipping {slide_name}: missing XML file.")
            continue

        # Extract glomeruli crops and save them to output_path
        polygons = parse_xml_annotations(xml_path)

        items = crop_glomeruli(
            slide_path,
            polygons,
            args.margin,
            (target_mean, target_std),
        )

        crops_output_path = output_path / "crops"
        masks_output_path = output_path / "masks"

        crops_output_path.mkdir(parents=True, exist_ok=True)
        masks_output_path.mkdir(parents=True, exist_ok=True)

        for name_counter, (crop, mask) in enumerate(items):
            crop = crop.resize(
                (args.images_size, args.images_size),
                resample=Image.Resampling.LANCZOS
            )

            mask = mask.resize(
                (args.images_size, args.images_size),
                resample=Image.Resampling.NEAREST
            )

            file_stem = f"{slide_name}_{name_counter:04d}"

            crop_output_path = crops_output_path / f"{file_stem}.png"
            mask_output_path = masks_output_path / f"{file_stem}_mask.png"

            crop.save(crop_output_path)
            mask.save(mask_output_path)


if __name__ == "__main__":
    main()
