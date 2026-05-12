import argparse
import random
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import preprocess


SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract train, validation, and test patches from SVS slides."
    )
    parser.add_argument(
        "slides_path",
        type=Path,
        help="Directory containing .svs slide files.",
    )
    parser.add_argument(
        "--split-percentages",
        type=float,
        nargs=2,
        default=(0.7, 0.15),
        metavar=("TRAIN", "VALIDATION"),
        help=(
            "Split proportions of train and validation. Accepts fractions summing to 1, then test is 0, "
            "or accepts fractions summing to less than 1, then test is the remainder. "
            "Default: 0.7 0.15"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used to shuffle slides before splitting. Default: 1.",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=None,
        help="Directory containing .xml annotation files. Defaults to slides_path.",
    )
    parser.add_argument(
        "--patch-extraction-level",
        type=int,
        default=0,
        help="OpenSlide level used as reference for patch extraction. Default: 0.",
    )
    parser.add_argument(
        "--mask-extraction-level",
        type=int,
        default=2,
        help="OpenSlide level used to extract the tissue mask. Default: 2.",
    )
    parser.add_argument(
        "--patch-original-size",
        type=int,
        default=2000,
        help="Patch side length read from level 0 before resizing. Default: 2000.",
    )
    parser.add_argument(
        "--patch-resize",
        type=int,
        default=400,
        help="Patch side length after resizing. Default: 400.",
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=None,
        help="Sliding-window stride at level 0. Defaults to patch_original_size.",
    )
    return parser.parse_args()


def split_dataset(
    slide_paths: list[Path],
    split_percentages: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Path]]:
    shuffled_slide_paths = slide_paths.copy()
    random.Random(seed).shuffle(shuffled_slide_paths)

    desired_counts = [percentage * len(shuffled_slide_paths) for percentage in split_percentages]
    split_counts = [int(count) for count in desired_counts]
    missing_count = len(shuffled_slide_paths) - sum(split_counts)

    remainders = sorted(
        range(len(desired_counts)),
        key=lambda index: desired_counts[index] - split_counts[index],
        reverse=True,
    )
    for index in remainders[:missing_count]:
        split_counts[index] += 1

    split_paths = {}
    start = 0
    for split_name, split_count in zip(SPLIT_NAMES, split_counts):
        end = start + split_count
        split_paths[split_name] = shuffled_slide_paths[start:end]
        start = end

    return split_paths


def main() -> None:
    args = parse_args()

    patch_stride = args.patch_stride
    if patch_stride is None:
        patch_stride = args.patch_original_size

    if args.mask_extraction_level < 2:
        warnings.warn("Setting mask extraction level below 2 can consume a huge amount of RAM.")

    slides_path = args.slides_path
    annotations_path = args.annotations_path or slides_path
    if not slides_path.is_dir():
        raise NotADirectoryError(f"Slides directory not found: {slides_path}")
    if not annotations_path.is_dir():
        raise NotADirectoryError(f"Annotations directory not found: {annotations_path}")

    slide_paths = sorted(slides_path.glob("*.svs"))
    if not slide_paths:
        raise FileNotFoundError(f"No .svs slides found in: {slides_path}")

    train_percentage, validation_percentage = args.split_percentages
    test_percentage = 1.0 - train_percentage - validation_percentage
    split_percentages = (
        train_percentage,
        validation_percentage,
        test_percentage,
    )
    split_paths = split_dataset(slide_paths, split_percentages, args.seed)

    patches_path = PROJECT_ROOT / "data" / "patches"
    for split_name, split_slide_paths in split_paths.items():
        output_path = patches_path / split_name
        (output_path / "img").mkdir(parents=True, exist_ok=True)
        (output_path / "mask").mkdir(parents=True, exist_ok=True)

        if not split_slide_paths:
            warnings.warn(f"No slides assigned to {split_name} split.")
            continue

        preprocess(
            slides_path=split_slide_paths,
            output_path=output_path,
            annotations_path=annotations_path,
            patch_stride=patch_stride,
            patch_extraction_level=args.patch_extraction_level,
            mask_extraction_level=args.mask_extraction_level,
            patch_original_size=args.patch_original_size,
            patch_resize=args.patch_resize,
        )


if __name__ == "__main__":
    main()
