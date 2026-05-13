import gc
import math
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

from src.data.extract_masks import parse_xml_annotations, create_patch_seg_mask
from src.data.reinhard_normalization import apply_reinhard


def preprocess_single_slide(
    *,
    slide_path: Path,
    output_path: Path,
    annotations_path: Path,
    tissue_masks_path: Path,
    source_stats: dict,
    reinhard_targets: dict | None = None,
    apply_reinhard_augmentation: bool = False,
    patch_stride: int,
    mask_extraction_level: int = 2,
    patch_original_size: int = 2000,
    patch_resize: int = 400,
) -> None:
    """
    Extract and preprocess image patches from a single whole slide image.

    The function extracts tissue-containing patches, creates the corresponding
    segmentation masks, and optionally applies Reinhard color augmentation
    using precomputed source statistics and selected target statistics.
    """

    slide_id = slide_path.stem

    annotation_path = annotations_path / f"{slide_id}.xml"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    tissue_mask_path = tissue_masks_path / f"{slide_id}.npy"
    if not tissue_mask_path.exists():
        raise FileNotFoundError(f"Tissue mask file not found: {tissue_mask_path}")

    if slide_id not in source_stats:
        raise KeyError(f"Source Reinhard statistics not found for slide: {slide_id}")

    source_mean = np.array(source_stats[slide_id]["mean"], dtype=np.float32)
    source_std = np.array(source_stats[slide_id]["std"], dtype=np.float32)

    annotation_polygons = parse_xml_annotations(annotation_path)
    tissue_mask = np.load(tissue_mask_path)

    slide = openslide.OpenSlide(str(slide_path))

    slide_width_0, slide_height_0 = slide.level_dimensions[0]
    ds_tissue_mask = slide.level_downsamples[mask_extraction_level]

    mask_patch_size = math.ceil(patch_original_size / ds_tissue_mask)
    stride_mask = max(1, int(round(patch_stride / ds_tissue_mask)))

    for y in range(0, tissue_mask.shape[0] - mask_patch_size + 1, stride_mask):
        for x in range(0, tissue_mask.shape[1] - mask_patch_size + 1, stride_mask):

            tissue_mask_patch = tissue_mask[
                y:y + mask_patch_size,
                x:x + mask_patch_size
            ]

            tissue_ratio = np.count_nonzero(tissue_mask_patch) / tissue_mask_patch.size

            if tissue_ratio < 0.05:
                continue

            slide_x = int(round(x * ds_tissue_mask))
            slide_y = int(round(y * ds_tissue_mask))

            if (
                slide_x + patch_original_size > slide_width_0
                or slide_y + patch_original_size > slide_height_0
            ):
                continue

            patch = slide.read_region(
                (slide_x, slide_y),
                0,
                (patch_original_size, patch_original_size)
            ).convert("RGB")

            patch = patch.resize(
                (patch_resize, patch_resize),
                resample=Image.Resampling.LANCZOS
            )

            seg_mask_patch = create_patch_seg_mask(
                annotation_polygons,
                (slide_x, slide_y),
                patch_original_size
            )

            seg_mask_patch = seg_mask_patch.resize(
                (patch_resize, patch_resize),
                resample=Image.Resampling.NEAREST
            )

            tissue_mask_patch = resize(
                tissue_mask_patch.astype(np.uint8),
                (patch_resize, patch_resize),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)

            base_filename = f"{slide_id}_{slide_x}_{slide_y}.png"

            patch.save(output_path / "img" / base_filename)
            seg_mask_patch.save(output_path / "mask" / base_filename)

            if apply_reinhard_augmentation:
                if reinhard_targets is None:
                    raise ValueError(
                        "apply_reinhard_augmentation=True but reinhard_targets is None."
                    )

                for target_id, target_stats in reinhard_targets.items():
                    if target_id == slide_id:
                        continue
                    target_mean = np.array(target_stats["mean"], dtype=np.float32)
                    target_std = np.array(target_stats["std"], dtype=np.float32)

                    patch_reinhard = apply_reinhard(
                        image=patch,
                        source_mean=source_mean,
                        source_std=source_std,
                        target_mean=target_mean,
                        target_std=target_std,
                        tissue_mask_patch=tissue_mask_patch,
                    )

                    aug_filename = (
                        f"{slide_id}_{slide_x}_{slide_y}_reinhard_{target_id}.png"
                    )

                    patch_reinhard.save(output_path / "img" / aug_filename)
                    seg_mask_patch.save(output_path / "mask" / aug_filename)

                    del patch_reinhard

            del patch, seg_mask_patch, tissue_mask_patch
            gc.collect()

    slide.close()

    del slide, tissue_mask
    gc.collect()
