import gc
import math
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

from src.data.extract_masks import extract_seg_mask, extract_tissue_mask
from src.data.reinhard_normalization import apply_reinhard, compute_reinhard


def preprocess(
    slides_path: list[Path],
    output_path: Path,
    annotations_path: Path,
    patch_stride: int,
    patch_extraction_level: int = 0,
    mask_extraction_level: int = 2,
    patch_original_size: int = 2000,
    patch_resize: int = 400
):
    """
    Extract and preprocess image patches from whole slide images (WSI).

    Processes a collection of WSI files by extracting patches at a specified stride,
    applying tissue filtering, semantic segmentation masking, and color normalization.
    Output patches and masks are saved as PNG files.

    Args:
        slides_path: List of Path objects pointing to input .svs slide files.
        output_path: Directory where output patches ("img" subfolder) and masks
                     ("mask" subfolder) will be saved.
        annotations_path: Directory containing XML annotation files corresponding
                          to each slide (named {slide_stem}.xml).
        patch_stride: Stride in pixels (at mask level) for extracting patches.
        patch_extraction_level: Resolution level of the slide to extract patches from.
                                Defaults to 0 (highest resolution).
        mask_extraction_level: Resolution level to extract tissue and compute Reinhard
                               normalization parameters. Defaults to 2.
        patch_original_size: Size of patches extracted from the slide in pixels.
                             Defaults to 2000.
        patch_resize: Target size to resize patches and masks to. Defaults to 400.

    Raises:
        FileNotFoundError: If the annotation file for a slide is not found.

    Note: patches with tissue ratio < 5% are skipped.
    """

    for slide_path in tqdm(slides_path, desc=f"Processing {len(slides_path)} slides"):
        slide = openslide.OpenSlide(str(slide_path))

        slide_width_0, slide_height_0 = slide.level_dimensions[0]
        ds_tissue_mask = slide.level_downsamples[mask_extraction_level]

        tissue_mask = extract_tissue_mask(slide, level=mask_extraction_level)

        slide_lowres = slide.read_region(
            (0, 0),
            mask_extraction_level,
            slide.level_dimensions[mask_extraction_level]
        ).convert("RGB")

        source_mean, source_std = compute_reinhard(
            slide_lowres,
            tissue_mask=tissue_mask
        )

        annotation_path = annotations_path / f"{slide_path.stem}.xml"
        if not annotation_path.exists():
            slide.close()
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        seg_mask_ds = slide.level_downsamples[patch_extraction_level]
        seg_mask = extract_seg_mask(
            str(annotation_path),
            slide.level_dimensions[patch_extraction_level],
            ds=seg_mask_ds
        )
        seg_mask_patch_size = math.ceil(patch_original_size / seg_mask_ds)

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
                    slide_x + patch_original_size > slide_width_0 or
                    slide_y + patch_original_size > slide_height_0
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

                seg_x = int(round(slide_x / seg_mask_ds))
                seg_y = int(round(slide_y / seg_mask_ds))

                seg_mask_patch = seg_mask[
                    seg_y:seg_y + seg_mask_patch_size,
                    seg_x:seg_x + seg_mask_patch_size
                ]
                if seg_mask_patch.shape != (seg_mask_patch_size, seg_mask_patch_size):
                    continue

                seg_mask_patch = resize(
                    seg_mask_patch.astype(np.uint8),
                    (patch_resize, patch_resize),
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(np.uint8)

                tissue_mask_patch = resize(
                    tissue_mask_patch.astype(np.uint8),
                    (patch_resize, patch_resize),
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(np.uint8)

                patch = apply_reinhard(
                    patch,
                    source_mean,
                    source_std,
                    tissue_mask_patch=tissue_mask_patch
                )

                filename = f"{slide_path.stem}_{slide_x}_{slide_y}.png"
                patch.save(output_path / "img" / filename)

                Image.fromarray((seg_mask_patch * 255).astype(np.uint8)).save(
                    output_path / "mask" / filename
                )

                del patch, seg_mask_patch, tissue_mask_patch
                gc.collect()

        slide.close()
        del slide, tissue_mask, seg_mask, slide_lowres
        gc.collect()
