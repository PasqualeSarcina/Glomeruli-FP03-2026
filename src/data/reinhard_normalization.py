import warnings
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from skimage import color


TARGET_MEAN = np.array([70.0, 8.0, 5.0])
TARGET_STD = np.array([15.0, 8.0, 8.0])


def compute_reinhard_source_stats(
    slide_path: Path,
    tissue_mask: np.ndarray,
    mask_extraction_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Reinhard source mean and standard deviation in Lab color space.

    The statistics are computed from a downsampled version of the WSI,
    considering only tissue pixels according to the provided tissue mask.

    Parameters
    ----------
    slide_path:
        Path to the WSI file.

    tissue_mask:
        Binary tissue/background mask at the same level used to read the slide.

        True / 1  = tissue
        False / 0 = background

    mask_extraction_level:
        OpenSlide level used both to generate the tissue mask and to read
        the image used for computing the Reinhard statistics.

    Returns
    -------
    source_mean:
        Mean Lab values of tissue pixels, with shape (3,).

    source_std:
        Standard deviation of Lab values of tissue pixels, with shape (3,).
    """

    slide = openslide.OpenSlide(str(slide_path))

    level_width, level_height = slide.level_dimensions[mask_extraction_level]

    image = slide.read_region(
            location=(0, 0),
            level=mask_extraction_level,
            size=(level_width, level_height),
    ).convert("RGB")

    slide.close()

    rgb = np.array(image).astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb)

    tissue_mask = tissue_mask > 0

    if tissue_mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"tissue_mask shape {tissue_mask.shape} does not match image shape "
            f"{rgb.shape[:2]} for slide {slide_path.stem}."
        )

    tissue_pixels = lab[tissue_mask]

    source_mean = tissue_pixels.mean(axis=0)
    source_std = tissue_pixels.std(axis=0)

    eps = 1e-6
    source_std = np.where(source_std < eps, 1.0, source_std)

    return source_mean.astype(np.float32), source_std.astype(np.float32)


def apply_reinhard(
    image: Image.Image,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    tissue_mask_patch: np.ndarray | None = None,
) -> Image.Image:
    """
    Apply Reinhard color normalization to a patch using precomputed
    source statistics and a selected target distribution.

    The transformation is applied only to tissue pixels.
    Background pixels are left unchanged.
    """

    image_rgb = image.convert("RGB")
    rgb = np.array(image_rgb).astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb)

    if tissue_mask_patch is None:
        tissue_mask = np.ones(rgb.shape[:2], dtype=bool)
    else:
        tissue_mask = tissue_mask_patch > 0

        if tissue_mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"tissue_mask_patch shape {tissue_mask.shape} "
                f"does not match image shape {rgb.shape[:2]}"
            )

    normalized_lab = lab.copy()
    eps = 1e-8

    for c in range(3):
        normalized_lab[:, :, c][tissue_mask] = (
            (lab[:, :, c][tissue_mask] - source_mean[c])
            / (source_std[c] + eps)
            * target_std[c]
            + target_mean[c]
        )

    normalized_lab[:, :, 0] = np.clip(normalized_lab[:, :, 0], 0, 100)
    normalized_lab[:, :, 1] = np.clip(normalized_lab[:, :, 1], -128, 127)
    normalized_lab[:, :, 2] = np.clip(normalized_lab[:, :, 2], -128, 127)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb_norm = color.lab2rgb(normalized_lab)

    rgb_norm = np.clip(rgb_norm * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb_norm)


def select_reinhard_targets_from_train(
    source_stats: dict,
    train_slide_ids: list[str],
    n_targets: int = 2,
) -> dict:
    """
    Select Reinhard target slides from the training set.

    The targets are selected according to the L-channel mean in Lab space,
    so that different brightness/stain profiles are included.
    """

    if n_targets < 1:
        raise ValueError("n_targets must be at least 1.")

    available_train_ids = [
        slide_id
        for slide_id in train_slide_ids
        if slide_id in source_stats
    ]

    if len(available_train_ids) < n_targets:
        raise ValueError(
            f"Requested {n_targets} Reinhard targets, but only "
            f"{len(available_train_ids)} training slides have source statistics."
        )

    sorted_ids = sorted(
        available_train_ids,
        key=lambda slide_id: source_stats[slide_id]["mean"][0],
    )

    if n_targets == 1:
        selected_indices = [len(sorted_ids) // 2]
    elif n_targets == 2:
        selected_indices = [0, len(sorted_ids) - 1]
    else:
        selected_indices = np.linspace(
            0,
            len(sorted_ids) - 1,
            n_targets,
            dtype=int,
        ).tolist()

    selected_ids = []
    for idx in selected_indices:
        slide_id = sorted_ids[idx]

        if slide_id not in selected_ids:
            selected_ids.append(slide_id)

    reinhard_targets = {
        slide_id: source_stats[slide_id]
        for slide_id in selected_ids
    }

    print("Selected Reinhard targets:")
    for slide_id in reinhard_targets:
        print(
            f"- {slide_id}: "
            f"mean={reinhard_targets[slide_id]['mean']}, "
            f"std={reinhard_targets[slide_id]['std']}"
        )

    return reinhard_targets