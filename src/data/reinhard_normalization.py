import warnings

import numpy as np
from PIL import Image
from skimage import color

TARGET_MEAN = np.array([70.0, 8.0, 5.0])
TARGET_STD = np.array([15.0, 8.0, 8.0])


def compute_reinhard(
    image: Image.Image,
    tissue_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola source_mean e source_std in LAB.
    Da usare una sola volta per slide, su una versione downsampled della WSI.
    """

    image_rgb = image.convert("RGB")
    rgb = np.array(image_rgb).astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb)

    tissue_mask = tissue_mask > 0

    if tissue_mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"tissue_mask shape {tissue_mask.shape} diversa da image shape {rgb.shape[:2]}"
        )

    if np.count_nonzero(tissue_mask) < 10:
        raise ValueError("Troppi pochi pixel di tessuto per calcolare Reinhard.")

    tissue_pixels = lab[tissue_mask]

    source_mean = tissue_pixels.mean(axis=0)
    source_std = tissue_pixels.std(axis=0)
    source_std = np.where(source_std < 1e-6, 1.0, source_std)

    return source_mean, source_std


def apply_reinhard(
    image: Image.Image,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    tissue_mask_patch: np.ndarray | None = None
) -> Image.Image:
    """
    Applica Reinhard a una patch usando source_mean/source_std già calcolati sulla slide.
    """

    image_rgb = image.convert("RGB")
    rgb = np.array(image_rgb).astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb)

    if tissue_mask_patch is None:
        gray = rgb.mean(axis=2)
        tissue_mask = gray < 0.85
    else:
        tissue_mask = tissue_mask_patch > 0

        if tissue_mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"tissue_mask_patch shape {tissue_mask.shape} diversa da image shape {rgb.shape[:2]}"
            )

    normalized_lab = lab.copy()

    for c in range(3):
        normalized_lab[:, :, c] = (
            (lab[:, :, c] - source_mean[c])
            / source_std[c]
            * TARGET_STD[c]
            + TARGET_MEAN[c]
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb_norm = color.lab2rgb(normalized_lab)

    rgb_norm = np.clip(rgb_norm * 255, 0, 255).astype(np.uint8)

    
    rgb_original = (rgb * 255).astype(np.uint8)
    rgb_norm[~tissue_mask] = rgb_original[~tissue_mask]

    return Image.fromarray(rgb_norm)