from pathlib import Path
from typing import Literal

import numpy as np
import openslide
from PIL import Image, ImageDraw
from skimage import color

from src.data.reinhard_normalization import apply_reinhard


def _compute_bounding_box(
    polygon: np.ndarray,
    slide_dimensions: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    """
    Compute a bounding box around one glomerulus polygon.

    Returns
    -------
    x_min, y_min, x_max, y_max:
        Bounding box coordinates clipped to WSI boundaries.
    """

    x_min = int(np.floor(polygon[:, 0].min())) - margin
    y_min = int(np.floor(polygon[:, 1].min())) - margin
    x_max = int(np.ceil(polygon[:, 0].max())) + margin
    y_max = int(np.ceil(polygon[:, 1].max())) + margin

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(slide_dimensions[0], x_max)
    y_max = min(slide_dimensions[1], y_max)

    return x_min, y_min, x_max, y_max

def _black_outside_polygon(img, polygon):
    img = img.convert("RGB")

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    pts = [tuple(p) for p in polygon.astype(int)]
    draw.polygon(pts, fill=255)

    result = Image.new("RGB", img.size, (0, 0, 0))
    result.paste(img, mask=mask)

    return result

def polygon_to_mask(
    image_size: tuple[int, int],
    polygon_local: np.ndarray,
) -> np.ndarray:
    """
    Create a binary mask from a local polygon.

    Parameters
    ----------
    image_size:
        PIL image size as (width, height).

    polygon_local:
        Polygon coordinates relative to the crop, shape (N, 2).

    Returns
    -------
    mask:
        Boolean mask with shape (height, width).
        True = inside polygon
        False = outside polygon
    """

    width, height = image_size

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    polygon_points = [
        (float(x), float(y))
        for x, y in polygon_local
    ]

    draw.polygon(polygon_points, outline=255, fill=255)

    mask = np.array(mask_img) > 0

    return mask


def crop_glomeruli(
        slide_path: Path,
        polygons: list[np.ndarray],
        margin: int,
        reinhard_target: tuple[np.ndarray, np.ndarray] | None = None,
        normalization_scope: Literal["glomerulus", "tissue"] = "glomerulus",
) -> list[tuple[Image.Image, Image.Image]]:
    """
    Crop glomeruli from a WSI based on polygon annotations.

    Returns:
        list of (crop_image, mask_image)

    crop_image:
        RGB crop extracted from the WSI.

    mask_image:
        binary mask of the glomerulus in the crop.
        white = glomerulus
        black = background

    If reinhard_target is not None, Reinhard color normalization is applied.

    reinhard_target:
        None
            No color normalization.

        (target_mean, target_std)
            Apply Reinhard normalization using the provided target statistics.

    normalization_scope:
        "glomerulus"
            Source statistics are computed only inside the glomerulus polygon.

        "tissue"
            Source statistics are computed on all tissue pixels in the crop,
            excluding the light background.
    """

    slide = openslide.OpenSlide(str(slide_path))
    slide_width, slide_height = slide.dimensions

    results = []

    apply_reinhard_normalization = reinhard_target is not None

    if apply_reinhard_normalization:
        target_mean, target_std = reinhard_target

        target_mean = np.asarray(target_mean, dtype=np.float32)
        target_std = np.asarray(target_std, dtype=np.float32)

        if target_mean.shape != (3,):
            raise ValueError(
                f"target_mean must have shape (3,), got {target_mean.shape}."
            )

        if target_std.shape != (3,):
            raise ValueError(
                f"target_std must have shape (3,), got {target_std.shape}."
            )

    for polygon in polygons:
        x_min, y_min, x_max, y_max = _compute_bounding_box(
            polygon,
            (slide_width, slide_height),
            margin
        )

        crop_width = x_max - x_min
        crop_height = y_max - y_min

        crop = slide.read_region(
            location=(x_min, y_min),
            level=0,
            size=(crop_width, crop_height),
        ).convert("RGB")

        # Coordinate globali WSI -> coordinate locali del crop
        polygon_local = polygon.copy().astype(float)
        polygon_local[:, 0] -= x_min
        polygon_local[:, 1] -= y_min

        # Maschera del glomerulo nel crop
        polygon_mask = polygon_to_mask(
            image_size=crop.size,
            polygon_local=polygon_local
        ).astype(bool)

        mask_image = Image.fromarray(
            (polygon_mask.astype(np.uint8) * 255),
            mode="L"
        )

        if polygon_mask.sum() == 0:
            print(f"Warning: empty polygon mask for slide {slide_path.stem}")
            results.append((crop, mask_image))
            continue

        if apply_reinhard_normalization:
            rgb = np.array(crop).astype(np.float32) / 255.0
            lab = color.rgb2lab(rgb)

            if normalization_scope == "glomerulus":
                normalization_mask = polygon_mask

            elif normalization_scope == "tissue":
                # Escludo lo sfondo chiaro del vetrino
                normalization_mask = rgb.mean(axis=2) < 0.85

                if normalization_mask.sum() == 0:
                    normalization_mask = polygon_mask

            else:
                raise ValueError(
                    "normalization_scope must be either 'glomerulus' or 'tissue'."
                )

            normalization_mask = normalization_mask.astype(bool)

            lab_tissue = lab[normalization_mask]

            source_mean = lab_tissue.mean(axis=0)
            source_std = lab_tissue.std(axis=0)

            eps = 1e-6
            source_std = np.where(source_std < eps, 1.0, source_std)

            crop = apply_reinhard(
                image=crop,
                source_mean=source_mean,
                source_std=source_std,
                target_mean=target_mean,
                target_std=target_std,
                tissue_mask_patch=normalization_mask,
            )

        results.append((crop, mask_image))

    slide.close()

    return results