from pathlib import Path

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
        remove_context: bool,
        apply_color_normalization: bool = False,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
) -> list[Image.Image]:
    """
    Crop glomeruli from a WSI based on their polygon annotations.

    If apply_color_normalization is True, Reinhard normalization is applied
    only inside the glomerulus polygon.

    Source mean and source standard deviation are computed directly from
    the current crop, using only pixels inside the polygon.

    If remove_context is True, everything outside the polygon is set to black.
    """

    slide = openslide.OpenSlide(str(slide_path))
    slide_width, slide_height = slide.dimensions

    crops = []

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
        )

        if polygon_mask.sum() == 0:
            print(f"Warning: empty polygon mask for slide {slide_path.stem}")
            crops.append(crop)
            continue

        # Reinhard solo dentro il polygon
        if apply_color_normalization:
            if target_mean is None or target_std is None:
                raise ValueError(
                    "target_mean and target_std must be provided when "
                    "apply_color_normalization=True."
                )

            # Calcolo source_mean e source_std dentro il crop,
            # usando solo i pixel interni al polygon
            rgb = np.array(crop).astype(np.float32) / 255.0
            lab = color.rgb2lab(rgb)

            lab_inside_polygon = lab[polygon_mask]

            source_mean = lab_inside_polygon.mean(axis=0)
            source_std = lab_inside_polygon.std(axis=0)

            eps = 1e-6
            source_std = np.where(source_std < eps, 1.0, source_std)

            crop = apply_reinhard(
                image=crop,
                source_mean=source_mean,
                source_std=source_std,
                target_mean=target_mean,
                target_std=target_std,
                tissue_mask_patch=polygon_mask,
            )

        # Nero fuori dal polygon
        if remove_context:
            crop_array = np.array(crop).copy()
            crop_array[~polygon_mask] = 0
            crop = Image.fromarray(crop_array)

        crops.append(crop)

    slide.close()

    return crops