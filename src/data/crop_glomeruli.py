from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw


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


def crop_glomeruli(
        slide_path: Path,
        polygons: list[np.ndarray],
        margin: int,
        remove_context: bool
) -> list[Image.Image]:
    """
    Crop glomeruli from a WSI based on their polygon annotations.

    Parameters
    ----------
    slide_path:
        Path to the SVS slide file.
    """
    slide = openslide.OpenSlide(str(slide_path))
    slide_width, slide_height = slide.dimensions

    crops = []

    for polygon in polygons:
        x_min, y_min, x_max, y_max = _compute_bounding_box(polygon, (slide_width, slide_height), margin)

        crop_width = x_max - x_min
        crop_height = y_max - y_min

        crop = slide.read_region(
            location=(x_min, y_min),
            level=0,
            size=(crop_width, crop_height),
        ).convert("RGB")

        if remove_context:
            crop = _black_outside_polygon(crop, polygon)

        crops.append(crop)

    return crops