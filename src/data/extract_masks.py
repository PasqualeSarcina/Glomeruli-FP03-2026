import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openslide
from scipy.signal import argrelextrema
import skimage
from skimage.filters.rank import entropy
from skimage.morphology import disk, opening, closing, remove_small_objects
from skimage.morphology import remove_small_holes
from PIL import Image


def filter_entropy_image(image, threshold, disk_radius=3):
    eimage = entropy(image, disk(disk_radius))
    return eimage < threshold


def extract_tissue_mask(
    slide_path: Path,
    level: int = 2,
    small_objects_max_size: int = 1499,
    small_holes_max_size: int = 5000
) -> np.ndarray:

    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]

    img = slide.read_region((0, 0), level, (w, h)).convert("L")
    img = np.array(img, dtype=np.uint8)

    ent = entropy(img, disk(5))

    hist_values, hist_bins = np.histogram(ent, 30)
    minindex = argrelextrema(hist_values, np.less)
    candidate_thresholds = []

    for idx in minindex[0]:
        temp_thresh = hist_bins[idx]

        if 0.5 < temp_thresh < 5.0:
            candidate_thresholds.append(temp_thresh)

    if len(candidate_thresholds) > 0:
        thresh_localminimal = candidate_thresholds[-1]
    else:
        print("No suitable local minimum found. Using default threshold of 2.0.")
        thresh_localminimal = 2.0

    thresh1 = filter_entropy_image(img, thresh_localminimal)

    # tessuto = True, background = False
    mask = ~thresh1
    mask = mask.astype(bool)

    clean = opening(mask, np.ones((3, 3), dtype=bool))
    clean = closing(clean, np.ones((7, 7), dtype=bool))

    # chiude piccole interruzioni nella maschera
    clean = closing(clean, disk(5))

    # rimuove piccoli oggetti isolati
    clean = remove_small_objects(
        clean.astype(bool),
        max_size=small_objects_max_size,
        connectivity=2
    )

    # riempie buchi interni al tessuto
    clean = remove_small_holes(
        clean.astype(bool),
        max_size=small_holes_max_size,
        connectivity=2
    )

    clean_mask = clean.astype(np.bool)

    return clean_mask


def parse_xml_annotations(xml_path: Path) -> list[np.ndarray]:
    """
    Read glomeruli polygon annotations from an XML file.

    Expected XML structure:
    <Annotations>
        <Annotation>
            <Coordinates>
                <Coordinate X="..." Y="..." />
                <Coordinate X="..." Y="..." />
                ...
            </Coordinates>
        </Annotation>
    </Annotations>

    Returns
    -------
    polygons:
        List of polygons. Each polygon is an array of shape (N, 2),
        containing x, y coordinates at WSI level 0.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons: list[np.ndarray] = []

    for annotation in root.iter():
        if annotation.tag.lower().endswith("annotation"):
            coords = []

            for coord in annotation.iter():
                if coord.tag.lower().endswith("coordinate"):
                    x = coord.attrib.get("X")
                    y = coord.attrib.get("Y")

                    if x is not None and y is not None:
                        coords.append([float(x), float(y)])

            if len(coords) >= 3:
                polygons.append(np.array(coords, dtype=np.float32))

    return polygons


def create_patch_seg_mask(
    polygons: list[np.ndarray],
    location: tuple[int, int],
    patch_size: int
) -> Image.Image:
    """
    Create a binary segmentation mask for a specific WSI patch.

    Parameters
    ----------
    polygons:
        List of glomeruli polygons in WSI level-0 coordinates.
    location:
        Top-left coordinates as tuple of the patch in WSI level-0 coordinates.
    patch_size:
        Size of the patch extracted from the WSI at level 0.

    Returns
    -------
    mask_img:
        Pillow Image containing the binary mask.
        Pixel values:
        0 = non-glomerulus
        1 = glomerulus
    """

    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

    patch_x_min = location[0]
    patch_y_min = location[1]
    patch_x_max = location[0] + patch_size
    patch_y_max = location[1] + patch_size

    for polygon in polygons:
        min_x, min_y = polygon.min(axis=0)
        max_x, max_y = polygon.max(axis=0)

        # Skip polygons that do not intersect the current patch
        if max_x < patch_x_min or min_x > patch_x_max:
            continue
        if max_y < patch_y_min or min_y > patch_y_max:
            continue

        # Convert WSI coordinates to patch-local coordinates
        local_polygon = polygon.copy().astype(np.float32)
        local_polygon[:, 0] -= location[0]
        local_polygon[:, 1] -= location[1]

        local_polygon = np.round(local_polygon).astype(np.int32)

        # skimage.draw.polygon wants rows and columns:
        # rows = y coordinates
        # cols = x coordinates
        rr, cc = skimage.draw.polygon(
            local_polygon[:, 1],
            local_polygon[:, 0],
            shape=mask.shape,
        )

        mask[rr, cc] = 1

    # Convert numpy mask to Pillow Image
    mask_img = Image.fromarray(mask, mode="L")

    return mask_img