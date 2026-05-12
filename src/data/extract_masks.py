import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openslide
from scipy.signal import argrelextrema
from skimage.draw import polygon
from skimage.filters.rank import entropy
from skimage.morphology import disk, opening, closing, remove_small_objects
from skimage.morphology import remove_small_holes


def filter_entropy_image(image, threshold, disk_radius=3):
    eimage = entropy(image, disk(disk_radius))
    return eimage < threshold


def extract_tissue_mask(
    slide: openslide.OpenSlide,
    level: int = 2,
    small_objects_max_size: int = 1499,
    small_holes_max_size: int = 5000
) -> np.ndarray:

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

    clean_mask = clean.astype(np.uint8)

    return clean_mask



def extract_seg_mask(annotation_path: str, slide_dimension: tuple, ds: float) -> np.ndarray:
    """
    Extracts a binary segmentation mask from an XML annotation file for a given slide.

    Parameters:
    - annotation_path: Path to the XML annotation file.
    - slide_dimension: Tuple containing the dimensions of the slide at the desired level (width, height).
    - ds: Downsampling factor at the desired level.
    """

    ann_file = ET.parse(Path(annotation_path))
    root = ann_file.getroot()

    mask = np.zeros((slide_dimension[1], slide_dimension[0]), dtype=np.uint8)

    for annotation in root.iter():
        if annotation.tag.lower().endswith("annotation"):
            points = []

            for coord in annotation.iter():
                if coord.tag.lower().endswith("coordinate"):
                    x0 = float(coord.attrib["X"])
                    y0 = float(coord.attrib["Y"])

                    # Coordinate level 0 -> coordinate chooses level
                    x = int(round(x0 / ds))
                    y = int(round(y0 / ds))

                    points.append([x, y])

            if len(points) >= 3:
                points = np.array(points, dtype=np.int32)

                xs = points[:, 0]
                ys = points[:, 1]

                rr, cc = polygon(ys, xs, shape=mask.shape)

                mask[rr, cc] = 1


    return mask.astype(np.uint8)