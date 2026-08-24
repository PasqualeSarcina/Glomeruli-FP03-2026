"""
Morphological descriptors used as a proxy for glomerulosclerosis severity.
Not a clinical ground truth: only a reproducible reference to compare backbones.
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from skimage.feature import graycomatrix, graycoprops


def _load_rgb_and_mask(
    image_path: Path,
    mask_path: Path | None,
    target_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB").resize((target_size, target_size))
    rgb = np.asarray(image, dtype=np.float64) / 255.0

    if mask_path is not None and Path(mask_path).exists():
        mask = Image.open(mask_path).convert("L").resize((target_size, target_size))
        mask = np.asarray(mask, dtype=np.float64) / 255.0
        mask = mask > 0.5
    else:
        mask = np.ones((target_size, target_size), dtype=bool)

    if mask.sum() < 10:  # degenerate mask -> fall back to the whole patch
        mask = np.ones_like(mask, dtype=bool)

    return rgb, mask


def extract_morphology_descriptor(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    target_size: int = 224,
) -> dict:
    """Extract the scalar descriptors of a single crop, measured inside the mask."""

    rgb, mask = _load_rgb_and_mask(Path(image_path), Path(mask_path) if mask_path else None, target_size)

    # rgb2hed channels: 0 = hematoxylin (nuclei), 1 = eosin (matrix/cytoplasm).
    hed = rgb2hed(rgb)
    hematoxylin = hed[..., 0]
    eosin = hed[..., 1]

    mask_pixels = mask

    eosin_vals = eosin[mask_pixels]
    hema_vals = hematoxylin[mask_pixels]

    result = {
        "eosin_intensity": float(np.mean(eosin_vals)),
        "eosin_std": float(np.std(eosin_vals)),
        "hematoxylin_intensity": float(np.mean(hema_vals)),
        "eosin_hema_ratio": float(np.mean(eosin_vals) / (np.mean(hema_vals) + 1e-6)),
        "area_fraction": float(mask_pixels.sum() / mask_pixels.size),
    }

    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    gray_u8 = (gray * 255).astype(np.uint8)

    ys, xs = np.where(mask_pixels)
    if len(ys) > 0:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = gray_u8[y0:y1, x0:x1]
    else:
        patch = gray_u8

    # fewer grey levels keep the GLCM stable
    levels = 32
    patch_q = (patch.astype(np.float64) / 256 * levels).astype(np.uint8)
    patch_q = np.clip(patch_q, 0, levels - 1)

    try:
        glcm = graycomatrix(
            patch_q,
            distances=[1],
            angles=[0, np.pi / 2],
            levels=levels,
            symmetric=True,
            normed=True,
        )
        result["texture_contrast"] = float(graycoprops(glcm, "contrast").mean())
        result["texture_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
        result["texture_energy"] = float(graycoprops(glcm, "energy").mean())
    except Exception:
        result["texture_contrast"] = np.nan
        result["texture_homogeneity"] = np.nan
        result["texture_energy"] = np.nan

    return result


def build_morphology_matrix(
    image_paths: list[str | Path],
    mask_paths: list[str | Path] | None = None,
    target_size: int = 224,
) -> tuple[np.ndarray, list[str]]:
    """
    Stack the descriptors of every crop into an (n_samples, n_descriptors) matrix.
    Rows follow the order of image_paths, so they align with the embeddings.
    """

    if mask_paths is not None and len(mask_paths) != len(image_paths):
        raise ValueError("image_paths and mask_paths must have the same length.")

    rows = []
    feature_names: list[str] | None = None

    for i, img_path in enumerate(image_paths):
        m_path = mask_paths[i] if mask_paths is not None else None
        desc = extract_morphology_descriptor(img_path, m_path, target_size)
        if feature_names is None:
            feature_names = sorted(desc.keys())
        rows.append([desc[k] for k in feature_names])

    matrix = np.asarray(rows, dtype=np.float64)
    return matrix, feature_names


def morphology_severity_axis(
    morphology_matrix: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce the descriptors to a single continuous severity axis (PC1 after
    standardisation). The sign is not guaranteed to point towards severity,
    but the gradation metrics use correlations and distances, so it does not matter.
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = np.asarray(morphology_matrix, dtype=np.float64)

    # replace non-finite values with the column median
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.any(~np.isfinite(col)):
            med = np.nanmedian(col[np.isfinite(col)]) if np.any(np.isfinite(col)) else 0.0
            col[~np.isfinite(col)] = med
            X[:, j] = col

    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1, random_state=random_state)
    axis = pca.fit_transform(X_std)[:, 0]
    return axis.astype(np.float64)
