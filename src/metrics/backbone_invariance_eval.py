from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def load_rgb_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def apply_augmentation(
    image: Image.Image,
    augmentation_name: str,
) -> Image.Image:
    if augmentation_name == "rot90":
        return image.transpose(Image.Transpose.ROTATE_90)

    if augmentation_name == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)

    if augmentation_name == "rot270":
        return image.transpose(Image.Transpose.ROTATE_270)

    if augmentation_name == "flip_h":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if augmentation_name == "flip_v":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    raise ValueError(f"Augmentation non riconosciuta: {augmentation_name}")


def embedding_to_1d_array(embedding) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim == 1:
        return embedding

    if embedding.ndim == 2 and embedding.shape[0] == 1:
        return embedding[0]

    raise ValueError(
        f"L'embedding deve avere shape (d,) oppure (1, d), "
        f"ma ha shape {embedding.shape}."
    )


def compute_augmentation_robustness_metrics(
    image_paths: Sequence[str | Path],
    embed_fn: Callable[[Image.Image], np.ndarray],
    backbone_name: str,
    augmentations: Sequence[str] = ("rot90", "rot180", "rot270", "flip_h", "flip_v"),
    retrieval_ks: Sequence[int] = (5, 10, 20),
    subset_size: int | None = None,
    random_state: int = 42,
    normalize_l2: bool = True,
) -> pd.DataFrame:
    """
    Calcola solo le metriche principali di robustezza alle augmentations.

    Metriche restituite:
    - self-retrieval@5
    - self-retrieval@10
    - self-retrieval@20
    - median intra/nearest-inter ratio
    """

    image_paths = list(map(Path, image_paths))

    if subset_size is not None:
        if subset_size > len(image_paths):
            raise ValueError(
                f"subset_size={subset_size} è maggiore del numero di immagini: "
                f"{len(image_paths)}."
            )

        rng = np.random.default_rng(random_state)
        selected_indices = rng.choice(
            len(image_paths),
            size=subset_size,
            replace=False,
        )
        image_paths = [image_paths[i] for i in selected_indices]

    n_images = len(image_paths)

    if n_images < 2:
        raise ValueError("Servono almeno 2 immagini.")

    if max(retrieval_ks) > n_images:
        raise ValueError(
            f"Il massimo k richiesto è {max(retrieval_ks)}, "
            f"ma ci sono solo {n_images} immagini."
        )

    original_embeddings = []
    augmented_embeddings = []
    augmented_original_indices = []

    for original_index, path in enumerate(image_paths):
        image = load_rgb_image(path)

        original_embedding = embed_fn(image)
        original_embedding = embedding_to_1d_array(original_embedding)
        original_embeddings.append(original_embedding)

        for augmentation_name in augmentations:
            augmented_image = apply_augmentation(
                image=image,
                augmentation_name=augmentation_name,
            )

            augmented_embedding = embed_fn(augmented_image)
            augmented_embedding = embedding_to_1d_array(augmented_embedding)

            augmented_embeddings.append(augmented_embedding)
            augmented_original_indices.append(original_index)

    original_embeddings = np.vstack(original_embeddings).astype(np.float32)
    augmented_embeddings = np.vstack(augmented_embeddings).astype(np.float32)
    augmented_original_indices = np.asarray(augmented_original_indices)

    if normalize_l2:
        original_embeddings = normalize(original_embeddings, norm="l2", axis=1)
        augmented_embeddings = normalize(augmented_embeddings, norm="l2", axis=1)

    distance_matrix = pairwise_distances(
        augmented_embeddings,
        original_embeddings,
        metric="cosine",
        n_jobs=-1,
    )

    original_ranks = []
    intra_nearest_inter_ratios = []

    for aug_idx, original_index in enumerate(augmented_original_indices):
        distances = distance_matrix[aug_idx]

        intra_distance = distances[original_index]

        inter_distances = np.delete(distances, original_index)
        nearest_inter_distance = np.min(inter_distances)

        intra_nearest_inter_ratio = intra_distance / (nearest_inter_distance + 1e-12)
        intra_nearest_inter_ratios.append(intra_nearest_inter_ratio)

        original_rank = 1 + np.sum(distances < intra_distance)
        original_ranks.append(original_rank)

    original_ranks = np.asarray(original_ranks)
    intra_nearest_inter_ratios = np.asarray(intra_nearest_inter_ratios)

    results = {
        "Backbone": backbone_name,
        "median intra/nearest-inter ratio": float(
            np.median(intra_nearest_inter_ratios)
        ),
    }

    for k in retrieval_ks:
        results[f"self-retrieval@{k}"] = float(np.mean(original_ranks <= k))

    return pd.DataFrame([results])