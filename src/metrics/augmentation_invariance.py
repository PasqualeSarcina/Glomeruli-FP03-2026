"""
Robustezza dell'embedding di una backbone alle trasformazioni geometriche
che non cambiano l'identita' biologica del glomerulo (rotazioni di 90/180/270
gradi, flip orizzontale/verticale).

Perche' serve
-------------
Un glomerulo ritagliato da una WSI non ha un orientamento "corretto": e'
tessuto tagliato, non un oggetto con un verso naturale. Lo stesso glomerulo
ruotato o specchiato e' biologicamente lo stesso oggetto e dovrebbe finire
vicino a se stesso nello spazio dell'embedding. Se una backbone e' sensibile
all'orientamento, l'assegnazione a un cluster puo' dipendere dal verso con
cui il glomerulo e' stato ritagliato invece che dalla sua reale gravita'
istologica: questo introduce rumore artificiale nella classificazione finale.

Differenza rispetto a backbone_invariance_eval.py originale
-------------------------------------------------------------
La versione originale accetta solo l'immagine. Qui l'embed_fn ha la stessa
firma di src.backbones.backbone.Backbone.__call__(image, mask): alcune
backbone (es. DinoV2/DinoV3 in modalita' "patch" o "both") usano la maschera
per pesare i patch token sulla sola regione del glomerulo, quindi la maschera
deve essere ruotata/specchiata insieme all'immagine, altrimenti la metrica
di invarianza sarebbe scorretta (penalizzerebbe la backbone per un
disallineamento introdotto da noi, non per un suo limite reale).
"""

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

AUGMENTATION_TO_PIL_TRANSPOSE = {
    "rot90": Image.Transpose.ROTATE_90,
    "rot180": Image.Transpose.ROTATE_180,
    "rot270": Image.Transpose.ROTATE_270,
    "flip_h": Image.Transpose.FLIP_LEFT_RIGHT,
    "flip_v": Image.Transpose.FLIP_TOP_BOTTOM,
}


def load_rgb_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def load_mask_image(path: str | Path) -> Image.Image:
    mask = Image.open(path)
    mask = ImageOps.exif_transpose(mask)
    return mask.convert("L")


def apply_augmentation(image: Image.Image, augmentation_name: str) -> Image.Image:
    if augmentation_name not in AUGMENTATION_TO_PIL_TRANSPOSE:
        raise ValueError(f"Augmentation non riconosciuta: {augmentation_name}")
    return image.transpose(AUGMENTATION_TO_PIL_TRANSPOSE[augmentation_name])


def embedding_to_1d_array(embedding) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim == 1:
        return embedding
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        return embedding[0]

    raise ValueError(
        f"L'embedding deve avere shape (d,) oppure (1, d), ma ha shape {embedding.shape}."
    )


def compute_augmentation_robustness_metrics(
    image_paths: Sequence[str | Path],
    embed_fn: Callable[[Image.Image, Image.Image | None], np.ndarray],
    mask_paths: Sequence[str | Path] | None = None,
    augmentations: Sequence[str] = ("rot90", "rot180", "rot270", "flip_h", "flip_v"),
    retrieval_ks: Sequence[int] = (5, 10, 20),
    subset_size: int | None = None,
    random_state: int = 42,
    normalize_l2: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Calcola le metriche di robustezza alle augmentation, sia aggregate su
    tutte le trasformazioni sia per singola trasformazione.

    Se mask_paths e' None, embed_fn viene chiamata con mask=None (va bene
    per backbone in modalita' "cls", che non usano la maschera).

    Ritorna un dizionario con due DataFrame:
    - "overall": una riga, metriche aggregate su tutte le augmentation
    - "by_augmentation": una riga per augmentation, stesse metriche
    """

    image_paths = list(map(Path, image_paths))
    use_masks = mask_paths is not None
    if use_masks:
        mask_paths = list(map(Path, mask_paths))
        if len(mask_paths) != len(image_paths):
            raise ValueError(
                f"image_paths ({len(image_paths)}) e mask_paths ({len(mask_paths)}) "
                "devono avere la stessa lunghezza."
            )

    if subset_size is not None and subset_size < len(image_paths):
        rng = np.random.default_rng(random_state)
        selected_indices = np.sort(
            rng.choice(len(image_paths), size=subset_size, replace=False)
        )
        image_paths = [image_paths[i] for i in selected_indices]
        if use_masks:
            mask_paths = [mask_paths[i] for i in selected_indices]

    n_images = len(image_paths)
    if n_images < 2:
        raise ValueError("Servono almeno 2 immagini.")
    if max(retrieval_ks) > n_images:
        raise ValueError(
            f"Il massimo k richiesto e' {max(retrieval_ks)}, ma ci sono solo {n_images} immagini."
        )

    original_embeddings = []
    # Per ogni augmentation teniamo separati gli embedding, per poter
    # calcolare sia le metriche aggregate sia quelle per-augmentation.
    augmented_embeddings_by_aug: dict[str, list[np.ndarray]] = {
        aug: [] for aug in augmentations
    }

    for i, path in enumerate(image_paths):
        image = load_rgb_image(path)
        mask = load_mask_image(mask_paths[i]) if use_masks else None

        original_embedding = embed_fn(image, mask)
        original_embeddings.append(embedding_to_1d_array(original_embedding))

        for augmentation_name in augmentations:
            augmented_image = apply_augmentation(image, augmentation_name)
            augmented_mask = (
                apply_augmentation(mask, augmentation_name) if use_masks else None
            )

            augmented_embedding = embed_fn(augmented_image, augmented_mask)
            augmented_embeddings_by_aug[augmentation_name].append(
                embedding_to_1d_array(augmented_embedding)
            )

    original_embeddings = np.vstack(original_embeddings).astype(np.float32)

    if normalize_l2:
        original_embeddings = normalize(original_embeddings, norm="l2", axis=1)

    def compute_metrics_for_augmented(augmented_embeddings: np.ndarray) -> dict:
        if normalize_l2:
            augmented_embeddings = normalize(augmented_embeddings, norm="l2", axis=1)

        distance_matrix = pairwise_distances(
            augmented_embeddings, original_embeddings, metric="cosine", n_jobs=-1
        )

        original_ranks = []
        intra_nearest_inter_ratios = []

        for idx in range(distance_matrix.shape[0]):
            distances = distance_matrix[idx]
            intra_distance = distances[idx]

            inter_distances = np.delete(distances, idx)
            nearest_inter_distance = np.min(inter_distances)

            intra_nearest_inter_ratios.append(
                intra_distance / (nearest_inter_distance + 1e-12)
            )
            original_ranks.append(1 + int(np.sum(distances < intra_distance)))

        original_ranks = np.asarray(original_ranks)
        intra_nearest_inter_ratios = np.asarray(intra_nearest_inter_ratios)

        result = {
            "median intra/nearest-inter ratio": float(
                np.median(intra_nearest_inter_ratios)
            ),
        }
        for k in retrieval_ks:
            result[f"self-retrieval@{k}"] = float(np.mean(original_ranks <= k))
        return result

    per_augmentation_rows = []
    all_augmented_embeddings = []

    for augmentation_name in augmentations:
        augmented_embeddings = np.vstack(
            augmented_embeddings_by_aug[augmentation_name]
        ).astype(np.float32)
        all_augmented_embeddings.append(augmented_embeddings)

        row = {"augmentation": augmentation_name}
        row.update(compute_metrics_for_augmented(augmented_embeddings))
        per_augmentation_rows.append(row)

    # L'overall confronta ogni versione augmentata (di qualunque tipo) con
    # il proprio originale: ogni blocco di n_images righe in
    # all_augmented_embeddings corrisponde, nell'ordine, agli n_images
    # originali (stesso ordine di image_paths).
    repeated_original_indices = np.tile(np.arange(n_images), len(augmentations))
    stacked_augmented = np.vstack(all_augmented_embeddings)

    if normalize_l2:
        stacked_augmented_norm = normalize(stacked_augmented, norm="l2", axis=1)
    else:
        stacked_augmented_norm = stacked_augmented

    distance_matrix_overall = pairwise_distances(
        stacked_augmented_norm, original_embeddings, metric="cosine", n_jobs=-1
    )

    original_ranks_overall = []
    ratios_overall = []
    for row_idx, original_index in enumerate(repeated_original_indices):
        distances = distance_matrix_overall[row_idx]
        intra_distance = distances[original_index]
        inter_distances = np.delete(distances, original_index)
        nearest_inter_distance = np.min(inter_distances)
        ratios_overall.append(intra_distance / (nearest_inter_distance + 1e-12))
        original_ranks_overall.append(1 + int(np.sum(distances < intra_distance)))

    original_ranks_overall = np.asarray(original_ranks_overall)
    ratios_overall = np.asarray(ratios_overall)

    overall_row = {
        "median intra/nearest-inter ratio": float(np.median(ratios_overall)),
    }
    for k in retrieval_ks:
        overall_row[f"self-retrieval@{k}"] = float(
            np.mean(original_ranks_overall <= k)
        )

    return {
        "overall": pd.DataFrame([overall_row]),
        "by_augmentation": pd.DataFrame(per_augmentation_rows),
    }
