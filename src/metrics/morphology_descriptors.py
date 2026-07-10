"""
Descrittori morfologici proxy della gravita' (glomerulosclerosi/necrotizzazione)
estratti dai crop dei glomeruli, usando la maschera per limitare le misure
alla sola regione del glomerulo.

Perche' servono
---------------
Non esistono etichette di gravita' annotate da un patologo. Per valutare se
una backbone dispone i glomeruli lungo una GRADAZIONE CONTINUA di gravita'
(invece che in cluster discreti) serve un riferimento morfologico oggettivo
e riproducibile da usare come proxy della gravita'.

I descrittori scelti sono marcatori morfologici noti della glomerulosclerosi:

  - eosin_intensity: intensita' del canale "rosa/eosina". La sclerosi
    deposita matrice extracellulare/collagene, che si colora di eosina;
    piu' alto -> tendenzialmente piu' sclerotico.
  - hematoxylin_intensity: intensita' del canale "blu-viola/ematossilina"
    (nuclei cellulari). La sclerosi sostituisce le cellule con matrice,
    quindi tende a calare; incluso come segnale complementare.
  - texture_homogeneity / texture_contrast: misure GLCM. Un glomerulo sano
    ha struttura ricca (capillari, nuclei -> texture complessa, contrasto
    alto); uno sclerotico diventa piu' omogeneo/liscio. La perdita di
    dettaglio strutturale e' un marcatore forte.
  - area_fraction: frazione di pixel del glomerulo nella patch (proxy
    grezzo di dimensione/atrofia).

Nota metodologica
-----------------
Questo NON e' la vera gravita' clinica: e' un proxy morfologico. Serve solo
come riferimento comune per confrontare le backbone tra loro ("quale dispone
i glomeruli in modo piu' coerente con la morfologia?"), non come ground truth
diagnostico. Va dichiarato esplicitamente nel report.
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

    if mask.sum() < 10:  # maschera vuota/degenere -> usa tutta la patch
        mask = np.ones_like(mask, dtype=bool)

    return rgb, mask


def extract_morphology_descriptor(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    target_size: int = 224,
) -> dict:
    """
    Estrae i descrittori morfologici da un singolo crop di glomerulo,
    misurati solo sui pixel dentro la maschera.

    Ritorna un dizionario di feature scalari.
    """

    rgb, mask = _load_rgb_and_mask(Path(image_path), Path(mask_path) if mask_path else None, target_size)

    # --- Deconvoluzione dei coloranti H&E (Hematoxylin-Eosin-DAB) ----------
    # rgb2hed separa i canali di colorazione: 0 = ematossilina (nuclei, blu),
    # 1 = eosina (matrice/citoplasma, rosa). E' molto piu' robusto che usare
    # i canali RGB grezzi, perche' isola i coloranti istologici reali.
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

    # --- Texture (GLCM) sul grigio, dentro la bounding box della maschera --
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    gray_u8 = (gray * 255).astype(np.uint8)

    ys, xs = np.where(mask_pixels)
    if len(ys) > 0:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = gray_u8[y0:y1, x0:x1]
    else:
        patch = gray_u8

    # riduco i livelli per una GLCM stabile
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
    Estrae i descrittori morfologici per un elenco di crop e li impila in
    una matrice (n_samples, n_descriptors).

    Ritorna (matrice, nomi_descrittori). Le righe sono nello stesso ordine
    di image_paths (fondamentale per allinearle agli embedding).
    """

    if mask_paths is not None and len(mask_paths) != len(image_paths):
        raise ValueError("image_paths e mask_paths devono avere la stessa lunghezza.")

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
    Riduce i descrittori morfologici a un singolo asse continuo di "gravita'
    morfologica" tramite la prima componente principale (dopo
    standardizzazione).

    L'assunzione e' che la direzione di massima variazione morfologica
    corrisponda grosso modo all'asse sano <-> sclerotico. Non e' garantito
    che il segno sia "verso la gravita'", ma per le metriche di gradazione
    (che usano correlazioni e distanze) il segno e' irrilevante.

    Ritorna un vettore (n_samples,) con il punteggio morfologico continuo.
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = np.asarray(morphology_matrix, dtype=np.float64)

    # gestisco eventuali colonne con NaN sostituendo con la mediana
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
