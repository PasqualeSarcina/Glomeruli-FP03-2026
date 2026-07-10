"""
Metriche di GRADAZIONE CONTINUA per valutare quanto gli embedding di una
backbone catturano una variabile che varia con continuita' (la gravita'
morfologica dei glomeruli), invece di formare cluster discreti.

Motivazione
-----------
Le metriche di clustering (silhouette, DBCV) premiano la separazione in
gruppi netti. Ma se la necrotizzazione e' un continuum (lieve -> moderato
-> grave), quello che serve non e' "quanto separa in cluster" ma "quanto
dispone i glomeruli lungo un asse graduale coerente con la morfologia".

Queste metriche usano un proxy morfologico continuo (vedi
morphology_descriptors.py) come riferimento e misurano quanto la geometria
dello spazio degli embedding e' allineata a quel gradiente:

1. distance_morphology_correlation
   Correlazione di Spearman tra le distanze nello spazio embedding e le
   differenze nel punteggio morfologico. Alta = coppie di glomeruli
   morfologicamente diverse sono anche lontane nell'embedding (l'embedding
   rispetta la gradazione).

2. morphology_neighborhood_consistency
   Per ogni glomerulo, i suoi k vicini nell'embedding sono anche vicini nel
   punteggio morfologico? Misura se la struttura LOCALE preserva la
   gradazione.

3. morphology_gradient_smoothness (indice di Moran)
   Autocorrelazione spaziale del punteggio morfologico sul grafo dei vicini
   nell'embedding: i valori morfologici variano in modo liscio e
   progressivo lungo lo spazio (gradiente continuo) o a salti/casuale?

Tutte e tre sono deterministiche (nessuna componente stocastica).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def distance_morphology_correlation(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    max_pairs: int = 200000,
    random_state: int = 42,
) -> float:
    """
    Correlazione di Spearman tra distanze nello spazio embedding e
    differenze assolute nel punteggio morfologico, su un campione di coppie.

    Valori piu' alti (verso 1) = la distanza nell'embedding cresce in modo
    monotono con la differenza morfologica: l'embedding riflette la
    gradazione continua della gravita'.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    morphology_score = np.asarray(morphology_score, dtype=np.float64)
    n = embeddings.shape[0]

    rng = np.random.default_rng(random_state)

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        emb_dist = pdist(embeddings, metric="euclidean")
        morph_diff = pdist(morphology_score.reshape(-1, 1), metric="cityblock")
    else:
        # campiono coppie casuali (i, j), i < j
        i_idx = rng.integers(0, n, size=max_pairs)
        j_idx = rng.integers(0, n, size=max_pairs)
        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]
        emb_dist = np.linalg.norm(embeddings[i_idx] - embeddings[j_idx], axis=1)
        morph_diff = np.abs(morphology_score[i_idx] - morphology_score[j_idx])

    rho, _ = spearmanr(emb_dist, morph_diff)
    return float(rho)


def morphology_neighborhood_consistency(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    k: int = 15,
) -> float:
    """
    Per ogni punto, misura quanto i k vicini nello spazio embedding sono
    simili nel punteggio morfologico, rispetto a quanto sarebbe atteso a
    caso.

    Implementazione: per ogni punto calcola la deviazione media assoluta del
    punteggio morfologico dei suoi k vicini; la normalizza rispetto alla
    deviazione media assoluta globale. Ritorna 1 - (locale/globale):
      ~1  -> i vicini nell'embedding hanno morfologia molto simile (ottimo)
      ~0  -> i vicini non sono piu' simili di due punti a caso
      <0  -> i vicini sono addirittura piu' diversi della media (pessimo)
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    morphology_score = np.asarray(morphology_score, dtype=np.float64)
    n = embeddings.shape[0]

    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]  # rimuovo il punto stesso

    global_mad = float(np.mean(np.abs(morphology_score - np.mean(morphology_score))))
    if global_mad == 0:
        return float("nan")

    local_mads = []
    for i in range(n):
        neighbor_scores = morphology_score[indices[i]]
        local_mads.append(np.mean(np.abs(neighbor_scores - morphology_score[i])))

    mean_local_mad = float(np.mean(local_mads))
    return float(1.0 - mean_local_mad / global_mad)


def morphology_gradient_smoothness(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    k: int = 15,
) -> float:
    """
    Indice di Moran (autocorrelazione spaziale) del punteggio morfologico
    sul grafo dei k-nearest-neighbor nello spazio embedding.

    Misura se il punteggio morfologico varia in modo liscio lungo lo spazio
    (punti vicini hanno valori simili -> gradiente continuo) oppure in modo
    casuale.
      ~1  -> gradiente molto liscio (forte struttura continua)
      ~0  -> nessuna autocorrelazione (valori casuali nello spazio)
      <0  -> anti-correlazione (vicini sistematicamente diversi)
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    z = np.asarray(morphology_score, dtype=np.float64)
    n = embeddings.shape[0]

    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]

    z_centered = z - np.mean(z)
    denom = np.sum(z_centered ** 2)
    if denom == 0:
        return float("nan")

    # somma pesata (pesi binari 1 per i k vicini)
    numerator = 0.0
    w_total = 0.0
    for i in range(n):
        for j in indices[i]:
            numerator += z_centered[i] * z_centered[j]
            w_total += 1.0

    morans_i = (n / w_total) * (numerator / denom)
    return float(morans_i)


def evaluate_gradation(
    embeddings: np.ndarray,
    morphology_score: np.ndarray,
    k: int = 15,
    random_state: int = 42,
) -> dict:
    """
    Calcola tutte le metriche di gradazione continua per una backbone e le
    ritorna in un dizionario, pronte per una riga di tabella comparativa.
    """

    return {
        "grad_distance_morph_corr": distance_morphology_correlation(
            embeddings, morphology_score, random_state=random_state
        ),
        "grad_neighborhood_consistency": morphology_neighborhood_consistency(
            embeddings, morphology_score, k=k
        ),
        "grad_gradient_smoothness_moran": morphology_gradient_smoothness(
            embeddings, morphology_score, k=k
        ),
    }
