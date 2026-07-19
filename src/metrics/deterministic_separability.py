"""
Metriche di separabilità DETERMINISTICHE per confrontare backbone diverse.

Motivazione
-----------
UMAP e HDBSCAN hanno componenti stocastiche (inizializzazioni casuali,
sampling interno). Se si usano per confrontare backbone diverse, una parte
della varianza osservata tra backbone è in realta' dovuta al seed, non alla
qualita' dell'embedding.

Il clustering gerarchico agglomerativo con linkage "ward" e' invece
completamente deterministico (nessun seed, nessuna inizializzazione
casuale): a parita' di dati, il risultato e' sempre lo stesso. Per questo
e' usato qui come "sonda" di separabilita', non come pipeline finale di
clustering del progetto.

Le due famiglie di metriche in questo modulo:

1. effective_dimensionality(...)
   Quanto l'embedding e' concentrato su poche direzioni principali
   (PCA). Un embedding troppo "piatto" (varianza spalmata su moltissime
   componenti) tende a essere meno separabile in pratica.

2. ward_clustering_curve(...) / summarize_ward_curve(...)
   Silhouette / Davies-Bouldin / Calinski-Harabasz calcolati con Ward
   su un range di k. Poiche' Ward e' deterministico, la curva ottenuta
   e' riproducibile al 100% e permette un confronto equo tra backbone.
"""


import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def effective_dimensionality(
    X: np.ndarray,
    variance_targets: tuple[float, ...] = (0.90, 0.95, 0.99),
    random_state: int = 42,
) -> dict:
    """
    Calcola la dimensionalita' effettiva di un embedding tramite PCA.

    Ritorna, per ciascuna soglia di varianza spiegata richiesta, il numero
    di componenti necessarie, sia in valore assoluto che come frazione
    della dimensionalita' originale (piu' basso = embedding piu'
    concentrato/strutturato).

    Include anche il "participation ratio" (PR), una misura continua
    della dimensionalita' effettiva basata sugli autovalori:

        PR = (sum(lambda_i))^2 / sum(lambda_i^2)

    PR vale 1 se tutta la varianza e' su una sola componente, e vale
    n_features se la varianza e' distribuita uniformemente su tutte.
    """

    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    max_components = min(n_samples, n_features)

    pca = PCA(n_components=max_components, svd_solver="full", random_state=random_state)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    result = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
    }

    for target in variance_targets:
        n_components_needed = int(np.searchsorted(cumulative, target) + 1)
        n_components_needed = min(n_components_needed, max_components)

        key = f"{int(target * 100)}"
        result[f"n_components_var{key}"] = n_components_needed
        result[f"frac_dims_var{key}"] = float(n_components_needed / n_features)

    eigenvalues = pca.explained_variance_
    participation_ratio = float(
        (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    )

    result["participation_ratio"] = participation_ratio
    result["participation_ratio_frac"] = float(participation_ratio / n_features)

    return result


def ward_clustering_curve(
    X: np.ndarray,
    k_values: tuple[int, ...] = tuple(range(2, 16)),
    max_samples_for_metrics: int | None = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcola una curva di metriche interne di clustering usando Ward
    (deterministico) per ciascun valore di k in k_values.

    Se il dataset e' molto grande, le metriche interne (silhouette in
    particolare, che e' O(n^2)) vengono calcolate su un sottoinsieme
    campionato in modo riproducibile; il clustering stesso viene invece
    fatto su tutti i punti disponibili.

    Ritorna un DataFrame con una riga per k, colonne:
    k, silhouette, davies_bouldin, calinski_harabasz,
    min_cluster_size, max_cluster_size.
    """

    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    rng = np.random.default_rng(random_state)

    if max_samples_for_metrics is not None and n_samples > max_samples_for_metrics:
        metric_idx = np.sort(
            rng.choice(n_samples, size=max_samples_for_metrics, replace=False)
        )
    else:
        metric_idx = np.arange(n_samples)

    rows = []

    for k in k_values:
        if k < 2 or k >= n_samples:
            continue

        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)

        counts = np.bincount(labels, minlength=k)

        labels_eval = labels[metric_idx]
        X_eval = X[metric_idx]

        row = {
            "k": int(k),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
        }

        if len(np.unique(labels_eval)) < 2:
            row.update(
                {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
            )
        else:
            try:
                row["silhouette"] = float(
                    silhouette_score(X_eval, labels_eval, metric="euclidean")
                )
            except Exception:
                row["silhouette"] = np.nan

            try:
                row["davies_bouldin"] = float(davies_bouldin_score(X_eval, labels_eval))
            except Exception:
                row["davies_bouldin"] = np.nan

            try:
                row["calinski_harabasz"] = float(
                    calinski_harabasz_score(X_eval, labels_eval)
                )
            except Exception:
                row["calinski_harabasz"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_ward_curve(curve_df: pd.DataFrame) -> dict:
    """
    Riassume la curva Ward in poche metriche scalari, comode per
    confrontare backbone diverse in una singola tabella:

    - best_silhouette / best_k_silhouette: picco della curva Silhouette
      e k al quale si ottiene (piu' alto e' meglio: -1..1).
    - best_calinski_harabasz / best_k_calinski_harabasz
    - best_davies_bouldin / best_k_davies_bouldin (piu' basso e' meglio)
    - mean_silhouette: media della curva, per non premiare solo un k
      "fortunato".
    """

    if curve_df.empty:
        return {
            "best_silhouette": np.nan,
            "best_k_silhouette": None,
            "mean_silhouette": np.nan,
            "best_calinski_harabasz": np.nan,
            "best_k_calinski_harabasz": None,
            "best_davies_bouldin": np.nan,
            "best_k_davies_bouldin": None,
        }

    result = {}

    if curve_df["silhouette"].notna().any():
        idx_best_sil = curve_df["silhouette"].idxmax()
        result["best_silhouette"] = float(curve_df.loc[idx_best_sil, "silhouette"])
        result["best_k_silhouette"] = int(curve_df.loc[idx_best_sil, "k"])
        result["mean_silhouette"] = float(curve_df["silhouette"].mean(skipna=True))
    else:
        result["best_silhouette"] = np.nan
        result["best_k_silhouette"] = None
        result["mean_silhouette"] = np.nan

    if curve_df["calinski_harabasz"].notna().any():
        idx_best_ch = curve_df["calinski_harabasz"].idxmax()
        result["best_calinski_harabasz"] = float(
            curve_df.loc[idx_best_ch, "calinski_harabasz"]
        )
        result["best_k_calinski_harabasz"] = int(curve_df.loc[idx_best_ch, "k"])
    else:
        result["best_calinski_harabasz"] = np.nan
        result["best_k_calinski_harabasz"] = None

    if curve_df["davies_bouldin"].notna().any():
        idx_best_db = curve_df["davies_bouldin"].idxmin()
        result["best_davies_bouldin"] = float(curve_df.loc[idx_best_db, "davies_bouldin"])
        result["best_k_davies_bouldin"] = int(curve_df.loc[idx_best_db, "k"])
    else:
        result["best_davies_bouldin"] = np.nan
        result["best_k_davies_bouldin"] = None

    return result


def pca_reduce(
    X: np.ndarray,
    variance_target: float = 0.95,
    max_components: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Riduce X tramite PCA mantenendo variance_target della varianza.

    PCA e' deterministico (a parita' di dati e seed dell'SVD randomizzato
    da', nella pratica, lo stesso risultato), quindi il confronto tra
    backbone resta riproducibile, a differenza di UMAP.

    Questo serve a valutare la separabilita' nello STESSO tipo di spazio
    ridotto che verra' usato nella pipeline vera (dove il clustering non
    avviene mai sullo spazio grezzo ad alta dimensione, ma dopo riduzione
    dimensionale). In spazi con migliaia di dimensioni le distanze
    euclidee si appiattiscono (maledizione della dimensionalita') e la
    silhouette risulta artificialmente bassa: valutare dopo PCA da' un
    quadro piu' realistico.

    Ritorna (X_ridotto, info) dove info contiene il numero di componenti
    effettivamente usate e la varianza spiegata cumulata.
    """

    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    upper = min(n_samples, n_features)
    if max_components is not None:
        upper = min(upper, max_components)

    pca = PCA(n_components=variance_target, svd_solver="full", random_state=random_state)
    X_reduced = pca.fit_transform(X)

    n_used = int(pca.n_components_)
    if max_components is not None and n_used > max_components:
        # Ricalcolo troncando al numero massimo di componenti richiesto.
        pca = PCA(n_components=max_components, svd_solver="full", random_state=random_state)
        X_reduced = pca.fit_transform(X)
        n_used = int(pca.n_components_)

    info = {
        "pca_n_components": n_used,
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "pca_variance_target": float(variance_target),
    }

    return X_reduced.astype(np.float64), info


def hdbscan_clustering_metrics(
    X: np.ndarray,
    min_cluster_sizes: tuple[int, ...] = (10, 20, 40),
    min_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sonda di clustering basata su densita' con HDBSCAN (deterministico).

    A differenza di Ward, HDBSCAN:
      - non richiede di fissare il numero di cluster: lo scopre da solo;
      - puo' marcare punti come rumore (label -1), utile con dataset
        sbilanciati dove alcuni glomeruli rari non formano un cluster netto;
      - trova cluster di forma e densita' arbitraria, non solo sferici.

    HDBSCAN (l'implementazione di scikit-learn, sklearn.cluster.HDBSCAN) e'
    DETERMINISTICO: a parita' di dati e parametri restituisce sempre lo
    stesso risultato, nessun seed. La stocasticita' della pipeline reale
    viene da UMAP, non da HDBSCAN; qui HDBSCAN gira direttamente sullo
    spazio PCA (deterministico), quindi tutto resta riproducibile.

    Viene provato un range di min_cluster_size (analogo al range di k di
    Ward) e per ciascuno si calcolano:
      - n_clusters: numero di cluster trovati (escluso il rumore)
      - noise_fraction: frazione di punti marcati come rumore
      - silhouette_no_noise: silhouette calcolata SOLO sui punti
        clusterizzati (il rumore falserebbe la metrica). NaN se restano
        meno di 2 cluster dopo aver escluso il rumore.
      - largest_cluster_fraction: frazione di punti nel cluster piu' grande
        (per accorgersi se collassa tutto in un unico gruppone)

    Ritorna un DataFrame con una riga per min_cluster_size.
    """

    from sklearn.cluster import HDBSCAN

    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    rows = []
    for mcs in min_cluster_sizes:
        if mcs >= n_samples:
            continue

        model = HDBSCAN(
            min_cluster_size=int(mcs),
            min_samples=min_samples,
            copy=True,
        )
        labels = model.fit_predict(X)

        noise_mask = labels == -1
        n_noise = int(np.sum(noise_mask))
        cluster_labels = labels[~noise_mask]
        unique_clusters = np.unique(cluster_labels)
        n_clusters = int(len(unique_clusters))

        row = {
            "min_cluster_size": int(mcs),
            "n_clusters": n_clusters,
            "noise_fraction": float(n_noise / n_samples),
        }

        if n_clusters >= 1:
            counts = np.array([np.sum(cluster_labels == c) for c in unique_clusters])
            row["largest_cluster_fraction"] = float(counts.max() / n_samples)
        else:
            row["largest_cluster_fraction"] = np.nan

        # Silhouette solo sui punti clusterizzati, e solo se restano >= 2 cluster.
        if n_clusters >= 2:
            try:
                row["silhouette_no_noise"] = float(
                    silhouette_score(X[~noise_mask], cluster_labels, metric="euclidean")
                )
            except Exception:
                row["silhouette_no_noise"] = np.nan
            # DBCV: metrica di validazione specifica per cluster di densita'
            # (Moulavi et al. 2014). Calcolata sull'intero set con le label
            # HDBSCAN (il rumore e' escluso internamente dalla funzione).
            try:
                row["dbcv"] = density_based_clustering_validation(X, labels, noise_label=-1)
            except Exception:
                row["dbcv"] = np.nan
        else:
            row["silhouette_no_noise"] = np.nan
            row["dbcv"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_hdbscan_metrics(hdbscan_df: pd.DataFrame) -> dict:
    """
    Riassume la tabella HDBSCAN in poche metriche scalari, scegliendo la
    configurazione "migliore" come quella con la silhouette_no_noise piu'
    alta tra quelle che trovano almeno 2 cluster e non collassano tutto
    nel rumore.

    Ritorna:
      - hdbscan_best_silhouette / hdbscan_best_min_cluster_size
      - hdbscan_best_n_clusters
      - hdbscan_best_noise_fraction
      - hdbscan_min_noise_fraction: il minimo rumore osservato (per capire
        se HDBSCAN riesce a spiegare la maggior parte dei punti)
    """

    if hdbscan_df.empty:
        return {
            "hdbscan_best_silhouette": np.nan,
            "hdbscan_best_min_cluster_size": None,
            "hdbscan_best_n_clusters": None,
            "hdbscan_best_noise_fraction": np.nan,
            "hdbscan_min_noise_fraction": np.nan,
        }

    valid = hdbscan_df[hdbscan_df["silhouette_no_noise"].notna()]

    result = {
        "hdbscan_min_noise_fraction": float(hdbscan_df["noise_fraction"].min()),
    }

    if not valid.empty:
        idx = valid["silhouette_no_noise"].idxmax()
        result["hdbscan_best_silhouette"] = float(valid.loc[idx, "silhouette_no_noise"])
        result["hdbscan_best_min_cluster_size"] = int(valid.loc[idx, "min_cluster_size"])
        result["hdbscan_best_n_clusters"] = int(valid.loc[idx, "n_clusters"])
        result["hdbscan_best_noise_fraction"] = float(valid.loc[idx, "noise_fraction"])
    else:
        result["hdbscan_best_silhouette"] = np.nan
        result["hdbscan_best_min_cluster_size"] = None
        result["hdbscan_best_n_clusters"] = None
        result["hdbscan_best_noise_fraction"] = np.nan

    # Miglior DBCV (metrica di densita' dedicata, Moulavi 2014): riportato
    # separatamente perche' e' l'indice piu' appropriato per HDBSCAN.
    if "dbcv" in hdbscan_df.columns and hdbscan_df["dbcv"].notna().any():
        idx_dbcv = hdbscan_df["dbcv"].idxmax()
        result["hdbscan_best_dbcv"] = float(hdbscan_df.loc[idx_dbcv, "dbcv"])
        result["hdbscan_best_dbcv_n_clusters"] = int(hdbscan_df.loc[idx_dbcv, "n_clusters"])
        result["hdbscan_best_dbcv_min_cluster_size"] = int(
            hdbscan_df.loc[idx_dbcv, "min_cluster_size"]
        )
    else:
        result["hdbscan_best_dbcv"] = np.nan
        result["hdbscan_best_dbcv_n_clusters"] = None
        result["hdbscan_best_dbcv_min_cluster_size"] = None

    return result


def density_based_clustering_validation(
    X: np.ndarray,
    labels: np.ndarray,
    noise_label: int = -1,
) -> float:
    """
    DBCV — Density-Based Clustering Validation (Moulavi et al., 2014).

    Metrica interna di validazione pensata specificamente per clustering
    basato su densita' (come HDBSCAN), a differenza della silhouette che
    assume cluster convessi/sferici. Va da -1 a +1: valori piu' alti
    indicano cluster piu' densi internamente e meglio separati.

    A differenza della silhouette, DBCV:
      - gestisce cluster di forma arbitraria (non solo sferici);
      - tiene conto della densita' interna dei cluster, non solo delle
        distanze medie.

    Riferimento:
      Moulavi, Jaskowiak, Campello, Zimek, Sander (2014),
      "Density-Based Clustering Validation", SDM 2014.

    Parametri
    ---------
    X: array (n_samples, n_features)
    labels: etichette di cluster; i punti con label == noise_label sono
            esclusi dal calcolo (coerentemente con la definizione DBCV,
            che valuta solo i punti effettivamente clusterizzati).

    Ritorna il DBCV globale (media pesata sui cluster). NaN se ci sono
    meno di 2 cluster validi.

    Nota implementativa: questa e' un'implementazione diretta della
    definizione del paper (mutual reachability distance -> MST per
    cluster -> density sparseness/separation). E' O(sum_i n_i^2) sui
    punti di ciascun cluster, adeguata a dataset di dimensioni moderate
    come quello dei glomeruli.
    """

    from scipy.spatial.distance import cdist
    from scipy.sparse.csgraph import minimum_spanning_tree

    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)

    core_mask = labels != noise_label
    X_core = X[core_mask]
    labels_core = labels[core_mask]

    unique_labels = np.unique(labels_core)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return float("nan")

    n_total = X_core.shape[0]
    n_features = X_core.shape[1]

    # --- core distance (apts) di ogni punto, cluster per cluster ----------
    # apts_i = ( mean_j ( 1 / d(i,j)^dim ) ) ^ (-1/dim), j != i nello stesso cluster
    all_core_dist = {}
    intra_index = {}
    for lab in unique_labels:
        idx = np.where(labels_core == lab)[0]
        intra_index[lab] = idx
        pts = X_core[idx]
        n_i = len(idx)
        if n_i <= 1:
            continue
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)

        # Core distance (all-points-core-distance del paper DBCV). Due
        # accorgimenti numerici:
        #  1) se due punti coincidono (D=0), D**(-n_features) darebbe inf:
        #     lo escludiamo trattandolo come contributo nullo (riga
        #     inv[~isfinite]=0), come gia' fatto.
        #  2) se la somma degli inversi e' 0 (punto isolatissimo) oppure il
        #     punto ha dei duplicati esatti, l'elevamento a -1/n_features
        #     puo' dividere per zero. Proteggiamo la base con un epsilon e
        #     forziamo core=0 per i punti con duplicati esatti (densita'
        #     localmente infinita => core distance nulla).
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = D ** (-n_features)
        has_exact_duplicate = np.any(~np.isfinite(inv), axis=1)
        inv[~np.isfinite(inv)] = 0.0

        inv_sum = inv.sum(axis=1)
        mean_inv = inv_sum / (n_i - 1)

        core = np.zeros(n_i, dtype=np.float64)
        # Punti con almeno un duplicato esatto: densita' infinita -> core = 0.
        # Punti "normali" (mean_inv > 0): formula standard.
        valid = (mean_inv > 0) & (~has_exact_duplicate)
        with np.errstate(divide="ignore", invalid="ignore"):
            core[valid] = mean_inv[valid] ** (-1.0 / n_features)
        # Punti isolati (mean_inv == 0, nessun vicino utile): core distance
        # molto grande -> usiamo la massima distanza finita nel cluster.
        isolated = (mean_inv == 0) & (~has_exact_duplicate)
        if np.any(isolated):
            finite_D = D.copy()
            finite_D[~np.isfinite(finite_D)] = 0.0
            core[isolated] = finite_D.max()

        for local_j, global_i in enumerate(idx):
            all_core_dist[global_i] = float(core[local_j])

    def mutual_reachability(i, j, dij):
        return max(all_core_dist.get(i, 0.0), all_core_dist.get(j, 0.0), dij)

    # --- density sparseness interna (DSC) via MST del grafo mutual-reach --
    dsc = {}
    for lab in unique_labels:
        idx = intra_index[lab]
        n_i = len(idx)
        if n_i <= 1:
            dsc[lab] = 0.0
            continue
        pts = X_core[idx]
        D = cdist(pts, pts)
        M = np.zeros_like(D)
        for a in range(n_i):
            for b in range(a + 1, n_i):
                mr = mutual_reachability(idx[a], idx[b], D[a, b])
                M[a, b] = mr
                M[b, a] = mr
        mst = minimum_spanning_tree(M).toarray()
        # density sparseness = max edge dell'MST (edge interno piu' "largo")
        dsc[lab] = float(mst.max()) if mst.size else 0.0

    # --- density separation tra cluster (DSPC) ----------------------------
    def cluster_separation(lab_a, lab_b):
        ia, ib = intra_index[lab_a], intra_index[lab_b]
        D = cdist(X_core[ia], X_core[ib])
        best = np.inf
        for a in range(len(ia)):
            for b in range(len(ib)):
                mr = mutual_reachability(ia[a], ib[b], D[a, b])
                if mr < best:
                    best = mr
        return best

    # --- validity index per cluster e media pesata ------------------------
    total_score = 0.0
    for lab in unique_labels:
        min_sep = np.inf
        for other in unique_labels:
            if other == lab:
                continue
            sep = cluster_separation(lab, other)
            if sep < min_sep:
                min_sep = sep
        denom = max(min_sep, dsc[lab])
        v = 0.0 if denom == 0 else (min_sep - dsc[lab]) / denom
        weight = len(intra_index[lab]) / n_total
        total_score += weight * v

    return float(total_score)