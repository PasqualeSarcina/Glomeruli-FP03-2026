import igraph as ig
import leidenalg
import numpy as np
import umap
from sklearn.neighbors import NearestNeighbors


def _build_knn_graph(
    X,
    k,
    metric,
    weight_mode
):
    """
    Costruisce un grafo kNN pesato a partire da un embedding.

    X: embedding su cui costruire il grafo, es. UMAP 20D
    k: numero di vicini
    metric: metrica per kNN
    weight_mode:
        - "gaussian": exp(-d^2 / 2 sigma^2), consigliato per euclidean
        - "inverse": 1 / (1 + d)
        - "cosine_similarity": 1 - cosine_distance
        - "binary": tutti gli archi peso 1
    """

    n = X.shape[0]

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
    )
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    neighbor_distances = distances[:, 1:].ravel()
    positive_distances = neighbor_distances[neighbor_distances > 0]

    if len(positive_distances) > 0:
        sigma = np.median(positive_distances)
    else:
        sigma = 1.0

    sigma = max(float(sigma), 1e-12)

    edge_weights = {}

    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            j = int(j)
            dist = float(dist)

            if i == j:
                continue

            if weight_mode == "gaussian":
                weight = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

            elif weight_mode == "inverse":
                weight = 1.0 / (1.0 + dist)

            elif weight_mode == "cosine_similarity":
                weight = max(0.0, 1.0 - dist)

            elif weight_mode == "binary":
                weight = 1.0

            else:
                raise ValueError(
                    "weight_mode must be one of: "
                    "'gaussian', 'inverse', 'cosine_similarity', 'binary'"
                )

            if weight <= 0:
                continue

            # grafo non diretto: ordino la coppia
            a, b = sorted((i, j))

            # se l'arco compare due volte, tengo il peso massimo
            if (a, b) not in edge_weights:
                edge_weights[(a, b)] = weight
            else:
                edge_weights[(a, b)] = max(edge_weights[(a, b)], weight)

    edges = list(edge_weights.keys())
    weights = list(edge_weights.values())

    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es["weight"] = weights

    return graph


def fit_umap_leiden(
        X,
        seed,
        n_components,
        n_neighbors,
        k,
        resolution,
        umap_metric = "euclidean",
        min_dist = 0.05,
        leiden_metric="euclidean",
        n_iterations=-1,
        weight_mode="inverse",
        partition_type="rb",
):
    umap_embedding = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=umap_metric,
        min_dist=min_dist,
        random_state=seed
    ).fit_transform(X)

    graph = _build_knn_graph(
        X=umap_embedding,
        k=k,
        metric=leiden_metric,
        weight_mode=weight_mode,
    )

    if partition_type == "rb":
        partition_cls = leidenalg.RBConfigurationVertexPartition

    elif partition_type == "cpm":
        partition_cls = leidenalg.CPMVertexPartition

    else:
        raise ValueError("partition_type must be 'rb' or 'cpm'")

    partition = leidenalg.find_partition(
        graph,
        partition_cls,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )

    labels = np.asarray(partition.membership, dtype=int)

    return labels, umap_embedding, graph, partition