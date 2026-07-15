import numpy as np


def modularity_cluster_consensus(
    consensus_labels,
    graphs,
    resolution=1.0,
    weight_attr="weight",
):
    """
    Average per-cluster modularity contributions across one or more graphs.

    Assumes undirected graphs and RB modularity.

    Returns
    -------
    dict
        Mapping from cluster label to mean modularity contribution.
    """

    labels = np.asarray(consensus_labels)

    # Accetta un singolo grafo oppure una sequenza.
    if hasattr(graphs, "vcount"):
        graphs = [graphs]

    clusters = sorted(set(labels) - {-1})
    modularity_values = {c: [] for c in clusters}

    for graph in graphs:
        if graph.is_directed():
            raise ValueError(
                "This implementation requires undirected graphs."
            )

        if graph.vcount() != len(labels):
            raise ValueError(
                f"Graph has {graph.vcount()} nodes, but labels has "
                f"{len(labels)} samples."
            )

        if (
            weight_attr is not None
            and weight_attr in graph.es.attributes()
        ):
            weights = np.asarray(
                graph.es[weight_attr],
                dtype=float,
            )
        else:
            weights = np.ones(graph.ecount(), dtype=float)

        if np.any(weights < 0):
            raise ValueError(
                "Modularity requires non-negative edge weights."
            )

        total_edge_weight = np.sum(weights)

        if total_edge_weight <= 0:
            for c in clusters:
                modularity_values[c].append(np.nan)
            continue

        strengths = np.zeros(graph.vcount(), dtype=float)
        internal_weights = {c: 0.0 for c in clusters}

        for (u, v), weight in zip(
            graph.get_edgelist(),
            weights,
        ):
            if u == v:
                strengths[u] += 2.0 * weight
            else:
                strengths[u] += weight
                strengths[v] += weight

            if labels[u] == labels[v] and labels[u] in internal_weights:
                internal_weights[labels[u]] += weight

        for c in clusters:
            cluster_strength = np.sum(
                strengths[labels == c]
            )

            contribution = (
                internal_weights[c] / total_edge_weight
                - resolution
                * (
                    cluster_strength
                    / (2.0 * total_edge_weight)
                ) ** 2
            )

            modularity_values[c].append(float(contribution))

    return {
        c: (
            float(np.nanmean(values))
            if np.any(np.isfinite(values))
            else np.nan
        )
        for c, values in modularity_values.items()
    }