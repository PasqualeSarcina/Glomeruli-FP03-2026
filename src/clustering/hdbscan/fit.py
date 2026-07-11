import hdbscan
import umap

def fit_umap_hdbscan(
        X,

        n_components,
        n_neighbors,
        min_cluster_size,
        min_samples,
        umap_metric,
        min_dist,
        seed=None,
        hdbscan_metric = "euclidean",
        cluster_selection_method = "eom"
):
    umap_embedding = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=umap_metric,
        min_dist=min_dist,
        random_state=seed
    ).fit_transform(X)

    cluster_labels = hdbscan.HDBSCAN(
        min_samples=min_samples,
        metric=hdbscan_metric,
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method
    ).fit_predict(umap_embedding)

    return cluster_labels, umap_embedding

