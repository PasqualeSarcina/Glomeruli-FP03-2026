import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


from clustering.consensus import consensus_clustering
from clustering.cosine import mean_intracluster_cosine
from clustering.hdbscan.consensus_dbcv import dbcv_cluster_consensus
from clustering.hdbscan.fit import fit_umap_hdbscan
from clustering.hdbscan.params import make_umap_hdbscan_grid_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, required=True,
                        help="input name (without the extension) of the embedding + csv to process")


def main():
    args = parse_args()

    try:
        embedding = np.load(args.input.with_suffix(".npy"))
        csv = pd.read_csv(args.input.with_suffix(".csv"))
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        return

    N_SAMPLES = embedding.shape[0]

    def preprocess(embedding):
        l2 = normalize(embedding, norm='l2')
        pca = PCA(n_components=0.9).fit(l2)
        pca_l2 = normalize(pca.transform(l2), norm='l2')
        return pca_l2

    preprocessed_embedding = preprocess(embedding)

    params = make_umap_hdbscan_grid_params(
        n_neighbors_values=[
            0.05 * N_SAMPLES, 0.03 * N_SAMPLES, 0.07 * N_SAMPLES
        ],
        min_samples_values=[3, 5, 7]
    )

    runs = []

    for param in params:
        labels_for_consensus = []
        umap_embeddings_for_consensus = []

        for seed in range(0, 50):

            labels, umap_embeddings = fit_umap_hdbscan(
                preprocessed_embedding,
                n_components=10,
                n_neighbors=param["n_neighbors"],
                min_cluster_size=(0.03 * N_SAMPLES),
                min_samples=param["min_samples"],
                umap_metric="cosine",
                min_dist=0.02
            )
            labels_for_consensus.append(labels)
            umap_embeddings_for_consensus.append(umap_embeddings)

        consensus_labels = consensus_clustering(
            labels_for_consensus,
            n_clusters="median",
            max_noise_frequency=0.25,
            min_consensus_strength=0.60,
            min_final_cluster_size=(0.03 * N_SAMPLES),
        )

        dbcv = dbcv_cluster_consensus(
            consensus_labels,
            umap_embeddings_for_consensus
        )
        cosine_sim = mean_intracluster_cosine(
            preprocessed_embedding,
            consensus_labels
        )

        dbcv_values = np.array(list(dbcv.values()), dtype=float)

        mean_dbcv = np.nanmean(dbcv_values)
        min_dbcv = np.nanmin(dbcv_values)

        bad_cluster_penalty = max(0.0, -min_dbcv)

        score = (
                mean_dbcv
                - 0.50 * bad_cluster_penalty
                + 0.20 * cosine_sim
        )

        runs.append({
            "params": param,
            "consensus_labels": consensus_labels,
            "mean_dbcv": mean_dbcv,
            "min_dbcv": min_dbcv,
            "cosine_sim": cosine_sim,
            "score": score
        })

    best_run = max(
        runs,
        key=lambda r: r["score"] if np.isfinite(r["score"]) else -np.inf
    )









if __name__ == "__main__":
    main()
