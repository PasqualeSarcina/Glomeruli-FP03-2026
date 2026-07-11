import argparse
from pathlib import Path
import sys

from tqdm import tqdm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


from clustering.consensus import consensus_clustering
from clustering.cosine import mean_intracluster_cosine
from clustering.hdbscan.consensus_dbcv import dbcv_cluster_consensus
from clustering.hdbscan.fit import fit_umap_hdbscan
from clustering.hdbscan.params import make_umap_hdbscan_grid_params
from visualization.samples_plot import save_clustering_results
from clustering.hdbscan.score import compute_clustering_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
                        help="input name (without the extension) of the embedding + csv to process")
    return parser.parse_args()


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

    MIN_CLUSTER_SIZE = int(round(0.03 * N_SAMPLES))

    params = make_umap_hdbscan_grid_params(
        n_neighbors_values=[
            round(0.03 * N_SAMPLES),
            round(0.05 * N_SAMPLES),
            round(0.075 * N_SAMPLES),
        ],
        min_samples_values=[3, 5, 7, 10, 12]
    )

    runs = []

    for param in tqdm(params):
        labels_for_consensus = []
        umap_embeddings_for_consensus = []

        if param["min_samples"] >= MIN_CLUSTER_SIZE:
            continue

        for seed in range(0, 50):

            labels, umap_embeddings = fit_umap_hdbscan(
                preprocessed_embedding,
                n_components=10,
                n_neighbors=param["n_neighbors"],
                min_cluster_size=MIN_CLUSTER_SIZE,
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
            min_consensus_strength=0.70,
            min_final_cluster_size=MIN_CLUSTER_SIZE,
        )

        dbcv = dbcv_cluster_consensus(
            consensus_labels,
            umap_embeddings_for_consensus
        )
        cosine_sim = mean_intracluster_cosine(
            preprocessed_embedding,
            consensus_labels
        )

        score = compute_clustering_score(
            dbcv,
            cosine_sim,
            consensus_labels,
            bad_cluster_weight=0.40,
            cosine_weight=0.35,
            noise_weight=0.15
        )

        runs.append({
            "params": param,
            "consensus_labels": consensus_labels,
            "dbcv": dbcv,
            "cosine_sim": cosine_sim,
            "score": score
        })

    best_run = max(
        runs,
        key=lambda r: r["score"] if np.isfinite(r["score"]) else -np.inf
    )
    print(best_run["params"])

    best_labels = np.asarray(best_run["consensus_labels"])
    cluster_rows = []

    for cluster in sorted(set(best_labels) - {-1}):
        cluster_mask = best_labels == cluster
        cluster_embedding = preprocessed_embedding[cluster_mask]
        cluster_labels = np.zeros(cluster_mask.sum(), dtype=int)

        cluster_rows.append({
            "cluster": cluster,
            "size": int(cluster_mask.sum()),
            "dbcv": best_run["dbcv"].get(cluster, np.nan),
            "cosine_similarity": mean_intracluster_cosine(
                cluster_embedding,
                cluster_labels,
                noise_label=None,
            ),
        })

    cluster_df = pd.DataFrame(
        cluster_rows,
        columns=["cluster", "size", "dbcv", "cosine_similarity"],
    )

    if "image_path" not in csv.columns:
        raise ValueError("The input CSV must contain an 'image_path' column.")

    save_clustering_results(
        embeddings=preprocessed_embedding,
        labels=best_labels,
        image_paths=csv["image_path"].tolist(),
        cluster_df=cluster_df,
        x=10,
        base_dir=PROJECT_ROOT,
        image_size=2.0,
    )


if __name__ == "__main__":
    main()
