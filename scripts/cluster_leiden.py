import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clustering.leiden.score import compute_leiden_clustering_score
from clustering.consensus import consensus_clustering
from clustering.leiden.fit import build_knn_graph, fit_umap_leiden
from clustering.leiden.params import make_umap_leiden_grid_params
from clustering.leiden.consensus_modularity import modularity_cluster_consensus
from clustering.cosine import mean_intracluster_cosine
from visualization.visualize_clustering import save_clustering_results

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

    evaluation_resolution = 1.0
    evaluation_graph = build_knn_graph(
        X=preprocessed_embedding,
        k=round(0.05 * N_SAMPLES),
        metric="cosine",
        weight_mode="gaussian",
    )

    params = make_umap_leiden_grid_params(
        n_neighbors_values=[
            round(0.03 * N_SAMPLES),
            round(0.05 * N_SAMPLES),
            round(0.075 * N_SAMPLES),
        ],
        k_neighbors_values=[
            round(0.03 * N_SAMPLES),
            round(0.05 * N_SAMPLES),
            round(0.075 * N_SAMPLES),
        ],
        resolution_values=[0.2, 0.3, 0.4, 0.5, 0.6]
    )

    runs = []

    for param in tqdm(params):
        labels_for_consensus = []
        #umap_embeddings_for_consensus = []

        for seed in range (0, 50):
            labels, umap_embeddings, _ = fit_umap_leiden(
                preprocessed_embedding,
                n_components=40,
                n_neighbors=param["n_neighbors"],
                umap_metric="cosine",
                k=param["k_neighbors"],
                resolution=param["resolution"],
                leiden_metric="euclidean",
                weight_mode="gaussian"
            )

            labels_for_consensus.append(labels)
            #umap_embeddings_for_consensus.append(umap_embeddings)

        consensus_labels = consensus_clustering(
            labels_for_consensus,
            n_clusters="median",
            max_noise_frequency=0.25,
            min_consensus_strength=0.70,
            min_final_cluster_size=int(round(0.03 * N_SAMPLES)),
            allow_final_noise=False
        )

        modularity = modularity_cluster_consensus(
            consensus_labels,
            evaluation_graph,
            resolution=evaluation_resolution,
        )
        cosine_sim = mean_intracluster_cosine(
            preprocessed_embedding,
            consensus_labels
        )

        score = compute_leiden_clustering_score(
            modularity=modularity,
            cosine_sim=cosine_sim,
            labels=consensus_labels,
            bad_cluster_weight=0.5,
            cosine_weight=0.2,
            noise_weight=0.15,
            max_noise_fraction=0.30,
            cluster_weight=0.05,
        )

        runs.append({
            "param": param,
            "consensus_labels": consensus_labels,
            "modularity": modularity,
            "cosine_sim": cosine_sim,
            "noise_fraction": float(np.mean(consensus_labels == -1)),
            "score": score
        })

    best_run = max(
        runs,
        key=lambda r: r["score"] if np.isfinite(r["score"]) else -np.inf
    )

    if not np.isfinite(best_run["score"]):
        raise RuntimeError(
            "No clustering configuration satisfies the coverage threshold."
        )

    print(best_run["param"])

    best_labels = np.asarray(best_run["consensus_labels"])
    cluster_rows = []

    for cluster in sorted(set(best_labels) - {-1}):
        cluster_mask = best_labels == cluster
        cluster_embedding = preprocessed_embedding[cluster_mask]
        cluster_labels = np.zeros(cluster_mask.sum(), dtype=int)

        cluster_rows.append({
            "cluster": cluster,
            "size": int(cluster_mask.sum()),
            "modularity": best_run["modularity"].get(cluster, np.nan),
            "cosine_similarity": mean_intracluster_cosine(
                cluster_embedding,
                cluster_labels,
                noise_label=None,
            ),
        })

    cluster_df = pd.DataFrame(
        cluster_rows,
        columns=["cluster", "size", "modularity", "cosine_similarity"],
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
