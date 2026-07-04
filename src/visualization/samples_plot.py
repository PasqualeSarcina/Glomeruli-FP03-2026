from __future__ import annotations

from datetime import datetime
from html import escape
import os
from pathlib import Path
from typing import Sequence

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors


def show_clustering_samples(
    labels: Sequence[int],
    probabilities: Sequence[float] | np.ndarray,
    image_paths: Sequence[str | Path],
    x: int,
    base_dir: str | Path | None,
    image_size: float,
) -> tuple[Figure, np.ndarray]:
    """
    Show the top x images per cluster ordered by descending probability.

    The function also creates:
    - clustering results/<timestamp>/samples_per_cluster.png
    - clustering results/<timestamp>/index.html
    - clustering results/<timestamp>/manifest.csv
    - clustering results/<timestamp>/cluster_<label>.html for every cluster
    - clustering results/<timestamp>/noise.html when HDBSCAN noise is present.
    """

    labels_array = np.asarray(labels)
    image_paths = list(image_paths)

    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array.")

    if len(image_paths) != labels_array.shape[0]:
        raise ValueError("image_paths and labels must have the same length.")

    if x < 1:
        raise ValueError("x must be at least 1.")

    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    assigned_scores = _assigned_cluster_probabilities(
        probabilities,
        labels_array,
    )

    cluster_labels = _cluster_labels(labels_array, include_noise=False)

    if not cluster_labels:
        raise ValueError("No clusters to display.")

    results_dir = _create_results_dir()

    figure, axes = plt.subplots(
        len(cluster_labels),
        x,
        figsize=(image_size * x, image_size * len(cluster_labels)),
        squeeze=False,
    )

    for row, cluster_label in enumerate(cluster_labels):

        cluster_indices = np.flatnonzero(labels_array == cluster_label)
        selected_indices = _sort_indices_by_score(
            indices=cluster_indices,
            assigned_scores=assigned_scores,
        )[:x]

        for column in range(x):

            axis = axes[row, column]
            axis.axis("off")

            if column == 0:
                axis.set_ylabel(
                    _format_cluster_label(cluster_label),
                    rotation=0,
                    labelpad=45,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

            if column >= len(selected_indices):
                continue

            sample_index = int(selected_indices[column])
            image_path = _resolve_path(
                image_paths[sample_index],
                base_dir,
            )

            _show_image(axis, image_path)

            axis.set_title(
                _sample_title(
                    sample_index=sample_index,
                    assigned_scores=assigned_scores,
                ),
                fontsize=9,
            )

    figure.suptitle("Samples per cluster")
    figure.tight_layout()

    figure.savefig(
        results_dir / "samples_per_cluster.png",
        dpi=300,
        bbox_inches="tight",
    )
    _write_html_report(
        labels=labels_array,
        assigned_scores=assigned_scores,
        image_paths=image_paths,
        base_dir=base_dir,
        output_dir=results_dir,
    )

    plt.show()

    return figure, axes


def _create_results_dir() -> Path:
    results_root = _repository_root() / "clustering results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_root / timestamp

    suffix = 2
    while output_dir.exists():
        output_dir = results_root / f"{timestamp}_{suffix}"
        suffix += 1

    output_dir.mkdir(parents=True, exist_ok=False)

    return output_dir


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_html_report(
    labels: np.ndarray,
    assigned_scores: np.ndarray,
    image_paths: Sequence[str | Path],
    base_dir: str | Path | None,
    output_dir: Path,
) -> None:
    manifest = _build_manifest(
        labels=labels,
        assigned_scores=assigned_scores,
        image_paths=image_paths,
        base_dir=base_dir,
        output_dir=output_dir,
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    summary_rows = []

    for cluster_label in _cluster_labels(labels, include_noise=True):
        cluster_manifest = manifest.loc[
            manifest["cluster"] == cluster_label
        ].sort_values(
            by="probability",
            ascending=False,
            na_position="last",
        )

        cluster_html = _cluster_html_filename(cluster_label)

        _write_cluster_html(
            output_path=output_dir / cluster_html,
            cluster_label=cluster_label,
            rows=cluster_manifest,
        )

        summary_rows.append({
            "cluster": int(cluster_label),
            "label": _format_cluster_label(cluster_label),
            "size": int(len(cluster_manifest)),
            "mean_probability": float(cluster_manifest["probability"].mean()),
            "html": cluster_html,
        })

    _write_index_html(
        output_path=output_dir / "index.html",
        summary_rows=summary_rows,
    )


def _build_manifest(
    labels: np.ndarray,
    assigned_scores: np.ndarray,
    image_paths: Sequence[str | Path],
    base_dir: str | Path | None,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    for cluster_label in _cluster_labels(labels, include_noise=True):
        cluster_indices = np.flatnonzero(labels == cluster_label)
        sorted_indices = _sort_indices_by_score(
            indices=cluster_indices,
            assigned_scores=assigned_scores,
        )

        for sample_index in sorted_indices:
            sample_index = int(sample_index)
            resolved_path = _resolve_path(image_paths[sample_index], base_dir)
            probability = float(assigned_scores[sample_index])

            rows.append({
                "id": sample_index,
                "cluster": int(cluster_label),
                "probability": probability if np.isfinite(probability) else np.nan,
                "image_path": str(resolved_path),
                "image_href": _relative_href(resolved_path, output_dir),
            })

    return pd.DataFrame(rows)


def _write_index_html(output_path: Path, summary_rows: list[dict]) -> None:
    rows_html = "\n".join(
        (
            "<tr>"
            f"<td><a href=\"{escape(row['html'])}\">"
            f"{escape(row['label'])}</a></td>"
            f"<td>{row['size']}</td>"
            f"<td>{_format_probability(row['mean_probability'])}</td>"
            "</tr>"
        )
        for row in summary_rows
    )

    output_path.write_text(
        _html_document(
            title="Clustering results",
            body=f"""
<h1>Clustering results</h1>
<p><a href="manifest.csv">manifest.csv</a></p>
<figure>
  <a href="samples_per_cluster.png">
    <img class="preview" src="samples_per_cluster.png" alt="Samples per cluster">
  </a>
  <figcaption>Top samples per cluster</figcaption>
</figure>
<table>
  <thead>
    <tr>
      <th>Cluster</th>
      <th>Size</th>
      <th>Mean probability</th>
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
""",
        ),
        encoding="utf-8",
    )


def _write_cluster_html(
    output_path: Path,
    cluster_label: int,
    rows: pd.DataFrame,
) -> None:
    items_html = "\n".join(
        _sample_card_html(row)
        for _, row in rows.iterrows()
    )

    output_path.write_text(
        _html_document(
            title=_format_cluster_label(cluster_label),
            body=f"""
<nav><a href="index.html">Index</a> | <a href="manifest.csv">manifest.csv</a></nav>
<h1>{escape(_format_cluster_label(cluster_label))}</h1>
<p>{len(rows)} samples</p>
<section class="grid">
  {items_html}
</section>
""",
        ),
        encoding="utf-8",
    )


def _sample_card_html(row: pd.Series) -> str:
    image_href = escape(str(row["image_href"]))
    image_path = escape(str(row["image_path"]))
    probability = _format_probability(row["probability"])
    sample_id = int(row["id"])

    return f"""
<article class="sample">
  <a href="{image_href}">
    <img src="{image_href}" loading="lazy" alt="id {sample_id}">
  </a>
  <div class="meta">
    <div><strong>id</strong>: {sample_id}</div>
    <div><strong>p</strong>: {probability}</div>
    <div class="path">{image_path}</div>
  </div>
</article>
"""


def _html_document(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      color: #1f2933;
      font-family: Arial, sans-serif;
      line-height: 1.4;
      margin: 24px;
    }}
    table {{
      border-collapse: collapse;
      margin-top: 18px;
      min-width: 420px;
    }}
    th, td {{
      border-bottom: 1px solid #d9e2ec;
      padding: 8px 12px;
      text-align: left;
    }}
    th {{
      background: #f0f4f8;
    }}
    .preview {{
      border: 1px solid #d9e2ec;
      max-width: min(100%, 1200px);
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    }}
    .sample {{
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      padding: 8px;
    }}
    .sample img {{
      display: block;
      height: 140px;
      object-fit: contain;
      width: 100%;
    }}
    .meta {{
      font-size: 12px;
      margin-top: 8px;
    }}
    .path {{
      color: #52606d;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _relative_href(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, start=output_dir)).as_posix()


def _format_probability(value: float) -> str:
    if pd.isna(value):
        return "NA"

    return f"{float(value):.3f}"


def _sort_indices_by_score(
    indices: np.ndarray,
    assigned_scores: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(assigned_scores[indices], dtype=float)
    finite_mask = np.isfinite(scores)
    finite_indices = indices[finite_mask]
    finite_scores = scores[finite_mask]
    non_finite_indices = indices[~finite_mask]

    sorted_finite_indices = finite_indices[np.argsort(-finite_scores, kind="stable")]

    return np.concatenate([sorted_finite_indices, non_finite_indices])


def _sample_title(
    sample_index: int,
    assigned_scores: np.ndarray,
) -> str:
    score = float(assigned_scores[sample_index])

    if np.isfinite(score):
        return f"id={sample_index}\np={score:.3f}"

    return f"id={sample_index}\np=NA"


def _assigned_cluster_probabilities(
    probabilities: Sequence[float] | np.ndarray | None,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Restituisce uno score per ogni sample.

    Se probabilities è None, restituisce un array di 1.
    Questo permette di usare la funzione anche con Leiden.
    """

    if probabilities is None:
        return np.ones(labels.shape[0], dtype=float)

    probabilities_array = np.asarray(probabilities, dtype=float)

    if probabilities_array.ndim == 1:

        if probabilities_array.shape[0] != labels.shape[0]:
            raise ValueError(
                "probabilities and labels must have the same length."
            )

        return probabilities_array

    if probabilities_array.ndim == 2:

        if probabilities_array.shape[0] != labels.shape[0]:
            raise ValueError(
                "probabilities and labels must have the same length."
            )

        if np.any((labels >= probabilities_array.shape[1]) & (labels != -1)):
            raise ValueError(
                "2D probabilities require non-noise labels to be valid "
                "column indices."
            )

        assigned = np.zeros(labels.shape[0], dtype=float)

        non_noise_mask = labels != -1

        row_indices = np.arange(labels.shape[0])[non_noise_mask]
        column_indices = labels[non_noise_mask].astype(int)

        assigned[non_noise_mask] = probabilities_array[
            row_indices,
            column_indices,
        ]

        return assigned

    raise ValueError("probabilities must be None, a 1D array or a 2D array.")


def _cluster_labels(labels: np.ndarray, include_noise: bool) -> list[int]:
    unique_labels = [int(label) for label in np.unique(labels)]
    if not include_noise:
        unique_labels = [label for label in unique_labels if label != -1]
    return sorted(unique_labels, key=lambda label: (label == -1, label))


def _resolve_path(path: str | Path, base_dir: str | Path | None) -> Path:
    image_path = Path(path)
    if base_dir is not None and not image_path.is_absolute():
        image_path = Path(base_dir) / image_path
    return image_path


def _show_image(axis: Axes, image_path: Path) -> None:
    if not image_path.exists():
        axis.text(0.5, 0.5, "missing", ha="center", va="center")
        return

    try:
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
    except (OSError, UnidentifiedImageError):
        axis.text(0.5, 0.5, "unreadable", ha="center", va="center")


def _format_cluster_label(label: int) -> str:
    return "Noise" if label == -1 else f"Cluster {label}"


def _cluster_html_filename(label: int) -> str:
    return "noise.html" if int(label) == -1 else f"cluster_{int(label)}.html"


__all__ = ["show_clustering_samples"]




def plot_nearest_neighbors_with_distances(
    X: np.ndarray,
    csv_path: str | Path,
    query_indices: Sequence[int],
    k: int = 5,
    image_column: str = "image_path",
    base_dir: str | Path | None = None,
    metric: str = "cosine",
    normalize_l2: bool = True,
    figsize_per_image: float = 2.8,
) -> None:
    """
    Plotta, per ogni indice query, il glomerulo selezionato e i suoi k nearest neighbors.

    Sopra ogni immagine mostra:
        - idx
        - filename estratto da image_path

    Sotto ogni immagine mostra:
        - distanza dal query sample

    Cornice:
        - verde per la query
        - grigia per i vicini
    """

    X = np.asarray(X)
    metadata = pd.read_csv(csv_path)

    if X.ndim != 2:
        raise ValueError(
            f"X deve essere una matrice 2D, ma ha shape {X.shape}."
        )

    if len(metadata) != X.shape[0]:
        raise ValueError(
            f"CSV ed embedding non hanno la stessa lunghezza: "
            f"CSV={len(metadata)}, X={X.shape[0]}."
        )

    if image_column not in metadata.columns:
        raise ValueError(
            f"La colonna '{image_column}' non esiste nel CSV. "
            f"Colonne disponibili: {list(metadata.columns)}"
        )

    n_samples = X.shape[0]

    if k >= n_samples:
        raise ValueError(
            f"k={k} è troppo grande. Deve essere minore di n_samples={n_samples}."
        )

    query_indices = list(query_indices)

    for idx in query_indices:
        if idx < 0 or idx >= n_samples:
            raise IndexError(
                f"Indice {idx} non valido. Deve essere tra 0 e {n_samples - 1}."
            )

    base_dir = Path(base_dir) if base_dir is not None else None

    if metric == "cosine" and normalize_l2:
        X_nn = normalize(X, norm="l2")
    else:
        X_nn = X

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric
    )

    nn.fit(X_nn)

    for query_idx in query_indices:
        distances, indices = nn.kneighbors(
            X_nn[query_idx].reshape(1, -1)
        )

        distances = distances[0]
        indices = indices[0]

        # Rimuove il punto stesso
        mask = indices != query_idx
        neighbor_indices = indices[mask][:k]
        neighbor_distances = distances[mask][:k]

        indices_to_plot = [query_idx] + list(neighbor_indices)
        distances_to_plot = [0.0] + list(neighbor_distances)

        n_images = len(indices_to_plot)

        fig, axes = plt.subplots(
            1,
            n_images,
            figsize=(figsize_per_image * n_images, figsize_per_image + 1.8)
        )

        if n_images == 1:
            axes = [axes]

        for rank, (axis, sample_idx, distance) in enumerate(
            zip(axes, indices_to_plot, distances_to_plot)
        ):
            raw_image_path = Path(str(metadata.iloc[sample_idx][image_column]))

            if base_dir is not None and not raw_image_path.is_absolute():
                image_path = base_dir / raw_image_path
            else:
                image_path = raw_image_path

            filename = raw_image_path.name

            image = Image.open(image_path).convert("RGB")

            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])

            # Cornice verde per la query
            if rank == 0:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("green")
                    spine.set_linewidth(4)
            else:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("lightgray")
                    spine.set_linewidth(1)

            # Sopra: indice e filename
            if rank == 0:
                axis.set_title(
                    f"Query\nidx={sample_idx}\n{filename}",
                    fontsize=8
                )
            else:
                axis.set_title(
                    f"NN {rank}\nidx={sample_idx}\n{filename}",
                    fontsize=8
                )

            # Sotto: distanza
            axis.set_xlabel(
                f"d={distance:.4f}",
                fontsize=9
            )

        fig.suptitle(
            f"Nearest neighbors | query index = {query_idx} | metric = {metric}",
            fontsize=12
        )

        plt.tight_layout()
        plt.show()
