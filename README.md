# Glomeruli detection — FP03

Machine Learning in Applications, Politecnico di Torino.

End-to-end pipeline that goes from raw kidney-biopsy Whole Slide Images to the
unsupervised analysis of glomeruli, in the context of diabetic kidney disease.

The dataset provides the *position* of each glomerulus but no label describing
its type or its degree of alteration, so the characterisation stage is posed as
an unsupervised problem. The pipeline addresses three questions:

- **RQ1** — can glomeruli be detected and segmented using only positional annotations?
- **RQ2** — do the resulting embeddings form discrete, separable classes?
- **RQ3** — is that structure aligned with the level of necrotization?

**Main finding.** Glomerular necrotization behaves as a *continuum*, not as a set
of discrete classes. No backbone induces a genuine density-based cluster
structure; the glomeruli lie along a continuous, partly stain-confounded gradient
of morphological severity. Any downstream grading is best read as a coarse
discretisation of that axis rather than as a set of natural categories.

Full method and discussion: see the project report.

---

## Repository layout

```
src/
  data/            WSI preprocessing: tissue mask, Reinhard normalisation,
                   patch extraction, glomerulus crops and masks from the XML
  segmentation/    SegNet encoder-decoder (VGG19 or ResNet50) and its tf.data
                   pipeline, with the on-the-fly augmentation
  backbones/       frozen feature extractors (MobileNet, DenseNet, Xception,
                   NASNet, DINOv2/v3, KimiaNet) behind a common interface
  clustering/      UMAP + HDBSCAN and UMAP + Leiden, with 50-seed consensus
  metrics/         deterministic backbone-evaluation metrics
  visualization/   UMAP plots and per-cluster HTML reports

scripts/           command-line entry points (see Pipeline below)
experiments/       SLURM job scripts for the HPC runs; legacy/ archives the
                   historical run configs for provenance only
notebooks/         backbone and clustering evaluation, with saved outputs
results/           committed results (see Results below)
data/              inputs and intermediates — not versioned
models/            training checkpoints and run logs — not versioned
```

`src/` is a plain package tree, so no installation step is required: each script
prepends the directory it needs (the project root, or `src/`) to `sys.path`
before importing.

---

## Setup

Python 3.12. OpenSlide needs its native library, which `openslide-bin` provides
on Windows and macOS; on Linux install the system package instead
(`apt install libopenslide0`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

Not versioned. Place the slides and their annotations anywhere, then let the
scripts write their outputs under `data/`:

```
<slides>/                  RECHERCHE-***.svs  +  RECHERCHE-***.xml
data/
  tissue_masks/            entropy-based tissue masks, one .npy per slide
  dataset/{train,validation,test}/{img,mask}/     patches for segmentation
  glomeruli/
    crops/                 one PNG per annotated glomerulus (677 total)
    masks/                 the matching binary mask
    embeddings/            <backbone>_embeddings.npy + .csv with the crop order
```

The dataset used in the report is 9 annotated slides and 677 glomeruli.

---

## Pipeline

### 1. Preprocessing — slides to patches

Extracts the tissue mask, estimates the Reinhard statistics, and cuts the
sliding-window patches with their rasterised glomerulus masks. The split is made
at slide level, so all the patches of a slide stay in the same subset.

```bash
python scripts/preprocess_data.py <slides> --split-percentages 0.7 0.15
```

Writes `data/tissue_masks/` and `data/dataset/{train,validation,test}/`.

### 2. Detection and segmentation

Detection and segmentation are carried out by the same fully convolutional
encoder-decoder: the network predicts a per-pixel glomerulus map, and the
connected components of that map localise the individual glomeruli.

`src/segmentation/` holds the architecture and the tf.data input pipeline.
Training is two-phase: a frozen-encoder warm-up, then a full fine-tune.

```bash
python scripts/train_segmentation.py data/dataset \
    --phase-1-epochs 10 --phase-1-lr 0.01 \
    --phase-2-epochs 20 --phase-2-lr 0.001 \
    --flip-horizontal --stain-jitter 0.05 \
    --output-dir models/run
```

Writes `best_model.keras` (selected on validation mean IoU), the CSV training log
and the history into `--output-dir`. Evaluation is inference-only and reports the
per-class IoU over a split from a single accumulated confusion matrix; `--tta`
averages the 8 D4 orientations.

```bash
python scripts/evaluate_segmentation.py models/run/best_model.keras data/dataset/test --tta
```

**Leave-one-slide-out cross-validation.** With nine annotated slides a single
fixed split is not a reliable estimator — and the one-slide validation set
repeatedly ranked models the opposite way from the test set — so the reported
segmentation results use LOSO: each slide is held out in turn, the model is
trained on the other eight for a fixed budget, and the final model is evaluated
on the held-out slide. There is deliberately no validation set and no early
stopping, since selecting an epoch would reintroduce the very problem LOSO
exists to avoid.

```bash
python scripts/loso_cv.py data/dataset --stain-jitter 0.05 --tta \
    --output-dir models/loso
```

Writes a per-slide breakdown and the mean ± std to `loso_results.json`.

The encoder is VGG19 with ImageNet weights by default. `--encoder resnet50`
instead uses a ResNet50 initialised from pathology-pretrained weights, which must
first be converted from PyTorch offline (this needs torch and network access, so
run it on a login node, not a GPU node):

```bash
bash scripts/setup_and_convert.sh barlowtwins   # or swav / mocov2 / imagenet
python scripts/loso_cv.py data/dataset --encoder resnet50 \
    --encoder-weights models/encoders/barlowtwins_resnet50.weights.h5
```

The SLURM jobs used on the cluster are in `experiments/`.

### 3. Glomerulus crops — for the unsupervised stage

Independent of step 1: crops each annotated glomerulus at full resolution from
the bounding box of its polygon, together with a binary mask rasterised from the
same polygon.

```bash
python scripts/extract_glomeruli_from_annotations.py <slides> --images-size 1024
```

Writes `data/glomeruli/crops/` and `data/glomeruli/masks/`.

### 4. Embeddings

Each crop is encoded by a frozen, pretrained backbone. Convolutional feature maps
are aggregated by a masked global average pooling, so the embedding describes the
glomerulus and not the surrounding tissue; for the vision transformers the class
token and a mask-weighted average of the patch tokens are used.

```bash
python scripts/extract_glomeruli_embeds.py mobilenet data/glomeruli
python scripts/extract_glomeruli_embeds.py dinov2 data/glomeruli --backbone-size base --mode cls
```

Subcommands: `mobilenet`, `densenet169`, `densenet201`, `xception`, `nasnet`,
`kimianet`, `dinov2`, `dinov3`. Writes a `.npy` and a `.csv` (the crop order,
needed to align labels back to images) into `data/glomeruli/embeddings/`.

### 5. Clustering

Both algorithms share the same preprocessing — L2, then PCA keeping 90% of the
variance, then L2 again — followed by UMAP. Each hyperparameter configuration is
run 50 times and the runs are combined into a single stable partition by
consensus clustering.

```bash
python scripts/cluster_hdbscan.py data/glomeruli/embeddings/mobilenet_crops_embeddings
python scripts/cluster_leiden.py  data/glomeruli/embeddings/mobilenet_crops_embeddings
```

The argument is the path **without extension**; the `.npy` and the `.csv` are
read from it. Each run creates `results/clustering/<timestamp>_<method>/` with
the labels (`manifest.csv`), the per-cluster metrics, the UMAP plot and a
browsable HTML report (open `index.html`).

### 6. Evaluation

Two notebooks, both committed with their outputs:

- `notebooks/backbone_evaluation.ipynb` — compares the eleven embeddings under a
  deterministic, seed-independent protocol and produces `results/backbone_evaluation/`.
- `notebooks/clustering_evaluation.ipynb` — reads the saved clustering runs and
  adds what the pipeline does not compute: the slide confound, the alignment with
  the severity proxy, and the HDBSCAN-versus-Leiden comparison. Produces
  `results/clustering_evaluation/`.

Two further notebooks are kept as working material rather than as results:
`evaluate_embeddings.ipynb` (the embedding sanity checks, later folded into the
backbone evaluation) and `colab_training.ipynb` (the Colab version of the
segmentation training, used before the runs moved to the cluster). Both contain
absolute paths from the machine they were run on.

The clustering notebook auto-detects the most recent run of each method under
`results/clustering/`, so run step 5 first.

Three further scripts produce material for the report rather than for the
pipeline: `make_preprocessing_figure.py` (slide to tissue mask to patches),
`make_qualitative_figure.py` (patch, ground truth and prediction side by side,
sampled across the IoU distribution rather than cherry-picked) and
`check_rectangular_masks.py` (counts the annotations drawn as bounding boxes
instead of outlines, via the mask fill ratio).

---

## Results

### Segmentation

Checkpoints and run logs are not versioned, so this stage has no committed
artifacts under `results/` — rerunning `scripts/loso_cv.py` regenerates
`loso_results.json` with the per-slide breakdown.

The reference recipe is SegNet-VGG19 with an ImageNet encoder, two-phase
training and on-the-fly D4 + horizontal flip + HED stain jitter augmentation,
which reaches a glomerulus IoU of 0.702 ± 0.094 over the nine leave-one-slide-out
folds. The spread across folds is as informative as the mean: it is what the
single fixed split hides. The pathology-pretrained ResNet50 encoders were run as
a comparison on the same protocol. Full per-run discussion is in the report.

### Backbone selection

Eleven embeddings compared over nine preprocessing configurations
(`results/backbone_evaluation/SUMMARY.csv` and the `*_meanOf9.csv` rankings).

**MobileNet** is retained: first on morphological neighbourhood consistency
(0.135) and effectively tied for the smoothness of the severity gradient
(Moran's I 0.399 against 0.404 for DenseNet-169), the most stable across
preprocessing choices, and the most representative of the cross-backbone
consensus (0.59). DenseNet-169 is the closest competitor and leads on the global
distance-morphology correlation (0.303).

DBCV is **negative for every backbone and every configuration** and the
silhouette never exceeds ≈0.30: the embedding spaces do not contain
well-separated clusters. The highest silhouette (KimiaNet-1000) is an artefact of
its very low effective dimensionality — a participation ratio of 9.7 against 58
for MobileNet.

### Clustering

Committed runs: `results/clustering/20260723_150036_hdbscan` and
`20260723_152312_leiden`, both on the MobileNet embeddings.

| | clusters | noise |
|---|---|---|
| HDBSCAN (baseline) | 3 (244 / 339 / 59) | 35 (5.2%) |
| Leiden (extension) | 7 | none |

The two partitions agree only moderately (ARI 0.57, NMI 0.74): Leiden is a
refinement of HDBSCAN. Where it splits an HDBSCAN cluster, the subgroups differ
significantly along the severity proxy (Mann-Whitney p = 6.4e-14 and 1.1e-23),
and the 34 glomeruli of Leiden cluster 1 — median severity +1.20, among the
highest — are all labelled as noise by HDBSCAN. The graph-based extension
therefore also recovers a coherent, matrix-rich group that the baseline discards.

### Caveats

- The severity proxy is derived from handcrafted descriptors (colour
  deconvolution and GLCM texture inside the mask), **not** from expert grading.
  It is a reproducible reference, not a clinical ground truth.
- The proxy is partly sensitive to staining intensity, so the severity ordering
  is a coarse trend rather than a clean monotone axis.
- The group formed by HDBSCAN cluster 2 and, in the finer partition, Leiden
  clusters 3 and 4 draws up to 52% of its members from a single slide
  (`RECHERCHE-016`) and is best read as partly a batch effect.
- Nine annotated slides is the dominant limitation. A single fixed split is not a
  reliable estimator at this size, which is why the segmentation results are
  reported under leave-one-slide-out cross-validation.

---

## Authors

Federico Mameli (s348829), Zein Alabedin Ismail (s349315),
Alexandra Elena Holota (s338437), Pasquale Sarcina (s347503).
