from pathlib import Path

import tensorflow as tf

# Ruifrok-Johnston stain vectors (skimage HED convention). Used for in-graph
# HED stain augmentation. _HED_FROM_RGB is the colour-deconvolution matrix;
# with alpha=1 the separate/combine round-trip is the identity (up to clipping),
# so stain_jitter=0.0 reproduces the un-augmented image exactly.
_RGB_FROM_HED = tf.constant(
    [[0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78]],
    dtype=tf.float32,
)
_HED_FROM_RGB = tf.linalg.inv(_RGB_FROM_HED)


class SegmentationDataset:
    """
    TF Dataset builder for SegNet training.

    Expects the directory layout produced by scripts/preprocessing/preprocess_data.py:
        root/
          img/   *.png   (400x400 RGB patches)
          mask/  *.png   (400x400 binary masks, 0=background 1=glomerulus)

    Image and mask must share the exact same filename across the two folders;
    pairing is established by `sorted()` on both sides.

    On-the-fly augmentation (full D4 group: rotations 0/90/180/270 + flips)
    is applied when `augment=True`. Apply only to the training split.
    `stain_jitter` adds HED-space stain perturbation, the domain-appropriate
    colour augmentation for histology (see `_stain_jitter`).
    """

    INPUT_SIZE = (384, 384)

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 4,
        shuffle: bool = True,
        augment: bool = False,
        flip_horizontal: bool = False,
        brightness_delta: float = 0.0,
        stain_jitter: float = 0.0,
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.flip_horizontal = flip_horizontal
        self.brightness_delta = brightness_delta
        self.stain_jitter = stain_jitter

        image_paths = sorted((self.root / "img").glob("*.png"))
        mask_paths = sorted((self.root / "mask").glob("*.png"))

        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Image/mask count mismatch in {self.root}: "
                f"{len(image_paths)} images vs {len(mask_paths)} masks."
            )

        self._image_paths = [str(p) for p in image_paths]
        self._mask_paths = [str(p) for p in mask_paths]

    def build(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(
            (self._image_paths, self._mask_paths)
        )

        if self.shuffle:
            ds = ds.shuffle(
                buffer_size=len(self._image_paths),
                reshuffle_each_iteration=True,
            )

        ds = ds.map(self._load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            # Concatenate original dataset with an augmented copy.
            # concatenate() produces a proper finite dataset that Keras can
            # iterate epoch-by-epoch without flat_map epoch-boundary issues.
            aug_ds = ds.map(self._augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.concatenate(aug_ds)
            if self.shuffle:
                ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _load_pair(self, image_path: tf.Tensor, mask_path: tf.Tensor):
        return self._load_image(image_path), self._load_mask(mask_path)

    def _load_image(self, path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        image = tf.image.decode_png(raw, channels=3)
        image = tf.image.resize(image, self.INPUT_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def _load_mask(self, path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        mask = tf.image.decode_png(raw, channels=1)
        mask = tf.image.resize(mask, self.INPUT_SIZE, method='nearest')
        mask = tf.cast(mask, tf.int32)
        return mask

    def _augment_pair(self, image: tf.Tensor, mask: tf.Tensor):
        # Full 90-degree rotation group: 0, 90, 180, 270. Histology has no
        # canonical orientation, so all four are label-preserving (the earlier
        # {0,90,270} set threw away 180 for no reason).
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)

        do_flip_v = tf.random.uniform([]) > 0.5
        image = tf.cond(do_flip_v, lambda: tf.image.flip_up_down(image), lambda: image)
        mask  = tf.cond(do_flip_v, lambda: tf.image.flip_up_down(mask),  lambda: mask)

        if self.flip_horizontal:
            do_flip_h = tf.random.uniform([]) > 0.5
            image = tf.cond(do_flip_h, lambda: tf.image.flip_left_right(image), lambda: image)
            mask  = tf.cond(do_flip_h, lambda: tf.image.flip_left_right(mask),  lambda: mask)

        if self.stain_jitter > 0.0:
            image = self._stain_jitter(image)

        if self.brightness_delta > 0.0:
            image = tf.image.random_brightness(image, max_delta=self.brightness_delta)
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image, mask

    def _stain_jitter(self, image: tf.Tensor) -> tf.Tensor:
        """
        Multiplicative HED-space stain augmentation (Tellez et al. 2019 style).

        Deconvolves the RGB patch into H/E/DAB stain concentrations, scales each
        channel by a random factor in [1-s, 1+s], then recombines. This mimics
        the stain-intensity variation between slides/scanners — the main source
        of domain shift when training on only a handful of WSIs.

        Multiplicative only (no additive term): white background has zero stain
        concentration, so it maps to itself and is left untouched.
        """
        s = self.stain_jitter
        log_adjust = tf.math.log(1e-6)
        od = tf.math.log(tf.maximum(image, 1e-6)) / log_adjust
        stains = tf.tensordot(od, _HED_FROM_RGB, axes=1)
        alpha = tf.random.uniform([3], 1.0 - s, 1.0 + s)
        stains = stains * alpha
        log_rgb = tf.tensordot(stains, _RGB_FROM_HED, axes=1) * log_adjust
        return tf.clip_by_value(tf.exp(log_rgb), 0.0, 1.0)

    def __len__(self) -> int:
        return len(self._image_paths)


if __name__ == "__main__":
    # Sanity check for the HED stain math (run on any box with TF).
    ds = SegmentationDataset.__new__(SegmentationDataset)
    img = tf.random.uniform([8, 8, 3])

    # alpha~1 => separate/combine round-trip is the identity (up to clip).
    ds.stain_jitter = 1e-9
    out0 = ds._stain_jitter(img)
    assert tf.reduce_max(tf.abs(out0 - img)) < 1e-4, "sigma~0 must be identity"

    ds.stain_jitter = 0.05
    out = ds._stain_jitter(img)
    assert out.shape == img.shape
    assert tf.reduce_min(out) >= 0.0 and tf.reduce_max(out) <= 1.0
    # White background stays white under multiplicative jitter.
    white = tf.ones([4, 4, 3])
    assert tf.reduce_max(tf.abs(ds._stain_jitter(white) - white)) < 1e-4
    print("dataset.py stain-jitter self-check passed.")
