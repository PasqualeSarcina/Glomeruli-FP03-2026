from pathlib import Path

import tensorflow as tf


class SegmentationDataset:
    """
    TF Dataset builder for SegNet training.

    Expects the directory layout produced by scripts/preprocess_data.py:
        root/
          img/   *.png   (400x400 RGB patches)
          mask/  *.png   (400x400 binary masks, 0=background 1=glomerulus)

    Image and mask must share the exact same filename across the two folders;
    pairing is established by `sorted()` on both sides.

    On-the-fly augmentation (rotations 90/270, vertical flip) is applied
    when `augment=True`. Apply only to the training split.
    """

    INPUT_SIZE = (384, 384)

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 4,
        shuffle: bool = True,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

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
            # Each image produces two samples: original + one randomly augmented.
            # A second shuffle breaks up the resulting adjacent pairs before batching.
            ds = ds.flat_map(self._expand_with_augment)
            if self.shuffle:
                ds = ds.shuffle(
                    buffer_size=500,
                    reshuffle_each_iteration=True,
                )

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

    def _expand_with_augment(self, image: tf.Tensor, mask: tf.Tensor) -> tf.data.Dataset:
        """Returns a 2-element dataset: the original pair followed by one augmented pair."""
        aug_image, aug_mask = self._augment_pair(image, mask)
        return tf.data.Dataset.from_tensors((image, mask)).concatenate(
            tf.data.Dataset.from_tensors((aug_image, aug_mask))
        )

    def _augment_pair(self, image: tf.Tensor, mask: tf.Tensor):
        # Same k for image and mask: 0 (identity), 1 (90 CCW), 3 (270 CCW).
        k_index = tf.random.uniform([], 0, 3, dtype=tf.int32)
        k = tf.gather(tf.constant([0, 1, 3], dtype=tf.int32), k_index)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)

        do_flip = tf.random.uniform([]) > 0.5
        image = tf.cond(
            do_flip,
            lambda: tf.image.flip_up_down(image),
            lambda: image,
        )
        mask = tf.cond(
            do_flip,
            lambda: tf.image.flip_up_down(mask),
            lambda: mask,
        )
        return image, mask

    def __len__(self) -> int:
        return len(self._image_paths)
