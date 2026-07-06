import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input


class MobileNet:
    def __init__(self, input_size: int = 224):
        assert input_size >= 32, "Input size should be at least 32"
        self.input_size = input_size
        self.backbone = keras.applications.MobileNet(
            include_top=False,
            input_shape=(self.input_size, self.input_size, 3),
            weights="imagenet",
            pooling=None,
            name="mobilenet",
        )
        self.backbone.trainable = False
        self.hidden_dim = self.backbone.output_shape[-1]

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize(
            (self.input_size, self.input_size),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        return image_array

    def _masked_global_average_pooling(
        self,
        feature_map,
        mask: Image.Image | None,
    ) -> np.ndarray:
        feature_map = tf.convert_to_tensor(feature_map, dtype=tf.float32)

        if mask is None:
            embedding = tf.reduce_mean(feature_map, axis=(1, 2))
            return embedding.numpy().astype("float32")

        _, feature_h, feature_w, _ = feature_map.shape

        mask = mask.convert("L")
        mask = mask.resize(
            (feature_w, feature_h),
            Image.Resampling.BILINEAR,
        )

        weights = np.asarray(mask, dtype=np.float32) / 255.0
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        weights = tf.reshape(weights, shape=(1, feature_h, feature_w, 1))

        numerator = tf.reduce_sum(feature_map * weights, axis=(1, 2))
        denominator = tf.reduce_sum(weights, axis=(1, 2))

        embedding = numerator / (denominator + 1e-8)

        return embedding.numpy().astype("float32")

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
    ) -> np.ndarray:
        x = self._preprocess_image(image)
        feature_map = self.backbone(x, training=False)
        embedding = self._masked_global_average_pooling(feature_map, mask)

        return embedding.squeeze(0).astype("float32")