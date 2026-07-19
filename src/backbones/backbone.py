from abc import ABC, abstractmethod

import keras
import numpy as np
from PIL import Image
import tensorflow as tf


class Backbone(ABC):
    def __init__(
            self,
            backbone,
            input_size,
            hidden_dim
    ):
        self.backbone = backbone
        self.input_size = input_size
        self.hidden_dim = hidden_dim

    @abstractmethod
    def _preprocess_image(
            self,
            image: Image.Image,
    ):
        pass

    @staticmethod
    def _make_backbone_input(
            processed_image
    ):
        """
        Default for keras.applications:
            self.backbone(processed_image, training=False)
        """
        return processed_image

    def _forward(
            self,
            image: Image.Image
    ) -> np.ndarray:
        preprocessed_image = self._preprocess_image(image)
        input = self._make_backbone_input(preprocessed_image)
        output = self.backbone(input)

        return keras.ops.convert_to_numpy(output)

    def _postprocess(
            self,
            embedding,
            mask: Image.Image,
    ):
        """
        feature_map: tensore DenseNet con shape (1, H, W, C)
        mask_image: maschera PIL del glomerulo

        ritorna:
            embedding con shape (1, C)
        """

        feature_map = tf.convert_to_tensor(embedding, dtype=tf.float32)

        _, feature_h, feature_w, _ = feature_map.shape

        mask = mask.convert("L")

        mask = mask.resize(
            (feature_w, feature_h),
            Image.Resampling.BILINEAR,
        )

        weights = np.asarray(mask, dtype=np.float32)

        weights = weights / 255.0

        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        weights = tf.reshape(weights, shape=(1, feature_h, feature_w, 1))

        weighted_feature_map = feature_map * weights

        numerator = tf.reduce_sum(
            weighted_feature_map,
            axis=(1, 2),
        )

        denominator = tf.reduce_sum(
            weights,
            axis=(1, 2),
        )

        embedding = numerator / (denominator + 1e-8)

        return embedding.numpy().astype("float32")

    def __call__(
            self,
            image: Image.Image,
            mask: Image.Image | None = None,
    ) -> np.ndarray:
        preprocessed_image = self._preprocess_image(image)
        input_array = self._make_backbone_input(preprocessed_image)
        tensor = self.backbone(input_array)
        output = self._postprocess(tensor, mask)
        return output