import keras
from PIL import Image
import numpy as np
from keras.src.applications.densenet import preprocess_input
import tensorflow as tf

from src.backbones.backbone import Backbone


class DenseNet169(Backbone):
    def __init__(
            self,
            input_size
    ):
        assert input_size > 32, "Input size should be greater than 32"
        backbone = keras.applications.DenseNet169(
            include_top=False,
            input_shape=(input_size, input_size, 3),
            pooling=None
        )
        backbone.trainable = False

        super().__init__(backbone, input_size, backbone.output_shape[-1])

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize(
            (self.input_size, self.input_size),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        image_array = preprocess_input(image_array, data_format=None)

        return image_array

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
