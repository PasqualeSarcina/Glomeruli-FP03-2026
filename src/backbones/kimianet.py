from pathlib import Path

import keras
from PIL import Image
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
import tensorflow as tf

from src.backbones.backbone import Backbone


class Kimianet(Backbone):
    def __init__(
            self,
            input_size= 224
    ):
        assert input_size > 32, "Input size should be greater than 32"
        backbone = tf.keras.applications.DenseNet121(
            include_top=False,
            input_shape=(input_size, input_size, 3),
            pooling=None
        )
        backbone.load_weights(Path(__file__).parent.parent.parent / "data" / "checkpoints" / "KimiaNetKerasWeights.h5")
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
