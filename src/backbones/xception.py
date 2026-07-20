import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

from backbones.backbone import Backbone


class Xception(Backbone):
    def __init__(self, input_size: int = 299):
        assert input_size >= 71, "Input size should be at least 71 for Xception"
        self.input_size = input_size
        self.backbone = keras.applications.Xception(
            include_top=False,
            input_shape=(self.input_size, self.input_size, 3),
            weights="imagenet",
            pooling=None,
            name="xception",
        )
        self.backbone.trainable = False
        self.hidden_dim = self.backbone.output_shape[-1]

        super().__init__(self.backbone, self.input_size, self.hidden_dim)

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
