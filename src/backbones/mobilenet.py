import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input

from src.backbones.backbone import Backbone


class MobileNet(Backbone):
    def __init__(self, input_size: int = 224):
        assert input_size >= 32, "Input size should be at least 32"
        backbone = keras.applications.MobileNet(
            include_top=False,
            input_shape=(input_size, input_size, 3),
            weights="imagenet",
            pooling=None,
            name="mobilenet",
        )
        backbone.trainable = False
        hidden_dim = backbone.output_shape[-1]

        super().__init__(backbone, input_size, hidden_dim)

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
