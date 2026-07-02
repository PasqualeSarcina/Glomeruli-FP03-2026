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
