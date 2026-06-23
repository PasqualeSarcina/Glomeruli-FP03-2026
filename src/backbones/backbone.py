from abc import ABC, abstractmethod

import keras
import numpy as np
from PIL import Image


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

    def _preprocess_image(
            self,
            image: Image.Image,
            mask: Image.Image | None = None,
            mean: np.ndarray | None = None,
            std: np.ndarray | None = None
    ) -> np.ndarray:
        image = image.convert("RGB").resize(self.input_size)
        if mask is not None:
            black_background = Image.new("RGB", self.input_size, (0, 0, 0))
            image = Image.composite(image, black_background, mask)

        array = keras.preprocessing.image.img_to_array(image).astype(np.float32)

        if mean is not None and std is not None:
            # Scale to  [0, 1]
            array = array / 255.0

            # ImageNet norm
            array = (array - mean) / std

        expanded_array = np.expand_dims(array, axis=0)
        return expanded_array

    @abstractmethod
    def forward(
            self,
            image: Image.Image,
            mask: Image.Image | None = None,
    ):
        pass