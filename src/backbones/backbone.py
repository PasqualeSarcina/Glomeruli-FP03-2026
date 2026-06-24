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
            mask
    ):
        return embedding

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