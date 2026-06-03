from typing import Literal

import PIL
import keras
import keras_hub
import numpy as np
from PIL import Image

DinoV2ModelName = Literal[
    "small",
    "base",
    "large",
    "giant"
]

class DinoV2:
    def __init__(
            self,
            model_name: DinoV2ModelName,
            input_size
    ):
        backbone = keras_hub.models.DINOV2Backbone.from_preset(
            "dinov2_" + model_name,
            image_shape=(input_size, input_size, 3),
        )
        backbone.trainable = False
        self.backbone = backbone

        assert input_size % self.backbone.patch_size == 0, "Input size must be a multiple of 14"
        self.input_size = (input_size, input_size)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(self.input_size)
        array = keras.preprocessing.image.img_to_array(image)
        expanded_array = np.expand_dims(array, axis=0)
        return expanded_array

    @staticmethod
    def _get_cls_token(embedding: np.ndarray) -> np.ndarray:
        cls_token = embedding[0, 0, :]
        return cls_token

    @staticmethod
    def _get_patch_embedding(embedding: np.ndarray) -> np.ndarray:
        patch_embedding = embedding[0, 1:, :]
        return patch_embedding

    def __call__(
            self,
            image,
            return_type: Literal["cls", "patch"]
    ) -> np.ndarray:
        processed_image = self._preprocess_image(image)
        features = self.backbone({"images": processed_image})
        features = keras.ops.convert_to_numpy(features)

        if return_type == "cls":
            cls_token = self._get_cls_token(features)
            return cls_token
        elif return_type == "patch":
            patch_embedding = self._get_patch_embedding(features)
            return patch_embedding
        else:
            raise ValueError("return_type must be either 'cls' or 'patch'")