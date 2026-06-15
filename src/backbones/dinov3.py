from typing import Literal

import keras
import keras_hub
import numpy as np
from PIL import Image

DinoV3ModelName = Literal[
    "small",
    "small_plus",
    "base",
    "large",
    "giant",
    "huge_plus",
    "7b",
]

_MODEL_PRESETS: dict[str, str] = {
    "small": "dinov3_vit_small_lvd1689m",
    "small_plus": "dinov3_vit_small_plus_lvd1689m",
    "base": "dinov3_vit_base_lvd1689m",
    "large": "dinov3_vit_large_lvd1689m",
    "huge_plus": "dinov3_vit_huge_plus_lvd1689m",
    "7b": "dinov3_vit_7b_lvd1689m"
}


class DinoV3:
    def __init__(
            self,
            model_name: DinoV3ModelName,
            input_size
    ):
        if model_name not in _MODEL_PRESETS:
            raise ValueError(f"Invalid model name: {model_name}")
        preset = _MODEL_PRESETS[model_name]
        backbone = keras_hub.models.DINOV3Backbone.from_preset(
            preset,
            image_shape=(input_size, input_size, 3),
        )
        backbone.trainable = False
        self.backbone = backbone
        self.preset = preset

        assert input_size % self.backbone.patch_size == 0, (
            f"Input size must be a multiple of {self.backbone.patch_size}"
        )
        self.input_size = (input_size, input_size)
        self.num_prefix_tokens = 1 + self.backbone.num_register_tokens
        self.hidden_dim = self.backbone.hidden_dim

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(self.input_size)
        array = keras.preprocessing.image.img_to_array(image)
        expanded_array = np.expand_dims(array, axis=0)
        return expanded_array

    @staticmethod
    def _get_cls_token(embedding: np.ndarray) -> np.ndarray:
        cls_token = embedding[0, 0, :]
        return cls_token

    def _get_patch_embedding(self, embedding: np.ndarray) -> np.ndarray:
        patch_embedding = embedding[0, self.num_prefix_tokens:, :]
        return patch_embedding

    def __call__(
            self,
            image: Image.Image,
            return_type: Literal["cls", "patch"],
    ) -> np.ndarray:
        processed_image = self._preprocess_image(image)
        features = self.backbone({"pixel_values": processed_image})
        features = keras.ops.convert_to_numpy(features)

        if return_type == "cls":
            cls_token = self._get_cls_token(features)
            return cls_token
        elif return_type == "patch":
            patch_embedding = self._get_patch_embedding(features)
            return patch_embedding
        else:
            raise ValueError("return_type must be either 'cls' or 'patch'")
