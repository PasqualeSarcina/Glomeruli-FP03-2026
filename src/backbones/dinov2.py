from os import PathLike
from typing import Literal

import PIL
import keras
import keras_hub
import numpy as np
from PIL import Image

ImageSource = Image.Image | str | PathLike[str]

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
        self.hidden_dim = self.backbone.hidden_dim

        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _load_image(image: ImageSource) -> Image.Image:
        if isinstance(image, Image.Image):
            return image

        with Image.open(image) as loaded_image:
            return loaded_image.copy()

    def _preprocess_mask(self, mask: ImageSource) -> Image.Image:
        mask = self._load_image(mask).convert("L").resize(
            self.input_size,
            Image.Resampling.NEAREST,
        )
        mask_array = np.asarray(mask, dtype=np.uint8)
        binary_mask = (mask_array > 127).astype(np.uint8) * 255
        return Image.fromarray(binary_mask)

    def _preprocess_image(
            self,
            image: ImageSource,
            mask: Image.Image | None = None
    ) -> np.ndarray:
        image = self._load_image(image).convert("RGB").resize(self.input_size)
        if mask is not None:
            black_background = Image.new("RGB", self.input_size, (0, 0, 0))
            image = Image.composite(image, black_background, mask)

        array = keras.preprocessing.image.img_to_array(image).astype(np.float32)

        # Scale to  [0, 1]
        array = array / 255.0

        # ImageNet norm
        array = (array - self.imagenet_mean) / self.imagenet_std

        expanded_array = np.expand_dims(array, axis=0)
        return expanded_array

    @staticmethod
    def _get_cls_token(embedding: np.ndarray) -> np.ndarray:
        cls_token = embedding[0, 0, :]
        return cls_token

    def _get_patch_embedding(
            self,
            embedding: np.ndarray,
            mask: Image.Image | None = None
    ) -> np.ndarray:
        patch_embedding = embedding[0, 1:, :]
        if mask is None:
            return patch_embedding

        patch_size = self.backbone.patch_size
        mask_array = np.asarray(mask, dtype=bool)
        grid_height = mask_array.shape[0] // patch_size
        grid_width = mask_array.shape[1] // patch_size
        patch_mask = mask_array.reshape(
            grid_height,
            patch_size,
            grid_width,
            patch_size,
        ).any(axis=(1, 3)).reshape(-1)

        patch_mask = patch_mask >= 0.3
        patch_embedding = patch_embedding[patch_mask]
        return patch_embedding

    def __call__(
            self,
            image: ImageSource,
            return_type: Literal["cls", "patch"],
            mask: ImageSource | None = None
    ) -> np.ndarray:
        processed_mask = self._preprocess_mask(mask) if mask is not None else None
        processed_image = self._preprocess_image(image, processed_mask)
        features = self.backbone({"images": processed_image})
        features = keras.ops.convert_to_numpy(features)

        if return_type == "cls":
            cls_token = self._get_cls_token(features)
            return cls_token
        elif return_type == "patch":
            patch_embedding = self._get_patch_embedding(features, processed_mask)
            return patch_embedding
        else:
            raise ValueError("return_type must be either 'cls' or 'patch'")
