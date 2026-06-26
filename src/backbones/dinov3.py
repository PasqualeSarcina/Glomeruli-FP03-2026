from typing import Literal

import keras
import keras_hub
import numpy as np
from PIL import Image
import tensorflow as tf

from src.backbones.backbone import Backbone

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


class DinoV3(Backbone):
    def __init__(
            self,
            model_name: DinoV3ModelName,
            input_size,
            mode: Literal["cls", "patch", "both"]
    ):
        if model_name not in _MODEL_PRESETS:
            raise ValueError(f"Invalid model name: {model_name}")
        preset = _MODEL_PRESETS[model_name]
        backbone = keras_hub.models.DINOV3Backbone.from_preset(
            preset,
            image_shape=(input_size, input_size, 3),
        )
        backbone.trainable = False

        assert input_size % backbone.patch_size == 0, (
            f"Input size must be a multiple of {backbone.patch_size}"
        )
        self.num_prefix_tokens = 1 + backbone.num_register_tokens
        hidden_dim = backbone.hidden_dim
        if mode == "both":
            hidden_dim *= 2

        super().__init__(backbone, input_size, hidden_dim)
        self.mode = mode
        self.preset = preset

    def _preprocess_image(
            self,
            image: Image.Image
    ):
        image = image.convert("RGB")
        image_converter = keras_hub.layers.DINOV3ImageConverter.from_preset(
            self.preset,
            image_size=(self.input_size, self.input_size),
        )
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)
        processed = image_converter(array)
        return processed

    @staticmethod
    def _make_backbone_input(
            processed_image
    ):
        return {
            "pixel_values": processed_image
        }

    def _masked_mean_from_patch_tokens(
            self,
            patch_tokens,
            mask: Image.Image,
    ):
        """
        Calcola la media pesata dei patch token usando una maschera PIL.

        patch_tokens:
            Tensor senza CLS token.
            Shape attesa: [B, N, D]

        mask:
            PIL Image della maschera del glomerulo.
            Deve avere la stessa dimensione dell'immagine vista dalla backbone.

        patch_size:
            Dimensione del patch della ViT.
            Per DINOv2 ViT/14: patch_size = 14.

        Ritorna:
            weight_map:
                Tensor [1, Gh, Gw]
        """

        EPS = 1e-6

        patch_tokens = tf.convert_to_tensor(patch_tokens, dtype=tf.float32)

        if len(patch_tokens.shape) != 3:
            raise ValueError(
                "patch_tokens deve avere shape [B, N, D]. "
                "Il CLS token deve essere già stato rimosso."
            )

        num_tokens = int(patch_tokens.shape[1])
        patch_size = self.backbone.patch_size
        grid_h = self.input_size // patch_size
        grid_w = self.input_size // patch_size
        if grid_h * grid_w != num_tokens:
            raise ValueError(
                f"Mask token grid {grid_h}x{grid_w} does not match "
                f"{num_tokens} patch tokens."
            )

        # PIL mask -> array [H, W]
        mask = mask.convert("L").resize(
            (self.input_size, self.input_size),
            Image.Resampling.NEAREST,
        )
        mask_array = np.asarray(mask, dtype=np.float32)

        # Binarizzazione: tutto ciò che è > 0 diventa glomerulo
        mask_array = (mask_array > 0).astype(np.float32)

        # [H, W] -> [1, H, W, 1]
        mask_tensor = tf.convert_to_tensor(mask_array, dtype=tf.float32)
        mask_tensor = tf.expand_dims(mask_tensor, axis=0)
        mask_tensor = tf.expand_dims(mask_tensor, axis=-1)

        # Maschera pixel-level -> maschera token-level
        # Ogni valore è la frazione di copertura del token.
        weight_map = tf.image.resize(
            mask_tensor,
            size=(grid_h, grid_w),
            method="area",
            antialias=False,
        )

        # [1, Gh, Gw, 1] -> [1, Gh, Gw]
        weight_map = tf.squeeze(weight_map, axis=-1)

        # [1, Gh, Gw] -> [1, N]
        weights = tf.reshape(weight_map, [1, num_tokens])

        # Se batch > 1, stessa maschera applicata a tutti gli elementi del batch
        batch_size = tf.shape(patch_tokens)[0]
        weights = tf.tile(weights, [batch_size, 1])

        denominator = tf.reduce_sum(weights, axis=1, keepdims=True)
        denominator = tf.maximum(denominator, EPS)

        masked_mean = tf.reduce_sum(
            patch_tokens * tf.expand_dims(weights, axis=-1),
            axis=1,
        ) / denominator

        return masked_mean

    def _postprocess(
            self,
            embedding,
            mask
    ):
        if self.mode == "patch":
            patch_tokens = embedding[:, self.num_prefix_tokens:, :]
            return self._masked_mean_from_patch_tokens(patch_tokens, mask)

        cls_token = embedding[:, 0, :]

        if self.mode == "cls":
            return cls_token

        patch_tokens = embedding[:, self.num_prefix_tokens:, :]
        patch_mean = self._masked_mean_from_patch_tokens(patch_tokens, mask)
        return tf.concat([patch_mean, cls_token], axis=-1)
