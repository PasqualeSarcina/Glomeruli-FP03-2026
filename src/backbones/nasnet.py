from tensorflow import keras
import numpy as np
from PIL import Image
from tensorflow.keras.applications.nasnet import preprocess_input

from src.backbones.backbone import Backbone


class NASNet(Backbone):
    def __init__(
            self,
            input_size: int = 331
    ):
        assert input_size > 32, "Input size should be greater than 32"
        
        backbone = keras.applications.NASNetLarge(
            include_top=False,
            input_shape=(input_size, input_size, 3),
            weights="imagenet",
            pooling=None,
            name="nasnet_large",
        )
        backbone.trainable = False
        hidden_dim = backbone.output_shape[-1]
        
        super().__init__(backbone, input_size, hidden_dim)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((self.input_size, self.input_size))

        array = keras.preprocessing.image.img_to_array(image)
        array = array.astype("float32")

        expanded_array = np.expand_dims(array, axis=0)

        preprocessed_array = preprocess_input(expanded_array)

        return preprocessed_array
