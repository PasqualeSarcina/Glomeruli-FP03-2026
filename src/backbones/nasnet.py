import keras
import numpy as np
from PIL import Image
from keras.src.applications.nasnet import preprocess_input


class NASNet():
    def __init__(
            self,
            input_size: int = 331
    ):
        assert input_size > 32 == 0, "Input size should be greater than 32"
        self.input_size = input_size
        self.backbone = keras.applications.NASNetLarge(
            include_top=False,
            input_shape=(self.input_size, self.input_size, 3),
            weights="imagenet",
            pooling="avg",
            name="nasnet_large",
        )
        self.backbone.trainable = False

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((self.input_size, self.input_size))

        array = keras.preprocessing.image.img_to_array(image)
        array = array.astype("float32")

        expanded_array = np.expand_dims(array, axis=0)

        preprocessed_array = preprocess_input(expanded_array)

        return preprocessed_array

    def __call__(
        self,
        image: Image.Image
    ) -> np.ndarray:
        x = self._preprocess_image(image)
        embedding = self.backbone.predict(x, verbose=0)
        return embedding.squeeze(0).astype("float32")