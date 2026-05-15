import tensorflow as tf
import keras
from keras import layers


def build_segnet_vgg19(
    input_shape: tuple = (400, 400, 3),
    num_classes: int = 2,
) -> keras.Model:
    """
    SegNet-VGG19 encoder-decoder for glomerulus segmentation.

    Architecture overview:
    - Encoder: VGG19 pretrained on ImageNet, 5 pooling stages.
      Each max-pool halves spatial dimensions:
        input (400x400) -> block1_pool (200x200) -> block2_pool (100x100)
                        -> block3_pool (50x50)   -> block4_pool (25x25)
                        -> block5_pool (12x12)
    - Decoder: 5 upsampling blocks mirroring encoder depth.
    - Output: 1x1 Conv2D + softmax for per-pixel class probabilities.

    Reference: Bueno et al. 2020.
    """
    inputs = keras.Input(shape=input_shape)

    vgg19 = keras.applications.VGG19(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet",
    )

    e5 = vgg19.get_layer("block5_pool").output

    x = _decoder_block(e5, filters=512, name="dec5")
    x = _decoder_block(x,  filters=512, name="dec4")
    x = _decoder_block(x,  filters=256, name="dec3")
    x = _decoder_block(x,  filters=128, name="dec2")
    x = _decoder_block(x,  filters=64,  name="dec1")

    outputs = keras.layers.Conv2D(
        num_classes,
        kernel_size=(1, 1),
        activation="softmax",
        name="output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def _decoder_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.UpSampling2D(size=(2, 2), name=f"{name}_up")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def compile_segnet(model: keras.Model, initial_lr: float = 0.1) -> keras.Model:
    """
    Compile SegNet with SGD + momentum and sparse categorical crossentropy.

    SGD + momentum is chosen over Adam following Bueno et al. (2020):
    more stable convergence under heavy class imbalance
    (background >> glomerulus pixels).

    sparse_categorical_crossentropy expects integer masks (0/1),
    not one-hot. MeanIoU is reported alongside accuracy because accuracy
    is misleading when background pixels dominate (~99% of each patch).
    """
    optimizer = keras.optimizers.SGD(
        learning_rate=initial_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.MeanIoU(num_classes=2)],
    )

    return model


def lr_step_decay(epoch: int, lr: float) -> float:
    """Step decay x0.1 every 2 epochs. Pass to keras.callbacks.LearningRateScheduler."""
    if (epoch + 1) % 2 == 0:
        return lr * 0.1
    return lr
