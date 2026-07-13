import tensorflow as tf
import keras
from keras import layers

from src.segmentation.segnet import compile_segnet


def _decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    name: str,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """SegNet-style decoder block with a U-Net skip concatenation."""
    x = layers.UpSampling2D(size=(2, 2), name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])

    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)

    return x


def build_unet_vgg19(
    input_shape: tuple = (384, 384, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.0,
) -> keras.Model:
    """
    U-Net with the same ImageNet-pretrained VGG19 encoder and decoder widths
    used by Federico's SegNet-VGG19.

    The intended comparison changes the decoder connectivity:
    SegNet has no encoder-to-decoder skip concatenations; this model does.
    """
    inputs = keras.Input(shape=input_shape)

    vgg19 = keras.applications.VGG19(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet",
    )

    skip1 = vgg19.get_layer("block1_conv2").output  # 384 x 384 x 64
    skip2 = vgg19.get_layer("block2_conv2").output  # 192 x 192 x 128
    skip3 = vgg19.get_layer("block3_conv4").output  # 96 x 96 x 256
    skip4 = vgg19.get_layer("block4_conv4").output  # 48 x 48 x 512
    skip5 = vgg19.get_layer("block5_conv4").output  # 24 x 24 x 512

    x = vgg19.get_layer("block5_pool").output       # 12 x 12 x 512

    x = _decoder_block(x, skip5, 512, "dec5", dropout_rate)
    x = _decoder_block(x, skip4, 512, "dec4", dropout_rate)
    x = _decoder_block(x, skip3, 256, "dec3", dropout_rate)
    x = _decoder_block(x, skip2, 128, "dec2", dropout_rate)
    x = _decoder_block(x, skip1, 64, "dec1", dropout_rate)

    outputs = layers.Conv2D(
        num_classes,
        kernel_size=(1, 1),
        activation="softmax",
        name="output",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="unet_vgg19")


def freeze_encoder(model: keras.Model) -> None:
    """Freeze VGG19 block layers. Recompile after calling."""
    for layer in model.layers:
        if layer.name.startswith("block"):
            layer.trainable = False


def unfreeze_encoder(model: keras.Model) -> None:
    """Unfreeze the complete network. Recompile after calling."""
    for layer in model.layers:
        layer.trainable = True


def compile_unet_vgg19(
    model: keras.Model,
    initial_lr: float = 0.01,
    loss_fn: str = "combined",
    miou_metric: keras.metrics.MeanIoU | None = None,
) -> keras.Model:
    """Reuse Federico's optimizer, loss, and metric configuration exactly."""
    return compile_segnet(
        model=model,
        initial_lr=initial_lr,
        loss_fn=loss_fn,
        miou_metric=miou_metric,
    )
