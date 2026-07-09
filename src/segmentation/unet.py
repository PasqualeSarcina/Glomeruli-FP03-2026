"""
U-Net model for glomeruli semantic segmentation.

Input:
    RGB patch: (384, 384, 3)

Output:
    Pixel-wise 2-class softmax mask:
        class 0 = background
        class 1 = glomerulus
"""

import tensorflow as tf
import keras
from keras import layers


def conv_block(x, filters, name):
    """
    Standard U-Net convolution block:
    Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    """

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)

    return x


def encoder_block(x, filters, name):
    """
    U-Net encoder block:
    convolution block followed by max pooling.
    Returns both the skip connection and the pooled output.
    """

    skip = conv_block(x, filters, name=f"{name}_convblock")
    pooled = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(skip)

    return skip, pooled


def decoder_block(x, skip, filters, name):
    """
    U-Net decoder block:
    upsample, concatenate with skip connection, then convolution block.
    """

    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name=f"{name}_upsample",
    )(x)

    x = layers.Concatenate(name=f"{name}_concat")([x, skip])

    x = conv_block(x, filters, name=f"{name}_convblock")

    return x


def build_unet(
    input_shape=(384, 384, 3),
    num_classes=2,
    base_filters=64,
    dropout_rate=0.0,
):
    """
    Build standard U-Net model.

    Args:
        input_shape: input image shape.
        num_classes: number of output segmentation classes.
        base_filters: number of filters in the first encoder block.
        dropout_rate: dropout applied at the bottleneck.

    Returns:
        Keras Model.
    """

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Encoder
    skip1, x = encoder_block(inputs, base_filters, name="encoder1")
    skip2, x = encoder_block(x, base_filters * 2, name="encoder2")
    skip3, x = encoder_block(x, base_filters * 4, name="encoder3")
    skip4, x = encoder_block(x, base_filters * 8, name="encoder4")

    # Bottleneck
    x = conv_block(x, base_filters * 16, name="bottleneck")

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="bottleneck_dropout")(x)

    # Decoder
    x = decoder_block(x, skip4, base_filters * 8, name="decoder4")
    x = decoder_block(x, skip3, base_filters * 4, name="decoder3")
    x = decoder_block(x, skip2, base_filters * 2, name="decoder2")
    x = decoder_block(x, skip1, base_filters, name="decoder1")

    # Output
    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation="softmax",
        name="segmentation_mask",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="unet_glomeruli_segmentation",
    )

    return model


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for the glomerulus class.

    y_true:
        shape (batch, height, width, 1)
        integer labels 0 or 1

    y_pred:
        shape (batch, height, width, 2)
        softmax probabilities
    """

    y_true = tf.cast(y_true, tf.float32)

    if len(y_true.shape) == 4:
        y_true = tf.squeeze(y_true, axis=-1)

    y_pred_glomerulus = y_pred[..., 1]

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_glomerulus, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice


def combined_loss(y_true, y_pred):
    """
    Sparse categorical cross-entropy + Dice loss.

    This helps because most pixels are background.
    Dice loss encourages the model to learn the glomerulus class.
    """

    if len(y_true.shape) == 4:
        y_true_squeezed = tf.squeeze(y_true, axis=-1)
    else:
        y_true_squeezed = y_true

    ce = keras.losses.sparse_categorical_crossentropy(
        y_true_squeezed,
        y_pred,
    )

    ce = tf.reduce_mean(ce)

    return ce + dice_loss(y_true, y_pred)


def compile_unet(
    model,
    initial_lr=0.001,
    loss_fn="combined",
    miou_metric=None,
):
    """
    Compile U-Net using the same general style as Federico's SegNet:
        optimizer: SGD
        momentum: 0.9
        weight_decay: 1e-4
        metric: MeanIoU
    """

    optimizer = keras.optimizers.SGD(
        learning_rate=initial_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    if loss_fn == "combined":
        loss = combined_loss
    elif loss_fn in ("crossentropy", "sparse_ce", "sparse_categorical_crossentropy"):
        loss = "sparse_categorical_crossentropy"
    else:
        raise ValueError(
            f"Unknown loss_fn: {loss_fn}. "
            "Use 'combined' or 'crossentropy'."
        )

    if miou_metric is None:
        miou_metric = keras.metrics.MeanIoU(
            num_classes=2,
            sparse_y_pred=False,
            name="mean_io_u",
        )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            miou_metric,
        ],
    )

    return model