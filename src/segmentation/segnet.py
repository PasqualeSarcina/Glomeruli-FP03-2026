import tensorflow as tf
import keras
from keras import layers


def build_segnet_vgg19(
    input_shape: tuple = (384, 384, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.0,
) -> keras.Model:
    """
    SegNet-VGG19 encoder-decoder for glomerulus segmentation.

    Architecture overview:
    - Encoder: VGG19 pretrained on ImageNet, 5 pooling stages.
      Each max-pool halves spatial dimensions:
        input (384x384) -> block1_pool (192x192) -> block2_pool (96x96)
                        -> block3_pool (48x48)   -> block4_pool (24x24)
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

    x = _decoder_block(e5, filters=512, name="dec5", dropout_rate=dropout_rate)
    x = _decoder_block(x,  filters=512, name="dec4", dropout_rate=dropout_rate)
    x = _decoder_block(x,  filters=256, name="dec3", dropout_rate=dropout_rate)
    x = _decoder_block(x,  filters=128, name="dec2", dropout_rate=dropout_rate)
    x = _decoder_block(x,  filters=64,  name="dec1", dropout_rate=dropout_rate)

    outputs = keras.layers.Conv2D(
        num_classes,
        kernel_size=(1, 1),
        activation="softmax",
        name="output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def _decoder_block(x: tf.Tensor, filters: int, name: str, dropout_rate: float = 0.0) -> tf.Tensor:
    x = layers.UpSampling2D(size=(2, 2), name=f"{name}_up")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)
    return x


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Soft Dice loss on the glomerulus class (class 1).

    y_true: (B, H, W, 1) int32  — integer masks from the dataset pipeline
    y_pred: (B, H, W, 2) float32 — softmax probabilities
    """
    y_true_f = tf.cast(tf.squeeze(y_true, axis=-1), tf.float32)
    y_pred_f = y_pred[..., 1]
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Sparse categorical cross-entropy + Dice loss.

    Cross-entropy handles per-pixel calibration across both classes.
    Dice directly optimises overlap on the minority glomerulus class,
    compensating for the heavy background dominance that makes
    cross-entropy alone collapse toward predicting all-background.
    """
    bce = tf.reduce_mean(
        keras.losses.sparse_categorical_crossentropy(
            tf.squeeze(y_true, axis=-1), y_pred
        )
    )
    return bce + dice_loss(y_true, y_pred)


def compile_segnet(
    model: keras.Model,
    initial_lr: float = 0.01,
    loss_fn: str = "combined",
    miou_metric: keras.metrics.MeanIoU | None = None,
) -> keras.Model:
    """
    Compile SegNet with SGD + momentum.

    loss_fn: "combined" (BCE + Dice, default) or "crossentropy".
    initial_lr default is 0.01 — the original 0.1 delayed val IoU
    improvement to epoch 5 in the first training run.

    miou_metric: pass a shared MeanIoU instance to keep the same metric
    name (e.g. val_mean_io_u) across multiple compile() calls. Without
    this, Keras dedupes by name on re-compile and suffixes the new one
    (mean_io_u_1), which breaks ModelCheckpoint(monitor=...) in two-phase
    training. If None, a fresh instance is created (single-phase usage).
    """
    optimizer = keras.optimizers.SGD(
        learning_rate=initial_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    loss = combined_loss if loss_fn == "combined" else "sparse_categorical_crossentropy"

    if miou_metric is None:
        miou_metric = keras.metrics.MeanIoU(num_classes=2, sparse_y_pred=False)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy", miou_metric],
    )

    return model


def freeze_encoder(model: keras.Model) -> None:
    """Freeze all VGG19 encoder layers (block1–block5). Call compile_segnet after."""
    for layer in model.layers:
        if layer.name.startswith("block"):
            layer.trainable = False


def unfreeze_encoder(model: keras.Model) -> None:
    """Unfreeze all layers. Call compile_segnet after to apply the change."""
    for layer in model.layers:
        layer.trainable = True
