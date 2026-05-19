import tensorflow as tf
import keras
from keras import layers


def build_segnet_vgg19(
    input_shape: tuple = (384, 384, 3),
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
) -> keras.Model:
    """
    Compile SegNet with SGD + momentum.

    loss_fn: "combined" (BCE + Dice, default) or "crossentropy".
    initial_lr default is 0.01 — the original 0.1 delayed val IoU
    improvement to epoch 5 in the first training run.

    SGD + momentum is used over Adam following Bueno et al. (2020):
    more stable convergence under heavy class imbalance.
    MeanIoU is tracked alongside accuracy because accuracy is
    misleading when background pixels dominate each patch.
    """
    optimizer = keras.optimizers.SGD(
        learning_rate=initial_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    loss = combined_loss if loss_fn == "combined" else "sparse_categorical_crossentropy"

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy", keras.metrics.MeanIoU(num_classes=2, sparse_y_pred=False)],
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


def lr_step_decay(epoch: int, lr: float) -> float:
    """Step decay x0.1 every 2 epochs. Pass to keras.callbacks.LearningRateScheduler."""
    if (epoch + 1) % 2 == 0:
        return lr * 0.1
    return lr
