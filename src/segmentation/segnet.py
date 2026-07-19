# Makes the `str | None` style type hints below lazy (evaluated as strings),
# so this module also imports under Python 3.9 — needed because the weight
# conversion venv (scripts/encoders/setup_and_convert.sh) runs on the login
# node's system Python 3.9 and imports build_resnet50_v15_encoder from here.
# Harmless on the 3.12 training env.
from __future__ import annotations

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


# ImageNet mean/std, channel order matches RGB (SegmentationDataset decodes
# PNGs as RGB, not BGR). Lunit's README doesn't document the exact
# normalization used to pretrain the SwAV weights, so this is the standard
# torchvision-SSL convention, not a confirmed fact — see the conversion
# script's docstring for the caveat.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# BatchNorm epsilon: torch's default is 1e-5, but Keras's BatchNormalization
# defaults to 1e-3. Using Keras's default would make the converted encoder
# numerically diverge from the PyTorch original — so every BN here pins 1e-5.
_BN_EPS = 1e-5
# momentum 0.9 ≈ torch BN momentum 0.1 (they use opposite conventions:
# keras_running = m*running + (1-m)*batch, torch_running = (1-m)*running +
# m*batch). Only affects fine-tuning updates, not the copied inference stats.
_BN_MOMENTUM = 0.9


def _resnet_bottleneck(x, width: int, out_ch: int, stride: int, name: str, downsample: bool):
    """
    One torchvision-style (v1.5) bottleneck block.

    v1.5 vs v1: the stride-2 lives on the 3x3 conv (conv2), NOT on the first
    1x1 (conv1) — this is the single difference that makes torchvision and
    keras.applications.ResNet50 weight-incompatible, and the reason this
    encoder is hand-built instead of taken from keras.applications.

    All convs use_bias=False (BatchNorm right after absorbs the bias), exactly
    like torchvision — so the weight transplant copies kernels only, no bias.

    The 3x3 conv uses explicit ZeroPadding2D(1) + padding='valid' rather than
    padding='same'. For stride 2, Keras 'same' pads asymmetrically (0 on the
    top/left, 1 on the bottom/right), whereas torch pads symmetrically (1 each
    side). Explicit symmetric padding reproduces torch's exact output.
    """
    shortcut = x
    if downsample:
        # Projection shortcut: present on the first block of every stage
        # (channel count always changes; spatial also changes when stride=2).
        shortcut = layers.Conv2D(out_ch, 1, strides=stride, use_bias=False, name=f"{name}_ds_conv")(x)
        shortcut = layers.BatchNormalization(epsilon=_BN_EPS, momentum=_BN_MOMENTUM, name=f"{name}_ds_bn")(shortcut)

    y = layers.Conv2D(width, 1, strides=1, use_bias=False, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(epsilon=_BN_EPS, momentum=_BN_MOMENTUM, name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)

    y = layers.ZeroPadding2D(padding=1, name=f"{name}_pad2")(y)
    y = layers.Conv2D(width, 3, strides=stride, padding="valid", use_bias=False, name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(epsilon=_BN_EPS, momentum=_BN_MOMENTUM, name=f"{name}_bn2")(y)
    y = layers.ReLU(name=f"{name}_relu2")(y)

    y = layers.Conv2D(out_ch, 1, strides=1, use_bias=False, name=f"{name}_conv3")(y)
    y = layers.BatchNormalization(epsilon=_BN_EPS, momentum=_BN_MOMENTUM, name=f"{name}_bn3")(y)

    y = layers.Add(name=f"{name}_add")([y, shortcut])
    y = layers.ReLU(name=f"{name}_out")(y)
    return y


def build_resnet50_v15_encoder(
    input_shape: tuple = (384, 384, 3),
    name: str = "resnet50_encoder",
) -> keras.Model:
    """
    ResNet50 **v1.5** (torchvision-compatible) feature extractor, output stride 32.

    Hand-built to match `torchvision.models.resnet50` layer-for-layer so that
    Lunit's SwAV pathology weights (trained on that exact architecture) can be
    transplanted with ~zero numerical drift. `keras.applications.ResNet50` is
    v1 (stride on the 1x1) and is therefore NOT weight-compatible — see
    `_resnet_bottleneck` and `scripts/encoders/convert_swav_resnet50.py`.

    Single source of truth for the encoder architecture: both the training
    code (`build_segnet_resnet50`) and the weight-conversion script import
    THIS function, so the graph the weights are copied into is byte-identical
    to the graph that trains. For a 384x384 input the output is 12x12x2048.
    """
    inputs = keras.Input(shape=input_shape)

    # Stem: 7x7 conv stride 2 (pad 3) -> BN -> ReLU -> 3x3 maxpool stride 2 (pad 1).
    # Explicit ZeroPadding (not 'same') to match torch's symmetric padding.
    x = layers.ZeroPadding2D(padding=3, name="stem_pad")(inputs)
    x = layers.Conv2D(64, 7, strides=2, padding="valid", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(epsilon=_BN_EPS, momentum=_BN_MOMENTUM, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="stem_pool")(x)

    # (num_blocks, bottleneck_width, output_channels, stride_of_first_block).
    # These are exactly torchvision's layer1..layer4.
    stage_cfg = [
        (3, 64, 256, 1),    # layer1 (stride 1: channels change but spatial doesn't)
        (4, 128, 512, 2),   # layer2
        (6, 256, 1024, 2),  # layer3
        (3, 512, 2048, 2),  # layer4
    ]
    for s, (n_blocks, width, out_ch, first_stride) in enumerate(stage_cfg, start=1):
        for b in range(1, n_blocks + 1):
            stride = first_stride if b == 1 else 1
            # First block of every stage needs a projection shortcut (channels
            # always change from the stage input); later blocks use identity.
            x = _resnet_bottleneck(x, width, out_ch, stride, name=f"s{s}b{b}", downsample=(b == 1))

    return keras.Model(inputs, x, name=name)


def build_segnet_resnet50(
    input_shape: tuple = (384, 384, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.0,
    encoder_weights_path: str | None = None,
) -> keras.Model:
    """
    SegNet-ResNet50 encoder-decoder, same decoder as build_segnet_vgg19 —
    only the encoder changes. This is what makes it a fair one-axis
    comparison against Run 4/6/7: identical decoder, loss, optimizer,
    training protocol; the only thing that differs is what the encoder was
    pretrained on (ImageNet natural images vs pathology WSI patches).

    Why ResNet50 slots in without changing the decoder at all: it has
    exactly 5 stride-2 downsampling stages (stem conv, stem maxpool, and
    the first block of each of the 3 remaining residual stages), same as
    VGG19's 5 max-pools — so a 384x384 input still bottlenecks at 12x12,
    and the existing 5-block decoder (each block upsamples x2) still
    reconstructs back to 384x384 with no changes.

    encoder_weights_path: path to the converted pathology-pretrained
    weights (see scripts/encoders/convert_swav_resnet50.py). If None, the
    encoder is left randomly initialized (mostly useful for architecture
    sanity-checks, not real training).
    """
    inputs = keras.Input(shape=input_shape)

    # SegmentationDataset already scales images to [0, 1] (see
    # dataset.py:_load_image), same as what VGG19 receives today. But
    # unlike VGG19 here we additionally apply ImageNet mean/std
    # normalization, because the SwAV pathology weights were pretrained
    # with that normalization (the standard torchvision-SSL convention) —
    # skipping it would feed the pretrained encoder inputs it never saw
    # during pretraining, quietly wasting most of the benefit of using
    # domain-pretrained weights in the first place.
    normalized = layers.Normalization(
        mean=_IMAGENET_MEAN,
        variance=[s ** 2 for s in _IMAGENET_STD],  # Normalization wants variance, not std
        name="imagenet_normalize",
    )(inputs)

    # Hand-built ResNet50-v1.5 encoder (torchvision-compatible) — NOT
    # keras.applications.ResNet50, which is v1 and weight-incompatible with
    # the SwAV checkpoint (see build_resnet50_v15_encoder). Built from the
    # SAME function the conversion script transplants into, so load_weights()
    # is an exact-topology load. It stays a single nested submodel named
    # "resnet50_encoder", so freeze/unfreeze flips the whole encoder with one
    # `.trainable` toggle (see freeze_resnet_encoder).
    encoder = build_resnet50_v15_encoder(input_shape=input_shape, name="resnet50_encoder")
    if encoder_weights_path is not None:
        encoder.load_weights(encoder_weights_path)

    e5 = encoder(normalized)  # (12, 12, 2048) for a 384x384 input — same spatial size as VGG19's block5_pool, 4x the channels

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
    miou_metric: keras.metrics.MeanIoU | None = None,
) -> keras.Model:
    """
    Compile SegNet with SGD + momentum and the combined BCE+Dice loss.

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

    if miou_metric is None:
        miou_metric = keras.metrics.MeanIoU(num_classes=2, sparse_y_pred=False)

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
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


def freeze_resnet_encoder(model: keras.Model) -> None:
    """
    Freeze the whole ResNet50 encoder in one shot. build_segnet_resnet50
    wraps the backbone as a single nested submodel named "resnet50_encoder",
    so setting that one layer's `.trainable = False` recursively freezes
    every conv/BN weight inside it — no per-layer name matching to get
    wrong. The decoder ("dec*") and output head are separate top-level
    layers and stay trainable. Call compile_segnet after to apply.

    Side effect (desirable here): a frozen nested model also runs its
    BatchNorm layers in inference mode, so the encoder's pretrained BN
    running statistics are NOT disturbed during the frozen warm-up phase —
    standard transfer-learning practice.
    """
    model.get_layer("resnet50_encoder").trainable = False


def unfreeze_resnet_encoder(model: keras.Model) -> None:
    """Unfreeze the ResNet50 encoder for full fine-tuning. Call compile_segnet after."""
    model.get_layer("resnet50_encoder").trainable = True


def tta_predict(model, images):
    """
    Test-time augmentation over the full D4 group (8 orientations: 4 rotations x
    optional horizontal flip). For each orientation g we predict on g(images),
    map the softmax probabilities back to the original frame with g^-1, and
    average across all 8. Histology has no canonical orientation, so each g is
    label-preserving; averaging the views is a free inference-time boost (no
    retraining). Returns averaged softmax probabilities, same shape as one
    forward pass. Shared by evaluate.py and loso_cv.py.
    """
    acc = tf.zeros_like(model(images, training=False))
    for flip in (False, True):
        x = tf.image.flip_left_right(images) if flip else images
        for k in range(4):
            p = model(tf.image.rot90(x, k=k), training=False)
            p = tf.image.rot90(p, k=(4 - k) % 4)      # undo rotation
            if flip:
                p = tf.image.flip_left_right(p)        # undo flip
            acc += p
    return acc / 8.0
