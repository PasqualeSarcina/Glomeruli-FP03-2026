"""
Convert Lunit's pathology-pretrained SwAV ResNet50 (PyTorch) into a Keras
encoder that build_segnet_resnet50() loads as its backbone, in place of the
ImageNet-pretrained VGG19.

WHY THIS SCRIPT EXISTS (and why the Keras side is hand-built)
--------------------------------------------------------------
torchvision.models.resnet50 and keras.applications.ResNet50 are NOT the same
architecture: torchvision is ResNet50 **v1.5** (the bottleneck's stride-2 is
on the 3x3 conv), Keras applications is **v1** (stride-2 on the first 1x1).
Same layer shapes, so a naive layer-by-layer copy into keras.applications
*runs* but produces wrong features — an earlier attempt did exactly this and
the verification below caught it (output diff ~5.2, not floating-point noise).

Automatic converters were ruled out on this cluster: onnx2tf needs Python
>=3.10 (the conversion venv is 3.9) plus a heavy dependency stack, and nobuco
only supports Keras 2 (training runs on Keras 3 / TF 2.21). So the encoder is
hand-built to match torchvision v1.5 exactly — see
src/segmentation/segnet.py::build_resnet50_v15_encoder, which this script
imports so the transplant target is byte-identical to the training graph.
The layer-by-layer copy then needs zero extra runtime dependency, and the
numerical verification below is the gate: if torch and Keras don't agree to
floating-point precision, nothing is saved.

This script runs ONCE (offline, CPU is enough — one ResNet50 forward pass),
producing a `.weights.h5` the GPU training job just loads. It needs no CUDA,
so it runs in its own throwaway CPU-only venv (see setup_and_convert.sh),
keeping torch out of the `glomeruli` conda env that Run 1-7 depend on.

WHAT WE DON'T KNOW FOR CERTAIN (flagged so you can fix it fast if it breaks)
-----------------------------------------------------------------------------
1. The exact key layout inside the downloaded .torch checkpoint. Lunit's
   README shows loading via `torch.hub.load_state_dict_from_url(...)` into
   their own `resnet50(pretrained=True, key="SwAV")` helper, but does not
   publish the raw state_dict key names. SwAV/MoCo-style checkpoints are
   *commonly* wrapped (e.g. {"state_dict": {...}}, or keys prefixed with
   "module." from DataParallel training). We defensively unwrap both cases
   below. If loading still fails, print the actual top-level keys you see
   and adjust `_unwrap_state_dict`.
2. The exact input normalization Lunit used at pretraining time. Their
   README doesn't state it. We assume the near-universal convention for
   torchvision-based SSL pipelines (ImageNet mean/std), which is what
   SwAV's own reference implementation uses by default. If downstream
   results look off, this is the first thing to revisit.

Both of these are exactly why step 5 (numerical verification) exists: it
will fail loudly if (1) is wrong, and gives you a concrete number to judge
whether (2) plausibly matters.

USAGE (inside the CPU-only conversion venv — see train_run8_swav.sbatch)
---------------------------------------------------------------------------
    python scripts/convert_encoder_weights.py \
        --output models/encoders/swav_resnet50.weights.h5
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Pathology-pretrained ResNet50 checkpoints, keyed by source name.
# - Lunit (swav/mocov2/barlowtwins): lunit-io/benchmark-ssl-pathology GitHub
#   Releases; ResNet50 backbones self-supervised on TCGA, different SSL methods.
#   Non-commercial / research-only (Lunit Public License).
# - retccl: Xiyue-Wang/RetCCL, contrastive on ~32k WSIs; here via an unofficial
#   HuggingFace mirror (direct URL, no Google Drive). GPLv3 (academic use OK).
#   RetCCL uses a *modified* ResNet50 — its checkpoint may not map cleanly to
#   torchvision's key names. _unwrap_state_dict/_load_torch_resnet50 fail LOUDLY
#   on the login node if the backbone doesn't line up (no GPU wasted); adjust
#   _unwrap_state_dict then.
_LUNIT_BASE = ("https://github.com/lunit-io/benchmark-ssl-pathology/releases/"
               "download/pretrained-weights/")
PATHOLOGY_URLS = {
    "swav": _LUNIT_BASE + "swav_rn50_ep200.torch",
    "mocov2": _LUNIT_BASE + "mocov2_rn50_ep200.torch",
    "barlowtwins": _LUNIT_BASE + "bt_rn50_ep200.torch",
    "retccl": "https://huggingface.co/jamesdolezal/RetCCL/resolve/main/retccl.pth",
}

# Standard ImageNet normalization stats. See docstring point (2) above —
# this is an informed assumption, not a confirmed fact from Lunit's docs.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 384  # matches SegmentationDataset.INPUT_SIZE used everywhere else


def _download_checkpoint(url: str, cache_path: Path) -> Path:
    """Download the .torch checkpoint once; reuse it on subsequent runs."""
    if cache_path.exists():
        print(f"Using cached checkpoint: {cache_path}")
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {cache_path}")
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def _unwrap_state_dict(raw: dict) -> dict:
    """
    Defensively normalize whatever `torch.load(...)` handed us into a flat
    {layer_name: tensor} state_dict matching torchvision.models.resnet50's
    own naming (conv1.weight, bn1.weight, layer1.0.conv1.weight, ...).

    SwAV/MoCo-style checkpoints are commonly saved either as a raw
    state_dict, or wrapped one level deeper under a "state_dict" key, and
    sometimes every key is prefixed with "module." because the reference
    training code wrapped the model in torch.nn.DataParallel. We peel off
    both wrappers if present. This is exactly the point flagged as
    uncertain in the module docstring — if this still doesn't line up,
    print `raw.keys()` (or the nested dict's keys) to see what you actually
    got and adjust here.
    """
    state_dict = raw
    # unwrap common one-level containers ("state_dict", "model", "network")
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "network"):
            if key in raw and isinstance(raw[key], dict):
                state_dict = raw[key]
                break

    # strip "module." (DataParallel) always
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    # If the backbone stem isn't at torchvision's expected key ("conv1.weight"),
    # try stripping a common wrapper prefix so RetCCL-style checkpoints line up.
    # We only accept a prefix that actually exposes "conv1.weight" — no blind
    # renaming.
    if "conv1.weight" not in state_dict:
        for prefix in ("backbone.", "encoder.", "resnet.", "model.", "features."):
            stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if "conv1.weight" in stripped:
                print(f"  (unwrapped backbone prefix '{prefix}')")
                state_dict = stripped
                break

    return state_dict


def _load_torch_resnet50(checkpoint_path: Path):
    """Build a torchvision ResNet50 and load the SwAV pathology weights into it."""
    import torch
    import torchvision

    model = torchvision.models.resnet50(weights=None)

    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(raw)

    # strict=False: SwAV checkpoints don't have a matching `fc` head (SwAV's
    # projection head has a different shape than ImageNet's 1000-way fc),
    # so a full strict load would fail on that layer even though everything
    # we actually need (conv1..layer4) matches fine. We check explicitly
    # that the backbone loaded, rather than trusting silence.
    result = model.load_state_dict(state_dict, strict=False)
    backbone_missing = [k for k in result.missing_keys if not k.startswith("fc.")]
    if backbone_missing:
        raise RuntimeError(
            "ResNet50 backbone weights did not load correctly — missing keys "
            f"outside of the (expected-to-differ) fc head: {backbone_missing}\n"
            "This means the checkpoint's key layout doesn't match "
            "torchvision's resnet50 naming. Inspect the checkpoint with "
            "`torch.load(path, map_location='cpu').keys()` and adjust "
            "_unwrap_state_dict()."
        )
    print(f"Loaded PyTorch ResNet50 backbone. Ignored (head-only) keys: "
          f"{[k for k in result.missing_keys if k.startswith('fc.')]}")

    model.eval()  # freeze BatchNorm running stats for the verification pass
    return model


def _load_torch_imagenet_resnet50():
    """
    Build a torchvision ResNet50 with its standard ImageNet-supervised weights.

    This is the Run 9 control: same v1.5 architecture as the SwAV path, only
    the pretraining differs (ImageNet natural images vs pathology WSIs). Same
    ImageNet mean/std input normalization applies, so nothing downstream
    changes. torchvision fetches these weights from its own CDN on first use
    (needs internet -> login node), caching them under ~/.cache/torch/hub.
    IMAGENET1K_V2 is torchvision's stronger training recipe; V1 would also be
    a valid ImageNet baseline.
    """
    import torchvision

    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    print("Loaded PyTorch ResNet50 with ImageNet (IMAGENET1K_V2) weights.")
    model.eval()
    return model


def _build_keras_encoder():
    """
    Build the target Keras encoder via build_resnet50_v15_encoder from the
    training code — the SINGLE source of truth for the encoder architecture.
    Using the same builder here guarantees the graph we transplant weights
    into is byte-identical to the one build_segnet_resnet50 uses at training
    time, so the saved .weights.h5 loads with an exact topology match.

    (This hand-built encoder is torchvision-v1.5-compatible; keras.applications
    .ResNet50 is v1 and would NOT match the SwAV weights — the reason the
    earlier layer-by-layer attempt into keras.applications failed verification.)
    """
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.segmentation.segnet import build_resnet50_v15_encoder
    return build_resnet50_v15_encoder(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), name="resnet50_encoder")


def _copy_conv(torch_conv, keras_model, keras_name: str) -> None:
    """
    Copy one Conv2D's kernel. PyTorch stores conv kernels as
    (out_channels, in_channels, kH, kW); Keras/TF wants (kH, kW,
    in_channels, out_channels) — a straight axis transpose, no reshaping.

    Both our hand-built encoder and torchvision use use_bias=False on every
    conv (the following BatchNorm absorbs the bias), so there is exactly one
    weight array per conv. (The len()==2 branch is a defensive fallback in
    case a conv ever gains a bias; it should not trigger here.)
    """
    w = torch_conv.weight.detach().numpy().transpose(2, 3, 1, 0)
    keras_layer = keras_model.get_layer(keras_name)
    if len(keras_layer.weights) == 2:
        bias = np.zeros(w.shape[-1], dtype=w.dtype)
        keras_layer.set_weights([w, bias])
    else:
        keras_layer.set_weights([w])


def _copy_bn(torch_bn, keras_model, keras_name: str) -> None:
    """
    Copy one BatchNorm's parameters. Both frameworks store these as plain
    1D vectors in the same [gamma, beta, moving_mean, moving_var] order
    (PyTorch: weight, bias, running_mean, running_var) — no transpose
    needed. The epsilon (1e-5, set to match torch in build_resnet50_v15_encoder)
    is a layer hyperparameter, not a weight, so it isn't copied here.
    """
    keras_model.get_layer(keras_name).set_weights([
        torch_bn.weight.detach().numpy(),
        torch_bn.bias.detach().numpy(),
        torch_bn.running_mean.detach().numpy(),
        torch_bn.running_var.detach().numpy(),
    ])


def _copy_bottleneck_block(torch_block, keras_model, name: str) -> None:
    """
    One residual bottleneck: conv1(1x1) -> conv2(3x3) -> conv3(1x1), plus an
    optional projection shortcut (`downsample`) on the first block of each
    stage. Keras layer names match build_resnet50_v15_encoder's scheme:
    `<name>_conv1/bn1`, `_conv2/bn2`, `_conv3/bn3`, and `_ds_conv/_ds_bn`.
    """
    _copy_conv(torch_block.conv1, keras_model, f"{name}_conv1")
    _copy_bn(torch_block.bn1, keras_model, f"{name}_bn1")
    _copy_conv(torch_block.conv2, keras_model, f"{name}_conv2")
    _copy_bn(torch_block.bn2, keras_model, f"{name}_bn2")
    _copy_conv(torch_block.conv3, keras_model, f"{name}_conv3")
    _copy_bn(torch_block.bn3, keras_model, f"{name}_bn3")

    if torch_block.downsample is not None:
        _copy_conv(torch_block.downsample[0], keras_model, f"{name}_ds_conv")
        _copy_bn(torch_block.downsample[1], keras_model, f"{name}_ds_bn")


def transplant_weights(torch_model, keras_model) -> None:
    """
    Walk both models in lockstep, copying every weight. torchvision's
    layer1..layer4 map to our stages s1..s4; block i (0-based in torch) maps
    to our `s{s}b{i+1}` naming. The stem is torch conv1/bn1 -> keras
    stem_conv/stem_bn.
    """
    _copy_conv(torch_model.conv1, keras_model, "stem_conv")
    _copy_bn(torch_model.bn1, keras_model, "stem_bn")

    for s, stage_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
        torch_stage = getattr(torch_model, stage_name)
        for b in range(len(torch_stage)):
            _copy_bottleneck_block(torch_stage[b], keras_model, f"s{s}b{b + 1}")

    print("Weight transplant complete: stem + 16 residual blocks copied (3+4+6+3).")


def verify_conversion(torch_model, keras_model) -> float:
    """
    The safety net: run the SAME random image through both models and
    compare outputs. A correct transplant should match near-exactly
    (floating point only); a wrong stage mapping, a missed transpose, or a
    checkpoint that didn't actually load will show up as a large,
    unmistakable difference — not a subtle one. If this raises, do NOT use
    the resulting .weights.h5 for training.
    """
    import torch

    rng = np.random.default_rng(seed=0)
    image = rng.uniform(0.0, 1.0, size=(1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)

    # Same ImageNet normalization both models are assumed to expect
    # (see docstring point 2). Applied here explicitly so this check
    # exercises the exact numbers, not just the architecture.
    normalized = (image - IMAGENET_MEAN) / IMAGENET_STD

    with torch.no_grad():
        torch_input = torch.from_numpy(normalized.transpose(0, 3, 1, 2))  # NHWC -> NCHW
        torch_out = torch_model.conv1(torch_input)
        torch_out = torch_model.bn1(torch_out)
        torch_out = torch_model.relu(torch_out)
        torch_out = torch_model.maxpool(torch_out)
        torch_out = torch_model.layer1(torch_out)
        torch_out = torch_model.layer2(torch_out)
        torch_out = torch_model.layer3(torch_out)
        torch_out = torch_model.layer4(torch_out)
        torch_out = torch_out.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC

    keras_out = keras_model(normalized, training=False).numpy()

    if torch_out.shape != keras_out.shape:
        raise RuntimeError(
            f"Shape mismatch after conversion: torch={torch_out.shape} "
            f"keras={keras_out.shape}. The two graphs are not aligned — do "
            "not trust these weights."
        )

    max_abs_diff = float(np.max(np.abs(torch_out - keras_out)))
    print(f"Verification: output shape {keras_out.shape}, "
          f"max abs difference PyTorch vs Keras = {max_abs_diff:.6f}")

    # Loose-ish tolerance on purpose: cuDNN vs PyTorch's conv/BN kernels
    # don't guarantee bit-identical results even for a correct transplant.
    # A REAL bug (wrong stage, missed layer, garbage weights) produces
    # differences of order 1-100, not 1e-2 — so this threshold still
    # catches what matters.
    if max_abs_diff > 0.05:
        raise RuntimeError(
            f"Verification FAILED: max abs difference {max_abs_diff:.4f} is "
            "too large to be floating-point noise. Do not train with these "
            "weights — re-check the stage/block mapping in _STAGE_MAP and "
            "the checkpoint key layout in _unwrap_state_dict()."
        )
    return max_abs_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("swav", "mocov2", "barlowtwins", "retccl", "imagenet"), default="swav",
        help="Which pretrained weights to convert: 'swav'/'mocov2'/'barlowtwins' "
             "(Lunit pathology ResNet50), 'retccl' (RetCCL pathology ResNet50, via "
             "HF mirror), or 'imagenet' (torchvision control). Same v1.5 arch.",
    )
    parser.add_argument(
        "--checkpoint-cache", type=Path, default=None,
        help="Where to cache the downloaded Lunit checkpoint. Defaults to "
             "models/encoders/<source>_rn50.torch. Ignored for --source imagenet.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/encoders/swav_resnet50.weights.h5"),
        help="Where to write the converted Keras weights.",
    )
    args = parser.parse_args()

    if args.output.exists():
        print(f"{args.output} already exists — skipping conversion. "
              "Delete it first if you want to redo the conversion.")
        return

    if args.source in PATHOLOGY_URLS:
        cache = args.checkpoint_cache or Path(f"models/encoders/{args.source}_rn50.torch")
        checkpoint_path = _download_checkpoint(PATHOLOGY_URLS[args.source], cache)
        print(f"\n=== Loading PyTorch {args.source} ResNet50 (pathology-pretrained) ===")
        torch_model = _load_torch_resnet50(checkpoint_path)
    else:
        print("\n=== Loading PyTorch ImageNet ResNet50 (control) ===")
        torch_model = _load_torch_imagenet_resnet50()

    print("\n=== Building untrained Keras ResNet50-v1.5 encoder (torchvision-matched) ===")
    keras_model = _build_keras_encoder()

    print("\n=== Transplanting weights ===")
    transplant_weights(torch_model, keras_model)

    print("\n=== Verifying conversion (PyTorch vs Keras on the same input) ===")
    verify_conversion(torch_model, keras_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    keras_model.save_weights(str(args.output))
    print(f"\nSaved converted encoder weights to {args.output}")
    print("Safe to use with build_segnet_resnet50(encoder_weights_path=...).")


if __name__ == "__main__":
    main()
