"""Algorithmic image resizing, matching ComfyUI's `ImageScale` body method for method.

The resample chain resizes a picture in three places — after a model upscale, in place of one when a
unit carries no model, and once more when a unit asks for its exact target back after the engine
rounded it — and all three have to agree, both with each other and with the node a user would have
built the same chain from in ComfyUI. The original five methods stay matched to
`comfy/utils.py:common_upscale`; Forge's pixel-space `Nearest` is additive and deliberately uses
Pillow rather than Torch:

- `nearest-exact`, `bilinear`, `area`, `bicubic` go through `torch.nn.functional.interpolate` with
  torch's own `align_corners` default, exactly as Comfy passes them.
- `nearest` uses Pillow's nearest-neighbour resize, matching Forge's pixel upscaler entry. It is not
  an alias for Torch's existing `nearest-exact` compatibility mode.
- `lanczos` is not an interpolate mode at all. Comfy converts to 8-bit RGB, calls PIL's
  `Image.resize(LANCZOS)` and converts back, so the channel order and the quantisation to bytes are
  both part of the result — which is why it is done the same way rather than approximated.

Comfy works in NHWC float tensors because that is what travels between its nodes; this pipeline
holds pictures as PIL images, so the conversion happens at the edges of this module. The arithmetic
in between is the same.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# `nodes.py:1883`. `bislerp` is deliberately absent: it is a latent-space method Comfy exposes on the
# latent nodes, not on `ImageScale`, and nothing in the chain resizes latents.
RESIZE_METHODS = ("nearest", "nearest-exact", "bilinear", "area", "bicubic", "lanczos")
DEFAULT_RESIZE_METHOD = "lanczos"


def scaled_size(size, scale: float):
    """`round(edge × scale)`, the size `ImageScaleBy` computes (`nodes.py:1929-1930`).

    Python's `round` is half-to-even, which is what Comfy gets, so the tie cases match rather than
    being off by one. No alignment is applied: a chain size is exact, and an engine that needs its
    own grid rounds from here.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Resample scale must be a positive finite number")
    width, height = (size.size if isinstance(size, Image.Image) else tuple(size))
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def align_size(size, alignment: int):
    """The nearest size the engine's latent grid accepts, never below one cell."""
    if not isinstance(alignment, int) or alignment < 1:
        raise ValueError("Alignment must be a positive integer")
    return tuple(max(alignment, int(round(edge / alignment)) * alignment) for edge in tuple(size))


def _lanczos(image: Image.Image, width: int, height: int):
    # `comfy/utils.py:1059-1067`: uint8 RGB through PIL and back. Comfy round-trips a tensor, so the
    # only step it takes that this does not is the one that would undo itself.
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def _nearest(image: Image.Image, width: int, height: int):
    """Forge's pixel `Nearest`, kept distinct from Torch `nearest-exact`."""
    return image.resize((width, height), resample=Image.Resampling.NEAREST)


def _interpolate(image: Image.Image, width: int, height: int, method: str):
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).unsqueeze(0)
    # `comfy/utils.py:1095`: size is (height, width) and `align_corners` is left at torch's default.
    resized = F.interpolate(tensor, size=(height, width), mode=method)
    # Comfy leaves the overshoot bicubic produces in the tensor and lets the VAE or the PNG writer
    # clamp it; this returns a picture, so the clamp happens here instead of nowhere.
    values = resized.squeeze(0).permute(1, 2, 0).clamp_(0, 1).numpy()
    return Image.fromarray(np.rint(values * 255).astype(np.uint8), "RGB")


def resize_image(image: Image.Image, width: int, height: int, method: str = DEFAULT_RESIZE_METHOD):
    """Resize `image` to exactly `width × height`, stretching (Comfy's `crop="disabled"`)."""
    if not isinstance(image, Image.Image):
        raise TypeError("resize_image requires a PIL image")
    if method not in RESIZE_METHODS:
        raise ValueError(f"Unsupported resize algorithm: {method}")
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("Resize target must be at least one pixel on each edge")
    source = image if image.mode == "RGB" else image.convert("RGB")
    if source.size == (width, height):
        return source
    if method == "lanczos":
        return _lanczos(source, width, height)
    if method == "nearest":
        return _nearest(source, width, height)
    return _interpolate(source, width, height, method)
