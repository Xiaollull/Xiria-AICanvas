"""Running a quantised checkpoint without expanding it first.

A quantised checkpoint stores each linear's weight in a narrow type beside the scale that turns it
back into a real number.  There are two places that multiplication can happen.  ComfyUI does it
inside the linear op, so the weight stays in its stored form and only the layer being evaluated is
ever full precision.  This project used to do it once at load, folding every scale into a bf16
copy of the whole model, which is where two problems came from:

* **Load time.**  Expanding fp8 to bf16 on the CPU runs at about 0.2 GiB/s per core and cannot use
  the GPU, because the result is far too large to keep there.  Krea 2's 11.6 GiB of fp8 weights
  took 78 seconds of pure arithmetic before the first step could run.
* **Residency.**  The expansion doubles the model.  Krea 2 became 23.9 GiB against a 24 GB card's
  23.5 GiB, so it could never be resident: every step streamed blocks across PCIe, and a 2048x2944
  image took 690 seconds where ComfyUI took 110 on a slower card.

Folding at load is not more accurate, either.  For every scalar-scale format the two orders are
bit-identical, because casting the stored value to the compute dtype is exact and the scale is
applied once either way — verified across all 508 quantised tensors of Krea 2's transformer and
text encoder in `test_quantized_linear.py`.  What it costs is the two things above, so this module
moves the multiplication to where ComfyUI keeps it.

The scale must stay float32 to get that equality.  Rounding it to bfloat16 first changes 210 of
Krea 2's 256 transformer tensors, by up to 3.1e-2 — the one shortcut that is not free.

What a layer's marker means, and the arithmetic each layout runs, come from :mod:`comfy_quant`,
which ports ComfyUI 0.33.0 and the comfy-kitchen it pins (P9).  The formats fall into three groups:

* **FP8** — ``float8_e4m3fn`` and ``float8_e5m2`` store the layer's own matrix and one number.
  They are dequantised per layer and multiplied at full precision, exactly as before P9.
* **Weight-only integer** — ``int8_tensorwise`` (optionally ConvRot-rotated) and ``convrot_w4a4``.
  ComfyUI keeps the activation in the compute dtype and hands it to the layout's own matmul, which
  rotates it (ConvRot), quantises it per row and multiplies integers.  That is what runs here too,
  unless the marker asks for ``full_precision_matrix_mult`` or a LoRA is mounted on the layer — the
  two cases where ComfyUI dequantises instead.
* **Block-scaled float** — ``nvfp4`` and ``mxfp8`` pack their values and keep one scale per group
  of 16 or 32 in cuBLAS's tiled order.  ComfyUI's native matmul for them needs SM 10.0; below that
  it disables the format and dequantises the layer, which is the path taken here on every card.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

try:
    from . import comfy_quant
    from .comfy_quant import (
        COMFY_QUANT_SUFFIX,
        FP4_E2M1_VALUES,
        MXFP8_GROUP_SIZE,
        NVFP4_GROUP_SIZE,
        LayerQuantConfig,
        UnsupportedQuantization,
        config_for_legacy_fp8,
        parse_layer_config,
    )
except ImportError:  # pragma: no cover - flat import when run from backend/
    import comfy_quant
    from comfy_quant import (
        COMFY_QUANT_SUFFIX,
        FP4_E2M1_VALUES,
        MXFP8_GROUP_SIZE,
        NVFP4_GROUP_SIZE,
        LayerQuantConfig,
        UnsupportedQuantization,
        config_for_legacy_fp8,
        parse_layer_config,
    )

__all__ = [
    "COMFY_QUANT_SUFFIX", "FP4_E2M1_VALUES", "MXFP8_GROUP_SIZE", "NVFP4_GROUP_SIZE",
    "QuantizedLinear", "UnsupportedQuantization", "apply_quantized_linears", "checkpoint_runtime_bytes",
    "combine_lora_adapters", "dequantize_mxfp8", "dequantize_nvfp4", "dequantize_scalar_scale",
    "expand_quantized_layers", "kron_adapter_output", "logical_shape_view", "quantized_state_dict",
    "quantized_weight_bytes", "scan_quantized_layers",
]

# The two spellings of a per-tensor scale. ComfyUI's own repackages write the first; the Krea 2
# and Qwen3-VL fp8 checkpoints write the second. `.scale` alone is deliberately not a scale key:
# these checkpoints use it for ordinary norm weights.
SCALE_WEIGHT_SUFFIXES = (".scale_weight", ".weight_scale")

# An activation scale only matters to a kernel that quantises its input with a fixed scale — the
# fp8/nvfp4/mxfp8 tensor-core paths, none of which runs here. It is read and dropped.
INPUT_SCALE_SUFFIXES = (".scale_input", ".input_scale")

# AWQ-style smoothing. On a layout that declares it (NVFP4), ComfyUI multiplies the activation by it
# before anything else, so it changes the result and is carried into the module rather than dropped.
PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"

BLOCK_SCALE_SUFFIX = ".weight_scale"
GLOBAL_SCALE_SUFFIX = ".weight_scale_2"

# Module buffer name -> ComfyUI parameter name.
BUFFER_PARAMETERS = {"scale_weight": "weight_scale", "scale_weight_2": "weight_scale_2"}


@dataclass(frozen=True)
class QuantFormat:
    """How one quantisation format is stored and which scales sit beside it."""

    name: str
    storage_dtype: torch.dtype
    # Buffer name -> the suffix its tensor is stored under, relative to the layer.
    scales: tuple[tuple[str, str], ...]
    group_size: int = 0

    @property
    def packed(self) -> bool:
        """Whether the stored tensor holds two values per byte, so its shape is not the layer's."""
        return self.name in ("nvfp4", "convrot_w4a4")


_ONE_SCALE = (("scale_weight", BLOCK_SCALE_SUFFIX),)

QUANT_FORMATS: dict[str, QuantFormat] = {
    "float8_e4m3fn": QuantFormat("float8_e4m3fn", torch.float8_e4m3fn, _ONE_SCALE),
    "float8_e5m2": QuantFormat("float8_e5m2", torch.float8_e5m2, _ONE_SCALE),
    "int8_tensorwise": QuantFormat("int8_tensorwise", torch.int8, _ONE_SCALE),
    "convrot_w4a4": QuantFormat("convrot_w4a4", torch.int8, _ONE_SCALE),
    "nvfp4": QuantFormat(
        "nvfp4",
        torch.uint8,
        (("scale_weight", BLOCK_SCALE_SUFFIX), ("scale_weight_2", GLOBAL_SCALE_SUFFIX)),
        group_size=NVFP4_GROUP_SIZE,
    ),
    "mxfp8": QuantFormat("mxfp8", torch.float8_e4m3fn, _ONE_SCALE, group_size=MXFP8_GROUP_SIZE),
}

SCALAR_SCALE_FORMATS = ("float8_e4m3fn", "float8_e5m2")
INTEGER_MATMUL_FORMATS = ("int8_tensorwise", "convrot_w4a4")


@dataclass(frozen=True)
class QuantizedLayer:
    """One quantised linear, as located in a checkpoint."""

    layer: str
    format: str
    weight_key: str
    # Buffer name -> the key its tensor lives under in the checkpoint.
    scale_keys: Mapping[str, str] = field(default_factory=dict)
    # Keys consumed by this layer that the module has no parameter for.
    dropped_keys: tuple[str, ...] = ()
    config: LayerQuantConfig | None = None
    # Buffer name -> key, for parameters only some files carry (``pre_quant_scale``).
    optional_keys: Mapping[str, str] = field(default_factory=dict)

    @property
    def layer_config(self) -> LayerQuantConfig:
        return self.config or LayerQuantConfig(self.format)


def _scale_layer(key: str) -> str | None:
    """The layer a per-tensor scale belongs to, or None when the key is not one."""
    for suffix in SCALE_WEIGHT_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def scan_quantized_layers(state_dict: Mapping[str, torch.Tensor], label: str = "checkpoint") -> dict[str, QuantizedLayer]:
    """Locate every quantised linear in a checkpoint without reading its weights.

    Two layouts describe the same thing. The newer one marks each layer with ``comfy_quant`` JSON
    naming the format; the older fp8 one has no marker at all and is recognised by an fp8 weight
    sitting beside a per-tensor scale. A marker always wins, so a file carrying both spellings is
    read as the format it declares. Files that keep their markers in the safetensors header are
    read through :func:`comfy_quant.load_safetensors_state_dict`, which turns them into the same
    ``comfy_quant`` entries first.
    """
    layers: dict[str, QuantizedLayer] = {}

    for key, marker in state_dict.items():
        if not key.endswith(COMFY_QUANT_SUFFIX):
            continue
        layer = key[: -len(COMFY_QUANT_SUFFIX)]
        config = parse_layer_config(marker, layer, label)
        weight_key = f"{layer}.weight"
        if weight_key not in state_dict:
            raise UnsupportedQuantization(f"{label} layer {layer!r} is quantised but carries no weight")
        layers[layer] = _describe(state_dict, layer, config, weight_key, key, label)

    for key, value in state_dict.items():
        layer = _scale_layer(key)
        if layer is None or layer in layers:
            continue
        weight_key = f"{layer}.weight"
        weight = state_dict.get(weight_key)
        if weight is None or weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        layers[layer] = _describe(
            state_dict, layer, config_for_legacy_fp8(weight.dtype), weight_key, None, label, scale_key=key
        )

    return layers


def describe_quantized_layer(state_dict: Mapping[str, torch.Tensor], layer: str, marker, label: str = "checkpoint") -> QuantizedLayer:
    """One marked layer, validated — for loaders that expand layers one at a time."""
    config = parse_layer_config(marker, layer, label)
    weight_key = f"{layer}.weight"
    if weight_key not in state_dict:
        raise UnsupportedQuantization(f"{label} layer {layer!r} is quantised but carries no weight")
    return _describe(state_dict, layer, config, weight_key, f"{layer}{COMFY_QUANT_SUFFIX}", label)


def _describe(state_dict, layer, config: LayerQuantConfig, weight_key, marker_key, label, scale_key=None) -> QuantizedLayer:
    quant_format = QUANT_FORMATS[config.format]
    weight = state_dict[weight_key]
    if weight.dtype != quant_format.storage_dtype:
        raise UnsupportedQuantization(
            f"{label} layer {layer!r} declares {quant_format.name} but stores {weight.dtype}"
        )

    scale_keys: dict[str, str] = {}
    for buffer_name, suffix in quant_format.scales:
        if buffer_name == "scale_weight" and scale_key is not None:
            candidate = scale_key
        else:
            candidate = f"{layer}{suffix}"
            if candidate not in state_dict and buffer_name == "scale_weight":
                for alternative in SCALE_WEIGHT_SUFFIXES:
                    if f"{layer}{alternative}" in state_dict:
                        candidate = f"{layer}{alternative}"
                        break
        if candidate not in state_dict:
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} is {config.label} but carries no {BUFFER_PARAMETERS[buffer_name]}"
            )
        scale_keys[buffer_name] = candidate

    comfy_quant.validate_layer(
        config,
        tuple(weight.shape),
        {BUFFER_PARAMETERS[name]: tuple(state_dict[key].shape) for name, key in scale_keys.items()},
        None,
        None,
        layer,
        label,
    )

    # `pre_quant_scale` is a parameter of exactly the layouts whose QUANT_ALGOS entry lists it
    # (NVFP4); ComfyUI loads it there and nowhere else, so on any other layout it is dropped unread.
    optional: dict[str, str] = {}
    dropped = [key for key in (marker_key,) if key]
    candidate = f"{layer}{PRE_QUANT_SCALE_SUFFIX}"
    if candidate in state_dict:
        if "pre_quant_scale" in config.algo.parameters:
            optional["pre_quant_scale"] = candidate
        else:
            dropped.append(candidate)
    for suffix in INPUT_SCALE_SUFFIXES:
        candidate = f"{layer}{suffix}"
        if candidate in state_dict:
            dropped.append(candidate)
    return QuantizedLayer(
        layer=layer,
        format=quant_format.name,
        weight_key=weight_key,
        scale_keys=scale_keys,
        dropped_keys=tuple(dropped),
        config=config,
        optional_keys=optional,
    )


# -- dequantisation ----------------------------------------------------------------------------


def dequantize_scalar_scale(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``weight.to(dtype) * scale``, with the scale kept in float32.

    Bit-identical to folding the scale in float32 at load for every scalar-scale format, which is
    what makes deferring it free. Casting the scale to the compute dtype first is not — it is the
    one shortcut this module refuses.
    """
    scaled = weight.to(dtype=dtype) * scale.to(device=weight.device, dtype=torch.float32).reshape(())
    return scaled.to(dtype=dtype)


def dequantize_nvfp4(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Expand NVFP4 the way comfy-kitchen does: high nibble first, block scales in cuBLAS's tiled
    order. Before P9 this read the low nibble first and the scales row-major; on a real Krea 2
    layer that decoded to noise (cosine 0.001 against the BF16 build, 0.996 read correctly)."""
    return comfy_quant.dequantize_nvfp4(weight, global_scale, block_scale, dtype, out_features, in_features)


def dequantize_mxfp8(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    dtype: torch.dtype,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Expand MXFP8: fp8 values times one E8M0 scale per 32, the scales in cuBLAS's tiled order."""
    return comfy_quant.dequantize_mxfp8(weight, block_scale, dtype, out_features, in_features)


def _dequantize(config: LayerQuantConfig, weight, scales: Mapping[str, torch.Tensor], dtype, rows, cols):
    """One layer's full-precision weight; ``scales`` is keyed by buffer name."""
    if config.format in SCALAR_SCALE_FORMATS:
        return dequantize_scalar_scale(weight, scales["scale_weight"], dtype)
    parameters = {BUFFER_PARAMETERS[name]: value for name, value in scales.items()}
    return comfy_quant.dequantize_layer(weight, parameters, config, dtype, rows, cols)


# -- the module --------------------------------------------------------------------------------


_INTEGER_VIEW_DTYPES = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _apply_without_cast(fn, tensor: torch.Tensor) -> torch.Tensor:
    """Run ``Module._apply``'s function over a tensor without letting it change the dtype.

    ``fn`` is a closure over both a device and a dtype, and it only casts what reports itself as
    floating point. Viewing the storage as an integer of the same width is what makes the cast a
    no-op while the device move still happens.
    """
    if not tensor.dtype.is_floating_point:
        return fn(tensor)
    view_dtype = _INTEGER_VIEW_DTYPES.get(tensor.element_size())
    if view_dtype is None:  # pragma: no cover - every torch float width is 1, 2, 4 or 8 bytes
        return fn(tensor)
    stored = tensor.dtype
    return fn(tensor.contiguous().view(view_dtype)).view(stored)


class QuantizedLinear(nn.Linear):
    """A linear whose weight stays in the checkpoint's quantised form until it is used.

    Subclassing ``nn.Linear`` rather than ``nn.Module`` is deliberate: Diffusers' group offload,
    the hook registry that removes it, and every ``isinstance`` check in the offload machinery all
    look for a Linear, so a plain Module would quietly opt the layer out of them.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        quant_format: QuantFormat,
        storage_shape: tuple[int, ...],
        scale_shapes: Mapping[str, tuple[int, ...]],
        scale_dtypes: Mapping[str, torch.dtype],
        device=None,
        compute_dtype: torch.dtype = torch.bfloat16,
        config: LayerQuantConfig | None = None,
        optional_shapes: Mapping[str, tuple[tuple[int, ...], torch.dtype]] | None = None,
    ):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.quant_format = quant_format.name
        self.quant_config = config or LayerQuantConfig(quant_format.name)
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(
            torch.empty(storage_shape, dtype=quant_format.storage_dtype, device=device),
            requires_grad=False,
        )
        for buffer_name, _ in quant_format.scales:
            self.register_buffer(
                buffer_name,
                torch.empty(
                    scale_shapes.get(buffer_name, ()),
                    dtype=scale_dtypes.get(buffer_name, torch.float32),
                    device=device,
                ),
            )
        self._optional_buffers = tuple(sorted((optional_shapes or {})))
        for buffer_name, (shape, dtype) in (optional_shapes or {}).items():
            self.register_buffer(buffer_name, torch.empty(shape, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=compute_dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, spec: QuantizedLayer, state_dict: Mapping[str, torch.Tensor], compute_dtype):
        quant_format = QUANT_FORMATS[spec.format]
        weight = state_dict[spec.weight_key]
        scale_shapes, scale_dtypes = {}, {}
        for buffer_name, _ in quant_format.scales:
            scale = state_dict[spec.scale_keys[buffer_name]]
            scale_shapes[buffer_name] = tuple(scale.shape)
            scale_dtypes[buffer_name] = scale.dtype
        optional = {
            name: (tuple(state_dict[key].shape), state_dict[key].dtype) for name, key in spec.optional_keys.items()
        }
        rows, cols = linear.out_features, linear.in_features
        comfy_quant.validate_layer(
            spec.layer_config, tuple(weight.shape),
            {BUFFER_PARAMETERS[name]: shape for name, shape in scale_shapes.items()},
            rows, cols, spec.layer, "checkpoint",
        )
        return cls(
            in_features=cols,
            out_features=rows,
            bias=linear.bias is not None,
            quant_format=quant_format,
            storage_shape=tuple(weight.shape),
            scale_shapes=scale_shapes,
            scale_dtypes=scale_dtypes,
            device="meta",
            compute_dtype=compute_dtype,
            config=spec.layer_config,
            optional_shapes=optional,
        )

    def _protected_names(self):
        return [name for name, _ in QUANT_FORMATS[self.quant_format].scales] + list(self._optional_buffers)

    def _apply(self, fn, recurse=True):
        """Let a device move through while refusing a dtype cast.

        ``Module.to(device=..., dtype=...)`` casts every floating-point tensor it finds, and the
        runtime moves components that way. On this layer that would expand the storage the layer
        exists to keep, and — worse, because it is silent — round the float32 scale to bfloat16,
        which changes 210 of Krea 2's 256 transformer tensors. Both are handed to ``fn`` through an
        integer view of the same width, so the move happens and the cast has nothing to act on.
        """
        protected_weight = self._parameters.pop("weight", None)
        protected_scales = {
            name: self._buffers.pop(name) for name in self._protected_names() if name in self._buffers
        }
        try:
            module = super()._apply(fn, recurse)
        finally:
            if protected_weight is not None:
                moved = _apply_without_cast(fn, protected_weight.data)
                self._parameters["weight"] = nn.Parameter(moved, requires_grad=False)
            for name, tensor in protected_scales.items():
                self._buffers[name] = _apply_without_cast(fn, tensor) if tensor is not None else None
        return module

    def _scales(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name, _ in QUANT_FORMATS[self.quant_format].scales}

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return _dequantize(self.quant_config, self.weight, self._scales(), dtype, self.out_features, self.in_features)

    def set_lora_adapter(self, down: torch.Tensor, up: torch.Tensor) -> None:
        """Carry a LoRA alongside the quantised weight instead of inside it.

        Fusing is the cheaper arrangement when the weight is full precision, and it is what every
        other engine here does. It cannot be done to a quantised weight: ``W + m·BA`` has to be
        cast back to the weight's own dtype, and rounding that sum back into fp8 rounds the adapter
        away again. Expanding the layer instead works, but a Krea 2 LoRA names nearly every linear
        in the model, so mounting one expanded the whole transformer — 12.2 GiB of fp8 became
        23.9 GiB of bf16, took 255 seconds to load, and stopped fitting on the card.

        ``(x A^T) B^T`` is the same value by association, computed against the sequence rather than
        against the weight. Several adapters combine into one pair by concatenation, so the cost is
        one extra pair of rank-sized matmuls however many are mounted.

        On an integer layer this also moves the base matmul to full precision, because that is
        what ComfyUI does once a LoRA patches the layer: its weight function dequantises the weight
        and the quantised matmul is skipped.
        """
        if down.shape[0] != up.shape[1]:
            raise ValueError(f"LoRA rank mismatch: down is {tuple(down.shape)}, up is {tuple(up.shape)}")
        if down.shape[1] != self.in_features or up.shape[0] != self.out_features:
            raise ValueError(
                f"LoRA shape {tuple(up.shape)}x{tuple(down.shape)} does not fit a "
                f"{self.out_features}x{self.in_features} linear"
            )
        self.register_buffer("lora_down", down)
        self.register_buffer("lora_up", up)

    def set_kron_adapters(self, terms) -> None:
        """Carry LoKr products beside the quantised weight: ``[(w1, w2, row_offset), ...]``.

        A full-matrix LoKr delta ``w1 ⊗ w2`` is as large as the weight, so it is neither a rank-sized
        pair nor something to round into the stored weight. :func:`kron_adapter_output` applies each
        product against the sequence instead; the scale and multiplier are already in ``w1``. A
        product may cover a row slice (one of a fused qkv's q/k/v), which ``row_offset`` places.
        Like a LoRA adapter, it moves an integer layer's base matmul to full precision.
        """
        self.kron_terms = _register_kron_terms(self, terms)

    @property
    def has_lora_adapter(self) -> bool:
        return getattr(self, "lora_down", None) is not None or bool(getattr(self, "kron_terms", ()))

    @property
    def uses_integer_matmul(self) -> bool:
        """ComfyUI's ``_use_quantized`` for a weight-only layout (``mixed_precision_ops``)."""
        return (
            self.quant_format in INTEGER_MATMUL_FORMATS
            and not self.quant_config.full_precision_mm
            and not self.has_lora_adapter
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pre_quant_scale = getattr(self, "pre_quant_scale", None)
        if pre_quant_scale is not None:
            input = input * pre_quant_scale.to(device=input.device, dtype=input.dtype)
        if self.uses_integer_matmul:
            config = self.quant_config
            if self.quant_format == "int8_tensorwise":
                out = comfy_quant.int8_linear(
                    input, self.weight, self.scale_weight, self.bias, input.dtype,
                    convrot=config.convrot, convrot_groupsize=config.convrot_groupsize,
                )
            else:
                out = comfy_quant.convrot_w4a4_linear(
                    input, self.weight, self.scale_weight, self.bias,
                    convrot_groupsize=config.convrot_groupsize,
                    quant_group_size=config.quant_group_size,
                    linear_dtype=config.linear_dtype,
                )
        else:
            out = F.linear(input, self.dequantized_weight(input.dtype), self.bias)
        if getattr(self, "lora_down", None) is not None:
            out = out + F.linear(F.linear(input, self.lora_down.to(input.dtype)), self.lora_up.to(input.dtype))
        return add_kron_terms(self, out, input)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant={self.quant_config.label}"
        )


# The safetensors header spells dtypes itself; reading it avoids opening the tensors at all.
SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

QUANTIZED_STORAGE_DTYPE_NAMES = ("F8_E4M3", "F8_E5M2", "I8", "U8")


def read_safetensors_header(path, with_metadata: bool = False):
    """The tensor table at the front of a safetensors file, without mapping any tensor data."""
    import struct

    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    metadata = header.pop("__metadata__", None) or {}
    return (header, metadata) if with_metadata else header


def checkpoint_runtime_bytes(path, compute_dtype_bytes: int = 2) -> int:
    """What one component costs once loaded, counting quantised weights in their stored form.

    Budgeting every tensor at the compute dtype is what the loader used to need, because it
    expanded everything. Now that a quantised weight stays quantised, counting it at two bytes
    per element overstates Krea 2's transformer by 11.6 GiB — enough to send a model that fits
    comfortably down the block-streaming path it no longer needs.
    """
    header, metadata = read_safetensors_header(path, with_metadata=True)

    quantised_layers = {
        key[: -len(COMFY_QUANT_SUFFIX)] for key in header if key.endswith(COMFY_QUANT_SUFFIX)
    }
    quantised_layers.update(comfy_quant.header_quant_layers(metadata, str(path)))
    for key in header:
        layer = _scale_layer(key)
        if layer is None:
            continue
        weight = header.get(f"{layer}.weight")
        if weight is not None and weight.get("dtype") in ("F8_E4M3", "F8_E5M2"):
            quantised_layers.add(layer)

    total = 0
    for key, entry in header.items():
        dtype = entry.get("dtype", "")
        elements = 1
        for dimension in entry.get("shape", ()):
            elements *= int(dimension)
        layer = key.rsplit(".", 1)[0] if "." in key else ""
        if layer in quantised_layers:
            if key.endswith(COMFY_QUANT_SUFFIX) or key.endswith(INPUT_SCALE_SUFFIXES):
                continue  # dropped rather than loaded
            total += elements * SAFETENSORS_DTYPE_BYTES.get(dtype, compute_dtype_bytes)
            continue
        if dtype in SAFETENSORS_DTYPE_BYTES and SAFETENSORS_DTYPE_BYTES[dtype] > compute_dtype_bytes:
            # An fp32 tensor is narrowed to the compute dtype like everything else.
            total += elements * compute_dtype_bytes
        elif dtype in QUANTIZED_STORAGE_DTYPE_NAMES:
            # Quantised storage with no scale beside it is loaded as-is rather than widened.
            total += elements * SAFETENSORS_DTYPE_BYTES[dtype]
        else:
            total += elements * compute_dtype_bytes
    return total


def logical_weight_shape(spec: QuantizedLayer, weight: torch.Tensor) -> tuple[int, ...]:
    """The shape the layer has, which is not the stored shape once a format packs two per byte."""
    if not QUANT_FORMATS[spec.format].packed:
        return tuple(weight.shape)
    rows, packed = int(weight.shape[0]), int(weight.shape[1])
    return (rows, packed * 2)


class logical_shape_view(Mapping):
    """A checkpoint whose quantised weights report the layer's shape rather than the stored one.

    Configuration is inferred from tensor shapes, and a packed format would otherwise report half
    an input width and be read as a different model. Only ``.shape`` is corrected — this is a view
    for shape inference, not a dequantiser — so anything that reads values still sees the storage.
    """

    class _Shaped:
        __slots__ = ("shape", "dtype")

        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    def __init__(self, state_dict: Mapping[str, torch.Tensor], layers: Mapping[str, QuantizedLayer]):
        self._state_dict = state_dict
        self._packed = {
            spec.weight_key: logical_weight_shape(spec, state_dict[spec.weight_key])
            for spec in layers.values()
            if QUANT_FORMATS[spec.format].packed and spec.weight_key in state_dict
        }

    def __getitem__(self, key):
        value = self._state_dict[key]
        if key in self._packed:
            return self._Shaped(self._packed[key], value.dtype)
        return value

    def __iter__(self):
        return iter(self._state_dict)

    def __len__(self):
        return len(self._state_dict)


def expanded_layer_weight(spec: QuantizedLayer, state_dict: Mapping[str, torch.Tensor], dtype: torch.dtype,
                          fold_pre_quant_scale: bool = True) -> torch.Tensor:
    """One quantised layer as a full-precision weight — the value ComfyUI's full-precision path
    multiplies. A ``pre_quant_scale`` scales the activation's columns, which is the same as scaling
    the weight's input columns, so an expanded linear carries it inside the weight."""
    weight = state_dict[spec.weight_key]
    rows, cols = logical_weight_shape(spec, weight)
    scales = {name: state_dict[key] for name, key in spec.scale_keys.items()}
    value = _dequantize(spec.layer_config, weight, scales, dtype, rows, cols)
    pre_key = spec.optional_keys.get("pre_quant_scale")
    if fold_pre_quant_scale and pre_key is not None:
        pre = state_dict[pre_key].to(device=value.device, dtype=torch.float32).reshape(1, -1)
        value = (value.to(torch.float32) * pre).to(dtype)
    return value


def expand_quantized_layers(
    state_dict: Mapping[str, torch.Tensor],
    layers: Mapping[str, QuantizedLayer],
    targets,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Fold the scales of the named layers back into full-precision weights.

    Used for the layers a module cannot hold quantised — a quantised embedding, say. Every other
    layer stays quantised, so this costs only the layers it actually touches.
    """
    targets = {layer for layer in targets if layer in layers}
    if not targets:
        return dict(state_dict)

    consumed: set[str] = set()
    expanded: dict[str, torch.Tensor] = {}
    for layer in targets:
        spec = layers[layer]
        expanded[spec.weight_key] = expanded_layer_weight(spec, state_dict, dtype)
        consumed.update(spec.scale_keys.values())
        consumed.update(spec.dropped_keys)
        consumed.update(spec.optional_keys.values())

    resolved = {key: value for key, value in state_dict.items() if key not in consumed}
    resolved.update(expanded)
    return resolved


def combine_lora_adapters(adapters, dtype: torch.dtype):
    """Fold several LoRAs for one layer into a single ``(down, up)`` pair.

    ``sum_i m_i (x A_i^T) B_i^T`` is ``(x A^T) B^T`` with ``A`` the multiplied downs stacked on the
    rank axis and ``B`` the ups concatenated on the same axis, so a stack of adapters costs one
    pair of matmuls rather than one pair each. The multiplier goes into the down, where it is a
    scale on the smaller of the two matrices.
    """
    downs, ups = [], []
    for down, up, multiplier in adapters:
        downs.append(down.to(dtype=torch.float32) * float(multiplier))
        ups.append(up.to(dtype=torch.float32))
    return torch.cat(downs, dim=0).to(dtype), torch.cat(ups, dim=1).to(dtype)


def kron_adapter_output(input: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """``x (w1 ⊗ w2)^T`` without forming the Kronecker product.

    Read each row of ``x`` (``in = b·d``) as a ``b × d`` matrix ``X``; then ``(w1 ⊗ w2) vec(X)`` is
    ``vec(w1 X w2^T)``, in the row-major order :func:`torch.kron` uses. Two small matmuls against
    the sequence replace one against an ``(a·c) × (b·d)`` matrix the size of the layer's weight.
    """
    a, b = w1.shape
    c, d = w2.shape
    rows = input.reshape(-1, b, d)
    return torch.matmul(w1, torch.matmul(rows, w2.t())).reshape(*input.shape[:-1], a * c)


def _register_kron_terms(module: nn.Module, terms) -> tuple[tuple[int, int], ...]:
    """Validate ``(w1, w2, row_offset)`` products against ``module`` and keep them as buffers."""
    placed = []
    for index, (w1, w2, offset) in enumerate(terms):
        if w1.ndim != 2 or w2.ndim != 2:
            raise ValueError("LoKr factors must be 2-D")
        rows, cols = w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]
        offset = int(offset)
        if cols != module.in_features or offset < 0 or offset + rows > module.out_features:
            raise ValueError(
                f"LoKr product {rows}x{cols} at row {offset} does not fit a "
                f"{module.out_features}x{module.in_features} linear"
            )
        module.register_buffer(f"kron_w1_{index}", w1)
        module.register_buffer(f"kron_w2_{index}", w2)
        placed.append((rows, offset))
    return tuple(placed)


def add_kron_terms(module: nn.Module, out: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """``out`` plus every LoKr product ``module`` carries, each on its own rows."""
    for index, (rows, offset) in enumerate(getattr(module, "kron_terms", ())):
        delta = kron_adapter_output(
            input, getattr(module, f"kron_w1_{index}").to(input.dtype), getattr(module, f"kron_w2_{index}").to(input.dtype)
        )
        if rows != module.out_features:
            delta = F.pad(delta, (offset, module.out_features - offset - rows))
        out = out + delta
    return out


def quantized_weight_bytes(spec: QuantizedLayer, state_dict: Mapping[str, torch.Tensor]) -> int:
    """What one quantised linear occupies in its stored form, scales included."""
    total = state_dict[spec.weight_key].nbytes
    for key in spec.scale_keys.values():
        total += state_dict[key].nbytes
    return total


def _resolve_parent(root: nn.Module, layer: str):
    parts = layer.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None, None
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_quantized_linears(
    module: nn.Module,
    layers: Mapping[str, QuantizedLayer],
    state_dict: Mapping[str, torch.Tensor],
    compute_dtype: torch.dtype,
    label: str = "checkpoint",
) -> list[str]:
    """Replace each named ``nn.Linear`` with one that keeps its weight quantised.

    Returns ``(replaced, deferred)``. A name the module does not carry is neither: a checkpoint may
    quantise a tensor this runtime drops, and the strict load that follows is what decides whether
    the two key sets agree.

    ``deferred`` is the layers whose module is not a linear — a quantised embedding, say. There is
    nowhere to apply a scale at compute time in a module that is not a matmul, and refusing the
    file would make a checkpoint that used to load stop loading, so the caller expands those and
    keeps the fast path for everything else.
    """
    del label
    replaced, deferred = [], []
    for layer, spec in layers.items():
        parent, attribute = _resolve_parent(module, layer)
        if parent is None or not hasattr(parent, attribute):
            continue
        original = getattr(parent, attribute)
        if not isinstance(original, nn.Linear):
            deferred.append(layer)
            continue
        setattr(parent, attribute, QuantizedLinear.from_linear(original, spec, state_dict, compute_dtype))
        replaced.append(layer)
    return replaced, deferred


def quantized_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    layers: Mapping[str, QuantizedLayer],
    replaced: set[str],
) -> dict[str, torch.Tensor]:
    """Rewrite a checkpoint so the quantised layers match the modules that now hold them.

    Scales move to the module's own buffer name, the format markers and activation scales are
    dropped, and every other tensor is passed through untouched.
    """
    consumed: set[str] = set()
    renamed: dict[str, torch.Tensor] = {}
    for layer in replaced:
        spec = layers[layer]
        for buffer_name, key in spec.scale_keys.items():
            renamed[f"{layer}.{buffer_name}"] = state_dict[key]
            consumed.add(key)
        for buffer_name, key in spec.optional_keys.items():
            renamed[f"{layer}.{buffer_name}"] = state_dict[key]
            consumed.add(key)
        consumed.update(spec.dropped_keys)
    for layer, spec in layers.items():
        if layer in replaced:
            continue
        consumed.update(spec.dropped_keys)
    resolved = {key: value for key, value in state_dict.items() if key not in consumed}
    resolved.update(renamed)
    return resolved


def quantization_summary(layers: Mapping[str, QuantizedLayer], replaced) -> list[str]:
    """The distinct layout labels a loaded component runs, for the load report."""
    return sorted({spec.layer_config.label for layer, spec in layers.items() if layer in replaced})
