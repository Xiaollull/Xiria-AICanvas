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

Formats fall into two groups:

* **Scalar scale** — ``float8_e4m3fn``, ``float8_e5m2`` and ``int8_tensorwise`` store an ordinary
  matrix of the layer's own shape and one number.  Recovering the weight is a cast and a multiply,
  which is what makes them exactly foldable and exactly deferrable.
* **Block scale** — ``nvfp4`` and ``mxfp8`` store a packed or sub-byte matrix with one scale per
  group of elements along the input axis.  ComfyUI reads these through ``comfy_kitchen``'s
  compiled kernels, which are not a dependency here; the dequantisation below is the same
  arithmetic in plain PyTorch, so such a file loads and runs rather than being refused, without
  claiming ComfyUI's fused matmul performance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

# `<layer>.comfy_quant` holds JSON naming the format. Everything else beside a quantised weight is
# addressed relative to the layer, so one layer name is enough to gather a whole quantised linear.
COMFY_QUANT_SUFFIX = ".comfy_quant"

# The two spellings of a per-tensor scale. ComfyUI's own repackages write the first; the Krea 2
# and Qwen3-VL fp8 checkpoints write the second. `.scale` alone is deliberately not a scale key:
# these checkpoints use it for ordinary norm weights.
SCALE_WEIGHT_SUFFIXES = (".scale_weight", ".weight_scale")

# An activation scale only matters to a kernel that quantises its input. Nothing here does, so it
# is read and dropped rather than carried into a module that has no use for it.
INPUT_SCALE_SUFFIXES = (".scale_input", ".input_scale", ".pre_quant_scale")

BLOCK_SCALE_SUFFIX = ".weight_scale"
GLOBAL_SCALE_SUFFIX = ".weight_scale_2"

# FP4 E2M1, the 16 values a nibble can hold. Index is the raw code: sign in bit 3, then two
# exponent bits and one mantissa bit.
FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

NVFP4_GROUP_SIZE = 16
MXFP8_GROUP_SIZE = 32


class UnsupportedQuantization(ValueError):
    """A quantised layer this loader cannot express, named rather than silently approximated."""


@dataclass(frozen=True)
class QuantFormat:
    """How one quantisation format is stored and how it is turned back into a weight."""

    name: str
    storage_dtype: torch.dtype
    # Buffer name -> the suffix its tensor is stored under, relative to the layer.
    scales: tuple[tuple[str, str], ...]
    group_size: int = 0

    @property
    def packed(self) -> bool:
        """Whether the stored tensor's shape differs from the layer's shape."""
        return self.name == "nvfp4"


QUANT_FORMATS: dict[str, QuantFormat] = {
    "float8_e4m3fn": QuantFormat("float8_e4m3fn", torch.float8_e4m3fn, (("scale_weight", BLOCK_SCALE_SUFFIX),)),
    "float8_e5m2": QuantFormat("float8_e5m2", torch.float8_e5m2, (("scale_weight", BLOCK_SCALE_SUFFIX),)),
    "int8_tensorwise": QuantFormat("int8_tensorwise", torch.int8, (("scale_weight", BLOCK_SCALE_SUFFIX),)),
    "nvfp4": QuantFormat(
        "nvfp4",
        torch.uint8,
        (("scale_weight", BLOCK_SCALE_SUFFIX), ("scale_weight_2", GLOBAL_SCALE_SUFFIX)),
        group_size=NVFP4_GROUP_SIZE,
    ),
    "mxfp8": QuantFormat(
        "mxfp8",
        torch.float8_e4m3fn,
        (("scale_weight", BLOCK_SCALE_SUFFIX),),
        group_size=MXFP8_GROUP_SIZE,
    ),
}

SCALAR_SCALE_FORMATS = ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise")


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


def _scale_layer(key: str) -> str | None:
    """The layer a per-tensor scale belongs to, or None when the key is not one."""
    for suffix in SCALE_WEIGHT_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _quant_format_name(marker: torch.Tensor, layer: str) -> str:
    """Read the format out of a ``comfy_quant`` tensor, which is JSON stored as bytes."""
    try:
        text = bytes(marker.to(dtype=torch.uint8).reshape(-1).tolist()).decode("utf-8")
        configuration = json.loads(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise UnsupportedQuantization(f"Layer {layer!r} has an unreadable comfy_quant marker") from error
    if not isinstance(configuration, dict) or not configuration.get("format"):
        raise UnsupportedQuantization(f"Layer {layer!r} declares no quantisation format")
    return str(configuration["format"])


def scan_quantized_layers(state_dict: Mapping[str, torch.Tensor], label: str = "checkpoint") -> dict[str, QuantizedLayer]:
    """Locate every quantised linear in a checkpoint without reading its weights.

    Two layouts describe the same thing. The newer one marks each layer with ``comfy_quant`` JSON
    naming the format; the older fp8 one has no marker at all and is recognised by an fp8 weight
    sitting beside a per-tensor scale. A marker always wins, so a file carrying both spellings is
    read as the format it declares.
    """
    layers: dict[str, QuantizedLayer] = {}

    for key, marker in state_dict.items():
        if not key.endswith(COMFY_QUANT_SUFFIX):
            continue
        layer = key[: -len(COMFY_QUANT_SUFFIX)]
        name = _quant_format_name(marker, layer)
        if name not in QUANT_FORMATS:
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} is quantised as {name!r}, which this loader cannot read; "
                "use an fp8, int8, nvfp4 or unquantised file instead"
            )
        weight_key = f"{layer}.weight"
        if weight_key not in state_dict:
            raise UnsupportedQuantization(f"{label} layer {layer!r} is quantised but carries no weight")
        layers[layer] = _describe(state_dict, layer, QUANT_FORMATS[name], weight_key, key, label)

    for key, value in state_dict.items():
        layer = _scale_layer(key)
        if layer is None or layer in layers:
            continue
        weight_key = f"{layer}.weight"
        weight = state_dict.get(weight_key)
        if weight is None or weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        name = "float8_e4m3fn" if weight.dtype == torch.float8_e4m3fn else "float8_e5m2"
        layers[layer] = _describe(state_dict, layer, QUANT_FORMATS[name], weight_key, None, label, scale_key=key)

    return layers


def _describe(state_dict, layer, quant_format, weight_key, marker_key, label, scale_key=None) -> QuantizedLayer:
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
                f"{label} layer {layer!r} is {quant_format.name} but carries no {buffer_name}"
            )
        scale_keys[buffer_name] = candidate

    if quant_format.group_size == 0:
        scale = state_dict[scale_keys["scale_weight"]]
        if scale.numel() != 1:
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} carries a {scale.numel()}-element scale where "
                f"{quant_format.name} expects one per-tensor value"
            )

    dropped = [key for key in (marker_key,) if key]
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
    """Expand NVFP4: two E2M1 nibbles per byte, one fp8 scale per 16 elements, one global scale."""
    codes = weight.reshape(out_features, -1)
    low = codes & 0x0F
    high = (codes >> 4) & 0x0F
    # The low nibble is the earlier element, so the two halves interleave rather than concatenate.
    nibbles = torch.stack((low, high), dim=-1).reshape(out_features, -1)
    if nibbles.shape[1] < in_features:
        raise UnsupportedQuantization(
            f"An NVFP4 weight unpacked to {nibbles.shape[1]} columns, short of the layer's {in_features}"
        )
    nibbles = nibbles[:, :in_features]

    table = torch.tensor(FP4_E2M1_VALUES, device=weight.device, dtype=torch.float32)
    values = table[nibbles.long()]

    scales = block_scale.to(device=weight.device, dtype=torch.float32).reshape(out_features, -1)
    groups = scales.shape[1]
    expanded = scales.repeat_interleave(NVFP4_GROUP_SIZE, dim=1)
    if expanded.shape[1] < in_features:
        raise UnsupportedQuantization(
            f"An NVFP4 weight carries {groups} block scales, too few for {in_features} columns"
        )
    values = values * expanded[:, :in_features]
    values = values * global_scale.to(device=weight.device, dtype=torch.float32).reshape(())
    return values.to(dtype=dtype)


def dequantize_mxfp8(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    dtype: torch.dtype,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Expand MXFP8: an fp8 matrix with one power-of-two scale per 32 elements."""
    values = weight.reshape(out_features, -1).to(dtype=torch.float32)
    scales = block_scale.reshape(out_features, -1)
    if scales.dtype == torch.uint8:
        # E8M0: the byte is a raw exponent, biased by 127, with 255 reserved for NaN.
        exponent = scales.to(device=weight.device, dtype=torch.float32) - 127.0
        scales = torch.exp2(exponent)
    else:
        scales = scales.to(device=weight.device, dtype=torch.float32)
    expanded = scales.repeat_interleave(MXFP8_GROUP_SIZE, dim=1)
    if expanded.shape[1] < in_features or values.shape[1] < in_features:
        raise UnsupportedQuantization(
            f"An MXFP8 weight expands to {min(expanded.shape[1], values.shape[1])} columns, "
            f"short of the layer's {in_features}"
        )
    return (values[:, :in_features] * expanded[:, :in_features]).to(dtype=dtype)


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
    ):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.quant_format = quant_format.name
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
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            quant_format=quant_format,
            storage_shape=tuple(weight.shape),
            scale_shapes=scale_shapes,
            scale_dtypes=scale_dtypes,
            device="meta",
            compute_dtype=compute_dtype,
        )

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
            name: self._buffers.pop(name)
            for name, _ in QUANT_FORMATS[self.quant_format].scales
            if name in self._buffers
        }
        try:
            module = super()._apply(fn, recurse)
        finally:
            if protected_weight is not None:
                moved = _apply_without_cast(fn, protected_weight.data)
                self._parameters["weight"] = nn.Parameter(moved, requires_grad=False)
            for name, tensor in protected_scales.items():
                self._buffers[name] = _apply_without_cast(fn, tensor)
        return module

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.quant_format in SCALAR_SCALE_FORMATS:
            return dequantize_scalar_scale(self.weight, self.scale_weight, dtype)
        if self.quant_format == "nvfp4":
            return dequantize_nvfp4(
                self.weight, self.scale_weight, self.scale_weight_2, dtype, self.out_features, self.in_features
            )
        if self.quant_format == "mxfp8":
            return dequantize_mxfp8(self.weight, self.scale_weight, dtype, self.out_features, self.in_features)
        raise UnsupportedQuantization(f"No dequantisation for {self.quant_format!r}")

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

    @property
    def has_lora_adapter(self) -> bool:
        return getattr(self, "lora_down", None) is not None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight(input.dtype)
        out = F.linear(input, weight, self.bias)
        if self.has_lora_adapter:
            out = out + F.linear(F.linear(input, self.lora_down.to(input.dtype)), self.lora_up.to(input.dtype))
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant={self.quant_format}"
        )


# The safetensors header spells dtypes itself; reading it avoids opening the tensors at all.
SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

QUANTIZED_STORAGE_DTYPE_NAMES = ("F8_E4M3", "F8_E5M2", "I8", "U8")


def read_safetensors_header(path) -> dict:
    """The tensor table at the front of a safetensors file, without mapping any tensor data."""
    import json as _json
    import struct

    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = _json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def checkpoint_runtime_bytes(path, compute_dtype_bytes: int = 2) -> int:
    """What one component costs once loaded, counting quantised weights in their stored form.

    Budgeting every tensor at the compute dtype is what the loader used to need, because it
    expanded everything. Now that a quantised weight stays quantised, counting it at two bytes
    per element overstates Krea 2's transformer by 11.6 GiB — enough to send a model that fits
    comfortably down the block-streaming path it no longer needs.
    """
    header = read_safetensors_header(path)

    quantised_layers = {
        key[: -len(COMFY_QUANT_SUFFIX)] for key in header if key.endswith(COMFY_QUANT_SUFFIX)
    }
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


def expand_quantized_layers(
    state_dict: Mapping[str, torch.Tensor],
    layers: Mapping[str, QuantizedLayer],
    targets,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Fold the scales of the named layers back into full-precision weights.

    Used for the layers a LoRA patches. ``fuse_flux_lora_state_dict`` adds its delta into the base
    weight and casts the sum back to the base dtype, which for a quantised weight would round the
    adapter away; expanding first is what keeps a LoRA exact. Every other layer stays quantised, so
    mounting one costs only the layers it actually touches.
    """
    targets = {layer for layer in targets if layer in layers}
    if not targets:
        return dict(state_dict)

    consumed: set[str] = set()
    expanded: dict[str, torch.Tensor] = {}
    for layer in targets:
        spec = layers[layer]
        weight = state_dict[spec.weight_key]
        quant_format = QUANT_FORMATS[spec.format]
        scales = {name: state_dict[key] for name, key in spec.scale_keys.items()}
        if spec.format in SCALAR_SCALE_FORMATS:
            value = dequantize_scalar_scale(weight, scales["scale_weight"], dtype)
        elif spec.format == "nvfp4":
            rows, columns = logical_weight_shape(spec, weight)
            value = dequantize_nvfp4(
                weight, scales["scale_weight"], scales["scale_weight_2"], dtype, rows, columns
            )
        elif spec.format == "mxfp8":
            rows, columns = logical_weight_shape(spec, weight)
            value = dequantize_mxfp8(weight, scales["scale_weight"], dtype, rows, columns)
        else:  # pragma: no cover - every registered format is handled above
            raise UnsupportedQuantization(f"No dequantisation for {spec.format!r}")
        expanded[spec.weight_key] = value
        consumed.update(spec.scale_keys.values())
        consumed.update(spec.dropped_keys)
        del quant_format

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
        consumed.update(spec.dropped_keys)
    for layer, spec in layers.items():
        if layer in replaced:
            continue
        consumed.update(spec.dropped_keys)
    resolved = {key: value for key, value in state_dict.items() if key not in consumed}
    resolved.update(renamed)
    return resolved
