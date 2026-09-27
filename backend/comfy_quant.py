"""ComfyUI's quantised layouts, read and executed the way ComfyUI reads and executes them.

A quantised safetensors file is only half a description of a model. The other half is the code
that knows what each layer's ``comfy_quant`` marker means: which tensor is the storage, which
scales sit beside it, whether the weight was rotated before it was quantised, and which matmul
the layer runs through. This module is that half, ported rather than re-derived, so a file means
here exactly what it means in ComfyUI.

Provenance (every port below keeps the reference's arithmetic; modifications are listed):

* ComfyUI 0.33.0 (GPL-3.0, Copyright Comfy Org), the checkout at ``F:\\AI\\Ai ComfyUi\\ComfyUI``:
  ``comfy/quant_ops.py::QUANT_ALGOS`` (format table), ``comfy/ops.py::_load_quantized_module``
  (marker and scale reading, per-format parameters and their defaults),
  ``comfy/ops.py::mixed_precision_ops`` (which layers use the quantised matmul and which
  dequantise), ``comfy/utils.py::convert_old_quants`` (header ``_quantization_metadata``) and
  ``comfy/model_management.py::supports_{fp8,nvfp4,mxfp8}_compute`` (capability gates).
* comfy-kitchen 0.2.31 (Apache-2.0, Copyright Comfy Org; its NVFP4 decoding derives from PyTorch
  AO, BSD-3-Clause), the version ComfyUI 0.33.0 pins: ``tensor/int8.py``, ``tensor/int8_utils.py``,
  ``tensor/convrot_w4a4.py``, ``backends/eager/quantization.py``, ``backends/eager/convrot_w4a4.py``,
  ``backends/eager/svdquant.py`` (int4 packing) and ``float_utils.py`` (cuBLAS scale swizzle, E8M0).
  Licence texts and notices: ``backend/resources/comfy-quant/``.

Modifications: the kernels are plain functions instead of ``torch.library`` custom ops and
registry entries; ``input_act`` fusion and stochastic rounding are left out (nothing here uses
them); ConvRot W4A4 accumulates its integer matmul exactly in int32 and picks int4 or int8
activations the way the pinned CUDA backend does (``backends/cuda/__init__.py``), where the eager
reference multiplies unpacked integers in the output dtype; markers are validated strictly and
unknown fields are refused instead of ignored.

On CUDA the two quantised matmuls run on the fastest of three backends: the pinned comfy-kitchen
when it is installed with a usable CUDA backend (never installed by this program), else the built-in
Triton kernels of :mod:`comfy_quant_triton` — bit-identical to the ports below, verified against
them on first use and dropped in favour of them on any failure — else the ports themselves. None of
them touches a layer that carries no marker.
"""

from __future__ import annotations

import contextlib
import json
import math
import struct
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

COMFY_QUANT_SUFFIX = ".comfy_quant"
HEADER_QUANT_METADATA_KEY = "_quantization_metadata"
PINNED_COMFY_KITCHEN = "0.2.31"
COMFY_REFERENCE = "ComfyUI 0.33.0 / comfy-kitchen 0.2.31"


class UnsupportedQuantization(ValueError):
    """A quantised layer this runtime cannot express, named rather than silently approximated."""


# -- the format table (comfy/quant_ops.py::QUANT_ALGOS) ---------------------------------------


@dataclass(frozen=True)
class QuantAlgo:
    name: str
    label: str
    storage_dtype: torch.dtype
    parameters: frozenset
    layout: str
    group_size: int = 0
    # False for the weight-only layouts: the layer keeps its activation in the compute dtype and
    # hands it to the layout's own matmul, which quantises it internally.
    quantize_input: bool = True


QUANT_ALGOS: dict[str, QuantAlgo] = {
    "float8_e4m3fn": QuantAlgo(
        "float8_e4m3fn", "FP8 E4M3", torch.float8_e4m3fn,
        frozenset({"weight_scale", "input_scale"}), "TensorCoreFP8E4M3Layout",
    ),
    "float8_e5m2": QuantAlgo(
        "float8_e5m2", "FP8 E5M2", torch.float8_e5m2,
        frozenset({"weight_scale", "input_scale"}), "TensorCoreFP8E5M2Layout",
    ),
    "nvfp4": QuantAlgo(
        "nvfp4", "NVFP4", torch.uint8,
        frozenset({"weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale"}),
        "TensorCoreNVFP4Layout", group_size=16,
    ),
    "mxfp8": QuantAlgo(
        "mxfp8", "MXFP8", torch.float8_e4m3fn,
        frozenset({"weight_scale", "input_scale"}), "TensorCoreMXFP8Layout", group_size=32,
    ),
    "int8_tensorwise": QuantAlgo(
        "int8_tensorwise", "INT8", torch.int8,
        frozenset({"weight_scale"}), "TensorWiseINT8Layout", quantize_input=False,
    ),
    "convrot_w4a4": QuantAlgo(
        "convrot_w4a4", "ConvRot W4A4 INT4", torch.int8,
        frozenset({"weight_scale"}), "TensorCoreConvRotW4A4Layout", quantize_input=False,
    ),
}

# Formats ComfyUI 0.33.0 knows that are not ported. A file using one is refused by its real name,
# never read as a neighbouring format that happens to share a storage type.
UNPORTED_FORMATS = {
    "asym_w4a8_int8": "ConvRot W4A8 (asym_w4a8_int8)",
}

# Fields a marker may carry. Everything else is refused: a field ComfyUI adds later may change
# the arithmetic, and ignoring one of those is how a file loads cleanly and draws noise.
#
# The second group is what third-party exporters write beside the format (wikeeyang's FLUX.2 True
# V3 NVFP4/MXFP8/INT8 builds, 2026-09). ComfyUI never reads them; they are accepted here only after
# they are checked to *agree* with the layout — a `group_size` that is not the format's own, an
# `orig_shape` that is not the layer's, or `per_row` over a single scale is refused.
MARKER_FIELDS = frozenset({
    "format", "full_precision_matrix_mult", "convrot", "convrot_groupsize", "linear_dtype", "params",
    "group_size", "orig_shape", "orig_dtype", "per_row",
})
_LAYOUT_GROUP_SIZES = {"nvfp4": 16, "mxfp8": 32, "convrot_w4a4": 64}

CONVROT_DEFAULT_GROUPSIZE = 256
W4A4_QUANT_GROUP_SIZE = 64  # comfy-kitchen's _INT4_GROUP_SIZE; ComfyUI sets it for every layer
INT4_MAX = 7
NVFP4_GROUP_SIZE = 16
MXFP8_GROUP_SIZE = 32


@dataclass(frozen=True)
class LayerQuantConfig:
    """One layer's marker, resolved to the parameters its layout is built with."""

    format: str
    full_precision_mm: bool = False
    convrot: bool = False
    convrot_groupsize: int = CONVROT_DEFAULT_GROUPSIZE
    linear_dtype: str = "int4"
    quant_group_size: int = 0
    # Descriptive exporter fields, kept only to be checked against the stored tensors.
    declared_shape: tuple[int, ...] | None = None
    per_row: bool | None = None

    @property
    def algo(self) -> QuantAlgo:
        return QUANT_ALGOS[self.format]

    @property
    def label(self) -> str:
        if self.format == "int8_tensorwise" and self.convrot:
            return "ConvRot INT8"
        return self.algo.label

    @property
    def rotated(self) -> bool:
        return self.convrot or self.format == "convrot_w4a4"

    def marker(self) -> dict:
        """The JSON ComfyUI writes back for this layer (``_quantized_weight_state_dict``)."""
        conf = {"format": self.format}
        if self.full_precision_mm:
            conf["full_precision_matrix_mult"] = True
        if self.format == "int8_tensorwise" and self.convrot:
            conf["convrot"] = True
            conf["convrot_groupsize"] = self.convrot_groupsize
        elif self.format == "convrot_w4a4":
            conf["convrot_groupsize"] = self.convrot_groupsize
            if self.linear_dtype != "int4":
                conf["linear_dtype"] = self.linear_dtype
        return conf


def decode_marker(marker, layer: str) -> dict:
    """The JSON object a ``comfy_quant`` entry holds, from a uint8 tensor, bytes, str or dict."""
    if isinstance(marker, Mapping):
        return dict(marker)
    try:
        if isinstance(marker, torch.Tensor):
            raw = bytes(marker.to(dtype=torch.uint8).reshape(-1).tolist())
        elif isinstance(marker, (bytes, bytearray)):
            raw = bytes(marker)
        else:
            raw = str(marker).encode("utf-8")
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise UnsupportedQuantization(f"Layer {layer!r} has an unreadable comfy_quant marker") from error
    if not isinstance(value, dict):
        raise UnsupportedQuantization(f"Layer {layer!r} has a comfy_quant marker that is not a JSON object")
    return value


def _marker_int(value, layer: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise UnsupportedQuantization(f"Layer {layer!r} declares {name}={value!r}, which is not an integer")
    return int(value)


def parse_layer_config(marker, layer: str, label: str = "checkpoint") -> LayerQuantConfig:
    """Read one marker the way ``_load_quantized_module`` does, and refuse what it cannot mean.

    The per-format parameters and their defaults are ComfyUI's: ``convrot`` and
    ``convrot_groupsize`` may sit on the marker or inside its ``params``, the group size defaults
    to 256, and W4A4's ``linear_dtype`` defaults to ``int4`` with a fixed quantisation group of
    64. What ComfyUI would silently ignore is refused instead — a rotation flag on a format that
    cannot carry one, a group size no Hadamard matrix has, a field this port does not know.
    """
    conf = decode_marker(marker, layer)
    fmt = conf.get("format")
    if not fmt or not isinstance(fmt, str):
        if conf.get("convrot") or "convrot_groupsize" in conf:
            # ComfyUI core refuses this too ("Unknown quantization format"): a rotation with no
            # format is the ComfyUI-INT8-Fast custom node's layout, not one ComfyUI defines.
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} carries a ConvRot marker with no format — the "
                "ComfyUI-INT8-Fast custom node's layout, which ComfyUI itself does not load; use a "
                "build marked int8_tensorwise + convrot (ComfyUI's INT8 ConvRot) instead"
            )
        raise UnsupportedQuantization(f"{label} layer {layer!r} declares no quantisation format")
    if fmt in UNPORTED_FORMATS:
        raise UnsupportedQuantization(
            f"{label} layer {layer!r} is {UNPORTED_FORMATS[fmt]}, a ComfyUI layout this runtime does "
            "not implement; use the INT8 ConvRot, W4A4, FP8 or BF16 build of this model instead"
        )
    if fmt not in QUANT_ALGOS:
        raise UnsupportedQuantization(
            f"{label} layer {layer!r} is quantised as {fmt!r}, which ComfyUI 0.33.0 does not define "
            "and this runtime cannot read"
        )
    unknown = sorted(set(conf) - MARKER_FIELDS)
    if unknown:
        raise UnsupportedQuantization(
            f"{label} layer {layer!r} ({fmt}) carries marker fields this runtime does not know "
            f"({', '.join(unknown)}); they may change the arithmetic, so the file is refused"
        )
    params = conf.get("params", {})
    if not isinstance(params, dict):
        params = {}  # ComfyUI treats a non-object `params` as absent
    full_precision = conf.get("full_precision_matrix_mult", False)
    if not isinstance(full_precision, bool):
        raise UnsupportedQuantization(f"{label} layer {layer!r} has a non-boolean full_precision_matrix_mult")
    described = _descriptive_fields(conf, fmt, layer, label)

    convrot = conf.get("convrot", params.get("convrot", False))
    if not isinstance(convrot, bool):
        raise UnsupportedQuantization(f"{label} layer {layer!r} has a non-boolean convrot flag")
    has_groupsize = "convrot_groupsize" in conf or "convrot_groupsize" in params
    groupsize = _marker_int(
        conf.get("convrot_groupsize", params.get("convrot_groupsize", CONVROT_DEFAULT_GROUPSIZE)),
        layer, "convrot_groupsize",
    )

    if fmt == "int8_tensorwise":
        if convrot:
            _require_hadamard_size(groupsize, layer, label)
        elif has_groupsize:
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} names a ConvRot group size without the convrot flag"
            )
        return LayerQuantConfig(
            fmt, full_precision, convrot, groupsize if convrot else CONVROT_DEFAULT_GROUPSIZE, **described
        )
    if fmt == "convrot_w4a4":
        _require_hadamard_size(groupsize, layer, label)
        linear_dtype = conf.get("linear_dtype", params.get("linear_dtype", "int4"))
        if linear_dtype not in ("int4", "int8"):
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} declares ConvRot W4A4 linear_dtype {linear_dtype!r}; "
                "only 'int4' and 'int8' exist"
            )
        return LayerQuantConfig(
            fmt, full_precision, False, groupsize, linear_dtype, W4A4_QUANT_GROUP_SIZE, **described
        )
    if convrot or has_groupsize or "linear_dtype" in conf:
        raise UnsupportedQuantization(
            f"{label} layer {layer!r} is {fmt} with a ConvRot field, which that layout cannot carry; "
            "ComfyUI would ignore it and multiply a rotated weight by an unrotated input"
        )
    return LayerQuantConfig(fmt, full_precision, **described)


def _descriptive_fields(conf: Mapping, fmt: str, layer: str, label: str) -> dict:
    """Check the exporter-written fields ComfyUI ignores, and keep what the tensors must agree with."""
    if "group_size" in conf:
        group = conf["group_size"]
        expected = _LAYOUT_GROUP_SIZES.get(fmt)
        if expected is None or isinstance(group, bool) or group != expected:
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} declares group_size={group!r}, which is not the {fmt} layout's "
                f"{expected if expected is not None else 'absence of a'} group"
            )
    if "orig_dtype" in conf and not isinstance(conf["orig_dtype"], str):
        raise UnsupportedQuantization(f"{label} layer {layer!r} has a non-string orig_dtype")
    described = {}
    if "orig_shape" in conf:
        shape = conf["orig_shape"]
        if (not isinstance(shape, list) or len(shape) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape)):
            raise UnsupportedQuantization(f"{label} layer {layer!r} declares an unusable orig_shape {shape!r}")
        described["declared_shape"] = tuple(shape)
    if "per_row" in conf:
        if fmt != "int8_tensorwise" or not isinstance(conf["per_row"], bool):
            raise UnsupportedQuantization(f"{label} layer {layer!r} declares per_row on {fmt}")
        described["per_row"] = conf["per_row"]
    return described


def _require_hadamard_size(size: int, layer: str, label: str) -> None:
    if size < 4 or size & (size - 1) or round(math.log(size, 4)) != math.log(size, 4):
        raise UnsupportedQuantization(
            f"{label} layer {layer!r} declares ConvRot group size {size}; a regular Hadamard "
            "rotation exists only for powers of 4"
        )


def config_for_legacy_fp8(dtype: torch.dtype) -> LayerQuantConfig:
    """The config an unmarked fp8 weight beside a scale is read as."""
    return LayerQuantConfig("float8_e4m3fn" if dtype == torch.float8_e4m3fn else "float8_e5m2")


def encode_marker(conf: Mapping) -> torch.Tensor:
    return torch.tensor(list(json.dumps(dict(conf)).encode("utf-8")), dtype=torch.uint8)


# -- header metadata (comfy/utils.py::convert_old_quants) -------------------------------------


def read_safetensors_metadata(path) -> dict:
    """The ``__metadata__`` table of a safetensors file, without touching any tensor data."""
    with open(path, "rb") as handle:
        head = handle.read(8)
        if len(head) != 8:
            raise ValueError(f"{Path(path).name} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", head)
        header = json.loads(handle.read(length))
    metadata = header.get("__metadata__") or {}
    return metadata if isinstance(metadata, dict) else {}


def header_quant_layers(metadata: Mapping, label: str = "checkpoint") -> dict[str, dict]:
    """The per-layer markers a file declares in its header instead of as tensors."""
    raw = metadata.get(HEADER_QUANT_METADATA_KEY) if metadata else None
    if raw is None:
        return {}
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as error:
        raise UnsupportedQuantization(f"{label} has an unreadable {HEADER_QUANT_METADATA_KEY} header") from error
    layers = value.get("layers") if isinstance(value, dict) else None
    if not isinstance(layers, dict):
        raise UnsupportedQuantization(f"{label} has a {HEADER_QUANT_METADATA_KEY} header with no layer table")
    return {str(layer): conf for layer, conf in layers.items()}


def apply_header_quant_metadata(state_dict, metadata: Mapping, label: str = "checkpoint"):
    """Turn a header's layer table into ``<layer>.comfy_quant`` entries, as ComfyUI does at load.

    ComfyUI's newer exports (Krea 2's MXFP8 and NVFP4 among them) keep the table in the header and
    no marker tensor at all, so reading only the tensors finds an unmarked packed ``uint8`` weight
    and nothing that says what it is. A header entry replaces a tensor marker of the same name,
    which is ComfyUI's order too.
    """
    layers = header_quant_layers(metadata, label)
    if not layers:
        return state_dict
    converted = dict(state_dict)
    for layer, conf in layers.items():
        if not isinstance(conf, dict):
            raise UnsupportedQuantization(f"{label} header declares a non-object marker for {layer!r}")
        converted[f"{layer}{COMFY_QUANT_SUFFIX}"] = encode_marker(conf)
    return converted


def load_safetensors_state_dict(path, load_file, label: str = "checkpoint"):
    """``load_file`` plus the header markers it drops, so both marker spellings reach the loader."""
    state_dict = load_file(str(path), device="cpu")
    try:
        metadata = read_safetensors_metadata(path)
    except (OSError, ValueError, struct.error):
        # `load_file` has just parsed this header, so a real file cannot land here; only a
        # stand-in that is not a safetensors file at all can, and it has no header markers.
        return state_dict
    return apply_header_quant_metadata(state_dict, metadata, label)


# -- rotation and integer kernels (comfy-kitchen int8_utils / eager quantization) -------------

_HADAMARD_CACHE: dict = {}


def build_hadamard(size: int, device="cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The normalised regular Hadamard matrix ConvRot rotates by: Kronecker powers of one 4x4."""
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if size < 4 or size & (size - 1) or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h
    return h


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """``W @ H_block^T`` over each group of input columns — its own inverse, since H is symmetric
    and orthogonal, so the same call rotates a weight in and a dequantised one back out."""
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    return torch.matmul(grouped, h.T.to(dtype=weight.dtype, device=weight.device)).reshape(out_features, in_features)


def rotate_activation(x: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    features = x.shape[-1]
    if features % group_size:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    grouped = x.reshape(-1, features // group_size, group_size)
    return torch.matmul(grouped, h.to(dtype=x.dtype, device=x.device)).reshape(x.shape)


def _scale_for_math(scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=x.device, dtype=x.dtype)
    return torch.where(scale == 0, torch.full_like(scale, torch.finfo(x.dtype).tiny), scale)


def quantize_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax INT8, how every INT8 layer quantises its activation."""
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    q = (x / _scale_for_math(scale, x)).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale


def _is_turing(device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (7, 5)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def int8_matmul_accumulate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a[M,K] @ b[K,N]`` in int8 with exact int32 accumulators, padded to cuBLASLt's shapes."""
    rows, columns, depth = a.size(0), b.size(1), a.size(1)
    if rows == 0 or depth == 0:
        return torch._int_mm(a, b)
    padded_rows = _round_up(max(rows, 32), 32) if a.is_cuda else rows
    if padded_rows != rows:
        a = torch.cat((a, a.new_zeros((padded_rows - rows, depth))), dim=0)
    padded_depth = _round_up(depth, 8)
    if padded_depth != depth:
        a = torch.cat((a, a.new_zeros((a.size(0), padded_depth - depth))), dim=1)
        b = torch.cat((b, b.new_zeros((padded_depth - depth, b.size(1)))), dim=0)
    padded_columns = _round_up(columns, 32 if a.is_cuda and _is_turing(a.device) else 8)
    if padded_columns != columns:
        b = torch.cat((b, b.new_zeros((b.size(0), padded_columns - columns))), dim=1)
    result = torch._int_mm(a, b)
    return result[:rows, :columns] if result.shape != (rows, columns) else result


def _scale_accumulators(acc: torch.Tensor, row_scale: torch.Tensor, column_scale: torch.Tensor,
                        bias, out_dtype: torch.dtype) -> torch.Tensor:
    """``acc * row_scale * column_scale + bias`` in float32, chunked the way comfy-kitchen does
    so a long sequence never materialises one full float32 copy of the output."""
    rows, columns = acc.shape
    chunk = max(1, min(rows, 256 * 1024 * 1024 // max(1, columns * 4)))
    column_scale = column_scale.to(device=acc.device, dtype=torch.float32).reshape(1, -1)
    row_scale = row_scale.to(device=acc.device, dtype=torch.float32).reshape(-1, 1)
    parts = []
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        parts.append((acc[start:end].float() * (row_scale[start:end] * column_scale)).to(out_dtype))
    out = torch.cat(parts, dim=0)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype).reshape(1, -1)
    return out


_BACKEND = threading.local()


@contextlib.contextmanager
def backend_override(name: str):
    """Run the two integer matmuls on one backend (``port``) for the enclosed calls only."""
    previous = getattr(_BACKEND, "name", None)
    _BACKEND.name = name
    try:
        yield
    finally:
        _BACKEND.name = previous


def _triton_module():
    try:
        from . import comfy_quant_triton as kernels  # noqa: PLC0415 - optional, loaded on first use
    except ImportError:
        import comfy_quant_triton as kernels  # noqa: PLC0415
    return kernels


def triton_kernels(device):
    """The built-in Triton kernels when they run (and reproduce the port) on ``device``, else None."""
    kernels = _triton_module()
    return kernels if kernels.available(device) else None


def _run_accelerated(name, ops, x, call):
    """Run ``call(ops)``; a Triton failure other than running out of memory switches the kernels off
    and returns None, so the caller falls through to the port instead of failing the generation."""
    if name != "triton":
        return call(ops)
    try:
        return call(ops)
    except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
        raise
    except Exception as error:  # noqa: BLE001 - any kernel failure hands the layer back to the port
        ops.disable(x.device, f"{type(error).__name__}: {error}")
        return None


def _accelerated(x: torch.Tensor):
    """``(name, ops)`` for the fastest usable backend of the two integer matmuls, or None."""
    if not x.is_cuda or getattr(_BACKEND, "name", None) == "port":
        return None
    kitchen = kitchen_ops()
    if kitchen is not None:
        return "kitchen", kitchen
    kernels = triton_kernels(x.device)
    if kernels is not None:
        return "triton", kernels
    return None


def _decoded_on_triton(qdata: torch.Tensor, dtype: torch.dtype, call):
    """``call(kernels)`` when the built-in NVFP4/MXFP8 decoders reproduce the port on this device,
    else None. A failure other than running out of memory switches the decoders off (the integer
    kernels are unaffected) and returns None, so the port decodes the layer instead."""
    if not qdata.is_cuda or getattr(_BACKEND, "name", None) == "port":
        return None
    kernels = _triton_module()
    if dtype not in kernels.DECODE_DTYPES or not kernels.decode_available(qdata.device):
        return None
    try:
        return call(kernels)
    except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError):
        raise
    except Exception as error:  # noqa: BLE001 - any decoder failure hands the layer back to the port
        kernels.disable_decode(qdata.device, f"{type(error).__name__}: {error}")
        return None


def int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias=None,
                out_dtype: torch.dtype | None = None, convrot: bool = False,
                convrot_groupsize: int = CONVROT_DEFAULT_GROUPSIZE) -> torch.Tensor:
    """ComfyUI's INT8 linear: rotate the activation (ConvRot), quantise it per row, int8 matmul.

    ``weight`` is ``[N, K]`` int8 and ``weight_scale`` one value or one per output row. On CUDA the
    work goes to comfy-kitchen when the pinned version is installed, else to the built-in Triton
    kernels (:mod:`comfy_quant_triton`, bit-identical to the code below), else stays here.
    """
    out_dtype = out_dtype or x.dtype
    backend = _accelerated(x)
    if backend is not None:
        result = _run_accelerated(*backend, x, lambda ops: ops.int8_linear(
            x, weight, weight_scale, bias, out_dtype, convrot=convrot, convrot_groupsize=convrot_groupsize,
        ))
        if result is not None:
            return result
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}")
    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {weight_scale.numel()} values "
            f"for weight shape {tuple(weight.shape)}"
        )
    if convrot:
        x = rotate_activation(x, build_hadamard(convrot_groupsize, x.device, x.dtype), convrot_groupsize)
    shape = x.shape
    x_int8, x_scale = quantize_int8_rowwise(x.reshape(-1, shape[-1]))
    acc = int8_matmul_accumulate(x_int8, weight.T.contiguous())
    out = _scale_accumulators(acc, x_scale, weight_scale, bias, out_dtype)
    return out.reshape(*shape[:-1], weight.shape[0])


def dequantize_int8(qdata: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype,
                    convrot: bool = False, convrot_groupsize: int = CONVROT_DEFAULT_GROUPSIZE) -> torch.Tensor:
    """``TensorWiseINT8Layout.dequantize``: scale in float32, then rotate back if ConvRot."""
    values = qdata.float() * scale.to(device=qdata.device, dtype=torch.float32)
    if convrot:
        values = rotate_weight(values, build_hadamard(convrot_groupsize, qdata.device, torch.float32), convrot_groupsize)
    return values.to(dtype)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """``(..., K)`` int4 values into ``(..., K/2)`` int8, the even column in the low nibble."""
    if values.shape[-1] % 2:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    low = values[..., 0::2].to(torch.int32) & 0x0F
    high = values[..., 1::2].to(torch.int32) & 0x0F
    return (low | (high << 4)).to(torch.int8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`pack_int4`, reading each nibble as signed.

    Arithmetic shifts on int8 sign-extend, so ``(b << 4) >> 4`` is the low nibble and ``b >> 4``
    the high one — the same values comfy-kitchen's masked int32 decode produces, without the
    int32 intermediates that made it the slowest step of the W4A4 matmul.
    """
    packed = packed.view(torch.int8)
    low = torch.bitwise_right_shift(torch.bitwise_left_shift(packed, 4), 4)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)


def _int4_values_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / INT4_MAX
    q = (x / scale).round_().clamp_(-INT4_MAX, INT4_MAX).to(torch.int8)
    return q, scale.reshape(x.shape[0]).to(torch.float32)


def quantize_int4_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT4 in [-7, 7], packed; scales are float32 ``[rows]``."""
    q, scale = _int4_values_rowwise(x)
    return pack_int4(q), scale


def w4a4_activation_bits(device, linear_dtype: str, rows: int | None = None) -> int:
    """4 where the pinned CUDA backend runs a native int4 MMA (SM 8.x, or Turing's CUTLASS
    kernel once M exceeds its GEMV cut-off); 8 where it quantises the activation to INT8 instead
    (Hopper, Blackwell, ``linear_dtype='int8'``). The eager reference used off-GPU is int4."""
    if linear_dtype == "int8":
        return 8
    device = torch.device(device) if device is not None else None
    if device is None or device.type != "cuda":
        return 4
    major, minor = torch.cuda.get_device_capability(device)
    if major == 8:
        return 4
    if (major, minor) == (7, 5) and (rows is None or rows > 8):
        return 4
    return 8


def convrot_w4a4_linear(x: torch.Tensor, qweight: torch.Tensor, wscales: torch.Tensor, bias=None,
                        convrot_groupsize: int = CONVROT_DEFAULT_GROUPSIZE,
                        quant_group_size: int = W4A4_QUANT_GROUP_SIZE, linear_dtype: str = "int4") -> torch.Tensor:
    """``x @ W^T + b`` for a ConvRot W4A4 layer: rotate, quantise the activation, integer matmul."""
    if quant_group_size != W4A4_QUANT_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {W4A4_QUANT_GROUP_SIZE}")
    backend = _accelerated(x)
    if backend is not None:
        result = _run_accelerated(*backend, x, lambda ops: ops.convrot_w4a4_linear(
            x, qweight, wscales, bias=bias, convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size, linear_dtype=linear_dtype,
        ))
        if result is not None:
            return result
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1] * 2}")
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    x_rot = rotate_activation(x2d, build_hadamard(convrot_groupsize, x2d.device, x2d.dtype), convrot_groupsize)
    if w4a4_activation_bits(x2d.device, linear_dtype, x2d.shape[0]) == 4:
        # The values quantize_int4_rowwise would pack, taken before packing: the matmul below
        # wants them one per int8 anyway.
        x_int, x_scale = _int4_values_rowwise(x_rot.contiguous())
    else:
        x_int, x_scale = quantize_int8_rowwise(x_rot)
    w_int = unpack_int4(qweight.to(device=x.device))
    acc = int8_matmul_accumulate(x_int.contiguous(), w_int.T.contiguous())
    out = _scale_accumulators(acc, x_scale, wscales, bias, x.dtype)
    return out.reshape(*shape[:-1], qweight.shape[0])


def dequantize_convrot_w4a4(qdata: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype,
                            convrot_groupsize: int = CONVROT_DEFAULT_GROUPSIZE) -> torch.Tensor:
    values = unpack_int4(qdata).to(torch.float32) * scales.to(device=qdata.device, dtype=torch.float32).reshape(-1, 1)
    h = build_hadamard(convrot_groupsize, qdata.device, torch.float32)
    return rotate_weight(values, h, convrot_groupsize).to(dtype)


# -- block-scaled float formats (comfy-kitchen float_utils / eager quantization) --------------

FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def swizzled_scale_shape(rows: int, blocks: int) -> tuple[int, int]:
    """Where cuBLAS keeps one scale per block: 128-row by 4-block tiles, padded out to whole tiles."""
    return _ceil_div(rows, 128) * 128, _ceil_div(blocks, 4) * 4


def to_blocked(matrix: torch.Tensor) -> torch.Tensor:
    """Lay a ``[rows, blocks]`` scale matrix out in cuBLAS's tiled order (how files store it)."""
    rows, cols = matrix.shape
    padded_rows, padded_cols = swizzled_scale_shape(rows, cols)
    padded = matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=matrix.device, dtype=matrix.dtype)
        padded[:rows, :cols] = matrix
    blocks = padded.view(padded_rows // 128, 128, padded_cols // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(padded_rows, padded_cols)


def _require_blocked_size(blocked: torch.Tensor, rows: int, cols: int) -> tuple[int, int]:
    padded_rows, padded_cols = swizzled_scale_shape(rows, cols)
    if blocked.numel() != padded_rows * padded_cols:
        raise UnsupportedQuantization(
            f"A block-scale table holds {blocked.numel()} values; a {rows}x{cols} block grid needs "
            f"{padded_rows * padded_cols} in cuBLAS's tiled layout"
        )
    return padded_rows, padded_cols


def from_blocked(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo :func:`to_blocked` and trim the padding back off."""
    padded_rows, padded_cols = _require_blocked_size(blocked, rows, cols)
    step = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    step = step.reshape(padded_rows // 128, padded_cols // 4, 128, 4).permute(0, 2, 1, 3)
    return step.reshape(padded_rows, padded_cols)[:rows, :cols]


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    """An E8M0 byte is a bare biased exponent; 0 decodes to zero, as comfy-kitchen reads it."""
    biased = x.view(torch.uint8).to(torch.int32)
    bits = torch.where(biased == 0, torch.zeros_like(biased), biased << 23)
    return bits.view(torch.float32)


def dequantize_nvfp4(qdata: torch.Tensor, global_scale: torch.Tensor, block_scale: torch.Tensor,
                     dtype: torch.dtype, rows: int, cols: int) -> torch.Tensor:
    """NVFP4 back to ``dtype``: two E2M1 codes per byte, **high nibble first**; one E4M3 scale per
    16 values, stored in cuBLAS's tiled order; one float32 global scale. The scale product and the
    multiply run in ``dtype``, as comfy-kitchen's eager decoder does."""
    storage_rows, storage_cols = qdata.shape
    columns = storage_cols * 2
    if columns % NVFP4_GROUP_SIZE:
        raise UnsupportedQuantization(
            f"An NVFP4 weight stores {columns} values per row, not whole blocks of {NVFP4_GROUP_SIZE}"
        )
    if storage_rows < rows or columns < cols:
        raise UnsupportedQuantization(
            f"An NVFP4 weight stores {storage_rows}x{columns} values, short of the layer's {rows}x{cols}"
        )
    _require_blocked_size(block_scale, storage_rows, columns // NVFP4_GROUP_SIZE)
    decoded = _decoded_on_triton(qdata, dtype, lambda kernels: kernels.dequantize_nvfp4(
        qdata, global_scale, block_scale, dtype, rows, cols,
    ))
    if decoded is not None:
        return decoded
    high, low = qdata >> 4, qdata & 0x0F
    codes = torch.stack([high, low], dim=-1).reshape(storage_rows, columns)
    table = torch.tensor(FP4_E2M1_VALUES, device=qdata.device, dtype=dtype)
    values = table[codes.long()].reshape(storage_rows, -1, NVFP4_GROUP_SIZE)
    if block_scale.dtype == torch.uint8:
        block_scale = block_scale.view(torch.float8_e4m3fn)
    scales = from_blocked(block_scale.to(device=qdata.device), storage_rows, columns // NVFP4_GROUP_SIZE)
    total = global_scale.to(device=qdata.device, dtype=dtype) * scales.to(dtype)
    return (values * total.unsqueeze(-1)).reshape(storage_rows, columns)[:rows, :cols].to(dtype)


def dequantize_mxfp8(qdata: torch.Tensor, block_scale: torch.Tensor, dtype: torch.dtype,
                     rows: int, cols: int) -> torch.Tensor:
    """MXFP8 back to ``dtype``: fp8 values times one power-of-two (E8M0) scale per 32, the scales
    stored in cuBLAS's tiled order."""
    storage_rows, storage_cols = qdata.shape
    if storage_rows < rows or storage_cols < cols:
        raise UnsupportedQuantization(
            f"An MXFP8 weight stores {storage_rows}x{storage_cols} values, short of the layer's {rows}x{cols}"
        )
    if block_scale.dtype.is_floating_point and block_scale.dtype != getattr(torch, "float8_e8m0fnu", None):
        raise UnsupportedQuantization(
            f"An MXFP8 block scale must be E8M0 bytes, not {block_scale.dtype}"
        )
    blocks = storage_cols // MXFP8_GROUP_SIZE
    _require_blocked_size(block_scale, storage_rows, blocks)
    decoded = _decoded_on_triton(qdata, dtype, lambda kernels: kernels.dequantize_mxfp8(
        qdata, block_scale, dtype, rows, cols,
    ))
    if decoded is not None:
        return decoded
    scales = e8m0_to_f32(from_blocked(block_scale.view(torch.uint8).to(device=qdata.device), storage_rows, blocks))
    values = qdata.to(torch.float32).reshape(storage_rows, blocks, MXFP8_GROUP_SIZE) * scales.unsqueeze(-1)
    return values.reshape(storage_rows, storage_cols)[:rows, :cols].to(dtype)


def dequantize_layer(qdata: torch.Tensor, scales: Mapping[str, torch.Tensor], config: LayerQuantConfig,
                     dtype: torch.dtype, rows: int, cols: int) -> torch.Tensor:
    """Any ported layout back to a full-precision ``[rows, cols]`` weight — the value ComfyUI's
    ``QuantizedTensor.dequantize`` produces, which is what its full-precision matmul multiplies."""
    fmt = config.format
    if fmt in ("float8_e4m3fn", "float8_e5m2"):
        return (qdata.to(dtype=dtype) * scales["weight_scale"].to(device=qdata.device, dtype=torch.float32)
                .reshape(())).to(dtype)
    if fmt == "int8_tensorwise":
        return dequantize_int8(qdata, scales["weight_scale"], dtype, config.convrot, config.convrot_groupsize)
    if fmt == "convrot_w4a4":
        return dequantize_convrot_w4a4(qdata, scales["weight_scale"], dtype, config.convrot_groupsize)
    if fmt == "nvfp4":
        return dequantize_nvfp4(qdata, scales["weight_scale_2"], scales["weight_scale"], dtype, rows, cols)
    if fmt == "mxfp8":
        return dequantize_mxfp8(qdata, scales["weight_scale"], dtype, rows, cols)
    raise UnsupportedQuantization(f"No dequantisation for {fmt!r}")


def validate_layer(config: LayerQuantConfig, weight_shape, scale_shapes: Mapping[str, tuple],
                   rows: int | None, cols: int | None, layer: str, label: str) -> tuple[int, int]:
    """Check a layer's stored tensors against its layout and return the logical ``(rows, cols)``.

    ``rows``/``cols`` are the module's own shape when it is known; otherwise the logical shape is
    read off the storage (packed formats store half the columns).
    """
    fmt = config.format
    if len(weight_shape) != 2:
        raise UnsupportedQuantization(f"{label} layer {layer!r} ({config.label}) stores a {len(weight_shape)}-D weight")
    stored_rows, stored_cols = (int(v) for v in weight_shape)
    packed = fmt in ("nvfp4", "convrot_w4a4")
    logical = (rows if rows is not None else stored_rows, cols if cols is not None else stored_cols * (2 if packed else 1))

    def numel(shape):
        total = 1
        for value in shape:
            total *= int(value)
        return total

    def fail(reason):
        raise UnsupportedQuantization(f"{label} layer {layer!r} ({config.label}, {fmt}): {reason}")

    if config.declared_shape is not None:
        declared = tuple(config.declared_shape)
        if rows is not None and cols is not None:
            if declared != (rows, cols):
                fail(f"the marker declares orig_shape {list(declared)} but the layer is {[rows, cols]}")
        elif declared[0] > logical[0] or declared[1] > logical[1]:
            # Before the module is known only the storage is, and a padded layout may exceed the layer.
            fail(f"the marker declares orig_shape {list(declared)}, larger than the stored {list(logical)}")
    if fmt == "int8_tensorwise":
        if numel(scale_shapes["weight_scale"]) not in (1, stored_rows):
            fail(f"weight_scale holds {numel(scale_shapes['weight_scale'])} values; one or one per row ({stored_rows}) expected")
        if config.per_row and numel(scale_shapes["weight_scale"]) != stored_rows:
            fail("the marker declares per_row, but the scale is not one per row")
        if config.convrot and stored_cols % config.convrot_groupsize:
            fail(f"{stored_cols} input features are not a multiple of the ConvRot group size {config.convrot_groupsize}")
    elif fmt == "convrot_w4a4":
        k = stored_cols * 2
        if numel(scale_shapes["weight_scale"]) != stored_rows:
            fail(f"weight_scale holds {numel(scale_shapes['weight_scale'])} values; one per row ({stored_rows}) expected")
        if k % config.convrot_groupsize or k % config.quant_group_size:
            fail(f"{k} input features are not a multiple of group sizes {config.convrot_groupsize}/{config.quant_group_size}")
    elif fmt == "nvfp4":
        if (stored_cols * 2) % NVFP4_GROUP_SIZE:
            fail(f"{stored_cols * 2} values per row are not whole groups of {NVFP4_GROUP_SIZE}")
        expected = swizzled_scale_shape(stored_rows, stored_cols * 2 // NVFP4_GROUP_SIZE)
        if numel(scale_shapes["weight_scale"]) != expected[0] * expected[1]:
            fail(f"weight_scale holds {numel(scale_shapes['weight_scale'])} block scales; the tiled layout needs {expected[0] * expected[1]}")
        if numel(scale_shapes["weight_scale_2"]) != 1:
            fail("weight_scale_2 must be one global value")
    elif fmt == "mxfp8":
        if stored_cols % MXFP8_GROUP_SIZE:
            fail(f"{stored_cols} columns are not whole groups of {MXFP8_GROUP_SIZE}")
        expected = swizzled_scale_shape(stored_rows, stored_cols // MXFP8_GROUP_SIZE)
        if numel(scale_shapes["weight_scale"]) != expected[0] * expected[1]:
            fail(f"weight_scale holds {numel(scale_shapes['weight_scale'])} block scales; the tiled layout needs {expected[0] * expected[1]}")
    else:  # fp8: one per-tensor scale
        if numel(scale_shapes["weight_scale"]) != 1:
            fail(f"a {numel(scale_shapes['weight_scale'])}-element scale where {fmt} expects one per-tensor value")
    if logical[0] > stored_rows or logical[1] > stored_cols * (2 if packed else 1):
        fail(f"stored {stored_rows}x{stored_cols} cannot hold a {logical[0]}x{logical[1]} layer")
    return logical


# -- capability (comfy/model_management.py) and the optional comfy-kitchen backend -------------


def _cuda_capability(device=None):
    if torch.version.cuda is None or not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability(device))
    except (RuntimeError, AssertionError):
        return None


def _torch_version() -> tuple[int, int]:
    parts = torch.__version__.split("+", 1)[0].split(".")
    return int(parts[0]), int(parts[1])


def supports_fp8_compute(device=None) -> bool:
    capability = _cuda_capability(device)
    if capability is None:
        return False
    if capability[0] >= 9:
        return True
    return capability >= (8, 9) and _torch_version() >= (2, 4)


def supports_nvfp4_compute(device=None) -> bool:
    capability = _cuda_capability(device)
    return capability is not None and capability[0] >= 10


def supports_mxfp8_compute(device=None) -> bool:
    capability = _cuda_capability(device)
    return capability is not None and capability[0] >= 10 and _torch_version() >= (2, 10)


@lru_cache(maxsize=1)
def kitchen_status() -> dict:
    """Whether the pinned comfy-kitchen is importable, and whether its CUDA kernels may be used.

    Never installs anything. ComfyUI's own start-up rules apply: the Triton backend stays off,
    and the CUDA backend is disabled under a PyTorch built for CUDA older than 13.
    """
    status = {"available": False, "version": None, "cuda": False, "backends": {}, "reason": ""}
    try:
        import comfy_kitchen as ck  # noqa: PLC0415 - optional, imported on first use only
    except ImportError as error:
        status["reason"] = f"not installed ({error.name or error})"
        return status
    version = getattr(ck, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as package_version  # noqa: PLC0415
            version = package_version("comfy-kitchen")
        except Exception:  # noqa: BLE001 - any metadata failure means "unknown version"
            version = None
    status["version"] = version
    if version != PINNED_COMFY_KITCHEN:
        status["reason"] = f"version {version} is not the pinned {PINNED_COMFY_KITCHEN}"
        return status
    try:
        ck.registry.disable("triton")
        cuda_ok = torch.version.cuda is not None and int(str(torch.version.cuda).split(".")[0]) >= 13
        if not cuda_ok:
            ck.registry.disable("cuda")
        status["backends"] = {str(k): str(v) for k, v in ck.list_backends().items()}
    except Exception as error:  # noqa: BLE001 - a broken optional package must not break loading
        status["reason"] = f"failed to initialise: {error}"
        return status
    status["available"] = True
    status["cuda"] = bool(cuda_ok and torch.cuda.is_available())
    status["reason"] = "" if status["cuda"] else "CUDA backend needs PyTorch built for CUDA 13+"
    return status


def kitchen_ops():
    """The comfy-kitchen module when its CUDA kernels are usable, else None (use the ports)."""
    status = kitchen_status()
    if not (status["available"] and status["cuda"]):
        return None
    import comfy_kitchen as ck  # noqa: PLC0415

    return ck


def execution_info(config: LayerQuantConfig, device=None, lora: bool = False) -> dict:
    """How a layer with this config runs here: a stable ``path`` code for the UI, and a description.

    ``path`` is ``integer`` (the layout's own integer matmul), ``emulated`` (ComfyUI would use a
    native kernel on newer hardware and dequantises here, as it does itself below that hardware) or
    ``dequantized`` (dequantised per layer and multiplied at full precision on any hardware).
    """
    if config.format in ("int8_tensorwise", "convrot_w4a4") and not (config.full_precision_mm or lora):
        kernel = _kernel_label(device)
        rotation = "ConvRot " if config.rotated else ""
        if config.format == "int8_tensorwise":
            weight_bits, activation_bits = 8, 8
        else:
            if device is None:
                device = "cuda" if _cuda_capability() is not None else "cpu"
            weight_bits, activation_bits = 4, w4a4_activation_bits(device, config.linear_dtype)
        return {
            "path": "integer", "kernel": kernel, "weight_bits": weight_bits, "activation_bits": activation_bits,
            "text": f"{rotation}W{weight_bits}A{activation_bits} integer matmul ({kernel})",
        }
    if config.format in ("nvfp4", "mxfp8"):
        decoder = _decoder_label(device)
        if config.format == "nvfp4" and not supports_nvfp4_compute(device):
            return {"path": "emulated", "decoder": decoder,
                    "text": f"emulated: dequantised per layer ({decoder} decoder; NVFP4 kernels need SM 10.0+)"}
        if config.format == "mxfp8" and not supports_mxfp8_compute(device):
            return {"path": "emulated", "decoder": decoder,
                    "text": f"emulated: dequantised per layer ({decoder} decoder; MXFP8 kernels need SM 10.0+ and PyTorch 2.10+)"}
        return {"path": "dequantized", "decoder": decoder,
                "text": f"dequantised per layer ({decoder} decoder), full-precision matmul"}
    return {"path": "dequantized", "text": "dequantised per layer, full-precision matmul"}


def _decoder_label(device=None) -> str:
    """Which code decodes NVFP4/MXFP8 layers on ``device``, without compiling anything to find out."""
    try:
        verdict = _triton_module().decode_peek(device)["available"]
    except Exception:  # noqa: BLE001 - a broken optional module leaves the port in charge
        verdict = False
    return "built-in Triton" if verdict is not False else "ported kernel"


def _kernel_label(device=None) -> str:
    """Which code runs the integer matmuls on ``device``, without compiling anything to find out."""
    if kitchen_ops() is not None:
        return "comfy-kitchen CUDA"
    try:
        verdict = _triton_module().peek(device)["available"]
    except Exception:  # noqa: BLE001 - a broken optional module leaves the port in charge
        verdict = False
    return "built-in Triton" if verdict is not False else "ported kernel"


def execution_path(config: LayerQuantConfig, device=None, lora: bool = False) -> str:
    """How a layer with this config runs here, in the words the model status shows."""
    return execution_info(config, device, lora)["text"]


def quantization_capabilities(device=None) -> dict:
    """The per-format report the health endpoint publishes."""
    capability = _cuda_capability(device)
    kitchen = kitchen_status()
    formats = {}
    for name, algo in QUANT_ALGOS.items():
        configs = [LayerQuantConfig(name)]
        if name == "int8_tensorwise":
            configs.append(LayerQuantConfig(name, convrot=True))
        if name == "convrot_w4a4":
            configs = [LayerQuantConfig(name, quant_group_size=W4A4_QUANT_GROUP_SIZE)]
        for config in configs:
            native = {
                "nvfp4": supports_nvfp4_compute(device),
                "mxfp8": supports_mxfp8_compute(device),
                "float8_e4m3fn": supports_fp8_compute(device),
                "float8_e5m2": supports_fp8_compute(device),
            }.get(name, capability is not None and capability >= (7, 5))
            info = execution_info(config, device)
            formats[config.label] = {
                "format": name,
                "loadable": True,
                "hardware_native": bool(native),
                "execution": info["text"],
                **{key: value for key, value in info.items() if key != "text"},
            }
    try:
        triton_state = _triton_module().peek(device)
        decode_state = _triton_module().decode_peek(device)
    except Exception as error:  # noqa: BLE001
        triton_state = decode_state = {"available": False, "reason": f"{type(error).__name__}: {error}"}
    return {
        "reference": COMFY_REFERENCE,
        "cuda_capability": list(capability) if capability else None,
        "comfy_kitchen": {k: v for k, v in kitchen.items()},
        "triton_kernels": triton_state,
        "triton_decoders": decode_state,
        "formats": formats,
        "refused": dict(UNPORTED_FORMATS),
    }
