"""Quantised storage pieces the MOSS-Image engine needs beyond :mod:`quantized_linear`.

:class:`quantized_linear.QuantizedLinear` already runs every ComfyUI linear layout the project reads
(ConvRot INT8 W8A8, FP8 scaled, ...). Three things MOSS needs are not linears and are kept here,
owned by this engine, so no shared helper changes behaviour for the six existing engines:

* :class:`QuantizedEmbedding` -- Qwen3.5-9B's INT8 ConvRot ``embed_tokens``. ComfyUI's
  ``MixedPrecisionOps.Embedding`` gathers the requested rows first and dequantises only those
  (comfy-kitchen 0.2.37 ``dequantize_int8_embedding``): ``rows * scale[rows]`` in float32, then the
  ConvRot rotation undone on those rows alone, then the compute dtype. Expanding the whole
  248 320 x 4096 table would cost 1.9 GiB of BF16 to look up a few hundred tokens. Token ids are
  never rotated: the rotation lives on the embedding's feature axis.
* :class:`RawFP8Linear` -- an FP8 weight with no scale at all, which ComfyUI runs by casting the
  stored value to the compute dtype per use. There is no scale to invent, so none is.
* The ``comfy_attention.config`` marker the Comfy-Org INT8 DiT carries per block, and the
  capability record each loaded component reports.

FP8 support beyond what :mod:`quantized_linear` already validates is labelled
``compatibility-ready`` (no real MOSS FP8 file exists as of 2026-10-16); MXFP8, NVFP4, INT4 and GGUF
are refused for MOSS components even though other engines read them.
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
    from .comfy_quant import COMFY_QUANT_SUFFIX, LayerQuantConfig, UnsupportedQuantization, parse_layer_config
    from .quantized_linear import QuantizedLinear, _apply_without_cast, _register_kron_terms, add_kron_terms
except ImportError:  # pragma: no cover - flat import when run from backend/
    import comfy_quant
    from comfy_quant import COMFY_QUANT_SUFFIX, LayerQuantConfig, UnsupportedQuantization, parse_layer_config
    from quantized_linear import QuantizedLinear, _apply_without_cast, _register_kron_terms, add_kron_terms

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
FP8_FORMAT_BY_DTYPE = {torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e5m2: "float8_e5m2"}

# Linear layouts a MOSS component may carry. Everything else is refused by name for this engine,
# even where another engine in the project reads it: support elsewhere is not MOSS validation.
MOSS_LINEAR_FORMATS = frozenset({"int8_tensorwise", "float8_e4m3fn", "float8_e5m2"})
MOSS_EMBEDDING_FORMATS = frozenset({"int8_tensorwise", "float8_e4m3fn", "float8_e5m2"})

ATTENTION_CONFIG_SUFFIX = ".comfy_attention.config"
# `comfy/ldm/modules/attention.py::ComfyAttention._load_from_state_dict` at ComfyUI bdc63e7 knows
# exactly these. comfy-kitchen is not installed (and must not be, by the user's rule), so the INT8
# kernel is never available here and the block runs PyTorch SDPA -- which is what ComfyUI itself
# does when the kernel is missing: the preference resolves to no function and the default
# attention runs. The marker carries no tensors; the INT8 *weights* are independent of it.
KNOWN_ATTENTION_METHODS = frozenset({"comfy_kitchen_int8"})

VALIDATION_LEVELS = ("unsupported", "compatibility-ready", "fixture-verified", "real-model-verified")


def decode_json_tensor(value, label: str) -> object:
    try:
        if isinstance(value, torch.Tensor):
            raw = bytes(value.to(dtype=torch.uint8).reshape(-1).tolist())
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:
            raw = str(value).encode("utf-8")
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise UnsupportedQuantization(f"{label} is not readable JSON") from error


def parse_attention_config(value, block: str) -> dict:
    """One ``comfy_attention.config`` marker, validated. Unknown methods are refused, not ignored."""
    decoded = decode_json_tensor(value, f"{block} comfy_attention.config")
    configs = decoded if isinstance(decoded, list) else [decoded]
    if not configs:
        raise UnsupportedQuantization(f"{block} comfy_attention.config is an empty list")
    methods = []
    for config in configs:
        if not isinstance(config, dict) or set(config) != {"attention"}:
            raise UnsupportedQuantization(
                f"{block} comfy_attention.config entry {config!r} has fields this runtime does not know"
            )
        method = config["attention"]
        if method not in KNOWN_ATTENTION_METHODS:
            raise UnsupportedQuantization(
                f"{block} asks for attention method {method!r}, whose semantics this runtime cannot establish"
            )
        methods.append(method)
    return {"methods": methods, "raw": decoded}


def attention_execution(requested_methods) -> dict:
    """Which attention backend a marked block actually runs, and why."""
    if not requested_methods:
        return {"backend": "sdpa", "requested": None, "fallback_reason": None}
    return {
        "backend": "sdpa",
        "requested": list(requested_methods),
        "fallback_reason": "comfy_kitchen_int8 kernel unavailable (comfy-kitchen not installed); "
                           "ComfyUI's ComfyAttention falls back to its default attention the same way",
    }


# -- embedding --------------------------------------------------------------------------------


class QuantizedEmbedding(nn.Module):
    """An embedding table kept in its stored INT8 / FP8 form, dequantised one gathered row at a time."""

    def __init__(self, num_embeddings: int, embedding_dim: int, config: LayerQuantConfig, scale_shape,
                 compute_dtype: torch.dtype = torch.bfloat16, device=None, storage_dtype=None):
        super().__init__()
        if config.format not in MOSS_EMBEDDING_FORMATS:
            raise UnsupportedQuantization(f"embedding layout {config.label} is not supported for MOSS")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.quant_config = config
        self.compute_dtype = compute_dtype
        storage = storage_dtype or comfy_quant.QUANT_ALGOS[config.format].storage_dtype
        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), dtype=storage, device=device), requires_grad=False
        )
        self.register_buffer("scale_weight", torch.empty(tuple(scale_shape), dtype=torch.float32, device=device))

    @property
    def quant_format(self) -> str:
        return self.quant_config.format

    def _apply(self, fn, recurse=True):
        # The same protection QuantizedLinear has: a `.to(dtype=...)` sweep must not expand the
        # table or round the float32 scale.
        weight = self._parameters.pop("weight", None)
        scale = self._buffers.pop("scale_weight", None)
        try:
            module = super()._apply(fn, recurse)
        finally:
            if weight is not None:
                self._parameters["weight"] = nn.Parameter(_apply_without_cast(fn, weight.data), requires_grad=False)
            if scale is not None:
                self._buffers["scale_weight"] = _apply_without_cast(fn, scale)
        return module

    def forward(self, input_ids: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        out_dtype = out_dtype or self.compute_dtype
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("embedding indices must be integer token ids")
        indices = input_ids.to(device=self.weight.device, dtype=torch.long)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.num_embeddings):
            raise IndexError("token id outside the embedding table")
        config = self.quant_config
        scale = self.scale_weight
        if config.format == "int8_tensorwise":
            # comfy-kitchen `dequantize_int8_embedding`: gather, scale per row in float32, un-rotate.
            rows = F.embedding(indices, self.weight).to(torch.float32)
            if scale.dim() >= 2:
                rows = rows * F.embedding(indices, scale).to(torch.float32)
            else:
                rows = rows * scale.to(torch.float32)
            if config.convrot:
                h = comfy_quant.build_hadamard(config.convrot_groupsize, rows.device, torch.float32)
                flat = rows.reshape(-1, rows.shape[-1])
                rows = comfy_quant.rotate_weight(flat, h, config.convrot_groupsize).reshape(rows.shape)
            return rows.to(out_dtype)
        # FP8 (ComfyUI `forward_comfy_cast_weights`): gather in fp8, cast to the target dtype, then
        # multiply by the scale cast to that dtype -- in the target dtype, as ComfyUI does it.
        rows = F.embedding(indices, self.weight).to(out_dtype)
        if scale.numel() != 1:
            raise UnsupportedQuantization("an FP8 embedding needs one per-tensor scale")
        return rows * scale.to(dtype=out_dtype)

    def dequantized_rows(self, indices: torch.Tensor) -> torch.Tensor:
        return self.forward(indices)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, quant={self.quant_config.label}"


def embedding_from_state(layer: str, state: Mapping[str, torch.Tensor], compute_dtype: torch.dtype,
                         label: str) -> QuantizedEmbedding:
    """Build the quantised embedding a marked (or legacy scaled-FP8) table describes; validate first."""
    weight = state[f"{layer}.weight"]
    marker = state.get(f"{layer}{COMFY_QUANT_SUFFIX}")
    if marker is not None:
        config = parse_layer_config(marker, layer, label)
    elif weight.dtype in FP8_DTYPES:
        config = LayerQuantConfig(FP8_FORMAT_BY_DTYPE[weight.dtype])
    else:
        raise UnsupportedQuantization(f"{label} embedding {layer!r} is not quantised")
    if config.format not in MOSS_EMBEDDING_FORMATS:
        raise UnsupportedQuantization(f"{label} embedding {layer!r} uses {config.label}, which MOSS does not read")
    if config.full_precision_mm:
        raise UnsupportedQuantization(f"{label} embedding {layer!r} sets full_precision_matrix_mult, which an embedding cannot honour")
    expected_storage = comfy_quant.QUANT_ALGOS[config.format].storage_dtype
    if weight.dtype != expected_storage or weight.ndim != 2:
        raise UnsupportedQuantization(f"{label} embedding {layer!r} declares {config.label} but stores {weight.dtype} {tuple(weight.shape)}")
    scale = None
    for suffix in (".weight_scale", ".scale_weight"):
        if f"{layer}{suffix}" in state:
            scale = state[f"{layer}{suffix}"]
            break
    if scale is None:
        raise UnsupportedQuantization(f"{label} embedding {layer!r} is {config.label} but carries no weight_scale")
    rows, dim = (int(v) for v in weight.shape)
    if config.format == "int8_tensorwise":
        if scale.numel() not in (1, rows) or (scale.numel() == rows and tuple(scale.shape) not in ((rows, 1), (rows,))):
            raise UnsupportedQuantization(f"{label} embedding {layer!r} has a {tuple(scale.shape)} scale for {rows} rows")
        if config.convrot and dim % config.convrot_groupsize:
            raise UnsupportedQuantization(f"{label} embedding {layer!r} width {dim} is not a multiple of its ConvRot group")
        scale_shape = (rows, 1) if scale.numel() == rows else ()
    else:
        if scale.numel() != 1:
            raise UnsupportedQuantization(f"{label} embedding {layer!r} has a non-scalar FP8 scale; that layout is not defined")
        scale_shape = ()
    module = QuantizedEmbedding(rows, dim, config, scale_shape, compute_dtype=compute_dtype, device="meta")
    return module


def embedding_state_for(layer: str, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors a :class:`QuantizedEmbedding` assigns, under the module's own names."""
    out = {"weight": state[f"{layer}.weight"]}
    for suffix in (".weight_scale", ".scale_weight"):
        if f"{layer}{suffix}" in state:
            scale = state[f"{layer}{suffix}"].to(torch.float32)
            if scale.numel() > 1 and scale.ndim == 1:
                scale = scale.reshape(-1, 1)
            elif scale.numel() == 1:
                scale = scale.reshape(())
            out["scale_weight"] = scale
            break
    return out


def consumed_embedding_keys(layer: str, state: Mapping[str, torch.Tensor]) -> set[str]:
    keys = {f"{layer}.weight"}
    for suffix in (".weight_scale", ".scale_weight", COMFY_QUANT_SUFFIX, ".input_scale", ".scale_input"):
        if f"{layer}{suffix}" in state:
            keys.add(f"{layer}{suffix}")
    return keys


# -- raw FP8 linear -----------------------------------------------------------------------------


class RawFP8Linear(nn.Linear):
    """An FP8-stored linear with no scale: the stored value *is* the weight, cast per use."""

    def __init__(self, in_features: int, out_features: int, bias: bool, storage_dtype: torch.dtype,
                 compute_dtype: torch.dtype = torch.bfloat16, device=None):
        nn.Module.__init__(self)
        if storage_dtype not in FP8_DTYPES:
            raise ValueError("RawFP8Linear stores float8_e4m3fn or float8_e5m2")
        self.in_features = in_features
        self.out_features = out_features
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=storage_dtype, device=device), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=compute_dtype, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @property
    def quant_format(self) -> str:
        return "raw_" + FP8_FORMAT_BY_DTYPE[self.storage_dtype]

    def _apply(self, fn, recurse=True):
        weight = self._parameters.pop("weight", None)
        try:
            module = super()._apply(fn, recurse)
        finally:
            if weight is not None:
                self._parameters["weight"] = nn.Parameter(_apply_without_cast(fn, weight.data), requires_grad=False)
        return module

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.weight.to(dtype=dtype)

    def set_lora_adapter(self, down: torch.Tensor, up: torch.Tensor) -> None:
        if down.shape[0] != up.shape[1] or down.shape[1] != self.in_features or up.shape[0] != self.out_features:
            raise ValueError("LoRA shape does not fit this raw FP8 linear")
        self.register_buffer("lora_down", down)
        self.register_buffer("lora_up", up)

    def set_kron_adapters(self, terms) -> None:
        """LoKr products beside the stored weight, as :meth:`QuantizedLinear.set_kron_adapters`."""
        self.kron_terms = _register_kron_terms(self, terms)

    @property
    def has_lora_adapter(self) -> bool:
        return getattr(self, "lora_down", None) is not None or bool(getattr(self, "kron_terms", ()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, self.weight.to(input.dtype), self.bias)
        if getattr(self, "lora_down", None) is not None:
            out = out + F.linear(F.linear(input, self.lora_down.to(input.dtype)), self.lora_up.to(input.dtype))
        return add_kron_terms(self, out, input)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quant={self.quant_format}"


def raw_fp8_layers(state: Mapping[str, torch.Tensor], quantized_layers, linear_layers) -> list[str]:
    """Linears whose weight is FP8 with neither a marker nor a scale beside it (raw FP8 storage)."""
    found = []
    for layer in linear_layers:
        if layer in quantized_layers:
            continue
        weight = state.get(f"{layer}.weight")
        if weight is not None and weight.dtype in FP8_DTYPES:
            found.append(layer)
    return found


def apply_raw_fp8_linears(module: nn.Module, layers, state: Mapping[str, torch.Tensor], compute_dtype: torch.dtype) -> list[str]:
    replaced = []
    for layer in layers:
        parts = layer.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        original = getattr(parent, parts[-1])
        if not isinstance(original, nn.Linear):
            raise UnsupportedQuantization(f"raw FP8 tensor {layer!r} is not a linear weight")
        weight = state[f"{layer}.weight"]
        if tuple(weight.shape) != (original.out_features, original.in_features):
            raise UnsupportedQuantization(f"raw FP8 layer {layer!r} is {tuple(weight.shape)}, the model expects "
                                          f"{(original.out_features, original.in_features)}")
        setattr(parent, parts[-1], RawFP8Linear(original.in_features, original.out_features, original.bias is not None,
                                                weight.dtype, compute_dtype, device="meta"))
        replaced.append(layer)
    return replaced


def refuse_unsupported_layouts(quantized_layers: Mapping, label: str) -> None:
    for layer, spec in quantized_layers.items():
        if spec.format not in MOSS_LINEAR_FORMATS:
            raise UnsupportedQuantization(
                f"{label} layer {layer!r} uses {spec.layer_config.label}; MOSS reads BF16, ConvRot INT8 and "
                "scaled/raw FP8 only (MXFP8, NVFP4, INT4 and GGUF are separate layouts this engine does not validate)"
            )


# -- capability records ---------------------------------------------------------------------------


@dataclass
class ComponentCapability:
    """What one loaded MOSS component is and how it actually runs (kept in caches, jobs and logs)."""

    role: str
    architecture: str
    variant: str
    storage_dtypes: list[str]
    quantization_layouts: list[str]
    compute_dtype: str
    matmul_backends: list[str]
    attention_backend: str | None = None
    fallback_reasons: list[str] = field(default_factory=list)
    validation_level: str = "fixture-verified"
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "architecture": self.architecture,
            "variant": self.variant,
            "storage_dtypes": sorted(set(self.storage_dtypes)),
            "quantization_layouts": sorted(set(self.quantization_layouts)),
            "compute_dtype": self.compute_dtype,
            "matmul_backends": sorted(set(self.matmul_backends)),
            "attention_backend": self.attention_backend,
            "fallback_reasons": list(self.fallback_reasons),
            "validation_level": self.validation_level,
            "details": dict(self.details),
        }


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def describe_linears(module: nn.Module) -> tuple[list[str], list[str], list[str]]:
    """``(storage dtypes, layouts, matmul backends)`` of every linear in a module."""
    storage, layouts, backends = [], [], []
    for child in module.modules():
        if isinstance(child, QuantizedLinear):
            storage.append(_dtype_name(child.weight.dtype))
            layouts.append(child.quant_config.label)
            info = comfy_quant.execution_info(child.quant_config, lora=child.has_lora_adapter)
            backends.append(info["text"])
        elif isinstance(child, RawFP8Linear):
            storage.append(_dtype_name(child.storage_dtype))
            layouts.append("raw FP8 " + ("E4M3" if child.storage_dtype == torch.float8_e4m3fn else "E5M2"))
            backends.append("raw FP8 cast per layer, full-precision matmul")
        elif isinstance(child, nn.Linear):
            storage.append(_dtype_name(child.weight.dtype))
            layouts.append("unquantised")
            backends.append(f"{_dtype_name(child.weight.dtype)} matmul")
    return storage, layouts, backends


def validation_level_for(layouts) -> str:
    """FP8 is compatibility-ready only until a real MOSS FP8 file is certified."""
    labels = {str(label) for label in layouts}
    if any("FP8" in label for label in labels):
        return "compatibility-ready"
    return "fixture-verified"
