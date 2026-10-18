"""Native MOSS-Image runtime: DiT + Qwen3.5-9B + FLUX.2 VAE from component files.

The seventh engine. It duck-types :class:`krea2_pipeline.Krea2Runtime` so the server's staged load,
group-offload escape hatch and job bookkeeping drive it unchanged, and it owns everything that is
MOSS-specific:

* **Components.** A Comfy-Org single file (``diffusion_models/``, ``text_encoders/``, ``vae/``) or the
  pinned HF shard layout (an ``*.safetensors.index.json`` or the directory holding one). Every
  tensor is accounted for: the model's own keys, the quantisation markers and scales, the per-block
  ``comfy_attention.config`` and ``rope.bands`` (both validated, neither trusted), and for the text
  encoder the versioned discard list. Anything else refuses the file before a weight is built.
* **Precision.** Quantised weights stay stored (ConvRot INT8 runs W8A8, FP8 dequantises per layer,
  the INT8 embedding gathers rows); the VAE is float32, as is everything the work order lists:
  scales, BN statistics, RoPE phases, the timestep sinusoid (float64), ``A_log``/linear-attention
  norm and the recurrence.
* **Text-encoder placement.** ``gpu_resident`` moves the encoder for the encode stage and parks it
  again; ``gpu_streamed`` moves one decoder layer at a time (the embedding gathers on the CPU), which
  is what an 8 GB card needs for a ~7 GiB INT8 encoder; ``cpu`` encodes on the CPU only when chosen.
  ``auto`` picks resident, then streamed, from measured free memory -- never a silent CPU fallback.
* **Sampling.** :mod:`moss_sampling`: ComfyUI's discrete-flow schedule at the request's shift, CFG
  on CONST denoised estimates, negative-branch block skipping inside its sigma window.
* **Refinement.** ``refine_batch`` serves the full-frame Hires redraw and ADetailer regions: the
  float32 VAE encodes the source, the 2x2 pack and BN normalisation mirror the decode, and the redraw
  runs the denoise suffix of the same schedule with the same guidance plan.

Not offered in this round, and refused rather than approximated: image-to-image as a base pass, USDU
tiled redraw (no shared conditioning/sigmas), PAG, CFG-Zero*, standard positive-branch SLG,
text-encoder LoRA, the small-decoder VAE.
"""

from __future__ import annotations

import gc
import json
import math
import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torch import nn

try:
    from . import moss_sampling
    from .comfy_quant import COMFY_QUANT_SUFFIX, apply_header_quant_metadata, read_safetensors_metadata
    from .flux_pipeline import CUDA_OOM_ERRORS, _decoded_tensor_to_image, refuse_prepared_refinement, sample_or_stream
    from .flux2_pipeline import (
        FLUX2_VAE_CONFIG,
        _load_flux2_vae,
        _runtime_dependencies as _flux2_dependencies,
        patchify_flux2_latents,
        unpatchify_flux2_latents,
    )
    from .moss_model import (
        MOSS_DIT_ARCHITECTURE,
        MOSS_EXPECTED_LATENT_CHANNELS,
        MOSS_MAX_EDGE,
        MOSS_PIXEL_ALIGNMENT,
        MOSS_VARIANT_LABELS,
        MossTransformer2DModel,
        canonical_moss_state,
        check_rope_bands,
        convert_moss_lora_state_dict,
        infer_moss_config,
        linear_layer_names,
        lokr_terms_for,
        lora_pair_for,
        variant_fingerprint,
    )
    from .moss_quant import (
        ATTENTION_CONFIG_SUFFIX,
        FP8_DTYPES,
        ComponentCapability,
        RawFP8Linear,
        apply_raw_fp8_linears,
        attention_execution,
        describe_linears,
        parse_attention_config,
        raw_fp8_layers,
        refuse_unsupported_layouts,
        validation_level_for,
    )
    from .moss_text_encoder import (
        DISCARDED_SUBGRAPHS,
        LANGUAGE_PREFIXES,
        MossTokenizer,
        build_text_encoder,
        classify_text_encoder,
        layer_runtime_bytes,
        text_encoder_runtime_bytes,
    )
    from .quantized_linear import (
        QuantizedLinear,
        SAFETENSORS_DTYPE_BYTES,
        apply_quantized_linears,
        combine_lora_adapters,
        logical_shape_view,
        quantized_state_dict,
        read_safetensors_header,
        scan_quantized_layers,
    )
except ImportError:  # pragma: no cover - flat import when run from backend/
    import moss_sampling
    from comfy_quant import COMFY_QUANT_SUFFIX, apply_header_quant_metadata, read_safetensors_metadata
    from flux_pipeline import CUDA_OOM_ERRORS, _decoded_tensor_to_image, refuse_prepared_refinement, sample_or_stream
    from flux2_pipeline import (
        FLUX2_VAE_CONFIG,
        _load_flux2_vae,
        _runtime_dependencies as _flux2_dependencies,
        patchify_flux2_latents,
        unpatchify_flux2_latents,
    )
    from moss_model import (
        MOSS_DIT_ARCHITECTURE,
        MOSS_EXPECTED_LATENT_CHANNELS,
        MOSS_MAX_EDGE,
        MOSS_PIXEL_ALIGNMENT,
        MOSS_VARIANT_LABELS,
        MossTransformer2DModel,
        canonical_moss_state,
        check_rope_bands,
        convert_moss_lora_state_dict,
        infer_moss_config,
        linear_layer_names,
        lokr_terms_for,
        lora_pair_for,
        variant_fingerprint,
    )
    from moss_quant import (
        ATTENTION_CONFIG_SUFFIX,
        FP8_DTYPES,
        ComponentCapability,
        RawFP8Linear,
        apply_raw_fp8_linears,
        attention_execution,
        describe_linears,
        parse_attention_config,
        raw_fp8_layers,
        refuse_unsupported_layouts,
        validation_level_for,
    )
    from moss_text_encoder import (
        DISCARDED_SUBGRAPHS,
        LANGUAGE_PREFIXES,
        MossTokenizer,
        build_text_encoder,
        classify_text_encoder,
        layer_runtime_bytes,
        text_encoder_runtime_bytes,
    )
    from quantized_linear import (
        QuantizedLinear,
        SAFETENSORS_DTYPE_BYTES,
        apply_quantized_linears,
        combine_lora_adapters,
        logical_shape_view,
        quantized_state_dict,
        read_safetensors_header,
        scan_quantized_layers,
    )

MOSS_ENGINE = "MOSS"
MOSS_FAMILY = "moss"
MOSS_GUIDANCE_MODES = frozenset({"none"})
TEXT_ENCODER_PLACEMENTS = ("auto", "gpu_resident", "gpu_streamed", "cpu")
MOSS_FULL_DECODER_WIDTH = 512
MOSS_TILED_DECODE_EDGE = 1536
# A float32 FLUX.2 full-decoder pass, per output pixel. Provisional (twice FLUX.2's bf16 scale) until
# the xiria-39 measurement replaces it; answering "does not fit" only costs a tiled decode.
MOSS_FULL_DECODE_BYTES_PER_PIXEL = 6000
# Encoder activations beyond the weights, per conditioning token (measured figures pending on
# xiria-39; generous: an INT8 MLP's int32 accumulators are L x 12288 x 4 bytes).
TEXT_ENCODE_BASE_WORKSPACE_BYTES = 512 * 1024 ** 2
TEXT_ENCODE_TOKEN_WORKSPACE_BYTES = 160 * 1024
CONDITIONING_CACHE_BYTES = 512 * 1024 ** 2

SHARD_INDEX_SUFFIX = ".safetensors.index.json"


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _module_nbytes(module: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


def _discard_module_storage(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.data = torch.empty(0, dtype=parameter.dtype)
    for name, buffer in list(module.named_buffers()):
        if buffer is not None:
            buffer.data = torch.empty(0, dtype=buffer.dtype)


# -- component files ------------------------------------------------------------------------------


def _shard_index(path: Path) -> Path | None:
    if path.is_dir():
        found = sorted(path.glob(f"*{SHARD_INDEX_SUFFIX}"))
        if len(found) != 1:
            raise ValueError(f"{path} must contain exactly one *{SHARD_INDEX_SUFFIX}")
        return found[0]
    if path.name.endswith(SHARD_INDEX_SUFFIX):
        return path
    return None


def resolve_component_files(path_value) -> tuple[list[Path], Path | None]:
    """The safetensors file(s) one component mount names, validated against its shard index."""
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MOSS component not found: {path}")
    index = _shard_index(path)
    if index is None:
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"MOSS components must be .safetensors files or an HF shard index: {path.name}")
        return [path], None
    mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{index.name} has no weight_map")
    root = index.parent.resolve()
    shards = []
    for name in sorted(set(mapping.values())):
        shard = (root / name).resolve()
        if root not in shard.parents:
            raise ValueError(f"{index.name} names a shard outside its directory: {name}")
        if not shard.is_file():
            raise FileNotFoundError(f"{index.name} names a missing shard: {name}")
        shards.append(shard)
    return shards, index


def load_component_state(path_value, keep: Callable[[str], bool] | None = None, label: str = "MOSS component") -> tuple[dict, dict]:
    """Read only the tensors ``keep`` accepts (all by default); shard contents must match the index."""
    from safetensors import safe_open

    files, index = resolve_component_files(path_value)
    expected = None
    if index is not None:
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        expected = {}
        for key, name in mapping.items():
            expected.setdefault((index.parent / name).resolve(), set()).add(key)
    state, skipped = {}, 0
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if expected is not None and keys != expected.get(file, set()):
                raise ValueError(f"{label}: shard {file.name} does not hold exactly the tensors its index assigns it")
            for key in sorted(keys):
                if key in state:
                    raise ValueError(f"{label}: tensor {key} appears in more than one shard")
                if keep is not None and not keep(key):
                    skipped += 1
                    continue
                state[key] = handle.get_tensor(key)
        try:
            state = apply_header_quant_metadata(state, read_safetensors_metadata(file), label)
        except (OSError, ValueError, struct.error):
            pass
    return state, {"files": [f.name for f in files], "sharded": index is not None, "skipped_tensors": skipped}


def _keep_text_encoder_key(key: str) -> bool:
    return not any(pattern.match(key) for pattern in DISCARDED_SUBGRAPHS)


# -- DiT -----------------------------------------------------------------------------------------


def load_moss_transformer(path_value, dtype: torch.dtype = torch.bfloat16, loras=(), state=None, label: str = "MOSS diffusion model"):
    """Build the DiT strictly; LoRAs fuse into full-precision layers and ride quantised ones."""
    from accelerate import init_empty_weights

    source = {}
    if state is None:
        state, source = load_component_state(path_value, label=label)
    state, layout = canonical_moss_state(state)
    attention_markers = {}
    for key in [key for key in state if key.endswith(ATTENTION_CONFIG_SUFFIX)]:
        block = key[: -len(ATTENTION_CONFIG_SUFFIX)]
        attention_markers[block] = parse_attention_config(state.pop(key), block)["methods"]
    quantized = scan_quantized_layers(state, label)
    refuse_unsupported_layouts(quantized, label)
    config = infer_moss_config(logical_shape_view(state, quantized))
    head_dim = int(config["dim"]) // int(config["num_heads"])
    rope_status = check_rope_bands(state, head_dim)
    state.pop("rope.bands", None)
    for block in attention_markers:
        if not block.startswith("layers.") or not block[len("layers."):].isdigit() or int(block[len("layers."):]) >= int(config["num_layers"]):
            raise ValueError(f"{label}: comfy_attention.config on {block!r}, which is not a DiT block")
    variant, fingerprint = variant_fingerprint(state)

    lora_specs = []
    for index, descriptor in enumerate(loras or ()):
        lora_path, multiplier = (descriptor["path"], descriptor["multiplier"]) if isinstance(descriptor, Mapping) else descriptor
        multiplier = float(multiplier)
        if not math.isfinite(multiplier):
            raise ValueError("LoRA multiplier must be finite")
        if multiplier == 0.0:
            continue
        lora_path = Path(lora_path)
        if lora_path.suffix.lower() != ".safetensors":
            raise ValueError(f"MOSS LoRA must be a .safetensors file: {lora_path.name}")
        from safetensors.torch import load_file

        converted = convert_moss_lora_state_dict(load_file(str(lora_path), device="cpu"), config, f"MOSS LoRA {lora_path.name}")
        lora_specs.append((lora_path, multiplier, converted))

    with init_empty_weights():
        transformer = MossTransformer2DModel(**config)
    transformer.compute_dtype = dtype
    kept, deferred = apply_quantized_linears(transformer, quantized, state, dtype, label)
    if deferred:
        raise ValueError(f"{label} quantises non-linear tensors: {sorted(deferred)[:4]}")
    replaced = set(kept)
    raw = raw_fp8_layers(state, replaced, linear_layer_names(config))
    apply_raw_fp8_linears(transformer, raw, state, dtype)
    state = quantized_state_dict(state, quantized, replaced)
    expected = set(transformer.state_dict())
    unknown = sorted(set(state) - expected)
    if unknown:
        raise ValueError(f"{label} carries tensors the MOSS DiT does not define: {unknown[:4]}")
    protected = {f"{layer}.weight" for layer in replaced | set(raw)} | {f"{layer}.scale_weight" for layer in replaced}
    weights = {}
    for key, value in state.items():
        if key in protected or not value.is_floating_point():
            weights[key] = value
        elif value.dtype in FP8_DTYPES:
            raise ValueError(f"{label} stores {key} in FP8 outside a linear weight")
        else:
            weights[key] = value.to(dtype)
    del state

    # Fuse every LoRA and LoKr delta for a full-precision layer in float32 and round once; quantised
    # layers keep their stored weight and take LoRAs on the rank axis and LoKr products beside it.
    deltas: dict[str, torch.Tensor] = {}
    adapters: dict[str, list] = {}
    krons: dict[str, list] = {}
    report = []
    for lora_path, multiplier, converted in lora_specs:
        fused = adapted = 0
        for target, entries in converted.items():
            module = transformer.get_submodule(target)
            out_features, in_features = module.out_features, module.in_features
            pairs = [entry for entry in entries if entry["kind"] == "lora"]
            terms = [(w1 * multiplier, w2, offset) for w1, w2, offset in
                     lokr_terms_for([entry for entry in entries if entry["kind"] == "lokr"], out_features, in_features)]
            down, up = lora_pair_for(pairs, out_features, in_features) if pairs else (None, None)
            if isinstance(module, (QuantizedLinear, RawFP8Linear)):
                if pairs:
                    adapters.setdefault(target, []).append((down, up, multiplier))
                if terms:
                    krons.setdefault(target, []).extend(terms)
                adapted += 1
            else:
                delta = (up @ down) * multiplier if pairs else torch.zeros((out_features, in_features), dtype=torch.float32)
                for w1, w2, offset in terms:
                    product = torch.kron(w1, w2)
                    delta[offset:offset + product.shape[0]] += product
                deltas[target] = deltas[target] + delta if target in deltas else delta
                fused += 1
        report.append({"name": lora_path.name, "multiplier": multiplier, "patched_modules": fused + adapted,
                       "fused_modules": fused, "adapted_modules": adapted, "skipped_modules": []})
    for target, delta in deltas.items():
        key = f"{target}.weight"
        weights[key] = (weights[key].to(torch.float32) + delta).to(weights[key].dtype)
    incompatible = transformer.load_state_dict(weights, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict {label} load failed: missing={incompatible.missing_keys[:4]}, unexpected={incompatible.unexpected_keys[:4]}")
    del weights
    for target, stack in adapters.items():
        transformer.get_submodule(target).set_lora_adapter(*combine_lora_adapters(stack, dtype))
    for target, terms in krons.items():
        transformer.get_submodule(target).set_kron_adapters([(w1.to(dtype), w2.to(dtype), offset) for w1, w2, offset in terms])
    for name, tensor in list(transformer.named_parameters()) + list(transformer.named_buffers()):
        if tensor.is_meta:
            raise RuntimeError(f"{label} left {name} unloaded")
    methods = sorted({method for values in attention_markers.values() for method in values})
    load_report = {
        "architecture": MOSS_DIT_ARCHITECTURE,
        "variant": variant,
        "variant_label": MOSS_VARIANT_LABELS[variant],
        "variant_fingerprint": fingerprint,
        "layout": layout,
        "source": source,
        "rope_bands": rope_status,
        "quantized_layers": sorted(replaced),
        "raw_fp8_layers": sorted(raw),
        "adapted_layers": sorted(set(adapters) | set(krons)),
        "attention_markers": {"blocks": len(attention_markers), "methods": methods},
        "loras": report,
    }
    return transformer.eval().requires_grad_(False), config, load_report


# -- text encoder ----------------------------------------------------------------------------------


def load_moss_text_encoder(path_value, dtype: torch.dtype = torch.bfloat16, label: str = "MOSS text encoder"):
    state, source = load_component_state(path_value, keep=_keep_text_encoder_key, label=label)
    if classify_text_encoder(state.keys()) != "qwen3_5":
        raise ValueError(f"{label}: the selected file is not a Qwen3.5 text encoder")
    model, config, report = build_text_encoder(state, dtype, label)
    report["source"] = source
    return model, config, report


# -- VAE ----------------------------------------------------------------------------------------


def vae_signature(path_value) -> dict:
    """Which FLUX.2 autoencoder a file is, from its header alone (no tensors read)."""
    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower() != ".safetensors":
        raise ValueError("The MOSS VAE must be a .safetensors file")
    header = read_safetensors_header(path)
    for key in ("bn.running_mean", "bn.running_var"):
        if key not in header:
            raise ValueError(f"The selected VAE is not a FLUX.2 autoencoder: it carries no {key} packing statistic")
    if tuple(header["bn.running_mean"]["shape"]) != (MOSS_EXPECTED_LATENT_CHANNELS,):
        raise ValueError("The selected VAE's packing statistics are not the 128-channel FLUX.2 layout")
    conv_in = header.get("decoder.conv_in.weight")
    if conv_in is None:
        raise ValueError("The selected VAE has no decoder")
    width = int(conv_in["shape"][0])
    latent = int(conv_in["shape"][1])
    if latent != 32:
        raise ValueError(f"The selected VAE decodes {latent}-channel latents; MOSS needs FLUX.2's 32")
    if width != MOSS_FULL_DECODER_WIDTH:
        raise ValueError(
            f"The selected VAE is a {width}-wide FLUX.2 small decoder (the HF MOSS bundle's). This round "
            "validates only the full decoder (flux2-vae.safetensors); the small decoder is a follow-up"
        )
    return {"signature": "flux2_full_decoder", "decoder_width": width, "latent_channels": latent}


def load_moss_vae(path_value):
    signature = vae_signature(path_value)
    deps = _flux2_dependencies()
    vae, statistics = _load_flux2_vae(Path(path_value).expanduser().resolve(), torch.float32, deps)
    return vae, statistics, signature


# -- conditioning -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MossConditioning:
    embeddings: torch.Tensor  # [1, L, 4096] compute dtype, CPU
    diagnostics: dict


class MossRuntime:
    """A loaded MOSS-Image model behind the surface every native engine exposes."""

    def __init__(self, transformer, text_encoder, vae, tokenizer, dtype, config, text_config, statistics,
                 device=None, reports=None):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = vae
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.config = dict(config)
        self.text_config = text_config
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        self._latent_mean = statistics["bn.running_mean"].to(torch.float32).reshape(1, -1, 1, 1)
        self._latent_std = torch.sqrt(
            statistics["bn.running_var"].to(torch.float32) + float(FLUX2_VAE_CONFIG["batch_norm_eps"])
        ).reshape(1, -1, 1, 1)
        self.components = {"transformer": transformer, "text_encoder": text_encoder, "vae": vae}
        self.weight_sizes = {
            "transformer": _module_nbytes(transformer),
            "text_encoder": _module_nbytes(text_encoder),
            "vae": _module_nbytes(vae),
        }
        self.weight_sizes["total"] = sum(self.weight_sizes.values())
        block_bytes = [_module_nbytes(block) for block in transformer.layers]
        self.weight_sizes["transformer_max_block"] = max(block_bytes, default=0)
        self.weight_sizes["transformer_unmatched"] = max(0, self.weight_sizes["transformer"] - sum(block_bytes))
        te_layers = layer_runtime_bytes(text_encoder)
        self.weight_sizes["text_encoder_max_layer"] = max(te_layers, default=0)
        self.weight_sizes["text_encoder_embedding"] = _module_nbytes(text_encoder.embed_tokens)
        self.reports = dict(reports or {})
        self.lora_report = list(self.reports.get("transformer", {}).get("loras", []))
        self.unclaimed_tensors = []
        self.last_generation_metrics = {}
        self.attention_backend = "native"
        self.batch_cfg = False
        self.keep_transformer_resident = False
        self.unload_before_decode = True
        self.noise_device = "cpu"
        self.text_encoder_placement = "auto"
        self.keep_text_encoder_resident = False
        self._text_encoder_on_device = False
        self._vae_tiling_required = False
        self._closed = False
        self._poisoned = False
        self._transformer_group_offload = False
        self._transformer_blocks_per_group = 0
        self._transformer_resident = False
        self._last_sampling_execution = {}
        self._conditioning_cache: dict = {}
        self.last_conditioning_reused = False

    # -- lifecycle --------------------------------------------------------------------------------

    def _require_open(self):
        if self._closed:
            raise RuntimeError("MOSS runtime is closed")
        if self._poisoned:
            raise RuntimeError("MOSS runtime is poisoned by an interrupted transformer restoration")

    @property
    def transformer_group_offload_enabled(self):
        return bool(self._transformer_group_offload)

    @property
    def transformer_resident(self):
        return bool(self._transformer_resident)

    def configure_attention_backend(self, backend="native"):
        """MOSS runs its own SDPA attention; only ``native`` is true to report."""
        self._require_open()
        if backend != "native":
            raise ValueError(f"MOSS attention runs natively; {backend!r} has no dispatch to apply")
        self.attention_backend = backend
        return backend

    def configure_vae_tiling(self, required=False):
        self._vae_tiling_required = bool(required)
        return "tiled" if self._vae_tiling_required else "auto"

    def configure_transformer_residency(self, keep=False):
        self.keep_transformer_resident = bool(keep)
        if not self.keep_transformer_resident and self.transformer_resident:
            self._park_transformer_on_cpu()
            _empty_cuda_cache()

    def configure_decode_residency(self, unload=True):
        self.unload_before_decode = bool(unload)
        return self.unload_before_decode

    def configure_text_encoder_placement(self, placement="auto", keep_resident=False):
        if placement not in TEXT_ENCODER_PLACEMENTS:
            raise ValueError(f"text encoder placement must be one of {TEXT_ENCODER_PLACEMENTS}")
        self.text_encoder_placement = placement
        self.keep_text_encoder_resident = bool(keep_resident) and placement in ("auto", "gpu_resident")
        if not self.keep_text_encoder_resident:
            self._park_text_encoder()
        return placement

    def enable_transformer_group_offload(self, blocks_per_group=1):
        self._require_open()
        if not isinstance(blocks_per_group, int) or isinstance(blocks_per_group, bool) or blocks_per_group < 1:
            raise ValueError("blocks_per_group must be a positive integer")
        if self.transformer_group_offload_enabled:
            if self._transformer_blocks_per_group != blocks_per_group:
                raise RuntimeError("MOSS transformer group offload is already configured with a different group size")
            return
        if any(parameter.device.type != "cpu" for parameter in self.transformer.parameters()):
            raise RuntimeError("MOSS transformer must be on CPU before enabling group offload")
        self.keep_transformer_resident = False
        self._transformer_resident = False
        self.transformer.enable_group_offload(
            onload_device=self.device, offload_device=torch.device("cpu"), offload_type="block_level",
            num_blocks_per_group=blocks_per_group, non_blocking=False, use_stream=False, record_stream=False,
        )
        self._transformer_group_offload = True
        self._transformer_blocks_per_group = blocks_per_group

    def _remove_transformer_group_offload(self):
        if not self.transformer_group_offload_enabled or self.transformer is None:
            return
        groups = {}
        for module in self.transformer.modules():
            registry = getattr(module, "_diffusers_hook", None)
            hook = registry.get_hook("group_offloading") if registry is not None else None
            if hook is not None:
                groups[id(hook.group)] = hook.group
        for group in groups.values():
            group.offload_()
        registry = getattr(self.transformer, "_diffusers_hook", None)
        if registry is not None:
            for name in ("lazy_prefetch_group_offloading", "layer_execution_tracker", "group_offloading"):
                registry.remove_hook(name, recurse=True)
        self._transformer_group_offload = False
        self._transformer_blocks_per_group = 0
        self.transformer.to("cpu")

    def _ensure_transformer_on_device(self):
        if self._transformer_resident:
            return
        self.transformer.to(self.device)
        self._transformer_resident = True

    # The server's shared helpers know this name.
    _ensure_transformer_on_cuda = _ensure_transformer_on_device

    def _park_transformer_on_cpu(self):
        if not self._transformer_resident:
            return
        self.transformer.to("cpu")
        self._transformer_resident = False

    def _park_text_encoder(self):
        if self._text_encoder_on_device and self.text_encoder is not None:
            self.text_encoder.to("cpu")
            self._text_encoder_on_device = False
            _empty_cuda_cache()

    def _release_transformer_for_decode(self):
        if not self.unload_before_decode or self.transformer_group_offload_enabled or not self._transformer_resident:
            return False
        self._park_transformer_on_cpu()
        _empty_cuda_cache()
        return True

    def to_cpu(self):
        if self._closed:
            return
        if not self.transformer_group_offload_enabled:
            self._park_transformer_on_cpu()
        self._park_text_encoder()
        if self.vae is not None:
            self.vae.to("cpu")
        _empty_cuda_cache()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.transformer_group_offload_enabled:
                self._remove_transformer_group_offload()
        except BaseException:
            self._poisoned = True
        self._conditioning_cache.clear()
        for name in ("transformer", "text_encoder", "vae"):
            module = getattr(self, name, None)
            if module is not None:
                _discard_module_storage(module)
                setattr(self, name, None)
        self.components = {}
        _empty_cuda_cache()

    # -- capabilities ------------------------------------------------------------------------------

    def capabilities(self) -> list[dict]:
        transformer_report = self.reports.get("transformer", {})
        text_report = self.reports.get("text_encoder", {})
        storage, layouts, backends = describe_linears(self.transformer)
        attention_methods = transformer_report.get("attention_markers", {}).get("methods", [])
        attention = attention_execution(attention_methods)
        dit = ComponentCapability(
            role="diffusion_model", architecture=MOSS_DIT_ARCHITECTURE,
            variant=transformer_report.get("variant", "unrecognised"), storage_dtypes=storage,
            quantization_layouts=layouts, compute_dtype=str(self.dtype).replace("torch.", ""),
            matmul_backends=backends, attention_backend=attention["backend"],
            fallback_reasons=[attention["fallback_reason"]] if attention["fallback_reason"] else [],
            validation_level=validation_level_for(layouts),
            details={"layout": transformer_report.get("layout"), "fingerprint": transformer_report.get("variant_fingerprint")},
        )
        te_storage, te_layouts, te_backends = describe_linears(self.text_encoder)
        embed = self.text_encoder.embed_tokens
        te = ComponentCapability(
            role="text_encoder", architecture="qwen3_5_text", variant="qwen3.5-9b",
            storage_dtypes=te_storage + [str(embed.weight.dtype).replace("torch.", "")],
            quantization_layouts=te_layouts + ([embed.quant_config.label + " embedding"] if hasattr(embed, "quant_config") else []),
            compute_dtype=str(self.dtype).replace("torch.", ""), matmul_backends=te_backends,
            attention_backend="sdpa", validation_level=validation_level_for(te_layouts),
            details={"placement": self.text_encoder_placement, "discarded": text_report.get("discarded_subgraphs", [])},
        )
        vae = ComponentCapability(
            role="vae", architecture="flux2_autoencoder", variant=self.reports.get("vae", {}).get("signature", "flux2_full_decoder"),
            storage_dtypes=["float32"], quantization_layouts=["unquantised"], compute_dtype="float32",
            matmul_backends=["float32 convolution"], validation_level="fixture-verified",
        )
        return [dit.as_dict(), te.as_dict(), vae.as_dict()]

    # -- prompt encoding ------------------------------------------------------------------------------

    def token_diagnostics(self, prompt: str):
        self._require_open()
        _ids, diagnostics = self.tokenizer.encode(prompt)
        return {"llm": {**diagnostics, "sequence_length": diagnostics["token_count"], "family": "qwen3_5_9b"}}

    def _free_device_bytes(self) -> int | None:
        if self.device.type != "cuda":
            return None
        try:
            free_bytes, _total = torch.cuda.mem_get_info(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
        except (AttributeError, RuntimeError):
            return None
        return int(free_bytes) + max(0, int(reserved) - int(allocated))

    def _resolve_text_placement(self, tokens: int) -> tuple[str, dict]:
        requested = self.text_encoder_placement
        workspace = TEXT_ENCODE_BASE_WORKSPACE_BYTES + tokens * TEXT_ENCODE_TOKEN_WORKSPACE_BYTES
        if self.device.type != "cuda":
            return "cpu", {"requested": requested, "reason": "runtime device is CPU", "workspace_bytes": workspace}
        if requested in ("gpu_resident", "gpu_streamed", "cpu"):
            return requested, {"requested": requested, "reason": "explicit", "workspace_bytes": workspace}
        free = self._free_device_bytes()
        resident_need = self.weight_sizes["text_encoder"] - self.weight_sizes["text_encoder_embedding"] + workspace
        if self._text_encoder_on_device or free is None or free >= int(resident_need * 1.05):
            return "gpu_resident", {"requested": "auto", "free_bytes": free, "required_bytes": resident_need, "workspace_bytes": workspace}
        stream_need = self.weight_sizes["text_encoder_max_layer"] * 2 + workspace
        if free >= stream_need:
            return "gpu_streamed", {"requested": "auto", "free_bytes": free, "required_bytes": stream_need, "workspace_bytes": workspace}
        raise RuntimeError(
            f"MOSS text encoding cannot fit: {free / 1024 ** 3:.2f} GiB free, one streamed layer plus the "
            f"encode workspace needs {stream_need / 1024 ** 3:.2f} GiB. Free GPU memory, shorten the prompt, "
            "or select CPU text encoding explicitly"
        )

    def _encode_uncached(self, prompt: str, checkpoint=None) -> MossConditioning:
        ids, diagnostics = self.tokenizer.encode(prompt)
        placement, decision = self._resolve_text_placement(len(ids))
        tensor = torch.tensor([ids], dtype=torch.long)
        encoder = self.text_encoder
        device = self.device if placement != "cpu" else torch.device("cpu")
        hook = None
        if placement == "gpu_resident":
            if not self._text_encoder_on_device:
                # Device-only moves: a dtype argument would round the float32 A_log and norms.
                for name, child in encoder.named_children():
                    if name != "embed_tokens":
                        child.to(device)
                self._text_encoder_on_device = True
        elif placement == "gpu_streamed":
            self._park_text_encoder()

            def hook(event, index, module):
                if event == "enter":
                    if checkpoint is not None:
                        checkpoint()
                    module.to(device)
                else:
                    module.to("cpu")
        try:
            with torch.inference_mode():
                hidden = encoder(tensor, device=device, layer_hook=hook)
        finally:
            if placement == "gpu_streamed":
                for layer in encoder.layers:
                    layer.to("cpu")
                encoder.norm.to("cpu")
            elif placement == "gpu_resident" and not self.keep_text_encoder_resident:
                self._park_text_encoder()
        embeddings = hidden.to(device="cpu", dtype=self.dtype).contiguous()
        diagnostics = {**diagnostics, "placement": placement, "placement_decision": decision}
        return MossConditioning(embeddings=embeddings, diagnostics=diagnostics)

    def _encode(self, prompt: str, checkpoint=None) -> MossConditioning:
        if prompt in self._conditioning_cache:
            self._conditioning_cache[prompt] = self._conditioning_cache.pop(prompt)
            self.last_conditioning_reused = True
            return self._conditioning_cache[prompt]
        conditioning = self._encode_uncached(prompt, checkpoint)
        self.last_conditioning_reused = False
        self._conditioning_cache[prompt] = conditioning
        while len(self._conditioning_cache) > 1 and sum(
            c.embeddings.numel() * c.embeddings.element_size() for c in self._conditioning_cache.values()
        ) > CONDITIONING_CACHE_BYTES:
            del self._conditioning_cache[next(iter(self._conditioning_cache))]
        return conditioning

    def _encode_stage(self, name: str, prompt: str, checkpoint=None) -> MossConditioning:
        conditioning = self._run_cuda_stage(name, lambda: self._encode(prompt, checkpoint))
        metric = self.last_generation_metrics.setdefault(name, {})
        metric.update({
            "conditioning_reused": self.last_conditioning_reused,
            "tokens": conditioning.diagnostics["token_count"],
            "truncated": conditioning.diagnostics["truncated"],
            "placement": conditioning.diagnostics["placement"],
        })
        return conditioning

    def _conditioning(self, prompt: str, negative_prompt: str, plan, checkpoint=None, stage: str = ""):
        positive = self._encode_stage(f"{stage}prompt_encode", prompt, checkpoint)
        negative = None
        if plan.runs_unconditional:
            negative = self._encode_stage(f"{stage}negative_prompt_encode", negative_prompt, checkpoint)
        if not self.keep_text_encoder_resident:
            self._park_text_encoder()
        return positive, negative

    # -- latents ----------------------------------------------------------------------------------

    def _initial_latents(self, generators, height: int, width: int, sigma) -> torch.Tensor:
        shape = (1, int(self.config["latent_channels"]), height // MOSS_PIXEL_ALIGNMENT, width // MOSS_PIXEL_ALIGNMENT)
        noise = torch.cat([torch.randn(shape, generator=g, dtype=torch.float32, device="cpu") for g in generators])
        return noise * float(torch.as_tensor(sigma, dtype=torch.float32).item())

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self._latent_mean.to(device=latents.device, dtype=torch.float32)
        std = self._latent_std.to(device=latents.device, dtype=torch.float32)
        return (latents.float() - mean) / std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self._latent_mean.to(device=latents.device, dtype=torch.float32)
        std = self._latent_std.to(device=latents.device, dtype=torch.float32)
        return latents.float() * std + mean

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """RGB images to packed, BN-normalised 128-channel latents (the inverse of ``_decode``)."""
        device = self.device
        width, height = images[0].size
        # The encoder's full-resolution working set is the decoder's mirror, so the decode rule decides.
        tiled = self._resolved_tiled_decode(height, width)
        pixels = []
        for image in images:
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(image.height, image.width, 3)
            pixels.append(data.permute(2, 0, 1))
        try:
            self.vae.to(device)
            self._configure_vae_tiling(tiled)
            vae_input = torch.stack(pixels).to(device=device, dtype=torch.float32) / 127.5 - 1.0
            with torch.inference_mode():
                encoded = self.vae.encode(vae_input)
                posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
                latents = self._normalize_latents(patchify_flux2_latents(posterior.mode().float().contiguous()))
                latents = latents.to(device="cpu", dtype=torch.float32)
            self.last_generation_metrics.setdefault("refinement.vae_encode", {}).update({
                "actual_vae_mode": "tiled" if tiled else "full",
                "vae_dtype": "float32",
            })
            return latents
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    def _full_decode_fits(self, height: int, width: int) -> bool:
        free = self._free_device_bytes()
        if free is None:
            return True
        return free >= int(height) * int(width) * MOSS_FULL_DECODE_BYTES_PER_PIXEL

    def _resolved_tiled_decode(self, height: int, width: int) -> bool:
        if self._vae_tiling_required or max(height, width) > MOSS_TILED_DECODE_EDGE:
            return True
        # A full frame only while it fits beside what is resident; evicting the DiT to buy room
        # costs more than the tiled pass (the Krea 2 measurement), so the choice is between decodes.
        return not self._full_decode_fits(height, width)

    def _configure_vae_tiling(self, tiled: bool) -> None:
        if not tiled:
            self.vae.disable_tiling()
            return
        self.vae.tile_sample_min_size = 512
        self.vae.tile_latent_min_size = 512 // 8
        self.vae.tile_overlap_factor = 64 / 512
        self.vae.enable_tiling()

    def _decode(self, latents: torch.Tensor, force_tiled_decode: bool = False, metrics_key: str = "vae_decode") -> list[Image.Image]:
        device = self.device
        height = latents.shape[2] * MOSS_PIXEL_ALIGNMENT
        width = latents.shape[3] * MOSS_PIXEL_ALIGNMENT
        tiled = bool(force_tiled_decode) or self._resolved_tiled_decode(height, width)
        evicted = self._release_transformer_for_decode() if not tiled else False
        try:
            self.vae.to(device)
            self._configure_vae_tiling(tiled)
            unpacked = unpatchify_flux2_latents(self._denormalize_latents(latents.to(device=device, dtype=torch.float32)))
            with torch.inference_mode():
                decoded = self.vae.decode(unpacked.to(torch.float32), return_dict=False)[0]
                images = [_decoded_tensor_to_image(frame.float()) for frame in decoded]
            self.last_generation_metrics.setdefault(metrics_key, {}).update({
                "actual_vae_mode": "tiled" if tiled else "full",
                "vae_dtype": "float32",
                "transformer_evicted_for_decode": evicted,
            })
            return images
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    # -- sampling ----------------------------------------------------------------------------------

    def _run_cuda_stage(self, name: str, operation):
        measured = False
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
                measured = True
            except Exception:
                pass
        self.last_generation_metrics.pop(name, None)
        started = time.perf_counter()
        try:
            return operation()
        finally:
            if measured:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            metric = {"seconds": round(time.perf_counter() - started, 3)}
            if measured:
                try:
                    metric.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                except Exception:
                    pass
            recorded = self.last_generation_metrics.get(name)
            self.last_generation_metrics[name] = {**recorded, **metric} if isinstance(recorded, dict) else metric

    @staticmethod
    def _validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step):
        if sampler not in moss_sampling.MOSS_SAMPLERS:
            raise ValueError(f"Unsupported MOSS sampler: {sampler}")
        if scheduler not in moss_sampling.MOSS_SCHEDULERS:
            raise ValueError(f"Unsupported MOSS scheduler: {scheduler}")
        if guidance not in MOSS_GUIDANCE_MODES:
            raise ValueError("MOSS supports plain CFG only (guidance 'none'); PAG and CFG-Zero* are not available")
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % MOSS_PIXEL_ALIGNMENT:
                raise ValueError(f"{name} must be a positive multiple of {MOSS_PIXEL_ALIGNMENT}")
            if value > MOSS_MAX_EDGE:
                raise ValueError(f"{name} cannot exceed the MOSS maximum of {MOSS_MAX_EDGE}")
        moss_sampling.validate_steps(steps)
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(float(cfg)):
            raise ValueError("cfg must be finite")
        if not isinstance(generators, Sequence) or isinstance(generators, (str, bytes)) or len(generators) < 1:
            raise ValueError("generators must contain at least one CPU torch.Generator")
        for index, generator in enumerate(generators):
            if not isinstance(generator, torch.Generator) or torch.device(generator.device).type != "cpu":
                raise ValueError(f"generators[{index}] must be a CPU torch.Generator")
        if on_step is not None and not callable(on_step):
            raise ValueError("on_step must be callable")

    def _sample(self, positive: MossConditioning, negative: MossConditioning | None, latents: torch.Tensor,
                sigmas: torch.Tensor, sampler: str, plan, generators, on_step, on_step_checkpoint,
                source_latents=None, source_noise=None, latent_mask=None):
        device = self.device
        batch = latents.shape[0]
        latents = latents.to(device=device, dtype=torch.float32)
        if plan.runs_unconditional and negative is None:
            raise ValueError("MOSS classifier-free guidance requires the negative conditioning")
        blend = None
        if latent_mask is not None:
            if source_latents is None or source_noise is None:
                raise ValueError("A latent mask requires the source latents it composites against")
            source = source_latents.to(device=device, dtype=torch.float32)
            source_noise = source_noise.to(device=device, dtype=torch.float32)
            mask = torch.nn.functional.interpolate(
                latent_mask.to(device=device, dtype=torch.float32).view(batch, 1, *latent_mask.shape[-2:]),
                size=tuple(latents.shape[-2:]), mode="nearest",
            )

            # After every update the unmasked area is put back on the source's own trajectory at the
            # next sigma -- the rule FLUX.2 and Krea 2 follow -- so it ends on the source exactly.
            def blend(x, sigma_next):
                sigma_next = sigma_next.to(dtype=torch.float32)
                return x * mask + ((1.0 - sigma_next) * source + sigma_next * source_noise) * (1.0 - mask)

        def context(conditioning):
            embeddings = conditioning.embeddings.to(device=device, dtype=self.dtype)
            return embeddings.expand(batch, -1, -1), torch.ones(embeddings.shape[:2], dtype=torch.bool, device=device).expand(batch, -1)

        cond_ctx, cond_mask = context(positive)
        uncond_ctx, uncond_mask = context(negative) if negative is not None else (None, None)
        invocations = {"cond": 0, "uncond": 0, "uncond_skipped": 0}

        def denoise(x, sigma_batch, step_index):
            timestep = (sigma_batch * moss_sampling.MOSS_MULTIPLIER).float()
            cond_v = self.transformer(x, timestep, cond_ctx, cond_mask).float()
            invocations["cond"] += 1
            uncond_v = None
            if plan.runs_unconditional:
                skip = plan.skip_layers_at(sigma_batch)
                uncond_v = self.transformer(x, timestep, uncond_ctx, uncond_mask, skip_layers=skip).float()
                invocations["uncond"] += 1
                invocations["uncond_skipped"] += 1 if skip else 0
            return moss_sampling.cfg_denoised(x, sigma_batch, cond_v, uncond_v, plan.cfg)

        def step_callback(step, total, state):
            if on_step_checkpoint is not None:
                on_step_checkpoint(step, total, state)
            if on_step is not None:
                on_step(step, total, state)

        try:
            if not self.transformer_group_offload_enabled:
                self._ensure_transformer_on_device()
            with torch.inference_mode():
                result, evaluations = moss_sampling.sample_moss(sampler, denoise, latents, sigmas, generators, step_callback, blend)
            self._last_sampling_execution = {
                "actual_transformer_invocations": invocations["cond"] + invocations["uncond"],
                "conditioned_invocations": invocations["cond"],
                "unconditioned_invocations": invocations["uncond"],
                "uncond_skip_invocations": invocations["uncond_skipped"],
                "model_evaluations": evaluations,
                "peak_batch_copies": 1,
                "cfg_branches": 2 if plan.runs_unconditional else 1,
            }
            return result.to("cpu")
        finally:
            if not self.transformer_group_offload_enabled and self.transformer is not None and not self.keep_transformer_resident:
                self._park_transformer_on_cpu()
            if not self.transformer_resident:
                _empty_cuda_cache()

    def _require_device(self):
        if self.device.type != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Native MOSS generation requires CUDA")
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native MOSS BF16 execution requires a BF16-capable CUDA device")

    def _guidance_plan(self, cfg, moss):
        options = dict(moss or {})
        if options.get("standard_slg"):
            raise ValueError("Standard (positive-branch) SLG is not available for MOSS in this release")
        return moss_sampling.build_guidance_plan(
            cfg,
            options.get("uncond_skip_layers", moss_sampling.MOSS_DEFAULT_UNCOND_SKIP_LAYERS),
            options.get("uncond_skip_start", moss_sampling.MOSS_DEFAULT_SKIP_START),
            options.get("uncond_skip_end", moss_sampling.MOSS_DEFAULT_SKIP_END),
            options.get("shift", moss_sampling.MOSS_DEFAULT_SHIFT),
            depth=int(self.config["num_layers"]),
        )

    def generate_batch(self, prompt, negative_prompt, width, height, steps, cfg, sampler, scheduler, generators,
                       guidance="none", pag_scale=0.0, pag_applied_layers="mid", on_step=None, on_step_checkpoint=None,
                       sampling_batch_size=None, moss=None) -> list[Image.Image]:
        self._require_open()
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("prompt and negative_prompt must be strings")
        self._validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step)
        if pag_scale:
            raise ValueError("MOSS does not support PAG")
        del pag_applied_layers
        self._require_device()
        plan = self._guidance_plan(cfg, moss)
        sigmas = moss_sampling.moss_sigma_schedule(steps, scheduler, plan.shift)
        checkpoint = (lambda: on_step_checkpoint(0, steps, None)) if on_step_checkpoint is not None else None
        positive, negative = self._conditioning(prompt, negative_prompt, plan, checkpoint)
        chunk_size = max(1, int(sampling_batch_size or len(generators)))
        chunks = [list(generators[start:start + chunk_size]) for start in range(0, len(generators), chunk_size)]
        steps_executed = len(sigmas) - 1
        total = len(chunks) * steps_executed
        images = []
        executions = []
        for index, chunk in enumerate(chunks):
            offset = index * steps_executed

            def chunked(callback, offset=offset):
                if callback is None:
                    return None
                return lambda step, _total, latents: callback(offset + step, total, latents)

            initial = self._initial_latents(chunk, height, width, sigmas[0])
            latents = sample_or_stream(
                self, chunk,
                lambda initial=initial, chunk=chunk: self._sample(
                    positive, negative, initial, sigmas, sampler, plan, chunk, chunked(on_step), chunked(on_step_checkpoint)
                ),
            )
            executions.append(dict(self._last_sampling_execution))
            if on_step_checkpoint is not None:
                on_step_checkpoint(offset + steps_executed, total, None)
            images.extend(self._run_cuda_stage("vae_decode", lambda latents=latents: self._decode(latents)))
        diagnostics = moss_sampling.moss_sampling_diagnostics(sampler, scheduler, width, height, plan)
        invocations = sum(int(item.get("actual_transformer_invocations", 0)) for item in executions)
        self.last_generation_metrics.setdefault("sampling", {}).update({
            **diagnostics,
            "requested_steps": int(steps),
            "executed_denoise_updates": steps_executed * len(chunks),
            "actual_transformer_invocations": invocations,
            "uncond_skip_invocations": sum(int(item.get("uncond_skip_invocations", 0)) for item in executions),
            "transformer_input_dtype": str(self.dtype).replace("torch.", ""),
            "latent_state_dtype": "float32",
            "noise": "cpu_float32_per_image_generator",
            "prompt_tokens": positive.diagnostics["token_count"],
            "prompt_truncated": positive.diagnostics["truncated"],
            "negative_prompt_tokens": negative.diagnostics["token_count"] if negative is not None else 0,
            "text_encoder_placement": positive.diagnostics["placement"],
        })
        self.last_generation_metrics["capabilities"] = {"components": self.capabilities()}
        self._last_sampling_execution = {
            "actual_transformer_invocations": invocations, "peak_batch_copies": 1,
            "cfg_branches": 2 if plan.runs_unconditional else 1,
        }
        return images

    def refine_batch(self, images, prompt, negative_prompt, steps, denoise, cfg, sampler, scheduler, generators,
                     guidance="none", masks=None, on_step=None, pag_scale=0.0, pag_applied_layers="mid",
                     prepared_conditioning=None, prepared_sigmas=None, force_tiled_decode=False,
                     on_step_checkpoint=None, moss=None) -> list[Image.Image]:
        """A Hires or ADetailer redraw: ComfyUI's KSampler with a denoise, on VAE-encoded source latents.

        The redraw starts at ``(1 - s0) * source + s0 * noise`` on the denoise suffix of the schedule
        and runs the guidance plan a base pass with the same ``moss`` options would run. A mask keeps
        the unmasked area on the source's trajectory. USDU's shared conditioning and sigmas are refused:
        MOSS does not run the tiled path.
        """
        self._require_open()
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("prompt and negative_prompt must be strings")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)) or not images:
            raise ValueError("images must contain at least one RGB PIL image")
        for index, image in enumerate(images):
            if not isinstance(image, Image.Image) or image.mode != "RGB":
                raise ValueError(f"images[{index}] must be an RGB PIL image")
        width, height = images[0].size
        if any(image.size != (width, height) for image in images):
            raise ValueError("all refinement images must have identical dimensions")
        self._validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step)
        if len(generators) != len(images):
            raise ValueError(f"Expected one CPU generator per image ({len(images)}), got {len(generators)}")
        moss_sampling.validate_denoise(denoise)
        refuse_prepared_refinement(MOSS_ENGINE, prepared_conditioning, prepared_sigmas)
        # PAG is ruled out by `guidance` above. The server forwards the request's PAG slider whatever the
        # guidance mode, as it does to every native engine, so the scale alone asks for nothing.
        del pag_scale, pag_applied_layers
        self._require_device()

        mask_tensor = None
        if masks is not None:
            if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)) or len(masks) != len(images):
                raise ValueError("masks must contain one PIL mask per image")
            values = []
            for index, mask in enumerate(masks):
                if not isinstance(mask, Image.Image) or mask.size != (width, height):
                    raise ValueError(f"masks[{index}] must be a same-size PIL image")
                data = torch.frombuffer(bytearray(mask.convert("L").tobytes()), dtype=torch.uint8)
                values.append(data.view(1, 1, height, width).float() / 255.0)
            mask_tensor = torch.cat(values)

        plan = self._guidance_plan(cfg, moss)
        sigmas, schedule = moss_sampling.moss_refinement_sigma_schedule(steps, float(denoise), scheduler, plan.shift)
        checkpoint = (lambda: on_step_checkpoint(0, steps, None)) if on_step_checkpoint is not None else None
        positive, negative = self._conditioning(prompt, negative_prompt, plan, checkpoint, stage="refinement.")
        source = self._run_cuda_stage("refinement.vae_encode", lambda: self._encode_images(images))
        noise = moss_sampling._cpu_noise(source, generators)
        start_sigma = float(sigmas[0].item())
        initial = (1.0 - start_sigma) * source + start_sigma * noise
        latents = sample_or_stream(
            self, generators,
            lambda: self._sample(
                positive, negative, initial, sigmas, sampler, plan, list(generators), on_step, on_step_checkpoint,
                source_latents=source, source_noise=noise, latent_mask=mask_tensor,
            ),
            stage="refinement.sampling",
        )
        execution = dict(self._last_sampling_execution)
        self.last_generation_metrics.setdefault("refinement.sampling", {}).update({
            **moss_sampling.moss_sampling_diagnostics(sampler, scheduler, width, height, plan),
            **schedule,
            "requested_steps": int(steps),
            "denoise": float(denoise),
            "executed_denoise_updates": len(sigmas) - 1,
            "actual_transformer_invocations": int(execution.get("actual_transformer_invocations", 0)),
            "uncond_skip_invocations": int(execution.get("uncond_skip_invocations", 0)),
            "transformer_input_dtype": str(self.dtype).replace("torch.", ""),
            "latent_state_dtype": "float32",
            "noise": "cpu_float32_per_image_generator",
            "masked": mask_tensor is not None,
            "prompt_tokens": positive.diagnostics["token_count"],
            "prompt_truncated": positive.diagnostics["truncated"],
            "negative_prompt_tokens": negative.diagnostics["token_count"] if negative is not None else 0,
            "text_encoder_placement": positive.diagnostics["placement"],
        })
        images = self._run_cuda_stage(
            "refinement.vae_decode",
            lambda: self._decode(latents, force_tiled_decode=force_tiled_decode, metrics_key="refinement.vae_decode"),
        )
        self.last_generation_metrics["capabilities"] = {"components": self.capabilities()}
        return images

    def prepare_refinement_conditioning(self, *args, **kwargs):
        raise ValueError("MOSS does not prepare shared refinement conditioning; USDU tiled redraw is not available")

    @staticmethod
    def prepare_refinement_sigmas(*args, **kwargs):
        raise ValueError("MOSS does not prepare a shared refinement schedule; USDU tiled redraw is not available")


# -- sizing ------------------------------------------------------------------------------------------


def _header_entries(path_value, keep=None):
    files, _index = resolve_component_files(path_value)
    for file in files:
        header = read_safetensors_header(file)
        for key, entry in header.items():
            if keep is None or keep(key):
                yield key, entry


def moss_component_bytes(path_value, role: str, compute_dtype_bytes: int = 2) -> int:
    """Host/device bytes one component occupies once loaded, from headers alone.

    Quantised weights count in their stored form; the text encoder's discarded subgraphs (vision
    tower, ``lm_head``, MTP) do not count at all; the VAE counts at float32.
    """
    keep = _keep_text_encoder_key if role == "text_encoder" else None
    entries = dict(_header_entries(path_value, keep))
    quantized = {key[: -len(COMFY_QUANT_SUFFIX)] for key in entries if key.endswith(COMFY_QUANT_SUFFIX)}
    total = 0
    for key, entry in entries.items():
        dtype = entry.get("dtype", "")
        elements = 1
        for dimension in entry.get("shape", ()):
            elements *= int(dimension)
        layer = key.rsplit(".", 1)[0] if "." in key else ""
        if key.endswith(COMFY_QUANT_SUFFIX) or key.endswith(ATTENTION_CONFIG_SUFFIX) or key.endswith((".input_scale", ".scale_input")):
            continue
        if role == "vae":
            total += elements * (4 if dtype in ("F32", "F16", "BF16") else SAFETENSORS_DTYPE_BYTES.get(dtype, 4))
        elif layer in quantized or dtype in ("I8", "U8", "F8_E4M3", "F8_E5M2"):
            total += elements * SAFETENSORS_DTYPE_BYTES.get(dtype, 1)
        elif dtype == "F32" and (key.endswith(("A_log", "linear_attn.norm.weight", "weight_scale", "scale_weight"))):
            total += elements * 4
        else:
            total += elements * compute_dtype_bytes
    return total


def moss_weight_estimate(paths: Mapping[str, Path]) -> tuple[int, int]:
    sizes = {
        "diffusion_model": moss_component_bytes(paths["diffusion_model"], "diffusion_model"),
        "text_encoder": moss_component_bytes(paths["text_encoder"], "text_encoder"),
        "vae": moss_component_bytes(paths["vae"], "vae"),
    }
    return sum(sizes.values()), max(sizes.values())


# -- loading -------------------------------------------------------------------------------------------


def load_moss_runtime(diffusion_model, text_encoder, vae, tokenizer_path, dtype: torch.dtype = torch.bfloat16,
                      loras=(), device=None) -> MossRuntime:
    """Load the three component mounts (and the pinned tokenizer) into one runtime."""
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("MOSS computes in bfloat16 (or float32 for verification)")
    tokenizer = MossTokenizer.from_resource(tokenizer_path)
    text_model, text_config, text_report = load_moss_text_encoder(text_encoder, dtype)
    vae_module, statistics, vae_report = load_moss_vae(vae)
    transformer, config, transformer_report = load_moss_transformer(diffusion_model, dtype, loras=loras)
    if int(config["text_dim"]) != int(text_config.hidden_size):
        raise ValueError(
            f"This MOSS diffusion model expects {config['text_dim']}-wide conditioning, but the selected text "
            f"encoder produces {text_config.hidden_size}; pair it with Qwen3.5-9B"
        )
    if int(config["latent_channels"]) != MOSS_EXPECTED_LATENT_CHANNELS:
        raise ValueError(f"This MOSS diffusion model expects {config['latent_channels']} latent channels; the FLUX.2 VAE packs 128")
    if tokenizer.vocab_size > int(text_config.vocab_size):
        raise ValueError("The pinned Qwen3.5 tokenizer addresses rows the text encoder's embedding does not have")
    return MossRuntime(
        transformer, text_model, vae_module, tokenizer, dtype, config, text_config, statistics, device=device,
        reports={"transformer": transformer_report, "text_encoder": text_report, "vae": vae_report},
    )
