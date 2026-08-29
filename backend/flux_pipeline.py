"""Native FLUX.1 runtime built on the ComfyUI component layout.

ComfyUI runs FLUX.1 from four separate files — a diffusion model, two text encoders and a VAE —
loaded by ``UNETLoader``, ``DualCLIPLoader(type=flux)`` and ``VAELoader``.  This module reproduces
that graph: it reads the same files, encodes with the same two encoders, and samples the same
rectified-flow trajectory ``flux_sampling`` builds, so a run here is the blueprint's run rather
than a lookalike.

Two properties of FLUX.1 shape everything below and are worth stating once:

* It is **guidance distilled**.  The blueprint runs ``KSampler`` at ``cfg=1`` with the negative
  branch zeroed out, and steers with a ``guidance`` scalar baked into the timestep embedding
  instead.  So a step is one transformer forward, never two, and there is no negative prompt,
  no PAG and no CFG-Zero* — those all need an unconditional branch this model does not have.
* The transformer works on **2x2-packed latents**.  A 1024x1024 canvas is a 128x128 latent, which
  is a 4096-token sequence of 64 channels.  That is why every canvas edge has to be a multiple of
  16 rather than the 8 an ordinary VAE would need.

The module deliberately duck-types :class:`anima_pipeline.AnimaRuntime`: the server drives both
native engines through the same staged-load, group-offload and refinement call sites, and a
second incompatible interface would fork every one of them.
"""

import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import torch
from PIL import Image

try:
    # Shared rectified-flow primitives. Both native engines integrate the same CONST-parametrised
    # flow, so these live in one place: a fix to the update or to the seeded-noise contract has to
    # reach both engines, and two copies would mean it reaches one.
    from .anima_pipeline import (
        _cast_state_dict,
        _cpu_noise_like_batch,
        _discard_module_storage,
        _empty_cuda_cache,
        _module_nbytes,
        _require_safetensors,
        _strict_assign,
        euler_ancestral_rf_step,
        euler_rf_step,
    )
    from .flux_sampling import (
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_refinement_sigma_schedule,
        flux_resolution_shift,
        flux_sigma_schedule,
        resolve_flux_sampler,
    )
    from .prompt_encoding import tokenize_weighted_prompt
except ImportError:
    from anima_pipeline import (
        _cast_state_dict,
        _cpu_noise_like_batch,
        _discard_module_storage,
        _empty_cuda_cache,
        _module_nbytes,
        _require_safetensors,
        _strict_assign,
        euler_ancestral_rf_step,
        euler_rf_step,
    )
    from flux_sampling import (
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_refinement_sigma_schedule,
        flux_resolution_shift,
        flux_sigma_schedule,
        resolve_flux_sampler,
    )
    from prompt_encoding import tokenize_weighted_prompt


GIB = 1024**3

FLUX_CHECKPOINT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# sum(axes_dim) for FLUX.1's 3-axis RoPE, which is also the per-head dimension.
FLUX_AXES_DIMS_ROPE = (16, 56, 56)
FLUX_ATTENTION_HEAD_DIM = sum(FLUX_AXES_DIMS_ROPE)
FLUX_LATENT_CHANNELS = 16
FLUX_VAE_SCALE_FACTOR = 8
FLUX_LATENT_PATCH = 2
# The VAE stride times the transformer's latent patch: a canvas edge that is not a multiple of
# this cannot be packed into whole tokens.
FLUX_PIXEL_ALIGNMENT = FLUX_VAE_SCALE_FACTOR * FLUX_LATENT_PATCH
FLUX_MAX_EDGE = 4096
FLUX_CLIP_SEQUENCE_LENGTH = 77
# FLUX.1 [dev] and its derivatives are trained at a 512-token T5 context; [schnell] at 256. Both
# numbers are the reference implementation's, and the guidance embedding is what tells them apart.
FLUX_GUIDED_T5_SEQUENCE_LENGTH = 512
FLUX_DISTILLED_T5_SEQUENCE_LENGTH = 256

# ComfyUI's scaled-fp8 checkpoints carry this marker plus one `<layer>.scale_weight` per quantised
# linear. The stored weight is float8_e4m3fn and the real weight is `stored * scale`.
FLUX_SCALED_FP8_MARKER = "scaled_fp8"
FLUX_SCALE_WEIGHT_SUFFIX = ".scale_weight"
FLUX_INPUT_SCALE_SUFFIX = ".scale_input"

# ComfyUI's newer mixed-precision checkpoints describe every quantised linear individually: a
# `<layer>.comfy_quant` tensor holding JSON bytes names the format, and the scales it needs sit
# beside the weight. `comfy/ops.py::_load_quantized_module` reads exactly these keys.
FLUX_COMFY_QUANT_SUFFIX = ".comfy_quant"
FLUX_WEIGHT_SCALE_SUFFIX = ".weight_scale"
# Everything a quantised layer may park next to its weight. They are consumed or dropped together
# with the format marker, because a full-precision weight leaves nothing for them to scale.
FLUX_COMFY_QUANT_PARAMETER_SUFFIXES = (
    FLUX_WEIGHT_SCALE_SUFFIX,
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
)
# The formats whose weight is still an ordinary matrix scaled by one number, which is all it takes
# to recover the full-precision weight without ComfyUI's kernels. Everything else — nvfp4, mxfp8,
# the int4 layouts — stores a packed or rotated tensor whose shape is not the layer's shape, and
# unpacking those correctly needs the quantisation library rather than a multiply.
FLUX_SCALAR_SCALE_QUANT_FORMATS = ("float8_e4m3fn", "float8_e5m2")

# ComfyUI's UltimateSDUpscale decodes every tile through VAEDecodeTiled(tile_size=512, overlap=64).
COMFY_VAE_TILE_PIXELS = 512
COMFY_VAE_OVERLAP_PIXELS = 64
COMFY_VAE_STRIDE_PIXELS = COMFY_VAE_TILE_PIXELS - COMFY_VAE_OVERLAP_PIXELS

# The published FLUX.1 autoencoder. Its geometry is fixed across every FLUX.1 release, and the
# shift/scale pair is applied around the latent exactly as `AutoencoderKL` documents.
FLUX_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "act_fn": "silu",
    "latent_channels": FLUX_LATENT_CHANNELS,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "force_upcast": True,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "mid_block_add_attention": True,
}

# google/t5-v1_1-xxl, encoder half only. Declared rather than downloaded so a Flux run never
# depends on a network round trip at generation time.
FLUX_T5_CONFIG = {
    "vocab_size": 32128,
    "d_model": 4096,
    "d_kv": 64,
    "d_ff": 10240,
    "num_layers": 24,
    "num_decoder_layers": 24,
    "num_heads": 64,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "initializer_factor": 1.0,
    "feed_forward_proj": "gated-gelu",
    "is_encoder_decoder": True,
    "use_cache": False,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "tie_word_embeddings": False,
}

# openai/clip-vit-large-patch14's text tower, which is byte-for-byte the `clip_l.safetensors`
# FLUX.1 ships with. `eos_token_id` is 2 because that is what the published config says, and
# transformers keys its pooling on that value: it takes the argmax token position, which is the
# first end-of-text token in a prompt padded with end-of-text — the position ComfyUI pools from.
FLUX_CLIP_CONFIG = {
    "vocab_size": 49408,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": FLUX_CLIP_SEQUENCE_LENGTH,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "initializer_range": 0.02,
    "initializer_factor": 1.0,
    "projection_dim": 768,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "pad_token_id": 1,
}

CLIP_START_TOKEN = "<|startoftext|>"
CLIP_END_TOKEN = "<|endoftext|>"


def _runtime_dependencies():
    try:
        import diffusers
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKL, FluxTransformer2DModel
        from diffusers.loaders.lora_conversion_utils import (
            _convert_kohya_flux_lora_to_diffusers,
            _convert_xlabs_flux_lora_to_diffusers,
        )
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        from safetensors.torch import load_file
        from transformers import (
            CLIPTextConfig,
            CLIPTextModel,
            CLIPTokenizer,
            T5Config,
            T5EncoderModel,
            T5TokenizerFast,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Native Flux requires accelerate, safetensors, transformers with CLIP and T5, and "
            "diffusers 0.38 with FluxTransformer2DModel, AutoencoderKL and the Flux converters"
        ) from error
    if not str(diffusers.__version__).startswith("0.38."):
        raise RuntimeError(f"Native Flux requires diffusers 0.38.x; found {diffusers.__version__}")
    return {
        "init_empty_weights": init_empty_weights,
        "AutoencoderKL": AutoencoderKL,
        "FluxTransformer2DModel": FluxTransformer2DModel,
        "convert_ldm_vae": convert_ldm_vae_checkpoint,
        "convert_kohya_lora": _convert_kohya_flux_lora_to_diffusers,
        "convert_xlabs_lora": _convert_xlabs_flux_lora_to_diffusers,
        "load_file": load_file,
        "CLIPTextConfig": CLIPTextConfig,
        "CLIPTextModel": CLIPTextModel,
        "CLIPTokenizer": CLIPTokenizer,
        "T5Config": T5Config,
        "T5EncoderModel": T5EncoderModel,
        "T5TokenizerFast": T5TokenizerFast,
    }


def normalize_flux_checkpoint_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the wrapper prefixes a Flux diffusion checkpoint may be published under."""
    normalized = {}
    sources = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in FLUX_CHECKPOINT_PREFIXES:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        if new_key in normalized:
            raise ValueError(f"Flux checkpoint key collision: {sources[new_key]!r} and {key!r} -> {new_key!r}")
        normalized[new_key] = value
        sources[new_key] = key
    return normalized


def checkpoint_is_scaled_fp8(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return FLUX_SCALED_FP8_MARKER in state_dict or any(
        key.endswith(FLUX_SCALE_WEIGHT_SUFFIX) for key in state_dict
    )


def dequantize_scaled_fp8(state_dict: Mapping[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Fold ComfyUI's per-tensor fp8 scales back into full-precision weights.

    ComfyUI keeps the fp8 storage and multiplies by the scale inside its own linear op.  Diffusers'
    modules have no such op, so the scale has to be applied once here instead — dropping it would
    not fail, it would quietly produce a differently-scaled model.
    """
    scales = {
        key[: -len(FLUX_SCALE_WEIGHT_SUFFIX)]: value
        for key, value in state_dict.items()
        if key.endswith(FLUX_SCALE_WEIGHT_SUFFIX)
    }
    resolved = {}
    for key, value in state_dict.items():
        if key == FLUX_SCALED_FP8_MARKER or key.endswith(FLUX_SCALE_WEIGHT_SUFFIX):
            continue
        if key.endswith(FLUX_INPUT_SCALE_SUFFIX):
            # An activation scale only matters to an fp8 matmul kernel. Full-precision weights make
            # it a no-op, and carrying it forward would break the strict load.
            continue
        if key.endswith(".weight"):
            layer = key[: -len(".weight")]
            scale = scales.get(layer)
            if scale is not None:
                resolved[key] = value.to(dtype=torch.float32) * scale.to(dtype=torch.float32).reshape(())
                continue
        resolved[key] = value
    unused = set(scales) - {key[: -len(".weight")] for key in state_dict if key.endswith(".weight")}
    if unused:
        raise ValueError(
            f"Flux checkpoint carries {len(unused)} fp8 scales with no matching weight: {sorted(unused)[:3]}"
        )
    return _cast_state_dict(resolved, dtype)


def checkpoint_is_mixed_precision(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(key.endswith(FLUX_COMFY_QUANT_SUFFIX) for key in state_dict)


def _comfy_quant_format(marker: torch.Tensor, layer: str) -> str:
    """Read the format out of a `comfy_quant` tensor, which is JSON stored as bytes."""
    try:
        text = bytes(marker.to(dtype=torch.uint8).reshape(-1).tolist()).decode("utf-8")
        configuration = json.loads(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Flux checkpoint layer {layer!r} has an unreadable comfy_quant marker") from error
    if not isinstance(configuration, dict) or not configuration.get("format"):
        raise ValueError(f"Flux checkpoint layer {layer!r} declares no quantisation format")
    return str(configuration["format"])


def dequantize_mixed_precision(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    label: str = "Flux checkpoint",
) -> dict[str, torch.Tensor]:
    """Fold ComfyUI's per-layer mixed-precision scales back into full-precision weights.

    The same reasoning as :func:`dequantize_scaled_fp8`, one layout later: ComfyUI keeps the fp8
    storage and applies the scale inside its own quantised linear, and Diffusers' modules have no
    such op, so the scale is applied once here instead.  A format whose stored tensor is packed
    rather than scaled is refused by name — expanding it would need ComfyUI's own kernels, and
    guessing at the packing would load a model that is quietly wrong rather than one that fails.
    """
    resolved = {}
    consumed = set()
    for key, marker in state_dict.items():
        if not key.endswith(FLUX_COMFY_QUANT_SUFFIX):
            continue
        layer = key[: -len(FLUX_COMFY_QUANT_SUFFIX)]
        quant_format = _comfy_quant_format(marker, layer)
        weight = state_dict.get(f"{layer}.weight")
        if weight is None:
            raise ValueError(f"{label} layer {layer!r} is quantised but carries no weight")
        if quant_format not in FLUX_SCALAR_SCALE_QUANT_FORMATS:
            raise ValueError(
                f"{label} is quantised as {quant_format!r}, which stores packed weights this "
                "loader cannot expand; use an fp8 (float8_e4m3fn) or unquantised file instead"
            )
        scale = state_dict.get(f"{layer}{FLUX_WEIGHT_SCALE_SUFFIX}")
        if scale is None:
            raise ValueError(f"{label} layer {layer!r} is {quant_format} but carries no weight scale")
        if scale.numel() != 1:
            raise ValueError(
                f"{label} layer {layer!r} carries a {scale.numel()}-element block scale rather than "
                "one per-tensor scale; that layout needs ComfyUI's quantisation kernels"
            )
        resolved[f"{layer}.weight"] = weight.to(dtype=torch.float32) * scale.to(dtype=torch.float32).reshape(())
        consumed.add(key)
        consumed.add(f"{layer}.weight")
        consumed.update(f"{layer}{suffix}" for suffix in FLUX_COMFY_QUANT_PARAMETER_SUFFIXES)
    for key, value in state_dict.items():
        if key in consumed:
            continue
        resolved[key] = value
    return _cast_state_dict(resolved, dtype)


def resolve_quantized_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    label: str = "Flux checkpoint",
) -> dict[str, torch.Tensor]:
    """Expand whichever ComfyUI quantisation a component file uses, or hand it back unchanged.

    Every component can arrive quantised — ComfyUI quantises text encoders as readily as diffusion
    models — so this sits in front of each loader rather than only the transformer's.
    """
    if checkpoint_is_mixed_precision(state_dict):
        return dequantize_mixed_precision(state_dict, dtype, label)
    if checkpoint_is_scaled_fp8(state_dict):
        return dequantize_scaled_fp8(state_dict, dtype)
    return dict(state_dict)


def _shape_of(state_dict: Mapping[str, torch.Tensor], key: str, label: str = "Flux diffusion checkpoint") -> tuple[int, ...]:
    tensor = state_dict.get(key)
    if tensor is None:
        raise ValueError(f"{label} is missing required tensor {key!r}")
    return tuple(tensor.shape)


def _count_blocks(keys, prefix: str, label: str = "Flux diffusion checkpoint") -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    indices = {int(match.group(1)) for match in map(pattern.match, keys) if match}
    if not indices:
        return 0
    if indices != set(range(len(indices))):
        raise ValueError(f"{label} has a gap in its {prefix}* block numbering")
    return len(indices)


def infer_flux_transformer_config(state_dict: Mapping[str, torch.Tensor]) -> dict:
    """Derive the Diffusers transformer configuration from the checkpoint itself.

    ComfyUI's ``model_detection`` reads the same tensors rather than trusting a file name, which is
    what lets one code path serve [dev], [schnell], [Krea dev] and Fill.  The guidance embedding is
    the load-bearing difference: its presence is what distinguishes a distilled-guidance model from
    [schnell], and it decides both the transformer class configuration and the T5 context length.
    """
    keys = list(state_dict.keys())
    num_layers = _count_blocks(keys, "double_blocks.")
    num_single_layers = _count_blocks(keys, "single_blocks.")
    if num_layers < 1 or num_single_layers < 1:
        raise ValueError(
            "The selected file is not a FLUX.1 diffusion model: it has no double_blocks/single_blocks stacks"
        )
    image_in = _shape_of(state_dict, "img_in.weight")
    text_in = _shape_of(state_dict, "txt_in.weight")
    vector_in = _shape_of(state_dict, "vector_in.in_layer.weight")
    proj_out = _shape_of(state_dict, "final_layer.linear.weight")
    inner_dim = int(image_in[0])
    if inner_dim % FLUX_ATTENTION_HEAD_DIM:
        raise ValueError(
            f"Flux hidden size {inner_dim} is not a multiple of the {FLUX_ATTENTION_HEAD_DIM}-dimension attention head"
        )
    if int(text_in[0]) != inner_dim:
        raise ValueError("Flux img_in and txt_in disagree about the hidden size")
    return {
        "patch_size": 1,
        "in_channels": int(image_in[1]),
        "out_channels": int(proj_out[0]),
        "num_layers": num_layers,
        "num_single_layers": num_single_layers,
        "attention_head_dim": FLUX_ATTENTION_HEAD_DIM,
        "num_attention_heads": inner_dim // FLUX_ATTENTION_HEAD_DIM,
        "joint_attention_dim": int(text_in[1]),
        "pooled_projection_dim": int(vector_in[1]),
        "guidance_embeds": any(key.startswith("guidance_in.") for key in keys),
        "axes_dims_rope": FLUX_AXES_DIMS_ROPE,
    }


def flux_t5_sequence_length(guidance_embeds: bool) -> int:
    return FLUX_GUIDED_T5_SEQUENCE_LENGTH if guidance_embeds else FLUX_DISTILLED_T5_SEQUENCE_LENGTH


def _swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    """BFL's final AdaLN emits (shift, scale); Diffusers' reads (scale, shift)."""
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def convert_flux_transformer_state_dict(state_dict: Mapping[str, torch.Tensor], config: Mapping) -> tuple[dict, list[str]]:
    """Rewrite a ComfyUI/BFL FLUX.1 checkpoint into Diffusers' key layout.

    Diffusers ships a converter for this, but it hard-codes a 3072 hidden size and a 4.0 MLP ratio,
    which makes it untestable below full scale and silently wrong for any variant that departs from
    them.  This reads both dimensions off the checkpoint instead, so the same code path serves the
    model and a fixture small enough to load in a unit test.

    Returns the converted state dict and the source keys nothing claimed.
    """
    source = dict(state_dict)

    def take(key: str) -> torch.Tensor:
        try:
            return source.pop(key)
        except KeyError:
            raise ValueError(f"FLUX.1 diffusion checkpoint is missing required tensor {key!r}") from None

    converted = {}

    def move_linear(target: str, key: str) -> None:
        converted[f"{target}.weight"] = take(f"{key}.weight")
        converted[f"{target}.bias"] = take(f"{key}.bias")

    move_linear("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer")
    move_linear("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer")
    move_linear("time_text_embed.text_embedder.linear_1", "vector_in.in_layer")
    move_linear("time_text_embed.text_embedder.linear_2", "vector_in.out_layer")
    if config["guidance_embeds"]:
        move_linear("time_text_embed.guidance_embedder.linear_1", "guidance_in.in_layer")
        move_linear("time_text_embed.guidance_embedder.linear_2", "guidance_in.out_layer")
    move_linear("context_embedder", "txt_in")
    move_linear("x_embedder", "img_in")

    inner_dim = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    image_projections = ("to_q", "to_k", "to_v")
    text_projections = ("add_q_proj", "add_k_proj", "add_v_proj")
    for index in range(int(config["num_layers"])):
        target = f"transformer_blocks.{index}."
        origin = f"double_blocks.{index}."
        move_linear(f"{target}norm1.linear", f"{origin}img_mod.lin")
        move_linear(f"{target}norm1_context.linear", f"{origin}txt_mod.lin")
        for stream, projections in (("img", image_projections), ("txt", text_projections)):
            for suffix in ("weight", "bias"):
                parts = torch.chunk(take(f"{origin}{stream}_attn.qkv.{suffix}"), 3, dim=0)
                for name, part in zip(projections, parts):
                    converted[f"{target}attn.{name}.{suffix}"] = part
        converted[f"{target}attn.norm_q.weight"] = take(f"{origin}img_attn.norm.query_norm.scale")
        converted[f"{target}attn.norm_k.weight"] = take(f"{origin}img_attn.norm.key_norm.scale")
        converted[f"{target}attn.norm_added_q.weight"] = take(f"{origin}txt_attn.norm.query_norm.scale")
        converted[f"{target}attn.norm_added_k.weight"] = take(f"{origin}txt_attn.norm.key_norm.scale")
        move_linear(f"{target}ff.net.0.proj", f"{origin}img_mlp.0")
        move_linear(f"{target}ff.net.2", f"{origin}img_mlp.2")
        move_linear(f"{target}ff_context.net.0.proj", f"{origin}txt_mlp.0")
        move_linear(f"{target}ff_context.net.2", f"{origin}txt_mlp.2")
        move_linear(f"{target}attn.to_out.0", f"{origin}img_attn.proj")
        move_linear(f"{target}attn.to_add_out", f"{origin}txt_attn.proj")

    for index in range(int(config["num_single_layers"])):
        target = f"single_transformer_blocks.{index}."
        origin = f"single_blocks.{index}."
        move_linear(f"{target}norm.linear", f"{origin}modulation.lin")
        # One fused projection holds Q, K, V and the MLP input. The MLP width is whatever is left
        # after the three attention projections, which is how a non-4.0 MLP ratio stays supported.
        fused_weight = take(f"{origin}linear1.weight")
        fused_bias = take(f"{origin}linear1.bias")
        mlp_hidden = int(fused_weight.shape[0]) - 3 * inner_dim
        if mlp_hidden <= 0:
            raise ValueError(
                f"FLUX.1 single block {index} projects {int(fused_weight.shape[0])} rows, which cannot hold "
                f"three {inner_dim}-wide attention projections and an MLP"
            )
        split = (inner_dim, inner_dim, inner_dim, mlp_hidden)
        for suffix, fused in (("weight", fused_weight), ("bias", fused_bias)):
            query, key, value, mlp = torch.split(fused, split, dim=0)
            converted[f"{target}attn.to_q.{suffix}"] = query
            converted[f"{target}attn.to_k.{suffix}"] = key
            converted[f"{target}attn.to_v.{suffix}"] = value
            converted[f"{target}proj_mlp.{suffix}"] = mlp
        converted[f"{target}attn.norm_q.weight"] = take(f"{origin}norm.query_norm.scale")
        converted[f"{target}attn.norm_k.weight"] = take(f"{origin}norm.key_norm.scale")
        move_linear(f"{target}proj_out", f"{origin}linear2")

    move_linear("proj_out", "final_layer.linear")
    converted["norm_out.linear.weight"] = _swap_scale_shift(take("final_layer.adaLN_modulation.1.weight"))
    converted["norm_out.linear.bias"] = _swap_scale_shift(take("final_layer.adaLN_modulation.1.bias"))
    return converted, sorted(source)


def flux_component_bytes(paths) -> int:
    """Host memory a Flux load will hold, read from the files' headers rather than their size.

    Every component is expanded to the BF16 compute dtype, so an fp8 file costs twice what it
    occupies on disk and an fp32 file costs half.  Reporting that before the first tensor is read
    is the difference between a precondition the user can act on and an allocation failure
    thousands of tensors in.
    """
    from safetensors import safe_open

    total = 0
    for path in paths:
        with safe_open(str(Path(path)), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                elements = 1
                for dimension in handle.get_slice(key).get_shape():
                    elements *= int(dimension)
                total += elements * 2
    return total


def _lora_descriptors(loras, label: str = "Flux LoRA") -> list[tuple[Path, float]]:
    descriptors = []
    for descriptor in loras or []:
        if isinstance(descriptor, dict):
            path, multiplier = descriptor["path"], descriptor["multiplier"]
        else:
            path, multiplier = descriptor
        multiplier = float(multiplier)
        if not math.isfinite(multiplier):
            raise ValueError("LoRA multiplier must be finite")
        if multiplier == 0.0:
            continue
        descriptors.append((_require_safetensors(path, label), multiplier))
    return descriptors


def convert_flux_lora_state_dict(state_dict: Mapping[str, torch.Tensor], deps, label: str) -> dict[str, torch.Tensor]:
    """Bring any published Flux LoRA layout into Diffusers' ``lora_A``/``lora_B`` naming.

    The three layouts in circulation are kohya/sd-scripts (``lora_unet_*`` with ``lora_down``),
    XLabs (``double_blocks.*.processor.*``) and Diffusers' own.  The first two converters fold
    ``alpha / rank`` into the matrices, so after conversion the delta is exactly ``B @ A``.
    """
    keys = list(state_dict.keys())
    if any(key.startswith(("lora_unet_", "lora_te_", "lora_te1_")) for key in keys):
        converted = deps["convert_kohya_lora"](dict(state_dict))
    elif any(".processor." in key for key in keys):
        converted = deps["convert_xlabs_lora"](dict(state_dict))
    elif any(".lora_A." in key or ".lora_B." in key for key in keys):
        converted = _scale_diffusers_lora_alphas(state_dict, label)
    else:
        raise ValueError(f"{label} is not a recognised FLUX.1 LoRA layout")
    if not converted:
        raise ValueError(f"{label} produced no usable LoRA tensors")
    return {key[len("transformer."):] if key.startswith("transformer.") else key: value for key, value in converted.items()}


def _scale_diffusers_lora_alphas(state_dict: Mapping[str, torch.Tensor], label: str) -> dict[str, torch.Tensor]:
    """Apply any explicit ``alpha`` to a Diffusers-layout LoRA so every layout ends up pre-scaled."""
    alphas = {}
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = float(value.reshape(()).item())
    converted = {}
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            continue
        if key.endswith(".lora_A.weight"):
            target = key[: -len(".lora_A.weight")]
            alpha = alphas.get(target)
            if alpha is not None:
                rank = int(value.shape[0])
                if rank < 1:
                    raise ValueError(f"{label} declares a zero-rank LoRA for {target}")
                value = value * (alpha / rank)
        converted[key] = value
    return converted


def fuse_flux_lora_state_dict(
    transformer_state: dict[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
    multiplier: float,
    label: str,
) -> int:
    """Add ``multiplier * (B @ A)`` into the base weights and report how many were patched.

    Fusing on the host before the model exists is what keeps LoRA compatible with group offload:
    the streamed blocks carry the patched weights, so there is no adapter to move, no per-forward
    overhead and nothing that a block eviction can leave half-applied.
    """
    targets = {}
    for key, value in lora_state.items():
        if key.endswith(".lora_A.weight"):
            targets.setdefault(key[: -len(".lora_A.weight")], {})["A"] = value
        elif key.endswith(".lora_B.weight"):
            targets.setdefault(key[: -len(".lora_B.weight")], {})["B"] = value
    if not targets:
        raise ValueError(f"{label} contains no lora_A/lora_B pairs")

    patched = 0
    missing = []
    for target, pair in sorted(targets.items()):
        down, up = pair.get("A"), pair.get("B")
        if down is None or up is None:
            raise ValueError(f"{label} has an unpaired LoRA tensor for {target}")
        base_key = f"{target}.weight"
        base = transformer_state.get(base_key)
        if base is None:
            missing.append(target)
            continue
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
            raise ValueError(f"{label} has a malformed LoRA pair for {target}")
        delta = (up.to(torch.float32) @ down.to(torch.float32)) * float(multiplier)
        if tuple(delta.shape) != tuple(base.shape):
            raise ValueError(
                f"{label} targets {target} with a {tuple(delta.shape)} delta but the model weight is {tuple(base.shape)}"
            )
        transformer_state[base_key] = (base.to(torch.float32) + delta).to(base.dtype)
        patched += 1
    if patched == 0:
        raise ValueError(f"{label} matches no weight in this model; it is for a different base")
    return patched


def unmatched_lora_targets(lora_state: Mapping[str, torch.Tensor], transformer_state: Mapping[str, torch.Tensor]) -> list[str]:
    """Targets a LoRA carries that this model has no weight for — text-encoder blocks, mostly."""
    unmatched = []
    for key in lora_state:
        if not key.endswith(".lora_A.weight"):
            continue
        target = key[: -len(".lora_A.weight")]
        if f"{target}.weight" not in transformer_state:
            unmatched.append(target)
    return sorted(unmatched)


def pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) latent grid -> (B, H/2 * W/2, C*4) token sequence."""
    batch, channels, height, width = latents.shape
    if height % FLUX_LATENT_PATCH or width % FLUX_LATENT_PATCH:
        raise ValueError("Flux latents must have even height and width")
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


def unpack_flux_latents(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, H/2 * W/2, C*4) token sequence -> (B, C, H, W) latent grid."""
    batch, sequence, channels = packed.shape
    if height % FLUX_LATENT_PATCH or width % FLUX_LATENT_PATCH:
        raise ValueError("Flux latents must have even height and width")
    if sequence != (height // 2) * (width // 2):
        raise ValueError(f"Flux token sequence {sequence} does not describe a {height} x {width} latent")
    latents = packed.view(batch, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch, channels // 4, height, width)


def flux_image_ids(height: int, width: int, device, dtype) -> torch.Tensor:
    """Row/column positions for every packed latent token, as the 3-axis RoPE expects them."""
    rows, columns = height // 2, width // 2
    ids = torch.zeros(rows, columns, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(rows, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(columns, device=device, dtype=dtype)[None, :]
    return ids.reshape(rows * columns, 3)


def _decoded_tensor_to_image(pixels: torch.Tensor) -> Image.Image:
    array = (pixels.clamp(-1.0, 1.0) + 1.0) * 127.5
    array = array.round().to(dtype=torch.uint8).permute(1, 2, 0).contiguous().cpu()
    return Image.frombytes("RGB", (array.shape[1], array.shape[0]), array.numpy().tobytes())


def _classify_text_encoder(keys) -> str:
    keys = set(keys)
    if any(key.startswith("text_model.encoder.layers.") for key in keys):
        return "clip_l"
    if any(key.startswith("encoder.block.") for key in keys) or "shared.weight" in keys:
        return "t5xxl"
    return "unknown"


def _load_clip_text_encoder(state_dict, dtype: torch.dtype, deps):
    config = deps["CLIPTextConfig"](**FLUX_CLIP_CONFIG)
    with deps["init_empty_weights"]():
        encoder = deps["CLIPTextModel"](config)
    state_dict = resolve_quantized_state_dict(state_dict, dtype, "FLUX.1 CLIP-L text encoder")
    weights = {
        key: value
        for key, value in state_dict.items()
        # `logit_scale`, `text_projection` and the vision tower belong to the full CLIP model, and
        # `position_ids` is a buffer current transformers derives rather than stores. FLUX.1 reads
        # the text tower's pooled output before any projection, so none of them are needed.
        if key.startswith("text_model.") and not key.endswith(".position_ids")
    }
    _strict_assign(encoder, _cast_state_dict(weights, dtype), "FLUX.1 CLIP-L text encoder")
    return encoder.eval().requires_grad_(False)


def _load_t5_text_encoder(state_dict, dtype: torch.dtype, deps):
    config = deps["T5Config"](**FLUX_T5_CONFIG)
    config.use_cache = False
    with deps["init_empty_weights"]():
        encoder = deps["T5EncoderModel"](config)
    state_dict = resolve_quantized_state_dict(state_dict, dtype, "FLUX.1 T5-XXL text encoder")
    weights = {key: value for key, value in state_dict.items() if not key.startswith("decoder.")}
    embedding = weights.get("shared.weight")
    if embedding is None:
        embedding = weights.get("encoder.embed_tokens.weight")
    if embedding is None:
        raise RuntimeError("FLUX.1 T5-XXL text encoder is missing its token embedding")
    # T5EncoderModel ties `shared` and `encoder.embed_tokens`, but a strict assign wants both
    # names present. ComfyUI's published file only carries one of them.
    weights["shared.weight"] = embedding
    weights["encoder.embed_tokens.weight"] = embedding
    weights.pop("lm_head.weight", None)
    _strict_assign(encoder, _cast_state_dict(weights, dtype), "FLUX.1 T5-XXL text encoder")
    return encoder.eval().requires_grad_(False)


def _load_flux_vae(path: Path, dtype: torch.dtype, deps):
    state_dict = deps["load_file"](str(path), device="cpu")
    state_dict = resolve_quantized_state_dict(state_dict, dtype, "FLUX.1 autoencoder")
    if any(key.startswith("decoder.up_blocks.") for key in state_dict):
        converted = dict(state_dict)
    else:
        try:
            converted = deps["convert_ldm_vae"](dict(state_dict), FLUX_VAE_CONFIG)
        except Exception as error:
            raise RuntimeError(f"Failed to convert the FLUX.1 autoencoder checkpoint: {error}") from error
    with deps["init_empty_weights"]():
        vae = deps["AutoencoderKL"](**FLUX_VAE_CONFIG)
    _strict_assign(vae, _cast_state_dict(converted, dtype), "FLUX.1 autoencoder")
    return vae.eval().requires_grad_(False)


def _load_flux_transformer(path: Path, dtype: torch.dtype, deps, loras=(), state_dict=None):
    if state_dict is None:
        state_dict = deps["load_file"](str(path), device="cpu")
    state_dict = normalize_flux_checkpoint_keys(state_dict)
    state_dict = resolve_quantized_state_dict(state_dict, dtype, "FLUX.1 diffusion model")
    config = infer_flux_transformer_config(state_dict)
    converted, unclaimed = convert_flux_transformer_state_dict(state_dict, config)
    del state_dict
    converted = _cast_state_dict(converted, dtype)

    lora_report = []
    for lora_path, multiplier in _lora_descriptors(loras):
        label = f"Flux LoRA {lora_path.name}"
        lora_state = convert_flux_lora_state_dict(deps["load_file"](str(lora_path), device="cpu"), deps, label)
        skipped = unmatched_lora_targets(lora_state, converted)
        patched = fuse_flux_lora_state_dict(converted, lora_state, multiplier, label)
        lora_report.append({
            "name": lora_path.name,
            "multiplier": multiplier,
            "patched_modules": patched,
            "skipped_modules": skipped,
        })
        del lora_state

    with deps["init_empty_weights"]():
        transformer = deps["FluxTransformer2DModel"](**config)
    _strict_assign(transformer, converted, "FLUX.1 transformer")
    del converted
    return transformer.eval().requires_grad_(False), config, {"loras": lora_report, "unclaimed_tensors": unclaimed}


def _load_flux_tokenizers(clip_tokenizer_directory: Path, t5_tokenizer_path: Path, deps):
    directory = Path(clip_tokenizer_directory)
    vocabulary = directory / "vocab.json"
    merges = directory / "merges.txt"
    if not vocabulary.is_file() or not merges.is_file():
        raise FileNotFoundError(
            f"FLUX.1 CLIP-L tokenizer files are missing from {directory}; run the environment configurator to restore them"
        )
    clip = deps["CLIPTokenizer"](
        vocab_file=str(vocabulary),
        merges_file=str(merges),
        errors="replace",
        unk_token=CLIP_END_TOKEN,
        bos_token=CLIP_START_TOKEN,
        eos_token=CLIP_END_TOKEN,
        pad_token=CLIP_END_TOKEN,
        model_max_length=FLUX_CLIP_SEQUENCE_LENGTH,
    )
    if clip.bos_token_id is None or clip.eos_token_id is None:
        raise RuntimeError("FLUX.1 CLIP-L tokenizer is missing its start or end token")
    t5 = deps["T5TokenizerFast"](tokenizer_file=str(t5_tokenizer_path), local_files_only=True)
    if t5.eos_token_id is None:
        raise RuntimeError("FLUX.1 T5 tokenizer.json does not preserve a usable EOS token")
    return clip, t5


class FluxRuntime:
    """A loaded FLUX.1 model, driven through the same surface the Anima runtime exposes."""

    def __init__(self, transformer, text_encoder, text_encoder_2, vae, clip_tokenizer, t5_tokenizer, dtype, config):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.clip_tokenizer = clip_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.dtype = dtype
        self.config = dict(config)
        self.guidance_embeds = bool(config["guidance_embeds"])
        self.t5_sequence_length = flux_t5_sequence_length(self.guidance_embeds)
        self.components = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "vae": vae,
        }
        self.weight_sizes = {
            "transformer": _module_nbytes(transformer),
            "text_encoder": _module_nbytes(text_encoder),
            "text_encoder_2": _module_nbytes(text_encoder_2),
            "vae": _module_nbytes(vae),
        }
        self.weight_sizes["total"] = sum(self.weight_sizes.values())
        # Group offload streams whole blocks, so the largest single block plus everything outside
        # the two block stacks is the resident floor the memory policy has to admit.
        block_bytes = [
            _module_nbytes(block)
            for stack in ("transformer_blocks", "single_transformer_blocks")
            for block in getattr(transformer, stack, ())
        ]
        self.weight_sizes["transformer_max_block"] = max(block_bytes, default=0)
        self.weight_sizes["transformer_unmatched"] = max(
            0, self.weight_sizes["transformer"] - sum(block_bytes)
        )
        # The device every stage is scheduled onto. Production is always CUDA; keeping it an
        # attribute rather than a literal is what lets the sampler, the packing and the decode be
        # exercised end to end on the CPU in a test, which is the only way to check them without a
        # 24 GB checkpoint.
        self.device = torch.device("cuda")
        self.lora_report = []
        self.last_generation_metrics = {}
        self.attention_backend = "native"
        self._vae_tiling_required = False
        # FLUX.1 is guidance distilled: one forward per step, so there is never a second branch to
        # batch and the resident/serial distinction the CFG engines draw does not exist here.
        self.batch_cfg = False
        self.keep_transformer_resident = False
        self.noise_device = "cpu"
        self._closed = False
        self._poisoned = False
        self._transformer_group_offload = False
        self._transformer_blocks_per_group = 0
        self._transformer_resident = False
        self._last_sampling_execution = {}

    # -- lifecycle ---------------------------------------------------------------------------

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Flux runtime is closed")
        if self._poisoned:
            raise RuntimeError("Flux runtime is poisoned by an interrupted transformer restoration")

    @property
    def transformer_group_offload_enabled(self):
        return bool(self._transformer_group_offload)

    @property
    def transformer_resident(self):
        return bool(self._transformer_resident)

    def configure_attention_backend(self, backend="native"):
        self._require_open()
        if not isinstance(backend, str) or not backend:
            raise ValueError("attention backend must be a non-empty string")
        self.attention_backend = backend
        return backend

    def configure_vae_tiling(self, required=False):
        """Force tiled decoding regardless of canvas size, when the memory policy demands it."""
        self._vae_tiling_required = bool(required)
        return "tiled" if self._vae_tiling_required else "auto"

    def configure_transformer_residency(self, keep=False):
        self.keep_transformer_resident = bool(keep)
        if not self.keep_transformer_resident and self.transformer_resident:
            self._park_transformer_on_cpu()
            _empty_cuda_cache()

    def enable_transformer_group_offload(self, blocks_per_group=1):
        self._require_open()
        if not isinstance(blocks_per_group, int) or isinstance(blocks_per_group, bool) or blocks_per_group < 1:
            raise ValueError("blocks_per_group must be a positive integer")
        if self.transformer_group_offload_enabled:
            if self._transformer_blocks_per_group != blocks_per_group:
                raise RuntimeError("Flux transformer group offload is already configured with a different group size")
            return
        if any(parameter.device.type != "cpu" for parameter in self.transformer.parameters()):
            raise RuntimeError("Flux transformer must be on CPU before enabling group offload")
        self.keep_transformer_resident = False
        self._transformer_resident = False
        self.transformer.enable_group_offload(
            onload_device=self.device,
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=blocks_per_group,
            non_blocking=False,
            use_stream=False,
            record_stream=False,
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

    def _ensure_transformer_on_cuda(self):
        if self._transformer_resident:
            return
        self.transformer.to(self.device)
        self._transformer_resident = True

    def _park_transformer_on_cpu(self):
        if not self._transformer_resident:
            return
        self.transformer.to("cpu")
        self._transformer_resident = False

    def to_cpu(self):
        if self._closed:
            return
        if not self.transformer_group_offload_enabled:
            self._park_transformer_on_cpu()
        for name in ("text_encoder", "text_encoder_2", "vae"):
            module = getattr(self, name, None)
            if module is not None:
                module.to("cpu")
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
        for name in ("transformer", "text_encoder", "text_encoder_2", "vae"):
            module = getattr(self, name, None)
            if module is None:
                continue
            # Release the storage without copying it: a stray reference held by a hook or an
            # attention processor follows the release rather than pinning what it captured.
            _discard_module_storage(module)
            setattr(self, name, None)
        self.components = {}
        _empty_cuda_cache()

    # -- prompt encoding ---------------------------------------------------------------------

    def _tokenize(self, texts: Sequence[str]):
        clip = [
            tokenize_weighted_prompt(
                self.clip_tokenizer,
                text,
                max_length=FLUX_CLIP_SEQUENCE_LENGTH,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            for text in texts
        ]
        t5 = [
            tokenize_weighted_prompt(
                self.t5_tokenizer,
                text,
                max_length=self.t5_sequence_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            for text in texts
        ]
        return clip, t5

    def token_diagnostics(self, prompt: str):
        self._require_open()
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        clip, t5 = self._tokenize([prompt])
        return {
            "clip": {
                "token_count": int(clip[0]["token_count"]),
                "weighted_token_count": int(clip[0]["weighted_token_count"]),
                "max_length": FLUX_CLIP_SEQUENCE_LENGTH,
            },
            "t5": {
                "token_count": int(t5[0]["token_count"]),
                "weighted_token_count": int(t5[0]["weighted_token_count"]),
                "max_length": self.t5_sequence_length,
            },
        }

    def _encode_prompt(self, prompt: str):
        """Return the T5 sequence conditioning and the CLIP-L pooled vector, both on CPU.

        ComfyUI weights a prompt by blending each token's embedding towards the same token's
        embedding in an empty prompt, so a weight of 1 is exactly the unweighted result.  The
        empty pass is only run when the prompt actually carries a weight — on T5-XXL it is not a
        free forward.
        """
        clip_tokens, t5_tokens = self._tokenize([prompt])
        weights = t5_tokens[0]["weights"]
        weighted = bool(t5_tokens[0]["weighted_token_count"])
        if weighted:
            empty_clip, empty_t5 = self._tokenize([""])
            t5_ids = torch.stack([t5_tokens[0]["input_ids"], empty_t5[0]["input_ids"]])
        else:
            t5_ids = t5_tokens[0]["input_ids"].unsqueeze(0)
        clip_ids = clip_tokens[0]["input_ids"].unsqueeze(0)
        device = self.device
        try:
            self.text_encoder.to(device=device, dtype=self.dtype)
            self.text_encoder_2.to(device=device, dtype=self.dtype)
            with torch.inference_mode():
                # FLUX.1 runs both encoders without an attention mask, exactly as the reference
                # implementation does: the padding is part of the trained context.
                pooled = self.text_encoder(input_ids=clip_ids.to(device), output_hidden_states=False).pooler_output
                embeddings = self.text_encoder_2(input_ids=t5_ids.to(device), output_hidden_states=False)[0]
                if weighted:
                    prompt_embeddings, empty_embeddings = embeddings[0], embeddings[1]
                    scale = weights.to(device=device, dtype=prompt_embeddings.dtype).unsqueeze(-1)
                    embeddings = (
                        (prompt_embeddings - empty_embeddings) * scale + empty_embeddings
                    ).unsqueeze(0)
            return embeddings.to(device="cpu", dtype=self.dtype), pooled.to(device="cpu", dtype=self.dtype)
        finally:
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")
            _empty_cuda_cache()

    # -- latents -----------------------------------------------------------------------------

    def _latent_scale(self):
        scaling = float(getattr(self.vae.config, "scaling_factor", FLUX_VAE_CONFIG["scaling_factor"]))
        shift = float(getattr(self.vae.config, "shift_factor", FLUX_VAE_CONFIG["shift_factor"]) or 0.0)
        return scaling, shift

    def _initial_latents(self, generators, height: int, width: int, sigma=1.0) -> torch.Tensor:
        shape = (1, FLUX_LATENT_CHANNELS, height // FLUX_VAE_SCALE_FACTOR, width // FLUX_VAE_SCALE_FACTOR)
        noise = torch.cat(
            [torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu") for generator in generators]
        )
        return noise * float(torch.as_tensor(sigma, dtype=torch.float32).item())

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        device = self.device
        scaling, shift = self._latent_scale()
        pixels = []
        for image in images:
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(image.height, image.width, 3)
            pixels.append(data.permute(2, 0, 1))
        vae_input = torch.stack(pixels).to(device=device, dtype=self.dtype) / 127.5 - 1.0
        try:
            self.vae.to(device=device, dtype=self.dtype)
            with torch.inference_mode():
                encoded = self.vae.encode(vae_input)
                posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
                latents = posterior.mode()
                return ((latents.float() - shift) * scaling).to(device="cpu", dtype=torch.float32)
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    def _resolved_tiled_decode(self, height: int, width: int, force_tiled_decode: bool):
        # A FLUX.1 decode of a 2048-edge canvas is a 4 GB activation on its own. Past the point
        # where a full-frame decode stops fitting beside the rest of the run, tile it — the same
        # 512/64 geometry ComfyUI's VAEDecodeTiled uses.
        if force_tiled_decode or self._vae_tiling_required:
            return True
        return max(height, width) > 1536

    def _configure_vae_tiling(self, tiled: bool) -> None:
        """Point `AutoencoderKL`'s tiling at ComfyUI's VAEDecodeTiled geometry.

        This model's `enable_tiling()` takes no arguments — the geometry lives in three attributes
        it reads per call — so the tile size and overlap are set here rather than passed in.
        """
        if not tiled:
            self.vae.disable_tiling()
            return
        self.vae.tile_sample_min_size = COMFY_VAE_TILE_PIXELS
        self.vae.tile_latent_min_size = COMFY_VAE_TILE_PIXELS // FLUX_VAE_SCALE_FACTOR
        self.vae.tile_overlap_factor = COMFY_VAE_OVERLAP_PIXELS / COMFY_VAE_TILE_PIXELS
        self.vae.enable_tiling()

    def _decode(self, latents: torch.Tensor, force_tiled_decode: bool = False) -> list[Image.Image]:
        device = self.device
        scaling, shift = self._latent_scale()
        height = latents.shape[2] * FLUX_VAE_SCALE_FACTOR
        width = latents.shape[3] * FLUX_VAE_SCALE_FACTOR
        tiled = self._resolved_tiled_decode(height, width, force_tiled_decode)
        try:
            self.vae.to(device=device, dtype=self.dtype)
            self._configure_vae_tiling(tiled)
            scaled = (latents.to(device=device, dtype=torch.float32) / scaling + shift).to(self.dtype)
            with torch.inference_mode():
                decoded = self.vae.decode(scaled, return_dict=False)[0]
                # Converted inside the inference scope: the decoded tensor is an inference tensor,
                # and every read of it belongs on the same side of the boundary that produced it.
                images = [_decoded_tensor_to_image(frame.float()) for frame in decoded]
            self.last_generation_metrics.setdefault("vae_decode", {}).update({
                "actual_vae_mode": "tiled" if tiled else "full",
                "requested_tiled_decode": {"tile": COMFY_VAE_TILE_PIXELS, "overlap": COMFY_VAE_OVERLAP_PIXELS},
                "resolved_tiled_decode": {
                    "tile": int(getattr(self.vae, "tile_sample_min_size", COMFY_VAE_TILE_PIXELS)),
                    "overlap_factor": float(getattr(self.vae, "tile_overlap_factor", 0.0)),
                },
            })
            return images
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    # -- sampling ----------------------------------------------------------------------------

    @staticmethod
    def _validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step):
        if sampler not in FLUX_SAMPLERS:
            raise ValueError(f"Unsupported Flux sampler: {sampler}")
        if scheduler not in FLUX_SCHEDULERS:
            raise ValueError(f"Unsupported Flux scheduler: {scheduler}")
        if guidance != "none":
            raise ValueError(
                "FLUX.1 is guidance distilled and runs a single forward per step; PAG and CFG-Zero* need an unconditional branch it does not have"
            )
        if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
            raise ValueError("width and height must be integers")
        if width <= 0 or height <= 0 or width % FLUX_PIXEL_ALIGNMENT or height % FLUX_PIXEL_ALIGNMENT:
            raise ValueError(f"width and height must be positive and divisible by {FLUX_PIXEL_ALIGNMENT}")
        if width > FLUX_MAX_EDGE or height > FLUX_MAX_EDGE:
            raise ValueError(f"width and height cannot exceed the Flux maximum of {FLUX_MAX_EDGE}")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(float(cfg)):
            raise ValueError("cfg must be finite")
        if not isinstance(generators, Sequence) or isinstance(generators, (str, bytes)) or len(generators) < 1:
            raise ValueError("generators must contain at least one CPU torch.Generator")
        for index, generator in enumerate(generators):
            if not isinstance(generator, torch.Generator) or torch.device(generator.device).type != "cpu":
                raise ValueError(f"generators[{index}] must be a CPU torch.Generator")
        if on_step is not None and not callable(on_step):
            raise ValueError("on_step must be callable")

    def _run_cuda_stage(self, name: str, operation):
        measured = False
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
                    metric.update(
                        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                    )
                except Exception:
                    pass
            recorded = self.last_generation_metrics.get(name)
            self.last_generation_metrics[name] = {**recorded, **metric} if isinstance(recorded, dict) else metric

    def _sample(
        self,
        embeddings,
        pooled,
        latents,
        sigmas,
        sampler,
        guidance_scale,
        generators,
        on_step,
        on_step_checkpoint,
        source_latents=None,
        source_noise=None,
        latent_mask=None,
    ):
        device = self.device
        sampler_implementation, _warning = resolve_flux_sampler(sampler)
        batch, _channels, latent_height, latent_width = latents.shape
        latents = latents.to(device=device, dtype=torch.float32)
        embeddings = embeddings.to(device=device, dtype=self.dtype)
        pooled = pooled.to(device=device, dtype=self.dtype)
        if embeddings.shape[0] == 1 and batch > 1:
            embeddings = embeddings.expand(batch, -1, -1)
        if pooled.shape[0] == 1 and batch > 1:
            pooled = pooled.expand(batch, -1)
        # Positions stay float32: the RoPE embedding upcasts them anyway, and a BF16 index stops
        # being exact past 256, which a 4096-edge Hires canvas would reach.
        image_ids = flux_image_ids(latent_height, latent_width, device, torch.float32)
        text_ids = torch.zeros(embeddings.shape[1], 3, device=device, dtype=torch.float32)
        guidance_tensor = (
            torch.full((batch,), float(guidance_scale), device=device, dtype=torch.float32)
            if self.guidance_embeds
            else None
        )
        sigmas = sigmas.to(device=device, dtype=torch.float32)
        source = source_latents.to(device=device, dtype=torch.float32) if source_latents is not None else None
        source_noise_f = source_noise.to(device=device, dtype=torch.float32) if source_noise is not None else None
        resized_mask = None
        if latent_mask is not None:
            if source is None or source_noise_f is None:
                raise ValueError("A latent mask requires the source latents it composites against")
            resized_mask = torch.nn.functional.interpolate(
                latent_mask.to(device=device, dtype=torch.float32).view(batch, 1, *latent_mask.shape[-2:]),
                size=(latent_height, latent_width),
                mode="nearest",
            )
        # Ancestral samplers draw fresh noise every step. Keeping that on the CPU generator the
        # caller owns is what makes a seed reproduce across machines.
        noise_generators = list(generators)
        sample_steps = len(sigmas) - 1
        invocations = 0
        previous_prediction = None

        def predict(state, sigma):
            nonlocal invocations
            invocations += 1
            timestep = torch.full((batch,), float(sigma), device=device, dtype=torch.float32)
            output = self.transformer(
                hidden_states=pack_flux_latents(state.to(self.dtype)),
                encoder_hidden_states=embeddings,
                pooled_projections=pooled,
                timestep=timestep,
                img_ids=image_ids,
                txt_ids=text_ids,
                guidance=guidance_tensor,
                return_dict=False,
            )[0]
            return unpack_flux_latents(output.float(), latent_height, latent_width)

        from diffusers.models.attention_dispatch import attention_backend

        try:
            if not self.transformer_group_offload_enabled:
                self._ensure_transformer_on_cuda()
            with attention_backend(self.attention_backend), torch.inference_mode():
                for index in range(sample_steps):
                    sigma = sigmas[index]
                    sigma_next = sigmas[index + 1]
                    prediction = predict(latents, sigma)

                    if sampler_implementation == "euler":
                        latents = euler_rf_step(latents, prediction, sigma, sigma_next)
                    elif sampler_implementation == "euler_ancestral":
                        denoised = latents - sigma.float() * prediction
                        latents = euler_ancestral_rf_step(
                            latents, denoised, sigma, sigma_next, noise_generators, eta=1.0, s_noise=1.0
                        )
                    elif sampler_implementation == "flow_lcm":
                        denoised = latents.float() - sigma.float() * prediction
                        if float(sigma_next) == 0.0:
                            latents = denoised
                        else:
                            sigma_next_device = torch.as_tensor(sigma_next, device=device, dtype=torch.float32)
                            noise = _cpu_noise_like_batch(latents.float(), noise_generators)
                            latents = (1.0 - sigma_next_device) * denoised + sigma_next_device * noise
                    elif sampler_implementation == "multistep":
                        delta = torch.as_tensor(sigma_next - sigma, device=device, dtype=torch.float32)
                        derivative = (
                            prediction if previous_prediction is None else 1.5 * prediction - 0.5 * previous_prediction
                        )
                        latents = latents.float() + delta * derivative
                        previous_prediction = prediction
                    elif float(sigma_next) == 0.0:
                        latents = euler_rf_step(latents, prediction, sigma, sigma_next)
                    else:
                        delta = torch.as_tensor(sigma_next - sigma, device=device, dtype=torch.float32)
                        if sampler_implementation == "midpoint":
                            sigma_eval = torch.as_tensor(sigma, device=device, dtype=torch.float32) + delta * 0.5
                            provisional = latents.float() + delta * 0.5 * prediction
                            corrected = predict(provisional, sigma_eval)
                            latents = latents.float() + delta * corrected
                        else:
                            provisional = latents.float() + delta * prediction
                            corrected = predict(provisional, sigma_next)
                            latents = latents.float() + delta * 0.5 * (prediction + corrected)
                    latents = latents.to(dtype=torch.float32)
                    if resized_mask is not None:
                        sigma_next_f = torch.as_tensor(sigma_next, dtype=torch.float32, device=device)
                        source_noised = (1.0 - sigma_next_f) * source + sigma_next_f * source_noise_f
                        latents = latents * resized_mask + source_noised * (1.0 - resized_mask)
                    if on_step_checkpoint is not None:
                        on_step_checkpoint(index + 1, sample_steps, latents)
                    if on_step is not None:
                        on_step(index + 1, sample_steps, latents)
            self._last_sampling_execution = {
                "actual_transformer_invocations": invocations,
                "peak_batch_copies": 1,
            }
            return latents.to("cpu")
        finally:
            if not self.transformer_group_offload_enabled and self.transformer is not None and not self.keep_transformer_resident:
                self._park_transformer_on_cpu()
            if not self.transformer_resident:
                _empty_cuda_cache()

    def _require_cuda(self):
        if self.device.type != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Native Flux generation requires CUDA")
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native Flux BF16 execution requires a BF16-capable CUDA device")

    def _record_sampling_metrics(self, name: str, steps: int, schedule_diagnostics: dict, shift: float):
        execution = self._last_sampling_execution or {}
        self.last_generation_metrics.setdefault(name, {}).update({
            **schedule_diagnostics,
            "shift": round(float(shift), 4),
            "transformer_input_dtype": str(self.dtype).replace("torch.", ""),
            "latent_state_dtype": "float32",
            "requested_steps": int(steps),
            "executed_denoise_updates": int(steps),
            "branch_invocations_per_update": 1,
            "sequential_transformer_invocations": int(steps),
            "actual_transformer_invocations": int(execution.get("actual_transformer_invocations", steps)),
            "peak_batch_copies": 1,
        })

    def generate_batch(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        sampler,
        scheduler,
        generators,
        guidance="none",
        pag_scale=0.0,
        pag_applied_layers="mid",
        on_step: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        on_step_checkpoint: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        sampling_batch_size=None,
    ) -> list[Image.Image]:
        self._require_open()
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("prompt and negative_prompt must be strings")
        if negative_prompt:
            raise ValueError("FLUX.1 has no unconditional branch; the negative prompt must be empty")
        self._validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step)
        self._require_cuda()
        del pag_scale, pag_applied_layers

        shift = flux_resolution_shift(width, height)
        sigmas = flux_sigma_schedule(steps, scheduler, shift)
        embeddings, pooled = self._run_cuda_stage("prompt_encode", lambda: self._encode_prompt(prompt))
        # A chunk is a subdivision of one batch, not a separate run: the progress callbacks are
        # offset so the bar counts one pass over `chunks x steps`, and the invocation total is
        # summed rather than overwritten by whichever chunk happened to finish last.
        chunk_size = max(1, int(sampling_batch_size or len(generators)))
        chunks = [list(generators[start:start + chunk_size]) for start in range(0, len(generators), chunk_size)]
        steps_executed = len(sigmas) - 1
        total = len(chunks) * steps_executed
        images = []
        invocations = 0
        for index, chunk in enumerate(chunks):
            offset = index * steps_executed

            def chunked(callback, offset=offset):
                if callback is None:
                    return None
                return lambda step, _total, latents: callback(offset + step, total, latents)

            initial = self._initial_latents(chunk, height, width, sigmas[0])
            latents = self._run_cuda_stage(
                "sampling",
                lambda initial=initial, chunk=chunk: self._sample(
                    embeddings, pooled, initial, sigmas, sampler, cfg, chunk,
                    chunked(on_step), chunked(on_step_checkpoint),
                ),
            )
            invocations += int((self._last_sampling_execution or {}).get("actual_transformer_invocations", 0))
            images.extend(self._run_cuda_stage("vae_decode", lambda latents=latents: self._decode(latents)))
        self._last_sampling_execution = {"actual_transformer_invocations": invocations, "peak_batch_copies": 1}
        self._record_sampling_metrics(
            "sampling", total, {"schedule_mode": "full", "schedule_steps": steps}, shift
        )
        return images

    def refine_batch(
        self,
        images,
        prompt,
        negative_prompt,
        steps,
        denoise,
        cfg,
        sampler,
        scheduler,
        generators,
        guidance="none",
        masks=None,
        on_step: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        pag_scale: float = 0.0,
        pag_applied_layers: str = "mid",
        prepared_conditioning=None,
        prepared_sigmas=None,
        force_tiled_decode: bool = False,
        on_step_checkpoint: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> list[Image.Image]:
        self._require_open()
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("prompt and negative_prompt must be strings")
        if negative_prompt:
            raise ValueError("FLUX.1 has no unconditional branch; the negative prompt must be empty")
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
        if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not math.isfinite(float(denoise)):
            raise ValueError("denoise must be finite")
        if float(denoise) <= 0.0 or float(denoise) > 1.0:
            raise ValueError("denoise must be greater than 0 and at most 1")
        self._require_cuda()
        del pag_scale, pag_applied_layers, prepared_conditioning, prepared_sigmas

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

        shift = flux_resolution_shift(width, height)
        sigmas, schedule_diagnostics = flux_refinement_sigma_schedule(steps, float(denoise), scheduler, shift)
        embeddings, pooled = self._run_cuda_stage("refinement.prompt_encode", lambda: self._encode_prompt(prompt))
        source = self._run_cuda_stage("refinement.vae_encode", lambda: self._encode_images(images))
        noise = _cpu_noise_like_batch(source, generators)
        start_sigma = float(sigmas[0].item())
        initial = (1.0 - start_sigma) * source + start_sigma * noise
        latents = self._run_cuda_stage(
            "refinement.sampling",
            lambda: self._sample(
                embeddings, pooled, initial, sigmas, sampler, cfg, list(generators),
                on_step, on_step_checkpoint,
                source_latents=source, source_noise=noise, latent_mask=mask_tensor,
            ),
        )
        self._record_sampling_metrics("refinement.sampling", len(sigmas) - 1, schedule_diagnostics, shift)
        return self._run_cuda_stage(
            "refinement.vae_decode", lambda: self._decode(latents, force_tiled_decode=force_tiled_decode)
        )


def load_flux_runtime(
    diffusion_model,
    text_encoder,
    text_encoder_2,
    vae,
    clip_tokenizer_directory,
    t5_tokenizer_path,
    dtype: torch.dtype = torch.bfloat16,
    loras=(),
) -> FluxRuntime:
    """Load the four ComfyUI component files into one runtime.

    ``text_encoder`` and ``text_encoder_2`` are classified from their own tensors rather than from
    their position, so a user who picks CLIP-L and T5-XXL the other way round gets a working run
    instead of a shape error thousands of tensors later.
    """
    deps = _runtime_dependencies()
    diffusion_path = _require_safetensors(diffusion_model, "Flux diffusion model")
    first_path = _require_safetensors(text_encoder, "Flux text encoder")
    second_path = _require_safetensors(text_encoder_2, "Flux second text encoder")
    vae_path = _require_safetensors(vae, "Flux VAE")
    if first_path == second_path:
        raise ValueError("Flux requires two different text encoders: CLIP-L and T5-XXL")

    encoders = {}
    for path in (first_path, second_path):
        state = deps["load_file"](str(path), device="cpu")
        kind = _classify_text_encoder(state.keys())
        if kind == "unknown":
            raise ValueError(f"{path.name} is neither a CLIP-L nor a T5-XXL text encoder")
        if kind in encoders:
            raise ValueError(f"Flux needs one CLIP-L and one T5-XXL text encoder; both files are {kind}")
        encoders[kind] = state
    if "clip_l" not in encoders or "t5xxl" not in encoders:
        raise ValueError("Flux requires both a CLIP-L and a T5-XXL text encoder")

    clip_tokenizer, t5_tokenizer = _load_flux_tokenizers(clip_tokenizer_directory, t5_tokenizer_path, deps)
    clip_encoder = _load_clip_text_encoder(encoders.pop("clip_l"), dtype, deps)
    t5_encoder = _load_t5_text_encoder(encoders.pop("t5xxl"), dtype, deps)
    flux_vae = _load_flux_vae(vae_path, dtype, deps)
    transformer, config, load_report = _load_flux_transformer(diffusion_path, dtype, deps, loras=loras)
    runtime = FluxRuntime(
        transformer, clip_encoder, t5_encoder, flux_vae, clip_tokenizer, t5_tokenizer, dtype, config
    )
    runtime.lora_report = load_report["loras"]
    runtime.unclaimed_tensors = load_report["unclaimed_tensors"]
    return runtime
