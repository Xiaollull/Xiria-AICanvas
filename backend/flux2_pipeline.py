"""Native FLUX.2 runtime built on the ComfyUI component layout.

ComfyUI runs FLUX.2 from three files — a diffusion model, one text encoder and a VAE — loaded by
``UNETLoader``, ``CLIPLoader(type=flux2)`` and ``VAELoader``.  This module reproduces that graph.
It shares FLUX.1's rectified-flow sampler and its whole staged-execution surface, because the two
models are the same family; what it does not share is any of the three components, and the
differences are worth stating once:

* **The text encoder is a large language model, not a text tower.**  FLUX.2 [dev] conditions on
  Mistral-Small-3.1-24B and FLUX.2 [klein] on Qwen3-4B/8B, and neither is read at its output: the
  conditioning is three *intermediate* hidden states concatenated along the channel axis, which is
  why ``joint_attention_dim`` is three times the language model's hidden size.  There is no CLIP
  tower and no pooled vector at all.
* **The autoencoder does the packing.**  FLUX.1 hands the transformer a 16-channel latent that the
  transformer packs 2x2 itself.  FLUX.2's autoencoder emits 32 channels at stride 8 and the packing
  happens outside it — a 2x2 pixel-unshuffle followed by a fixed batch normalisation — so the
  transformer runs at patch size 1 on 128-channel tokens of stride 16.
* **Position is four-dimensional.**  FLUX.2's RoPE has four axes: reference-image index, row,
  column, and a fourth axis that carries the *text* token position.  FLUX.1 leaves text at the
  origin; FLUX.2 orders it.

Like FLUX.1 it is guidance distilled — one forward per step, steered by a scalar in the timestep
embedding — so there is no negative prompt, no PAG and no CFG-Zero* here either.

The module deliberately duck-types :class:`anima_pipeline.AnimaRuntime` for the same reason
:mod:`flux_pipeline` does: the server drives every native engine through one set of staged-load,
group-offload and refinement call sites.
"""

import math
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import torch
from PIL import Image

try:
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
    from .flux2_sampling import (
        FLUX2_SAMPLERS,
        FLUX2_SCHEDULERS,
        flux2_refinement_sigma_schedule,
        flux2_resolution_shift,
        flux2_sigma_schedule,
        resolve_flux2_sampler,
    )
    from .flux_pipeline import (
        COMFY_VAE_OVERLAP_PIXELS,
        COMFY_VAE_TILE_PIXELS,
        _count_blocks,
        _decoded_tensor_to_image,
        _lora_descriptors,
        _scale_diffusers_lora_alphas,
        _shape_of,
        _swap_scale_shift,
        flux_component_bytes,
        fuse_flux_lora_state_dict,
        normalize_flux_checkpoint_keys,
        resolve_quantized_state_dict,
        unmatched_lora_targets,
    )
    from .tekken_tokenizer import load_tekken_tokenizer, tekken_tokenizer_from_state_dict
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
    from flux2_sampling import (
        FLUX2_SAMPLERS,
        FLUX2_SCHEDULERS,
        flux2_refinement_sigma_schedule,
        flux2_resolution_shift,
        flux2_sigma_schedule,
        resolve_flux2_sampler,
    )
    from flux_pipeline import (
        COMFY_VAE_OVERLAP_PIXELS,
        COMFY_VAE_TILE_PIXELS,
        _count_blocks,
        _decoded_tensor_to_image,
        _lora_descriptors,
        _scale_diffusers_lora_alphas,
        _shape_of,
        _swap_scale_shift,
        flux_component_bytes,
        fuse_flux_lora_state_dict,
        normalize_flux_checkpoint_keys,
        resolve_quantized_state_dict,
        unmatched_lora_targets,
    )
    from tekken_tokenizer import load_tekken_tokenizer, tekken_tokenizer_from_state_dict


# Four RoPE axes — reference index, row, column, text position — summing to the head dimension.
FLUX2_AXES_DIMS_ROPE = (32, 32, 32, 32)
FLUX2_ATTENTION_HEAD_DIM = sum(FLUX2_AXES_DIMS_ROPE)
FLUX2_ROPE_THETA = 2000
FLUX2_NORM_EPS = 1e-6

# The autoencoder emits 32 channels at stride 8; the 2x2 pack outside it makes the 128-channel,
# stride-16 token the transformer consumes.
FLUX2_VAE_LATENT_CHANNELS = 32
FLUX2_VAE_SCALE_FACTOR = 8
FLUX2_LATENT_PATCH = 2
FLUX2_LATENT_CHANNELS = FLUX2_VAE_LATENT_CHANNELS * FLUX2_LATENT_PATCH * FLUX2_LATENT_PATCH
# `EmptyFlux2LatentImage` steps width and height by 16 for exactly this reason.
FLUX2_PIXEL_ALIGNMENT = FLUX2_VAE_SCALE_FACTOR * FLUX2_LATENT_PATCH
FLUX2_MAX_EDGE = 4096

# `model_base.Flux2.extra_conds` left-pads the conditioning to 512 tokens with zeros. That is the
# trained context; a longer prompt is passed through rather than truncated, exactly as ComfyUI
# does, and the ceiling below only exists so a pathological prompt fails with a sentence instead
# of an allocation.
FLUX2_TEXT_SEQUENCE_LENGTH = 512
FLUX2_MAX_TEXT_TOKENS = 4096

# `comfy/text_encoders/flux.py`: three intermediate layers are tapped and concatenated. The tap
# depths are per language model and are what `joint_attention_dim` is three times the width of.
FLUX2_MISTRAL_TAP_LAYERS = (10, 20, 30)
FLUX2_QWEN_TAP_LAYERS = (9, 18, 27)
FLUX2_TAP_COUNT = 3

# `Flux2Tokenizer.llama_template`, verbatim.
FLUX2_MISTRAL_TEMPLATE = (
    "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured "
    "responses focusing on object relationships, object\nattribution and actions without "
    "speculation.[/SYSTEM_PROMPT][INST]{}[/INST]"
)
# `KleinTokenizer.llama_template`, verbatim.
FLUX2_QWEN_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# `Qwen3Tokenizer(min_length=512, pad_token=151643)`: the Klein encoders pad their *tokens* to the
# trained context and mask the padding, where Mistral pads its *embeddings* instead.
FLUX2_QWEN_PAD_TOKEN = 151643

# The published FLUX.2 autoencoder. `mid_block_add_attention`, the quantisation convolutions and
# the batch-normalised 2x2 packing are all what `comfy/sd.py` builds for this checkpoint.
FLUX2_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "act_fn": "silu",
    "latent_channels": FLUX2_VAE_LATENT_CHANNELS,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "force_upcast": True,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
    "mid_block_add_attention": True,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
    "patch_size": (FLUX2_LATENT_PATCH, FLUX2_LATENT_PATCH),
}

# ComfyUI publishes this VAE with the quantisation convolutions nested under the encoder and
# decoder; `comfy/sd.py` lifts them to the top level before building the module.
FLUX2_VAE_KEY_MOVES = (
    ("encoder.quant_conv.", "quant_conv."),
    ("decoder.post_quant_conv.", "post_quant_conv."),
)
FLUX2_VAE_BATCH_NORM_KEYS = ("bn.running_mean", "bn.running_var")

# mistralai/Mistral-Small-3.1-24B's language half, and Qwen3's, as `comfy/text_encoders/llama.py`
# declares them. Every dimension that can be read off the checkpoint is; these are the ones that
# cannot be — the rotary base and the norm epsilon leave no trace in the tensors.
FLUX2_MISTRAL_CONFIG = {
    "rope_theta": 1000000000.0,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 8192,
    "sliding_window": None,
    "attention_dropout": 0.0,
    "use_cache": False,
}
FLUX2_QWEN3_CONFIG = {
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 40960,
    "attention_dropout": 0.0,
    "use_cache": False,
}

TEXT_ENCODER_PREFIX = "model."


def _runtime_dependencies():
    try:
        import diffusers
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
        from diffusers.loaders.lora_conversion_utils import (
            _convert_kohya_flux2_lora_to_diffusers,
            _convert_non_diffusers_flux2_lora_to_diffusers,
        )
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        from safetensors.torch import load_file
        from transformers import (
            MistralConfig,
            MistralModel,
            PreTrainedTokenizerFast,
            Qwen3Config,
            Qwen3Model,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Native Flux2 requires accelerate, safetensors, transformers with Mistral and Qwen3, "
            "and diffusers 0.38 with Flux2Transformer2DModel and AutoencoderKLFlux2"
        ) from error
    if not str(diffusers.__version__).startswith("0.38."):
        raise RuntimeError(f"Native Flux2 requires diffusers 0.38.x; found {diffusers.__version__}")
    return {
        "init_empty_weights": init_empty_weights,
        "AutoencoderKLFlux2": AutoencoderKLFlux2,
        "Flux2Transformer2DModel": Flux2Transformer2DModel,
        "convert_ldm_vae": convert_ldm_vae_checkpoint,
        "convert_kohya_lora": _convert_kohya_flux2_lora_to_diffusers,
        "convert_ai_toolkit_lora": _convert_non_diffusers_flux2_lora_to_diffusers,
        "load_file": load_file,
        "MistralConfig": MistralConfig,
        "MistralModel": MistralModel,
        "Qwen3Config": Qwen3Config,
        "Qwen3Model": Qwen3Model,
        "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
    }


# -- transformer ------------------------------------------------------------------------------


def infer_flux2_transformer_config(state_dict: Mapping[str, torch.Tensor]) -> dict:
    """Derive the Diffusers FLUX.2 configuration from the checkpoint itself.

    ``model_detection`` recognises FLUX.2 by the shared modulation projection the per-block
    modulations were replaced with, so that tensor is the gate here too.  Everything else — the
    hidden size, the MLP ratio, the conditioning width and the two block counts — is read rather
    than assumed, which is what lets one path serve [dev] and [klein] and a fixture small enough
    to load in a unit test.
    """
    keys = list(state_dict.keys())
    if "double_stream_modulation_img.lin.weight" not in state_dict:
        raise ValueError(
            "The selected file is not a FLUX.2 diffusion model: it carries no shared "
            "double_stream_modulation_img projection"
        )
    label = "FLUX.2 diffusion checkpoint"
    num_layers = _count_blocks(keys, "double_blocks.", label)
    num_single_layers = _count_blocks(keys, "single_blocks.", label)
    if num_layers < 1 or num_single_layers < 1:
        raise ValueError(
            "The selected file is not a FLUX.2 diffusion model: it has no double_blocks/single_blocks stacks"
        )
    image_in = _shape_of(state_dict, "img_in.weight", label)
    text_in = _shape_of(state_dict, "txt_in.weight", label)
    proj_out = _shape_of(state_dict, "final_layer.linear.weight", label)
    time_in = _shape_of(state_dict, "time_in.in_layer.weight", label)
    mlp_in = _shape_of(state_dict, "double_blocks.0.img_mlp.0.weight", label)

    inner_dim = int(image_in[0])
    if inner_dim % FLUX2_ATTENTION_HEAD_DIM:
        raise ValueError(
            f"Flux2 hidden size {inner_dim} is not a multiple of the {FLUX2_ATTENTION_HEAD_DIM}-dimension attention head"
        )
    if int(text_in[0]) != inner_dim:
        raise ValueError("Flux2 img_in and txt_in disagree about the hidden size")
    # The SwiGLU projection emits gate and value side by side, so the MLP is half of it.
    mlp_hidden = int(mlp_in[0]) // 2
    if mlp_hidden * 2 != int(mlp_in[0]) or mlp_hidden < 1:
        raise ValueError(f"Flux2 MLP projection width {int(mlp_in[0])} is not a gated pair")
    mlp_ratio = mlp_hidden / inner_dim
    if int(inner_dim * mlp_ratio) != mlp_hidden:
        raise ValueError(f"Flux2 MLP ratio {mlp_ratio} does not reproduce a {mlp_hidden}-wide MLP")
    return {
        "patch_size": 1,
        "in_channels": int(image_in[1]),
        "out_channels": int(proj_out[0]),
        "num_layers": num_layers,
        "num_single_layers": num_single_layers,
        "attention_head_dim": FLUX2_ATTENTION_HEAD_DIM,
        "num_attention_heads": inner_dim // FLUX2_ATTENTION_HEAD_DIM,
        "joint_attention_dim": int(text_in[1]),
        "timestep_guidance_channels": int(time_in[1]),
        "mlp_ratio": mlp_ratio,
        "axes_dims_rope": FLUX2_AXES_DIMS_ROPE,
        "rope_theta": FLUX2_ROPE_THETA,
        "eps": FLUX2_NORM_EPS,
        "guidance_embeds": any(key.startswith("guidance_in.") for key in keys),
    }


def convert_flux2_transformer_state_dict(
    state_dict: Mapping[str, torch.Tensor], config: Mapping
) -> tuple[dict, list[str]]:
    """Rewrite a ComfyUI/BFL FLUX.2 checkpoint into Diffusers' key layout.

    Returns the converted state dict and the source keys nothing claimed.  FLUX.2 carries no
    biases anywhere (``ops_bias=False`` in ``comfy/ldm/flux/model.py``), and its single blocks keep
    the fused QKV+MLP projection Diffusers also keeps, so the only tensor that has to be taken
    apart is the double blocks' fused QKV.
    """
    source = dict(state_dict)

    def take(key: str) -> torch.Tensor:
        try:
            return source.pop(key)
        except KeyError:
            raise ValueError(f"FLUX.2 diffusion checkpoint is missing required tensor {key!r}") from None

    def move(target: str, key: str) -> None:
        converted[target] = take(key)

    def move_norm(target: str, key: str) -> None:
        """`process_unet_state_dict` renames `*_norm.scale` to `*_norm.weight`; take either."""
        if f"{key}.scale" in source:
            converted[target] = take(f"{key}.scale")
        else:
            converted[target] = take(f"{key}.weight")

    converted = {}
    move("time_guidance_embed.timestep_embedder.linear_1.weight", "time_in.in_layer.weight")
    move("time_guidance_embed.timestep_embedder.linear_2.weight", "time_in.out_layer.weight")
    if config["guidance_embeds"]:
        move("time_guidance_embed.guidance_embedder.linear_1.weight", "guidance_in.in_layer.weight")
        move("time_guidance_embed.guidance_embedder.linear_2.weight", "guidance_in.out_layer.weight")
    move("x_embedder.weight", "img_in.weight")
    move("context_embedder.weight", "txt_in.weight")
    move("double_stream_modulation_img.linear.weight", "double_stream_modulation_img.lin.weight")
    move("double_stream_modulation_txt.linear.weight", "double_stream_modulation_txt.lin.weight")
    move("single_stream_modulation.linear.weight", "single_stream_modulation.lin.weight")

    image_projections = ("to_q", "to_k", "to_v")
    text_projections = ("add_q_proj", "add_k_proj", "add_v_proj")
    for index in range(int(config["num_layers"])):
        target = f"transformer_blocks.{index}."
        origin = f"double_blocks.{index}."
        for stream, projections in (("img", image_projections), ("txt", text_projections)):
            parts = torch.chunk(take(f"{origin}{stream}_attn.qkv.weight"), 3, dim=0)
            for name, part in zip(projections, parts):
                converted[f"{target}attn.{name}.weight"] = part
        move_norm(f"{target}attn.norm_q.weight", f"{origin}img_attn.norm.query_norm")
        move_norm(f"{target}attn.norm_k.weight", f"{origin}img_attn.norm.key_norm")
        move_norm(f"{target}attn.norm_added_q.weight", f"{origin}txt_attn.norm.query_norm")
        move_norm(f"{target}attn.norm_added_k.weight", f"{origin}txt_attn.norm.key_norm")
        move(f"{target}attn.to_out.0.weight", f"{origin}img_attn.proj.weight")
        move(f"{target}attn.to_add_out.weight", f"{origin}txt_attn.proj.weight")
        move(f"{target}ff.linear_in.weight", f"{origin}img_mlp.0.weight")
        move(f"{target}ff.linear_out.weight", f"{origin}img_mlp.2.weight")
        move(f"{target}ff_context.linear_in.weight", f"{origin}txt_mlp.0.weight")
        move(f"{target}ff_context.linear_out.weight", f"{origin}txt_mlp.2.weight")

    for index in range(int(config["num_single_layers"])):
        target = f"single_transformer_blocks.{index}."
        origin = f"single_blocks.{index}."
        move(f"{target}attn.to_qkv_mlp_proj.weight", f"{origin}linear1.weight")
        move(f"{target}attn.to_out.weight", f"{origin}linear2.weight")
        move_norm(f"{target}attn.norm_q.weight", f"{origin}norm.query_norm")
        move_norm(f"{target}attn.norm_k.weight", f"{origin}norm.key_norm")

    move("proj_out.weight", "final_layer.linear.weight")
    converted["norm_out.linear.weight"] = _swap_scale_shift(take("final_layer.adaLN_modulation.1.weight"))
    return converted, sorted(source)


def convert_flux2_lora_state_dict(state_dict: Mapping[str, torch.Tensor], deps, label: str) -> dict[str, torch.Tensor]:
    """Bring any published FLUX.2 LoRA layout into Diffusers' ``lora_A``/``lora_B`` naming."""
    keys = list(state_dict.keys())
    if any("dora_scale" in key for key in keys):
        raise ValueError(f"{label} is a DoRA checkpoint, which this runtime cannot fuse")
    if any(".lora_down.weight" in key for key in keys):
        # kohya folds alpha / rank into the matrices itself.
        converted = deps["convert_kohya_lora"](dict(state_dict))
    else:
        prepared = {
            key.replace("base_model.model.", "diffusion_model."): value for key, value in state_dict.items()
        }
        if any(key.startswith("diffusion_model.") for key in prepared):
            converted = deps["convert_ai_toolkit_lora"](prepared)
        elif any(".lora_A." in key or ".lora_B." in key for key in prepared):
            converted = _scale_diffusers_lora_alphas(prepared, label)
        else:
            raise ValueError(f"{label} is not a recognised FLUX.2 LoRA layout")
    if not converted:
        raise ValueError(f"{label} produced no usable LoRA tensors")
    return {
        key[len("transformer."):] if key.startswith("transformer.") else key: value
        for key, value in converted.items()
    }


# -- latents ----------------------------------------------------------------------------------


def patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, 32, H, W) autoencoder latent -> (B, 128, H/2, W/2), channel-major within each 2x2 cell."""
    batch, channels, height, width = latents.shape
    if height % FLUX2_LATENT_PATCH or width % FLUX2_LATENT_PATCH:
        raise ValueError("Flux2 autoencoder latents must have even height and width")
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 1, 3, 5, 2, 4)
    return packed.reshape(batch, channels * 4, height // 2, width // 2)


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, 128, H, W) transformer latent -> (B, 32, H*2, W*2) autoencoder latent."""
    batch, channels, height, width = latents.shape
    if channels % 4:
        raise ValueError("Flux2 transformer latents must have a channel count divisible by four")
    unpacked = latents.reshape(batch, channels // 4, 2, 2, height, width)
    unpacked = unpacked.permute(0, 1, 4, 2, 5, 3)
    return unpacked.reshape(batch, channels // 4, height * 2, width * 2)


def pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C). The transformer runs at patch size 1; the VAE already packed."""
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def unpack_flux2_latents(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, H, W)."""
    batch, sequence, channels = packed.shape
    if sequence != height * width:
        raise ValueError(f"Flux2 token sequence {sequence} does not describe a {height} x {width} latent")
    return packed.permute(0, 2, 1).reshape(batch, channels, height, width)


def flux2_image_ids(height: int, width: int, device, dtype) -> torch.Tensor:
    """Four-axis positions for every latent token: (reference index, row, column, text position)."""
    ids = torch.zeros(height, width, len(FLUX2_AXES_DIMS_ROPE), device=device, dtype=dtype)
    ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    return ids.reshape(height * width, len(FLUX2_AXES_DIMS_ROPE))


def flux2_text_ids(length: int, device, dtype) -> torch.Tensor:
    """Text tokens sit at the spatial origin and are ordered along the fourth RoPE axis."""
    ids = torch.zeros(length, len(FLUX2_AXES_DIMS_ROPE), device=device, dtype=dtype)
    ids[:, 3] = torch.arange(length, device=device, dtype=dtype)
    return ids


# -- component loading ------------------------------------------------------------------------


def classify_flux2_text_encoder(keys) -> str:
    """Tell a Mistral text encoder from a Qwen3 one by its attention, not by its file name."""
    keys = set(keys)
    prefixed = {key[len(TEXT_ENCODER_PREFIX):] for key in keys if key.startswith(TEXT_ENCODER_PREFIX)}
    keys = keys | prefixed
    if "layers.0.post_attention_layernorm.weight" not in keys:
        return "unknown"
    # Qwen3 normalises queries and keys per head; Mistral does not.
    return "qwen3" if "layers.0.self_attn.q_norm.weight" in keys else "mistral3"


def _language_model_weights(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    weights = {}
    for key, value in state_dict.items():
        if key == "tekken_model":
            continue
        name = key[len(TEXT_ENCODER_PREFIX):] if key.startswith(TEXT_ENCODER_PREFIX) else key
        if name.startswith("lm_head.") or name.startswith("visual.") or name.startswith("vision_tower."):
            # FLUX.2 reads intermediate hidden states; a language-model head or a vision tower
            # bundled into the same file is never evaluated.
            continue
        weights[name] = value
    return weights


def infer_flux2_text_encoder_config(weights: Mapping[str, torch.Tensor], family: str) -> dict:
    """Read the language model's geometry off its own tensors."""
    label = f"Flux2 {family} text encoder"
    layers = _count_blocks(weights.keys(), "layers.", label)
    if layers < max(FLUX2_MISTRAL_TAP_LAYERS if family == "mistral3" else FLUX2_QWEN_TAP_LAYERS):
        raise ValueError(
            f"{label} has {layers} layers, too few for its {FLUX2_TAP_COUNT}-layer conditioning tap"
        )
    embedding = _shape_of(weights, "embed_tokens.weight", label)
    query = _shape_of(weights, "layers.0.self_attn.q_proj.weight", label)
    key_projection = _shape_of(weights, "layers.0.self_attn.k_proj.weight", label)
    mlp = _shape_of(weights, "layers.0.mlp.gate_proj.weight", label)
    hidden_size = int(embedding[1])
    head_dim = FLUX2_ATTENTION_HEAD_DIM
    if int(query[0]) % head_dim or int(key_projection[0]) % head_dim:
        raise ValueError(f"Flux2 {family} attention projections are not multiples of {head_dim}")
    base = dict(FLUX2_MISTRAL_CONFIG if family == "mistral3" else FLUX2_QWEN3_CONFIG)
    base.update(
        vocab_size=int(embedding[0]),
        hidden_size=hidden_size,
        intermediate_size=int(mlp[0]),
        num_hidden_layers=layers,
        num_attention_heads=int(query[0]) // head_dim,
        num_key_value_heads=int(key_projection[0]) // head_dim,
        head_dim=head_dim,
    )
    return base


def _load_flux2_text_encoder(state_dict, dtype: torch.dtype, deps):
    """Build the language model half of a FLUX.2 text encoder and load it strictly."""
    family = classify_flux2_text_encoder(state_dict.keys())
    if family == "unknown":
        raise ValueError("The selected file is not a FLUX.2 text encoder: it has no decoder layers")
    # The published [klein] encoders ship quantised as readily as the diffusion model does, and the
    # tap reads hidden states rather than logits, so the weights have to be real ones.
    state_dict = resolve_quantized_state_dict(state_dict, dtype, f"FLUX.2 {family} text encoder")
    weights = _language_model_weights(state_dict)
    config = infer_flux2_text_encoder_config(weights, family)
    if family == "mistral3":
        model_config = deps["MistralConfig"](**config)
        with deps["init_empty_weights"]():
            encoder = deps["MistralModel"](model_config)
    else:
        model_config = deps["Qwen3Config"](**config)
        with deps["init_empty_weights"]():
            encoder = deps["Qwen3Model"](model_config)

    if "norm.weight" not in weights:
        # ComfyUI prunes FLUX.2 [dev]'s Mistral to 30 layers and drops the final norm with it
        # (`Mistral3_24BModel` sets `final_norm=False` below 40 layers). The deepest tap is then
        # the last layer's raw output, so the norm has to disappear rather than be neutralised:
        # an RMS norm with unit weights still rescales.
        encoder.norm = torch.nn.Identity()
    _strict_assign(encoder, _cast_state_dict(weights, dtype), f"FLUX.2 {family} text encoder")
    encoder.config.use_cache = False
    return encoder.eval().requires_grad_(False), family


def _load_flux2_vae(path: Path, dtype: torch.dtype, deps):
    """Load the FLUX.2 autoencoder and the packing statistics that live beside it."""
    state_dict = deps["load_file"](str(path), device="cpu")
    state_dict = resolve_quantized_state_dict(state_dict, dtype, "FLUX.2 autoencoder")
    moved = {}
    for key, value in state_dict.items():
        for prefix, replacement in FLUX2_VAE_KEY_MOVES:
            if key.startswith(prefix):
                key = replacement + key[len(prefix):]
                break
        moved[key] = value
    statistics = {}
    for key in FLUX2_VAE_BATCH_NORM_KEYS:
        tensor = moved.get(key)
        if tensor is None:
            raise ValueError(
                f"The selected file is not a FLUX.2 autoencoder: it carries no {key!r} packing statistic"
            )
        statistics[key] = tensor.to(dtype=torch.float32)

    if any(key.startswith("decoder.up_blocks.") for key in moved):
        converted = {key: value for key, value in moved.items() if not key.startswith("bn.")}
    else:
        try:
            converted = deps["convert_ldm_vae"](dict(moved), FLUX2_VAE_CONFIG)
        except Exception as error:
            raise RuntimeError(f"Failed to convert the FLUX.2 autoencoder checkpoint: {error}") from error
    converted = _cast_state_dict(converted, dtype)
    # The running statistics stay float32: they are 128 numbers that rescale every latent, and the
    # converter drops them because no Diffusers VAE mapping mentions them.
    converted.update(statistics)
    converted.setdefault("bn.num_batches_tracked", torch.zeros((), dtype=torch.long))

    with deps["init_empty_weights"]():
        vae = deps["AutoencoderKLFlux2"](**FLUX2_VAE_CONFIG)
    _strict_assign(vae, converted, "FLUX.2 autoencoder")
    return vae.eval().requires_grad_(False), statistics


def _load_flux2_transformer(path: Path, dtype: torch.dtype, deps, loras=(), state_dict=None):
    if state_dict is None:
        state_dict = deps["load_file"](str(path), device="cpu")
    state_dict = normalize_flux_checkpoint_keys(state_dict)
    state_dict = resolve_quantized_state_dict(state_dict, dtype, "FLUX.2 diffusion model")
    config = infer_flux2_transformer_config(state_dict)
    converted, unclaimed = convert_flux2_transformer_state_dict(state_dict, config)
    del state_dict
    converted = _cast_state_dict(converted, dtype)

    lora_report = []
    for lora_path, multiplier in _lora_descriptors(loras, "Flux2 LoRA"):
        label = f"Flux2 LoRA {lora_path.name}"
        lora_state = convert_flux2_lora_state_dict(deps["load_file"](str(lora_path), device="cpu"), deps, label)
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
        transformer = deps["Flux2Transformer2DModel"](**config)
    _strict_assign(transformer, converted, "FLUX.2 transformer")
    del converted
    return transformer.eval().requires_grad_(False), config, {"loras": lora_report, "unclaimed_tensors": unclaimed}


def _load_flux2_tokenizer(family: str, state_dict, encoder_path: Path, qwen_tokenizer_path, deps):
    """The tokenizer a FLUX.2 text encoder needs, from wherever that family keeps it."""
    if family == "mistral3":
        # ComfyUI's published encoder carries the tekken table inside the checkpoint. A file
        # converted from HuggingFace will not, so a sibling `tekken.json` is accepted too rather
        # than turning a recoverable situation into a dead end.
        sidecar = Path(encoder_path).with_name("tekken.json")
        if "tekken_model" not in state_dict and sidecar.is_file():
            return load_tekken_tokenizer(sidecar.read_bytes())
        return tekken_tokenizer_from_state_dict(state_dict)
    path = Path(qwen_tokenizer_path) if qwen_tokenizer_path else None
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "The FLUX.2 [klein] Qwen tokenizer resource is missing; run the environment configurator to restore it"
        )
    tokenizer = deps["PreTrainedTokenizerFast"](tokenizer_file=str(path), local_files_only=True)
    if tokenizer.vocab_size < 1:
        raise RuntimeError("The FLUX.2 [klein] Qwen tokenizer.json does not declare a usable vocabulary")
    return tokenizer


class Flux2Runtime:
    """A loaded FLUX.2 model, driven through the same surface the Anima and Flux runtimes expose."""

    def __init__(self, transformer, text_encoder, vae, tokenizer, dtype, config, family, latent_statistics):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = vae
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.config = dict(config)
        self.family = family
        self.guidance_embeds = bool(config["guidance_embeds"])
        self.tap_layers = FLUX2_MISTRAL_TAP_LAYERS if family == "mistral3" else FLUX2_QWEN_TAP_LAYERS
        self.prompt_template = FLUX2_MISTRAL_TEMPLATE if family == "mistral3" else FLUX2_QWEN_TEMPLATE
        self.text_sequence_length = FLUX2_TEXT_SEQUENCE_LENGTH
        self._latent_mean = latent_statistics["bn.running_mean"].to(torch.float32).reshape(1, -1, 1, 1)
        self._latent_std = torch.sqrt(
            latent_statistics["bn.running_var"].to(torch.float32) + float(FLUX2_VAE_CONFIG["batch_norm_eps"])
        ).reshape(1, -1, 1, 1)
        self.components = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "vae": vae,
        }
        self.weight_sizes = {
            "transformer": _module_nbytes(transformer),
            "text_encoder": _module_nbytes(text_encoder),
            "vae": _module_nbytes(vae),
        }
        self.weight_sizes["total"] = sum(self.weight_sizes.values())
        block_bytes = [
            _module_nbytes(block)
            for stack in ("transformer_blocks", "single_transformer_blocks")
            for block in getattr(transformer, stack, ())
        ]
        self.weight_sizes["transformer_max_block"] = max(block_bytes, default=0)
        self.weight_sizes["transformer_unmatched"] = max(
            0, self.weight_sizes["transformer"] - sum(block_bytes)
        )
        self.device = torch.device("cuda")
        self.lora_report = []
        self.unclaimed_tensors = []
        self.last_generation_metrics = {}
        self.attention_backend = "native"
        self._vae_tiling_required = False
        # Guidance distilled: one forward per step, so there is never a second branch to batch.
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
            raise RuntimeError("Flux2 runtime is closed")
        if self._poisoned:
            raise RuntimeError("Flux2 runtime is poisoned by an interrupted transformer restoration")

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
                raise RuntimeError("Flux2 transformer group offload is already configured with a different group size")
            return
        if any(parameter.device.type != "cpu" for parameter in self.transformer.parameters()):
            raise RuntimeError("Flux2 transformer must be on CPU before enabling group offload")
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
        for name in ("text_encoder", "vae"):
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
        for name in ("transformer", "text_encoder", "vae"):
            module = getattr(self, name, None)
            if module is None:
                continue
            _discard_module_storage(module)
            setattr(self, name, None)
        self.components = {}
        _empty_cuda_cache()

    # -- prompt encoding ---------------------------------------------------------------------

    def _encode_tokens(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The token ids and attention mask ComfyUI feeds this family's language model.

        FLUX.2 encoders take the prompt unweighted: ``Flux2Tokenizer`` and ``KleinTokenizer`` both
        pass ``disable_weights=True``, so ``(word:1.2)`` is literal text here rather than an
        emphasis instruction.
        """
        text = self.prompt_template.format(prompt)
        if self.family == "mistral3":
            ids = self.tokenizer.encode(text, add_bos=True)
        else:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        token_count = len(ids)
        if token_count > FLUX2_MAX_TEXT_TOKENS:
            raise ValueError(
                f"This prompt tokenises to {token_count} FLUX.2 text tokens, beyond the "
                f"{FLUX2_MAX_TEXT_TOKENS}-token ceiling; shorten it"
            )
        mask = [1] * token_count
        if self.family != "mistral3" and token_count < self.text_sequence_length:
            # Klein pads its tokens to the trained context and masks the padding.
            padding = self.text_sequence_length - token_count
            ids = list(ids) + [FLUX2_QWEN_PAD_TOKEN] * padding
            mask = mask + [0] * padding
        return (
            torch.tensor([list(ids)], dtype=torch.long),
            torch.tensor([mask], dtype=torch.long),
            token_count,
        )

    def token_diagnostics(self, prompt: str):
        self._require_open()
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        _ids, _mask, token_count = self._encode_tokens(prompt)
        return {
            "llm": {
                "token_count": int(token_count),
                "weighted_token_count": 0,
                "max_length": self.text_sequence_length,
                "ceiling": FLUX2_MAX_TEXT_TOKENS,
                "family": self.family,
            }
        }

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Three intermediate hidden states, concatenated on the channel axis, on CPU.

        ``Flux2TEModel.encode_token_weights`` stacks the taps and folds them into the channel
        dimension, and ``model_base.Flux2.extra_conds`` then left-pads the result to 512 tokens
        with zeros.  Both happen here so the transformer sees exactly what ComfyUI hands it.
        """
        ids, mask, _token_count = self._encode_tokens(prompt)
        device = self.device
        try:
            self.text_encoder.to(device=device, dtype=self.dtype)
            with torch.inference_mode():
                output = self.text_encoder(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    output_hidden_states=True,
                    use_cache=False,
                )
                states = output.hidden_states
                if len(states) <= max(self.tap_layers):
                    raise RuntimeError(
                        f"FLUX.2 {self.family} text encoder produced {len(states)} hidden states, "
                        f"too few for taps {self.tap_layers}"
                    )
                stacked = torch.stack([states[index] for index in self.tap_layers], dim=1)
                batch, taps, sequence, width = stacked.shape
                embeddings = stacked.permute(0, 2, 1, 3).reshape(batch, sequence, taps * width)
                if sequence < self.text_sequence_length:
                    embeddings = torch.nn.functional.pad(
                        embeddings, (0, 0, self.text_sequence_length - sequence, 0)
                    )
            return embeddings.to(device="cpu", dtype=self.dtype)
        finally:
            self.text_encoder.to("cpu")
            _empty_cuda_cache()

    # -- latents -----------------------------------------------------------------------------

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self._latent_mean.to(device=latents.device, dtype=torch.float32)
        std = self._latent_std.to(device=latents.device, dtype=torch.float32)
        return (latents.float() - mean) / std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self._latent_mean.to(device=latents.device, dtype=torch.float32)
        std = self._latent_std.to(device=latents.device, dtype=torch.float32)
        return latents.float() * std + mean

    def _initial_latents(self, generators, height: int, width: int, sigma=1.0) -> torch.Tensor:
        shape = (
            1,
            FLUX2_LATENT_CHANNELS,
            height // FLUX2_PIXEL_ALIGNMENT,
            width // FLUX2_PIXEL_ALIGNMENT,
        )
        noise = torch.cat(
            [torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu") for generator in generators]
        )
        return noise * float(torch.as_tensor(sigma, dtype=torch.float32).item())

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        device = self.device
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
                latents = patchify_flux2_latents(posterior.mode())
                return self._normalize_latents(latents).to(device="cpu", dtype=torch.float32)
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    def _resolved_tiled_decode(self, height: int, width: int, force_tiled_decode: bool):
        if force_tiled_decode or self._vae_tiling_required:
            return True
        return max(height, width) > 1536

    def _configure_vae_tiling(self, tiled: bool) -> None:
        """Point the autoencoder's tiling at ComfyUI's VAEDecodeTiled geometry.

        The tile geometry lives in three attributes rather than in ``enable_tiling``'s arguments —
        it takes none — and the latent tile is measured in the autoencoder's own stride of 8,
        before the 2x2 pack.
        """
        if not tiled:
            self.vae.disable_tiling()
            return
        self.vae.tile_sample_min_size = COMFY_VAE_TILE_PIXELS
        self.vae.tile_latent_min_size = COMFY_VAE_TILE_PIXELS // FLUX2_VAE_SCALE_FACTOR
        self.vae.tile_overlap_factor = COMFY_VAE_OVERLAP_PIXELS / COMFY_VAE_TILE_PIXELS
        self.vae.enable_tiling()

    def _decode(self, latents: torch.Tensor, force_tiled_decode: bool = False) -> list[Image.Image]:
        device = self.device
        height = latents.shape[2] * FLUX2_PIXEL_ALIGNMENT
        width = latents.shape[3] * FLUX2_PIXEL_ALIGNMENT
        tiled = self._resolved_tiled_decode(height, width, force_tiled_decode)
        try:
            self.vae.to(device=device, dtype=self.dtype)
            self._configure_vae_tiling(tiled)
            unpacked = unpatchify_flux2_latents(
                self._denormalize_latents(latents.to(device=device, dtype=torch.float32))
            )
            with torch.inference_mode():
                decoded = self.vae.decode(unpacked.to(self.dtype), return_dict=False)[0]
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
        if sampler not in FLUX2_SAMPLERS:
            raise ValueError(f"Unsupported Flux2 sampler: {sampler}")
        if scheduler not in FLUX2_SCHEDULERS:
            raise ValueError(f"Unsupported Flux2 scheduler: {scheduler}")
        if guidance != "none":
            raise ValueError(
                "FLUX.2 is guidance distilled and runs a single forward per step; PAG and CFG-Zero* need an unconditional branch it does not have"
            )
        if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
            raise ValueError("width and height must be integers")
        if width <= 0 or height <= 0 or width % FLUX2_PIXEL_ALIGNMENT or height % FLUX2_PIXEL_ALIGNMENT:
            raise ValueError(f"width and height must be positive and divisible by {FLUX2_PIXEL_ALIGNMENT}")
        if width > FLUX2_MAX_EDGE or height > FLUX2_MAX_EDGE:
            raise ValueError(f"width and height cannot exceed the Flux2 maximum of {FLUX2_MAX_EDGE}")
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
        sampler_implementation, _warning = resolve_flux2_sampler(sampler)
        batch, _channels, latent_height, latent_width = latents.shape
        latents = latents.to(device=device, dtype=torch.float32)
        embeddings = embeddings.to(device=device, dtype=self.dtype)
        if embeddings.shape[0] == 1 and batch > 1:
            embeddings = embeddings.expand(batch, -1, -1)
        # Positions stay float32: RoPE upcasts them anyway, and a BF16 index stops being exact past
        # 256, which a 4096-edge canvas reaches on the text axis alone.
        image_ids = flux2_image_ids(latent_height, latent_width, device, torch.float32)
        text_ids = flux2_text_ids(embeddings.shape[1], device, torch.float32)
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
        noise_generators = list(generators)
        sample_steps = len(sigmas) - 1
        invocations = 0
        previous_prediction = None

        def predict(state, sigma):
            nonlocal invocations
            invocations += 1
            timestep = torch.full((batch,), float(sigma), device=device, dtype=torch.float32)
            output = self.transformer(
                hidden_states=pack_flux2_latents(state.to(self.dtype)),
                encoder_hidden_states=embeddings,
                timestep=timestep,
                img_ids=image_ids,
                txt_ids=text_ids,
                guidance=guidance_tensor,
                return_dict=False,
            )[0]
            return unpack_flux2_latents(output.float(), latent_height, latent_width)

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
            raise RuntimeError("Native Flux2 generation requires CUDA")
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native Flux2 BF16 execution requires a BF16-capable CUDA device")

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
            raise ValueError("FLUX.2 has no unconditional branch; the negative prompt must be empty")
        self._validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step)
        self._require_cuda()
        del pag_scale, pag_applied_layers

        shift = flux2_resolution_shift(width, height, steps)
        sigmas = flux2_sigma_schedule(steps, scheduler, shift)
        embeddings = self._run_cuda_stage("prompt_encode", lambda: self._encode_prompt(prompt))
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
                    embeddings, initial, sigmas, sampler, cfg, chunk,
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
            raise ValueError("FLUX.2 has no unconditional branch; the negative prompt must be empty")
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

        shift = flux2_resolution_shift(width, height, steps)
        sigmas, schedule_diagnostics = flux2_refinement_sigma_schedule(steps, float(denoise), scheduler, shift)
        embeddings = self._run_cuda_stage("refinement.prompt_encode", lambda: self._encode_prompt(prompt))
        source = self._run_cuda_stage("refinement.vae_encode", lambda: self._encode_images(images))
        noise = _cpu_noise_like_batch(source, generators)
        start_sigma = float(sigmas[0].item())
        initial = (1.0 - start_sigma) * source + start_sigma * noise
        latents = self._run_cuda_stage(
            "refinement.sampling",
            lambda: self._sample(
                embeddings, initial, sigmas, sampler, cfg, list(generators),
                on_step, on_step_checkpoint,
                source_latents=source, source_noise=noise, latent_mask=mask_tensor,
            ),
        )
        self._record_sampling_metrics("refinement.sampling", len(sigmas) - 1, schedule_diagnostics, shift)
        return self._run_cuda_stage(
            "refinement.vae_decode", lambda: self._decode(latents, force_tiled_decode=force_tiled_decode)
        )


def flux2_component_bytes(paths) -> int:
    """Host memory a FLUX.2 load will hold, read from the files' headers rather than their size."""
    return flux_component_bytes(paths)


def load_flux2_runtime(
    diffusion_model,
    text_encoder,
    vae,
    qwen_tokenizer_path=None,
    dtype: torch.dtype = torch.bfloat16,
    loras=(),
) -> Flux2Runtime:
    """Load the three ComfyUI component files into one runtime.

    The text encoder is classified from its own tensors, so a [dev] Mistral file and a [klein]
    Qwen3 file both mount through the same picker without the user declaring which is which.
    """
    deps = _runtime_dependencies()
    diffusion_path = _require_safetensors(diffusion_model, "Flux2 diffusion model")
    encoder_path = _require_safetensors(text_encoder, "Flux2 text encoder")
    vae_path = _require_safetensors(vae, "Flux2 VAE")

    encoder_state = deps["load_file"](str(encoder_path), device="cpu")
    family = classify_flux2_text_encoder(encoder_state.keys())
    if family == "unknown":
        raise ValueError(f"{encoder_path.name} is not a FLUX.2 text encoder")
    tokenizer = _load_flux2_tokenizer(family, encoder_state, encoder_path, qwen_tokenizer_path, deps)
    language_model, family = _load_flux2_text_encoder(encoder_state, dtype, deps)
    del encoder_state
    flux2_vae, statistics = _load_flux2_vae(vae_path, dtype, deps)
    transformer, config, load_report = _load_flux2_transformer(diffusion_path, dtype, deps, loras=loras)

    conditioning_width = int(config["joint_attention_dim"])
    expected = FLUX2_TAP_COUNT * int(language_model.config.hidden_size)
    if conditioning_width != expected:
        raise ValueError(
            f"This FLUX.2 diffusion model expects a {conditioning_width}-wide conditioning, but the "
            f"selected {family} text encoder produces {expected}; pair a [dev] model with its Mistral "
            "encoder and a [klein] model with its Qwen3 encoder"
        )

    runtime = Flux2Runtime(
        transformer, language_model, flux2_vae, tokenizer, dtype, config, family, statistics
    )
    runtime.lora_report = load_report["loras"]
    runtime.unclaimed_tensors = load_report["unclaimed_tensors"]
    return runtime
