"""Native Krea 2 (K2) runtime built on the ComfyUI component layout.

ComfyUI runs Krea 2 from three files — a diffusion model, one text encoder and a VAE — loaded by
``UNETLoader``, ``CLIPLoader(type=krea2)`` and ``VAELoader``.  This module reproduces that graph.
It duck-types :class:`flux2_pipeline.Flux2Runtime` for the same reason that module duck-types
:class:`anima_pipeline.AnimaRuntime`: the server drives every native engine through one set of
staged-load, group-offload and refinement call sites.

What makes Krea 2 different from the two Flux engines beside it:

* **It is not guidance distilled.**  FLUX.1 and FLUX.2 steer with a scalar folded into the timestep
  embedding and run one forward per step.  ``model_base.Krea2`` carries no guidance conditioning at
  all, so Krea 2 runs classifier-free guidance the ordinary way: a conditioned and an unconditioned
  forward per step, a real negative prompt, and — because there is an unconditional branch to
  rescale against — CFG-Zero*.
* **The text encoder is read at twelve depths, not three.**  Qwen3-VL-4B's hidden states at layers
  2, 5, 8 ... 35 are stacked into ``(B, seq, 12, 2560)`` and handed to an adapter *inside* the
  transformer (``txtfusion``) rather than being concatenated on the channel axis the way FLUX.2's
  three taps are.  ComfyUI flattens the tap axis into the feature dimension to move the tensor
  through its 3D conditioning contract; so does this, and the model unpacks it again.
* **The autoencoder is a video autoencoder.**  Krea 2's latent format is ``latent_formats.Wan21``:
  Wan 2.1's causal 3D VAE, 16 channels at stride 8, with a frame axis.  A still image is one frame,
  so the frame axis is added at the autoencoder boundary and dropped immediately after; everything
  between holds four-dimensional latents like the other engines.
* **The transformer is ours.**  Diffusers has no Krea 2 implementation, so
  :mod:`krea2_model` ports ComfyUI's ``SingleStreamDiT`` and the checkpoint loads in ComfyUI's own
  key naming — no conversion table, and a LoRA published against ``diffusion_model.blocks.N.``
  applies directly.

The schedule is :mod:`krea2_sampling`, which is FLUX.1's rectified-flow sigma table at the static
shift Krea 2's own model config declares.
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
        _load_diffusion_state_dict,
        _module_nbytes,
        _require_diffusion_weights,
        _require_safetensors,
        _strict_assign,
        euler_ancestral_rf_step,
        euler_rf_step,
    )
    from .flux_pipeline import (
        COMFY_VAE_OVERLAP_PIXELS,
        COMFY_VAE_TILE_PIXELS,
        _count_blocks,
        _decoded_tensor_to_image,
        _lora_descriptors,
        _scale_diffusers_lora_alphas,
        _shape_of,
        flux_component_bytes,
        fuse_flux_lora_state_dict,
        normalize_flux_checkpoint_keys,
        resolve_quantized_state_dict,
        unmatched_lora_targets,
    )
    from .krea2_model import (
        KREA2_TEXT_FUSION_LAYERWISE_BLOCKS,
        KREA2_TEXT_FUSION_REFINER_BLOCKS,
        Krea2Transformer2DModel,
    )
    from .gguf_loader import GGUF_SUFFIX
    from .quantized_linear import (
        QuantizedLinear,
        UnsupportedQuantization,
        apply_quantized_linears,
        checkpoint_runtime_bytes,
        combine_lora_adapters,
        expand_quantized_layers,
        logical_shape_view,
        quantized_state_dict,
        scan_quantized_layers,
    )
    from .krea2_sampling import (
        KREA2_SAMPLERS,
        KREA2_SCHEDULERS,
        KREA2_SHIFT,
        krea2_refinement_sigma_schedule,
        krea2_sigma_schedule,
        resolve_krea2_sampler,
    )
except ImportError:
    from anima_pipeline import (
        _cast_state_dict,
        _cpu_noise_like_batch,
        _discard_module_storage,
        _empty_cuda_cache,
        _load_diffusion_state_dict,
        _module_nbytes,
        _require_diffusion_weights,
        _require_safetensors,
        _strict_assign,
        euler_ancestral_rf_step,
        euler_rf_step,
    )
    from flux_pipeline import (
        COMFY_VAE_OVERLAP_PIXELS,
        COMFY_VAE_TILE_PIXELS,
        _count_blocks,
        _decoded_tensor_to_image,
        _lora_descriptors,
        _scale_diffusers_lora_alphas,
        _shape_of,
        flux_component_bytes,
        fuse_flux_lora_state_dict,
        normalize_flux_checkpoint_keys,
        resolve_quantized_state_dict,
        unmatched_lora_targets,
    )
    from krea2_model import (
        KREA2_TEXT_FUSION_LAYERWISE_BLOCKS,
        KREA2_TEXT_FUSION_REFINER_BLOCKS,
        Krea2Transformer2DModel,
    )
    from gguf_loader import GGUF_SUFFIX
    from quantized_linear import (
        QuantizedLinear,
        UnsupportedQuantization,
        apply_quantized_linears,
        checkpoint_runtime_bytes,
        combine_lora_adapters,
        expand_quantized_layers,
        logical_shape_view,
        quantized_state_dict,
        scan_quantized_layers,
    )
    from krea2_sampling import (
        KREA2_SAMPLERS,
        KREA2_SCHEDULERS,
        KREA2_SHIFT,
        krea2_refinement_sigma_schedule,
        krea2_sigma_schedule,
        resolve_krea2_sampler,
    )


# `model_detection.py` reads Krea 2's head count by dividing the query projection by a fixed 128.
# The width is not stored anywhere in the checkpoint, so this constant is part of the format.
KREA2_ATTENTION_HEAD_DIM = 128

# Wan 2.1's autoencoder: 16 channels at stride 8, with the transformer packing 2x2 on top.
KREA2_VAE_LATENT_CHANNELS = 16
KREA2_VAE_SCALE_FACTOR = 8
KREA2_LATENT_PATCH = 2
KREA2_PIXEL_ALIGNMENT = KREA2_VAE_SCALE_FACTOR * KREA2_LATENT_PATCH
KREA2_MAX_EDGE = 4096
# Where tiled decoding starts. Measured to be the faster path above it as well as the smaller one.
KREA2_TILED_DECODE_EDGE = 1536


# `comfy/latent_formats.py::Wan21`. `scale_factor` is 1.0, so `process_in` is a plain
# (latent - mean) / std and `process_out` its inverse.
KREA2_LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
KREA2_LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)

# `comfy/text_encoders/krea2.py::KREA2_TAP_LAYERS`: tap k is hidden_states[k], no offset. The
# deepest tap is 35, so the language model must have at least 36 layers.
KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TAP_COUNT = len(KREA2_TAP_LAYERS)

# `comfy/text_encoders/krea2.py::KREA2_TEMPLATE`, verbatim. `Krea2Tokenizer` passes
# `thinking=True`, which is what suppresses the empty `<think>` block Qwen3's tokenizer would
# otherwise append; the template below is therefore the complete encoded text.
KREA2_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
# Token ids the prefix strip walks: `<|im_start|>`, then `user`, then a newline.
KREA2_IM_START_TOKEN = 151644
KREA2_USER_TOKEN = 872
KREA2_NEWLINE_TOKEN = 198

# Krea 2 pads nothing — `Qwen3VLSDTokenizer` sets `pad_to_max_length=False` and `min_length=1`, so
# a prompt occupies exactly its own tokens and no attention mask is ever needed. The ceiling below
# only exists so a pathological prompt fails with a sentence instead of an allocation.
KREA2_MAX_TEXT_TOKENS = 4096

# Qwen3-VL-4B, as `comfy/text_encoders/llama.py::Qwen3VL_4BConfig` declares it. Every dimension
# that can be read off the checkpoint is; these are the ones that leave no trace in the tensors.
KREA2_QWEN3VL_CONFIG = {
    "rope_theta": 5000000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 262144,
    "attention_dropout": 0.0,
    "use_cache": False,
    "tie_word_embeddings": True,
    # `interleaved_mrope = True` with sections (24, 20, 20). For text-only conditioning all three
    # position axes carry the same ids, so the interleave resolves to ordinary RoPE — but the
    # sections still have to be declared or the rotary embedding cannot be constructed.
    "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True},
}

# HuggingFace naming inside a published Qwen3-VL checkpoint. `comfy/sd.py` rewrites exactly these
# three prefixes before handing the file to its own loader.
KREA2_TEXT_ENCODER_LANGUAGE_PREFIX = "model.language_model."
KREA2_TEXT_ENCODER_DISCARDED_PREFIXES = ("model.visual.", "visual.", "lm_head.", "model.lm_head.")

# The Wan 2.1 autoencoder's fingerprint, and the Wan 2.2 tensor that must be absent. Both are the
# tests `comfy/sd.py::VAE.__init__` makes.
KREA2_VAE_MARKER = "decoder.middle.0.residual.0.gamma"
KREA2_WAN22_VAE_MARKER = "decoder.upsamples.0.upsamples.0.residual.2.weight"

# Wan 2.1's fixed block structure, which no tensor in either layout records. `AutoencoderKLWan`
# defaults to exactly these values; restating them means a Diffusers default change cannot
# silently alter what a Krea 2 checkpoint loads into.
KREA2_VAE_GEOMETRY = {
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "is_residual": False,
    "patch_size": None,
}


def _runtime_dependencies():
    try:
        import diffusers
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKLWan
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        from safetensors.torch import load_file
        from transformers import PreTrainedTokenizerFast, Qwen3VLTextModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Native Krea2 requires accelerate, safetensors, transformers with Qwen3-VL, and "
            "diffusers 0.38 with AutoencoderKLWan"
        ) from error
    if not str(diffusers.__version__).startswith("0.38."):
        raise RuntimeError(f"Native Krea2 requires diffusers 0.38.x; found {diffusers.__version__}")
    return {
        "init_empty_weights": init_empty_weights,
        "AutoencoderKLWan": AutoencoderKLWan,
        "convert_wan_vae": convert_wan_vae_to_diffusers,
        "load_file": load_file,
        "Qwen3VLTextConfig": Qwen3VLTextConfig,
        "Qwen3VLTextModel": Qwen3VLTextModel,
        "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
    }


# -- transformer ------------------------------------------------------------------------------


def _swiglu_hidden(features: int, multiplier: int, multiple: int = 128) -> int:
    hidden = int(2 * features / 3) * multiplier
    return multiple * ((hidden + multiple - 1) // multiple)


def _infer_swiglu_multiplier(features: int, hidden: int, label: str) -> int:
    """Recover the SwiGLU multiplier the checkpoint was built with.

    ``SwiGLU`` rounds ``int(2 * features / 3) * multiplier`` up to a multiple of 128, so the width
    alone does not name the multiplier.  ComfyUI sidesteps this by leaving the constructor default
    in place; reading it back instead is what lets a fixture small enough for a unit test load
    through the same path a 6144-wide checkpoint does.
    """
    for multiplier in range(1, 17):
        if _swiglu_hidden(features, multiplier) == hidden:
            return multiplier
    raise ValueError(
        f"{label} has a {hidden}-wide SwiGLU that no multiplier between 1 and 16 reproduces "
        f"for a {features}-wide model"
    )


def infer_krea2_transformer_config(state_dict: Mapping[str, torch.Tensor]) -> dict:
    """Derive the transformer configuration from the checkpoint itself.

    ``model_detection.py`` recognises Krea 2 by ``txtfusion.projector.weight`` — the learned
    collapse of the twelve encoder taps, which no other architecture has — so that tensor is the
    gate here too.  Everything ComfyUI reads is read the same way; the two values it leaves at
    their constructor defaults (the MLP multiplier and the RoPE base) are recovered from the
    tensors where that is possible and pinned where it is not.
    """
    keys = list(state_dict.keys())
    label = "Krea 2 diffusion checkpoint"
    if "txtfusion.projector.weight" not in state_dict:
        raise ValueError(
            "The selected file is not a Krea 2 diffusion model: it carries no txtfusion.projector "
            "tensor collapsing the text-encoder taps"
        )
    layers = _count_blocks(keys, "blocks.", label)
    if layers < 1:
        raise ValueError("The selected file is not a Krea 2 diffusion model: it has no blocks stack")

    first = _shape_of(state_dict, "first.weight", label)
    features = int(first[0])
    patch = KREA2_LATENT_PATCH
    if int(first[1]) % (patch * patch):
        raise ValueError(f"{label} has a {int(first[1])}-wide input projection that is not a {patch}x{patch} patch")
    channels = int(first[1]) // (patch * patch)

    query = _shape_of(state_dict, "blocks.0.attn.wq.weight", label)
    key_projection = _shape_of(state_dict, "blocks.0.attn.wk.weight", label)
    if int(query[0]) % KREA2_ATTENTION_HEAD_DIM or int(key_projection[0]) % KREA2_ATTENTION_HEAD_DIM:
        raise ValueError(f"{label} attention projections are not multiples of {KREA2_ATTENTION_HEAD_DIM}")
    heads = int(query[0]) // KREA2_ATTENTION_HEAD_DIM
    kvheads = int(key_projection[0]) // KREA2_ATTENTION_HEAD_DIM

    projector = _shape_of(state_dict, "txtfusion.projector.weight", label)
    txtlayers = int(projector[1])
    if txtlayers != KREA2_TAP_COUNT:
        raise ValueError(
            f"{label} collapses {txtlayers} text-encoder taps, but Krea 2 conditioning is a "
            f"{KREA2_TAP_COUNT}-layer Qwen3-VL stack"
        )
    txtdim = int(_shape_of(state_dict, "txtfusion.layerwise_blocks.0.prenorm.scale", label)[0])
    txt_query = _shape_of(state_dict, "txtfusion.layerwise_blocks.0.attn.wq.weight", label)
    txt_key = _shape_of(state_dict, "txtfusion.layerwise_blocks.0.attn.wk.weight", label)
    txtheads = int(txt_query[0]) // KREA2_ATTENTION_HEAD_DIM
    txtkvheads = int(txt_key[0]) // KREA2_ATTENTION_HEAD_DIM
    if txtheads < 1 or txtkvheads < 1:
        raise ValueError(f"{label} text-fusion attention is narrower than one {KREA2_ATTENTION_HEAD_DIM}-wide head")

    tdim = int(_shape_of(state_dict, "tmlp.0.weight", label)[1])
    multiplier = _infer_swiglu_multiplier(features, int(_shape_of(state_dict, "blocks.0.mlp.gate.weight", label)[0]), label)
    text_multiplier = _infer_swiglu_multiplier(
        txtdim, int(_shape_of(state_dict, "txtfusion.layerwise_blocks.0.mlp.gate.weight", label)[0]), label
    )
    if text_multiplier != multiplier:
        raise ValueError(
            f"{label} uses SwiGLU multiplier {multiplier} in its blocks but {text_multiplier} in its "
            "text fusion; Krea 2 shares one multiplier across both"
        )
    return {
        "features": features,
        "tdim": tdim,
        "txtdim": txtdim,
        "heads": heads,
        "kvheads": kvheads,
        "multiplier": multiplier,
        "layers": layers,
        "patch": patch,
        "channels": channels,
        # `bias=False` everywhere in the published model; read rather than assumed so a future
        # biased variant fails on the tensors instead of on a silent shape mismatch.
        "bias": "blocks.0.attn.wq.bias" in state_dict,
        # `theta` leaves no trace in the weights: `SingleStreamDiT` takes it as a constructor
        # default and no checkpoint records it.
        "theta": 1000,
        "txtlayers": txtlayers,
        "txtheads": txtheads,
        "txtkvheads": txtkvheads,
    }


def krea2_lora_key_map(config: Mapping) -> dict[str, str]:
    """``comfy/utils.py::krea2_to_diffusers`` — Diffusers-style LoRA names to this model's weights.

    The transformer keeps ComfyUI's naming, so a LoRA already written against
    ``diffusion_model.blocks.N.attn.wq`` needs no map at all.  Published tooling emits the
    Diffusers spelling instead, and this is the table that accepts it.
    """
    block_map = {
        "attn.to_q": "attn.wq",
        "attn.to_k": "attn.wk",
        "attn.to_v": "attn.wv",
        "attn.to_gate": "attn.gate",
        "attn.to_out.0": "attn.wo",
        # Some tooling drops the `.0` from `to_out`; ComfyUI accepts both spellings.
        "attn.to_out": "attn.wo",
        "ff.gate": "mlp.gate",
        "ff.up": "mlp.up",
        "ff.down": "mlp.down",
    }
    key_map = {}

    def add_block(target: str, origin: str) -> None:
        for diffusers_name, comfy_name in block_map.items():
            key_map[f"{target}.{diffusers_name}"] = f"{origin}.{comfy_name}"

    for index in range(int(config["layers"])):
        add_block(f"transformer_blocks.{index}", f"blocks.{index}")
    for index in range(KREA2_TEXT_FUSION_LAYERWISE_BLOCKS):
        add_block(f"text_fusion.layerwise_blocks.{index}", f"txtfusion.layerwise_blocks.{index}")
    for index in range(KREA2_TEXT_FUSION_REFINER_BLOCKS):
        add_block(f"text_fusion.refiner_blocks.{index}", f"txtfusion.refiner_blocks.{index}")
    key_map.update({
        "img_in": "first",
        "time_embed.linear_1": "tmlp.0",
        "time_embed.linear_2": "tmlp.2",
        "time_mod_proj": "tproj.1",
        "txt_in.linear_1": "txtmlp.1",
        "txt_in.linear_2": "txtmlp.3",
        "text_fusion.projector": "txtfusion.projector",
        "final_layer.linear": "last.linear",
    })
    return key_map


def convert_krea2_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor], config: Mapping, label: str
) -> dict[str, torch.Tensor]:
    """Bring a published Krea 2 LoRA into this model's ``lora_A``/``lora_B`` naming.

    Three spellings reach here: kohya's ``lora_down``/``lora_up``, Diffusers' ``lora_A``/``lora_B``
    under either the Diffusers or the ComfyUI module names, and the ``diffusion_model.``/
    ``transformer.``/``lycoris_`` prefixes ``comfy/lora.py`` accepts for each.
    """
    if any("dora_scale" in key for key in state_dict):
        raise ValueError(f"{label} is a DoRA checkpoint, which this runtime cannot fuse")

    key_map = krea2_lora_key_map(config)
    # `comfy/lora.py` registers each Diffusers target under four spellings; the lycoris one flattens
    # every dot in the name, so it is matched by flattening the candidate the same way.
    lookup = {}
    for diffusers_name, comfy_name in key_map.items():
        for prefix in ("", "diffusion_model.", "transformer."):
            lookup[f"{prefix}{diffusers_name}"] = comfy_name
        lookup[f"lycoris_{diffusers_name.replace('.', '_')}"] = comfy_name
    # A LoRA already written in ComfyUI's own naming maps to itself.
    for comfy_name in set(key_map.values()):
        for prefix in ("", "diffusion_model.", "transformer."):
            lookup.setdefault(f"{prefix}{comfy_name}", comfy_name)

    normalized = {}
    for key, value in state_dict.items():
        for source, target in (
            (".lora_down.weight", ".lora_A.weight"),
            (".lora_up.weight", ".lora_B.weight"),
            (".lora_A.weight", ".lora_A.weight"),
            (".lora_B.weight", ".lora_B.weight"),
            (".alpha", ".alpha"),
        ):
            if key.endswith(source):
                normalized[key[: -len(source)] + target] = value
                break

    converted = {}
    unknown = set()
    for key, value in normalized.items():
        suffix = ".alpha" if key.endswith(".alpha") else key[key.rindex(".lora_") :]
        module = key[: len(key) - len(suffix)]
        target = lookup.get(module) or lookup.get(module.replace("base_model.model.", ""))
        if target is None:
            unknown.add(module)
            continue
        converted[f"{target}{suffix}"] = value
    if not converted:
        detail = f" (unrecognised targets: {', '.join(sorted(unknown)[:4])})" if unknown else ""
        raise ValueError(f"{label} is not a recognised Krea 2 LoRA layout{detail}")
    # kohya folds alpha into neither matrix, so every layout is pre-scaled the same way here.
    return _scale_diffusers_lora_alphas(converted, label)


def _load_krea2_transformer(path: Path, dtype: torch.dtype, deps, loras=(), state_dict=None):
    """Build the transformer, keeping every quantised weight in the form the file stores it in.

    Expanding the quantisation here instead — which is what this did — cost 78 seconds of CPU
    arithmetic and produced a 23.9 GiB model that cannot be resident on a 24 GB card, so every
    step then streamed blocks across PCIe.  The scales are applied inside the linear instead, on
    the same numbers: see :mod:`quantized_linear`.
    """
    if state_dict is None:
        state_dict = _load_diffusion_state_dict(path, dtype, deps, "Krea 2 diffusion model")
    state_dict = normalize_flux_checkpoint_keys(state_dict)

    quantized = scan_quantized_layers(state_dict, "Krea 2 diffusion model")
    config = infer_krea2_transformer_config(logical_shape_view(state_dict, quantized))

    lora_states = []
    for lora_path, multiplier in _lora_descriptors(loras, "Krea2 LoRA"):
        label = f"Krea2 LoRA {lora_path.name}"
        lora_state = convert_krea2_lora_state_dict(
            deps["load_file"](str(lora_path), device="cpu"), config, label
        )
        lora_states.append((lora_path, multiplier, label, lora_state))

    with deps["init_empty_weights"]():
        transformer = Krea2Transformer2DModel(**config)
    kept, deferred = apply_quantized_linears(
        transformer, quantized, state_dict, dtype, "Krea 2 diffusion model"
    )
    replaced = set(kept)
    if deferred:
        state_dict = expand_quantized_layers(state_dict, quantized, set(deferred), dtype)
    state_dict = quantized_state_dict(state_dict, quantized, replaced)

    # The transformer keeps ComfyUI's naming, so there is no conversion table to claim tensors
    # through — the module's own key set is what separates the model from anything a publisher
    # bundled beside it. Missing keys still fail the strict load below; extra ones are reported.
    expected = set(transformer.state_dict())
    weights = {}
    unclaimed = []
    for key, value in state_dict.items():
        if key in expected:
            weights[key] = value
        else:
            unclaimed.append(key)
    del state_dict
    # A quantised weight is already in the dtype its module declares; casting is for the rest.
    weights = _cast_quantized_state_dict(weights, replaced, dtype)

    # A LoRA reaches a full-precision layer by being fused into its weight, which is the cheaper
    # arrangement and what the other engines do. A quantised layer takes it as an adapter instead:
    # fusing would have to round the patched weight back into fp8, and expanding the layer to avoid
    # that expanded nearly the whole transformer, because a Krea 2 LoRA names nearly every linear.
    lora_report = []
    adapters: dict[str, list] = {}
    for lora_path, multiplier, label, lora_state in lora_states:
        skipped = unmatched_lora_targets(lora_state, weights)
        fusable, adapted = _split_lora_by_storage(lora_state, replaced)
        patched = fuse_flux_lora_state_dict(weights, fusable, multiplier, label) if fusable else 0
        for target, (down, up) in adapted.items():
            adapters.setdefault(target, []).append((down, up, multiplier))
        lora_report.append({
            "name": lora_path.name,
            "multiplier": multiplier,
            "patched_modules": patched + len(adapted),
            "fused_modules": patched,
            "adapted_modules": len(adapted),
            "skipped_modules": skipped,
        })
    del lora_states

    _strict_assign(transformer, weights, "Krea 2 transformer")
    del weights
    for target, stack in adapters.items():
        module = transformer.get_submodule(target)
        module.set_lora_adapter(*combine_lora_adapters(stack, dtype))
    return (
        transformer.eval().requires_grad_(False),
        config,
        {
            "loras": lora_report,
            "unclaimed_tensors": sorted(unclaimed),
            "quantized_layers": sorted(replaced),
            "adapted_layers": sorted(adapters),
            "quantization": sorted({spec.format for layer, spec in quantized.items() if layer in replaced}),
        },
    )


def _split_lora_by_storage(lora_state, quantized_layers):
    """Separate a converted LoRA into the part that can be fused and the part that cannot.

    Returns ``(fusable_state, {target: (down, up)})``. A pair whose target is a quantised layer is
    handed back whole rather than added into a weight, because that weight has nowhere to keep the
    sum at full precision.
    """
    fusable, adapted = {}, {}
    for key, value in lora_state.items():
        if key.endswith(".lora_A.weight"):
            target = key[: -len(".lora_A.weight")]
        elif key.endswith(".lora_B.weight"):
            target = key[: -len(".lora_B.weight")]
        else:
            fusable[key] = value
            continue
        if target in quantized_layers:
            down = lora_state.get(f"{target}.lora_A.weight")
            up = lora_state.get(f"{target}.lora_B.weight")
            if down is None or up is None:
                raise ValueError(f"Krea2 LoRA has an unpaired tensor for {target}")
            adapted[target] = (down, up)
        else:
            fusable[key] = value
    return fusable, adapted


def _cast_quantized_state_dict(weights, quantized_layers, dtype: torch.dtype):
    """Cast everything to the compute dtype except the tensors a quantised layer owns.

    A quantised weight and its scale are the two things that must keep their stored types: casting
    the weight would undo the point, and casting the scale to bfloat16 changes the result for most
    tensors — measurably, unlike deferring the multiply, which changes none.
    """
    protected = set()
    for layer in quantized_layers:
        protected.add(f"{layer}.weight")
        protected.add(f"{layer}.scale_weight")
        protected.add(f"{layer}.scale_weight_2")
    return {
        key: value if key in protected else (
            value.to(dtype=dtype) if value.is_floating_point() and value.dtype != dtype else value
        )
        for key, value in weights.items()
    }


# -- text encoder -----------------------------------------------------------------------------


def krea2_text_encoder_weights(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The language-model half of a published Qwen3-VL checkpoint, in Diffusers-free naming.

    ``comfy/sd.py`` rewrites ``model.language_model.`` to ``model.`` and keeps the vision tower
    beside it because ComfyUI also offers Krea 2's encoder for multimodal prompt generation.  This
    runtime only conditions, and conditioning reads hidden states, so the vision tower and the
    language-model head are dropped rather than loaded and never evaluated.
    """
    weights = {}
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in KREA2_TEXT_ENCODER_DISCARDED_PREFIXES):
            continue
        name = key
        if name.startswith(KREA2_TEXT_ENCODER_LANGUAGE_PREFIX):
            name = name[len(KREA2_TEXT_ENCODER_LANGUAGE_PREFIX) :]
        elif name.startswith("model."):
            name = name[len("model.") :]
        weights[name] = value
    return weights


def classify_krea2_text_encoder(keys) -> str:
    """Tell a Qwen3-VL text encoder from anything else by its per-head QK norms."""
    weights = krea2_text_encoder_weights({key: None for key in keys})
    if "layers.0.post_attention_layernorm.weight" not in weights:
        return "unknown"
    # Qwen3 normalises queries and keys per head; the Qwen2.x and Mistral encoders do not.
    return "qwen3vl" if "layers.0.self_attn.q_norm.weight" in weights else "unknown"


def infer_krea2_text_encoder_config(weights: Mapping[str, torch.Tensor]) -> dict:
    """Read Qwen3-VL-4B's geometry off its own tensors."""
    label = "Krea 2 text encoder"
    layers = _count_blocks(weights.keys(), "layers.", label)
    if layers <= max(KREA2_TAP_LAYERS):
        raise ValueError(
            f"{label} has {layers} layers, too few for Krea 2's {KREA2_TAP_COUNT}-layer "
            f"conditioning tap (the deepest is layer {max(KREA2_TAP_LAYERS)})"
        )
    embedding = _shape_of(weights, "embed_tokens.weight", label)
    query = _shape_of(weights, "layers.0.self_attn.q_proj.weight", label)
    key_projection = _shape_of(weights, "layers.0.self_attn.k_proj.weight", label)
    mlp = _shape_of(weights, "layers.0.mlp.gate_proj.weight", label)
    head_dim = int(_shape_of(weights, "layers.0.self_attn.q_norm.weight", label)[0])
    if int(query[0]) % head_dim or int(key_projection[0]) % head_dim:
        raise ValueError(f"{label} attention projections are not multiples of its {head_dim}-wide head")
    config = dict(KREA2_QWEN3VL_CONFIG)
    config.update(
        vocab_size=int(embedding[0]),
        hidden_size=int(embedding[1]),
        intermediate_size=int(mlp[0]),
        num_hidden_layers=layers,
        num_attention_heads=int(query[0]) // head_dim,
        num_key_value_heads=int(key_projection[0]) // head_dim,
        head_dim=head_dim,
    )
    return config


def _load_krea2_text_encoder(state_dict, dtype: torch.dtype, deps):
    if classify_krea2_text_encoder(state_dict.keys()) == "unknown":
        raise ValueError(
            "The selected file is not a Krea 2 text encoder: it carries no Qwen3-VL decoder layers"
        )
    weights = krea2_text_encoder_weights(state_dict)
    quantized = scan_quantized_layers(weights, "Krea 2 text encoder")
    config = infer_krea2_text_encoder_config(logical_shape_view(weights, quantized))
    model_config = deps["Qwen3VLTextConfig"](**config)
    with deps["init_empty_weights"]():
        encoder = deps["Qwen3VLTextModel"](model_config)
    # Qwen3-VL's fp8 repackage is quantised the same way the transformer is, and expanding it cost
    # 11 seconds and 6.8 GiB that the card then has to hold beside the diffusion model.
    kept, deferred = apply_quantized_linears(encoder, quantized, weights, dtype, "Krea 2 text encoder")
    replaced = set(kept)
    if deferred:
        # A quantised embedding has no matmul to fold the scale into, so it is expanded as before.
        weights = expand_quantized_layers(weights, quantized, set(deferred), dtype)
    weights = quantized_state_dict(weights, quantized, replaced)
    _strict_assign(encoder, _cast_quantized_state_dict(weights, replaced, dtype), "Krea 2 text encoder")
    encoder.config.use_cache = False
    return encoder.eval().requires_grad_(False)


# -- autoencoder ------------------------------------------------------------------------------


def krea2_vae_layout(state_dict) -> str:
    """Which naming a Krea 2 autoencoder file is written in.

    ``wan`` is what ComfyUI and the Wan release publish — the layout ``comfy/sd.py::VAE`` reads
    directly.  ``diffusers`` is what a file exported from ``AutoencoderKLWan`` carries; it needs no
    conversion, and accepting it costs one branch and removes a dead end for anyone who already
    has one.
    """
    keys = set(state_dict)
    if any(key.startswith("decoder.up_blocks.") for key in keys):
        return "diffusers"
    if KREA2_VAE_MARKER in keys:
        return "wan"
    return "unknown"


def infer_krea2_vae_config(state_dict: Mapping[str, torch.Tensor]) -> dict:
    """Read the autoencoder's geometry the way ``comfy/sd.py::VAE`` does.

    The four widths below are the only ones either layout records.  Wan 2.1's block structure —
    the channel ladder, the residual depth and which stages halve time — leaves no trace in the
    tensors, so it comes from :data:`KREA2_VAE_GEOMETRY` rather than being guessed.
    """
    label = "Krea 2 autoencoder"
    layout = krea2_vae_layout(state_dict)
    if layout == "unknown":
        raise ValueError(
            "The selected file is not a Krea 2 autoencoder: it carries no Wan 2.1 residual middle block"
        )
    if layout == "wan" and KREA2_WAN22_VAE_MARKER in state_dict:
        raise ValueError(
            "The selected file is a Wan 2.2 autoencoder; Krea 2's latent format is Wan 2.1 "
            "(16 channels at stride 8)"
        )
    if layout == "wan":
        base_dim_key, latent_key = "decoder.head.0.gamma", "encoder.head.2.weight"
        input_key, output_key = "encoder.conv1.weight", "decoder.head.2.weight"
    else:
        base_dim_key, latent_key = "decoder.norm_out.gamma", "encoder.conv_out.weight"
        input_key, output_key = "encoder.conv_in.weight", "decoder.conv_out.weight"
    latent_pair = int(_shape_of(state_dict, latent_key, label)[0])
    if latent_pair % 2:
        raise ValueError(f"{label} emits {latent_pair} channels, which is not a mean/log-variance pair")
    return {
        **KREA2_VAE_GEOMETRY,
        "base_dim": int(_shape_of(state_dict, base_dim_key, label)[0]),
        "z_dim": latent_pair // 2,
        "in_channels": int(_shape_of(state_dict, input_key, label)[1]),
        "out_channels": int(_shape_of(state_dict, output_key, label)[0]),
    }


def _load_krea2_vae(path: Path, dtype: torch.dtype, deps):
    state_dict = deps["load_file"](str(path), device="cpu")
    state_dict = resolve_quantized_state_dict(state_dict, dtype, "Krea 2 autoencoder")
    config = infer_krea2_vae_config(state_dict)
    if krea2_vae_layout(state_dict) == "diffusers":
        converted = dict(state_dict)
    else:
        try:
            converted = deps["convert_wan_vae"](dict(state_dict))
        except Exception as error:
            raise RuntimeError(f"Failed to convert the Krea 2 autoencoder checkpoint: {error}") from error
    del state_dict
    with deps["init_empty_weights"]():
        vae = deps["AutoencoderKLWan"](**config)
    _strict_assign(vae, _cast_state_dict(converted, dtype), "Krea 2 autoencoder")
    return vae.eval().requires_grad_(False), config


def _load_krea2_tokenizer(qwen_tokenizer_path, deps):
    path = Path(qwen_tokenizer_path) if qwen_tokenizer_path else None
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "The Krea 2 Qwen tokenizer resource is missing; run the environment configurator to restore it"
        )
    tokenizer = deps["PreTrainedTokenizerFast"](tokenizer_file=str(path), local_files_only=True)
    if tokenizer.vocab_size < 1:
        raise RuntimeError("The Krea 2 Qwen tokenizer.json does not declare a usable vocabulary")
    return tokenizer


class Krea2Runtime:
    """A loaded Krea 2 model, driven through the surface every native engine exposes."""

    def __init__(self, transformer, text_encoder, vae, tokenizer, dtype, config, vae_config):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = vae
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.config = dict(config)
        self.vae_config = dict(vae_config)
        self.tap_layers = KREA2_TAP_LAYERS
        self.prompt_template = KREA2_TEMPLATE
        latent_channels = int(vae_config["z_dim"])
        self._latent_mean = torch.tensor(KREA2_LATENTS_MEAN[:latent_channels], dtype=torch.float32).reshape(1, -1, 1, 1)
        self._latent_std = torch.tensor(KREA2_LATENTS_STD[:latent_channels], dtype=torch.float32).reshape(1, -1, 1, 1)
        self.latent_channels = latent_channels
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
        block_bytes = [_module_nbytes(block) for block in transformer.blocks]
        self.weight_sizes["transformer_max_block"] = max(block_bytes, default=0)
        self.weight_sizes["transformer_unmatched"] = max(0, self.weight_sizes["transformer"] - sum(block_bytes))
        self.device = torch.device("cuda")
        self.lora_report = []
        self.unclaimed_tensors = []
        self.last_generation_metrics = {}
        self.attention_backend = "native"
        self._vae_tiling_required = False
        # Krea 2 runs a real unconditional branch, but sequentially: the two forwards are executed
        # one after the other rather than as a batch of two, which halves the activation peak and
        # is what the Anima benchmark suite settled on for the same trade.
        self.batch_cfg = False
        self.keep_transformer_resident = False
        # The autoencoder decodes at full resolution, and by then the transformer has nothing left
        # to do but hold 12 GiB. Evicting it first is what lets a large canvas decode in one piece
        # rather than in tiles; the next job's sampling pulls it back.
        self.unload_before_decode = True
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
            raise RuntimeError("Krea2 runtime is closed")
        if self._poisoned:
            raise RuntimeError("Krea2 runtime is poisoned by an interrupted transformer restoration")

    @property
    def transformer_group_offload_enabled(self):
        return bool(self._transformer_group_offload)

    @property
    def transformer_resident(self):
        return bool(self._transformer_resident)

    def configure_attention_backend(self, backend="native"):
        """Krea 2 runs its own attention, so only the native path exists.

        The Flux engines mount Diffusers transformers, which means Diffusers' attention dispatch
        can steer them onto xFormers or a Sage kernel.  :mod:`krea2_model` calls
        ``F.scaled_dot_product_attention`` directly — PyTorch's own dispatcher already picks Flash
        where it is eligible — so there is nothing here for that context manager to redirect.
        Refusing the other names is what keeps the reported backend true: the server's existing
        fallback re-configures to ``native`` and the job says ``native``, rather than claiming a
        kernel that never ran.
        """
        self._require_open()
        if not isinstance(backend, str) or not backend:
            raise ValueError("attention backend must be a non-empty string")
        if backend != "native":
            raise ValueError(f"Krea2 attention runs natively; {backend!r} has no dispatch to apply")
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
        """Whether the transformer is evicted before the autoencoder runs."""
        self.unload_before_decode = bool(unload)
        return self.unload_before_decode

    def _release_transformer_for_decode(self):
        """Give the decoder the card. Group offload already holds nothing, so it is left alone."""
        if not self.unload_before_decode or self.transformer_group_offload_enabled:
            return False
        if not self._transformer_resident:
            return False
        self._park_transformer_on_cpu()
        _empty_cuda_cache()
        return True

    def enable_transformer_group_offload(self, blocks_per_group=1):
        self._require_open()
        if not isinstance(blocks_per_group, int) or isinstance(blocks_per_group, bool) or blocks_per_group < 1:
            raise ValueError("blocks_per_group must be a positive integer")
        if self.transformer_group_offload_enabled:
            if self._transformer_blocks_per_group != blocks_per_group:
                raise RuntimeError("Krea2 transformer group offload is already configured with a different group size")
            return
        if any(parameter.device.type != "cpu" for parameter in self.transformer.parameters()):
            raise RuntimeError("Krea2 transformer must be on CPU before enabling group offload")
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

    def _encode_tokens(self, prompt: str) -> tuple[torch.Tensor, int]:
        """The token ids ComfyUI feeds Qwen3-VL, and where its conditioning actually begins.

        ``Qwen3VLTokenizer`` passes ``disable_weights=True``, so ``(word:1.2)`` is literal text
        here rather than an emphasis instruction.
        """
        ids = self.tokenizer(self.prompt_template.format(prompt), add_special_tokens=False)["input_ids"]
        if len(ids) > KREA2_MAX_TEXT_TOKENS:
            raise ValueError(
                f"This prompt tokenises to {len(ids)} Krea 2 text tokens, beyond the "
                f"{KREA2_MAX_TEXT_TOKENS}-token ceiling; shorten it"
            )
        return torch.tensor([list(ids)], dtype=torch.long), krea2_template_end(ids)

    def token_diagnostics(self, prompt: str):
        self._require_open()
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        ids, template_end = self._encode_tokens(prompt)
        return {
            "llm": {
                "token_count": int(ids.shape[1]),
                "conditioning_token_count": int(ids.shape[1] - template_end),
                "weighted_token_count": 0,
                "template_tokens": int(template_end),
                "ceiling": KREA2_MAX_TEXT_TOKENS,
                "family": "qwen3vl_4b",
                "tap_layers": list(self.tap_layers),
            }
        }

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Twelve hidden states, flattened into the feature axis, on CPU.

        ``Krea2TEModel.encode_token_weights`` stacks the taps, drops the system and user-opening
        prefix, and folds the tap axis into the channel dimension.  All three happen here so the
        transformer sees exactly what ComfyUI hands it.
        """
        ids, template_end = self._encode_tokens(prompt)
        device = self.device
        try:
            self.text_encoder.to(device=device, dtype=self.dtype)
            with torch.inference_mode():
                output = self.text_encoder(
                    input_ids=ids.to(device),
                    attention_mask=torch.ones_like(ids, device=device),
                    output_hidden_states=True,
                    use_cache=False,
                )
                states = output.hidden_states
                if len(states) <= max(self.tap_layers):
                    raise RuntimeError(
                        f"Krea 2 text encoder produced {len(states)} hidden states, too few for taps {self.tap_layers}"
                    )
                stacked = torch.stack([states[index] for index in self.tap_layers], dim=1)
                stacked = stacked[:, :, template_end:]
                batch, taps, sequence, width = stacked.shape
                if sequence < 1:
                    raise RuntimeError("Krea 2 prompt encoding left no conditioning tokens after the template prefix")
                embeddings = stacked.permute(0, 2, 1, 3).reshape(batch, sequence, taps * width)
            return embeddings.to(device="cpu", dtype=self.dtype)
        finally:
            self.text_encoder.to("cpu")
            _empty_cuda_cache()

    # -- latents -----------------------------------------------------------------------------

    def _process_in(self, latents: torch.Tensor) -> torch.Tensor:
        """``latent_formats.Wan21.process_in``: (latent - mean) / std, at scale factor 1."""
        mean = self._latent_mean.to(device=latents.device, dtype=torch.float32)
        std = self._latent_std.to(device=latents.device, dtype=torch.float32)
        return (latents.float() - mean) / std

    def _process_out(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self._latent_mean.to(device=latents.device, dtype=torch.float32)
        std = self._latent_std.to(device=latents.device, dtype=torch.float32)
        return latents.float() * std + mean

    def _initial_latents(self, generators, height: int, width: int, sigma=1.0) -> torch.Tensor:
        shape = (
            1,
            self.latent_channels,
            height // KREA2_VAE_SCALE_FACTOR,
            width // KREA2_VAE_SCALE_FACTOR,
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
        # (B, 3, 1, H, W): the autoencoder is causal over frames and a still image is one frame.
        vae_input = torch.stack(pixels).to(device=device, dtype=self.dtype).unsqueeze(2) / 127.5 - 1.0
        try:
            self.vae.to(device=device, dtype=self.dtype)
            with torch.inference_mode():
                encoded = self.vae.encode(vae_input)
                posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
                latents = posterior.mode().squeeze(2)
                return self._process_in(latents).to(device="cpu", dtype=torch.float32)
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    def _resolved_tiled_decode(self, height: int, width: int, force_tiled_decode: bool):
        """Tiling above 1536 stands, and evicting the transformer first does not change it.

        Measured on a 24 GB card with nothing else resident: a 1024x1472 decode costs 6.18 GiB in
        one pass and 1.29 GiB in tiles, and the tiled pass is also the faster of the two (0.81s
        against 2.92s). Whatever room eviction frees, there is nothing here to spend it on — the
        untiled path is slower and quadratic in area, so the threshold is a speed choice as much
        as a memory one. A 2048x2944 decode takes 1.7 seconds tiled, which is noise beside
        sampling; this is documented because it is where the room went, not to invite raising it.
        """
        if force_tiled_decode or self._vae_tiling_required:
            return True
        return max(height, width) > KREA2_TILED_DECODE_EDGE

    def _configure_vae_tiling(self, tiled: bool) -> None:
        """Point the autoencoder's tiling at ComfyUI's VAEDecodeTiled geometry.

        ``AutoencoderKLWan.enable_tiling`` takes the geometry as arguments, so ComfyUI's 512-pixel
        tile and 64-pixel overlap map onto it directly as a 448-pixel stride.
        """
        if not tiled:
            self.vae.disable_tiling()
            return
        self.vae.enable_tiling(
            tile_sample_min_height=COMFY_VAE_TILE_PIXELS,
            tile_sample_min_width=COMFY_VAE_TILE_PIXELS,
            tile_sample_stride_height=COMFY_VAE_TILE_PIXELS - COMFY_VAE_OVERLAP_PIXELS,
            tile_sample_stride_width=COMFY_VAE_TILE_PIXELS - COMFY_VAE_OVERLAP_PIXELS,
        )

    def _decode(self, latents: torch.Tensor, force_tiled_decode: bool = False) -> list[Image.Image]:
        device = self.device
        height = latents.shape[2] * KREA2_VAE_SCALE_FACTOR
        width = latents.shape[3] * KREA2_VAE_SCALE_FACTOR
        self._release_transformer_for_decode()
        tiled = self._resolved_tiled_decode(height, width, force_tiled_decode)
        try:
            self.vae.to(device=device, dtype=self.dtype)
            self._configure_vae_tiling(tiled)
            restored = self._process_out(latents.to(device=device, dtype=torch.float32)).unsqueeze(2)
            with torch.inference_mode():
                decoded = self.vae.decode(restored.to(self.dtype), return_dict=False)[0]
                images = [_decoded_tensor_to_image(frame[:, 0].float()) for frame in decoded]
            self.last_generation_metrics.setdefault("vae_decode", {}).update({
                "actual_vae_mode": "tiled" if tiled else "full",
                "requested_tiled_decode": {"tile": COMFY_VAE_TILE_PIXELS, "overlap": COMFY_VAE_OVERLAP_PIXELS},
                "resolved_tiled_decode": {
                    "tile": int(getattr(self.vae, "tile_sample_min_height", COMFY_VAE_TILE_PIXELS)),
                    "stride": int(getattr(self.vae, "tile_sample_stride_height", COMFY_VAE_TILE_PIXELS)),
                },
            })
            return images
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    # -- sampling ----------------------------------------------------------------------------

    @staticmethod
    def _validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step):
        if sampler not in KREA2_SAMPLERS:
            raise ValueError(f"Unsupported Krea2 sampler: {sampler}")
        if scheduler not in KREA2_SCHEDULERS:
            raise ValueError(f"Unsupported Krea2 scheduler: {scheduler}")
        if guidance not in {"none", "cfg_zero_star"}:
            raise ValueError("Krea2 guidance must be 'none' or 'cfg_zero_star'")
        if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
            raise ValueError("width and height must be integers")
        if width <= 0 or height <= 0 or width % KREA2_PIXEL_ALIGNMENT or height % KREA2_PIXEL_ALIGNMENT:
            raise ValueError(f"width and height must be positive and divisible by {KREA2_PIXEL_ALIGNMENT}")
        if width > KREA2_MAX_EDGE or height > KREA2_MAX_EDGE:
            raise ValueError(f"width and height cannot exceed the Krea2 maximum of {KREA2_MAX_EDGE}")
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
        conditioning,
        latents,
        sigmas,
        sampler,
        cfg,
        generators,
        on_step,
        on_step_checkpoint,
        guidance="none",
        source_latents=None,
        source_noise=None,
        latent_mask=None,
    ):
        try:
            from .guidance import apply_cfg_zero_star
        except ImportError:
            from guidance import apply_cfg_zero_star

        device = self.device
        sampler_implementation, _warning = resolve_krea2_sampler(sampler)
        positive, negative = conditioning
        batch, _channels, latent_height, latent_width = latents.shape
        latents = latents.to(device=device, dtype=torch.float32)
        positive = self._batched_conditioning(positive, batch, device)
        negative = self._batched_conditioning(negative, batch, device) if negative is not None else None
        do_cfg = float(cfg) != 1.0 or guidance == "cfg_zero_star"
        if do_cfg and negative is None:
            raise ValueError("Krea2 classifier-free guidance requires an unconditional conditioning")
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

        def branch(state, sigma, context):
            nonlocal invocations
            invocations += 1
            timestep = torch.full((batch,), float(sigma), device=device, dtype=torch.float32)
            output = self.transformer(state.to(self.dtype), timestep, context)
            return output.float()

        def predict(state, sigma, step_index):
            conditioned = branch(state, sigma, positive)
            if not do_cfg:
                return conditioned
            unconditioned = branch(state, sigma, negative)
            if guidance == "cfg_zero_star":
                return apply_cfg_zero_star(conditioned, unconditioned, float(cfg), step_index, sample_steps).float()
            return unconditioned + float(cfg) * (conditioned - unconditioned)

        try:
            if not self.transformer_group_offload_enabled:
                self._ensure_transformer_on_cuda()
            with torch.inference_mode():
                for index in range(sample_steps):
                    sigma = sigmas[index]
                    sigma_next = sigmas[index + 1]
                    prediction = predict(latents, sigma, index)

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
                            corrected = predict(provisional, sigma_eval, index)
                            latents = latents.float() + delta * corrected
                        else:
                            provisional = latents.float() + delta * prediction
                            corrected = predict(provisional, sigma_next, index)
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
                "cfg_branches": 2 if do_cfg else 1,
            }
            return latents.to("cpu")
        finally:
            if not self.transformer_group_offload_enabled and self.transformer is not None and not self.keep_transformer_resident:
                self._park_transformer_on_cpu()
            if not self.transformer_resident:
                _empty_cuda_cache()

    def _batched_conditioning(self, embeddings, batch: int, device):
        embeddings = embeddings.to(device=device, dtype=self.dtype)
        if embeddings.shape[0] == 1 and batch > 1:
            embeddings = embeddings.expand(batch, -1, -1)
        return embeddings

    def _require_cuda(self):
        if self.device.type != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Native Krea2 generation requires CUDA")
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native Krea2 BF16 execution requires a BF16-capable CUDA device")

    def _record_sampling_metrics(self, name: str, steps: int, schedule_diagnostics: dict, guidance: str, cfg: float):
        execution = self._last_sampling_execution or {}
        branches = int(execution.get("cfg_branches", 1))
        self.last_generation_metrics.setdefault(name, {}).update({
            **schedule_diagnostics,
            "shift": round(float(KREA2_SHIFT), 4),
            "guidance": guidance,
            "transformer_input_dtype": str(self.dtype).replace("torch.", ""),
            "latent_state_dtype": "float32",
            "requested_steps": int(steps),
            "executed_denoise_updates": int(steps),
            "branch_invocations_per_update": branches,
            "sequential_transformer_invocations": int(steps) * branches,
            "actual_transformer_invocations": int(execution.get("actual_transformer_invocations", steps * branches)),
            "peak_batch_copies": 1,
            "cfg_scale": round(float(cfg), 4),
        })

    def _conditioning(self, prompt: str, negative_prompt: str, cfg: float, guidance: str, stage: str):
        positive = self._run_cuda_stage(f"{stage}prompt_encode", lambda: self._encode_prompt(prompt))
        if float(cfg) == 1.0 and guidance != "cfg_zero_star":
            # No unconditional branch will be evaluated, so encoding one would cost a full language
            # model pass for a tensor nothing reads.
            return positive, None
        negative = self._run_cuda_stage(
            f"{stage}negative_prompt_encode", lambda: self._encode_prompt(negative_prompt)
        )
        return positive, negative

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
        self._validate_request(width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step)
        self._require_cuda()
        del pag_scale, pag_applied_layers

        sigmas = krea2_sigma_schedule(steps, scheduler)
        conditioning = self._conditioning(prompt, negative_prompt, cfg, guidance, "")
        chunk_size = max(1, int(sampling_batch_size or len(generators)))
        chunks = [list(generators[start:start + chunk_size]) for start in range(0, len(generators), chunk_size)]
        steps_executed = len(sigmas) - 1
        total = len(chunks) * steps_executed
        images = []
        invocations = 0
        branches = 1
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
                    conditioning, initial, sigmas, sampler, cfg, chunk,
                    chunked(on_step), chunked(on_step_checkpoint), guidance=guidance,
                ),
            )
            execution = self._last_sampling_execution or {}
            invocations += int(execution.get("actual_transformer_invocations", 0))
            branches = int(execution.get("cfg_branches", branches))
            images.extend(self._run_cuda_stage("vae_decode", lambda latents=latents: self._decode(latents)))
        self._last_sampling_execution = {
            "actual_transformer_invocations": invocations,
            "peak_batch_copies": 1,
            "cfg_branches": branches,
        }
        self._record_sampling_metrics(
            "sampling", total, {"schedule_mode": "full", "schedule_steps": steps}, guidance, cfg
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

        sigmas, schedule_diagnostics = krea2_refinement_sigma_schedule(steps, float(denoise), scheduler)
        conditioning = self._conditioning(prompt, negative_prompt, cfg, guidance, "refinement.")
        source = self._run_cuda_stage("refinement.vae_encode", lambda: self._encode_images(images))
        noise = _cpu_noise_like_batch(source, generators)
        start_sigma = float(sigmas[0].item())
        initial = (1.0 - start_sigma) * source + start_sigma * noise
        latents = self._run_cuda_stage(
            "refinement.sampling",
            lambda: self._sample(
                conditioning, initial, sigmas, sampler, cfg, list(generators),
                on_step, on_step_checkpoint, guidance=guidance,
                source_latents=source, source_noise=noise, latent_mask=mask_tensor,
            ),
        )
        self._record_sampling_metrics("refinement.sampling", len(sigmas) - 1, schedule_diagnostics, guidance, cfg)
        return self._run_cuda_stage(
            "refinement.vae_decode", lambda: self._decode(latents, force_tiled_decode=force_tiled_decode)
        )


def krea2_template_end(ids) -> int:
    """Where Krea 2's conditioning starts inside the tokenised template.

    ``Krea2TEModel.encode_token_weights`` walks to the *second* ``<|im_start|>`` — the one opening
    the user turn — and, when it is followed by ``user`` and a newline, steps past all three.  The
    system instruction and the turn marker are dropped; the prompt, its ``<|im_end|>`` and the
    assistant opening are kept, because the model was trained reading them.
    """
    ids = list(ids)
    starts = [index for index, value in enumerate(ids) if value == KREA2_IM_START_TOKEN]
    if len(starts) < 2:
        raise ValueError("Krea 2 prompt template did not tokenise to two <|im_start|> markers")
    template_end = starts[1]
    if len(ids) > template_end + 3:
        if ids[template_end + 1] == KREA2_USER_TOKEN and ids[template_end + 2] == KREA2_NEWLINE_TOKEN:
            template_end += 3
    return template_end


def krea2_component_bytes(paths) -> int:
    """Host memory a Krea 2 load will hold, read from the files' headers rather than their size.

    Krea 2 keeps a quantised weight in its stored form, so — unlike the Flux engines beside it —
    an fp8 component costs what it occupies rather than twice that.  Budgeting the expansion that
    no longer happens overstated the transformer by 11.6 GiB, which was enough on its own to send
    a model that fits comfortably down the block-streaming path.

    A GGUF is the exception and still counts double: it has no compute-time form, so
    :func:`_load_diffusion_state_dict` expands it to the compute dtype before anything else runs.
    """
    total = 0
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == GGUF_SUFFIX:
            total += flux_component_bytes([path])
        else:
            total += checkpoint_runtime_bytes(path)
    return total


def load_krea2_runtime(
    diffusion_model,
    text_encoder,
    vae,
    qwen_tokenizer_path=None,
    dtype: torch.dtype = torch.bfloat16,
    loras=(),
) -> Krea2Runtime:
    """Load the three ComfyUI component files into one runtime."""
    deps = _runtime_dependencies()
    diffusion_path = _require_diffusion_weights(diffusion_model, "Krea2 diffusion model")
    encoder_path = _require_safetensors(text_encoder, "Krea2 text encoder")
    vae_path = _require_safetensors(vae, "Krea2 VAE")

    tokenizer = _load_krea2_tokenizer(qwen_tokenizer_path, deps)
    encoder_state = deps["load_file"](str(encoder_path), device="cpu")
    if classify_krea2_text_encoder(encoder_state.keys()) == "unknown":
        raise ValueError(f"{encoder_path.name} is not a Krea 2 Qwen3-VL text encoder")
    language_model = _load_krea2_text_encoder(encoder_state, dtype, deps)
    del encoder_state
    krea2_vae, vae_config = _load_krea2_vae(vae_path, dtype, deps)
    transformer, config, load_report = _load_krea2_transformer(diffusion_path, dtype, deps, loras=loras)

    if int(config["txtdim"]) != int(language_model.config.hidden_size):
        raise ValueError(
            f"This Krea 2 diffusion model expects a {int(config['txtdim'])}-wide conditioning, but the "
            f"selected text encoder produces {int(language_model.config.hidden_size)}; pair the model "
            "with its own Qwen3-VL-4B encoder"
        )
    if int(config["channels"]) != int(vae_config["z_dim"]):
        raise ValueError(
            f"This Krea 2 diffusion model expects {int(config['channels'])} latent channels, but the "
            f"selected autoencoder produces {int(vae_config['z_dim'])}; mount the Wan 2.1 VAE Krea 2 ships with"
        )

    runtime = Krea2Runtime(
        transformer, language_model, krea2_vae, tokenizer, dtype, config, vae_config
    )
    runtime.lora_report = load_report["loras"]
    runtime.unclaimed_tensors = load_report["unclaimed_tensors"]
    return runtime
