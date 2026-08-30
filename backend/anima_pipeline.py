# LLMAdapter implementation derived from NVIDIA/Cosmos code in sd-scripts
# library/anima_models.py, licensed under the Apache License 2.0.

import gc
import inspect
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

MIB = 1024**2
GIB = 1024**3

try:
    from .anima_sampling import (
        ANIMA_SAMPLERS,
        ANIMA_SCHEDULERS,
        anima_sampling_diagnostics,
        anima_refinement_sigma_schedule,
        anima_sigma_schedule,
        prepare_anima_refinement_sigmas,
        resolve_anima_sampler,
        validate_prepared_anima_refinement_sigmas,
    )
    from .prompt_encoding import tokenize_weighted_prompt
except ImportError:
    from anima_sampling import (
        ANIMA_SAMPLERS,
        ANIMA_SCHEDULERS,
        anima_sampling_diagnostics,
        anima_refinement_sigma_schedule,
        anima_sigma_schedule,
        prepare_anima_refinement_sigmas,
        resolve_anima_sampler,
        validate_prepared_anima_refinement_sigmas,
    )
    from prompt_encoding import tokenize_weighted_prompt


# ComfyUI's UltimateSDUpscale decodes every tile through VAEDecodeTiled(tile_size=512, overlap=64).
# Diffusers expresses the same geometry as a minimum tile plus a stride, where stride = tile - overlap.
COMFY_VAE_TILE_PIXELS = 512
COMFY_VAE_OVERLAP_PIXELS = 64
COMFY_VAE_STRIDE_PIXELS = COMFY_VAE_TILE_PIXELS - COMFY_VAE_OVERLAP_PIXELS
ANIMA_CHECKPOINT_PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.")
ANIMA_MAX_SEQUENCE_LENGTH = 512


class _GroupCfgBatchOom(RuntimeError):
    """Internal signal: replay the complete chunk sequentially from saved RNG."""
ANIMA_QWEN_PAD_EOS_TOKEN = 151643


# LLM Adapter: Bridges Qwen3 embeddings to T5-compatible space
class LLMAdapterRMSNorm(nn.Module):
    """RMSNorm specifically for the LLM Adapter (T5-style, no mean subtraction)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


def _adapter_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _adapter_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (_adapter_rotate_half(x) * sin)
    return x_embed


class AdapterRotaryEmbedding(nn.Module):
    """Rotary embedding for LLM Adapter."""

    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    """Attention module for LLM Adapter with QK-norm and separate RoPE for query/key."""

    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = LLMAdapterRMSNorm(self.head_dim)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = LLMAdapterRMSNorm(self.head_dim)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = _adapter_apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = _adapter_apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class LLMAdapterTransformerBlock(nn.Module):
    """Transformer block for LLM Adapter: optional self-attn + cross-attn + MLP."""

    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, self_attn=False, layer_norm=False):
        super().__init__()
        self.has_self_attn = self_attn

        if self.has_self_attn:
            self.norm_self_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
            self.self_attn = LLMAdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim // num_heads,
            )

        self.norm_cross_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.cross_attn = LLMAdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim // num_heads,
        )

        self.norm_mlp = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)), nn.GELU(), nn.Linear(int(model_dim * mlp_ratio), model_dim)
        )

    def forward(
        self,
        x,
        context,
        target_attention_mask=None,
        source_attention_mask=None,
        position_embeddings=None,
        position_embeddings_context=None,
    ):
        if self.has_self_attn:
            # Self-attention: target_attention_mask is not expected to be all zeros
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(
                normed,
                mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(
            normed,
            mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)


class LLMAdapter(nn.Module):
    """Bridge module: Qwen3 embeddings (source) → T5-compatible space (target).

    Uses T5 token IDs as target input, embeds them, and cross-attends to Qwen3 hidden states.
    """

    def __init__(
        self, source_dim, target_dim, model_dim, num_layers=6, num_heads=16, embed=None, self_attn=False, layer_norm=False
    ):
        super().__init__()
        if embed is not None:
            self.embed = nn.Embedding.from_pretrained(embed.weight)
        else:
            self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                LLMAdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, self_attn=self_attn, layer_norm=layer_norm)
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = LLMAdapterRMSNorm(target_dim)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


def normalize_checkpoint_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove at most one recognized Anima wrapper prefix and reject aliases."""
    normalized = {}
    sources = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ANIMA_CHECKPOINT_PREFIXES:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                break
        if new_key in normalized:
            raise ValueError(
                f"Anima checkpoint key collision after prefix normalization: {sources[new_key]!r} and {key!r} -> {new_key!r}"
            )
        normalized[new_key] = value
        sources[new_key] = key
    return normalized


def split_llm_adapter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    transformer = {}
    adapter = {}
    prefix = "llm_adapter."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            adapter[key[len(prefix) :]] = value
        else:
            transformer[key] = value
    return transformer, adapter


def shifted_sigmas(steps: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return anima_sigma_schedule(steps, "simple").to(device=device)


def euler_rf_step(
    sample: torch.Tensor, model_output: torch.Tensor, sigma: float | torch.Tensor, sigma_next: float | torch.Tensor
) -> torch.Tensor:
    sample_f = sample.float()
    return sample_f + (torch.as_tensor(sigma_next, device=sample.device) - torch.as_tensor(sigma, device=sample.device)).float() * model_output.float()


def _cpu_noise_like_batch(sample: torch.Tensor, generators: Sequence[torch.Generator]) -> torch.Tensor:
    if len(generators) != sample.shape[0]:
        raise ValueError(f"Expected {sample.shape[0]} CPU generators, got {len(generators)}")
    noise = []
    for index, generator in enumerate(generators):
        if not isinstance(generator, torch.Generator) or torch.device(generator.device).type != "cpu":
            raise ValueError(f"generators[{index}] must be a CPU torch.Generator")
        noise.append(torch.randn(sample[index : index + 1].shape, generator=generator, dtype=torch.float32, device="cpu"))
    return torch.cat(noise).to(sample.device)


def _cuda_noise_like_batch(sample: torch.Tensor, generators: Sequence[torch.Generator]) -> torch.Tensor:
    if len(generators) != sample.shape[0]:
        raise ValueError(f"Expected {sample.shape[0]} CUDA generators, got {len(generators)}")
    noise = []
    for index, generator in enumerate(generators):
        if not isinstance(generator, torch.Generator) or torch.device(generator.device).type != "cuda":
            raise ValueError(f"generators[{index}] must be a CUDA torch.Generator")
        noise.append(torch.randn(sample[index : index + 1].shape, generator=generator, dtype=torch.float32, device="cuda"))
    return torch.cat(noise)


def _derive_cuda_generators(generators: Sequence[torch.Generator]) -> list[torch.Generator]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA noise generation requires CUDA")
    derived = []
    for index, generator in enumerate(generators):
        if not isinstance(generator, torch.Generator) or torch.device(generator.device).type != "cpu":
            raise ValueError(f"generators[{index}] must be a CPU torch.Generator")
        derived.append(torch.Generator(device="cuda").manual_seed(generator.initial_seed()))
    return derived


def euler_ancestral_rf_step(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigma: float | torch.Tensor,
    sigma_next: float | torch.Tensor,
    generators: Sequence[torch.Generator],
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """LTX/Comfy CONST-parametrization Euler ancestral RF step."""
    sample_f = sample.float()
    denoised_f = denoised.float()
    sigma_value = float(torch.as_tensor(sigma, dtype=torch.float32).item())
    sigma_next_value = float(torch.as_tensor(sigma_next, dtype=torch.float32).item())
    if abs(sigma_next_value) < 1e-8:
        return denoised_f
    if abs(sigma_value) < 1e-8:
        raise ValueError("A nonterminal ancestral RF step requires a nonzero sigma")
    sigma_f = torch.as_tensor(sigma, dtype=torch.float32, device=sample.device)
    sigma_next_f = torch.as_tensor(sigma_next, dtype=torch.float32, device=sample.device)

    downstep_ratio = 1.0 + (sigma_next_f / sigma_f - 1.0) * eta
    sigma_down = sigma_next_f * downstep_ratio
    alpha_next = 1.0 - sigma_next_f
    alpha_down = 1.0 - sigma_down
    sigma_ratio = sigma_down / sigma_f
    result = sigma_ratio * sample_f + (1.0 - sigma_ratio) * denoised_f

    if eta > 0.0 and s_noise > 0.0:
        renoise_coeff = (
            sigma_next_f.square() - sigma_down.square() * alpha_next.square() / (alpha_down.square() + 1e-12)
        ).clamp(min=0.0).sqrt()
        result = (alpha_next / (alpha_down + 1e-12)) * result
        if generators and torch.device(generators[0].device).type == "cuda":
            result = result + _cuda_noise_like_batch(sample_f, generators) * float(s_noise) * renoise_coeff
        else:
            result = result + _cpu_noise_like_batch(sample_f, generators) * float(s_noise) * renoise_coeff
    return result


def _runtime_dependencies():
    try:
        import diffusers
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKLQwenImage, CosmosTransformer3DModel
        from diffusers.loaders.single_file_utils import (
            convert_cosmos_transformer_checkpoint_to_diffusers,
            convert_wan_vae_to_diffusers,
        )
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model, T5TokenizerFast
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Native Anima requires accelerate, safetensors, transformers with Qwen3, and diffusers 0.38 with "
            "CosmosTransformer3DModel, AutoencoderKLQwenImage, and the Cosmos/Wan converters"
        ) from error
    if not str(diffusers.__version__).startswith("0.38."):
        raise RuntimeError(f"Native Anima requires diffusers 0.38.x; found {diffusers.__version__}")
    return {
        "init_empty_weights": init_empty_weights,
        "CosmosTransformer3DModel": CosmosTransformer3DModel,
        "AutoencoderKLQwenImage": AutoencoderKLQwenImage,
        "convert_cosmos": convert_cosmos_transformer_checkpoint_to_diffusers,
        "convert_wan_vae": convert_wan_vae_to_diffusers,
        "load_file": load_file,
        "AutoTokenizer": AutoTokenizer,
        "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
        "T5TokenizerFast": T5TokenizerFast,
        "Qwen3Config": Qwen3Config,
        "Qwen3Model": Qwen3Model,
    }


def _require_safetensors(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} safetensors file not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"{label} must be a .safetensors checkpoint: {path}")
    return path


def _require_tokenizer_json(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tokenizer.json not found: {path}")
    if path.name.lower() != "tokenizer.json":
        raise ValueError(f"{label} tokenizer path must explicitly name tokenizer.json: {path}")
    return path


def _cast_state_dict(state_dict: Mapping[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {key: value.to(dtype=dtype) if value.is_floating_point() and value.dtype != dtype else value for key, value in state_dict.items()}


def _lora_descriptors(loras) -> list[tuple[Path, float]]:
    if loras is None:
        return []
    if not isinstance(loras, Sequence) or isinstance(loras, (str, bytes)):
        raise ValueError("loras must be an ordered sequence of {'path', 'multiplier'} mappings or (path, multiplier) tuples")
    descriptors = []
    for index, descriptor in enumerate(loras):
        if isinstance(descriptor, Mapping):
            unknown = set(descriptor) - {"path", "multiplier"}
            if unknown or "path" not in descriptor or "multiplier" not in descriptor:
                raise ValueError(
                    f"loras[{index}] must contain exactly path and multiplier; unknown={sorted(unknown)}"
                )
            path_value = descriptor["path"]
            multiplier = descriptor["multiplier"]
        elif isinstance(descriptor, (tuple, list)) and len(descriptor) == 2:
            path_value, multiplier = descriptor
        else:
            raise ValueError(f"loras[{index}] must be a path/multiplier mapping or 2-item tuple")
        if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or not math.isfinite(float(multiplier)):
            raise ValueError(f"loras[{index}].multiplier must be finite")
        path = _require_safetensors(path_value, f"Anima LoRA {index}")
        if float(multiplier) != 0.0:
            descriptors.append((path, float(multiplier)))
    return descriptors


def _lora_target_maps(
    diffusion_state: Mapping[str, torch.Tensor], qwen_state: Mapping[str, torch.Tensor]
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    native: dict[str, list[tuple[str, str]]] = {}
    official: dict[str, list[tuple[str, str]]] = {}
    for family, state, native_prefix, official_prefix in (
        ("diffusion", diffusion_state, "lora_unet_", "diffusion_model."),
        ("qwen", qwen_state, "lora_te_", "text_encoders.qwen3_06b.transformer.model."),
    ):
        for key in state:
            if not key.endswith(".weight"):
                continue
            module = key[: -len(".weight")]
            target = (family, key)
            native.setdefault(native_prefix + module.replace(".", "_"), []).append(target)
            official.setdefault(official_prefix + module, []).append(target)
    return native, official


def _resolve_lora_target(targets, name: str, label: str) -> tuple[str, str]:
    matches = targets.get(name, ())
    if not matches:
        raise ValueError(f"{label}: unknown LoRA target {name!r}")
    if len(matches) != 1:
        keys = [key for _family, key in matches]
        raise ValueError(f"{label}: ambiguous flattened LoRA alias {name!r} maps to {keys}")
    return matches[0]


def _parse_lora_key(key: str, native_targets, official_targets, label: str):
    lower = key.lower()
    if "lora_mid" in lower:
        raise ValueError(f"{label}: unsupported T-LoRA/LoCon middle tensor: {key!r}")
    if "magnitude_vector" in lower:
        raise ValueError(f"{label}: unsupported PEFT DoRA magnitude alias: {key!r}")

    suffix_roles = (
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".lora_A.weight", "down"),
        (".lora_B.weight", "up"),
        (".hada_w1_a", "hada_w1_a"),
        (".hada_w1_b", "hada_w1_b"),
        (".hada_w2_a", "hada_w2_a"),
        (".hada_w2_b", "hada_w2_b"),
        (".hada_t1", "hada_t1"),
        (".hada_t2", "hada_t2"),
        (".lokr_w1", "lokr_w1"),
        (".lokr_w1_a", "lokr_w1_a"),
        (".lokr_w1_b", "lokr_w1_b"),
        (".lokr_w2", "lokr_w2"),
        (".lokr_w2_a", "lokr_w2_a"),
        (".lokr_w2_b", "lokr_w2_b"),
        (".lokr_t2", "lokr_t2"),
        (".dora_scale", "dora_scale"),
        (".alpha", "alpha"),
    )
    for suffix, role in suffix_roles:
        if not key.endswith(suffix):
            continue
        name = key[: -len(suffix)]
        if name.startswith(("lora_unet_", "lora_te_")):
            return name, _resolve_lora_target(native_targets, name, label), role
        if name.startswith(("diffusion_model.", "text_encoders.qwen3_06b.transformer.model.")):
            return name, _resolve_lora_target(official_targets, name, label), role
        break
    raise ValueError(f"{label}: unknown or unsupported LoRA tensor key {key!r}")


def _finite_lora_tensor(tensor: torch.Tensor, role: str, group_name: str, label: str) -> torch.Tensor:
    if not tensor.is_floating_point():
        raise ValueError(f"{label}: {role} for {group_name!r} must be floating point")
    value = tensor.detach().to(dtype=torch.float32)
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{label}: {role} for {group_name!r} must contain only finite values")
    return value


def _lora_alpha(group, rank: int, group_name: str, label: str, *, reject_zero=False) -> float:
    alpha = float(rank)
    if "alpha" not in group:
        return alpha
    alpha_tensor = group["alpha"]
    if alpha_tensor.numel() != 1:
        raise ValueError(f"{label}: alpha for {group_name!r} must be a scalar tensor")
    alpha = float(alpha_tensor.detach().float().item())
    if not math.isfinite(alpha):
        raise ValueError(f"{label}: alpha for {group_name!r} must be finite")
    if reject_zero and alpha == 0.0:
        raise ValueError(f"{label}: explicit zero alpha for {group_name!r} is ambiguous")
    return alpha


def _linear_lora_base(group, states, group_name: str, label: str):
    family, target_key = group["target"]
    base = states[family][target_key]
    module_leaf = target_key[: -len(".weight")].rsplit(".", 1)[-1]
    if (
        base.ndim != 2
        or not base.is_floating_point()
        or module_leaf
        in {
            "embed",
            "embedding",
            "embeddings",
            "embed_tokens",
            "word_embeddings",
            "pos_embed",
            "position_embedding",
            "position_embeddings",
        }
    ):
        raise ValueError(f"{label}: target {target_key!r} is not a linear 2D floating-point weight")
    base_float = base.detach().to(dtype=torch.float32)
    if not torch.isfinite(base_float).all().item():
        raise ValueError(f"{label}: target {target_key!r} contains nonfinite values")
    return family, target_key, base, base_float


def _fuse_anima_lora_state_dict(
    diffusion_state: Mapping[str, torch.Tensor],
    qwen_state: Mapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
    multiplier: float,
    label: str = "Anima LoRA",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or not math.isfinite(float(multiplier)):
        raise ValueError(f"{label}: multiplier must be finite")
    if float(multiplier) == 0.0:
        return dict(diffusion_state), dict(qwen_state)

    native_targets, official_targets = _lora_target_maps(diffusion_state, qwen_state)
    groups = {}
    target_groups = {}
    for key, tensor in lora_state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label}: every entry must be a named tensor")
        group_name, target, role = _parse_lora_key(key, native_targets, official_targets, label)
        previous_group = target_groups.setdefault(target, group_name)
        if previous_group != group_name:
            raise ValueError(
                f"{label}: aliases {previous_group!r} and {group_name!r} duplicate target {target[1]!r}"
            )
        group = groups.setdefault(group_name, {"target": target})
        if role in group:
            raise ValueError(f"{label}: duplicate {role} tensor for {group_name!r}")
        group[role] = tensor

    if not groups:
        raise ValueError(f"{label}: file does not contain a supported Anima LoRA target")

    operations = []
    states = {"diffusion": diffusion_state, "qwen": qwen_state}
    for group_name, group in groups.items():
        roles = set(group) - {"target", "alpha"}
        has_loha = any(role.startswith("hada_") for role in roles)
        has_lokr = any(role.startswith("lokr_") for role in roles)
        has_lora = bool(roles & {"down", "up"})
        has_dora = "dora_scale" in roles
        family_count = sum((has_loha, has_lokr, has_lora))
        if family_count != 1 or (has_dora and not (has_lora or has_lokr)):
            raise ValueError(f"{label}: mixed or unsupported adapter families for {group_name!r}")

        family, target_key, base, base_float = _linear_lora_base(group, states, group_name, label)
        expected = tuple(base.shape)
        if has_loha:
            unsupported = roles & {"hada_t1", "hada_t2"}
            if unsupported:
                raise ValueError(f"{label}: LoHa Tucker/convolution tensors are unsupported for {group_name!r}")
            required = {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}
            missing = sorted(required - roles)
            extras = sorted(roles - required)
            if missing or extras:
                raise ValueError(
                    f"{label}: incomplete or unsupported LoHa group {group_name!r}; missing={missing}, extras={extras}"
                )
            w1a, w1b, w2a, w2b = (
                _finite_lora_tensor(group[role], role, group_name, label)
                for role in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
            )
            if any(tensor.ndim != 2 for tensor in (w1a, w1b, w2a, w2b)):
                raise ValueError(f"{label}: LoHa factors for {group_name!r} must all be 2D")
            rank = w1b.shape[0]
            if rank < 1 or w1a.shape[1] != rank or w2b.shape[0] != rank or w2a.shape[1] != rank:
                raise ValueError(f"{label}: LoHa factors for {group_name!r} must share one positive rank")
            dense1 = w1a @ w1b
            dense2 = w2a @ w2b
            if tuple(dense1.shape) != expected or tuple(dense2.shape) != expected:
                raise ValueError(f"{label}: LoHa product shape for {group_name!r} must equal {expected}")
            alpha = _lora_alpha(group, rank, group_name, label, reject_zero=True)
            delta = dense1 * dense2 * (alpha / rank)
        elif has_lokr:
            unsupported = roles & {"lokr_w1_a", "lokr_w1_b", "lokr_t2"}
            if unsupported:
                raise ValueError(f"{label}: unsupported LoKr decomposition for {group_name!r}: {sorted(unsupported)}")
            if "lokr_w1" not in roles:
                raise ValueError(f"{label}: incomplete LoKr group {group_name!r}; missing lokr_w1")
            w1 = _finite_lora_tensor(group["lokr_w1"], "lokr_w1", group_name, label)
            if w1.ndim != 2:
                raise ValueError(f"{label}: lokr_w1 for {group_name!r} must be 2D")
            full = "lokr_w2" in roles
            low_rank = bool(roles & {"lokr_w2_a", "lokr_w2_b"})
            if full == low_rank:
                raise ValueError(f"{label}: LoKr group {group_name!r} must contain exactly one second-factor form")
            if full:
                extras = roles - {"lokr_w1", "lokr_w2", "dora_scale"}
                if extras:
                    raise ValueError(f"{label}: full-matrix LoKr group {group_name!r} has ambiguous extras")
                _lora_alpha(group, 1, group_name, label)
                w2 = _finite_lora_tensor(group["lokr_w2"], "lokr_w2", group_name, label)
                if w2.ndim != 2:
                    raise ValueError(f"{label}: lokr_w2 for {group_name!r} must be 2D")
                delta = torch.kron(w1, w2)
            else:
                required = {"lokr_w1", "lokr_w2_a", "lokr_w2_b"}
                missing = sorted(required - roles)
                extras = sorted(roles - required - {"dora_scale"})
                if missing or extras:
                    raise ValueError(
                        f"{label}: incomplete or unsupported LoKr group {group_name!r}; missing={missing}, extras={extras}"
                    )
                w2a = _finite_lora_tensor(group["lokr_w2_a"], "lokr_w2_a", group_name, label)
                w2b = _finite_lora_tensor(group["lokr_w2_b"], "lokr_w2_b", group_name, label)
                if w2a.ndim != 2 or w2b.ndim != 2:
                    raise ValueError(f"{label}: low-rank LoKr second factors for {group_name!r} must be 2D")
                rank = w2b.shape[0]
                if rank < 1 or w2a.shape[1] != rank:
                    raise ValueError(f"{label}: incompatible LoKr rank shapes for {group_name!r}")
                alpha = _lora_alpha(group, rank, group_name, label)
                delta = torch.kron(w1, w2a @ w2b) * (alpha / rank)
            if tuple(delta.shape) != expected:
                raise ValueError(f"{label}: LoKr product shape for {group_name!r} is {tuple(delta.shape)}, expected {expected}")
        else:
            missing = sorted({"down", "up"} - roles)
            extras = sorted(roles - {"down", "up", "dora_scale"})
            if missing or extras:
                raise ValueError(
                    f"{label}: incomplete or unsupported LoRA pair for {group_name!r}; missing={missing}, extras={extras}"
                )
            down = _finite_lora_tensor(group["down"], "down", group_name, label)
            up = _finite_lora_tensor(group["up"], "up", group_name, label)
            if down.ndim != 2 or up.ndim != 2:
                raise ValueError(f"{label}: LoRA down/up tensors for {group_name!r} must both be 2D")
            rank = down.shape[0]
            if rank < 1 or up.shape[1] != rank:
                raise ValueError(
                    f"{label}: incompatible LoRA rank shapes for {group_name!r}: down={tuple(down.shape)}, up={tuple(up.shape)}"
                )
            actual = (up.shape[0], down.shape[1])
            if actual != expected:
                raise ValueError(f"{label}: LoRA product shape for {group_name!r} is {actual}, expected {expected}")
            alpha = _lora_alpha(group, rank, group_name, label)
            delta = (up @ down) * (alpha / rank)

        if not has_dora:
            result = base_float + float(multiplier) * delta
        else:
            magnitude = _finite_lora_tensor(group["dora_scale"], "dora_scale", group_name, label)
            if magnitude.ndim == 2 and magnitude.shape[1] == 1:
                magnitude = magnitude[:, 0]
            if magnitude.ndim != 1 or magnitude.shape[0] != base.shape[0]:
                raise ValueError(
                    f"{label}: DoRA magnitude for {group_name!r} must have shape ({base.shape[0]},) or ({base.shape[0]}, 1)"
                )
            direction = base_float + delta
            row_norm = torch.linalg.vector_norm(direction, dim=1)
            if not torch.isfinite(row_norm).all().item() or torch.any(row_norm == 0).item():
                raise ValueError(f"{label}: DoRA direction row norms for {group_name!r} must be finite and nonzero")
            target = direction * (magnitude / (row_norm + torch.finfo(torch.float32).eps)).unsqueeze(1)
            result = base_float + float(multiplier) * (target - base_float)

        if not torch.isfinite(result).all().item():
            raise ValueError(f"{label}: reconstructed adapter result for {group_name!r} is nonfinite")
        operations.append((family, target_key, result.to(dtype=base.dtype)))

    fused_diffusion = dict(diffusion_state)
    fused_qwen = dict(qwen_state)
    fused_states = {"diffusion": fused_diffusion, "qwen": fused_qwen}
    for family, target_key, result in operations:
        fused_states[family][target_key] = result
    return fused_diffusion, fused_qwen


# A LoRA names its targets in the *checkpoint's* namespace, but the modules that
# hold the weights were built from the converted Diffusers namespace, so patching
# one needs the translation between them. That translation already exists — it is
# the converter the loader ran — and re-implementing a corner of it by hand is
# what produced two silent gaps in a row: only `cross_attn`, `self_attn` and
# `mlp` were mapped, so a LoRA touching `adaln_modulation_*`, the patch embedder,
# the time embedder or the final layer had no live target and failed mid-sampling.
#
# So the map is *derived from the converter itself*: it renames keys and never
# merges or splits tensors, which means feeding it a dictionary whose values are
# the original key names returns the whole old-to-new mapping exactly, for
# whatever version of Diffusers is installed. Keys the converter drops
# (`logvar`, `pos_embedder.seq`, `_extra_state`) are absent, which is correct —
# they are not weights of the model that runs.
def _diffusers_transformer_key_map(transformer_keys, convert) -> dict[str, str]:
    probe = {f"net.{key}": key for key in transformer_keys}
    try:
        converted = convert(probe)
    except Exception:  # pragma: no cover - a converter that cannot read key names
        return {}
    mapping = {}
    for new_key, original in converted.items():
        if isinstance(original, str):
            mapping[original] = new_key
    return mapping


def _lora_target_modules(diffusion_keys, convert) -> dict[str, tuple[str, str]]:
    """Checkpoint key -> (runtime module attribute, parameter path on that module).

    The single answer to "where does this LoRA target actually live". Built once
    at load from the same conversion the loader performed, so a target either has
    a live parameter or is rejected before any sampling step runs.
    """
    prefix = "llm_adapter."
    targets: dict[str, tuple[str, str]] = {}
    transformer_keys = []
    for key in diffusion_keys:
        if key.startswith(prefix):
            targets[key] = ("llm_adapter", key[len(prefix) :])
        else:
            transformer_keys.append(key)
    for original, converted in _diffusers_transformer_key_map(transformer_keys, convert).items():
        targets[original] = ("transformer", converted)
    return targets


def _require_live_lora_targets(specs, target_modules, diffusion_shapes, qwen_shapes, label: str) -> None:
    """Fail at load, naming the file and the key, rather than deep inside sampling.

    A LoRA whose target does not exist here, or exists with another shape, cannot
    be applied at all. Discovering that after the model has loaded and the first
    sampling step has begun costs minutes and reports the fault from a stack that
    has nothing to do with the file that caused it.
    """
    for spec in specs:
        target = spec["target"]
        if spec["family"] == "qwen":
            shape = qwen_shapes.get(target)
        elif target in target_modules:
            shape = diffusion_shapes.get(target)
        else:
            raise ValueError(f"{label}: target {target!r} for {spec['group_name']!r} is not a weight of this model")
        if shape is None:
            raise ValueError(f"{label}: target {target!r} for {spec['group_name']!r} is not a weight of this model")
        if tuple(shape) != spec["expected"]:
            raise ValueError(
                f"{label}: target {target!r} for {spec['group_name']!r} is {tuple(shape)}, but the adapter reconstructs {spec['expected']}"
            )


def _analyze_anima_lora_patch(
    lora_state: Mapping[str, torch.Tensor],
    diffusion_keys: Sequence[str],
    qwen_keys: Sequence[str],
    multiplier: float,
    label: str = "Anima LoRA",
) -> list[dict]:
    """Parse and validate a LoRA without materializing any target delta.

    Returns one group spec per target: family, module path, expected shape,
    alpha, adapter family kind and the raw role tensors. The expensive delta
    matmuls happen later on the GPU in _apply_anima_lora_groups_on_gpu.
    """
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or not math.isfinite(float(multiplier)):
        raise ValueError(f"{label}: multiplier must be finite")
    if float(multiplier) == 0.0:
        return []

    diffusion_state = {key: None for key in diffusion_keys}
    qwen_state = {key: None for key in qwen_keys}
    native_targets, official_targets = _lora_target_maps(diffusion_state, qwen_state)
    groups: dict[str, dict] = {}
    target_groups: dict[str, str] = {}
    for key, tensor in lora_state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label}: every entry must be a named tensor")
        group_name, target, role = _parse_lora_key(key, native_targets, official_targets, label)
        previous_group = target_groups.setdefault(target[1], group_name)
        if previous_group != group_name:
            raise ValueError(f"{label}: aliases {previous_group!r} and {group_name!r} duplicate target {target[1]!r}")
        group = groups.setdefault(group_name, {"target": target})
        if role in group:
            raise ValueError(f"{label}: duplicate {role} tensor for {group_name!r}")
        group[role] = tensor

    if not groups:
        raise ValueError(f"{label}: file does not contain a supported Anima LoRA target")

    specs = []
    for group_name, group in groups.items():
        roles = set(group) - {"target", "alpha"}
        has_loha = any(role.startswith("hada_") for role in roles)
        has_lokr = any(role.startswith("lokr_") for role in roles)
        has_lora = bool(roles & {"down", "up"})
        has_dora = "dora_scale" in roles
        family_count = sum((has_loha, has_lokr, has_lora))
        if family_count != 1 or (has_dora and not (has_lora or has_lokr)):
            raise ValueError(f"{label}: mixed or unsupported adapter families for {group_name!r}")

        family, target_key = group["target"]
        expected_shape = None
        if has_loha:
            unsupported = roles & {"hada_t1", "hada_t2"}
            if unsupported:
                raise ValueError(f"{label}: LoHa Tucker/convolution tensors are unsupported for {group_name!r}")
            required = {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}
            missing = sorted(required - roles)
            extras = sorted(roles - required)
            if missing or extras:
                raise ValueError(
                    f"{label}: incomplete or unsupported LoHa group {group_name!r}; missing={missing}, extras={extras}"
                )
            w1a, w1b, w2a, w2b = (
                _finite_lora_tensor(group[role], role, group_name, label)
                for role in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
            )
            if any(tensor.ndim != 2 for tensor in (w1a, w1b, w2a, w2b)):
                raise ValueError(f"{label}: LoHa factors for {group_name!r} must all be 2D")
            rank = w1b.shape[0]
            if rank < 1 or w1a.shape[1] != rank or w2b.shape[0] != rank or w2a.shape[1] != rank:
                raise ValueError(f"{label}: LoHa factors for {group_name!r} must share one positive rank")
            expected_shape = (w1a.shape[0], w1b.shape[1])
            if w2a.shape[0] != expected_shape[0] or w2b.shape[1] != expected_shape[1]:
                raise ValueError(f"{label}: LoHa product shape for {group_name!r} must equal {expected_shape}")
            alpha = _lora_alpha(group, rank, group_name, label, reject_zero=True)
            kind = "loha"
        elif has_lokr:
            unsupported = roles & {"lokr_w1_a", "lokr_w1_b", "lokr_t2"}
            if unsupported:
                raise ValueError(f"{label}: unsupported LoKr decomposition for {group_name!r}: {sorted(unsupported)}")
            if "lokr_w1" not in roles:
                raise ValueError(f"{label}: incomplete LoKr group {group_name!r}; missing lokr_w1")
            w1 = _finite_lora_tensor(group["lokr_w1"], "lokr_w1", group_name, label)
            if w1.ndim != 2:
                raise ValueError(f"{label}: lokr_w1 for {group_name!r} must be 2D")
            full = "lokr_w2" in roles
            low_rank = bool(roles & {"lokr_w2_a", "lokr_w2_b"})
            if full == low_rank:
                raise ValueError(f"{label}: LoKr group {group_name!r} must contain exactly one second-factor form")
            if full:
                extras = roles - {"lokr_w1", "lokr_w2", "dora_scale"}
                if extras:
                    raise ValueError(f"{label}: full-matrix LoKr group {group_name!r} has ambiguous extras")
                # A full-matrix LoKr carries no rank to normalise by, so the
                # Kronecker product is applied as it is and the file's alpha is
                # validated rather than used as a scale. `spec["alpha"]` must
                # still be assigned here, and must be the scale that will
                # actually be applied: leaving it unset raised UnboundLocalError
                # when such a group came first, and let it silently inherit the
                # previous group's alpha when it did not.
                _lora_alpha(group, 1, group_name, label)
                alpha = 1.0
                w2 = _finite_lora_tensor(group["lokr_w2"], "lokr_w2", group_name, label)
                if w2.ndim != 2:
                    raise ValueError(f"{label}: lokr_w2 for {group_name!r} must be 2D")
                expected_shape = (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])
                kind = "lokr_full"
            else:
                required = {"lokr_w1", "lokr_w2_a", "lokr_w2_b"}
                missing = sorted(required - roles)
                extras = sorted(roles - required - {"dora_scale"})
                if missing or extras:
                    raise ValueError(
                        f"{label}: incomplete or unsupported LoKr group {group_name!r}; missing={missing}, extras={extras}"
                    )
                w2a = _finite_lora_tensor(group["lokr_w2_a"], "lokr_w2_a", group_name, label)
                w2b = _finite_lora_tensor(group["lokr_w2_b"], "lokr_w2_b", group_name, label)
                if w2a.ndim != 2 or w2b.ndim != 2:
                    raise ValueError(f"{label}: low-rank LoKr second factors for {group_name!r} must be 2D")
                rank = w2b.shape[0]
                if rank < 1 or w2a.shape[1] != rank:
                    raise ValueError(f"{label}: incompatible LoKr rank shapes for {group_name!r}")
                alpha = _lora_alpha(group, rank, group_name, label)
                expected_shape = (w1.shape[0] * w2a.shape[0], w1.shape[1] * w2b.shape[1])
                kind = "lokr_lowrank"
        else:
            missing = sorted({"down", "up"} - roles)
            extras = sorted(roles - {"down", "up", "dora_scale"})
            if missing or extras:
                raise ValueError(
                    f"{label}: incomplete or unsupported LoRA pair for {group_name!r}; missing={missing}, extras={extras}"
                )
            down = _finite_lora_tensor(group["down"], "down", group_name, label)
            up = _finite_lora_tensor(group["up"], "up", group_name, label)
            if down.ndim != 2 or up.ndim != 2:
                raise ValueError(f"{label}: LoRA down/up tensors for {group_name!r} must both be 2D")
            rank = down.shape[0]
            if rank < 1 or up.shape[1] != rank:
                raise ValueError(
                    f"{label}: incompatible LoRA rank shapes for {group_name!r}: down={tuple(down.shape)}, up={tuple(up.shape)}"
                )
            expected_shape = (up.shape[0], down.shape[1])
            alpha = _lora_alpha(group, rank, group_name, label)
            kind = "lora"

        dora = None
        if has_dora:
            magnitude = _finite_lora_tensor(group["dora_scale"], "dora_scale", group_name, label)
            if magnitude.ndim == 2 and magnitude.shape[1] == 1:
                magnitude = magnitude[:, 0]
            if magnitude.ndim != 1:
                raise ValueError(
                    f"{label}: DoRA magnitude for {group_name!r} must have shape ({expected_shape[0]},) or ({expected_shape[0]}, 1)"
                )
            dora = magnitude
        specs.append(
            {
                "group_name": group_name,
                "family": family,
                "target": target_key,
                "expected": tuple(expected_shape),
                "alpha": float(alpha),
                "kind": kind,
                "roles": {role: tensor for role, tensor in group.items() if role not in {"target", "alpha"}},
                "dora": dora,
            }
        )
    return specs


def _apply_anima_lora_groups_on_gpu(
    specs: Sequence[dict],
    get_weight: Callable[[str, str], torch.Tensor],
    multiplier: float,
    device: torch.device,
    dtype: torch.dtype,
    label: str = "Anima LoRA",
) -> None:
    """Fuse validated LoRA groups into module weights on the GPU in FP32."""
    if float(multiplier) == 0.0 or not specs:
        return
    for spec in specs:
        base = get_weight(spec["family"], spec["target"])
        module_leaf = spec["target"][: -len(".weight")].rsplit(".", 1)[-1]
        # "Not found" and "found but wrong" are different faults with different
        # fixes — one is ours, the other is the file's — and folding them into one
        # message sent the reader looking for a shape problem that did not exist.
        if base is None:
            raise ValueError(f"{label}: target {spec['target']!r} for {spec['group_name']!r} is not a weight of this model")
        if (
            tuple(base.shape) != spec["expected"]
            or not base.is_floating_point()
            or base.ndim != 2
            or module_leaf
            in {
                "embed",
                "embedding",
                "embeddings",
                "embed_tokens",
                "word_embeddings",
                "pos_embed",
                "position_embedding",
                "position_embeddings",
            }
        ):
            raise ValueError(f"{label}: target {spec['target']!r} shape mismatch for {spec['group_name']!r}")
        base_float = base.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(base_float).all().item():
            raise ValueError(f"{label}: target {spec['target']!r} contains nonfinite values")
        group = dict(spec["roles"])
        roles = set(group)
        if spec["kind"] == "loha":
            w1a, w1b, w2a, w2b = (
                group[role].to(device=device, dtype=torch.float32)
                for role in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
            )
            dense1 = w1a @ w1b
            dense2 = w2a @ w2b
            delta = dense1 * dense2 * (spec["alpha"] / w1b.shape[0])
        elif spec["kind"] == "lokr_full":
            w1 = group["lokr_w1"].to(device=device, dtype=torch.float32)
            w2 = group["lokr_w2"].to(device=device, dtype=torch.float32)
            delta = torch.kron(w1, w2)
        elif spec["kind"] == "lokr_lowrank":
            w1 = group["lokr_w1"].to(device=device, dtype=torch.float32)
            w2a = group["lokr_w2_a"].to(device=device, dtype=torch.float32)
            w2b = group["lokr_w2_b"].to(device=device, dtype=torch.float32)
            rank = w2b.shape[0]
            delta = torch.kron(w1, w2a @ w2b) * (spec["alpha"] / rank)
        else:
            down = group["down"].to(device=device, dtype=torch.float32)
            up = group["up"].to(device=device, dtype=torch.float32)
            delta = (up @ down) * (spec["alpha"] / down.shape[0])
        if tuple(delta.shape) != spec["expected"]:
            raise ValueError(f"{label}: product shape for {spec['group_name']!r} is {tuple(delta.shape)}")
        if spec["dora"] is None:
            result = base_float + float(multiplier) * delta
        else:
            magnitude = spec["dora"].to(device=device, dtype=torch.float32)
            if magnitude.shape[0] != base.shape[0]:
                raise ValueError(f"{label}: DoRA magnitude for {spec['group_name']!r} has wrong rows")
            direction = base_float + delta
            row_norm = torch.linalg.vector_norm(direction, dim=1)
            if not torch.isfinite(row_norm).all().item() or torch.any(row_norm == 0).item():
                raise ValueError(f"{label}: DoRA direction row norms for {spec['group_name']!r} must be finite and nonzero")
            target = direction * (magnitude / (row_norm + torch.finfo(torch.float32).eps)).unsqueeze(1)
            result = base_float + float(multiplier) * (target - base_float)
        if not torch.isfinite(result).all().item():
            raise ValueError(f"{label}: reconstructed adapter result for {spec['group_name']!r} is nonfinite")
        base.copy_(result.to(dtype=dtype))


def _strict_assign(module: nn.Module, state_dict: Mapping[str, torch.Tensor], label: str) -> None:
    if "assign" not in inspect.signature(module.load_state_dict).parameters:
        raise RuntimeError("Native Anima requires a PyTorch version whose load_state_dict supports assign=True")
    try:
        incompatible = module.load_state_dict(dict(state_dict), strict=True, assign=True)
    except RuntimeError as error:
        raise RuntimeError(f"Strict {label} checkpoint load failed: {error}") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict {label} checkpoint load failed: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def _module_nbytes(module: nn.Module) -> int:
    tensors = list(module.parameters()) + list(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _fused_rms_norm(norm, hidden_states: torch.Tensor) -> torch.Tensor:
    """Diffusers' RMSNorm in one kernel instead of six.

    ``diffusers.models.normalization.RMSNorm`` spells the norm out eagerly:
    upcast to float32, square, mean, multiply by the reciprocal square root,
    round back, then apply the weight.  On Anima's query and key tensors -
    ``[1, 16, 5888, 128]`` at 1472x1024 - that is six full-size intermediates,
    four of them float32, for one normalisation.  ``F.rms_norm`` is the same
    algorithm with the same float32 accumulation, done once.

    It is not bit-identical: Diffusers rounds to the weight dtype *before*
    applying the weight and the fused kernel rounds once at the end, so results
    differ by up to one bfloat16 ULP.  That is the rounding ComfyUI's own fused
    kernel performs, and matching it is the point.
    """
    if norm is None:
        return hidden_states
    weight = getattr(norm, "weight", None)
    if weight is None or getattr(norm, "bias", None) is not None:
        return norm(hidden_states)
    return F.rms_norm(hidden_states, tuple(norm.dim), weight, norm.eps)


def _split_half_rotary(hidden_states: torch.Tensor, image_rotary_emb) -> torch.Tensor:
    """Cosmos' rotary application without the rotated-copy detour.

    ``apply_rotary_emb(..., use_real_unbind_dim=-2)`` negates one half,
    concatenates a full rotated copy, then upcasts both that copy and the input
    to float32 before combining them.  Running the same arithmetic on the two
    halves directly is bit-for-bit identical - negation and multiplication are
    exact in IEEE-754, and the summation order is unchanged - and it never
    builds the rotated copy or its float32 upcast.
    """
    cos, sin = image_rotary_emb
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    half = hidden_states.shape[-1] // 2
    real, imaginary = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)
    rotated_real = (cos[..., :half] * real - sin[..., :half] * imaginary).to(hidden_states.dtype)
    rotated_imaginary = (cos[..., half:] * imaginary + sin[..., half:] * real).to(hidden_states.dtype)
    return torch.cat((rotated_real, rotated_imaginary), dim=-1)


def repair_inductor_template_encoding():
    """Make Inductor readable on a non-UTF-8 locale.

    ``torch/_inductor/utils.py::load_template`` opens PyTorch's own bundled
    Jinja templates with ``open()`` and no ``encoding``, so Python uses the
    locale default. On this machine that is cp936, one template holds a byte
    that is not valid GBK, and the failure lands during the *import* of
    ``torch._inductor`` - before any compilation is attempted - which makes
    ``torch.compile`` unusable on any CJK-locale Windows install.

    The templates are UTF-8; only the reader is wrong. Patching the reader keeps
    the blast radius at one function. The alternative, Python's UTF-8 mode, has
    to be set before the interpreter starts and would change text handling for
    every file the backend touches.
    """
    from torch._inductor import utils as inductor_utils

    if getattr(inductor_utils.load_template, "_xirai_utf8_reader", False):
        return False

    def load_template(name: str, template_dir) -> str:
        with open(Path(template_dir) / f"{name}.py.jinja", encoding="utf-8") as handle:
            return handle.read()

    load_template._xirai_utf8_reader = True
    inductor_utils.load_template = load_template
    return True


def sage_attention_callable():
    """SageAttention's entry point, or None when the package is not installed.

    Only the pure-Triton 1.x line is installable here: Diffusers' own `sage`
    backends require `sageattention>=2.1.1`, which ships no Windows wheel and
    needs an MSVC/CUDA build this machine has no C++ compiler for. Calling
    `sageattn` directly from the processor sidesteps that floor, and the 1.x
    kernel covers exactly what Anima needs - head_dim 128, BF16, no mask.
    """
    try:
        from sageattention import sageattn
    except ImportError:
        return None
    return sageattn


class AnimaCosmosAttnProcessor:
    """``CosmosAttnProcessor2_0`` with the pre-attention query/key work fused.

    ComfyUI does this whole span - query/key RMS normalisation plus the rotary
    application - in a single fused CUDA kernel
    (``comfy.quant_ops.ck.rms_rope_split_half``, called from
    ``comfy/ldm/cosmos/predict2.py``).  Diffusers' eager spelling writes roughly
    790 MB of intermediates per transformer block at 1472x1024, about a third of
    everything the block writes, and the tensors are far too large to stay in L2.
    The arithmetic here is unchanged; only the number of passes over memory is.
    """

    # SageAttention 1.x quantises Q/K to INT8 per block; it supports these head
    # dimensions and nothing else.
    SAGE_HEAD_DIMS = frozenset({64, 96, 128})

    def __init__(self, dispatch_attention_fn, sage_attention=None):
        self._dispatch_attention_fn = dispatch_attention_fn
        self._sage_attention = sage_attention

    def _use_sage(self, query, self_attention, attention_mask) -> bool:
        """Route to Sage only where its constraints hold and its win is real.

        The quantisation pays off on the 5888x5888 self-attention and is noise on
        a 5888x226 cross-attention, so cross-attention keeps the exact kernel.
        Sage also takes no attention mask and no non-CUDA tensor.
        """
        return bool(
            self._sage_attention is not None
            and self_attention
            and attention_mask is None
            and query.is_cuda
            and query.size(-1) in self.SAGE_HEAD_DIMS
            and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attention = encoder_hidden_states is None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)

        query = _fused_rms_norm(attn.norm_q, query)
        key = _fused_rms_norm(attn.norm_k, key)

        if image_rotary_emb is not None:
            query = _split_half_rotary(query, image_rotary_emb)
            key = _split_half_rotary(key, image_rotary_emb)

        # Anima's key and value head dimension already equals the query's, so
        # Diffusers' unconditional repeat_interleave is a full-tensor copy by a
        # factor of one.  Copy only when a real grouped-query ratio asks for it.
        key_repeats = query.size(3) // key.size(3)
        if key_repeats != 1:
            key = key.repeat_interleave(key_repeats, dim=3)
        value_repeats = query.size(3) // value.size(3)
        if value_repeats != 1:
            value = value.repeat_interleave(value_repeats, dim=3)

        if self._use_sage(query, self_attention, attention_mask):
            # Sage takes and returns "HND" - the [B, H, S, D] layout already in
            # hand - so this path transposes once at the end instead of thrice.
            hidden_states = self._sage_attention(
                query, key, value, tensor_layout="HND", is_causal=False
            ).transpose(1, 2)
        else:
            hidden_states = self._dispatch_attention_fn(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


# Installation is repeatable, so a module already carrying our processor is as
# valid a target as a stock one: `configure_sage_attention` reinstalls to switch
# the Sage kernel on or off, and refusing the second sweep would fail the job.
REPLACEABLE_COSMOS_PROCESSORS = ("CosmosAttnProcessor2_0", "AnimaCosmosAttnProcessor")


def install_anima_cosmos_attention_processor(transformer, sage_attention=None) -> int:
    """Put the fused processor on every Cosmos attention module."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    processor = AnimaCosmosAttnProcessor(dispatch_attention_fn, sage_attention)
    installed = 0
    for module in transformer.modules():
        if type(getattr(module, "processor", None)).__name__ not in REPLACEABLE_COSMOS_PROCESSORS:
            continue
        set_processor = getattr(module, "set_processor", None)
        if not callable(set_processor):
            continue
        set_processor(processor)
        installed += 1
    if installed == 0:
        raise RuntimeError(
            "Native Anima expected Diffusers' CosmosAttnProcessor2_0 on the Cosmos attention modules"
        )
    return installed


def _load_transformer_and_adapter(path: Path, dtype: torch.dtype, deps, state_dict=None):
    if state_dict is None:
        state_dict = normalize_checkpoint_keys(deps["load_file"](str(path), device="cpu"))
    else:
        state_dict = dict(state_dict)
    transformer_state, adapter_state = split_llm_adapter_state_dict(state_dict)
    if not adapter_state:
        raise RuntimeError("Anima diffusion checkpoint is missing required llm_adapter.* weights")
    converter_input = {f"net.{key}": value for key, value in transformer_state.items()}
    try:
        converted = deps["convert_cosmos"](converter_input)
    except Exception as error:
        raise RuntimeError(f"Failed to convert the Anima Cosmos transformer checkpoint: {error}") from error

    transformer_config = {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 28,
        "mlp_ratio": 4.0,
        "text_embed_dim": 1024,
        "adaln_lora_dim": 256,
        "max_size": (128, 512, 512),
        "patch_size": (1, 2, 2),
        "rope_scale": (1.0, 4.0, 4.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
        "use_crossattn_projection": False,
    }
    with deps["init_empty_weights"]():
        transformer = deps["CosmosTransformer3DModel"](**transformer_config)
        adapter = LLMAdapter(source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, self_attn=True)
    _strict_assign(transformer, _cast_state_dict(converted, dtype), "Anima transformer")
    _strict_assign(adapter, _cast_state_dict(adapter_state, dtype), "Anima LLM adapter")
    install_anima_cosmos_attention_processor(transformer)
    return transformer.eval().requires_grad_(False), adapter.eval().requires_grad_(False)


def _strip_qwen_model_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped = {}
    sources = {}
    for key, value in state_dict.items():
        new_key = key[len("model.") :] if key.startswith("model.") else key
        if new_key in stripped:
            raise ValueError(f"Qwen checkpoint key collision: {sources[new_key]!r} and {key!r} -> {new_key!r}")
        stripped[new_key] = value
        sources[new_key] = key
    return stripped


def _load_text_encoder(path: Path, dtype: torch.dtype, deps, state_dict=None):
    config_values = {
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "rope_theta": 1_000_000,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "use_cache": False,
        "bos_token_id": ANIMA_QWEN_PAD_EOS_TOKEN,
        "eos_token_id": ANIMA_QWEN_PAD_EOS_TOKEN,
        "pad_token_id": ANIMA_QWEN_PAD_EOS_TOKEN,
    }
    if "head_dim" in inspect.signature(deps["Qwen3Config"].__init__).parameters:
        config_values["head_dim"] = 128
    config = deps["Qwen3Config"](**config_values)
    config.use_cache = False
    with deps["init_empty_weights"]():
        text_encoder = deps["Qwen3Model"](config)

    if state_dict is None:
        state_dict = _strip_qwen_model_prefix(deps["load_file"](str(path), device="cpu"))
    else:
        state_dict = dict(state_dict)
    state_dict.pop("lm_head.weight", None)
    _strict_assign(text_encoder, _cast_state_dict(state_dict, dtype), "Qwen3-0.6B text encoder")
    return text_encoder.eval().requires_grad_(False)


def _load_vae(path: Path, dtype: torch.dtype, deps):
    state_dict = deps["load_file"](str(path), device="cpu")
    if "conv1.bias" in state_dict:
        try:
            state_dict = deps["convert_wan_vae"](dict(state_dict))
        except Exception as error:
            raise RuntimeError(f"Failed to convert the Comfy-format Qwen Image VAE checkpoint: {error}") from error
    with deps["init_empty_weights"]():
        vae = deps["AutoencoderKLQwenImage"]()
    _strict_assign(vae, _cast_state_dict(state_dict, dtype), "Qwen Image VAE")
    return vae.eval().requires_grad_(False)


def _load_tokenizers(qwen_path: Path, t5_path: Path, deps):
    qwen_config = qwen_path.parent / "tokenizer_config.json"
    qwen = (
        deps["AutoTokenizer"].from_pretrained(qwen_path.parent, local_files_only=True)
        if qwen_config.is_file()
        else deps["PreTrainedTokenizerFast"](tokenizer_file=str(qwen_path), local_files_only=True)
    )
    qwen_token = qwen.convert_ids_to_tokens(ANIMA_QWEN_PAD_EOS_TOKEN)
    if qwen_token is None:
        raise RuntimeError(f"Qwen tokenizer has no token ID {ANIMA_QWEN_PAD_EOS_TOKEN}")
    qwen.pad_token = qwen_token
    qwen.eos_token = qwen_token
    if qwen.pad_token_id != ANIMA_QWEN_PAD_EOS_TOKEN or qwen.eos_token_id != ANIMA_QWEN_PAD_EOS_TOKEN:
        raise RuntimeError("Qwen tokenizer could not set pad/eos token ID 151643")

    t5 = deps["T5TokenizerFast"](tokenizer_file=str(t5_path), local_files_only=True)
    if t5.eos_token_id is None:
        raise RuntimeError("T5 tokenizer.json does not preserve a usable EOS token")
    return qwen, t5


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_bf16_supported() -> bool:
    """Return CUDA BF16 capability without probing a CPU-only build."""
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "is_bf16_supported"):
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except (AssertionError, RuntimeError):
        # Some CPU-only builds expose the symbol but reject its invocation.
        return False


def _discard_module_storage(module: torch.nn.Module) -> None:
    """Release a module's tensor storage without copying any of it anywhere.

    Closing used to run `module.to("cpu")` over the whole runtime, which for the transformer alone is
    a ~4 GB device-to-host transfer performed purely to relocate memory that is about to be dropped —
    seconds of PCIe traffic on the critical path of an engine switch. The reason it was written that
    way is worth keeping, though: `.to()` reassigns `parameter.data` **in place**, so a stray
    reference held elsewhere (a compiled block's closure, a hook, an attention processor) follows the
    move and cannot pin VRAM. Pointing the same parameters at an empty CPU tensor keeps exactly that
    property and transfers nothing.
    """
    empty = torch.empty(0)
    for parameter in module.parameters(recurse=True):
        parameter.data = empty
        if parameter.grad is not None:
            parameter.grad = None
    for name, buffer in list(module.named_buffers(recurse=True)):
        if buffer is None:
            continue
        owner = module
        *path, attribute = name.split(".")
        for step in path:
            owner = getattr(owner, step)
        setattr(owner, attribute, empty)


def _decoded_tensor_to_image(pixels: torch.Tensor) -> Image.Image:
    pixels = ((pixels.float().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy(), mode="RGB").copy()


@dataclass(frozen=True)
class PreparedAnimaConditioning:
    """CPU-only reusable native conditioning for repeated refinement calls."""

    prompt: str
    negative_prompt: str
    cfg: float
    guidance: str
    embeddings: torch.Tensor


class AnimaRuntime:
    def __init__(
        self, transformer, text_encoder, llm_adapter, vae, qwen_tokenizer, t5_tokenizer, dtype, latent_state_mode="fp32"
    ):
        if latent_state_mode not in {"fp32", "bf16"}:
            raise ValueError("latent_state_mode must be 'fp32' or 'bf16'")
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.llm_adapter = llm_adapter
        self.vae = vae
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.dtype = dtype
        self.latent_state_mode = latent_state_mode
        self.components = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "llm_adapter": llm_adapter,
            "vae": vae,
            "qwen_tokenizer": qwen_tokenizer,
            "t5_tokenizer": t5_tokenizer,
        }
        self.weight_sizes = {
            "transformer": _module_nbytes(transformer),
            "text_encoder": _module_nbytes(text_encoder),
            "llm_adapter": _module_nbytes(llm_adapter),
            "vae": _module_nbytes(vae),
        }
        total_weight_bytes = sum(self.weight_sizes.values())
        blocks = getattr(transformer, "transformer_blocks", ())
        block_bytes = [_module_nbytes(block) for block in blocks]
        self.weight_sizes["transformer_max_block"] = max(block_bytes, default=0)
        self.weight_sizes["transformer_unmatched"] = max(
            0, self.weight_sizes["transformer"] - sum(block_bytes)
        )
        self.weight_sizes["total"] = total_weight_bytes
        self.last_generation_metrics = {}
        self._closed = False
        self._poisoned = False
        self._transformer_group_offload = False
        self._transformer_blocks_per_group = 0
        self.attention_backend = "native"
        self.keep_transformer_resident = False
        self.batch_cfg = False
        self._transformer_resident = False
        self._co_residency_failed = False
        self._vae_tiling_required = False
        self._offload_streams = []
        self._offload_stream_index = 0
        self._transformer_pinned = False
        self._transformer_host_buffers = None
        self._transformer_prefetch_started = False
        self._transformer_prefetch_stream = None
        self._use_async_transfer = False
        self.noise_device = "cpu"
        self.lora_plans = []
        self._lora_applied = False
        self._transformer_compiled = False
        self._transformer_compile_mode = None
        self._sage_attention_enabled = False

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Anima runtime is closed")
        if getattr(self, "_poisoned", False):
            raise RuntimeError("Anima runtime is poisoned by an interrupted transformer or attention restoration failure")

    @property
    def transformer_group_offload_enabled(self):
        return bool(getattr(self, "_transformer_group_offload", False))

    @property
    def transformer_resident(self):
        return bool(getattr(self, "_transformer_resident", False))

    def _init_offload_streams(self):
        if getattr(self, "_offload_streams", None) or not torch.cuda.is_available():
            return
        self._offload_streams = [torch.cuda.Stream() for _ in range(2)]

    def _next_offload_stream(self):
        self._init_offload_streams()
        if not getattr(self, "_offload_streams", None):
            return None
        index = getattr(self, "_offload_stream_index", 0)
        stream = self._offload_streams[index % len(self._offload_streams)]
        self._offload_stream_index = index + 1
        return stream

    def _pin_transformer(self):
        """Allocate the pinned host mirror once and keep owning it.

        Pinning the parameters in place is not enough: ``.to("cuda")`` replaces
        every ``parameter.data``, and the later ``.to("cpu")`` allocates fresh
        *pageable* storage. The pinned buffers were then unreferenced and freed,
        while ``_transformer_pinned`` still claimed otherwise - so every stage-in
        after the first copied 3.9 GB from pageable memory on a ``non_blocking``
        path that silently cannot overlap. Holding the buffers here means one
        host allocation for the life of the runtime, and both directions really
        are asynchronous.
        """
        if (
            getattr(self, "_transformer_pinned", False)
            or self.transformer is None
            or not torch.cuda.is_available()
        ):
            return
        try:
            named = list(self.transformer.named_parameters())
        except (TypeError, AttributeError, ValueError):
            return
        mirror = {}
        try:
            for name, parameter in named:
                data = parameter.data
                if data.device.type != "cpu":
                    continue
                mirror[name] = data if data.is_pinned() and data.is_contiguous() else data.contiguous().pin_memory()
        except RuntimeError:
            # CUDA may become unavailable after the initial capability check.
            # Do not leave a partial mirror or turn an optimization into a load failure.
            return
        for name, parameter in named:
            if name in mirror:
                parameter.data = mirror[name]
        self._transformer_host_buffers = mirror
        self._transformer_pinned = True

    def _release_transformer_host_buffers(self):
        """Drop the pinned mirror. Pinned memory is not swappable; do not keep it
        past a runtime that no longer owns the weights it mirrors."""
        self._transformer_host_buffers = None
        self._transformer_pinned = False

    def _park_transformer_on_cpu(self):
        """Return the Transformer to its own pinned buffers.

        ``Tensor.to("cpu")`` would allocate new pageable storage and strand the
        mirror. Copying into the buffers we already own keeps the pinned pages
        stable and preserves LoRA weights fused on the GPU, because the copy is
        of the live CUDA tensor.
        """
        transformer = getattr(self, "transformer", None)
        if transformer is None:
            return
        mirror = getattr(self, "_transformer_host_buffers", None)
        if not mirror or self.transformer_group_offload_enabled:
            transformer.to("cpu")
            self._transformer_resident = False
            return
        moved = False
        for name, parameter in transformer.named_parameters():
            buffer = mirror.get(name)
            data = parameter.data
            # Storage, not device, decides whether this weight is already parked.
            # `Tensor.data` hands back a fresh tensor object on every access, so an
            # `is` test would never hold and would copy each buffer into itself;
            # the storage pointer is the stable identity.
            if buffer is None or (data.device == buffer.device and data.data_ptr() == buffer.data_ptr()):
                continue
            if buffer.shape != data.shape or buffer.dtype != data.dtype:
                # A dtype or shape change invalidates the mirror; rebuild rather
                # than copy into a buffer that no longer describes the weight.
                self._release_transformer_host_buffers()
                transformer.to("cpu")
                self._transformer_resident = False
                return
            moved = moved or data.device.type == "cuda"
            buffer.copy_(data, non_blocking=True)
            parameter.data = buffer
        if moved and torch.cuda.is_available():
            # `non_blocking` device-to-host copies are only complete after a sync.
            torch.cuda.synchronize()
        # Buffers and any parameter the mirror does not cover still need moving.
        transformer.to("cpu")
        self._transformer_resident = False

    def _wait_transformer_transfer(self):
        stream = getattr(self, "_transformer_prefetch_stream", None)
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        self._transformer_prefetch_started = False
        self._transformer_prefetch_stream = None

    def _start_transformer_transfer(self):
        transformer = getattr(self, "transformer", None)
        if (
            transformer is None
            or self.transformer_group_offload_enabled
            or self.transformer_resident
            or getattr(self, "_transformer_prefetch_started", False)
            or not torch.cuda.is_available()
        ):
            return
        if not getattr(self, "_use_async_transfer", False):
            transformer.to(device=torch.device("cuda"), dtype=self.dtype)
            self._transformer_resident = True
            return
        self._pin_transformer()
        stream = self._next_offload_stream()
        if stream is None:
            transformer.to(device=torch.device("cuda"), dtype=self.dtype)
            self._transformer_resident = True
            return
        with torch.cuda.stream(stream):
            transformer.to(device=torch.device("cuda"), dtype=self.dtype, non_blocking=True)
        self._transformer_prefetch_started = True
        self._transformer_prefetch_stream = stream

    def _ensure_transformer_on_cuda(self):
        transformer = getattr(self, "transformer", None)
        if transformer is None or self.transformer_resident or self.transformer_group_offload_enabled:
            return
        if getattr(self, "_transformer_prefetch_started", False):
            self._wait_transformer_transfer()
            self._transformer_resident = True
            self._apply_diffusion_lora_on_gpu()
            return
        if not getattr(self, "_use_async_transfer", False) or not torch.cuda.is_available():
            transformer.to(device=torch.device("cuda"), dtype=self.dtype)
            self._transformer_resident = True
            self._apply_diffusion_lora_on_gpu()
            return
        stream = self._next_offload_stream()
        if stream is None:
            transformer.to(device=torch.device("cuda"), dtype=self.dtype)
            self._transformer_resident = True
            self._apply_diffusion_lora_on_gpu()
            return
        self._pin_transformer()
        with torch.cuda.stream(stream):
            transformer.to(device=torch.device("cuda"), dtype=self.dtype, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream)
        self._transformer_resident = True
        self._apply_diffusion_lora_on_gpu()

    def _lora_weight_resolver(self, family: str, target: str):
        if family == "diffusion":
            # One table, built at load from the loader's own conversion. Guessing
            # candidate names here is what hid two faults: a module attribute that
            # never existed, and a hand-written key map covering three of the
            # transformer's module kinds. A target the table does not know has no
            # live parameter, and saying so is the right answer.
            entry = (getattr(self, "lora_target_modules", None) or {}).get(target)
            if entry is None:
                return None
            module = getattr(self, entry[0], None)
            if module is None:
                return None
            try:
                return module.get_parameter(entry[1])
            except (AttributeError, KeyError, ValueError):
                return None
        module = getattr(self, "text_encoder", None)
        if module is None:
            return None
        try:
            return module.get_parameter(target)
        except (AttributeError, KeyError, ValueError):
            return None

    # Three modules can be patched, and each is only resident at its own moment:
    # the Qwen text encoder and the LLM adapter during prompt encoding, the
    # transformer during sampling. `llm_adapter.*` targets live in the diffusion
    # model file and so carry the "diffusion" family, but they belong to the text
    # pass — patched with the transformer they would be fused *after* the adapter
    # had already produced this run's conditioning, so the first image would not
    # carry them and every later one would.
    ADAPTER_TARGET_PREFIX = "llm_adapter."

    # Checkpoint key -> (module attribute, parameter path). Empty until a load
    # that has LoRAs to place fills it in.
    lora_target_modules: dict[str, tuple[str, str]] = {}

    def _lora_pass_specs(self, specs, pass_name: str):
        if pass_name == "qwen":
            return [spec for spec in specs if spec["family"] == "qwen"]
        adapter = pass_name == "adapter"
        return [
            spec for spec in specs
            if spec["family"] == "diffusion"
            and spec["target"].startswith(self.ADAPTER_TARGET_PREFIX) == adapter
        ]

    def _lora_family_plans(self, pass_name: str):
        """Select only the plans whose specs this pass patches.

        Each pass owns its own applied-flag, so they must never see another pass's
        specs: `_lora_weight_resolver` dispatches on `spec["family"]` alone, so
        passing the full list would fuse every target once per pass and double the
        effective LoRA strength.
        """
        plans = []
        for path, multiplier, specs in getattr(self, "lora_plans", None) or []:
            selected = self._lora_pass_specs(specs, pass_name)
            if selected:
                plans.append((path, multiplier, selected))
        return plans

    def _apply_diffusion_lora_on_gpu(self):
        if getattr(self, "_diffusion_lora_applied", False) or not getattr(self, "lora_plans", None):
            self._diffusion_lora_applied = True
            return
        if self.transformer is None or self.transformer.device.type != "cuda":
            return
        device = torch.device("cuda")
        for path, multiplier, specs in self._lora_family_plans("diffusion"):
            _apply_anima_lora_groups_on_gpu(
                specs,
                self._lora_weight_resolver,
                multiplier,
                device,
                self.dtype,
                label=f"Anima LoRA {path}",
            )
        self._diffusion_lora_applied = True

    def _apply_text_lora_on_gpu(self):
        if getattr(self, "_text_lora_applied", False) or not getattr(self, "lora_plans", None):
            self._text_lora_applied = True
            return
        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None or text_encoder.device.type != "cuda":
            return
        device = torch.device("cuda")
        # The adapter is moved to CUDA beside the text encoder by `_encode_prompts`
        # and parked again on the way out, so this is the only moment its weights
        # are resident and still unused.
        for pass_name in ("qwen", "adapter"):
            for path, multiplier, specs in self._lora_family_plans(pass_name):
                _apply_anima_lora_groups_on_gpu(
                    specs,
                    self._lora_weight_resolver,
                    multiplier,
                    device,
                    self.dtype,
                    label=f"Anima LoRA {path}",
                )
        self._text_lora_applied = True

    def _transformer_vae_co_residency_fits(self, megapixels=1.0):
        if not torch.cuda.is_available():
            return False
        physical_total = int(torch.cuda.get_device_properties(0).total_memory)
        weight_sizes = getattr(self, "weight_sizes", {}) or {}
        transformer_bytes = weight_sizes.get("transformer", 0)
        vae_bytes = weight_sizes.get("vae", 0)
        megapixels = max(1.0, float(megapixels))
        decode_estimate = int((1.9 + 3.2 * megapixels) * GIB)
        reserve = 600 * MIB
        return transformer_bytes + vae_bytes + decode_estimate + reserve <= physical_total

    @property
    def transformer_compiled(self):
        return bool(getattr(self, "_transformer_compiled", False))

    @property
    def sage_attention_enabled(self):
        return bool(getattr(self, "_sage_attention_enabled", False))

    def configure_sage_attention(self, enabled=False):
        """Reinstall the Cosmos processors with or without the Sage kernel.

        Returns what is actually in force, which is False whenever the package
        is missing, so a caller can report the truth rather than the request.
        """
        self._require_open()
        sage = sage_attention_callable() if enabled else None
        if enabled and sage is None:
            self._sage_attention_enabled = False
            return False
        target = sage is not None
        if target == self.sage_attention_enabled:
            # The load already installed the processor in this state. Reinstalling
            # every job would be pure churn on the common path.
            return target
        install_anima_cosmos_attention_processor(self.transformer, sage)
        self._sage_attention_enabled = target
        return target

    def configure_transformer_compilation(self, enabled=False, mode="default"):
        """Hand the 28 Cosmos blocks to Inductor, or take them back.

        Compiling the *block* rather than the whole model gives Inductor one
        graph to fuse and reuse 28 times, which keeps the first-call compile
        cost proportionate. It is assigned onto ``block.forward`` rather than by
        wrapping the module, because wrapping inserts ``_orig_mod`` into every
        parameter path - and the LoRA target table, ``_pag_targets`` and every
        ``get_parameter`` lookup address blocks by their real attribute path.

        This is what the eager processor above cannot reach: Inductor fuses the
        adaLN modulation, the three gated residuals and the norm chains that
        remain spread across separate kernels.
        """
        self._require_open()
        enabled = bool(enabled)
        if enabled and self.transformer_group_offload_enabled:
            raise RuntimeError("Anima transformer compilation and group offload cannot both be active")
        blocks = getattr(self.transformer, "transformer_blocks", None) if self.transformer is not None else None
        if enabled and (blocks is None or len(blocks) != 28):
            raise RuntimeError("Native Anima block compilation requires exactly 28 Cosmos transformer blocks")
        if enabled == self.transformer_compiled:
            return self.transformer_compiled
        if not enabled:
            # Clear the claim first. If unwrapping a block then fails, the runtime
            # must not go on reporting itself compiled - the fallback path reads
            # this flag to decide whether it has a remedy left to try.
            self._transformer_compiled = False
            self._transformer_compile_mode = None
            for block in blocks or ():
                if "forward" in vars(block):
                    del block.forward
            return False
        repair_inductor_template_encoding()
        compiled = []
        try:
            for block in blocks:
                block.forward = torch.compile(block.forward, mode=mode, dynamic=None)
                compiled.append(block)
        except BaseException:
            for block in compiled:
                if "forward" in vars(block):
                    del block.forward
            raise
        self._transformer_compiled = True
        self._transformer_compile_mode = mode
        return True

    @contextmanager
    def _eager_transformer_blocks(self, active=True):
        """Run a scope without Inductor, restoring exactly what was in force.

        PAG swaps ``attn1.processor`` inside the compiled region on every
        perturbed forward, which invalidates Dynamo's guards and would recompile
        each step. Sampling with PAG therefore steps outside compilation.
        """
        if not active or not self.transformer_compiled:
            yield
            return
        mode = getattr(self, "_transformer_compile_mode", "default")
        self.configure_transformer_compilation(False)
        try:
            yield
        finally:
            self.configure_transformer_compilation(True, mode)

    def configure_attention_backend(self, backend="native"):
        if not isinstance(backend, str) or not backend:
            raise ValueError("Anima attention backend must be a nonempty string")
        try:
            from diffusers.models.attention_dispatch import (
                AttentionBackendName,
                _AttentionBackendRegistry,
                _check_attention_backend_requirements,
            )
            backend = backend.lower()
            if backend not in {item.value for item in AttentionBackendName}:
                raise ValueError(f"Unsupported Anima attention backend: {backend}")
            name = AttentionBackendName(backend)
            _check_attention_backend_requirements(name)
            if name not in _AttentionBackendRegistry._backends:
                raise ValueError(f"Unavailable Anima attention backend: {backend}")
        except ImportError:
            if backend != "native":
                raise ValueError("Diffusers attention backend registry is unavailable")
            backend = "native"
        self.attention_backend = backend
        return backend

    def configure_transformer_residency(self, keep=False):
        self.keep_transformer_resident = bool(keep) and not getattr(self, "_co_residency_failed", False)
        if not self.keep_transformer_resident and self.transformer_resident:
            self._park_transformer_on_cpu()
            _empty_cuda_cache()

    @contextmanager
    def _attention_scope(self):
        from diffusers.models.attention_dispatch import attention_backend
        with attention_backend(getattr(self, "attention_backend", "native")):
            yield

    def enable_transformer_group_offload(self, blocks_per_group=1):
        self._require_open()
        if not isinstance(blocks_per_group, int) or isinstance(blocks_per_group, bool) or blocks_per_group < 1:
            raise ValueError("blocks_per_group must be a positive integer")
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if blocks is None or len(blocks) != 28:
            raise RuntimeError("Native Anima group offload requires exactly 28 Cosmos transformer blocks")
        if self.transformer_group_offload_enabled:
            if self._transformer_blocks_per_group != blocks_per_group:
                raise RuntimeError("Anima transformer group offload is already configured with a different group size")
            return
        if any(parameter.device.type != "cpu" for parameter in self.transformer.parameters()):
            raise RuntimeError("Anima transformer must be on CPU before enabling group offload")
        # Group offload is the OOM escape hatch, so it wins over compilation
        # rather than refusing to engage: block hooks move weights per forward,
        # which Inductor's graphs cannot see.
        self.configure_transformer_compilation(False)
        # Diffusers' hooks take ownership of every `parameter.data` from here, so
        # the pinned mirror stops describing the live weights. Release it rather
        # than hold 3.9 GB of unswappable host pages that can never be copied back.
        self._release_transformer_host_buffers()
        self.keep_transformer_resident = False
        self._transformer_resident = False
        self.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
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

    def _transformer_prediction(self, **kwargs):
        # Persistent trajectory state stays FP32. The model boundary alone
        # receives the configured weight/input dtype.
        kwargs = dict(kwargs)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = kwargs["hidden_states"].to(dtype=getattr(self, "dtype", kwargs["hidden_states"].dtype))
        def forward():
            if getattr(self, "attention_backend", "native") == "native":
                return self.transformer(return_dict=False, **kwargs)[0].float()
            with self._attention_scope():
                return self.transformer(return_dict=False, **kwargs)[0].float()

        try:
            return forward()
        except BaseException as error:
            if (
                self.transformer_compiled
                and isinstance(error, Exception)
                and not isinstance(error, torch.cuda.OutOfMemoryError)
            ):
                # Inductor can fail to compile or to run a graph for reasons that
                # have nothing to do with the model. Falling back to eager costs
                # speed; failing the job costs the picture.
                failed_mode = getattr(self, "_transformer_compile_mode", "default")
                try:
                    self.configure_transformer_compilation(False)
                except BaseException:
                    pass
                self.last_generation_metrics["compile_fallback"] = {
                    "from": f"inductor:{failed_mode}",
                    "to": "eager",
                    "reason": type(error).__name__,
                }
                try:
                    return forward()
                except BaseException as retry_error:
                    error = retry_error
            failed_backend = getattr(self, "attention_backend", "native")
            if (
                failed_backend != "native"
                and not isinstance(error, torch.cuda.OutOfMemoryError)
            ):
                # Optional SDPA kernels can be present but reject a model-specific shape at runtime.
                # Retry the identical forward with native SDPA before changing memory strategy.
                self.configure_attention_backend("native")
                self.last_generation_metrics["attention_fallback"] = {
                    "from": failed_backend,
                    "to": "native",
                    "reason": type(error).__name__,
                }
                # A native retry is still a group forward: if it also fails, it
                # must flow through the common poison/remove-hooks path below.
                try:
                    return forward()
                except BaseException as retry_error:
                    error = retry_error
            if self.transformer_group_offload_enabled and not isinstance(error, torch.cuda.OutOfMemoryError):
                self._poisoned = True
                try:
                    self._remove_transformer_group_offload()
                except BaseException:
                    pass
            raise error

    def _pag_targets(self, scope: str):
        blocks = getattr(self.transformer, "transformer_blocks", None)
        if blocks is None or len(blocks) != 28:
            raise RuntimeError("Native Anima PAG requires exactly 28 Cosmos transformer blocks")
        indices = range(len(blocks)) if scope == "all" else (len(blocks) // 2,)
        targets = []
        for index in indices:
            attention = getattr(blocks[index], "attn1", None)
            if attention is None or not hasattr(attention, "processor") or not callable(getattr(attention, "set_processor", None)):
                raise RuntimeError(f"Native Anima PAG target transformer_blocks.{index}.attn1 is unavailable")
            targets.append((f"transformer_blocks.{index}.attn1", attention))
        return targets

    def _pag_prediction(self, targets, forward):
        originals = [(name, attention, attention.processor) for name, attention in targets]
        processor = CosmosPAGIdentitySelfAttnProcessor()
        operation_error = None
        installation_failed = False
        result = None
        try:
            for _name, attention, _original in originals:
                attention.set_processor(processor)
        except BaseException as error:
            operation_error = error
            installation_failed = True
        if not installation_failed:
            try:
                result = forward()
            except BaseException as error:
                operation_error = error
        restoration_errors = []
        for name, attention, original in originals:
            try:
                attention.set_processor(original)
                if attention.processor is not original:
                    raise RuntimeError("processor identity did not restore")
            except BaseException as error:
                restoration_errors.append(f"{name}: {error}")
        if restoration_errors or installation_failed:
            self._poisoned = True
        if restoration_errors:
            raise RuntimeError(
                "Native Anima PAG failed to restore Cosmos attention processors: " + "; ".join(restoration_errors)
            ) from operation_error
        if operation_error is not None:
            raise operation_error
        return result

    def _tokenize_texts(self, texts: Sequence[str]):
        def tokenize_batch(tokenizer, *, keep_weights):
            encoded = [
                tokenize_weighted_prompt(
                    tokenizer,
                    text,
                    max_length=ANIMA_MAX_SEQUENCE_LENGTH,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True,
                )
                for text in texts
            ]
            weights = torch.stack([item["weights"] for item in encoded])
            if not keep_weights:
                weights = torch.ones_like(weights)
            return {
                "input_ids": torch.stack([item["input_ids"] for item in encoded]),
                "attention_mask": torch.stack([item["attention_mask"] for item in encoded]),
                "weights": weights,
                "token_count": [item["token_count"] for item in encoded],
                "weighted_token_count": [item["weighted_token_count"] if keep_weights else 0 for item in encoded],
            }

        qwen = tokenize_batch(self.qwen_tokenizer, keep_weights=False)
        t5 = tokenize_batch(self.t5_tokenizer, keep_weights=True)
        return qwen, t5

    def _run_cuda_stage(self, name: str, operation):
        measured = False
        try:
            torch.cuda.reset_peak_memory_stats()
            measured = True
        except Exception:
            pass
        if not hasattr(self, "last_generation_metrics"):
            self.last_generation_metrics = {}
        # The operation may record contract facts under this same name; drop any stale entry now and
        # merge what this stage records, so the timing metric never silently replaces them.
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

    def token_diagnostics(self, prompt: str):
        self._require_open()
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        qwen, t5 = self._tokenize_texts([prompt])

        def diagnostics(encoding):
            return {
                "token_count": int(encoding["token_count"][0]),
                "weighted_token_count": int(encoding["weighted_token_count"][0]),
                "max_length": ANIMA_MAX_SEQUENCE_LENGTH,
            }

        return {"qwen": diagnostics(qwen), "t5": diagnostics(t5)}

    @staticmethod
    def _validate_generation_request(
        width,
        height,
        steps,
        cfg,
        sampler,
        scheduler,
        generators,
        guidance,
        on_step=None,
        pag_scale=0.3,
        pag_applied_layers="mid",
    ):
        if sampler not in ANIMA_SAMPLERS:
            raise ValueError(f"Unsupported Anima sampler: {sampler}")
        if scheduler not in ANIMA_SCHEDULERS:
            raise ValueError(f"Unsupported Anima scheduler: {scheduler}")
        if guidance not in {"none", "pag", "cfg_zero_star"}:
            raise ValueError("guidance must be 'none', 'pag', or 'cfg_zero_star'")
        if not isinstance(pag_scale, (int, float)) or isinstance(pag_scale, bool) or not math.isfinite(float(pag_scale)):
            raise ValueError("pag_scale must be finite")
        if not 0.0 <= float(pag_scale) <= 5.0:
            raise ValueError("pag_scale must be between 0 and 5")
        if pag_applied_layers not in {"mid", "all"}:
            raise ValueError("pag_applied_layers must be 'mid' or 'all'")
        if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
            raise ValueError("width and height must be integers")
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("width and height must be positive and divisible by 32")
        if width > 4096 or height > 4096:
            raise ValueError("width and height cannot exceed the Anima maximum of 4096")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isinstance(cfg, (int, float)) or isinstance(cfg, bool) or not math.isfinite(float(cfg)):
            raise ValueError("cfg must be finite")
        if not isinstance(generators, Sequence) or isinstance(generators, (str, bytes)) or len(generators) < 1:
            raise ValueError("generators must contain at least one CPU torch.Generator")
        for index, generator in enumerate(generators):
            if not isinstance(generator, torch.Generator) or torch.device(generator.device).type != "cpu":
                raise ValueError(f"generators[{index}] must be a CPU torch.Generator")
        if on_step is not None and not callable(on_step):
            raise ValueError("on_step must be callable")

    def _encode_prompts(self, prompt: str, negative_prompt: str, do_cfg: bool):
        texts = [prompt, negative_prompt] if do_cfg else [prompt]
        qwen_tokens, t5_tokens = self._tokenize_texts(texts)
        device = torch.device("cuda")
        try:
            self.text_encoder.to(device=device, dtype=self.dtype)
            self.llm_adapter.to(device=device, dtype=self.dtype)
            self._apply_text_lora_on_gpu()
            qwen_ids = qwen_tokens["input_ids"].to(device)
            qwen_mask = qwen_tokens["attention_mask"].to(device)
            t5_ids = t5_tokens["input_ids"].to(device)
            t5_mask = t5_tokens["attention_mask"].to(device)
            t5_weights = t5_tokens["weights"].to(device)
            with torch.inference_mode():
                source = self.text_encoder(input_ids=qwen_ids, attention_mask=qwen_mask).last_hidden_state
                source = source * qwen_mask.unsqueeze(-1).to(source.dtype)
                adapted = self.llm_adapter(
                    source_hidden_states=source,
                    target_input_ids=t5_ids,
                    target_attention_mask=t5_mask,
                    source_attention_mask=qwen_mask,
                )
                adapted = adapted * t5_weights.unsqueeze(-1).to(adapted.dtype)
                adapted = adapted * t5_mask.unsqueeze(-1).to(adapted.dtype)
            return adapted.to("cpu"), t5_mask.to("cpu")
        finally:
            self.text_encoder.to("cpu")
            self.llm_adapter.to("cpu")
            _empty_cuda_cache()

    def prepare_refinement_conditioning(self, prompt, negative_prompt, cfg, guidance):
        """Encode once into a CPU-only object safe to reuse across tiled refine calls."""
        self._require_open()
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("prompt and negative_prompt must be strings")
        if not isinstance(cfg, (int, float)) or isinstance(cfg, bool) or not math.isfinite(float(cfg)):
            raise ValueError("cfg must be finite")
        if guidance not in {"none", "pag", "cfg_zero_star"}:
            raise ValueError("guidance must be 'none', 'pag', or 'cfg_zero_star'")
        do_cfg = float(cfg) > 1.0 if guidance == "pag" else float(cfg) != 1.0 or guidance == "cfg_zero_star"
        embeddings, _prompt_masks = self._run_cuda_stage(
            "refinement.prompt_encode", lambda: self._encode_prompts(prompt, negative_prompt, do_cfg)
        )
        return PreparedAnimaConditioning(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg=float(cfg),
            guidance=guidance,
            embeddings=embeddings.detach().to(device="cpu", dtype=self.dtype).contiguous().clone(),
        )

    @staticmethod
    def prepare_refinement_sigmas(steps, denoise, scheduler):
        return prepare_anima_refinement_sigmas(steps, denoise, scheduler)

    def _validate_prepared_conditioning(self, prepared, prompt, negative_prompt, cfg, guidance):
        if not isinstance(prepared, PreparedAnimaConditioning):
            raise ValueError("prepared_conditioning must be PreparedAnimaConditioning")
        if (prepared.prompt, prepared.negative_prompt, prepared.cfg, prepared.guidance) != (
            prompt, negative_prompt, float(cfg), guidance
        ):
            raise ValueError("prepared_conditioning does not match prompt, negative_prompt, cfg, and guidance")
        if not isinstance(prepared.embeddings, torch.Tensor):
            raise ValueError("prepared_conditioning embeddings must be a torch.Tensor")
        if prepared.embeddings.device.type != "cpu":
            raise ValueError("prepared_conditioning embeddings must be a CPU tensor")
        if prepared.embeddings.dtype != self.dtype:
            raise ValueError("prepared_conditioning embeddings dtype does not match runtime dtype")
        if prepared.embeddings.ndim != 3 or prepared.embeddings.shape[0] not in {1, 2}:
            raise ValueError("prepared_conditioning embeddings must have shape [1|2, sequence, hidden]")
        if not prepared.embeddings.is_contiguous():
            raise ValueError("prepared_conditioning embeddings must be contiguous")
        if not torch.isfinite(prepared.embeddings).all():
            raise ValueError("prepared_conditioning embeddings must be finite")
        # Prompt masks are consumed while creating the embedding, not by the
        # Cosmos transformer, whose padding mask is spatial and constructed in
        # _sample. Do not expose a misleading reusable prompt-mask field.
        return prepared.embeddings.clone()

    def _initial_latents(self, generators: Sequence[torch.Generator], height: int, width: int, sigma=1.0) -> torch.Tensor:
        shape = (1, 16, 1, height // 8, width // 8)
        noise = torch.cat(
            [torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu") for generator in generators]
        )
        return (noise * float(torch.as_tensor(sigma, dtype=torch.float32).item())).to(dtype=self._latent_state_dtype())

    def _latent_state_dtype(self):
        return torch.bfloat16 if getattr(self, "latent_state_mode", "fp32") == "bf16" else torch.float32

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        device = torch.device("cuda")
        pixels = []
        for image in images:
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(image.height, image.width, 3)
            pixels.append(data.permute(2, 0, 1))
        vae_input = torch.stack(pixels).unsqueeze(2).to(device=device, dtype=self.dtype)
        vae_input = vae_input / 127.5 - 1.0
        try:
            self.vae.to(device=device, dtype=self.dtype)
            with torch.inference_mode():
                encoded = self.vae.encode(vae_input)
                posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
                raw = posterior.mode()
                mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=raw.dtype).view(1, 16, 1, 1, 1)
                std = torch.tensor(self.vae.config.latents_std, device=device, dtype=raw.dtype).view(1, 16, 1, 1, 1)
                return ((raw - mean) / std).to(dtype=self._latent_state_dtype(), device="cpu")
        finally:
            self.vae.to("cpu")
            _empty_cuda_cache()

    def _refinement_start(self, source: torch.Tensor, generators, sigma) -> tuple[torch.Tensor, torch.Tensor]:
        source_f = source.to(device="cpu", dtype=torch.float32)
        noise = _cpu_noise_like_batch(source_f, generators)
        sigma_f = float(torch.as_tensor(sigma, dtype=torch.float32).item())
        initial = ((1.0 - sigma_f) * source_f + sigma_f * noise).to(dtype=self._latent_state_dtype())
        return initial, noise

    def _sample(
        self,
        embeddings,
        masks,
        width,
        height,
        steps,
        cfg,
        sampler,
        generators,
        guidance,
        on_step,
        *,
        pag_scale=0.3,
        pag_applied_layers="mid",
        scheduler="simple",
        initial_latents=None,
        sigmas=None,
        start_index=0,
        source_latents=None,
        source_noise=None,
        latent_mask=None,
        on_step_checkpoint=None,
    ):
        try:
            from .guidance import apply_cfg_zero_star, apply_pag
        except ImportError:
            from guidance import apply_cfg_zero_star, apply_pag

        device = torch.device("cuda")
        batch_size = len(generators)
        sampler_implementation, _sampler_warning = resolve_anima_sampler(sampler)
        noise_generators = generators
        if getattr(self, "noise_device", "cpu") == "cuda":
            noise_generators = _derive_cuda_generators(generators)
        do_cfg = (
            float(cfg) > 1.0
            if guidance == "pag"
            else float(cfg) != 1.0 or guidance == "cfg_zero_star"
        )
        condition = embeddings[0:1].expand(batch_size, -1, -1).to(device=device, dtype=self.dtype)
        negative = None
        if do_cfg:
            negative = embeddings[1:2].expand(batch_size, -1, -1).to(device=device, dtype=self.dtype)
        sigmas = anima_sigma_schedule(steps, scheduler) if sigmas is None else sigmas
        latents = (
            self._initial_latents(generators, height, width, sigmas[start_index])
            if initial_latents is None
            else initial_latents
        ).to(device=device, dtype=self._latent_state_dtype())
        sample_steps = len(sigmas) - 1 - start_index
        if sample_steps < 1:
            raise ValueError("sampling sigma suffix must contain at least one step")
        source = source_latents.to(device=device, dtype=torch.float32) if source_latents is not None else None
        source_noise_f = source_noise.to(device=device, dtype=torch.float32) if source_noise is not None else None
        resized_mask = None
        if latent_mask is not None:
            if source is None or source_noise_f is None:
                raise ValueError("masked refinement requires source latents and source noise")
            resized_mask = F.interpolate(
                latent_mask.to(device=device, dtype=torch.float32),
                size=source.shape[-3:],
                mode="nearest",
            )
        padding_mask = torch.zeros((1, 1, height // 8, width // 8), device=device, dtype=self.dtype)
        pag_targets = self._pag_targets(pag_applied_layers) if guidance == "pag" and float(pag_scale) != 0.0 else ()
        previous_prediction = None
        cfg_batch_enabled = bool(getattr(self, "batch_cfg", False) and do_cfg and not pag_targets)
        cfg_batch_attempts = 0
        actual_transformer_invocations = 0
        cfg_batch_fallback = None

        def guided_prediction(model_input, sigma, local_index):
            nonlocal cfg_batch_enabled, cfg_batch_attempts, cfg_batch_fallback, actual_transformer_invocations
            timestep = torch.as_tensor(sigma, dtype=torch.float32).expand(batch_size).to(
                device=device, dtype=self.dtype
            )
            conditioned = None
            unconditioned = None
            if cfg_batch_enabled:
                try:
                    cfg_batch_attempts += 1
                    actual_transformer_invocations += 1
                    combined = self._transformer_prediction(
                        hidden_states=torch.cat((model_input, model_input)),
                        timestep=torch.cat((timestep, timestep)),
                        encoder_hidden_states=torch.cat((condition, negative)),
                        # Cosmos expands this singleton mask by hidden-state batch internally.
                        padding_mask=padding_mask,
                    )
                    conditioned, unconditioned = combined.chunk(2)
                except torch.cuda.OutOfMemoryError:
                    # The caller owns the chunk/refinement boundary and must
                    # recreate CPU-generator derived inputs before sequential
                    # replay.  This applies equally to resident probes and
                    # group probes; do not continue a partially batched path.
                    raise _GroupCfgBatchOom("CFG batch CUDA OOM")
            if conditioned is None:
                actual_transformer_invocations += 1
                conditioned = self._transformer_prediction(
                    hidden_states=model_input,
                    timestep=timestep,
                    encoder_hidden_states=condition,
                    padding_mask=padding_mask,
                )
            prediction = conditioned
            perturbed = None
            if pag_targets:
                perturbed = self._pag_prediction(
                    pag_targets,
                    lambda: self._transformer_prediction(
                        hidden_states=model_input,
                        timestep=timestep,
                        encoder_hidden_states=condition,
                        padding_mask=padding_mask,
                    ),
                )
            if do_cfg:
                if unconditioned is None:
                    actual_transformer_invocations += 1
                    unconditioned = self._transformer_prediction(
                        hidden_states=model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative,
                        padding_mask=padding_mask,
                    )
                if guidance == "cfg_zero_star":
                    prediction = apply_cfg_zero_star(
                        conditioned, unconditioned, float(cfg), local_index, sample_steps
                    ).float()
                else:
                    prediction = unconditioned + float(cfg) * (conditioned - unconditioned)
            if perturbed is not None:
                prediction = apply_pag(
                    conditioned,
                    perturbed,
                    float(pag_scale),
                    unconditioned=unconditioned,
                    guidance_scale=float(cfg),
                ).float()
            return prediction

        try:
            if not self.transformer_group_offload_enabled:
                self._ensure_transformer_on_cuda()
            with self._eager_transformer_blocks(bool(pag_targets)), torch.inference_mode():
                for local_index, index in enumerate(range(start_index, len(sigmas) - 1)):
                    sigma = sigmas[index]
                    sigma_next = sigmas[index + 1]
                    model_input = latents
                    prediction = guided_prediction(model_input, sigma, local_index)

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
                            if torch.device(noise_generators[0].device).type == "cuda":
                                noise = _cuda_noise_like_batch(latents.float(), noise_generators)
                            else:
                                noise = _cpu_noise_like_batch(latents.float(), noise_generators)
                            latents = (1.0 - sigma_next_device) * denoised + sigma_next_device * noise
                    elif sampler_implementation == "multistep":
                        delta = torch.as_tensor(sigma_next - sigma, device=device, dtype=torch.float32)
                        derivative = (
                            prediction
                            if previous_prediction is None
                            else 1.5 * prediction - 0.5 * previous_prediction
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
                            corrected = guided_prediction(provisional, sigma_eval, local_index)
                            latents = latents.float() + delta * corrected
                        else:
                            provisional = latents.float() + delta * prediction
                            corrected = guided_prediction(provisional, sigma_next, local_index)
                            latents = latents.float() + delta * 0.5 * (prediction + corrected)
                    latents = latents.to(dtype=self._latent_state_dtype())
                    if resized_mask is not None:
                        sigma_next_f = torch.as_tensor(sigma_next, dtype=torch.float32, device=device)
                        source_noised = ((1.0 - sigma_next_f) * source + sigma_next_f * source_noise_f).to(
                            dtype=self._latent_state_dtype()
                        )
                        latents = (latents * resized_mask + source_noised * (1.0 - resized_mask)).to(
                            dtype=self._latent_state_dtype()
                        )
                    if on_step_checkpoint is not None:
                        on_step_checkpoint(local_index + 1, sample_steps, latents)
                    if on_step is not None:
                        on_step(local_index + 1, sample_steps, latents)
            self._last_sampling_execution = {
                "actual_transformer_invocations": actual_transformer_invocations,
                "peak_batch_copies": 2 if cfg_batch_attempts else 1,
                "cfg_batch_attempts": cfg_batch_attempts,
                "cfg_batch_fallback": cfg_batch_fallback,
            }
            if cfg_batch_fallback is not None:
                self.last_generation_metrics["cfg_batch_fallback"] = cfg_batch_fallback
            return latents.to("cpu")
        finally:
            if getattr(self, "_transformer_prefetch_started", False):
                self._wait_transformer_transfer()
            if not self.transformer_group_offload_enabled and self.transformer is not None and not getattr(self, "keep_transformer_resident", False):
                self._park_transformer_on_cpu()
            if not self.transformer_resident:
                _empty_cuda_cache()

    def _comfy_tiled_decode_kwargs(self):
        """Pass Comfy's 512/64 tile geometry explicitly when this VAE accepts it."""
        try:
            parameters = inspect.signature(self.vae.enable_tiling).parameters
        except (TypeError, ValueError):
            return {}
        requested = {
            "tile_sample_min_height": COMFY_VAE_TILE_PIXELS,
            "tile_sample_min_width": COMFY_VAE_TILE_PIXELS,
            "tile_sample_stride_height": COMFY_VAE_STRIDE_PIXELS,
            "tile_sample_stride_width": COMFY_VAE_STRIDE_PIXELS,
        }
        return {key: value for key, value in requested.items() if key in parameters}

    def _resolved_tiled_decode(self):
        """Report the geometry the VAE actually holds, never the geometry we asked for."""
        names = ("tile_sample_min_height", "tile_sample_min_width", "tile_sample_stride_height", "tile_sample_stride_width")
        resolved = {name: getattr(self.vae, name, None) for name in names}
        values = [resolved[name] for name in names]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            resolved["mode"] = "unknown"
            return resolved
        height, width, stride_height, stride_width = values
        resolved.update({
            "mode": "diffusers_explicit_geometry",
            "overlap_height": height - stride_height,
            "overlap_width": width - stride_width,
            "matches_comfy_contract": (height == width == COMFY_VAE_TILE_PIXELS
                                       and stride_height == stride_width == COMFY_VAE_STRIDE_PIXELS),
        })
        return resolved

    def _decode(self, latents: torch.Tensor, force_tiled_decode: bool = False) -> list[Image.Image]:
        device = torch.device("cuda")
        had_resident_transformer = self.transformer_resident
        evicted_for_decode = False

        def decode_once():
            images = []
            self.vae.to(device=device, dtype=self.dtype)
            mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=self.dtype).view(1, 16, 1, 1, 1)
            std = torch.tensor(self.vae.config.latents_std, device=device, dtype=self.dtype).view(1, 16, 1, 1, 1)
            with torch.inference_mode():
                for latent in latents.split(1):
                    vae_latent = latent.to(device=device, dtype=self.dtype) * std + mean
                    pixels = self.vae.decode(vae_latent, return_dict=False)[0][0, :, 0].float().clamp(-1.0, 1.0)
                    images.append(_decoded_tensor_to_image(pixels))
            return images

        temporary_tiling = False
        tiling_was_enabled = bool(getattr(self.vae, "use_tiling", False))
        persistent_tiling_required = bool(getattr(self, "_vae_tiling_required", False))
        if force_tiled_decode and not tiling_was_enabled and not persistent_tiling_required:
            enable_tiling = getattr(self.vae, "enable_tiling", None)
            disable_tiling = getattr(self.vae, "disable_tiling", None)
            if not callable(enable_tiling) or not callable(disable_tiling):
                raise RuntimeError("USDU tiled refinement requires reversible VAE tiled decode APIs; full decode is not allowed")
            # Request Comfy's exact tile/overlap contract when the VAE exposes it. This is
            # call-local: OOM fallback is the only path that permanently promotes the runtime's
            # tiled-decode requirement.
            enable_tiling(**self._comfy_tiled_decode_kwargs())
            temporary_tiling = True
        if force_tiled_decode:
            self.last_generation_metrics["refinement.vae_decode"] = {
                "requested_tiled_decode": {"tile": COMFY_VAE_TILE_PIXELS, "overlap": COMFY_VAE_OVERLAP_PIXELS,
                                           "stride": COMFY_VAE_STRIDE_PIXELS},
                "resolved_tiled_decode": self._resolved_tiled_decode(),
                "actual_vae_mode": "tiled",
            }

        try:
            if not hasattr(self, "last_generation_metrics"):
                self.last_generation_metrics = {}
            self.last_generation_metrics.setdefault("vae_transition", {
                "sampling_transformer_evicted": bool(not self.transformer_resident),
                "actual_vae_stage": "decode",
            })
            if (
                had_resident_transformer
                and not self.transformer_group_offload_enabled
                and not self._transformer_vae_co_residency_fits(
                    latents.shape[-2] * 8 * latents.shape[-1] * 8 / (1024 * 1024)
                )
            ):
                self._wait_transformer_transfer()
                self._park_transformer_on_cpu()
                evicted_for_decode = True
                self.last_generation_metrics["vae_transition"] = {
                    "sampling_transformer_evicted": True,
                    "actual_vae_stage": "decode",
                }
                _empty_cuda_cache()
            try:
                return decode_once()
            except torch.cuda.OutOfMemoryError:
                self.vae.to("cpu")
                if had_resident_transformer and not evicted_for_decode:
                    self._park_transformer_on_cpu()
                    self.keep_transformer_resident = False
                    self._co_residency_failed = True
                    _empty_cuda_cache()
                    try:
                        return decode_once()
                    except torch.cuda.OutOfMemoryError:
                        self.vae.to("cpu")
                if getattr(self.vae, "use_tiling", False):
                    raise
                enable_tiling = getattr(self.vae, "enable_tiling", None)
                if not callable(enable_tiling):
                    raise
                enable_tiling(**self._comfy_tiled_decode_kwargs())
                self._vae_tiling_required = True
                self.last_generation_metrics["vae_decode_fallback"] = {
                    "from": "full",
                    "to": "tiled",
                    "reason": "cuda_oom",
                }
                _empty_cuda_cache()
                return decode_once()
        finally:
            self.vae.to("cpu")
            if temporary_tiling:
                # Always restore on success, error, and cancellation. Do not
                # reset an existing OOM-derived persistent policy.
                self.vae.disable_tiling()
            if not self.transformer_resident:
                _empty_cuda_cache()

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
        guidance,
        on_step: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        on_step_checkpoint: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        sampling_batch_size: Optional[int] = None,
        pag_scale: float = 0.3,
        pag_applied_layers: str = "mid",
    ) -> list[Image.Image]:
        self._require_open()
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("prompt and negative_prompt must be strings")
        self._validate_generation_request(
            width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step, pag_scale, pag_applied_layers
        )
        if not torch.cuda.is_available():
            raise RuntimeError("Native Anima generation requires CUDA")
        if self.dtype == torch.bfloat16 and not _cuda_bf16_supported():
            raise RuntimeError("Native Anima BF16 execution requires a BF16-capable CUDA device")
        if sampling_batch_size is None:
            sampling_batch_size = len(generators)
        if not isinstance(sampling_batch_size, int) or isinstance(sampling_batch_size, bool) or sampling_batch_size < 1:
            raise ValueError("sampling_batch_size must be a positive integer")

        self.last_generation_metrics = {}
        sampling_diagnostics = anima_sampling_diagnostics(sampler, scheduler)
        do_cfg = float(cfg) > 1.0 if guidance == "pag" else float(cfg) != 1.0 or guidance == "cfg_zero_star"
        def encode_prompts():
            if not self.transformer_group_offload_enabled and not getattr(self, "_co_residency_failed", False):
                self._start_transformer_transfer()
            try:
                return self._encode_prompts(prompt, negative_prompt, do_cfg)
            except torch.cuda.OutOfMemoryError:
                if not self.transformer_resident:
                    raise
                self._wait_transformer_transfer()
                self._park_transformer_on_cpu()
                self.keep_transformer_resident = False
                self._co_residency_failed = True
                _empty_cuda_cache()
                return self._encode_prompts(prompt, negative_prompt, do_cfg)

        embeddings, masks = self._run_cuda_stage("prompt_encode", encode_prompts)
        chunks = [generators[index : index + sampling_batch_size] for index in range(0, len(generators), sampling_batch_size)]
        transactional_batch_job = bool(getattr(self, "batch_cfg", False))
        def sample_chunks():
            sampled = []
            resident_fallback = False
            chunk_executions = []
            fallback_history = []
            for chunk_index, chunk in enumerate(chunks):
                offset = chunk_index * steps
                transactional_callbacks = transactional_batch_job
                published_steps = []
                def checkpoint_callback(step, _total, latents, offset=offset):
                    if on_step_checkpoint is not None:
                        on_step_checkpoint(offset + step, len(chunks) * steps, latents)
                def chunk_callback(step, _total, latents, offset=offset):
                    if on_step is None:
                        return
                    if transactional_callbacks:
                        # Do not retain GPU tensors/graphs while a batch attempt is
                        # speculative. Anima previews are disabled, but this keeps
                        # the callback API safe for future consumers.
                        published_steps.append((offset + step, len(chunks) * steps, latents.detach().to("cpu").clone()))
                    else:
                        on_step(offset + step, len(chunks) * steps, latents)
                def commit_callbacks():
                    for step, total, latent in published_steps:
                        on_step(step, total, latent)
                    published_steps.clear()
                generator_states = [generator.get_state() for generator in chunk]
                chunk_history = []
                # A `_GroupCfgBatchOom` only reaches this branch after the
                # resident batch path. Group-batch OOM uses the same exception
                # but starts with hooks installed, so retain that distinction.
                batch_attempt_was_resident = not bool(getattr(self, "_transformer_group_offload", False))
                try:
                    latents = self._sample(
                        embeddings,
                        masks,
                        width,
                        height,
                        steps,
                        cfg,
                        sampler,
                        chunk,
                        guidance,
                        chunk_callback,
                        on_step_checkpoint=checkpoint_callback,
                        pag_scale=pag_scale,
                        pag_applied_layers=pag_applied_layers,
                        scheduler=scheduler,
                    )
                except BaseException as error:
                    if isinstance(error, _GroupCfgBatchOom):
                        # Sampling may already have consumed ancestral noise in earlier
                        # updates. Replay the entire chunk from its boundary, rather
                        # than continuing a partially batched trajectory.
                        self.batch_cfg = False
                        published_steps.clear()
                        for generator, state in zip(chunk, generator_states):
                            generator.set_state(state)
                        self.last_generation_metrics["cfg_batch_fallback"] = {
                            "from": "batched",
                            "to": "sequential",
                            "reason": "cuda_oom",
                            "stage": "sampling",
                            "attempts": 1,
                            "generator_states_restored": True,
                        }
                        transition = {
                            "from": "resident_batched", "to": "resident_sequential", "reason": "cuda_oom",
                            "stage": "sampling", "attempt": 1, "generator_restored": True,
                            "callback_buffer_discarded": True, "chunk_index": chunk_index,
                        }
                        chunk_history.append(transition); fallback_history.append(transition)
                        _empty_cuda_cache()
                        try:
                            latents = self._sample(
                                embeddings, masks, width, height, steps, cfg, sampler, chunk, guidance,
                                chunk_callback, pag_scale=pag_scale, pag_applied_layers=pag_applied_layers,
                                scheduler=scheduler, on_step_checkpoint=checkpoint_callback,
                            )
                        except BaseException as sequential_error:
                            if not ("outofmemory" in type(sequential_error).__name__.lower() or "out of memory" in str(sequential_error).lower()):
                                raise
                            # Batch retry has been consumed. A resident serial
                            # retry may now use the existing one-time group
                            # replay, never another batch retry.
                            self._park_transformer_on_cpu()
                            self.keep_transformer_resident = False
                            _empty_cuda_cache()
                            self.enable_transformer_group_offload(1)
                            # The sequential attempt may have buffered public
                            # progress; only the final group replay may commit.
                            published_steps.clear()
                            for generator, state in zip(chunk, generator_states):
                                generator.set_state(state)
                            resident_fallback = True
                            transition = {
                                "from": "resident_sequential", "to": "group_sequential", "reason": "cuda_oom",
                                "stage": "sampling", "attempt": 1, "generator_restored": True,
                                "callback_buffer_discarded": True, "chunk_index": chunk_index,
                            }
                            chunk_history.append(transition); fallback_history.append(transition)
                            latents = self._sample(
                                embeddings, masks, width, height, steps, cfg, sampler, chunk, guidance,
                                chunk_callback, pag_scale=pag_scale, pag_applied_layers=pag_applied_layers,
                                scheduler=scheduler, on_step_checkpoint=checkpoint_callback,
                            )
                        sampled.append(latents)
                        chunk_executions.append({**dict(getattr(self, "_last_sampling_execution", {})), "fallback_history": chunk_history})
                        commit_callbacks()
                        continue
                    is_oom = "outofmemory" in type(error).__name__.lower() or "out of memory" in str(error).lower()
                    if self.transformer_group_offload_enabled or not is_oom:
                        raise
                    try:
                        if getattr(self, "_transformer_prefetch_started", False):
                            self._wait_transformer_transfer()
                        self._park_transformer_on_cpu()
                        self.keep_transformer_resident = False
                        _empty_cuda_cache()
                        self.enable_transformer_group_offload(1)
                    except BaseException:
                        self._poisoned = True
                        raise error
                    for generator, state in zip(chunk, generator_states):
                        generator.set_state(state)
                    resident_fallback = True
                    transition = {
                        "from": "resident_sequential", "to": "group_sequential", "reason": "cuda_oom",
                        "stage": "sampling", "attempt": 1, "generator_restored": True,
                        "callback_buffer_discarded": False, "chunk_index": chunk_index,
                    }
                    chunk_history.append(transition); fallback_history.append(transition)
                    latents = self._sample(
                        embeddings,
                        masks,
                        width,
                        height,
                        steps,
                        cfg,
                        sampler,
                        chunk,
                        guidance,
                        chunk_callback,
                        pag_scale=pag_scale,
                        pag_applied_layers=pag_applied_layers,
                        scheduler=scheduler,
                        on_step_checkpoint=checkpoint_callback,
                    )
                sampled.append(latents)
                chunk_executions.append({**dict(getattr(self, "_last_sampling_execution", {})), "fallback_history": chunk_history})
                commit_callbacks()
            if resident_fallback:
                self.last_generation_metrics["sampling_fallback"] = {
                    "from": "staged_transformer_resident",
                    "to": "staged_transformer_group_offload",
                    "reason": "cuda_oom",
                    "stage": "sampling",
                    "attempts": 1,
                    "generator_states_restored": True,
                }
            self._last_sampling_execution = {
                "chunks": chunk_executions,
                "actual_transformer_invocations": sum(int(item.get("actual_transformer_invocations", 0)) for item in chunk_executions),
                "peak_batch_copies": max((int(item.get("peak_batch_copies", 1)) for item in chunk_executions), default=1),
                "cfg_batch_attempts": sum(int(item.get("cfg_batch_attempts", 0)) for item in chunk_executions),
                "fallback_history": fallback_history,
                "known_count": len(chunk_executions), "complete": len(chunk_executions) == len(chunks),
            }
            return torch.cat(sampled)

        latents = self._run_cuda_stage("sampling", sample_chunks)
        execution = getattr(self, "_last_sampling_execution", {})
        self.last_generation_metrics["sampling"].update({
            **{key: value for key, value in sampling_diagnostics.items() if key != "warning"},
            "transformer_input_dtype": str(self.dtype).replace("torch.", ""),
            "latent_state_dtype": str(self._latent_state_dtype()).replace("torch.", ""),
            "conditioning_reused": False,
            "requested_steps": int(steps),
            "executed_denoise_updates": int(steps) * len(chunks),
            "schedule_construction_steps": int(steps) * len(chunks),
            "branch_invocations_per_update": 1 if not do_cfg else (1 if execution.get("cfg_batch_attempts") else 2),
            "sequential_transformer_invocations": int(steps) * len(chunks) * (2 if do_cfg else 1),
            "actual_transformer_invocations": int(execution.get("actual_transformer_invocations", int(steps) * (2 if do_cfg else 1))),
            "peak_batch_copies": int(execution.get("peak_batch_copies", 1)),
            "chunk_execution_known_count": execution.get("known_count", 0),
            "chunk_execution_complete": execution.get("complete", False),
            "chunk_executions": execution.get("chunks", []),
            "fallback_history": execution.get("fallback_history", []),
        })
        return self._run_cuda_stage("vae_decode", lambda: self._decode(latents))

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
        guidance,
        masks=None,
        on_step: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        pag_scale: float = 0.3,
        pag_applied_layers: str = "mid",
        prepared_conditioning: Optional[PreparedAnimaConditioning] = None,
        prepared_sigmas: Optional[torch.Tensor] = None,
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
        self._validate_generation_request(
            width, height, steps, cfg, sampler, scheduler, generators, guidance, on_step, pag_scale, pag_applied_layers
        )
        if len(generators) != len(images):
            raise ValueError(f"Expected one CPU generator per image ({len(images)}), got {len(generators)}")
        if not isinstance(denoise, (int, float)) or isinstance(denoise, bool) or not math.isfinite(float(denoise)):
            raise ValueError("denoise must be finite")
        if float(denoise) <= 0.0 or float(denoise) > 1.0:
            raise ValueError("denoise must be greater than 0 and at most 1")
        mask_tensor = None
        if masks is not None:
            if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)) or len(masks) != len(images):
                raise ValueError("masks must contain one PIL mask per image")
            mask_values = []
            for index, mask in enumerate(masks):
                if not isinstance(mask, Image.Image) or mask.size != (width, height):
                    raise ValueError(f"masks[{index}] must be a same-size PIL image")
                values = torch.frombuffer(bytearray(mask.convert("L").tobytes()), dtype=torch.uint8)
                mask_values.append(values.view(1, 1, height, width).float() / 255.0)
            mask_tensor = torch.stack(mask_values)

        if not torch.cuda.is_available():
            raise RuntimeError("Native Anima refinement requires CUDA")
        if self.dtype == torch.bfloat16 and not _cuda_bf16_supported():
            raise RuntimeError("Native Anima BF16 execution requires a BF16-capable CUDA device")

        conditioning_reused = prepared_conditioning is not None
        refinement_do_cfg = float(cfg) > 1.0 if guidance == "pag" else float(cfg) != 1.0 or guidance == "cfg_zero_star"
        if conditioning_reused:
            embeddings = self._validate_prepared_conditioning(
                prepared_conditioning, prompt, negative_prompt, cfg, guidance
            )
        else:
            prepared_conditioning = self.prepare_refinement_conditioning(prompt, negative_prompt, cfg, guidance)
            embeddings = prepared_conditioning.embeddings.clone()
        if prepared_sigmas is None:
            sigmas, schedule_diagnostics = anima_refinement_sigma_schedule(steps, denoise, scheduler)
        else:
            sigmas = validate_prepared_anima_refinement_sigmas(prepared_sigmas, steps, denoise, scheduler)
            _expected, schedule_diagnostics = anima_refinement_sigma_schedule(steps, denoise, scheduler)
        source = self._run_cuda_stage("refinement.vae_encode", lambda: self._encode_images(images))
        generator_states = [generator.get_state() for generator in generators]
        transactional_callbacks = bool(getattr(self, "batch_cfg", False))
        published_steps = []
        fallback_history = []
        def checkpoint_callback(step, total, latents):
            if on_step_checkpoint is not None:
                on_step_checkpoint(step, total, latents)
        def visible_callback(step, total, latents):
            if on_step is None:
                return
            if transactional_callbacks:
                published_steps.append((step, total, latents.detach().to("cpu").clone()))
            else:
                on_step(step, total, latents)
        def sample_refinement():
            initial, source_noise = self._refinement_start(source, generators, sigmas[0])
            return self._sample(
                embeddings,
                None,
                width,
                height,
                steps,
                cfg,
                sampler,
                generators,
                guidance,
                visible_callback,
                pag_scale=pag_scale,
                pag_applied_layers=pag_applied_layers,
                scheduler=scheduler,
                initial_latents=initial,
                sigmas=sigmas,
                start_index=0,
                source_latents=source,
                source_noise=source_noise,
                latent_mask=mask_tensor,
                on_step_checkpoint=checkpoint_callback,
            )
        try:
            latents = self._run_cuda_stage("refinement.sampling", sample_refinement)
        except _GroupCfgBatchOom:
            # `_refinement_start` consumes per-image source noise before the
            # first forward. Replay from its generator boundary, not from the
            # partially initialized latent, and retain prepared inputs.
            self.batch_cfg = False
            published_steps.clear()
            for generator, state in zip(generators, generator_states):
                generator.set_state(state)
            self.last_generation_metrics["cfg_batch_fallback"] = {
                "from": "batched", "to": "sequential", "reason": "cuda_oom",
                "stage": "refinement.sampling", "attempts": 1,
                "generator_states_restored": True,
            }
            fallback_history.append({
                "from": "resident_batched", "to": "resident_sequential", "reason": "cuda_oom",
                "stage": "refinement.sampling", "attempt": 1, "generator_restored": True,
                "callback_buffer_discarded": True, "chunk_index": 0,
            })
            _empty_cuda_cache()
            try:
                latents = self._run_cuda_stage("refinement.sampling", sample_refinement)
            except BaseException as sequential_error:
                if not ("outofmemory" in type(sequential_error).__name__.lower() or "out of memory" in str(sequential_error).lower()):
                    raise
                if self.transformer_group_offload_enabled:
                    raise
                self._park_transformer_on_cpu()
                self.keep_transformer_resident = False
                _empty_cuda_cache()
                self.enable_transformer_group_offload(1)
                published_steps.clear()
                for generator, state in zip(generators, generator_states):
                    generator.set_state(state)
                latents = self._run_cuda_stage("refinement.sampling", sample_refinement)
                self.last_generation_metrics["sampling_fallback"] = {
                    "from": "staged_transformer_resident",
                    "to": "staged_transformer_group_offload",
                    "reason": "cuda_oom",
                    "stage": "refinement.sampling",
                    "attempts": 1,
                    "generator_states_restored": True,
                }
                fallback_history.append({
                    "from": "resident_sequential", "to": "group_sequential", "reason": "cuda_oom",
                    "stage": "refinement.sampling", "attempt": 1, "generator_restored": True,
                    "callback_buffer_discarded": True, "chunk_index": 0,
                })
        if transactional_callbacks:
            for step, total, latent in published_steps:
                on_step(step, total, latent)
            published_steps.clear()
        execution = getattr(self, "_last_sampling_execution", {})
        self.last_generation_metrics.setdefault("refinement.sampling", {}).update({
            **schedule_diagnostics,
            "transformer_input_dtype": str(self.dtype).replace("torch.", ""),
            "latent_state_dtype": str(self._latent_state_dtype()).replace("torch.", ""),
            "conditioning_reused": conditioning_reused,
            "sigmas_reused": prepared_sigmas is not None,
            "requested_steps": int(steps),
            "executed_denoise_updates": int(steps),
            "schedule_construction_steps": int(schedule_diagnostics["schedule_steps"]),
            "branch_invocations_per_update": 1 if not refinement_do_cfg else (1 if execution.get("cfg_batch_attempts") else 2),
            "sequential_transformer_invocations": int(steps) * (2 if refinement_do_cfg else 1),
            "actual_transformer_invocations": int(execution.get("actual_transformer_invocations", int(steps) * (2 if refinement_do_cfg else 1))),
            "peak_batch_copies": int(execution.get("peak_batch_copies", 1)),
            "fallback_history": fallback_history,
            "fallback_history_known_count": len(fallback_history),
            "fallback_history_complete": True,
        })
        return self._run_cuda_stage(
            "refinement.vae_decode", lambda: self._decode(latents, force_tiled_decode=force_tiled_decode)
        )

    def to_cpu(self):
        if self._closed:
            return
        if getattr(self, "_poisoned", False):
            self.close()
            raise RuntimeError("Discarded poisoned Anima runtime after interrupted transformer execution")
        for module in (self.text_encoder, self.llm_adapter, self.vae):
            module.to("cpu")
        if self.transformer_group_offload_enabled:
            if any(parameter.device.type != "cpu" for parameter in self.transformer.parameters()):
                self._poisoned = True
                self.close()
                raise RuntimeError("Discarded Anima runtime with an incomplete transformer group offload")
        else:
            self._park_transformer_on_cpu()
        _empty_cuda_cache()

    def close(self):
        if self._closed:
            return
        if self.transformer_group_offload_enabled:
            try:
                self._remove_transformer_group_offload()
            except Exception:
                pass
        for module in (self.text_encoder, self.llm_adapter, self.transformer, self.vae):
            if module is not None:
                try:
                    _discard_module_storage(module)
                except Exception:
                    pass
        self.components.clear()
        self.transformer = None
        self.text_encoder = None
        self.llm_adapter = None
        self.vae = None
        self.qwen_tokenizer = None
        self.t5_tokenizer = None
        # Pinned pages are not swappable, so a discarded runtime must not keep
        # holding several GB of them.
        self._release_transformer_host_buffers()
        self._closed = True
        self._transformer_resident = False
        _empty_cuda_cache()


class CosmosPAGIdentitySelfAttnProcessor:
    """Replace Cosmos self-attention weights with identity while preserving V/output projections."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        del image_rotary_emb
        if encoder_hidden_states is not None or attention_mask is not None:
            raise RuntimeError("Cosmos PAG identity attention was attached to a non-self-attention path")
        hidden_states = attn.to_v(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def load_anima_runtime(
    diffusion_path,
    text_encoder_path,
    vae_path,
    qwen_tokenizer_path,
    t5_tokenizer_path,
    dtype=torch.bfloat16,
    loras=None,
) -> AnimaRuntime:
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Anima dtype must be torch.float16, torch.bfloat16, or torch.float32")
    if dtype == torch.bfloat16 and not _cuda_bf16_supported():
        raise RuntimeError("torch.bfloat16 Anima loading requires a BF16-capable CUDA device")

    diffusion_path = _require_safetensors(diffusion_path, "Anima diffusion")
    text_encoder_path = _require_safetensors(text_encoder_path, "Qwen3 text encoder")
    vae_path = _require_safetensors(vae_path, "Qwen Image VAE")
    qwen_tokenizer_path = _require_tokenizer_json(qwen_tokenizer_path, "Qwen")
    t5_tokenizer_path = _require_tokenizer_json(t5_tokenizer_path, "T5")
    lora_descriptors = _lora_descriptors(loras)
    deps = _runtime_dependencies()

    lora_plans = []
    # Bound before the branch, not inside it: a name assigned on only one path is
    # exactly the fault this module has already been bitten by twice.
    target_modules: dict[str, tuple[str, str]] = {}
    if lora_descriptors:
        diffusion_state = normalize_checkpoint_keys(deps["load_file"](str(diffusion_path), device="cpu"))
        qwen_state = _strip_qwen_model_prefix(deps["load_file"](str(text_encoder_path), device="cpu"))
        qwen_state.pop("lm_head.weight", None)
        target_modules = _lora_target_modules(list(diffusion_state), deps["convert_cosmos"])
        diffusion_shapes = {key: tuple(value.shape) for key, value in diffusion_state.items()}
        qwen_shapes = {key: tuple(value.shape) for key, value in qwen_state.items()}
        for lora_path, multiplier in lora_descriptors:
            lora_state = deps["load_file"](str(lora_path), device="cpu")
            label = f"Anima LoRA {lora_path}"
            specs = _analyze_anima_lora_patch(
                lora_state,
                list(diffusion_state),
                list(qwen_state),
                multiplier,
                label=label,
            )
            # Every target is proven to name a live weight of the right shape
            # before anything is loaded onto the GPU, so an unusable LoRA is
            # reported against the file that caused it rather than from inside
            # the first sampling step.
            _require_live_lora_targets(specs, target_modules, diffusion_shapes, qwen_shapes, label)
            if specs:
                lora_plans.append((str(lora_path), multiplier, specs))
            del lora_state
        transformer, adapter = _load_transformer_and_adapter(diffusion_path, dtype, deps, diffusion_state)
        text_encoder = _load_text_encoder(text_encoder_path, dtype, deps, qwen_state)
    else:
        transformer, adapter = _load_transformer_and_adapter(diffusion_path, dtype, deps)
        text_encoder = _load_text_encoder(text_encoder_path, dtype, deps)
    vae = _load_vae(vae_path, dtype, deps)
    qwen_tokenizer, t5_tokenizer = _load_tokenizers(qwen_tokenizer_path, t5_tokenizer_path, deps)
    runtime = AnimaRuntime(transformer, text_encoder, adapter, vae, qwen_tokenizer, t5_tokenizer, dtype)
    runtime.lora_plans = lora_plans
    runtime.lora_target_modules = target_modules
    runtime._pin_transformer()
    runtime._use_async_transfer = True
    # Comfy's ancestral re-noise comes from a generator built on the *sampling* device and seeded
    # with the sampler seed (`k_diffusion/sampling.py:default_noise_sampler`, which only adds the
    # +1 offset when the latent is on CPU; `samplers.py:outer_sample` moves it to CUDA first).
    # Continuing the CPU initial-latent stream instead would diverge at the first injection, so the
    # matched default is a fresh CUDA generator whenever CUDA is actually available.
    default_noise_device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime.noise_device = os.environ.get("XIRAI_ANIMA_NOISE_DEVICE", default_noise_device).strip().lower()
    if runtime.noise_device not in {"cpu", "cuda"}:
        raise RuntimeError("XIRAI_ANIMA_NOISE_DEVICE must be cpu or cuda")
    return runtime
