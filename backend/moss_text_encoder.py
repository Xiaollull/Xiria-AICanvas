"""Qwen3.5-9B language model for MOSS-Image conditioning, plus its tokenizer contract.

MOSS-Image conditions on the **last hidden state** (after the final norm) of a stock Qwen3.5-9B: 32
hybrid layers, three Gated DeltaNet (linear attention) layers for every gated full-attention layer,
4096 wide. The installed Transformers (4.57) has no Qwen3.5, and the work order forbids upgrading the
main environment, so the language model is ported here in plain PyTorch. Numerics follow the model
authors' reference, ``transformers@v5.12.1 models/qwen3_5/modeling_qwen3_5.py`` (Apache-2.0), term for
term: zero-centred RMSNorm computed in float32 with ``1 + w``, the gated RMSNorm in float32, the
torch chunked gated-delta rule in float32 with ``l2norm(eps=1e-6)`` on q/k, sigmoid output gating
on full attention, partial RoPE over the first 64 of 256 head channels with theta 1e7 and cos/sin
cast to the activation dtype. One deliberate difference, required by the work order: ``A_log`` and
the linear-attention norm weight keep the float32 the checkpoint stores (the reference rounds them
to the load dtype); the recurrence itself runs in float32 in both.

Only the text-to-image subgraph is loaded. A Comfy-Org repackage also carries the vision tower,
``lm_head`` and the multi-token-prediction head; those are dropped through the exact, versioned
allowlist :data:`DISCARDED_SUBGRAPHS` and nothing else unknown is tolerated.

Tokenisation reproduces ComfyUI's text path for MOSS (plugin @9d799c1 and core PR #17095 agree on
it): ``disable_weights=True`` so ``(word:1.2)`` is literal text, the escape pair ``\\(``/``\\)`` loses
its backslash, the text is tokenised in segments split at ``(?<=\\s)embedding:``, no BOS/EOS, no
padding. The prompt is trimmed like the model's chat template (``|trim``) before it is placed in
``<|im_start|>user\\n{}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``, and the whole sequence,
template included, is cut at 4096 tokens -- which is always reported.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .moss_quant import (
        FP8_DTYPES,
        QuantizedEmbedding,
        apply_raw_fp8_linears,
        consumed_embedding_keys,
        embedding_from_state,
        embedding_state_for,
        raw_fp8_layers,
        refuse_unsupported_layouts,
    )
    from .quantized_linear import apply_quantized_linears, quantized_state_dict, scan_quantized_layers
except ImportError:  # pragma: no cover - flat import when run from backend/
    from moss_quant import (
        FP8_DTYPES,
        QuantizedEmbedding,
        apply_raw_fp8_linears,
        consumed_embedding_keys,
        embedding_from_state,
        embedding_state_for,
        raw_fp8_layers,
        refuse_unsupported_layouts,
    )
    from quantized_linear import apply_quantized_linears, quantized_state_dict, scan_quantized_layers

MOSS_TEXT_ENCODER_ARCHITECTURE = "qwen3_5_text"
MOSS_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
MOSS_MAX_TOKENS = 4096
MOSS_TEMPLATE_TOKENS = 10  # the template around an empty prompt, measured against the pinned tokenizer
EMBEDDING_IDENTIFIER = "embedding:"

# Pinned tokenizer: OpenMOSS-Team/MOSS-Image-Base@5f40f07a304541840b6019d2d34d30f966756787
# tokenizer/tokenizer.json, stored as a deterministic gzip (`gzip -9 -n`).
TOKENIZER_JSON_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
TOKENIZER_JSON_SIZE = 19_989_325
TOKENIZER_GZ_NAME = "moss-qwen35-tokenizer.json.gz"

# Architecture constants that leave no trace in the tensors (config.json of the pinned HF snapshot).
QWEN35_9B_TEXT = {
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25,
    "full_attention_interval": 4,
    "conv_kernel_size": 4,
    "vocab_size": 248320,
}

# Versioned exact allowlist of the subgraphs a text-to-image run never evaluates (D10, version 1).
DISCARD_ALLOWLIST_VERSION = 1
DISCARDED_SUBGRAPHS = (
    re.compile(r"^(model\.)?visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])\.(weight|bias|weight_scale|comfy_quant)"
               r"|blocks\.\d+\.norm[12]\.(weight|bias)"
               r"|merger\.(linear_fc[12]|norm)\.(weight|bias|weight_scale|comfy_quant)"
               r"|patch_embed\.proj\.(weight|bias)|pos_embed\.weight)$"),
    re.compile(r"^lm_head\.(weight|weight_scale|comfy_quant)$"),
    re.compile(r"^mtp\.(fc\.weight|norm\.weight|pre_fc_norm_(embedding|hidden)\.weight"
               r"|layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight"
               r"|layers\.\d+\.self_attn\.(q_norm|k_norm)\.weight"
               r"|layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.(weight|weight_scale|comfy_quant))$"),
)
LANGUAGE_PREFIXES = ("model.language_model.", "language_model.")

# Parameters stored in float32 that a blanket cast would corrupt.
FLOAT32_PRESERVED_SUFFIXES = (".linear_attn.A_log", ".linear_attn.norm.weight")


def escape_important(text: str) -> str:
    return text.replace("\\)", "\0\1").replace("\\(", "\0\2")


def unescape_important(text: str) -> str:
    return text.replace("\0\1", ")").replace("\0\2", "(")


def load_tokenizer_json(path: Path) -> tuple[str, dict]:
    """Read the bundled gzip, verifying the decompressed bytes against the pinned upstream file."""
    data = gzip.decompress(Path(path).read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != TOKENIZER_JSON_SIZE or digest != TOKENIZER_JSON_SHA256:
        raise ValueError(
            f"MOSS tokenizer resource does not decompress to the pinned tokenizer.json ({len(data)} bytes, sha256 {digest[:12]}…)"
        )
    return data.decode("utf-8"), {"sha256": digest, "size": len(data)}


class MossTokenizer:
    """The ComfyUI MOSS text path over the pinned Qwen3.5 tokenizer.json."""

    def __init__(self, tokenizer_json: str, max_tokens: int = MOSS_MAX_TOKENS):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_str(tokenizer_json)
        self.max_tokens = int(max_tokens)
        self.vocab_size = int(self._tokenizer.get_vocab_size(with_added_tokens=True))
        self.pad_token = 248044

    @classmethod
    def from_resource(cls, path) -> "MossTokenizer":
        text, _record = load_tokenizer_json(Path(path))
        return cls(text)

    def _ids(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def encode(self, prompt: str) -> tuple[list[int], dict]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        trimmed = prompt.strip()
        templated = MOSS_TEMPLATE.format(trimmed)
        text = unescape_important(escape_important(templated))
        segments = re.split(r"(?<=\s){}".format(re.escape(EMBEDDING_IDENTIFIER)), text)
        pieces = [segments[0]] + [EMBEDDING_IDENTIFIER + segment for segment in segments[1:]]
        ids: list[int] = []
        for piece in pieces:
            if piece:
                ids.extend(self._ids(piece))
        if not ids:
            ids = [self.pad_token]
        full_length = len(ids)
        truncated = full_length > self.max_tokens
        if truncated:
            ids = ids[: self.max_tokens]
        return ids, {
            "token_count": len(ids),
            "untruncated_token_count": full_length,
            "truncated": truncated,
            "max_tokens": self.max_tokens,
            "template_tokens": MOSS_TEMPLATE_TOKENS,
            "prompt_trimmed": trimmed != prompt,
            "escapes_removed": templated != text,
            "segments": len([p for p in pieces if p]),
            "weighted_token_count": 0,
        }


# -- model ----------------------------------------------------------------------------------------


@dataclass
class Qwen35TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    layer_types: list = field(default_factory=list)
    conv_kernel_size: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


class Qwen35RMSNorm(nn.Module):
    """Zero-centred: ``norm(x.float()) * (1 + w.float())``, then back to the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * (1.0 + self.weight.float())).type_as(x)


class Qwen35RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(input_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(input_dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size: int = 64):
    """``torch_chunk_gated_delta_rule`` with ``use_qk_l2norm_in_kernel=True`` and no initial state."""
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
    core = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        attn = attn.masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core[:, :, i] = attn_inter + attn @ v_new
        state = state * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    core = core.reshape(core.shape[0], core.shape[1], -1, core.shape[-1])[:, :, :sequence_length]
    return core.transpose(1, 2).contiguous().to(initial_dtype)


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.conv_kernel_size
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim, padding=self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core = chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(batch_size, seq_len, -1))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotary_cos_sin(config: Qwen35TextConfig, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only interleaved M-RoPE: all three axes carry the same position, so it is plain RoPE."""
    dim = config.rotary_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


class Qwen35Attention(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        rotary = cos.shape[-1]
        q_rot, q_pass = query[..., :rotary], query[..., rotary:]
        k_rot, k_pass = key[..., :rotary], key[..., rotary:]
        query = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
        key = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=self.scaling)
        output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        output = output * torch.sigmoid(gate)
        return self.o_proj(output)


class Qwen35MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35TextConfig, index: int):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config)
        else:
            self.self_attn = Qwen35Attention(config)
        self.mlp = Qwen35MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class Qwen35TextModel(nn.Module):
    """The language model only; ``forward`` returns the final-norm last hidden state."""

    def __init__(self, config: Qwen35TextConfig, compute_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if isinstance(self.embed_tokens, QuantizedEmbedding):
            return self.embed_tokens(input_ids, out_dtype=self.compute_dtype)
        return self.embed_tokens(input_ids.to(self.embed_tokens.weight.device)).to(self.compute_dtype)

    def forward(self, input_ids: torch.Tensor, device=None, layer_hook=None) -> torch.Tensor:
        """``layer_hook(index, layer)`` may place a layer before it runs and is told when it is done."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        hidden = self.embed(input_ids)
        device = torch.device(device) if device is not None else hidden.device
        hidden = hidden.to(device)
        cos, sin = rotary_cos_sin(self.config, hidden.shape[1], device, hidden.dtype)
        for index, layer in enumerate(self.layers):
            if layer_hook is not None:
                layer_hook("enter", index, layer)
            hidden = layer(hidden, cos, sin)
            if layer_hook is not None:
                layer_hook("exit", index, layer)
        if layer_hook is not None:
            layer_hook("enter", -1, self.norm)
        out = self.norm(hidden)
        if layer_hook is not None:
            layer_hook("exit", -1, self.norm)
        return out


# -- loading ------------------------------------------------------------------------------------


def _layer_count(keys) -> int:
    indices = {int(m.group(1)) for key in keys if (m := re.match(r"^layers\.(\d+)\.", key))}
    if not indices:
        return 0
    if indices != set(range(max(indices) + 1)):
        raise ValueError("Qwen3.5 text encoder has non-contiguous layer indices")
    return max(indices) + 1


def split_text_encoder_state(state: Mapping[str, torch.Tensor]) -> tuple[dict, dict]:
    """``(language-model tensors under canonical names, report)``; refuse anything unrecognised."""
    language, discarded, unknown = {}, [], []
    prefixes = [prefix for prefix in LANGUAGE_PREFIXES if any(key.startswith(prefix) for key in state)]
    if len(prefixes) > 1:
        raise ValueError("Qwen3.5 text encoder mixes language-model prefixes; the file layout is ambiguous")
    prefix = prefixes[0] if prefixes else None
    for key, value in state.items():
        if prefix is not None and key.startswith(prefix):
            language[key[len(prefix):]] = value
        elif any(pattern.match(key) for pattern in DISCARDED_SUBGRAPHS):
            discarded.append(key)
        else:
            unknown.append(key)
    if prefix is None:
        raise ValueError("The selected file is not a Qwen3.5 text encoder: no language-model weights were found")
    if unknown:
        raise ValueError(f"Qwen3.5 text encoder carries tensors outside the language model and the versioned discard list: {sorted(unknown)[:4]}")
    return language, {
        "language_prefix": prefix,
        "discarded_tensors": len(discarded),
        "discarded_subgraphs": sorted({key.split(".")[0] if not key.startswith("model.") else "model." + key.split(".")[1] for key in discarded}),
        "discard_allowlist_version": DISCARD_ALLOWLIST_VERSION,
    }


def classify_text_encoder(keys) -> str:
    placeholder = {key: None for key in keys}
    try:
        language, _report = split_text_encoder_state(placeholder)
    except ValueError:
        return "unknown"
    has_linear = any(".linear_attn.in_proj_qkv." in key for key in language)
    has_gated = any(".self_attn.q_norm." in key for key in language)
    return "qwen3_5" if has_linear and has_gated else "unknown"


def _shape(state, key: str) -> tuple[int, ...]:
    value = state.get(key)
    if value is None:
        raise ValueError(f"Qwen3.5 text encoder is missing {key}")
    return tuple(int(v) for v in value.shape)


def infer_text_config(language: Mapping[str, torch.Tensor], embedding_rows: int | None = None) -> Qwen35TextConfig:
    layers = _layer_count(language)
    if layers < 1:
        raise ValueError("Qwen3.5 text encoder has no decoder layers")
    layer_types = []
    for index in range(layers):
        linear = f"layers.{index}.linear_attn.in_proj_qkv.weight" in language
        full = f"layers.{index}.self_attn.q_proj.weight" in language
        if linear == full:
            raise ValueError(f"Qwen3.5 layer {index} is neither a linear-attention nor a full-attention layer")
        layer_types.append("linear_attention" if linear else "full_attention")
    interval = QWEN35_9B_TEXT["full_attention_interval"]
    expected_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
    if layer_types != expected_types:
        raise ValueError("Qwen3.5 text encoder layer pattern is not three linear-attention layers per full-attention layer")
    embed_rows, hidden = _shape(language, "embed_tokens.weight")
    full_index = layer_types.index("full_attention")
    linear_index = layer_types.index("linear_attention")
    head_dim = _shape(language, f"layers.{full_index}.self_attn.q_norm.weight")[0]
    q_out = _shape(language, f"layers.{full_index}.self_attn.q_proj.weight")[0]
    k_out = _shape(language, f"layers.{full_index}.self_attn.k_proj.weight")[0]
    if q_out % (2 * head_dim) or k_out % head_dim:
        raise ValueError("Qwen3.5 gated attention projections are not whole heads")
    heads = q_out // (2 * head_dim)
    kv_heads = k_out // head_dim
    intermediate = _shape(language, "layers.0.mlp.gate_proj.weight")[0]
    v_heads = _shape(language, f"layers.{linear_index}.linear_attn.A_log")[0]
    z_out = _shape(language, f"layers.{linear_index}.linear_attn.in_proj_z.weight")[0]
    if z_out % v_heads:
        raise ValueError("Qwen3.5 linear-attention value projection is not whole heads")
    head_v = z_out // v_heads
    qkv_out = _shape(language, f"layers.{linear_index}.linear_attn.in_proj_qkv.weight")[0]
    key_dim2 = qkv_out - z_out
    if key_dim2 % 2:
        raise ValueError("Qwen3.5 linear-attention q/k projection is not an even pair")
    if _shape(language, f"layers.{linear_index}.linear_attn.norm.weight")[0] != head_v:
        raise ValueError("Qwen3.5 linear-attention norm width does not match its value heads")
    # The q/k head width leaves no trace of its own: in_proj_qkv only records 2 * heads * width.
    # Every Qwen3.5 size uses equal key and value head widths (128), which is what fixes the split.
    head_k = head_v
    if (key_dim2 // 2) % head_k:
        raise ValueError("Qwen3.5 linear-attention key projection is not whole heads")
    k_heads = (key_dim2 // 2) // head_k
    if v_heads % k_heads:
        raise ValueError("Qwen3.5 value heads are not a multiple of key heads")
    config = Qwen35TextConfig(
        vocab_size=int(embedding_rows or embed_rows), hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
        linear_num_key_heads=k_heads, linear_num_value_heads=v_heads, linear_key_head_dim=head_k,
        linear_value_head_dim=head_v, layer_types=layer_types,
        conv_kernel_size=_shape(language, f"layers.{linear_index}.linear_attn.conv1d.weight")[-1],
        rms_norm_eps=QWEN35_9B_TEXT["rms_norm_eps"], rope_theta=QWEN35_9B_TEXT["rope_theta"],
        partial_rotary_factor=QWEN35_9B_TEXT["partial_rotary_factor"],
    )
    if config.rotary_dim < 2 or config.rotary_dim % 2:
        raise ValueError("Qwen3.5 partial rotary dimension must be even")
    return config


def _preserved(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in FLOAT32_PRESERVED_SUFFIXES)


def build_text_encoder(state: Mapping[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16, label: str = "MOSS text encoder"):
    """Build and strictly load the Qwen3.5 language model from a Comfy-Org or HF state dict."""
    language, report = split_text_encoder_state(state)
    quantized = scan_quantized_layers(language, label)
    refuse_unsupported_layouts(quantized, label)
    embedding_quantized = "embed_tokens" in quantized or (
        "embed_tokens.weight" in language and language["embed_tokens.weight"].dtype in FP8_DTYPES
    )
    linear_quantized = {layer: spec for layer, spec in quantized.items() if layer != "embed_tokens"}
    config = infer_text_config(language)
    with torch.device("meta"):
        model = Qwen35TextModel(config, compute_dtype=dtype)
    model.compute_dtype = dtype
    kept, deferred = apply_quantized_linears(model, linear_quantized, language, dtype, label)
    if deferred:
        raise ValueError(f"{label} quantises non-linear tensors this runtime cannot keep stored: {sorted(deferred)[:4]}")
    replaced = set(kept)
    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    raw_fp8 = raw_fp8_layers(language, replaced, linear_names)
    apply_raw_fp8_linears(model, raw_fp8, language, dtype)
    weights = {}
    consumed = set()
    if embedding_quantized:
        embedding = embedding_from_state("embed_tokens", language, dtype, label)
        if (embedding.num_embeddings, embedding.embedding_dim) != (config.vocab_size, config.hidden_size):
            raise ValueError(f"{label} embedding table does not match the model width")
        model.embed_tokens = embedding
        for name, tensor in embedding_state_for("embed_tokens", language).items():
            weights[f"embed_tokens.{name}"] = tensor
        consumed |= consumed_embedding_keys("embed_tokens", language)
    remaining = {key: value for key, value in language.items() if key not in consumed}
    remaining = quantized_state_dict(remaining, linear_quantized, replaced)
    protected = {f"{layer}.weight" for layer in replaced} | {f"{layer}.scale_weight" for layer in replaced}
    protected |= {f"{layer}.weight" for layer in raw_fp8}
    for key, value in remaining.items():
        if key in protected or not value.is_floating_point():
            weights[key] = value
        elif _preserved(key):
            weights[key] = value.to(torch.float32)
        elif value.dtype in FP8_DTYPES:
            raise ValueError(f"{label} stores {key} in FP8 outside a linear weight; that layout is not defined")
        else:
            weights[key] = value.to(dtype)
    incompatible = model.load_state_dict(weights, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict {label} load failed: missing={incompatible.missing_keys[:4]}, unexpected={incompatible.unexpected_keys[:4]}")
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_meta:
            raise RuntimeError(f"{label} left {name} unloaded")
    report.update({
        "architecture": MOSS_TEXT_ENCODER_ARCHITECTURE,
        "layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "quantized_linears": len(replaced),
        "raw_fp8_linears": len(raw_fp8),
        "quantized_embedding": embedding_quantized,
        "float32_preserved": sorted({key.rsplit(".", 1)[0].split(".")[-1] + "." + key.rsplit(".", 1)[-1]
                                     for key in weights if _preserved(key)}),
        "layouts": sorted({spec.layer_config.label for layer, spec in linear_quantized.items() if layer in replaced}),
    })
    return model.eval().requires_grad_(False), config, report


def text_encoder_runtime_bytes(model: nn.Module) -> int:
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def layer_runtime_bytes(model: Qwen35TextModel) -> list[int]:
    return [sum(t.numel() * t.element_size() for t in list(layer.parameters()) + list(layer.buffers())) for layer in model.layers]
