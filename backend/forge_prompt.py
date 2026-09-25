"""Clean-room Forge-compatible static CLIP prompt conditioning.

The behaviour here is specified by what Forge f2.0.1 observably produces; the
P4.1 differential oracle runs Forge's own parser and chunker beside this module
and compares token ids and multipliers. No Forge source is copied.

This module implements the static prompt kernel: attention syntax, chunking,
and the "Original" emphasis over a batch of texts. Scheduling, alternation and
composable ``AND`` plans live in the neighbouring runtime module.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch


CHUNK_CONTENT_TOKENS = 75
CHUNK_LENGTH = 77
COMMA_BACKTRACK = 20
ROUND_MULTIPLIER = 1.1
SQUARE_MULTIPLIER = 1 / 1.1
MAX_OPEN_GROUPS = 256
BREAK = object()
_BREAK = BREAK
_ESCAPABLE = frozenset("()[]\\")
_PLAIN_STOPS = frozenset("\\()[]:")
# `:number)` closes the innermost open round group with that weight. The number
# is any run of digits and dots, exactly as permissive (and as fallible) as
# Forge's reader; `inf`, `nan` and exponents are therefore literal text.
_EXPLICIT_WEIGHT = re.compile(r":\s*([+-]?[.\d]+)\s*\)")
_BREAK_WORD = re.compile(r"\s*\bBREAK\b\s*")


@dataclass(frozen=True)
class PromptFragment:
    text: str
    weight: float


@dataclass(frozen=True)
class EncodedBranch:
    cross_attention: torch.Tensor
    pooled: torch.Tensor | None
    diagnostics: dict


@dataclass(frozen=True)
class ConditioningPlan:
    family: str
    positive: EncodedBranch
    negative: EncodedBranch
    emphasis: str = "forge"


class PreparedConditioning(Mapping):
    """Diffusers-compatible dummy tensors plus the true unequal-length plan.

    Expanding this object into keyword arguments exposes only the rectangular
    one-block tensors Diffusers needs for preprocessing.  The UNet lease reads
    ``plan`` and replaces those tensors at the final forward boundary.
    """

    def __init__(self, values: dict[str, torch.Tensor], plan: ConditioningPlan | None):
        self._values = dict(values)
        self.plan = plan

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._values)


def _finite_weight(value, label="prompt weight") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _explicit_weight(number: str) -> float:
    try:
        return float(number)
    except ValueError as error:
        raise ValueError(f"explicit prompt weight is not a number: {number[:40]}") from error


def parse_prompt_attention(text: str):
    """Split a prompt into weighted fragments and ``BREAK`` markers.

    Observable rules (Forge f2.0.1):

    * ``\\(`` ``\\)`` ``\\[`` ``\\]`` ``\\\\`` are literal characters; any other
      backslash is dropped.
    * Round and square groups are tracked independently: ``)`` closes the most
      recent open ``(`` even across an open ``[``, and vice versa. A closer
      with nothing open of its kind is literal text.
    * Closing multiplies everything emitted since the opener by 1.1 (round) or
      1/1.1 (square); ``:number)`` closes a round group with that multiplier
      instead. Unclosed groups apply at end of input, round ones first.
    * ``BREAK`` is only recognised in plain text runs. A marker enclosed by a
      group is multiplied like text and then becomes the literal word with
      that weight, which is what Forge does with ``(a BREAK b)``.
    * Adjacent fragments with exactly equal weights are joined before
      tokenising, so ``(a:1.2)(b:1.2)`` tokenises as ``ab``.
    """

    if not isinstance(text, str):
        raise TypeError("prompt text must be a string")
    pieces: list[list] = []
    open_round: list[int] = []
    open_square: list[int] = []

    def emit(value, weight=1.0):
        pieces.append([value, weight])

    def scale_from(start, factor):
        for piece in pieces[start:]:
            piece[1] *= factor

    position = 0
    length = len(text)
    while position < length:
        character = text[position]
        if character == "\\":
            following = text[position + 1:position + 2]
            if following and following in _ESCAPABLE:
                emit(following)
                position += 2
            else:
                emit("")
                position += 1
            continue
        if character in "([":
            if len(open_round) + len(open_square) >= MAX_OPEN_GROUPS:
                raise ValueError(f"prompt emphasis cannot keep more than {MAX_OPEN_GROUPS} groups open")
            (open_round if character == "(" else open_square).append(len(pieces))
            position += 1
            continue
        if character == ":":
            explicit = _EXPLICIT_WEIGHT.match(text, position)
            if explicit is None:
                emit(":")
                position += 1
            elif open_round:
                scale_from(open_round.pop(), _explicit_weight(explicit.group(1)))
                position = explicit.end()
            else:
                emit(explicit.group(0))
                position = explicit.end()
            continue
        if character == ")" or character == "]":
            groups, factor = (open_round, ROUND_MULTIPLIER) if character == ")" else (open_square, SQUARE_MULTIPLIER)
            if groups:
                scale_from(groups.pop(), factor)
            else:
                emit(character)
            position += 1
            continue
        end = position
        while end < length and text[end] not in _PLAIN_STOPS:
            end += 1
        for index, part in enumerate(_BREAK_WORD.split(text[position:end])):
            if index:
                emit("BREAK", -1.0)
            emit(part)
        position = end

    for start in open_round:
        scale_from(start, ROUND_MULTIPLIER)
    for start in open_square:
        scale_from(start, SQUARE_MULTIPLIER)
    if not pieces:
        emit("")

    merged: list[list] = []
    for value, weight in pieces:
        if merged and merged[-1][1] == weight:
            merged[-1][0] += value
        else:
            merged.append([value, weight])

    result: list[PromptFragment | object] = []
    for value, weight in merged:
        if value == "BREAK" and weight == -1.0:
            result.append(BREAK)
        else:
            result.append(PromptFragment(value, _finite_weight(weight)))
    return result


def _token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _comma_token_id(tokenizer):
    vocabulary = getattr(tokenizer, "get_vocab", None)
    if callable(vocabulary):
        comma = vocabulary().get(",</w>")
        if comma is not None:
            return int(comma)
    ids = _token_ids(tokenizer, ",")
    return ids[0] if len(ids) == 1 else None


def build_forge_token_batches(tokenizer, text: str):
    """Build 75-content-token CLIP chunks the way Forge closes them.

    A chunk closes when a token arrives and it is already full. If that token
    is not a comma and the chunk's last comma sits within its final 20
    positions, everything after the comma moves to the next chunk; a comma
    arriving at a full chunk simply starts the next one. ``BREAK`` closes the
    current chunk even when it is empty, so a leading or doubled ``BREAK``
    yields an empty chunk.
    """

    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if bos is None or eos is None:
        raise ValueError("CLIP tokenizer is missing required BOS or EOS tokens")
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = eos if pad is None else pad
    comma = _comma_token_id(tokenizer)
    batches = []
    current: list[tuple[int, float]] = []
    last_comma = -1

    def close(content):
        ids = [int(bos), *(token for token, _weight in content), int(eos)]
        weights = [1.0, *(weight for _token, weight in content), 1.0]
        missing = CHUNK_LENGTH - len(ids)
        ids.extend([int(pad)] * missing)
        weights.extend([1.0] * missing)
        batches.append((ids, weights))

    for item in parse_prompt_attention(text):
        if item is BREAK:
            close(current)
            current, last_comma = [], -1
            continue
        for token in _token_ids(tokenizer, item.text):
            if len(current) == CHUNK_CONTENT_TOKENS:
                split = CHUNK_CONTENT_TOKENS
                if token != comma and last_comma >= CHUNK_CONTENT_TOKENS - COMMA_BACKTRACK:
                    split = last_comma + 1
                close(current[:split])
                current, last_comma = current[split:], -1
            if token == comma:
                last_comma = len(current)
            current.append((token, item.weight))
    if current or not batches:
        close(current)
    return batches


def forge_prompt_diagnostics(tokenizer, text: str):
    batches = build_forge_token_batches(tokenizer, text)
    token_count = sum(sum(token not in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id} for token in ids) for ids, _ in batches)
    return {
        "tokens": token_count,
        "blocks": len(batches),
        "weighted_tokens": sum(weight != 1.0 for _ids, weights in batches for weight in weights),
    }


def _attention_mask(text_encoder, input_ids, tokenizer):
    if not getattr(getattr(text_encoder, "config", None), "use_attention_mask", False):
        return None
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return input_ids.ne(int(pad)).long()


SD_DEFAULT_CLIP_SKIP = 1
SDXL_MINIMUM_CLIP_SKIP = 2


@dataclass(frozen=True)
class TowerLayer:
    """Which hidden state one CLIP tower is read from, counted from the last.

    ``skip`` 1 is the model's own final output. A deeper skip reads
    ``hidden_states[-skip]``; SD then applies the tower's final layer norm, so
    the result stays on the scale the UNet was trained against, while the SDXL
    towers never normalise (their default is already the penultimate layer).
    ``pooled`` asks for the tower's final projected output, which carries the
    same meaning at every skip.
    """

    skip: int
    final_norm: bool
    pooled: bool


def tower_layers(family: str, clip_skip: int | None = None) -> TowerLayer:
    if family == "sdxl":
        skip = SDXL_MINIMUM_CLIP_SKIP if clip_skip is None else int(clip_skip)
        if skip < SDXL_MINIMUM_CLIP_SKIP:
            raise ValueError("SDXL CLIP-L/G towers have a minimum effective CLIP skip of 2")
        return TowerLayer(skip, False, True)
    skip = SD_DEFAULT_CLIP_SKIP if clip_skip is None else int(clip_skip)
    if skip < 1:
        raise ValueError("CLIP skip must be at least 1")
    return TowerLayer(skip, skip > 1, False)


def _final_layer_norm(text_encoder):
    norm = getattr(getattr(text_encoder, "text_model", None), "final_layer_norm", None)
    if norm is None:
        raise ValueError("this text encoder has no final layer norm to apply after CLIP skip")
    return norm


def _encode_rows(tokenizer, text_encoder, rows, device, *, layer: TowerLayer):
    """Encode one chunk position for every text and apply Original emphasis.

    ``rows`` holds one ``(ids, weights)`` chunk per text. Forge normalises the
    whole batch with a single ``mean(z) / mean(z * w)`` factor, so texts that
    are encoded together (the schedule states of one prompt) share it.
    Identical rows are encoded once; the mean still counts every row.
    """

    unique: dict[tuple, int] = {}
    order = []
    for ids, _weights in rows:
        key = tuple(ids)
        if key not in unique:
            unique[key] = len(unique)
        order.append(unique[key])
    ids = torch.tensor([list(key) for key in unique], dtype=torch.long, device=device)
    kwargs = {"attention_mask": _attention_mask(text_encoder, ids, tokenizer)}
    # Skip 1 without a pooled read is SD's historical call, kept byte-for-byte: the final output
    # is already the last hidden state through the final layer norm.
    hidden = layer.skip > 1
    output = text_encoder(ids, output_hidden_states=True, **kwargs) if hidden else text_encoder(ids, **kwargs)
    if hidden:
        states = output.hidden_states
        if layer.skip >= len(states):
            raise ValueError(f"CLIP skip {layer.skip} is deeper than this text encoder's {len(states) - 1} layers")
        encoded = states[-layer.skip]
        if layer.final_norm:
            encoded = _final_layer_norm(text_encoder)(encoded)
    else:
        encoded = output[0]
    pooled = output[0] if layer.pooled else None
    gather = torch.tensor(order, dtype=torch.long, device=encoded.device)
    embeddings = encoded.index_select(0, gather)
    if pooled is not None:
        pooled = pooled.index_select(0, gather.to(pooled.device))
    weights = torch.tensor([row[1] for row in rows], dtype=embeddings.dtype, device=embeddings.device).unsqueeze(-1)
    if not torch.isfinite(weights).all():
        raise ValueError("prompt weights contain NaN or infinity")
    if not torch.all(weights == 1):
        original_mean = embeddings.float().mean()
        weighted = embeddings * weights
        weighted_mean = weighted.float().mean()
        if not torch.isfinite(original_mean) or not torch.isfinite(weighted_mean):
            raise ValueError("prompt emphasis produced a non-finite mean")
        if weighted_mean.abs().item() <= torch.finfo(torch.float32).tiny:
            raise ValueError("prompt emphasis has a zero normalization denominator")
        embeddings = weighted * (original_mean / weighted_mean).to(embeddings.dtype)
    if not torch.isfinite(embeddings).all():
        raise ValueError("prompt emphasis produced NaN or infinity")
    return embeddings, pooled


def _encode_tower(tokenizer, text_encoder, texts, device, *, layer: TowerLayer):
    """Encode texts as one batch: equal chunk counts, empty chunks as filler."""

    per_text = [build_forge_token_batches(tokenizer, text) for text in texts]
    empty = build_forge_token_batches(tokenizer, "")[0]
    blocks = []
    pooled = None
    for index in range(max(len(chunks) for chunks in per_text)):
        rows = [chunks[index] if index < len(chunks) else empty for chunks in per_text]
        embeddings, block_pooled = _encode_rows(tokenizer, text_encoder, rows, device, layer=layer)
        blocks.append(embeddings)
        if index == 0:
            pooled = block_pooled
    return torch.cat(blocks, dim=1), pooled, [len(chunks) for chunks in per_text]


def _pad_tower_blocks(tokenizer, text_encoder, embeddings, device, target_blocks, *, layer: TowerLayer):
    missing = target_blocks - embeddings.shape[1] // CHUNK_LENGTH
    if missing <= 0:
        return embeddings, 0
    empty_rows = [build_forge_token_batches(tokenizer, "")[0]] * embeddings.shape[0]
    empty, _pooled = _encode_rows(tokenizer, text_encoder, empty_rows, device, layer=layer)
    return torch.cat((embeddings, empty.repeat(1, missing, 1)), dim=1), missing


@torch.inference_mode()
def encode_forge_texts(pipeline, family: str, texts, device, *, negative=False, clip_skip: int | None = None) -> list[EncodedBranch]:
    """Encode texts that Forge conditions together, one branch per text.

    One call is one Forge ``get_learned_conditioning`` batch: a static prompt,
    or every schedule state of one prompt. An SDXL negative whose texts are all
    empty is the zero conditioning Forge substitutes; a positive never is.
    ``clip_skip`` is the effective layer for this job; ``None`` is the family
    default (SD 1, SDXL 2), which is the encoding every earlier job ran with.
    """

    texts = list(texts)
    if not texts:
        raise ValueError("at least one prompt text is required")
    layer = tower_layers(family, clip_skip)
    if family != "sdxl":
        embeddings, _pooled, counts = _encode_tower(
            pipeline.tokenizer, pipeline.text_encoder, texts, device, layer=layer
        )
        return [
            EncodedBranch(embeddings[index:index + 1], None, forge_prompt_diagnostics(pipeline.tokenizer, text))
            for index, text in enumerate(texts)
        ]

    local, _local_pooled, local_counts = _encode_tower(
        pipeline.tokenizer, pipeline.text_encoder, texts, device, layer=layer
    )
    global_, pooled, global_counts = _encode_tower(
        pipeline.tokenizer_2, pipeline.text_encoder_2, texts, device, layer=layer
    )
    target = max(local.shape[1], global_.shape[1]) // CHUNK_LENGTH
    local, _local_padding = _pad_tower_blocks(pipeline.tokenizer, pipeline.text_encoder, local, device, target, layer=layer)
    global_, _global_padding = _pad_tower_blocks(pipeline.tokenizer_2, pipeline.text_encoder_2, global_, device, target, layer=layer)
    cross = torch.cat((local, global_), dim=-1)
    if negative and all(text == "" for text in texts):
        cross = torch.zeros_like(cross)
        pooled = torch.zeros_like(pooled)
    branches = []
    for index, text in enumerate(texts):
        local_info = forge_prompt_diagnostics(pipeline.tokenizer, text)
        global_info = forge_prompt_diagnostics(pipeline.tokenizer_2, text)
        branches.append(EncodedBranch(cross[index:index + 1], pooled[index:index + 1], {
            "tokens": max(local_info["tokens"], global_info["tokens"]),
            "blocks": target,
            "weighted_tokens": max(local_info["weighted_tokens"], global_info["weighted_tokens"]),
            "clip_l_blocks": local_counts[index],
            "clip_g_blocks": global_counts[index],
            # Empty chunks appended to this text: batch alignment plus any L/G tower mismatch.
            "clip_l_empty_padding": target - local_counts[index],
            "clip_g_empty_padding": target - global_counts[index],
        }))
    return branches


def encode_forge_branch(pipeline, family: str, text: str, device, *, negative=False, clip_skip: int | None = None) -> EncodedBranch:
    return encode_forge_texts(pipeline, family, [text], device, negative=negative, clip_skip=clip_skip)[0]


def _conditioning_device(pipeline):
    return getattr(pipeline, "_execution_device", torch.device("cuda"))


def _dummy_values(plan: ConditioningPlan):
    positive = plan.positive.cross_attention[:, :CHUNK_LENGTH]
    negative = plan.negative.cross_attention[:, :CHUNK_LENGTH]
    values = {"prompt_embeds": positive, "negative_prompt_embeds": negative}
    if plan.family == "sdxl":
        values.update({
            "pooled_prompt_embeds": plan.positive.pooled,
            "negative_pooled_prompt_embeds": plan.negative.pooled,
        })
    return values


def prepare_prompt_conditioning(pipeline, family: str, prompt: str, negative_prompt: str, *, emphasis="forge", clip_skip: int | None = None):
    """Prepare static SD/SDXL conditioning.

    ``forge`` is the SD/iL product default. ``comfy`` remains available as an
    explicit regression oracle and returns the historical rectangular tensors;
    it has no CLIP skip, so a non-default one is refused there.
    """

    if emphasis not in {"forge", "comfy"}:
        raise ValueError("emphasis must be 'forge' or 'comfy'")
    if emphasis == "comfy":
        if tower_layers(family, clip_skip) != tower_layers(family):
            raise ValueError("the comfy emphasis oracle does not implement CLIP skip")
        try:
            from .prompt_encoding import prepare_prompt_conditioning as prepare_comfy
        except ImportError:
            from prompt_encoding import prepare_prompt_conditioning as prepare_comfy
        return PreparedConditioning(prepare_comfy(pipeline, family, prompt, negative_prompt), None)
    device = _conditioning_device(pipeline)
    positive = encode_forge_branch(pipeline, family, prompt, device, clip_skip=clip_skip)
    negative = encode_forge_branch(pipeline, family, negative_prompt, device, negative=True, clip_skip=clip_skip)
    plan = ConditioningPlan(family=family, positive=positive, negative=negative)
    return PreparedConditioning(_dummy_values(plan), plan)


__all__ = [
    "BREAK", "CHUNK_CONTENT_TOKENS", "CHUNK_LENGTH", "COMMA_BACKTRACK", "ConditioningPlan",
    "EncodedBranch", "PreparedConditioning", "SDXL_MINIMUM_CLIP_SKIP", "SD_DEFAULT_CLIP_SKIP", "TowerLayer",
    "build_forge_token_batches", "encode_forge_branch", "encode_forge_texts", "forge_prompt_diagnostics",
    "parse_prompt_attention", "prepare_prompt_conditioning", "tower_layers",
]
