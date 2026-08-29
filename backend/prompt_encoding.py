"""ComfyUI-style CLIP weighting and long-prompt conditioning for Diffusers."""

from __future__ import annotations

import math

import torch


_ESCAPED_OPEN = "\x00open"
_ESCAPED_CLOSE = "\x00close"
_MAX_PROMPT_WEIGHT_NESTING = 64


def parse_prompt_weights(text: str, current_weight: float = 1.0, *, _depth: int = 0):
    """Match ComfyUI's `(term)` and `(term:weight)` parenthesis semantics."""
    if not isinstance(text, str):
        raise TypeError("prompt text must be a string")
    if not isinstance(current_weight, (int, float)) or isinstance(current_weight, bool):
        raise ValueError("prompt weight must be a finite number")
    current_weight = float(current_weight)
    if not math.isfinite(current_weight):
        raise ValueError("prompt weight must be finite")
    if _depth > _MAX_PROMPT_WEIGHT_NESTING:
        raise ValueError(f"prompt weight nesting cannot exceed {_MAX_PROMPT_WEIGHT_NESTING} levels")
    text = text.replace(r"\(", _ESCAPED_OPEN).replace(r"\)", _ESCAPED_CLOSE)
    segments = _split_parentheses(text, max_nesting=_MAX_PROMPT_WEIGHT_NESTING - _depth)
    parsed = []
    for segment in segments:
        if len(segment) >= 2 and segment[0] == "(" and segment[-1] == ")":
            content = segment[1:-1]
            weight = current_weight * 1.1
            if not math.isfinite(weight):
                raise ValueError("prompt weight must be finite")
            separator = content.rfind(":")
            if separator > 0:
                try:
                    explicit_weight = float(content[separator + 1:])
                except ValueError:
                    pass
                else:
                    if not math.isfinite(explicit_weight):
                        raise ValueError("explicit prompt weight must be finite")
                    weight = explicit_weight
                    content = content[:separator]
            parsed.extend(parse_prompt_weights(content, weight, _depth=_depth + 1))
        else:
            parsed.append((segment.replace(_ESCAPED_OPEN, "(").replace(_ESCAPED_CLOSE, ")"), current_weight))
    return parsed


def _split_parentheses(text: str, *, max_nesting: int = _MAX_PROMPT_WEIGHT_NESTING):
    segments = []
    current = ""
    nesting = 0
    for character in text:
        if character == "(":
            if nesting == 0 and current:
                segments.append(current)
                current = "("
            else:
                current += character
            nesting += 1
            if nesting > max_nesting:
                raise ValueError(f"prompt weight nesting cannot exceed {_MAX_PROMPT_WEIGHT_NESTING} levels")
        elif character == ")":
            nesting -= 1
            if nesting == 0:
                segments.append(current + ")")
                current = ""
            else:
                current += character
        else:
            current += character
    if current:
        segments.append(current)
    return segments


def _tokenize_without_special_tokens(tokenizer, text: str):
    return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]


def tokenize_weighted_prompt(
    tokenizer,
    text: str,
    *,
    max_length: int,
    truncation: bool = True,
    padding: str = "max_length",
    add_special_tokens: bool = True,
):
    """Tokenize a weighted prompt into one fixed-length, tokenizer-aligned sequence."""
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if padding not in {"max_length", False, None}:
        raise ValueError("padding must be 'max_length' or disabled")

    content_ids = []
    content_weights = []
    for segment, weight in parse_prompt_weights(text):
        segment_ids = list(_tokenize_without_special_tokens(tokenizer, segment))
        content_ids.extend(segment_ids)
        content_weights.extend([float(weight)] * len(segment_ids))

    if add_special_tokens:
        if not hasattr(tokenizer, "build_inputs_with_special_tokens") or not hasattr(tokenizer, "get_special_tokens_mask"):
            raise ValueError("tokenizer cannot build aligned special tokens")
        empty_with_specials = list(tokenizer.build_inputs_with_special_tokens([]))
        available_content = max_length - len(empty_with_specials)
        if available_content < 0:
            raise ValueError("max_length is too short for tokenizer special tokens")
        if len(content_ids) > available_content:
            if not truncation:
                raise ValueError(
                    f"weighted prompt uses {len(content_ids) + len(empty_with_specials)} tokens, exceeding max_length={max_length}"
                )
            if getattr(tokenizer, "truncation_side", "right") == "left":
                content_ids = content_ids[-available_content:] if available_content else []
                content_weights = content_weights[-available_content:] if available_content else []
            else:
                content_ids = content_ids[:available_content]
                content_weights = content_weights[:available_content]
        input_ids = list(tokenizer.build_inputs_with_special_tokens(content_ids))
        special_mask = list(tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True))
        if len(input_ids) != len(special_mask):
            raise ValueError("tokenizer returned an invalid special-token mask")
        weights = []
        content_index = 0
        for is_special in special_mask:
            if is_special:
                weights.append(1.0)
            else:
                if content_index >= len(content_weights):
                    raise ValueError("tokenizer special-token mask does not align with content tokens")
                weights.append(content_weights[content_index])
                content_index += 1
        if content_index != len(content_weights):
            raise ValueError("tokenizer special-token mask omitted content tokens")
    else:
        input_ids = content_ids
        weights = content_weights

    if len(input_ids) > max_length:
        if not truncation:
            raise ValueError(f"weighted prompt uses {len(input_ids)} tokens, exceeding max_length={max_length}")
        if getattr(tokenizer, "truncation_side", "right") == "left":
            input_ids = input_ids[-max_length:]
            weights = weights[-max_length:]
        else:
            input_ids = input_ids[:max_length]
            weights = weights[:max_length]

    attention_mask = [1] * len(input_ids)
    token_count = len(input_ids)
    weighted_token_count = sum(weight != 1.0 for weight in weights)
    if padding == "max_length" and len(input_ids) < max_length:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer is missing a pad or eos token")
        missing = max_length - len(input_ids)
        pad_ids = [int(pad_token_id)] * missing
        pad_weights = [1.0] * missing
        pad_mask = [0] * missing
        if getattr(tokenizer, "padding_side", "right") == "left":
            input_ids = pad_ids + input_ids
            weights = pad_weights + weights
            attention_mask = pad_mask + attention_mask
        else:
            input_ids.extend(pad_ids)
            weights.extend(pad_weights)
            attention_mask.extend(pad_mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "token_count": token_count,
        "weighted_token_count": weighted_token_count,
    }


def build_weighted_token_batches(tokenizer, text: str):
    """Build fixed-size CLIP blocks without truncating a prompt after 75 tokens."""
    max_length = int(getattr(tokenizer, "model_max_length", 77) or 77)
    if max_length < 3:
        raise ValueError("CLIP tokenizer max length must be at least 3")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("CLIP tokenizer is missing required start or end tokens")

    token_groups = []
    for segment, weight in parse_prompt_weights(text):
        token_groups.append([(token, weight) for token in _tokenize_without_special_tokens(tokenizer, segment)])

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batches = []
    batch = [(tokenizer.bos_token_id, 1.0)]
    for group in token_groups:
        # ComfyUI splits eight-token words to preserve later block capacity.
        # The second condition keeps custom short-context tokenizers safe too.
        is_large = len(group) >= 8 or len(group) > max_length - 2
        while group:
            if len(group) + len(batch) > max_length - 1:
                remaining = max_length - len(batch) - 1
                if is_large:
                    batch.extend(group[:remaining])
                    group = group[remaining:]
                batch.append((tokenizer.eos_token_id, 1.0))
                batch.extend([(pad, 1.0)] * (max_length - len(batch)))
                batches.append(batch)
                batch = [(tokenizer.bos_token_id, 1.0)]
            else:
                batch.extend(group)
                group = []
    batch.append((tokenizer.eos_token_id, 1.0))
    batch.extend([(pad, 1.0)] * (max_length - len(batch)))
    batches.append(batch)
    batches = [([token for token, _ in batch], [weight for _, weight in batch]) for batch in batches]
    return batches


def prompt_diagnostics(tokenizer, text: str):
    """Return UI-safe details about the actual CLIP conditioning blocks used."""
    batches = build_weighted_token_batches(tokenizer, text)
    weighted_tokens = sum(weight != 1.0 for _, weights in batches for weight in weights)
    return {
        "tokens": sum(len(_tokenize_without_special_tokens(tokenizer, segment)) for segment, _ in parse_prompt_weights(text)),
        "blocks": len(batches),
        "weighted_tokens": weighted_tokens,
    }


def _attention_mask(text_encoder, input_ids, tokenizer):
    if not getattr(getattr(text_encoder, "config", None), "use_attention_mask", False):
        return None
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return input_ids.ne(pad).long()


def _encode_blocks(tokenizer, text_encoder, batches, device, penultimate_layer=False):
    input_ids = torch.tensor([token_ids for token_ids, _ in batches], device=device, dtype=torch.long)
    kwargs = {"attention_mask": _attention_mask(text_encoder, input_ids, tokenizer)}
    if penultimate_layer:
        output = text_encoder(input_ids, output_hidden_states=True, **kwargs)
        return output.hidden_states[-2], output[0][:1]
    return text_encoder(input_ids, **kwargs)[0], None


@torch.inference_mode()
def encode_weighted_prompt(tokenizer, text_encoder, text: str, device, penultimate_layer=False):
    """Encode ComfyUI-style blocks and apply its empty-conditioning weight blend."""
    batches = build_weighted_token_batches(tokenizer, text)
    embeddings, pooled = _encode_blocks(tokenizer, text_encoder, batches, device, penultimate_layer)
    empty_batches = build_weighted_token_batches(tokenizer, "")
    empty_embeddings, empty_pooled = _encode_blocks(tokenizer, text_encoder, empty_batches, device, penultimate_layer)
    if any(weight != 1.0 for _, weights in batches for weight in weights):
        weights = torch.tensor([weights for _, weights in batches], device=device, dtype=embeddings.dtype).unsqueeze(-1)
        embeddings = empty_embeddings + (embeddings - empty_embeddings) * weights
    return embeddings.reshape(1, -1, embeddings.shape[-1]), pooled if pooled is not None else empty_pooled, empty_embeddings


def _pad_to_length(embeddings, empty_block, target_length: int):
    if embeddings.shape[1] >= target_length:
        return embeddings[:, :target_length]
    missing = target_length - embeddings.shape[1]
    repeats = (missing + empty_block.shape[1] - 1) // empty_block.shape[1]
    padding = empty_block.repeat(1, repeats, 1)[:, :missing]
    return torch.cat((embeddings, padding), dim=1)


def _align_prompt_pair(prompt_embeddings, negative_embeddings, prompt_empty, negative_empty):
    target_length = max(prompt_embeddings.shape[1], negative_embeddings.shape[1])
    return (
        _pad_to_length(prompt_embeddings, prompt_empty, target_length),
        _pad_to_length(negative_embeddings, negative_empty, target_length),
    )


def _conditioning_device(pipeline):
    return getattr(pipeline, "_execution_device", torch.device("cuda"))


def prepare_prompt_conditioning(pipeline, family: str, prompt: str, negative_prompt: str):
    """Create explicit Diffusers conditioning with ComfyUI prompt semantics."""
    device = _conditioning_device(pipeline)
    if family != "sdxl":
        prompt_embeddings, _, prompt_empty = encode_weighted_prompt(pipeline.tokenizer, pipeline.text_encoder, prompt, device)
        negative_embeddings, _, negative_empty = encode_weighted_prompt(
            pipeline.tokenizer, pipeline.text_encoder, negative_prompt, device
        )
        prompt_embeddings, negative_embeddings = _align_prompt_pair(
            prompt_embeddings, negative_embeddings, prompt_empty, negative_empty
        )
        return {
            "prompt_embeds": prompt_embeddings,
            "negative_prompt_embeds": negative_embeddings,
        }

    prompt_l, _, empty_l = encode_weighted_prompt(
        pipeline.tokenizer, pipeline.text_encoder, prompt, device, penultimate_layer=True
    )
    prompt_g, pooled_prompt, empty_g = encode_weighted_prompt(
        pipeline.tokenizer_2, pipeline.text_encoder_2, prompt, device, penultimate_layer=True
    )
    negative_l, _, negative_empty_l = encode_weighted_prompt(
        pipeline.tokenizer, pipeline.text_encoder, negative_prompt, device, penultimate_layer=True
    )
    negative_g, negative_pooled, negative_empty_g = encode_weighted_prompt(
        pipeline.tokenizer_2, pipeline.text_encoder_2, negative_prompt, device, penultimate_layer=True
    )
    prompt_length = min(prompt_l.shape[1], prompt_g.shape[1])
    negative_length = min(negative_l.shape[1], negative_g.shape[1])
    prompt_embeddings = torch.cat((prompt_l[:, :prompt_length], prompt_g[:, :prompt_length]), dim=-1)
    negative_embeddings = torch.cat((negative_l[:, :negative_length], negative_g[:, :negative_length]), dim=-1)
    prompt_empty = torch.cat((empty_l, empty_g), dim=-1)
    negative_empty = torch.cat((negative_empty_l, negative_empty_g), dim=-1)
    prompt_embeddings, negative_embeddings = _align_prompt_pair(
        prompt_embeddings, negative_embeddings, prompt_empty, negative_empty
    )
    return {
        "prompt_embeds": prompt_embeddings,
        "negative_prompt_embeds": negative_embeddings,
        "pooled_prompt_embeds": pooled_prompt,
        "negative_pooled_prompt_embeds": negative_pooled,
    }
