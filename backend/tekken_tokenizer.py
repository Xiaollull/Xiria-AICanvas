"""Mistral *tekken* byte-level BPE, for the FLUX.2 [dev] text encoder.

FLUX.2 [dev] conditions on Mistral-Small-3.1-24B, whose tokenizer is not a HuggingFace
``tokenizer.json`` but Mistral's own *tekken* format: a flat table of ``(rank, token_bytes)``
pairs plus a block of reserved special-token ranks, split by one regular expression.  ComfyUI
carries that table *inside* the text encoder checkpoint as a ``tekken_model`` tensor of raw JSON
bytes (``comfy/sd.py`` reads it straight out of the state dict), which is why loading a FLUX.2
text encoder needs no pinned tokenizer resource the way FLUX.1's CLIP-L and T5 do.

The encoder below is tiktoken's algorithm rather than a merges table: split the text on the
declared pattern, then merge the adjacent byte pair whose *concatenation* has the lowest rank,
repeatedly, until no pair is in the vocabulary.  That is the definition tekken ranks encode, so it
reproduces Mistral's own tokenizer without having to reconstruct a merge order from it.
"""

import base64
import json
from typing import Mapping

import regex


# Tekken reserves a fixed block of low ids for control tokens; ordinary vocabulary ranks are
# offset past it. Both numbers are declared by the file itself and are read, never assumed.
TEKKEN_CONFIG_KEY = "config"
TEKKEN_SPECIAL_COUNT_KEY = "default_num_special_tokens"
TEKKEN_VOCAB_SIZE_KEY = "default_vocab_size"
TEKKEN_PATTERN_KEY = "pattern"

# The tensor ComfyUI stores the tekken JSON blob under inside a FLUX.2 text encoder checkpoint.
TEKKEN_STATE_DICT_KEY = "tekken_model"

# `<s>`: Mistral's beginning-of-sequence control token, which ComfyUI prepends to every FLUX.2
# prompt (`Mistral3Tokenizer(start_token=1)`).
TEKKEN_BOS_ID = 1


class TekkenTokenizer:
    """A loaded tekken vocabulary, encoding text to Mistral token ids."""

    def __init__(self, ranks: Mapping[bytes, int], special_tokens: Mapping[str, int], pattern: str):
        if not ranks:
            raise ValueError("tekken vocabulary is empty")
        self._ranks = dict(ranks)
        self._special_tokens = dict(special_tokens)
        self._decoder = {token_id: token for token, token_id in self._ranks.items()}
        self._special_decoder = {token_id: text for text, token_id in self._special_tokens.items()}
        self.pattern = pattern
        try:
            self._split = regex.compile(pattern)
        except regex.error as error:
            raise ValueError(f"tekken split pattern is not a usable regular expression: {error}") from error
        if self._special_tokens:
            # Longest first so `[/SYSTEM_PROMPT]` is never matched as `[` plus text.
            alternation = "|".join(
                regex.escape(text) for text in sorted(self._special_tokens, key=len, reverse=True)
            )
            self._special_split = regex.compile(f"({alternation})")
        else:
            self._special_split = None

    @property
    def vocabulary_size(self) -> int:
        return max(max(self._ranks.values(), default=0), max(self._special_tokens.values(), default=0)) + 1

    def _merge_piece(self, piece: bytes) -> list[int]:
        """tiktoken's byte-pair merge: always join the pair with the lowest merged rank."""
        if len(piece) == 1:
            token = self._ranks.get(piece)
            if token is None:
                raise ValueError(f"tekken vocabulary has no entry for byte {piece!r}")
            return [token]
        parts = [piece[index:index + 1] for index in range(len(piece))]
        while len(parts) > 1:
            best_rank = None
            best_index = -1
            for index in range(len(parts) - 1):
                rank = self._ranks.get(parts[index] + parts[index + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_rank is None:
                break
            parts[best_index:best_index + 2] = [parts[best_index] + parts[best_index + 1]]
        ids = []
        for part in parts:
            token = self._ranks.get(part)
            if token is None:
                raise ValueError(f"tekken vocabulary has no entry for byte sequence {part!r}")
            ids.append(token)
        return ids

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = [TEKKEN_BOS_ID] if add_bos else []
        segments = self._special_split.split(text) if self._special_split is not None else [text]
        for segment in segments:
            if not segment:
                continue
            special = self._special_tokens.get(segment)
            if special is not None:
                ids.append(special)
                continue
            for piece in self._split.findall(segment):
                if not piece:
                    continue
                ids.extend(self._merge_piece(piece.encode("utf-8")))
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        buffer = bytearray()
        for token_id in token_ids:
            token_id = int(token_id)
            token = self._decoder.get(token_id)
            if token is not None:
                buffer.extend(token)
                continue
            special = self._special_decoder.get(token_id)
            if special is not None and not skip_special_tokens:
                buffer.extend(special.encode("utf-8"))
        return buffer.decode("utf-8", errors="replace")


def load_tekken_tokenizer(data) -> TekkenTokenizer:
    """Build a tokenizer from a tekken JSON blob (``bytes``, ``str`` or a byte tensor)."""
    if hasattr(data, "numpy") and hasattr(data, "dtype"):
        data = bytes(data.to("cpu").contiguous().numpy().tobytes())
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if isinstance(data, bytes):
        data = data.rstrip(b"\x00")
    try:
        document = json.loads(data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"tekken tokenizer blob is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("tekken tokenizer blob is not a JSON object")

    config = document.get(TEKKEN_CONFIG_KEY)
    if not isinstance(config, dict):
        raise ValueError("tekken tokenizer blob has no config block")
    pattern = config.get(TEKKEN_PATTERN_KEY)
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("tekken tokenizer config declares no split pattern")
    special_count = config.get(TEKKEN_SPECIAL_COUNT_KEY)
    vocabulary_size = config.get(TEKKEN_VOCAB_SIZE_KEY)
    if not isinstance(special_count, int) or isinstance(special_count, bool) or special_count < 0:
        raise ValueError("tekken tokenizer config declares no special-token block size")
    if not isinstance(vocabulary_size, int) or isinstance(vocabulary_size, bool) or vocabulary_size <= special_count:
        raise ValueError("tekken tokenizer config declares an unusable vocabulary size")

    ranks = {}
    # Ranks are ids offset past the reserved control block, and the table is cut at the declared
    # vocabulary size: a file may carry more entries than the model's embedding matrix has rows.
    limit = vocabulary_size - special_count
    for entry in document.get("vocab", ()):
        if not isinstance(entry, dict):
            continue
        rank = entry.get("rank")
        encoded = entry.get("token_bytes")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank >= limit or not isinstance(encoded, str):
            continue
        try:
            token = base64.b64decode(encoded)
        except (ValueError, TypeError) as error:
            raise ValueError(f"tekken vocabulary entry {rank} is not valid base64: {error}") from error
        ranks[token] = rank + special_count

    special_tokens = {}
    for entry in document.get("special_tokens", ()):
        if not isinstance(entry, dict):
            continue
        rank = entry.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            continue
        text = entry.get("token_str")
        if not isinstance(text, str) and isinstance(entry.get("token_bytes"), str):
            text = base64.b64decode(entry["token_bytes"]).decode("utf-8", errors="replace")
        if isinstance(text, str) and text:
            special_tokens[text] = rank

    if len(ranks) < 256:
        raise ValueError("tekken vocabulary is missing its single-byte entries")
    return TekkenTokenizer(ranks, special_tokens, pattern)


def tekken_tokenizer_from_state_dict(state_dict: Mapping[str, object]) -> TekkenTokenizer:
    """Read the tokenizer ComfyUI packs into a FLUX.2 text encoder checkpoint."""
    blob = state_dict.get(TEKKEN_STATE_DICT_KEY)
    if blob is None:
        raise ValueError(
            "This FLUX.2 text encoder carries no embedded tekken tokenizer. ComfyUI's published "
            "flux2 text encoder stores it as a `tekken_model` tensor; place a `tekken.json` beside "
            "the checkpoint if yours was converted from another source."
        )
    return load_tekken_tokenizer(blob)
