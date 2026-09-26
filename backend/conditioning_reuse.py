"""One job's static SD/iL prompt encodings, shared by every stage that asks for the same text.

A job with a Hires redraw and three ADetailer units used to encode the same two strings five times:
once for the base pass, once for the redraw and once per detail unit, each a full CLIP-L/G forward
with its own transient allocation. For a static prompt — no `[a:b:n]` schedule, no `[a|b]`
alternation, no `AND` — every one of those forwards returns the same tensors, because the towers,
the tokens and the layer never change inside a job. The 2026-09-26 12 GB report ran out of memory
inside exactly such a repeat, in the redraw's CLIP-G pass. So a static prompt is encoded once and
the stages share it (QPlan P8 §3.5).

The key is everything the encoding depends on: positive and negative text, CLIP skip, emphasis,
family, the model (checkpoint and the tower objects themselves) and the LoRA fusion signature,
plus the timeline specification — always `static` here, because only prompts with one conditioning
state are ever cached. A change to any of them is a different key and therefore a fresh encode.
A dynamic or composable prompt, or a pipeline whose adapter state is dirty, is never cached at all.

The cache lives for one job and is cleared when it ends, so nothing encoded for one request can
reach another.
"""

import threading


STATIC_TIMELINE = "static"


class StaticConditioningCache:
    def __init__(self):
        self._entries = {}
        self._lock = threading.Lock()
        self.encodes = 0
        self.hits = 0
        self.bypassed = 0

    @staticmethod
    def key(*, family, prompt, negative_prompt, clip_skip, emphasis, model_signature, lora_signature,
            timeline=STATIC_TIMELINE):
        return (
            "sd-static-conditioning",
            str(family),
            str(prompt),
            str(negative_prompt),
            None if clip_skip is None else int(clip_skip),
            str(emphasis),
            tuple(model_signature) if isinstance(model_signature, (list, tuple)) else model_signature,
            tuple(lora_signature) if isinstance(lora_signature, (list, tuple)) else lora_signature,
            str(timeline),
        )

    def get_or_encode(self, key, encode):
        """The shared encoding for `key`, encoding it with `encode()` the first time only."""
        with self._lock:
            if key in self._entries:
                self.hits += 1
                return self._entries[key]
        value = encode()
        with self._lock:
            # Another stage cannot race this one inside a job — stages run in order — but if one
            # ever did, the first stored encoding wins so every stage shares the same tensors.
            existing = self._entries.setdefault(key, value)
            if existing is value:
                self.encodes += 1
            else:
                self.hits += 1
            return existing

    def bypass(self, encode):
        """Encode without caching: a prompt with more than one state, or an unknown adapter state."""
        with self._lock:
            self.bypassed += 1
        return encode()

    def clear(self):
        with self._lock:
            self._entries.clear()

    def stats(self):
        with self._lock:
            return {
                "encodes": self.encodes,
                "reused": self.hits,
                "uncached": self.bypassed,
                "entries": len(self._entries),
            }


def lora_fusion_signature(pipeline):
    """The adapter state a cached encoding depends on, or None when it cannot be trusted.

    A text-encoder LoRA changes what a string encodes to, so the fused set is part of the key. A
    `dirty` state means a failed fold left an unknown prefix applied; nothing is cached against it.
    """
    state = getattr(pipeline, "_xiriai_lora_fusion_state", None)
    if isinstance(state, dict):
        if state.get("state") != "clean":
            return None
        return ("clean", tuple(state.get("signature") or ()), state.get("revision"))
    fused = getattr(pipeline, "_xiriai_fused_loras", None)
    return ("legacy", tuple(fused or ()))


def tower_signature(pipeline, checkpoint=None):
    """Which model is encoding: the checkpoint and the exact tower objects.

    Derived pipelines (image-to-image, inpainting) share the base pipeline's towers, so they share
    the key; a reloaded checkpoint builds new towers and cannot.
    """
    return (
        str(checkpoint) if checkpoint is not None else None,
        id(getattr(pipeline, "text_encoder", None)),
        id(getattr(pipeline, "text_encoder_2", None)),
    )
