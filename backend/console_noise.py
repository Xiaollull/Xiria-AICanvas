"""Library messages that are false about how SD/iL drives the library, and how each is kept quiet.

Every message handled here was printed on a successful SD/iL job, in the colours and with the prefix
of an error, and each one is wrong about this app:

    "Token indices sequence length is longer than the specified maximum sequence length for this
    model (147 > 77). Running this sequence through the model will result in indexing errors"
        `prompt_encoding` tokenizes a prompt without truncation on purpose and then packs the ids into
        77-token CLIP blocks (`build_weighted_token_batches`). Nothing longer than 77 ever reaches
        the text encoder, so the indexing error it predicts cannot happen.

    "No LoRA keys associated to CLIPTextModel found with the prefix='text_encoder'. This is safe to
    ignore if LoRA state dict didn't originally have any CLIPTextModel related params."
        Most style LoRAs train only the UNet. Diffusers cannot tell that case from a text-encoder
        LoRA whose keys it failed to map, so it warns for both — here the file's own header decides,
        and a LoRA that *does* carry text-encoder weights still gets the warning.

    "Already found a `peft_config` attribute in the model. This will lead to having multiple
    adapters in the model."
        Several adapters are exactly what a LoRA stack is, and they are fused into the weights
        straight after loading.

    "Loading pipeline components...: 100%|██████████| 7/7"
        Not a message but a progress bar, printed beside the console's own "Model ready · loaded in
        2.0s" line for the same event, and the only one of its kind: the native engines load quietly.

Each is silenced at the one call that provokes it and only for as long as that call runs, never by a
process-wide filter — the same text from anywhere else still reaches the console.
"""

import logging
import re
import warnings
from contextlib import contextmanager


# The two Diffusers modules that emit the missing-text-encoder message.
LORA_LOGGERS = ("diffusers.loaders.lora_base", "diffusers.loaders.peft")

# How each LoRA format names the weights of a text encoder. Kohya writes `lora_te_` for SD 1.x's
# single encoder and `lora_te1_` / `lora_te2_` for SDXL's pair; Diffusers writes the component name.
TEXT_ENCODER_KEY_PREFIXES = {
    "text_encoder": ("lora_te_", "lora_te1_", "text_encoder."),
    "text_encoder_2": ("lora_te2_", "text_encoder_2."),
}

_MISSING_TEXT_ENCODER = re.compile(r"No LoRA keys associated to \S+ found with the prefix='([^']+)'")
_MULTIPLE_ADAPTERS = r"Already found a `peft_config` attribute in the model"

# The flag transformers sets on a tokenizer once it has printed the too-long-sequence warning.
LONG_SEQUENCE_WARNING = "sequence-length-is-longer-than-the-specified-maximum"


def lora_text_encoders(path):
    """Which text encoders a LoRA file carries weights for, read from its header alone.

    Returns None when the file cannot be read that way — a pickled `.pt`, or a damaged header —
    and None means "unknown", which never silences anything.
    """
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            keys = list(handle.keys())
    except Exception:
        return None
    return {
        encoder for encoder, prefixes in TEXT_ENCODER_KEY_PREFIXES.items()
        if any(key.startswith(prefixes) for key in keys)
    }


class _AbsentTextEncoderFilter(logging.Filter):
    """Drop "no LoRA keys for <encoder>" only for encoders the file provably has no weights for."""

    def __init__(self, absent):
        super().__init__()
        self.absent = frozenset(absent)

    def filter(self, record):
        match = _MISSING_TEXT_ENCODER.match(record.getMessage())
        return not (match and match.group(1) in self.absent)


@contextmanager
def quiet_lora_loading(path):
    """Load one LoRA without the two messages that are false about it."""
    present = lora_text_encoders(path)
    absent = () if present is None else set(TEXT_ENCODER_KEY_PREFIXES) - present
    record_filter = _AbsentTextEncoderFilter(absent)
    loggers = [logging.getLogger(name) for name in LORA_LOGGERS]
    for logger in loggers:
        logger.addFilter(record_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_MULTIPLE_ADAPTERS, category=UserWarning)
            yield
    finally:
        for logger in loggers:
            logger.removeFilter(record_filter)


def mark_long_prompts_handled(*tokenizers):
    """Tell SD/iL's CLIP tokenizers that a long prompt is expected and already taken care of.

    Transformers prints the warning once per tokenizer and then records that it has, in
    `deprecation_warnings`. Recording it up front is that same mechanism, applied to the tokenizers
    whose long inputs are always chunked — so a tokenizer anywhere else still warns.
    """
    for tokenizer in tokenizers:
        flags = getattr(tokenizer, "deprecation_warnings", None)
        if isinstance(flags, dict):
            flags[LONG_SEQUENCE_WARNING] = True


@contextmanager
def quiet_pipeline_loading():
    """Build a pipeline without Diffusers' component-loading progress bar, then restore it."""
    from diffusers.utils import logging as diffusers_logging

    was_enabled = diffusers_logging.is_progress_bar_enabled()
    diffusers_logging.disable_progress_bar()
    try:
        yield
    finally:
        if was_enabled:
            diffusers_logging.enable_progress_bar()
