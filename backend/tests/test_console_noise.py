"""The library messages an SD/iL job used to print as errors, and that each is quiet only where false.

Every test here pairs "silenced where it is false" with "still printed where it could be true", because
the failure worth guarding against is not a noisy console — it is a real warning that stopped showing.
"""

import logging
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from backend import inference_server
from backend.console_noise import (
    lora_text_encoders,
    mark_long_prompts_handled,
    quiet_lora_loading,
    quiet_pipeline_loading,
)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def missing_text_encoder_message(model_class, prefix):
    # Formatted exactly as `diffusers/loaders/lora_base.py` formats it.
    return (
        f"No LoRA keys associated to {model_class} found with the {prefix=}. "
        "This is safe to ignore if LoRA state dict didn't originally have any "
        f"{model_class} related params."
    )


class LoraFixture:
    def __init__(self, test_case):
        self.directory = tempfile.TemporaryDirectory()
        test_case.addCleanup(self.directory.cleanup)

    def write(self, name, keys):
        path = Path(self.directory.name) / name
        save_file({key: torch.zeros(1) for key in keys}, str(path))
        return path


UNET_ONLY = ["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"]
FIRST_ENCODER = UNET_ONLY + ["lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha"]
BOTH_ENCODERS = FIRST_ENCODER + ["lora_te2_text_model_encoder_layers_0_mlp_fc1.alpha"]


class LoraHeaderTests(unittest.TestCase):
    def test_the_header_says_which_text_encoders_a_lora_carries(self):
        # The same split the reference stack showed: three style LoRAs with no text-encoder weights
        # (and the warning pair each), three with 792 text-encoder keys (and none).
        files = LoraFixture(self)
        self.assertEqual(lora_text_encoders(files.write("unet.safetensors", UNET_ONLY)), set())
        self.assertEqual(lora_text_encoders(files.write("te1.safetensors", FIRST_ENCODER)), {"text_encoder"})
        self.assertEqual(
            lora_text_encoders(files.write("both.safetensors", BOTH_ENCODERS)),
            {"text_encoder", "text_encoder_2"},
        )

    def test_diffusers_and_sd1_key_names_count_too(self):
        files = LoraFixture(self)
        diffusers_style = files.write("d.safetensors", ["text_encoder_2.text_model.lora_A.weight"])
        sd1_style = files.write("s.safetensors", ["lora_te_text_model_encoder_layers_0_mlp_fc1.alpha"])
        self.assertEqual(lora_text_encoders(diffusers_style), {"text_encoder_2"})
        self.assertEqual(lora_text_encoders(sd1_style), {"text_encoder"})

    def test_a_file_whose_header_cannot_be_read_is_unknown_not_empty(self):
        # Unknown must never read as "no text-encoder weights", or a pickled LoRA whose encoder keys
        # failed to map would lose the one warning that says so.
        with tempfile.TemporaryDirectory() as directory:
            pickled = Path(directory) / "old.pt"
            pickled.write_bytes(b"not a safetensors file")
            self.assertIsNone(lora_text_encoders(pickled))


class QuietLoraLoadingTests(unittest.TestCase):
    def setUp(self):
        self.capture = Capture()
        self.logger = logging.getLogger("diffusers.loaders.lora_base")
        self.logger.addHandler(self.capture)
        self.addCleanup(self.logger.removeHandler, self.capture)

    def emit_both(self):
        self.logger.warning(missing_text_encoder_message("CLIPTextModel", "text_encoder"))
        self.logger.warning(missing_text_encoder_message("CLIPTextModelWithProjection", "text_encoder_2"))

    def test_a_unet_only_lora_loads_without_the_text_encoder_warnings(self):
        path = LoraFixture(self).write("unet.safetensors", UNET_ONLY)
        with quiet_lora_loading(path):
            self.emit_both()
        self.assertEqual(self.capture.messages, [])

    def test_a_lora_that_carries_an_encoder_still_warns_about_that_encoder(self):
        # This is the case the message exists for: weights that are in the file but did not map.
        path = LoraFixture(self).write("te1.safetensors", FIRST_ENCODER)
        with quiet_lora_loading(path):
            self.emit_both()
        self.assertEqual(len(self.capture.messages), 1)
        self.assertIn("prefix='text_encoder'", self.capture.messages[0])

    def test_an_unreadable_lora_keeps_every_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            pickled = Path(directory) / "old.pt"
            pickled.write_bytes(b"not a safetensors file")
            with quiet_lora_loading(pickled):
                self.emit_both()
        self.assertEqual(len(self.capture.messages), 2)

    def test_the_filter_lasts_exactly_as_long_as_the_load(self):
        path = LoraFixture(self).write("unet.safetensors", UNET_ONLY)
        with quiet_lora_loading(path):
            pass
        self.emit_both()
        self.assertEqual(len(self.capture.messages), 2)

    def test_unrelated_messages_pass_through_the_load(self):
        path = LoraFixture(self).write("unet.safetensors", UNET_ONLY)
        with quiet_lora_loading(path):
            self.logger.warning("Loading adapter weights from state_dict led to unexpected keys")
        self.assertEqual(len(self.capture.messages), 1)

    def test_stacking_adapters_is_not_reported_as_a_mistake_during_the_load_only(self):
        path = LoraFixture(self).write("unet.safetensors", UNET_ONLY)
        message = (
            "Already found a `peft_config` attribute in the model. This will lead to having multiple "
            "adapters in the model. Make sure to know what you are doing!"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with quiet_lora_loading(path):
                warnings.warn(message, UserWarning)
            self.assertEqual(caught, [])
            warnings.warn(message, UserWarning)
            self.assertEqual(len(caught), 1)


class LongPromptTests(unittest.TestCase):
    """Against transformers' own warning path, so a renamed flag fails here rather than in the console."""

    def warns(self, tokenizer):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        capture = Capture()
        logger = logging.getLogger("transformers.tokenization_utils_base")
        logger.addHandler(capture)
        try:
            PreTrainedTokenizerBase._eventual_warn_about_too_long_sequence(tokenizer, list(range(147)), None, True)
        finally:
            logger.removeHandler(capture)
        return [message for message in capture.messages if "Token indices sequence length" in message]

    def tokenizer(self):
        return SimpleNamespace(model_max_length=77, deprecation_warnings={})

    def test_a_chunked_tokenizer_no_longer_predicts_indexing_errors(self):
        tokenizer = self.tokenizer()
        mark_long_prompts_handled(tokenizer)
        self.assertEqual(self.warns(tokenizer), [])

    def test_a_tokenizer_nobody_marked_still_warns(self):
        # Only the SD/iL tokenizers are marked; another engine's tokenizer truncating nothing and
        # feeding 147 tokens to a 77-token model would be a real bug, and still says so.
        self.assertEqual(len(self.warns(self.tokenizer())), 1)

    def test_missing_tokenizers_are_skipped(self):
        mark_long_prompts_handled(None, SimpleNamespace())


class PipelineLoadingTests(unittest.TestCase):
    def test_the_loading_bar_is_off_for_the_load_and_restored_after(self):
        from diffusers.utils import logging as diffusers_logging

        self.assertTrue(diffusers_logging.is_progress_bar_enabled())
        with quiet_pipeline_loading():
            self.assertFalse(diffusers_logging.is_progress_bar_enabled())
        self.assertTrue(diffusers_logging.is_progress_bar_enabled())

    def test_a_failed_load_still_restores_the_bar(self):
        from diffusers.utils import logging as diffusers_logging

        with self.assertRaises(RuntimeError):
            with quiet_pipeline_loading():
                raise RuntimeError("checkpoint missing")
        self.assertTrue(diffusers_logging.is_progress_bar_enabled())


class VaeModeTests(unittest.TestCase):
    def test_the_vae_is_configured_on_the_autoencoder_not_through_deprecated_wrappers(self):
        calls = []

        def deprecated(*_args, **_kwargs):
            raise AssertionError("pipeline-level VAE wrappers are deprecated in Diffusers 0.38")

        vae = SimpleNamespace(**{
            name: (lambda name=name: calls.append(name))
            for name in ("disable_slicing", "disable_tiling", "enable_slicing", "enable_tiling")
        })
        pipeline = SimpleNamespace(
            vae=vae,
            disable_vae_slicing=deprecated, disable_vae_tiling=deprecated,
            enable_vae_slicing=deprecated, enable_vae_tiling=deprecated,
        )
        with patch.dict(inference_server.performance_settings, {"vae_mode": "auto"}), \
                patch.object(inference_server, "active_vae_mode", "none"):
            inference_server.configure_vae_mode(pipeline, {"mode": "low_vram"})
            self.assertEqual(inference_server.active_vae_mode, "tiled")
        self.assertEqual(calls, ["disable_slicing", "disable_tiling", "enable_slicing", "enable_tiling"])


if __name__ == "__main__":
    unittest.main()
