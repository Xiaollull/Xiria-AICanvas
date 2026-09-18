"""A prompt has a floor and no ceiling, on every engine.

The request model used to hold `prompt` and `negative_prompt` to 8000 characters. That number came
from nothing in any model — every engine encodes a prompt at whatever length it tokenises to — and
it failed the request outright, so a long prompt did not produce a shorter picture, it produced no
picture at all. These tests exist so no engine can quietly acquire a character limit again.
"""

import unittest

from pydantic import ValidationError

from backend.inference_server import (
    ADetailerInput,
    ADetailerUnitInput,
    GalleryPromptCreateInput,
    GalleryPromptUpdateInput,
    GenerateInput,
)


# Well past any limit the stack used to carry, and past the token count of any real prompt.
LONG_PROMPT = "a lantern in the rain, " * 20_000

ENGINE_REQUESTS = {
    "Anima": dict(
        engine="Anima",
        diffusion_model="anima/diffusion.safetensors",
        text_encoder="qwen/text-encoder.safetensors",
        vae="qwen/vae.safetensors",
    ),
    "Flux": dict(
        engine="Flux",
        diffusion_model="flux1-dev.safetensors",
        text_encoder="clip_l.safetensors",
        text_encoder_2="t5xxl_fp16.safetensors",
        vae="ae.safetensors",
    ),
    "Flux2": dict(
        engine="Flux2",
        diffusion_model="flux2-dev.safetensors",
        text_encoder="flux2-text-encoder.safetensors",
        vae="flux2-vae.safetensors",
    ),
    "Krea2": dict(
        engine="Krea2",
        diffusion_model="krea2_raw_bf16.safetensors",
        text_encoder="qwen3vl_4b_bf16.safetensors",
        vae="qwen_image_vae.safetensors",
    ),
    "SD": dict(engine="SD", checkpoint="model.safetensors"),
    "iL": dict(engine="iL", checkpoint="model.safetensors"),
}


def request_for(engine, **overrides):
    fields = dict(
        prompt="a lantern in the rain",
        width=1024,
        height=1024,
        steps=20,
        cfg=4.0,
        denoise=1.0,
        seed=1,
        sampler="euler",
        scheduler="simple",
        **ENGINE_REQUESTS[engine],
    )
    fields.update(overrides)
    return GenerateInput(**fields)


class PromptLengthContractTests(unittest.TestCase):
    def test_every_engine_accepts_a_prompt_of_any_length(self):
        for engine in ENGINE_REQUESTS:
            with self.subTest(engine=engine):
                request = request_for(engine, prompt=LONG_PROMPT)
                self.assertEqual(request.prompt, LONG_PROMPT)

    def test_a_long_negative_prompt_is_accepted_wherever_the_engine_takes_one(self):
        # Flux and Flux2 are guidance distilled and refuse a negative prompt for that reason, which
        # is a reason about the model rather than about length.
        for engine in ("Anima", "Krea2", "SD", "iL"):
            with self.subTest(engine=engine):
                request = request_for(engine, negative_prompt=LONG_PROMPT)
                self.assertEqual(request.negative_prompt, LONG_PROMPT)

    def test_an_empty_prompt_is_still_refused(self):
        # The floor stays: an empty prompt is a request with nothing to render, not a long one.
        with self.assertRaises(ValidationError):
            request_for("Krea2", prompt="")

    def test_an_adetailer_unit_prompt_has_no_ceiling_either(self):
        stage = ADetailerInput(
            enabled=True,
            units=[
                ADetailerUnitInput(
                    detector="face_yolov8n.pt",
                    prompt=LONG_PROMPT,
                    negative_prompt=LONG_PROMPT,
                )
            ],
        )
        self.assertEqual(stage.units[0].prompt, LONG_PROMPT)
        self.assertEqual(stage.units[0].negative_prompt, LONG_PROMPT)

    def test_the_prompt_library_takes_what_a_generation_takes(self):
        # Saving the prompt that just produced a picture must not be the step that refuses it.
        created = GalleryPromptCreateInput(title="Long", positive_prompt=LONG_PROMPT)
        self.assertEqual(created.positive_prompt, LONG_PROMPT)
        updated = GalleryPromptUpdateInput(negative_prompt=LONG_PROMPT)
        self.assertEqual(updated.negative_prompt, LONG_PROMPT)
        # The title is a label, not a prompt, and keeps its limit.
        with self.assertRaises(ValidationError):
            GalleryPromptCreateInput(title="x" * 161, positive_prompt="a lantern")


if __name__ == "__main__":
    unittest.main()
