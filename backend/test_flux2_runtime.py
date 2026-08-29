"""End-to-end exercise of the Flux2 runtime on the CPU with a miniature model.

Nothing here validates image quality — a randomly initialised 1-block DiT has none. What it does
validate is every seam a 32B checkpoint would otherwise be the only way to reach: the three-layer
conditioning tap and its zero left-pad, the 2x2 autoencoder pack and the batch normalisation
around it, the four-axis position ids, the sigma trajectory, the rectified-flow update, the mask
composite and the decode back to pixels.
"""

import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux2_pipeline import (
    FLUX2_LATENT_CHANNELS,
    FLUX2_MISTRAL_TAP_LAYERS,
    FLUX2_PIXEL_ALIGNMENT,
    FLUX2_QWEN_TAP_LAYERS,
    FLUX2_VAE_LATENT_CHANNELS,
    Flux2Runtime,
    _load_flux2_transformer,
    _runtime_dependencies,
)
from test_flux2_pipeline import JOINT, LLM_HIDDEN, comfy_flux2_checkpoint

# A four-channel-wide autoencoder with the real latent depth, the real 8x stride and the real 2x2
# pack: those are the properties the runtime's packing and normalisation depend on, at a size that
# loads in milliseconds.
TINY_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (4, 4, 4, 4),
    "layers_per_block": 1,
    "latent_channels": FLUX2_VAE_LATENT_CHANNELS,
    "norm_num_groups": 2,
    "sample_size": 64,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
    "batch_norm_eps": 1e-4,
    "patch_size": (2, 2),
}

TEXT_SEQUENCE = 16


class _StubTekkenTokenizer:
    """Enough of the tekken interface for the runtime: a deterministic id per word."""

    def encode(self, text, add_bos=True):
        ids = [(index % 7) + 2 for index in range(len(str(text).split()))]
        return [1, *ids] if add_bos else ids


class _StubQwenTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(index % 7) + 2 for index in range(len(str(text).split()))]}


class _StubLanguageModel(torch.nn.Module):
    """Deterministic stand-in: the hidden-state count and width are what the runtime depends on."""

    def __init__(self, width, depth):
        super().__init__()
        self.width = width
        self.depth = depth

        class _Config:
            hidden_size = width
            use_cache = False

        self.config = _Config()

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, use_cache=False):
        batch, length = input_ids.shape
        values = (input_ids.float().unsqueeze(-1) % 5) / 5.0
        base = values.expand(batch, length, self.width).contiguous()
        states = tuple(base * (index + 1) for index in range(self.depth + 1))
        return type("Output", (), {"hidden_states": states})()


def _tiny_runtime(family="mistral3"):
    deps = _runtime_dependencies()
    transformer, config, _report = _load_flux2_transformer(
        Path("unused.safetensors"), torch.float32, deps, state_dict=comfy_flux2_checkpoint()
    )
    vae = deps["AutoencoderKLFlux2"](**TINY_VAE_CONFIG).eval().requires_grad_(False)
    taps = FLUX2_MISTRAL_TAP_LAYERS if family == "mistral3" else FLUX2_QWEN_TAP_LAYERS
    # Non-trivial packing statistics, so a dropped normalisation cannot pass as a round trip.
    statistics = {
        "bn.running_mean": torch.linspace(-0.5, 0.5, FLUX2_LATENT_CHANNELS),
        "bn.running_var": torch.linspace(0.5, 1.5, FLUX2_LATENT_CHANNELS),
    }
    runtime = Flux2Runtime(
        transformer,
        _StubLanguageModel(LLM_HIDDEN, max(taps)),
        vae,
        _StubTekkenTokenizer() if family == "mistral3" else _StubQwenTokenizer(),
        torch.float32,
        config,
        family,
        statistics,
    )
    runtime.device = torch.device("cpu")
    # The stand-in encoder emits short sequences; keeping the padded context short keeps the
    # fixture fast without changing any of the code paths under test.
    runtime.text_sequence_length = TEXT_SEQUENCE
    return runtime


class Flux2RuntimeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.runtime = _tiny_runtime()

    def tearDown(self):
        self.runtime.close()

    def _generators(self, count, seed=7):
        return [torch.Generator(device="cpu").manual_seed(seed + index) for index in range(count)]

    def test_a_text_to_image_run_produces_one_image_per_generator(self):
        steps = []
        images = self.runtime.generate_batch(
            prompt="a lantern in the rain",
            negative_prompt="",
            width=64,
            height=64,
            steps=3,
            cfg=4.0,
            sampler="euler",
            scheduler="simple",
            generators=self._generators(2),
            on_step=lambda step, total, _latents: steps.append((step, total)),
        )
        self.assertEqual(len(images), 2)
        for image in images:
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.mode, "RGB")
        self.assertEqual(steps, [(1, 3), (2, 3), (3, 3)])
        metrics = self.runtime.last_generation_metrics["sampling"]
        self.assertEqual(metrics["requested_steps"], 3)
        self.assertEqual(metrics["actual_transformer_invocations"], 3)
        self.assertEqual(metrics["branch_invocations_per_update"], 1)

    def test_the_same_seed_reproduces_the_same_image(self):
        first = self.runtime.generate_batch(
            prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=4.0,
            sampler="euler", scheduler="simple", generators=self._generators(1, seed=11),
        )[0]
        second = self.runtime.generate_batch(
            prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=4.0,
            sampler="euler", scheduler="simple", generators=self._generators(1, seed=11),
        )[0]
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_a_subdivided_batch_counts_one_run(self):
        progress = []
        self.runtime.generate_batch(
            prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=4.0,
            sampler="euler", scheduler="simple", generators=self._generators(2, seed=5),
            sampling_batch_size=1,
            on_step=lambda step, total, _latents: progress.append((step, total)),
        )
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual(self.runtime.last_generation_metrics["sampling"]["actual_transformer_invocations"], 4)

    def test_every_sampler_implementation_completes_a_run(self):
        for sampler in ("euler", "euler_ancestral", "heun", "dpm_2", "lms", "lcm"):
            with self.subTest(sampler=sampler):
                images = self.runtime.generate_batch(
                    prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=4.0,
                    sampler=sampler, scheduler="simple", generators=self._generators(1),
                )
                self.assertEqual(len(images), 1)

    def test_a_refinement_pass_runs_every_requested_step_and_returns_the_same_canvas(self):
        source = Image.new("RGB", (64, 96), (40, 90, 160))
        steps = []
        images = self.runtime.refine_batch(
            images=[source],
            prompt="a lantern",
            negative_prompt="",
            steps=4,
            denoise=0.3,
            cfg=4.0,
            sampler="euler",
            scheduler="simple",
            generators=self._generators(1),
            on_step=lambda step, total, _latents: steps.append((step, total)),
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (64, 96))
        self.assertEqual(steps, [(1, 4), (2, 4), (3, 4), (4, 4)])
        diagnostics = self.runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual(diagnostics["schedule_mode"], "comfy_suffix")
        self.assertEqual(diagnostics["requested_steps"], 4)

    def test_a_mask_pins_the_region_it_leaves_out_to_the_source(self):
        source = torch.full((1, FLUX2_LATENT_CHANNELS, 4, 4), 0.25)
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(3))
        mask = torch.zeros(1, 1, 64, 64)
        mask[..., :32] = 1.0
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        embeddings = torch.zeros(1, 4, JOINT)

        final = self.runtime._sample(
            embeddings, source.clone(), sigmas, "euler", 4.0,
            [torch.Generator(device="cpu").manual_seed(1)], None, None,
            source_latents=source, source_noise=noise, latent_mask=mask,
        )

        # The last step lands at sigma 0, so an unmasked cell ends holding exactly the source.
        self.assertTrue(torch.allclose(final[..., 2:], source[..., 2:], atol=1e-5))
        self.assertFalse(torch.allclose(final[..., :2], source[..., :2], atol=1e-3))

    def test_the_conditioning_is_three_taps_wide_and_left_padded_to_the_context(self):
        # The system template alone is longer than the fixture's shortened context, so the pad is
        # only observable with room to spare above it.
        self.runtime.text_sequence_length = 64
        embeddings = self.runtime._encode_prompt("a lantern in the rain")
        self.assertEqual(tuple(embeddings.shape), (1, 64, 3 * LLM_HIDDEN))
        # The pad is at the front and is exactly zero, which is what `extra_conds` produces.
        _ids, _mask, tokens = self.runtime._encode_tokens("a lantern in the rain")
        self.assertLess(tokens, 64)
        self.assertTrue(torch.equal(embeddings[0, : 64 - tokens], torch.zeros(64 - tokens, 3 * LLM_HIDDEN)))
        self.assertFalse(torch.equal(embeddings[0, -1], torch.zeros(3 * LLM_HIDDEN)))

    def test_a_prompt_longer_than_the_context_is_passed_through_rather_than_truncated(self):
        # ComfyUI never truncates a FLUX.2 prompt; the transformer simply sees a longer sequence.
        embeddings = self.runtime._encode_prompt("a lantern in the rain")
        _ids, _mask, tokens = self.runtime._encode_tokens("a lantern in the rain")
        self.assertGreater(tokens, TEXT_SEQUENCE)
        self.assertEqual(tuple(embeddings.shape), (1, tokens, 3 * LLM_HIDDEN))

    def test_the_mistral_prompt_carries_the_system_template_and_a_bos(self):
        ids, mask, tokens = self.runtime._encode_tokens("a lantern")
        self.assertEqual(int(ids[0, 0]), 1)
        self.assertEqual(int(mask.sum()), tokens)
        self.assertEqual(tuple(ids.shape), (1, tokens))

    def test_a_klein_prompt_pads_its_tokens_to_the_context_and_masks_them(self):
        runtime = _tiny_runtime(family="qwen3")
        try:
            runtime.text_sequence_length = TEXT_SEQUENCE
            ids, mask, tokens = runtime._encode_tokens("a lantern")
            self.assertEqual(tuple(ids.shape), (1, TEXT_SEQUENCE))
            self.assertEqual(int(mask.sum()), tokens)
            self.assertEqual(int(ids[0, -1]), 151643)
            self.assertEqual(runtime.tap_layers, FLUX2_QWEN_TAP_LAYERS)
        finally:
            runtime.close()

    def test_token_diagnostics_report_the_language_model(self):
        diagnostics = self.runtime.token_diagnostics("a lantern in the rain")
        self.assertEqual(diagnostics["llm"]["max_length"], TEXT_SEQUENCE)
        self.assertEqual(diagnostics["llm"]["family"], "mistral3")
        self.assertGreater(diagnostics["llm"]["token_count"], 0)
        # FLUX.2 tokenisers disable prompt weighting, so nothing is ever reported as weighted.
        self.assertEqual(diagnostics["llm"]["weighted_token_count"], 0)

    def test_a_canvas_that_cannot_be_packed_into_whole_tokens_is_refused(self):
        with self.assertRaises(ValueError) as error:
            self.runtime.generate_batch(
                prompt="a lantern", negative_prompt="", width=72, height=64, steps=2, cfg=4.0,
                sampler="euler", scheduler="simple", generators=self._generators(1),
            )
        self.assertIn("divisible by 16", str(error.exception))

    def test_a_negative_prompt_and_a_guidance_enhancement_are_both_refused(self):
        with self.assertRaises(ValueError):
            self.runtime.generate_batch(
                prompt="a lantern", negative_prompt="blurry", width=64, height=64, steps=2, cfg=4.0,
                sampler="euler", scheduler="simple", generators=self._generators(1),
            )
        with self.assertRaises(ValueError):
            self.runtime.generate_batch(
                prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=4.0,
                sampler="euler", scheduler="simple", generators=self._generators(1), guidance="pag",
            )

    def test_the_latent_grid_follows_the_vae_stride_and_the_pack(self):
        latents = self.runtime._encode_images([Image.new("RGB", (64, 96), (10, 20, 30))])
        self.assertEqual(
            tuple(latents.shape),
            (1, FLUX2_LATENT_CHANNELS, 96 // FLUX2_PIXEL_ALIGNMENT, 64 // FLUX2_PIXEL_ALIGNMENT),
        )

    def test_the_packing_statistics_are_applied_and_undone(self):
        packed = torch.randn(1, FLUX2_LATENT_CHANNELS, 2, 2)
        normalized = self.runtime._normalize_latents(packed)
        self.assertFalse(torch.allclose(normalized, packed, atol=1e-3))
        self.assertTrue(torch.allclose(self.runtime._denormalize_latents(normalized), packed, atol=1e-5))

    def test_a_closed_runtime_refuses_further_work_and_releases_its_storage(self):
        transformer = self.runtime.transformer
        self.runtime.close()
        self.assertTrue(all(parameter.numel() == 0 for parameter in transformer.parameters()))
        with self.assertRaises(RuntimeError):
            self.runtime.token_diagnostics("a lantern")


class Flux2RuntimeAccountingTests(unittest.TestCase):
    def test_the_weight_sizes_describe_the_group_offload_floor(self):
        runtime = _tiny_runtime()
        try:
            sizes = runtime.weight_sizes
            self.assertEqual(sizes["total"], sum(
                sizes[name] for name in ("transformer", "text_encoder", "vae")
            ))
            self.assertGreater(sizes["transformer_max_block"], 0)
            self.assertGreaterEqual(sizes["transformer_unmatched"], 0)
            self.assertLessEqual(
                sizes["transformer_max_block"] + sizes["transformer_unmatched"], sizes["transformer"]
            )
            self.assertFalse(runtime.batch_cfg)
        finally:
            runtime.close()

    def test_forcing_tiled_decode_is_reported_in_the_metrics(self):
        runtime = _tiny_runtime()
        try:
            self.assertEqual(runtime.configure_vae_tiling(True), "tiled")
            runtime.generate_batch(
                prompt="a lantern", negative_prompt="", width=64, height=64, steps=1, cfg=4.0,
                sampler="euler", scheduler="simple",
                generators=[torch.Generator(device="cpu").manual_seed(3)],
            )
            decode = runtime.last_generation_metrics["vae_decode"]
            self.assertEqual(decode["actual_vae_mode"], "tiled")
            self.assertEqual(decode["requested_tiled_decode"], {"tile": 512, "overlap": 64})
            self.assertTrue(runtime.vae.use_tiling)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
