"""End-to-end exercise of the Flux runtime on the CPU with a miniature model.

Nothing here validates image quality — a randomly initialised 1-block DiT has none. What it does
validate is every seam a 24 GB checkpoint would otherwise be the only way to reach: the packing and
unpacking around the transformer, the call signature, the sigma trajectory, the rectified-flow
update, the mask composite, the latent scaling around the VAE and the decode back to pixels.
"""

import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux_pipeline import (
    FLUX_CLIP_SEQUENCE_LENGTH,
    FLUX_LATENT_CHANNELS,
    FLUX_VAE_SCALE_FACTOR,
    FluxRuntime,
    _load_flux_transformer,
    _runtime_dependencies,
)
from test_flux_pipeline import JOINT, POOLED, comfy_flux_checkpoint

# A four-channel-wide autoencoder with the real latent depth and the real 8x stride: those are the
# two properties the runtime's scaling and packing depend on, at a size that loads in milliseconds.
TINY_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "block_out_channels": (4, 4, 4, 4),
    "layers_per_block": 1,
    "latent_channels": FLUX_LATENT_CHANNELS,
    "norm_num_groups": 2,
    "sample_size": 64,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
}


class _StubTokenizer:
    """Enough of a tokenizer for `tokenize_weighted_prompt`: ids, specials and padding."""

    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 512
    truncation_side = "right"
    padding_side = "right"

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": [(index % 7) + 2 for index in range(len(str(text).split()))]}

    def build_inputs_with_special_tokens(self, ids):
        return [*ids, self.eos_token_id]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [0] * (len(ids) - 1) + [1]


def _tiny_runtime():
    deps = _runtime_dependencies()
    transformer, config, _report = _load_flux_transformer(
        Path("unused.safetensors"), torch.float32, deps, state_dict=comfy_flux_checkpoint()
    )
    vae = deps["AutoencoderKL"](**TINY_VAE_CONFIG).eval().requires_grad_(False)

    class _StubTextEncoder(torch.nn.Module):
        """Deterministic stand-in: shape and dtype are what the runtime actually depends on."""

        def __init__(self, width, pooled=False):
            super().__init__()
            self.width = width
            self.pooled = pooled

        def forward(self, input_ids=None, output_hidden_states=False):
            batch, length = input_ids.shape
            values = (input_ids.float().unsqueeze(-1) % 5) / 5.0
            hidden = values.expand(batch, length, self.width).contiguous()
            if self.pooled:
                return type("Output", (), {"pooler_output": hidden[:, 0, :].contiguous()})()
            return (hidden,)

    runtime = FluxRuntime(
        transformer,
        _StubTextEncoder(POOLED, pooled=True),
        _StubTextEncoder(JOINT),
        vae,
        _StubTokenizer(),
        _StubTokenizer(),
        torch.float32,
        config,
    )
    runtime.device = torch.device("cpu")
    # Both stand-in encoders emit short sequences; keeping the T5 context short keeps the fixture
    # fast without changing any of the code paths under test.
    runtime.t5_sequence_length = 16
    return runtime


class FluxRuntimeExecutionTests(unittest.TestCase):
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
            cfg=3.5,
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
            prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=3.5,
            sampler="euler", scheduler="simple", generators=self._generators(1, seed=11),
        )[0]
        second = self.runtime.generate_batch(
            prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=3.5,
            sampler="euler", scheduler="simple", generators=self._generators(1, seed=11),
        )[0]
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_a_subdivided_batch_counts_one_run_and_matches_the_undivided_one(self):
        whole = self.runtime.generate_batch(
            prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=3.5,
            sampler="euler", scheduler="simple", generators=self._generators(2, seed=5),
        )
        progress = []
        chunked = self.runtime.generate_batch(
            prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=3.5,
            sampler="euler", scheduler="simple", generators=self._generators(2, seed=5),
            sampling_batch_size=1,
            on_step=lambda step, total, _latents: progress.append((step, total)),
        )
        # Subdividing changes the matmul batch, so the pixels are equal to within rounding rather
        # than byte-identical; what must not change is which noise each image started from.
        for original, divided in zip(whole, chunked):
            difference = max(abs(int(a) - int(b)) for a, b in zip(original.tobytes(), divided.tobytes()))
            self.assertLessEqual(difference, 2)
        # One continuous count across both chunks rather than two runs of two.
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual(self.runtime.last_generation_metrics["sampling"]["actual_transformer_invocations"], 4)

    def test_every_sampler_implementation_completes_a_run(self):
        for sampler in ("euler", "euler_ancestral", "heun", "dpm_2", "lms", "lcm"):
            with self.subTest(sampler=sampler):
                images = self.runtime.generate_batch(
                    prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=3.5,
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
            cfg=3.5,
            sampler="euler",
            scheduler="simple",
            generators=self._generators(1),
            on_step=lambda step, total, _latents: steps.append((step, total)),
        )
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (64, 96))
        # The denoise moves the starting sigma; it never shortens the run.
        self.assertEqual(steps, [(1, 4), (2, 4), (3, 4), (4, 4)])
        diagnostics = self.runtime.last_generation_metrics["refinement.sampling"]
        self.assertEqual(diagnostics["schedule_mode"], "comfy_suffix")
        self.assertEqual(diagnostics["requested_steps"], 4)

    def test_a_mask_pins_the_region_it_leaves_out_to_the_source(self):
        # Asserted in latent space, because that is where the composite happens. The decoder is
        # convolutional over an 8x8 latent, so at this fixture size every output pixel sees every
        # latent cell and a pixel-space comparison could not isolate the two halves.
        source = torch.full((1, FLUX_LATENT_CHANNELS, 8, 8), 0.25)
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(3))
        mask = torch.zeros(1, 1, 64, 64)
        mask[..., :32] = 1.0
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        embeddings = torch.zeros(1, 4, JOINT)
        pooled = torch.zeros(1, POOLED)

        final = self.runtime._sample(
            embeddings, pooled, source.clone(), sigmas, "euler", 3.5,
            [torch.Generator(device="cpu").manual_seed(1)], None, None,
            source_latents=source, source_noise=noise, latent_mask=mask,
        )

        # The last step lands at sigma 0, so an unmasked cell ends holding exactly the source.
        self.assertTrue(torch.allclose(final[..., 4:], source[..., 4:], atol=1e-5))
        self.assertFalse(torch.allclose(final[..., :4], source[..., :4], atol=1e-3))

    def test_token_diagnostics_report_both_encoders(self):
        diagnostics = self.runtime.token_diagnostics("a lantern in the rain")
        self.assertEqual(diagnostics["clip"]["max_length"], FLUX_CLIP_SEQUENCE_LENGTH)
        self.assertEqual(diagnostics["t5"]["max_length"], 16)
        self.assertGreater(diagnostics["t5"]["token_count"], 0)

    def test_a_canvas_that_cannot_be_packed_into_whole_tokens_is_refused(self):
        with self.assertRaises(ValueError) as error:
            self.runtime.generate_batch(
                prompt="a lantern", negative_prompt="", width=72, height=64, steps=2, cfg=3.5,
                sampler="euler", scheduler="simple", generators=self._generators(1),
            )
        self.assertIn("divisible by 16", str(error.exception))

    def test_a_negative_prompt_and_a_guidance_enhancement_are_both_refused(self):
        with self.assertRaises(ValueError):
            self.runtime.generate_batch(
                prompt="a lantern", negative_prompt="blurry", width=64, height=64, steps=2, cfg=3.5,
                sampler="euler", scheduler="simple", generators=self._generators(1),
            )
        with self.assertRaises(ValueError):
            self.runtime.generate_batch(
                prompt="a lantern", negative_prompt="", width=64, height=64, steps=2, cfg=3.5,
                sampler="euler", scheduler="simple", generators=self._generators(1), guidance="pag",
            )

    def test_one_generator_per_image_is_required(self):
        with self.assertRaises(ValueError):
            self.runtime.refine_batch(
                images=[Image.new("RGB", (64, 64))], prompt="a", negative_prompt="", steps=2,
                denoise=0.5, cfg=3.5, sampler="euler", scheduler="simple",
                generators=self._generators(2),
            )

    def test_the_latent_grid_follows_the_vae_stride(self):
        latents = self.runtime._encode_images([Image.new("RGB", (64, 96), (10, 20, 30))])
        self.assertEqual(
            tuple(latents.shape),
            (1, FLUX_LATENT_CHANNELS, 96 // FLUX_VAE_SCALE_FACTOR, 64 // FLUX_VAE_SCALE_FACTOR),
        )

    def test_a_closed_runtime_refuses_further_work_and_releases_its_storage(self):
        transformer = self.runtime.transformer
        self.runtime.close()
        self.assertTrue(all(parameter.numel() == 0 for parameter in transformer.parameters()))
        with self.assertRaises(RuntimeError):
            self.runtime.token_diagnostics("a lantern")


class FluxRuntimeAccountingTests(unittest.TestCase):
    def test_the_weight_sizes_describe_the_group_offload_floor(self):
        runtime = _tiny_runtime()
        try:
            sizes = runtime.weight_sizes
            self.assertEqual(sizes["total"], sum(
                sizes[name] for name in ("transformer", "text_encoder", "text_encoder_2", "vae")
            ))
            self.assertGreater(sizes["transformer_max_block"], 0)
            self.assertGreaterEqual(sizes["transformer_unmatched"], 0)
            self.assertLessEqual(
                sizes["transformer_max_block"] + sizes["transformer_unmatched"], sizes["transformer"]
            )
            # A guidance-distilled step never has a second branch to batch.
            self.assertFalse(runtime.batch_cfg)
        finally:
            runtime.close()

    def test_forcing_tiled_decode_is_reported_in_the_metrics(self):
        runtime = _tiny_runtime()
        try:
            self.assertEqual(runtime.configure_vae_tiling(True), "tiled")
            runtime.generate_batch(
                prompt="a lantern", negative_prompt="", width=64, height=64, steps=1, cfg=3.5,
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
