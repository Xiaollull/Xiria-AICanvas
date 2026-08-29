import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from backend import inference_server
from backend.guidance import CFG_ZERO_STAR_ENGINES
from backend.krea2_pipeline import KREA2_MAX_EDGE, KREA2_PIXEL_ALIGNMENT
from backend.krea2_sampling import KREA2_SAMPLERS, KREA2_SCHEDULERS, KREA2_SHIFT
from backend.inference_server import GenerateInput
from backend.memory_policy import estimate_inference_bytes, estimate_largest_component_bytes


def krea2_request(**overrides):
    fields = dict(
        engine="Krea2",
        diffusion_model="krea2_raw_bf16.safetensors",
        text_encoder="qwen3vl_4b_bf16.safetensors",
        vae="qwen_image_vae.safetensors",
        prompt="a lantern in the rain",
        width=1024,
        height=1024,
        steps=20,
        cfg=4.0,
        denoise=1.0,
        seed=1,
        sampler="euler",
        scheduler="simple",
        preview_enabled=False,
    )
    fields.update(overrides)
    return GenerateInput(**fields)


class Krea2RequestContractTests(unittest.TestCase):
    def test_a_krea2_request_mounts_three_components_and_no_checkpoint(self):
        request = krea2_request()
        self.assertEqual(request.engine, "Krea2")
        self.assertIsNone(request.checkpoint)
        self.assertIsNone(request.text_encoder_2)

    def test_every_component_is_required(self):
        for field in ("diffusion_model", "text_encoder", "vae"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                krea2_request(**{field: None})

    def test_a_checkpoint_is_refused(self):
        with self.assertRaises(ValidationError) as error:
            krea2_request(checkpoint="model.safetensors")
        self.assertIn("Krea2 forbids checkpoint", str(error.exception))

    def test_a_second_text_encoder_belongs_only_to_flux1(self):
        with self.assertRaises(ValidationError) as error:
            krea2_request(text_encoder_2="t5xxl_fp16.safetensors")
        self.assertIn("text_encoder_2 belongs to Flux", str(error.exception))

    def test_a_negative_prompt_is_accepted_because_krea2_runs_an_unconditional_branch(self):
        # This is the one native engine that is not guidance distilled; refusing the negative
        # prompt here would be copying a Flux rule that does not apply.
        self.assertEqual(krea2_request(negative_prompt="blurry").negative_prompt, "blurry")

    def test_a_per_unit_adetailer_negative_prompt_is_accepted_too(self):
        unit = {"detector": "face_yolov8n.pt", "negative_prompt": "blurry"}
        request = krea2_request(adetailer={"enabled": True, "units": [unit]})
        self.assertEqual(request.adetailer.active_units[0].negative_prompt, "blurry")

    def test_cfg_zero_star_is_accepted_and_pag_is_refused_by_name(self):
        self.assertEqual(krea2_request(guidance="cfg_zero_star").guidance, "cfg_zero_star")
        with self.assertRaises(ValidationError) as error:
            krea2_request(guidance="pag")
        self.assertIn("Krea2 does not support PAG", str(error.exception))

    def test_process_previews_are_refused(self):
        with self.assertRaises(ValidationError):
            krea2_request(preview_enabled=True)

    def test_usdu_tiled_hires_remains_anima_only(self):
        with self.assertRaises(ValidationError) as error:
            krea2_request(hires={
                "enabled": True, "model": "RealESRGAN_x4plus_anime_6B.pth", "execution_mode": "usdu_tiled",
                "uniform_tiles": True, "tiled_decode": True,
            })
        self.assertIn("only by Anima", str(error.exception))

    def test_the_sampler_and_scheduler_come_from_the_krea2_vocabulary(self):
        self.assertIn("euler", KREA2_SAMPLERS)
        self.assertIn("simple", KREA2_SCHEDULERS)
        with self.assertRaises(ValidationError) as error:
            krea2_request(sampler="not_a_sampler")
        self.assertIn("Unsupported Krea2 sampler", str(error.exception))
        with self.assertRaises(ValidationError) as error:
            krea2_request(scheduler="not_a_scheduler")
        self.assertIn("Unsupported Krea2 scheduler", str(error.exception))

    def test_a_hires_chain_cannot_exceed_the_native_refinement_edge(self):
        with self.assertRaises(ValidationError) as error:
            krea2_request(width=2048, height=2048, hires={
                "enabled": True, "model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 4,
            })
        self.assertIn(str(KREA2_MAX_EDGE), str(error.exception))


class Krea2EngineWiringTests(unittest.TestCase):
    def test_krea2_is_a_native_family_that_is_not_guidance_distilled(self):
        self.assertIn("Krea2", inference_server.NATIVE_ENGINES)
        self.assertTrue(inference_server.is_native_family("krea2"))
        self.assertEqual(inference_server.NATIVE_FAMILY_BY_ENGINE["Krea2"], "krea2")
        self.assertNotIn("Krea2", inference_server.DISTILLED_GUIDANCE_ENGINES)

    def test_krea2_shares_the_component_directories_but_keeps_its_own_lora_root(self):
        anima_roots = inference_server.anima_model_roots()
        flux2_roots = inference_server.flux2_model_roots()
        krea2_roots = inference_server.krea2_model_roots()
        for shared in ("diffusion_model", "text_encoder", "vae"):
            self.assertEqual(anima_roots[shared], krea2_roots[shared])
        # A Krea 2 LoRA patches a single-stream DiT whose modules are named nothing like either
        # Flux transformer's, so it can only ever apply here.
        self.assertNotEqual(krea2_roots["lora"], flux2_roots["lora"])
        self.assertNotEqual(krea2_roots["lora"], anima_roots["lora"])
        self.assertEqual(krea2_roots["lora"].name, "krea2")
        self.assertEqual(inference_server.native_model_roots("Krea2"), krea2_roots)

    def test_a_krea2_run_executes_every_requested_step_whatever_the_denoise(self):
        request = krea2_request(denoise=0.4, source_image={
            "enabled": True,
            "image_data": _one_pixel_png(),
        })
        self.assertEqual(inference_server.base_sampling_steps(request, "krea2"), request.steps)
        self.assertLess(inference_server.base_sampling_steps(request, "sdxl"), request.steps)

    def test_the_memory_estimate_treats_krea2_as_its_own_family(self):
        # A 6144-wide single-stream DiT costs more per megapixel than FLUX.1's 3072-wide one, and
        # the twelve-tap text refinement adds a little on top of FLUX.2's per-branch cost.
        self.assertGreater(
            estimate_inference_bytes("krea2", 1024, 1024, 4.0, 1, guidance_copies=1),
            estimate_inference_bytes("flux", 1024, 1024, 3.5, 1, guidance_copies=1),
        )
        self.assertGreater(
            estimate_inference_bytes("krea2", 1024, 1024, 4.0, 1, guidance_copies=1),
            estimate_inference_bytes("flux2", 1024, 1024, 4.0, 1, guidance_copies=1),
        )
        # The published mount is a 26 GB transformer beside an 8.9 GB encoder, so the transformer
        # is a larger share than FLUX.2's and a smaller one than Anima's.
        self.assertGreater(
            estimate_largest_component_bytes(10 * 1024**3, "krea2"),
            estimate_largest_component_bytes(10 * 1024**3, "flux2"),
        )
        self.assertLess(
            estimate_largest_component_bytes(10 * 1024**3, "krea2"),
            estimate_largest_component_bytes(10 * 1024**3, "anima"),
        )

    def test_the_admission_budget_is_one_branch_even_with_two_guidance_copies(self):
        # The conditioned and unconditional forwards run one after the other, so budgeting two
        # would be the same over-count Anima's sequential path deliberately avoids.
        one = estimate_inference_bytes("krea2", 1024, 1024, 4.0, 1, guidance_copies=1)
        two = estimate_inference_bytes("krea2", 1024, 1024, 4.0, 1, guidance_copies=2)
        self.assertLess(one, two)
        diagnostics = inference_server.generation_memory_workload_diagnostics(krea2_request(), "krea2")
        width, height, cfg, batch = diagnostics["admission"]
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(batch, 1)
        self.assertAlmostEqual(cfg, 4.0)

    def test_the_hires_override_vocabulary_is_the_krea2_one(self):
        inference_server.validate_hires_sampling_override("euler", "simple", "krea2")
        with self.assertRaises(ValueError):
            inference_server.validate_hires_sampling_override("mystery", None, "krea2")

    def test_the_health_endpoint_publishes_krea2_beside_the_others(self):
        fields = inference_server.krea2_health_fields()
        self.assertEqual(fields["samplers"], list(KREA2_SAMPLERS))
        self.assertEqual(fields["schedulers"], list(KREA2_SCHEDULERS))
        features = fields["features"]
        self.assertTrue(features["cfg_zero_star"])
        self.assertTrue(features["negative_prompt"])
        self.assertFalse(features["distilled_guidance"])
        # PAG needs an identity-self-attention override of the native blocks, which is not installed.
        self.assertFalse(features["pag"])
        self.assertFalse(features["process_preview"])
        # `Qwen3VLTokenizer` passes `disable_weights=True`.
        self.assertFalse(features["prompt_weights"])
        for stage in ("hires", "adetailer", "rtx", "lora"):
            self.assertTrue(features[stage])

    def test_cfg_zero_star_is_published_as_available_rather_than_planned(self):
        with patch.object(inference_server, "health_scan_fields", return_value={
            "detector_models": [],
            "anima": inference_server.anima_health_fields(),
            "flux": inference_server.flux_health_fields(),
            "flux2": inference_server.flux2_health_fields(),
            "krea2": inference_server.krea2_health_fields(),
            "upscalers": {"models": []},
        }):
            health = inference_server.health()
        self.assertIn("Krea2", health["engines"])
        self.assertEqual(health["guidance"]["cfg_zero_star"]["engines"], sorted(CFG_ZERO_STAR_ENGINES))
        self.assertEqual(health["guidance"]["cfg_zero_star"]["planned_engines"], [])
        # PAG stays a three-engine feature.
        self.assertNotIn("Krea2", health["guidance"]["pag"]["engines"])

    def test_the_tokenizer_is_the_pinned_qwen_resource_the_other_engines_already_use(self):
        sources = inference_server.krea2_tokenizer_sources()
        self.assertEqual(set(sources), {"qwen"})
        self.assertTrue(Path(sources["qwen"]).is_file())
        self.assertEqual(sources, inference_server.flux2_tokenizer_sources())

    def test_the_unload_endpoint_resolves_krea2_against_the_diffusion_root(self):
        with patch.object(inference_server, "clear_pipeline") as clear:
            with self.assertRaises(Exception):
                inference_server.unload_model_cache(engine="Krea2", checkpoint="../escape.safetensors")
        clear.assert_not_called()

    def test_the_canvas_alignment_matches_the_latent_geometry(self):
        # Stride 8 through the autoencoder, then a 2x2 patch in the transformer.
        self.assertEqual(KREA2_PIXEL_ALIGNMENT, 16)
        krea2_request(width=1024, height=1536)
        with self.assertRaises(ValidationError):
            krea2_request(width=1000, height=1024)

    def test_the_recorded_shift_is_the_static_one(self):
        self.assertEqual(KREA2_SHIFT, 1.15)


def _one_pixel_png() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
