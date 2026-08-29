import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from backend import inference_server
from backend.flux2_pipeline import FLUX2_MAX_EDGE
from backend.flux2_sampling import FLUX2_SAMPLERS, FLUX2_SCHEDULERS
from backend.inference_server import GenerateInput
from backend.memory_policy import estimate_inference_bytes, estimate_largest_component_bytes


def flux2_request(**overrides):
    fields = dict(
        engine="Flux2",
        diffusion_model="flux2-dev.safetensors",
        text_encoder="flux2-text-encoder.safetensors",
        vae="flux2-vae.safetensors",
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


class Flux2RequestContractTests(unittest.TestCase):
    def test_a_flux2_request_mounts_three_components_and_no_checkpoint(self):
        request = flux2_request()
        self.assertEqual(request.engine, "Flux2")
        self.assertIsNone(request.checkpoint)
        self.assertIsNone(request.text_encoder_2)

    def test_every_component_is_required(self):
        for field in ("diffusion_model", "text_encoder", "vae"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                flux2_request(**{field: None})

    def test_a_checkpoint_is_refused(self):
        with self.assertRaises(ValidationError):
            flux2_request(checkpoint="model.safetensors")

    def test_a_second_text_encoder_belongs_only_to_flux1(self):
        # FLUX.2 conditions on one language model; a second encoder here would be a FLUX.1 request
        # wearing the wrong engine name.
        with self.assertRaises(ValidationError) as error:
            flux2_request(text_encoder_2="t5xxl_fp16.safetensors")
        self.assertIn("text_encoder_2 belongs to Flux", str(error.exception))

    def test_guidance_enhancements_are_refused_because_there_is_no_unconditional_branch(self):
        for guidance in ("pag", "cfg_zero_star"):
            with self.subTest(guidance=guidance), self.assertRaises(ValidationError) as error:
                flux2_request(guidance=guidance)
            self.assertIn("guidance distilled", str(error.exception))

    def test_a_negative_prompt_is_refused_rather_than_silently_dropped(self):
        with self.assertRaises(ValidationError) as error:
            flux2_request(negative_prompt="blurry")
        self.assertIn("no unconditional branch", str(error.exception))
        self.assertEqual(flux2_request(negative_prompt="").negative_prompt, "")

    def test_a_per_unit_adetailer_negative_prompt_is_refused_at_submit(self):
        unit = {"detector": "face_yolov8n.pt", "negative_prompt": "blurry"}
        with self.assertRaises(ValidationError) as error:
            flux2_request(adetailer={"enabled": True, "units": [unit]})
        self.assertIn("ADetailer negative prompts must be empty", str(error.exception))
        self.assertEqual(flux2_request(adetailer={"enabled": False, "units": [unit]}).adetailer.active_units, [])
        flux2_request(adetailer={"enabled": True, "units": [{"detector": "face_yolov8n.pt"}]})

    def test_process_previews_are_refused(self):
        with self.assertRaises(ValidationError):
            flux2_request(preview_enabled=True)

    def test_usdu_tiled_hires_remains_anima_only(self):
        with self.assertRaises(ValidationError) as error:
            flux2_request(hires={
                "enabled": True, "model": "RealESRGAN_x4plus_anime_6B.pth", "execution_mode": "usdu_tiled",
                "uniform_tiles": True, "tiled_decode": True,
            })
        self.assertIn("only by Anima", str(error.exception))

    def test_the_sampler_and_scheduler_come_from_the_flux2_vocabulary(self):
        self.assertIn("euler", FLUX2_SAMPLERS)
        self.assertIn("simple", FLUX2_SCHEDULERS)
        with self.assertRaises(ValidationError):
            flux2_request(sampler="not_a_sampler")
        with self.assertRaises(ValidationError):
            flux2_request(scheduler="not_a_scheduler")

    def test_a_hires_chain_cannot_exceed_the_native_refinement_edge(self):
        with self.assertRaises(ValidationError) as error:
            flux2_request(width=2048, height=2048, hires={
                "enabled": True, "model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 4,
            })
        self.assertIn(str(FLUX2_MAX_EDGE), str(error.exception))


class Flux2EngineWiringTests(unittest.TestCase):
    def test_flux2_is_a_native_family_alongside_anima_and_flux(self):
        self.assertEqual(inference_server.NATIVE_ENGINES, ("Anima", "Flux", "Flux2", "Krea2"))
        self.assertTrue(inference_server.is_native_family("flux2"))
        self.assertEqual(inference_server.NATIVE_FAMILY_BY_ENGINE["Flux2"], "flux2")
        # Krea 2 is native but not distilled, so it stays out of this tuple.
        self.assertEqual(inference_server.DISTILLED_GUIDANCE_ENGINES, ("Flux", "Flux2"))

    def test_flux2_shares_the_component_directories_but_keeps_its_own_lora_root(self):
        anima_roots = inference_server.anima_model_roots()
        flux_roots = inference_server.flux_model_roots()
        flux2_roots = inference_server.flux2_model_roots()
        for shared in ("diffusion_model", "text_encoder", "vae"):
            self.assertEqual(anima_roots[shared], flux2_roots[shared])
        # A FLUX.2 LoRA targets a different transformer from either neighbour's.
        self.assertNotEqual(flux2_roots["lora"], flux_roots["lora"])
        self.assertNotEqual(flux2_roots["lora"], anima_roots["lora"])
        self.assertEqual(flux2_roots["lora"].name, "flux2")
        self.assertEqual(inference_server.native_model_roots("Flux2"), flux2_roots)

    def test_a_flux2_run_executes_every_requested_step_whatever_the_denoise(self):
        request = flux2_request(denoise=0.4, source_image={
            "enabled": True,
            "image_data": _one_pixel_png(),
        })
        self.assertEqual(inference_server.base_sampling_steps(request, "flux2"), request.steps)
        self.assertLess(inference_server.base_sampling_steps(request, "sdxl"), request.steps)

    def test_a_flux2_workload_budgets_one_forward_per_step(self):
        diagnostics = inference_server.generation_memory_workload_diagnostics(flux2_request(), "flux2")
        width, height, cfg, batch = diagnostics["admission"]
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(batch, 1)
        self.assertAlmostEqual(cfg, 4.0)

    def test_the_memory_estimate_treats_flux2_as_its_own_family(self):
        # A 6144-wide DiT costs more per megapixel than FLUX.1's 3072-wide one, and its language
        # model encoder makes the transformer a smaller share of the mount.
        self.assertGreater(
            estimate_inference_bytes("flux2", 1024, 1024, 4.0, 1, guidance_copies=1),
            estimate_inference_bytes("flux", 1024, 1024, 3.5, 1, guidance_copies=1),
        )
        self.assertLess(
            estimate_largest_component_bytes(10 * 1024**3, "flux2"),
            estimate_largest_component_bytes(10 * 1024**3, "flux"),
        )

    def test_the_hires_override_vocabulary_is_the_flux2_one(self):
        inference_server.validate_hires_sampling_override("euler", "simple", "flux2")
        with self.assertRaises(ValueError):
            inference_server.validate_hires_sampling_override("mystery", None, "flux2")

    def test_the_health_endpoint_publishes_flux2_beside_the_others(self):
        fields = inference_server.flux2_health_fields()
        self.assertEqual(fields["samplers"], list(FLUX2_SAMPLERS))
        self.assertEqual(fields["schedulers"], list(FLUX2_SCHEDULERS))
        features = fields["features"]
        self.assertFalse(features["pag"])
        self.assertFalse(features["cfg_zero_star"])
        self.assertFalse(features["negative_prompt"])
        self.assertFalse(features["process_preview"])
        # ComfyUI runs both FLUX.2 tokenisers with weighting disabled.
        self.assertFalse(features["prompt_weights"])
        self.assertTrue(features["distilled_guidance"])
        for stage in ("hires", "adetailer", "rtx", "lora"):
            self.assertTrue(features[stage])

    def test_the_mistral_tokenizer_is_declared_as_travelling_with_the_checkpoint(self):
        runtime = inference_server.flux2_health_fields()["runtime"]
        self.assertEqual(runtime["tekken_tokenizer"]["source"], "embedded_in_text_encoder")
        self.assertTrue(runtime["tekken_tokenizer"]["installed"])
        self.assertIn("qwen_tokenizer", runtime)

    def test_the_klein_tokenizer_is_the_pinned_qwen_resource(self):
        sources = inference_server.flux2_tokenizer_sources()
        self.assertEqual(set(sources), {"qwen"})
        self.assertTrue(Path(sources["qwen"]).is_file())

    def test_the_unload_endpoint_resolves_flux2_against_the_diffusion_root(self):
        with patch.object(inference_server, "clear_pipeline") as clear:
            with self.assertRaises(Exception):
                inference_server.unload_model_cache(engine="Flux2", checkpoint="../escape.safetensors")
        clear.assert_not_called()


class PromptWeightSyntaxTests(unittest.TestCase):
    def test_emphasis_is_detected_so_the_run_can_say_it_is_inert(self):
        self.assertTrue(inference_server.prompt_carries_weight_syntax("a (lantern:1.4) in the rain"))
        self.assertTrue(inference_server.prompt_carries_weight_syntax("a (lantern) in the rain"))
        self.assertFalse(inference_server.prompt_carries_weight_syntax("a lantern in the rain"))

    def test_escaped_parentheses_are_literal_text_rather_than_emphasis(self):
        # Reading through the real parser is what gets this right: a note saying weighting is
        # inert would be noise on a prompt that never asked for any.
        self.assertFalse(inference_server.prompt_carries_weight_syntax(r"a \(lantern\) in the rain"))

    def test_a_prompt_the_parser_refuses_never_raises_out_of_the_helper(self):
        # It fails on its own terms elsewhere; this helper only decides whether to add a note.
        self.assertIsInstance(
            inference_server.prompt_carries_weight_syntax("(" * 64 + "x" + ")" * 64), bool
        )


def _one_pixel_png() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
