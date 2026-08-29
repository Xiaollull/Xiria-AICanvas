import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from backend import inference_server
from backend.flux_pipeline import FLUX_MAX_EDGE
from backend.flux_sampling import FLUX_SAMPLERS, FLUX_SCHEDULERS
from backend.inference_server import GenerateInput
from backend.memory_policy import estimate_inference_bytes, estimate_largest_component_bytes


def flux_request(**overrides):
    fields = dict(
        engine="Flux",
        diffusion_model="flux1-dev.safetensors",
        text_encoder="clip_l.safetensors",
        text_encoder_2="t5xxl_fp16.safetensors",
        vae="ae.safetensors",
        prompt="a lantern in the rain",
        width=1024,
        height=1024,
        steps=20,
        cfg=3.5,
        denoise=1.0,
        seed=1,
        sampler="euler",
        scheduler="simple",
        preview_enabled=False,
    )
    fields.update(overrides)
    return GenerateInput(**fields)


class FluxRequestContractTests(unittest.TestCase):
    def test_a_flux_request_mounts_four_components_and_no_checkpoint(self):
        request = flux_request()
        self.assertEqual(request.engine, "Flux")
        self.assertIsNone(request.checkpoint)
        self.assertEqual(request.text_encoder_2, "t5xxl_fp16.safetensors")

    def test_every_component_is_required(self):
        for field in ("diffusion_model", "text_encoder", "text_encoder_2", "vae"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                flux_request(**{field: None})

    def test_a_checkpoint_is_refused(self):
        with self.assertRaises(ValidationError):
            flux_request(checkpoint="model.safetensors")

    def test_the_two_text_encoders_must_be_different_files(self):
        with self.assertRaises(ValidationError) as error:
            flux_request(text_encoder_2="clip_l.safetensors")
        self.assertIn("two different text encoders", str(error.exception))

    def test_a_second_text_encoder_belongs_only_to_flux(self):
        for engine, extra in (("SD", {"checkpoint": "model.safetensors"}),
                              ("Anima", {"diffusion_model": "a.safetensors", "text_encoder": "b.safetensors", "vae": "c.safetensors"})):
            with self.subTest(engine=engine), self.assertRaises(ValidationError):
                GenerateInput(
                    engine=engine, prompt="x", width=512, height=512, steps=10, cfg=7, denoise=1,
                    seed=1, sampler="euler", scheduler="simple", preview_enabled=False,
                    text_encoder_2="t5xxl.safetensors", **extra,
                )

    def test_guidance_enhancements_are_refused_because_there_is_no_unconditional_branch(self):
        for guidance in ("pag", "cfg_zero_star"):
            with self.subTest(guidance=guidance), self.assertRaises(ValidationError) as error:
                flux_request(guidance=guidance)
            self.assertIn("guidance distilled", str(error.exception))

    def test_a_negative_prompt_is_refused_rather_than_silently_dropped(self):
        with self.assertRaises(ValidationError) as error:
            flux_request(negative_prompt="blurry")
        self.assertIn("no unconditional branch", str(error.exception))
        # An empty one is the normal case and must stay legal.
        self.assertEqual(flux_request(negative_prompt="").negative_prompt, "")

    def test_a_per_unit_adetailer_negative_prompt_is_refused_at_submit(self):
        # A detail pass encodes its own negative prompt through the same runtime, so this is the
        # same contradiction as the main one — and it has to fail at submit rather than several
        # stages into a job that has already sampled.
        unit = {"detector": "face_yolov8n.pt", "negative_prompt": "blurry"}
        with self.assertRaises(ValidationError) as error:
            flux_request(adetailer={"enabled": True, "units": [unit]})
        self.assertIn("ADetailer negative prompts must be empty", str(error.exception))
        # The wire carries only the units that will run, so a disabled stage carries no run plan and
        # its stored text is not a contradiction.
        self.assertEqual(flux_request(adetailer={"enabled": False, "units": [unit]}).adetailer.active_units, [])
        # An enabled stage whose units carry no negative prompt is the ordinary case.
        flux_request(adetailer={"enabled": True, "units": [{"detector": "face_yolov8n.pt"}]})

    def test_process_previews_are_refused(self):
        with self.assertRaises(ValidationError):
            flux_request(preview_enabled=True)

    def test_usdu_tiled_hires_remains_anima_only(self):
        with self.assertRaises(ValidationError) as error:
            flux_request(hires={
                "enabled": True, "model": "RealESRGAN_x4plus_anime_6B.pth", "execution_mode": "usdu_tiled",
                "uniform_tiles": True, "tiled_decode": True,
            })
        self.assertIn("only by Anima", str(error.exception))

    def test_the_sampler_and_scheduler_come_from_the_flux_vocabulary(self):
        self.assertIn("euler", FLUX_SAMPLERS)
        self.assertIn("simple", FLUX_SCHEDULERS)
        with self.assertRaises(ValidationError):
            flux_request(sampler="not_a_sampler")
        with self.assertRaises(ValidationError):
            flux_request(scheduler="not_a_scheduler")

    def test_a_hires_chain_cannot_exceed_the_native_refinement_edge(self):
        with self.assertRaises(ValidationError) as error:
            flux_request(width=2048, height=2048, hires={
                "enabled": True, "model": "RealESRGAN_x4plus_anime_6B.pth", "scale": 4,
            })
        self.assertIn(str(FLUX_MAX_EDGE), str(error.exception))


class FluxEngineWiringTests(unittest.TestCase):
    def test_flux_is_a_native_family_alongside_anima(self):
        self.assertEqual(inference_server.NATIVE_ENGINES, ("Anima", "Flux", "Flux2", "Krea2"))
        self.assertTrue(inference_server.is_native_family("flux"))
        self.assertTrue(inference_server.is_native_family("anima"))
        self.assertFalse(inference_server.is_native_family("sdxl"))
        self.assertEqual(inference_server.NATIVE_FAMILY_BY_ENGINE["Flux"], "flux")

    def test_flux_shares_the_component_directories_but_keeps_its_own_lora_root(self):
        anima_roots = inference_server.anima_model_roots()
        flux_roots = inference_server.flux_model_roots()
        for shared in ("diffusion_model", "text_encoder", "vae"):
            self.assertEqual(anima_roots[shared], flux_roots[shared])
        self.assertNotEqual(anima_roots["lora"], flux_roots["lora"])
        self.assertEqual(flux_roots["lora"].name, "flux")
        self.assertEqual(inference_server.native_model_roots("Flux"), flux_roots)
        self.assertEqual(inference_server.native_model_roots("Anima"), anima_roots)

    def test_a_flux_run_executes_every_requested_step_whatever_the_denoise(self):
        # The same rule the ADetailer and Hires accounting follow: a native refinement pass moves
        # its starting sigma, it does not shorten the run.
        request = flux_request(denoise=0.4, source_image={
            "enabled": True,
            "image_data": _one_pixel_png(),
        })
        self.assertEqual(inference_server.base_sampling_steps(request, "flux"), request.steps)
        self.assertLess(inference_server.base_sampling_steps(request, "sdxl"), request.steps)

    def test_a_flux_workload_budgets_one_forward_per_step(self):
        diagnostics = inference_server.generation_memory_workload_diagnostics(flux_request(), "flux")
        width, height, cfg, batch = diagnostics["admission"]
        self.assertEqual((width, height), (1024, 1024))
        self.assertEqual(batch, 1)
        self.assertAlmostEqual(cfg, 3.5)

    def test_the_memory_estimate_treats_flux_as_its_own_family(self):
        flux_bytes = estimate_inference_bytes("flux", 1024, 1024, 3.5, 1, guidance_copies=1)
        anima_bytes = estimate_inference_bytes("anima", 1024, 1024, 3.5, 1, guidance_copies=1)
        self.assertGreater(flux_bytes, anima_bytes)
        self.assertLess(
            estimate_largest_component_bytes(10 * 1024**3, "flux"),
            estimate_largest_component_bytes(10 * 1024**3, "anima"),
        )

    def test_the_health_endpoint_publishes_flux_beside_anima(self):
        fields = inference_server.flux_health_fields()
        self.assertEqual(fields["samplers"], list(FLUX_SAMPLERS))
        self.assertEqual(fields["schedulers"], list(FLUX_SCHEDULERS))
        features = fields["features"]
        # Guidance distillation is declared as an absence, not left undefined: the UI fails closed
        # on a native engine, so an undeclared capability would silently disable the stage.
        self.assertFalse(features["pag"])
        self.assertFalse(features["cfg_zero_star"])
        self.assertFalse(features["negative_prompt"])
        self.assertFalse(features["process_preview"])
        self.assertTrue(features["distilled_guidance"])
        for stage in ("hires", "adetailer", "rtx", "lora"):
            self.assertTrue(features[stage])

    def test_the_clip_tokenizer_is_taken_from_an_installed_runtime_config(self):
        with patch.object(inference_server, "find_pipeline_config", return_value=None):
            self.assertIsNone(inference_server.flux_clip_tokenizer_directory())
        with self.assertRaises(ValueError):
            with patch.object(inference_server, "flux_clip_tokenizer_directory", return_value=None):
                inference_server.flux_tokenizer_sources()

    def test_the_unload_endpoint_resolves_flux_against_the_diffusion_root(self):
        with patch.object(inference_server, "clear_pipeline") as clear:
            with self.assertRaises(Exception):
                # A path outside the root must be refused before anything is released.
                inference_server.unload_model_cache(engine="Flux", checkpoint="../escape.safetensors")
        clear.assert_not_called()


def _one_pixel_png() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
