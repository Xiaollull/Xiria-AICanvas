import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
from pydantic import ValidationError

from backend import inference_server, upscaler
from backend.inference_server import GenerateInput, HiresInput


def generation_request(**updates):
    values = {
        "engine": "SD",
        "checkpoint": "model.safetensors",
        "prompt": "test",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 7,
        "denoise": 1,
        "seed": 1,
        "sampler": "euler",
        "scheduler": "simple",
    }
    values.update(updates)
    return GenerateInput(**values)


class FakeArchitecture:
    name = "TestSR"


class FakeDescriptor:
    scale = 2
    input_channels = 3
    output_channels = 3
    purpose = "SR"
    supports_half = False
    architecture = FakeArchitecture()

    def __init__(self):
        self.model = torch.nn.Identity()

    def __call__(self, tensor):
        return torch.nn.functional.interpolate(tensor, scale_factor=2, mode="nearest")


class UpscalerTests(unittest.TestCase):
    def test_hires_seed_schema_is_strict_lossless_and_legacy_defaults_to_inherit(self):
        legacy = HiresInput()
        self.assertEqual((legacy.mode, legacy.seed), ("inherit", None))
        comfy = HiresInput(mode="fixed", seed="885289963651097")
        maximum = HiresInput(mode="fixed", seed="18446744073709551615")
        self.assertEqual(comfy.seed, "885289963651097")
        self.assertEqual(maximum.seed, "18446744073709551615")
        self.assertEqual(HiresInput(mode="fixed", seed="0007").seed, "7")

        invalid = (
            {"mode": "fixed"},
            {"mode": "inherit", "seed": "1"},
            {"mode": "random", "seed": "1"},
            {"mode": "fixed", "seed": True},
            {"mode": "fixed", "seed": 1},
            {"mode": "fixed", "seed": "-1"},
            {"mode": "fixed", "seed": "1.0"},
            {"mode": "fixed", "seed": "18446744073709551616"},
            {"mode": "illegal"},
            {"mode": True},
            {"seed_mode": "fixed"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                HiresInput(**values)

    def test_effective_hires_seed_resolution_is_per_output_secure_and_disabled_is_null(self):
        self.assertIsNone(inference_server.resolve_effective_hires_seed(HiresInput(enabled=False), 9))
        self.assertEqual(
            inference_server.resolve_effective_hires_seed(HiresInput(enabled=True, model="model.pth"), 9),
            9,
        )
        fixed = HiresInput(enabled=True, model="model.pth", mode="fixed", seed="885289963651097")
        self.assertEqual(inference_server.resolve_effective_hires_seed(fixed, 9), 885289963651097)
        calls = []
        values = iter((17, 18))
        random_settings = HiresInput(enabled=True, model="model.pth", mode="random")

        def random_uint64():
            value = next(values)
            calls.append(value)
            return value

        first = inference_server.resolve_effective_hires_seed(random_settings, 100, random_uint64)
        second = inference_server.resolve_effective_hires_seed(random_settings, 101, random_uint64)
        self.assertEqual((first, second, calls), (17, 18, [17, 18]))
        for invalid in (True, -1, 0x10000000000000000, "17"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                inference_server.resolve_effective_hires_seed(random_settings, 1, lambda value=invalid: value)

    def test_hires_schema_accepts_frontend_contract_and_rejects_invalid_tiled_modes(self):
        full = HiresInput(model="model.pth", execution_mode="full_frame")
        self.assertEqual(full.execution_mode, "full_frame")
        tiled = HiresInput(
            enabled=True, model="model.pth", execution_mode="usdu_tiled", sampler="euler_ancestral",
            scheduler="normal", tile_width="auto", tile_height="auto", padding=32, mask_blur=8,
            seam_mode="none", uniform_tiles=True, tiled_decode=True,
        )
        self.assertEqual(tiled.tile_width, "auto")
        with self.assertRaises(ValidationError):
            HiresInput(enabled=True, model="model.pth", execution_mode="usdu_tiled", uniform_tiles=False)
        for tile_dimension in (1, 0, -1, True):
            with self.subTest(tile_dimension=tile_dimension), self.assertRaises(ValidationError):
                HiresInput(model="model.pth", tile_width=tile_dimension)
        with self.assertRaises(ValidationError):
            HiresInput(model="model.pth", unexpected=True)
        with self.assertRaisesRegex(ValidationError, "only by Anima"):
            generation_request(hires={"enabled": True, "model": "model.pth", "execution_mode": "usdu_tiled"})

    def test_hires_schema_defaults_to_one_x_and_normalizes_tenths(self):
        request = generation_request()
        self.assertFalse(request.hires.enabled)
        self.assertEqual(request.hires.scale, 1.0)
        self.assertEqual(HiresInput(model="model.pth", scale=2.26).scale, 2.3)
        with self.assertRaises(ValidationError):
            HiresInput(model="model.pth", scale=4.1)
        HiresInput(enabled=True, model="model.pth", steps=1, denoise=0.05)
        HiresInput(enabled=False, tile_size=32, tile_overlap=512)
        with self.assertRaises(ValidationError):
            HiresInput(enabled=True, model="model.pth", tile_size=32, tile_overlap=17)

    def test_memory_workload_uses_larger_real_base_or_serial_hires_request(self):
        base_heavy = generation_request(width=512, height=512, images_per_batch=10, hires={"enabled": True, "model": "model.pth", "scale": 2, "cfg": 7})
        hires_heavy = generation_request(width=512, height=512, images_per_batch=1, hires={"enabled": True, "model": "model.pth", "scale": 4, "cfg": 7})
        self.assertEqual(inference_server.generation_memory_workload(base_heavy, "sd"), (512, 512, 7.0, 10))
        self.assertEqual(inference_server.generation_memory_workload(hires_heavy, "sd"), (2048, 2048, 7.0, 1))

    def test_anima_usdu_memory_admission_uses_processing_tile_not_full_target(self):
        request = generation_request(
            engine="Anima", checkpoint=None, diffusion_model="a.safetensors", text_encoder="b.safetensors", vae="c.safetensors", preview_enabled=False, width=1088, height=1472, images_per_batch=1,
            hires={"enabled": True, "model": "model.pth", "scale": 2, "cfg": 7,
                   "execution_mode": "usdu_tiled", "padding": 32},
        )
        details = inference_server.generation_memory_workload_diagnostics(request, "anima")
        self.assertEqual(details["admission"], (1120, 1504, 7.0, 1))
        self.assertEqual(details["base_dimensions"], (1088, 1472))
        self.assertEqual(details["target_dimensions"], (2176, 2944))
        self.assertEqual(details["core_dimensions"], (1088, 1472))
        self.assertEqual(details["processing_dimensions"], (1120, 1504))
        self.assertTrue(details["target_excluded"])

    def test_anima_full_frame_memory_admission_keeps_target_and_non_two_x_tiles_round(self):
        request = generation_request(
            engine="Anima", checkpoint=None, diffusion_model="a.safetensors", text_encoder="b.safetensors", vae="c.safetensors", preview_enabled=False, width=704, height=960,
            hires={"enabled": True, "model": "model.pth", "scale": 1.5, "cfg": 7,
                   "execution_mode": "full_frame"},
        )
        details = inference_server.generation_memory_workload_diagnostics(request, "anima")
        self.assertEqual(details["target_dimensions"], (1088, 1472))
        self.assertEqual(details["admission_dimensions"], (1088, 1472))
        self.assertFalse(details["target_excluded"])

    def test_anima_usdu_rtx_before_hires_uses_rtx_canvas_as_auto_core(self):
        request = generation_request(
            engine="Anima", checkpoint=None, diffusion_model="a.safetensors", text_encoder="b.safetensors", vae="c.safetensors", preview_enabled=False,
            width=512, height=768, postprocess_order=["rtx", "hires", "adetailer"],
            rtx={"enabled": True, "scale": 2},
            hires={"enabled": True, "model": "model.pth", "scale": 2, "cfg": 7,
                   "execution_mode": "usdu_tiled", "padding": 32},
        )
        details = inference_server.generation_memory_workload_diagnostics(request, "anima")
        self.assertEqual(details["base_dimensions"], (512, 768))
        self.assertEqual(details["core_dimensions"], (1024, 1536))
        self.assertEqual(details["target_dimensions"], (2048, 3072))
        self.assertEqual(details["processing_dimensions"], (1056, 1568))
        self.assertEqual(details["admission_dimensions"], (1056, 1568))

    def test_logical_diffusion_step_accounts_for_second_order_callbacks(self):
        pipeline = SimpleNamespace(num_timesteps=40)
        self.assertEqual(inference_server.logical_diffusion_step(pipeline, 0, 20), 1)
        self.assertEqual(inference_server.logical_diffusion_step(pipeline, 1, 20), 1)
        self.assertEqual(inference_server.logical_diffusion_step(pipeline, 2, 20), 2)
        self.assertEqual(inference_server.logical_diffusion_step(pipeline, 39, 20), 20)

    def test_postprocessing_order_follows_user_selection(self):
        before = generation_request(
            hires={"enabled": True, "model": "model.pth"},
            adetailer={"enabled": True, "units": [{"detector": "face.pt"}]},
            postprocess_order=["hires", "adetailer", "rtx"],
        )
        after = before.model_copy(update={"postprocess_order": ["adetailer", "hires", "rtx"]})
        self.assertEqual(inference_server.postprocessing_stages(before), ["hires", "adetailer"])
        self.assertEqual(inference_server.postprocessing_stages(after), ["adetailer", "hires"])

    def test_renamed_model_is_discovered_by_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "renamed-anything.pth"
            path.write_bytes(b"weights")
            with patch.object(upscaler, "runtime_available", return_value=True), patch.object(upscaler, "load_descriptor", return_value=FakeDescriptor()):
                models = upscaler.discover_models(root)
            self.assertEqual(models[0]["id"], "renamed-anything.pth")
            self.assertTrue(models[0]["compatible"])
            self.assertEqual(models[0]["architecture"], "TestSR")
            self.assertEqual(models[0]["scale"], 2)

    def test_one_x_refines_without_pixel_upscale_and_four_x_uses_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "model.pth"
            path.write_bytes(b"weights")
            metadata = {"compatible": True, "architecture": "TestSR", "scale": 2}
            image = Image.new("RGB", (64, 64), "red")
            with patch.object(upscaler, "resolve_model", return_value=(path, metadata)), patch.object(upscaler, "load_descriptor", return_value=FakeDescriptor()):
                one_x, one_result = upscaler.upscale_image(image, root, "model.pth", 1.0, tile_size=64, overlap=0)
                four_x, four_result = upscaler.upscale_image(image, root, "model.pth", 4.0, tile_size=64, overlap=0)
            self.assertEqual(one_x.size, (64, 64))
            self.assertFalse(one_result["upscaler_applied"])
            self.assertEqual(four_x.size, (256, 256))
            self.assertTrue(four_result["upscaler_applied"])

    def test_overlapping_tiles_are_blended_instead_of_hard_pasted(self):
        image = Image.new("RGB", (56, 32), "black")

        def local_gradient(_descriptor, patch):
            row = np.linspace(0, 255, patch.width * 2, dtype=np.uint8)
            pixels = np.repeat(row.reshape(1, -1, 1), patch.height * 2, axis=0)
            return Image.fromarray(np.repeat(pixels, 3, axis=2), "RGB")

        with patch.object(upscaler, "_upscale_patch", side_effect=local_gradient):
            output, completed = upscaler._upscale_once(image, FakeDescriptor(), 32, 8, None, None, 0, 2)
        values = np.asarray(output)[32, :, 0].astype(np.int16)
        self.assertEqual(output.size, (112, 64))
        self.assertEqual(completed, 2)
        self.assertLess(np.abs(np.diff(values)).max(), 40)

    def test_torch_models_are_loaded_with_weights_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pth"
            path.write_bytes(b"weights")
            with patch.object(torch, "load", return_value={}) as loader:
                upscaler._state_dict(path)
            loader.assert_called_once_with(str(path), map_location="cpu", weights_only=True)

    def test_staged_hires_requests_latents_and_uses_staged_vae_decode(self):
        request = generation_request(hires={"enabled": True, "model": "model.pth", "scale": 1, "denoise": 1, "steps": 1})
        image = Image.new("RGB", (64, 64), "red")
        calls = {}

        class ImagePipeline:
            def set_progress_bar_config(self, **_kwargs):
                pass

            def __call__(self, **kwargs):
                calls.update(kwargs)
                return SimpleNamespace(images=torch.zeros((1, 4, 8, 8)))

            def maybe_free_model_hooks(self):
                pass

        class Control:
            cancelled = False

            def checkpoint(self, *_args):
                pass

            def active_elapsed(self, _started_at):
                return 0

            def total_paused(self):
                return 0

        diagnostics = {"model": "model.pth"}
        with patch.dict(inference_server.performance_settings, {"staged_vae_decode": True}), \
                patch.object(inference_server, "upscale_image", return_value=(image, diagnostics)), \
                patch.object(inference_server.AutoPipelineForImage2Image, "from_pipe", return_value=ImagePipeline()), \
                patch.object(inference_server, "prepare_prompt_conditioning", return_value={}), \
                patch.object(inference_server, "decode_staged_latents", return_value=[image]) as decode:
            result, result_diagnostics = inference_server.apply_hires_fix(
                image, object(), "sd", request, "missing-job", Control(), 0,
                effective_hires_seed=request.seed,
            )
        self.assertEqual(calls["output_type"], "latent")
        self.assertEqual(result.size, (64, 64))
        self.assertEqual(result_diagnostics["effective_steps"], 1)
        decode.assert_called_once()

    def test_sd_full_frame_hires_generator_uses_the_effective_hires_seed(self):
        image = Image.new("RGB", (64, 64), "red")
        calls = {}

        class ImagePipeline:
            def set_progress_bar_config(self, **_kwargs):
                pass

            def __call__(self, **kwargs):
                calls.update(kwargs)
                return SimpleNamespace(images=torch.zeros((1, 4, 8, 8)))

            def maybe_free_model_hooks(self):
                pass

        class Control:
            cancelled = False

            def checkpoint(self, *_args):
                pass

            def active_elapsed(self, _started_at):
                return 0

            def total_paused(self):
                return 0

        cases = (
            ({}, 1, "inherit"),
            ({"mode": "fixed", "seed": "885289963651097"}, 885289963651097, "fixed"),
            ({"mode": "random"}, 9007199254740993, "random"),
        )
        for overrides, effective, mode in cases:
            with self.subTest(mode=mode):
                calls.clear()
                request = generation_request(
                    hires={"enabled": True, "model": "model.pth", "scale": 1, "denoise": 1, "steps": 1, **overrides},
                )
                with patch.dict(inference_server.performance_settings, {"staged_vae_decode": True}), \
                        patch.object(inference_server, "upscale_image", return_value=(image, {"model": "model.pth"})), \
                        patch.object(inference_server.AutoPipelineForImage2Image, "from_pipe", return_value=ImagePipeline()), \
                        patch.object(inference_server, "prepare_prompt_conditioning", return_value={}), \
                        patch.object(inference_server, "decode_staged_latents", return_value=[image]):
                    _, diagnostics = inference_server.apply_hires_fix(
                        image, object(), "sd", request, "missing-job", Control(), 0,
                        image_seed=1, effective_hires_seed=effective,
                    )
                self.assertEqual(calls["generator"].initial_seed(), effective)
                self.assertEqual(diagnostics["execution_mode"], "full_frame")
                self.assertEqual(diagnostics["base_seed"], "1")
                self.assertEqual(diagnostics["hires_seed_mode"], mode)
                self.assertEqual(diagnostics["hires_seed"], str(effective))

    def test_hires_stage_and_save_refuse_an_unresolved_effective_hires_seed(self):
        request = generation_request(hires={"enabled": True, "model": "model.pth", "scale": 1, "denoise": 1, "steps": 1})
        image = Image.new("RGB", (8, 8), "red")

        class Control:
            cancelled = False

            def checkpoint(self, *_args):
                pass

            def active_elapsed(self, _started_at):
                return 0

            def total_paused(self):
                return 0

        with self.assertRaisesRegex(ValueError, "must be resolved before the Hires stage"):
            inference_server.apply_hires_fix(image, object(), "sd", request, "missing-job", Control(), 0, image_seed=1)

        disabled = generation_request()
        self.assertEqual(inference_server.apply_hires_fix(image, object(), "sd", disabled, "missing-job", Control(), 0), (image, None))

        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                with self.assertRaisesRegex(ValueError, "required when saving a Hires output"):
                    inference_server.save_image(image, "job", request, 1, 1, 0, hires_result={"model": "model.pth"})
                self.assertEqual(list(Path(temporary).rglob("*.png")), [])
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_anima_hires_uses_native_refinement_without_diffusers_or_staged_decode(self):
        request = GenerateInput(
            engine="Anima",
            diffusion_model="diffusion.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            prompt="parent prompt",
            negative_prompt="negative prompt",
            width=512,
            height=512,
            steps=20,
            cfg=6,
            denoise=1,
            seed=5,
            sampler="euler_ancestral",
            scheduler="simple",
            guidance="pag",
            pag={"scale": 0.8, "applied_layers": "all"},
            preview_enabled=False,
            hires={"enabled": True, "model": "model.pth", "mode": "fixed", "seed": "885289963651097", "scale": 2, "steps": 20, "denoise": 0.5, "cfg": 4},
        )
        source = Image.new("RGB", (64, 64), "red")
        upscaled = Image.new("RGB", (128, 128), "green")
        refined = Image.new("RGB", (128, 128), "blue")

        class Runtime:
            last_generation_metrics = {"sampling": {"seconds": 1.0}}

            def refine_batch(self, **kwargs):
                self.kwargs = kwargs
                kwargs["on_step"](1, 20, None)
                kwargs["on_step"](20, 20, None)
                self.last_generation_metrics = {
                    **self.last_generation_metrics,
                    "refinement.sampling": {
                        "seconds": 0.2,
                        "schedule_steps": 40,
                        "schedule_mode": "comfy_suffix",
                        "start_sigma": 0.61,
                    },
                }
                return [refined]

        class Control:
            cancelled = False

            def checkpoint(self, *_args):
                pass

            def active_elapsed(self, _started_at):
                return 0

            def total_paused(self):
                return 0

        runtime = Runtime()
        inference_server.jobs["anima-hires"] = {"runtime_metrics": {"sampling": {"seconds": 1.0}}}
        try:
            with (
                patch.dict(inference_server.performance_settings, {"staged_vae_decode": True}),
                patch.object(inference_server, "pipeline_cpu_parked", False),
                patch.object(inference_server.torch.cuda, "is_available", return_value=True),
                patch.object(inference_server, "upscale_image", return_value=(upscaled, {"model": "model.pth"})),
                patch.object(inference_server, "park_pipeline_for_external_stage") as park,
                patch.object(inference_server, "restore_parked_pipeline") as restore,
                patch.object(
                    inference_server.AutoPipelineForImage2Image,
                    "from_pipe",
                    side_effect=AssertionError("Diffusers image pipeline used"),
                ),
                patch.object(
                    inference_server,
                    "prepare_prompt_conditioning",
                    side_effect=AssertionError("CLIP conditioning used"),
                ),
                patch.object(
                    inference_server,
                    "decode_staged_latents",
                    side_effect=AssertionError("staged VAE decode used"),
                ),
            ):
                result, diagnostics = inference_server.apply_hires_fix(
                    source, runtime, "anima", request, "anima-hires", Control(), 0,
                    image_seed=17, effective_hires_seed=885289963651097,
                )
            park.assert_called_once_with(runtime, "anima")
            restore.assert_not_called()
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(runtime.kwargs["images"][0].size, (128, 128))
            self.assertEqual(runtime.kwargs["prompt"], "parent prompt")
            self.assertEqual(runtime.kwargs["negative_prompt"], "negative prompt")
            self.assertEqual(runtime.kwargs["steps"], 20)
            self.assertEqual(runtime.kwargs["denoise"], 0.5)
            self.assertEqual(runtime.kwargs["cfg"], 4)
            self.assertEqual(runtime.kwargs["sampler"], "euler_ancestral")
            self.assertEqual(runtime.kwargs["scheduler"], "simple")
            self.assertEqual(runtime.kwargs["guidance"], "pag")
            self.assertEqual(runtime.kwargs["pag_scale"], 0.8)
            self.assertEqual(runtime.kwargs["pag_applied_layers"], "all")
            self.assertEqual(runtime.kwargs["generators"][0].initial_seed(), 885289963651097)
            self.assertEqual(diagnostics["base_seed"], "17")
            self.assertEqual(diagnostics["hires_seed_mode"], "fixed")
            self.assertEqual(diagnostics["hires_seed"], "885289963651097")
            self.assertIsNone(runtime.kwargs["masks"])
            self.assertEqual(diagnostics["effective_steps"], 20)
            self.assertEqual(diagnostics["schedule_steps"], 40)
            self.assertEqual(diagnostics["schedule_mode"], "comfy_suffix")
            self.assertEqual(diagnostics["start_sigma"], 0.61)
            self.assertEqual(inference_server.jobs["anima-hires"]["stage_step"], 20)
            self.assertEqual(inference_server.jobs["anima-hires"]["stage_total"], 20)
            self.assertIn("sampling", inference_server.jobs["anima-hires"]["runtime_metrics"])
            self.assertIn(
                "hires.refinement.sampling",
                inference_server.jobs["anima-hires"]["runtime_metrics"],
            )
        finally:
            inference_server.jobs.pop("anima-hires", None)

    def test_anima_hires_propagates_refinement_schedule_limit_error(self):
        request = GenerateInput(
            engine="Anima", diffusion_model="diffusion.safetensors", text_encoder="text.safetensors", vae="vae.safetensors",
            prompt="test", width=512, height=512, steps=20, cfg=7, denoise=1, seed=1,
            sampler="euler_ancestral", scheduler="normal", preview_enabled=False,
            hires={"enabled": True, "model": "model.pth", "scale": 1, "steps": 100, "denoise": 100 / 4097},
        )
        image = Image.new("RGB", (64, 64), "red")

        class Runtime:
            def refine_batch(self, **_kwargs):
                raise ValueError(
                    "Anima refinement denoise requires 4097 schedule steps, exceeding the limit of 4096; "
                    "increase denoise or reduce steps"
                )

        class Control:
            cancelled = False

            def checkpoint(self, *_args):
                pass

            def active_elapsed(self, _started_at):
                return 0

            def total_paused(self):
                return 0

        with patch.object(inference_server, "upscale_image", return_value=(image, {"model": "model.pth"})):
            with self.assertRaisesRegex(ValueError, "4097 schedule steps.*increase denoise or reduce steps"):
                inference_server.apply_hires_fix(
                    image, Runtime(), "anima", request, "missing-job", Control(), 0,
                    effective_hires_seed=request.seed,
                )

    def test_anima_usdu_auto_tiles_prepare_once_reseed_and_compose_row_major(self):
        request = GenerateInput(
            engine="Anima", diffusion_model="diffusion.safetensors", text_encoder="text.safetensors", vae="vae.safetensors",
            prompt="parent", width=1088, height=1472, steps=30, cfg=5, denoise=1, seed=9,
            sampler="euler", scheduler="simple", preview_enabled=False,
            hires={"enabled": True, "model": "model.pth", "mode": "fixed", "seed": "885289963651097", "scale": 2, "steps": 12, "denoise": .2, "cfg": 5,
                   "execution_mode": "usdu_tiled", "sampler": "euler_ancestral", "scheduler": "normal"},
        )
        source, upscaled = Image.new("RGB", (1088, 1472), "black"), Image.new("RGB", (2176, 2944), "black")

        class Runtime:
            last_generation_metrics = {}
            def __init__(self): self.calls = []; self.prepared = []
            def prepare_refinement_conditioning(self, *args): self.prepared.append(("conditioning", args)); return "prepared-conditioning"
            def prepare_refinement_sigmas(self, *args): self.prepared.append(("sigmas", args)); return "prepared-sigmas"
            def refine_batch(self, **kwargs):
                self.calls.append(kwargs)
                kwargs["on_step"](1, 12, None); kwargs["on_step"](12, 12, None)
                index = len(self.calls)
                self.last_generation_metrics = {
                    "refinement.vae_encode": {"seconds": index / 10},
                    "refinement.sampling": {"seconds": index, "schedule_steps": 60, "schedule_construction_steps": 60, "executed_denoise_updates": 12, "sequential_transformer_invocations": 24, "actual_transformer_invocations": 24, "peak_batch_copies": 1, "schedule_mode": f"mode-{index}", "start_sigma": index / 10, "latent_state_dtype": "float32", "transformer_input_dtype": "bfloat16", "conditioning_reused": True, "sigmas_reused": True},
                    "refinement.vae_decode": {"seconds": index / 100, "requested_tiled_decode": {"tile": 512, "overlap": 64}, "resolved_tiled_decode": {"mode": f"fake-{index}"}, "actual_vae_mode": "tiled"},
                }
                return [Image.new("RGB", kwargs["images"][0].size, (len(self.calls), 0, 0))]

        class Control:
            cancelled = False
            def checkpoint(self, *_args): pass
            def active_elapsed(self, _started_at): return 0
            def total_paused(self): return 0

        runtime = Runtime()
        inference_server.jobs["usdu"] = {}
        try:
            with patch.object(inference_server, "upscale_image", return_value=(upscaled, {"model": "model.pth"})), \
                    patch.object(inference_server.torch.cuda, "is_available", return_value=False):
                result, diagnostics = inference_server.apply_hires_fix(
                    source, runtime, "anima", request, "usdu", Control(), 0,
                    image_seed=17, effective_hires_seed=885289963651097,
                )
            self.assertEqual(result.size, (2176, 2944))
            self.assertEqual(len(runtime.prepared), 2)
            self.assertEqual(len(runtime.calls), 4)
            self.assertEqual([call["generators"][0].initial_seed() for call in runtime.calls], [885289963651097] * 4)
            self.assertEqual(len({id(call["generators"][0]) for call in runtime.calls}), 4)
            self.assertTrue(all(call["prepared_conditioning"] == "prepared-conditioning" for call in runtime.calls))
            self.assertTrue(all(call["prepared_sigmas"] == "prepared-sigmas" for call in runtime.calls))
            self.assertTrue(all(call["force_tiled_decode"] for call in runtime.calls))
            self.assertEqual([call["images"][0].size for call in runtime.calls], [(1120, 1504)] * 4)
            self.assertEqual((diagnostics["rows"], diagnostics["cols"], diagnostics["tile_count"]), (2, 2, 4))
            self.assertEqual(diagnostics["core_tile"], (1088, 1472))
            self.assertEqual(diagnostics["model_size"], (1120, 1504))
            self.assertEqual(diagnostics["total_steps"], 48)
            self.assertEqual(inference_server.jobs["usdu"]["stage_step"], 48)
            self.assertEqual(inference_server.jobs["usdu"]["stage_total"], 48)
            self.assertEqual(diagnostics["sampler_resolved"], "euler_ancestral")
            self.assertEqual(diagnostics["scheduler_resolved"], "normal")
            self.assertEqual([item["row"] for item in diagnostics["tile_metrics"]], [0, 0, 1, 1])
            self.assertEqual([item["col"] for item in diagnostics["tile_metrics"]], [0, 1, 0, 1])
            self.assertEqual([item["hires_seed"] for item in diagnostics["tile_metrics"]], ["885289963651097"] * 4)
            self.assertEqual((diagnostics["base_seed"], diagnostics["hires_seed_mode"], diagnostics["hires_seed"]), ("17", "fixed", "885289963651097"))
            self.assertEqual([item["sampling_metrics"]["schedule_mode"] for item in diagnostics["tile_metrics"]], ["mode-1", "mode-2", "mode-3", "mode-4"])
            self.assertEqual([item["tiled_decode_resolved"]["mode"] for item in diagnostics["tile_metrics"]], ["fake-1", "fake-2", "fake-3", "fake-4"])
            self.assertEqual(diagnostics["aggregate"]["conditioning_reused_count"], 4)
            aggregate = diagnostics["aggregate"]
            self.assertEqual(aggregate["schedule_construction_steps"], 240)
            self.assertEqual(aggregate["executed_denoise_updates"], 48)
            self.assertEqual(aggregate["sequential_transformer_invocations"], 96)
            self.assertEqual(aggregate["actual_transformer_invocations"], 96)
            self.assertEqual(aggregate["peak_batch_copies"], 1)
            self.assertTrue(aggregate["actual_transformer_invocations_complete"])
            json.dumps(diagnostics)
        finally:
            inference_server.jobs.pop("usdu", None)

    def test_usdu_sampling_aggregate_handles_batch_and_unknown_metrics(self):
        batch_tiles = [{"sampling_metrics": {
            "schedule_construction_steps": 60, "executed_denoise_updates": 12,
            "sequential_transformer_invocations": 24, "actual_transformer_invocations": 12,
            "peak_batch_copies": 2,
        }} for _ in range(4)]
        aggregate = inference_server.aggregate_usdu_sampling_metrics(batch_tiles)
        self.assertEqual(aggregate["schedule_construction_steps"], 240)
        self.assertEqual(aggregate["executed_denoise_updates"], 48)
        self.assertEqual(aggregate["sequential_transformer_invocations"], 96)
        self.assertEqual(aggregate["actual_transformer_invocations"], 48)
        self.assertEqual(aggregate["peak_batch_copies"], 2)
        self.assertTrue(aggregate["actual_transformer_invocations_complete"])

        partial = inference_server.aggregate_usdu_sampling_metrics([
            batch_tiles[0], {"sampling_metrics": {"actual_transformer_invocations": "unknown"}}, {},
        ])
        self.assertEqual(partial["actual_transformer_invocations"], 12)
        self.assertEqual(partial["actual_transformer_invocations_known_count"], 1)
        self.assertFalse(partial["actual_transformer_invocations_complete"])

        unknown = inference_server.aggregate_usdu_sampling_metrics([{}, {"sampling_metrics": {"peak_batch_copies": None}}])
        self.assertIsNone(unknown["actual_transformer_invocations"])
        self.assertEqual(unknown["actual_transformer_invocations_known_count"], 0)
        self.assertFalse(unknown["actual_transformer_invocations_complete"])
        self.assertIsNone(unknown["peak_batch_copies"])
        self.assertEqual(unknown["peak_batch_copies_known_count"], 0)
        self.assertFalse(unknown["peak_batch_copies_complete"])

    def test_hires_metadata_records_actual_output_size(self):
        request = generation_request(hires={"enabled": True, "model": "renamed.pth", "mode": "fixed", "seed": "18446744073709551615", "scale": 2.0})
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                output = inference_server.save_image(
                    Image.new("RGB", (1024, 768)),
                    "job",
                    request,
                    1.0,
                    1.0,
                    0.0,
                    hires_result={"model": "renamed.pth", "scale": 2.0},
                    seed=9007199254740993,
                    effective_hires_seed=0xFFFFFFFFFFFFFFFF,
                )
                with Image.open(output) as saved:
                    parameters = json.loads(saved.info["parameters"])
                self.assertEqual(parameters["output_width"], 1024)
                self.assertEqual(parameters["output_height"], 768)
                self.assertEqual(parameters["hires"]["model"], "renamed.pth")
                self.assertEqual(parameters["seed"], "9007199254740993")
                self.assertEqual(parameters["base_seed"], "9007199254740993")
                self.assertEqual(parameters["hires_seed_mode"], "fixed")
                self.assertEqual(parameters["hires_seed"], "18446744073709551615")
                record = inference_server.history_file_record(output)
                self.assertEqual(record["base_seed"], parameters["base_seed"])
                self.assertEqual(record["hires_seed_mode"], parameters["hires_seed_mode"])
                self.assertEqual(record["hires_seed"], parameters["hires_seed"])
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_non_hires_png_seed_facts_keep_effective_hires_seed_null(self):
        request = generation_request(seed=0xFFFFFFFFFFFFFFFF)
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                output = inference_server.save_image(Image.new("RGB", (8, 8)), "job", request, 1, 1, 0)
                with Image.open(output) as saved:
                    parameters = json.loads(saved.info["parameters"])
                self.assertEqual(parameters["seed"], "18446744073709551615")
                self.assertEqual(parameters["base_seed"], parameters["seed"])
                self.assertEqual(parameters["hires_seed_mode"], "inherit")
                self.assertIsNone(parameters["hires_seed"])
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output


if __name__ == "__main__":
    unittest.main()
