"""The resample chain: its five resize methods, its size arithmetic, and what one unit executes.

The chain replaced a single Hires.fix pass whose size was always rounded up to a multiple of 64.
Two things follow from that and are what these tests hold:

- a unit's target is exact — `round(edge x scale)`, the number ComfyUI's `ImageScaleBy` computes —
  and the engine's own grid is applied separately and reported separately;
- a unit is three optional parts, so "no model", "no redraw" and "no enlargement" all have to run.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError

from backend import inference_server, resize
from backend.inference_server import GenerateInput, ResampleUnitInput


def request_with(chain, **overrides):
    fields = dict(
        engine="SD", checkpoint="model.safetensors", prompt="a lantern", width=1024, height=1024,
        steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
        postprocess_chain=chain,
    )
    fields.update(overrides)
    return GenerateInput(**fields)


def gradient(width, height):
    """A picture with structure in both axes, so a resize that ignores one is visible."""
    columns = np.linspace(0, 255, width, dtype=np.uint8)
    rows = np.linspace(0, 255, height, dtype=np.uint8)
    pixels = np.stack([
        np.repeat(columns.reshape(1, width), height, axis=0),
        np.repeat(rows.reshape(height, 1), width, axis=1),
        np.full((height, width), 128, dtype=np.uint8),
    ], axis=2)
    return Image.fromarray(pixels, "RGB")


class ResizeMethodTests(unittest.TestCase):
    def test_every_method_comfy_offers_is_usable_and_lands_on_the_requested_size(self):
        # ComfyUI's ImageScale list, plus Forge's PIL `nearest` ahead of it.
        self.assertEqual(resize.RESIZE_METHODS, ("nearest", "nearest-exact", "bilinear", "area", "bicubic", "lanczos"))
        source = gradient(64, 48)
        for method in resize.RESIZE_METHODS:
            with self.subTest(method=method):
                result = resize.resize_image(source, 97, 33, method)
                self.assertEqual(result.size, (97, 33))
                self.assertEqual(result.mode, "RGB")

    def test_forge_nearest_is_pils_own_pass_and_not_torch_nearest_exact(self):
        source = gradient(40, 30)
        result = resize.resize_image(source, 97, 33, "nearest")
        self.assertEqual(result.tobytes(), source.resize((97, 33), Image.Resampling.NEAREST).tobytes())
        self.assertNotEqual(result.tobytes(), resize.resize_image(source, 97, 33, "nearest-exact").tobytes())

    def test_an_unknown_method_is_refused_rather_than_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, "Unsupported resize algorithm"):
            resize.resize_image(gradient(8, 8), 4, 4, "bislerp")

    def test_lanczos_is_pils_own_pass_so_the_result_is_reproducible(self):
        source = gradient(40, 30)
        first = resize.resize_image(source, 80, 60, "lanczos")
        second = resize.resize_image(source, 80, 60, "lanczos")
        self.assertEqual(first.tobytes(), second.tobytes())
        # Comfy's `lanczos` is uint8 RGB through PIL and back, so the same call reproduces it.
        self.assertEqual(first.tobytes(), source.resize((80, 60), Image.Resampling.LANCZOS).tobytes())

    def test_an_interpolate_method_matches_the_torch_call_comfy_makes(self):
        source = gradient(40, 30)
        result = resize.resize_image(source, 80, 60, "bilinear")
        self.assertEqual(result.tobytes(), resize.resize_image(source, 80, 60, "bilinear").tobytes())
        # `comfy/utils.py:1095`: BCHW floats, size as (height, width), torch's own align_corners.
        tensor = torch.from_numpy(np.asarray(source, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        expected = torch.nn.functional.interpolate(tensor, size=(60, 80), mode="bilinear")
        expected_image = Image.fromarray(
            np.rint(expected.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8), "RGB",
        )
        self.assertEqual(result.tobytes(), expected_image.tobytes())

    def test_a_resize_to_the_size_it_already_is_changes_no_pixel(self):
        source = gradient(24, 24)
        for method in resize.RESIZE_METHODS:
            with self.subTest(method=method):
                self.assertEqual(resize.resize_image(source, 24, 24, method).tobytes(), source.tobytes())


class ChainSizeTests(unittest.TestCase):
    def test_a_target_is_the_exact_product_and_not_a_multiple_of_sixty_four(self):
        # The acceptance example: 1088 x 1.5 is 1632, which is not a multiple of 64 and runs anyway.
        self.assertEqual(resize.scaled_size((1088, 1088), 1.5), (1632, 1632))
        self.assertEqual(resize.scaled_size((1024, 768), 1.5), (1536, 1152))
        # `round` is half-to-even in both Python and Comfy, so the ties agree.
        self.assertEqual(resize.scaled_size((5, 5), 1.1), (6, 6))
        self.assertEqual(resize.scaled_size((5, 7), 1.5), (8, 10))

    def test_each_engine_rounds_the_redraw_canvas_to_its_own_grid(self):
        self.assertEqual(inference_server.engine_pixel_alignment("sd"), 8)
        self.assertEqual(inference_server.engine_pixel_alignment("anima"), 32)
        for family in ("flux", "flux2", "krea2"):
            self.assertEqual(inference_server.engine_pixel_alignment(family), 16)
        unit = ResampleUnitInput(enabled=True, scale=1.5, denoise=0.3)
        # 1090 x 1.5 = 1635: off every grid, and each engine moves it to its nearest own multiple.
        self.assertEqual(inference_server.resample_unit_sizes((1090, 1090), unit, "sd")["redraw"], (1632, 1632))
        self.assertEqual(inference_server.resample_unit_sizes((1090, 1090), unit, "anima")["redraw"], (1632, 1632))
        self.assertEqual(inference_server.resample_unit_sizes((1090, 1090), unit, "flux")["redraw"], (1632, 1632))
        # The target itself never moves, whichever engine is mounted.
        for family in ("sd", "anima", "flux", "flux2", "krea2"):
            self.assertEqual(inference_server.resample_unit_sizes((1090, 1090), unit, family)["target"], (1635, 1635))

    def test_a_unit_that_only_resizes_lands_on_its_exact_target(self):
        # No redraw means no latent grid to satisfy, so nothing rounds it.
        resize_only = ResampleUnitInput(enabled=True, scale=1.5, denoise=0)
        sizes = inference_server.resample_unit_sizes((1090, 1090), resize_only, "anima")
        self.assertEqual((sizes["target"], sizes["redraw"], sizes["output"]), ((1635, 1635),) * 3)

    def test_exact_output_returns_the_target_the_engine_rounded_away(self):
        rounded = ResampleUnitInput(enabled=True, scale=1.5, denoise=0.3)
        exact = ResampleUnitInput(enabled=True, scale=1.5, denoise=0.3, exact_output=True)
        self.assertEqual(inference_server.resample_unit_sizes((1090, 1090), rounded, "sd")["output"], (1632, 1632))
        self.assertEqual(inference_server.resample_unit_sizes((1090, 1090), exact, "sd")["output"], (1635, 1635))

    def test_the_chain_accumulates_each_unit_over_the_last_one(self):
        request = request_with([
            {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.3},
            {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0},
            {"type": "adetailer", "enabled": True, "units": [{"detector": "face.pt"}]},
        ], width=1024, height=1024)
        plan = inference_server.resample_chain_plan(request)
        self.assertEqual([entry["output_size"] for entry in plan], [(1536, 1536), (2304, 2304), (2304, 2304)])
        self.assertEqual(inference_server.chain_output_size(request), (2304, 2304))

    def test_a_disabled_unit_is_skipped_but_keeps_its_card_number(self):
        request = request_with([
            {"type": "resample", "enabled": False, "scale": 2},
            {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0},
        ])
        plan = inference_server.resample_chain_plan(request)
        self.assertEqual([(entry["unit_number"], entry["output_size"]) for entry in plan], [(2, (1536, 1536))])

    def test_the_envelope_error_names_the_unit_that_crossed_it(self):
        with self.assertRaisesRegex(ValidationError, "Resample unit 3"):
            request_with([
                {"type": "resample", "enabled": True, "scale": 2, "denoise": 0},
                {"type": "resample", "enabled": True, "scale": 2, "denoise": 0},
                {"type": "resample", "enabled": True, "scale": 2.5, "denoise": 0},
            ])

    def test_eight_units_fit_and_a_ninth_does_not(self):
        light = {"type": "resample", "enabled": True, "scale": 1.0, "denoise": 0.2}
        request_with([dict(light) for _ in range(8)])
        with self.assertRaisesRegex(ValidationError, "at most 8 units"):
            request_with([dict(light) for _ in range(9)])


class Control:
    cancelled = False

    def checkpoint(self, *_args):
        pass

    def active_elapsed(self, _started_at):
        return 0

    def total_paused(self):
        return 0


class ResampleUnitExecutionTests(unittest.TestCase):
    def test_a_unit_with_no_model_resizes_and_never_loads_an_upscaler(self):
        request = request_with([{"type": "resample", "enabled": True, "scale": 2, "denoise": 0}])
        source = gradient(64, 64)
        with patch.object(inference_server, "upscale_image") as upscale:
            result, diagnostics = inference_server.apply_resample_unit(
                source, object(), "sd", request, request.resample_units[0], "missing-job", Control(), 0,
                image_seed=1,
            )
        upscale.assert_not_called()
        self.assertEqual(result.size, (128, 128))
        self.assertFalse(diagnostics["upscaler_applied"])
        self.assertIsNone(diagnostics["upscaler"])
        self.assertEqual(diagnostics["execution_mode"], "resize_only")
        self.assertEqual(diagnostics["resize_algorithm"], "lanczos")
        # The picture is the algorithm's own output, not an approximation of it.
        self.assertEqual(result.tobytes(), resize.resize_image(source, 128, 128, "lanczos").tobytes())

    def test_a_unit_at_denoise_zero_skips_the_redraw_entirely(self):
        request = request_with([{"type": "resample", "enabled": True, "scale": 2, "denoise": 0, "upscaler": "x.pth"}])
        source = gradient(64, 64)
        enlarged = gradient(128, 128)

        class Pipeline:
            def __getattr__(self, name):
                raise AssertionError(f"the redraw touched the pipeline ({name})")

        with patch.object(inference_server, "upscale_image", return_value=(enlarged, {"upscaler_applied": True})):
            result, diagnostics = inference_server.apply_resample_unit(
                source, Pipeline(), "sd", request, request.resample_units[0], "missing-job", Control(), 0,
                image_seed=1,
            )
        self.assertEqual(result.size, (128, 128))
        self.assertEqual(diagnostics["execution_mode"], "resize_only")
        self.assertIsNone(diagnostics["seed"])
        self.assertEqual(diagnostics["target_size"], [128, 128])
        self.assertEqual(diagnostics["executed_size"], [128, 128])

    def test_the_model_pass_is_told_the_destination_and_the_algorithm(self):
        request = request_with([{
            "type": "resample", "enabled": True, "scale": 1.5, "denoise": 0,
            "upscaler": "x.pth", "resize_algorithm": "bicubic",
        }])
        calls = {}

        def upscale(image, directory, reference, scale, *args, **kwargs):
            calls.update(kwargs)
            calls["reference"] = reference
            return gradient(*kwargs["destination"]), {"upscaler_applied": True}

        source = gradient(1090, 1090)
        with patch.object(inference_server, "upscale_image", side_effect=upscale):
            result, _ = inference_server.apply_resample_unit(
                source, object(), "sd", request, request.resample_units[0], "missing-job", Control(), 0,
                image_seed=1,
            )
        self.assertEqual(calls["reference"], "x.pth")
        self.assertEqual(calls["method"], "bicubic")
        # No redraw follows, so the destination is the exact target rather than an aligned one.
        self.assertEqual(calls["destination"], (1635, 1635))
        self.assertEqual(result.size, (1635, 1635))

    def test_the_target_and_the_size_actually_executed_are_both_recorded(self):
        request = request_with([{"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.5, "steps": 4}])
        source = gradient(64, 64)

        class ImagePipeline:
            def set_progress_bar_config(self, **_kwargs):
                pass

            def __call__(self, **kwargs):
                return type("Output", (), {"images": [Image.new("RGB", kwargs["image"].size, "red")]})()

            def maybe_free_model_hooks(self):
                pass

        with (
            patch.dict(inference_server.performance_settings, {"staged_vae_decode": False}),
            patch.object(inference_server.AutoPipelineForImage2Image, "from_pipe", return_value=ImagePipeline()),
            patch.object(inference_server, "prepare_dynamic_prompt_conditioning", return_value={}),
        ):
            result, diagnostics = inference_server.apply_resample_unit(
                source, object(), "sd", request, request.resample_units[0], "missing-job", Control(), 0,
                image_seed=1, effective_seed=1,
            )
        # 64 x 1.5 = 96, already on SD's grid of 8, so the two agree and the record says so.
        self.assertEqual(diagnostics["target_size"], [96, 96])
        self.assertEqual(diagnostics["executed_size"], [96, 96])
        self.assertTrue(diagnostics["size_matches_target"])
        self.assertEqual(result.size, (96, 96))
        self.assertEqual(diagnostics["steps"], 4)
        # `cfg` was left to inherit, so the record shows what the request carried.
        self.assertEqual(diagnostics["cfg"], 7.0)

    def test_exact_output_resizes_the_redrawn_picture_back_onto_its_target(self):
        request = request_with([{
            "type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.5, "steps": 4, "exact_output": True,
        }])
        # 70 x 1.5 = 105, which SD's grid of 8 moves to 104. `exact_output` asks for 105 back.
        source = gradient(70, 70)

        class ImagePipeline:
            def set_progress_bar_config(self, **_kwargs):
                pass

            def __call__(self, **kwargs):
                return type("Output", (), {"images": [Image.new("RGB", kwargs["image"].size, "red")]})()

            def maybe_free_model_hooks(self):
                pass

        with (
            patch.dict(inference_server.performance_settings, {"staged_vae_decode": False}),
            patch.object(inference_server.AutoPipelineForImage2Image, "from_pipe", return_value=ImagePipeline()),
            patch.object(inference_server, "prepare_dynamic_prompt_conditioning", return_value={}),
        ):
            result, diagnostics = inference_server.apply_resample_unit(
                source, object(), "sd", request, request.resample_units[0], "missing-job", Control(), 0,
                image_seed=1, effective_seed=1,
            )
        self.assertEqual(result.size, (105, 105))
        self.assertEqual(diagnostics["target_size"], [105, 105])
        self.assertEqual(diagnostics["pre_exact_size"], [104, 104])
        self.assertTrue(diagnostics["exact_output_applied"])
        self.assertTrue(diagnostics["size_matches_target"])


if __name__ == "__main__":
    unittest.main()
