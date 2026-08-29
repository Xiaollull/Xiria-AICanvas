import os
import tempfile
import unittest
import base64
import io
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from pydantic import ValidationError

from backend.adetailer import (
    Detection,
    detection_mask,
    discover_detector_models,
    expand_prompt,
    render_detection_preview,
    resolve_detector_model,
    select_detections,
)
from backend import inference_server
from backend.inference_server import (
    ADetailerInput,
    ADetailerUnitInput,
    GenerateInput,
    builtin_yolo_models,
    write_generation_failure_log,
)


def _encoded_source(size=(512, 512)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return {
        "enabled": True,
        "image_data": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


SOURCE = _encoded_source()


class ADetailerTests(unittest.TestCase):
    def test_generation_batch_limits_and_seed_sequence(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=0xFFFFFFFFFFFFFFFE, images_per_batch=3, batch_count=4,
            sampler="euler", scheduler="simple",
        )
        self.assertEqual(inference_server.batch_seeds(request, 0), [0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF, 0])
        self.assertEqual(inference_server.batch_seeds(request, 1), [1, 2, 3])
        generators = [object(), object(), object()]
        callback = object()
        kwargs = inference_server.batch_pipeline_kwargs(request, generators, callback, {"prompt_embeds": "encoded"})
        self.assertEqual(kwargs["num_images_per_prompt"], 3)
        self.assertIs(kwargs["generator"], generators)
        self.assertIs(kwargs["callback_on_step_end"], callback)
        self.assertEqual(kwargs["prompt_embeds"], "encoded")
        pipeline_calls = []

        class Pipeline:
            def __call__(self, **pipeline_kwargs):
                pipeline_calls.append(pipeline_kwargs)
                return type("Result", (), {"images": ["one", "two", "three"]})()

        self.assertEqual(inference_server.run_pipeline_batch(Pipeline(), kwargs), ["one", "two", "three"])
        self.assertEqual(len(pipeline_calls), 1)
        with self.assertRaises(ValidationError):
            GenerateInput(
                engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
                steps=20, cfg=7, denoise=1, seed=1, images_per_batch=11,
                sampler="euler", scheduler="simple",
            )
        with self.assertRaisesRegex(ValueError, "Generator count"):
            inference_server.batch_pipeline_kwargs(request, generators[:2], callback, {})
        with self.assertRaises(ValidationError):
            GenerateInput(
                engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
                steps=20, cfg=7, denoise=1, seed=1, batch_count=21,
                sampler="euler", scheduler="simple",
            )

    def test_detector_discovery_and_containment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            nested = root / "face"
            nested.mkdir()
            model = nested / "face.pt"
            model.write_bytes(b"model")
            (nested / "ignored.onnx").write_bytes(b"model")
            self.assertEqual(discover_detector_models(root), ["face/face.pt"])
            self.assertEqual(resolve_detector_model(root, "face/face.pt"), model)
            with self.assertRaisesRegex(ValueError, "relative"):
                resolve_detector_model(root, str(model))
            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_detector_model(root, "../outside.pt")

    def test_selection_filters_area_and_sorts_largest_first(self):
        detections = [
            Detection((0, 0, 10, 10), 0.9, "tiny"),
            Detection((10, 10, 80, 80), 0.7, "large"),
            Detection((20, 20, 60, 60), 0.8, "medium"),
        ]
        selected = select_detections(detections, (100, 100), 0.02, 0.6, 2)
        self.assertEqual([item.class_name for item in selected], ["large", "medium"])

    def test_mask_dilation_and_blur_preserve_hard_and_soft_masks(self):
        detection = Detection((4, 4, 7, 7), 0.9, "face")
        hard, soft = detection_mask(detection, (12, 12), dilate_erode=3, blur=2)
        hard_values = np.asarray(hard)
        soft_values = np.asarray(soft)
        self.assertTrue(set(np.unique(hard_values)).issubset({0, 255}))
        self.assertGreater(np.count_nonzero(hard_values), 16)
        self.assertTrue(np.any((soft_values > 0) & (soft_values < 255)))

    def test_detection_preview_draws_boxes_and_segmentation(self):
        image = Image.new("RGB", (100, 100), "black")
        mask_values = np.zeros((100, 100), dtype=np.uint8)
        mask_values[30:70, 30:70] = 255
        detection = Detection((25, 25, 75, 75), 0.876, "face", Image.fromarray(mask_values))
        preview = render_detection_preview(image, [detection])
        pixels = np.asarray(preview)
        self.assertEqual(preview.size, image.size)
        self.assertTrue(np.any(pixels[25, 25] != 0))
        self.assertTrue(np.all(pixels[50, 50] == 0))

    def test_lossless_detection_preview_keeps_source_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_directory = inference_server.PREVIEW_DIRECTORY
            inference_server.PREVIEW_DIRECTORY = Path(temporary)
            try:
                source = Image.new("RGB", (1024, 768), (32, 64, 96))
                output = inference_server.save_pil_preview(source, "job", "adetailer_detection", lossless=True)
                self.assertEqual(output.suffix, ".png")
                with Image.open(output) as preview:
                    self.assertEqual(preview.size, source.size)
                    self.assertEqual(preview.format, "PNG")
            finally:
                inference_server.PREVIEW_DIRECTORY = original_directory

    def test_prompt_inheritance_and_placeholder(self):
        self.assertEqual(expand_prompt("", "parent"), "parent")
        self.assertEqual(expand_prompt("portrait, [PROMPT]", "parent"), "portrait, parent")

    def test_schema_defaults_disabled_and_validates_enabled_configuration(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
        )
        self.assertFalse(request.adetailer.enabled)
        self.assertEqual(request.adetailer.active_units, [])
        # An enabled stage with nothing to run is a configuration error, not a
        # no-op: the user asked for detail work that cannot happen.
        with self.assertRaises(ValidationError):
            ADetailerInput(enabled=True)
        with self.assertRaises(ValidationError):
            ADetailerInput(enabled=True, units=[{"detector": "face.pt", "mask_min_ratio": 0.7, "mask_max_ratio": 0.2}])
        # A unit that will run and cannot do anything is a configuration error, and the message
        # names the slot: "ADetailer is not ready" does not say which page to open.
        with self.assertRaises(ValidationError) as error:
            ADetailerInput(enabled=True, units=[{"detector": "face.pt"}, {}])
        self.assertIn("ADetailer unit 2 requires a detector", str(error.exception))
        with self.assertRaises(ValidationError):
            ADetailerInput(enabled=True, units=[{"detector": f"m{index}.pt"} for index in range(7)])

    def test_a_disabled_stage_cannot_refuse_the_generation_it_is_not_part_of(self):
        # The editor always holds a first unit and that unit starts with no detector chosen, so a
        # request can carry `enabled: false` beside a unit that could never run. Validating the
        # detector per unit rejected the whole generation before the stage's own switch was read.
        stage = ADetailerInput(enabled=False, units=[{}])
        self.assertEqual(stage.active_units, [])
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
            adetailer={"enabled": False, "units": [{"detector": "", "denoise": 0.4}]},
        )
        self.assertFalse(request.adetailer.enabled)
        self.assertEqual(request.adetailer.active_units, [])
        # The unit's own settings are still validated; only the detector is the stage's business.
        with self.assertRaises(ValidationError):
            ADetailerUnitInput(mask_min_ratio=0.7, mask_max_ratio=0.2)

    def test_units_run_in_order_each_over_the_previous_result(self):
        """A face model and a hand model in one render: two passes, two configurations."""

        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="parent", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
            adetailer={
                "enabled": True,
                "units": [
                    {"detector": "face.pt", "denoise": 0.5, "prompt": "face"},
                    {"detector": "hand.pt", "denoise": 0.3, "prompt": "hand", "use_cfg": True, "cfg": 4},
                ],
            },
        )
        self.assertEqual([unit.detector for unit in request.adetailer.active_units], ["face.pt", "hand.pt"])

        seen = []

        def fake_unit(image, _pipeline, _family, _request, settings, *_args, **kwargs):
            seen.append({
                "image": image,
                "detector": settings.detector,
                "denoise": settings.denoise,
                "band": (kwargs["progress_start"], kwargs["progress_end"]),
                "index": kwargs["unit_index"],
                "total": kwargs["unit_total"],
            })
            produced = Image.new("RGB", (8, 8), "white")
            seen[-1]["produced"] = produced
            warning = "no regions" if settings.detector == "face.pt" else None
            return produced, {"detector": settings.detector, "detections": []}, warning

        source = Image.new("RGB", (8, 8), "black")
        with patch.object(inference_server, "apply_adetailer_unit", side_effect=fake_unit):
            result, diagnostics, warning = inference_server.apply_adetailer(
                source, Mock(), "sd", request, "job", Mock(), 0, progress_start=90, progress_end=98,
            )

        # Sequential, in the configured order, each unit told which pass it is.
        self.assertEqual([item["detector"] for item in seen], ["face.pt", "hand.pt"])
        self.assertEqual([item["denoise"] for item in seen], [0.5, 0.3])
        self.assertEqual([(item["index"], item["total"]) for item in seen], [(0, 2), (1, 2)])
        # The second pass works on what the first produced, not on the original,
        # and the stage returns what the last pass produced.
        self.assertIs(seen[0]["image"], source)
        self.assertIs(seen[1]["image"], seen[0]["produced"])
        self.assertIs(result, seen[1]["produced"])
        # Each pass owns its own slice of the stage's progress band.
        self.assertEqual([item["band"] for item in seen], [(90, 94), (94, 98)])

        # One pass finding nothing is a warning; the passes after it still ran.
        self.assertEqual(warning, "no regions")
        self.assertEqual([item["unit"] for item in diagnostics["units"]], [1, 2])
        self.assertEqual(diagnostics["detector"], "face.pt · hand.pt")

    def test_a_disabled_stage_runs_nothing_at_all(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="parent", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
            adetailer={"enabled": False, "units": [{"detector": "face.pt"}]},
        )
        source = Image.new("RGB", (8, 8), "black")
        with patch.object(
            inference_server,
            "apply_adetailer_unit",
            side_effect=AssertionError("a disabled stage ran a unit"),
        ):
            result, diagnostics, warning = inference_server.apply_adetailer(
                source, Mock(), "sd", request, "job", Mock(), 0,
            )
        self.assertIs(result, source)
        self.assertIsNone(diagnostics)
        self.assertIsNone(warning)

    def test_anima_memory_workload_includes_native_detail_canvas_and_cfg(self):
        request = GenerateInput(
            engine="Anima",
            diffusion_model="diffusion.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            prompt="test",
            width=512,
            height=512,
            steps=20,
            cfg=1,
            denoise=1,
            seed=1,
            sampler="euler",
            scheduler="simple",
            guidance="pag",
            pag={"scale": 0.7, "applied_layers": "all"},
            preview_enabled=False,
            adetailer={
                "enabled": True,
                "units": [{"detector": "face.pt", "use_cfg": True, "cfg": 20}],
            },
        )
        self.assertEqual(inference_server.generation_memory_workload(request, "anima"), (1024, 1024, 20.0, 1))

    def test_anima_detail_processing_enlarges_small_regions_to_native_canvas(self):
        self.assertEqual(inference_server.anima_detail_processing_size((0, 0, 128, 96)), (1024, 768))
        self.assertEqual(inference_server.anima_detail_processing_size((0, 0, 2048, 1024)), (1024, 512))

    def test_anima_adetailer_refines_masked_crops_with_region_seeds_without_diffusers(self):
        request = GenerateInput(
            engine="Anima",
            diffusion_model="diffusion.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            prompt="parent prompt",
            negative_prompt="parent negative",
            width=512,
            height=512,
            steps=20,
            cfg=6,
            denoise=1,
            seed=7,
            sampler="euler",
            scheduler="simple",
            guidance="pag",
            pag={"scale": 0.7, "applied_layers": "all"},
            preview_enabled=False,
            adetailer={
                "enabled": True,
                "units": [{
                    "detector": "face.pt",
                    "dilate_erode": 0,
                    "mask_blur": 4,
                    "padding": 16,
                    "denoise": 0.5,
                    "use_steps": True,
                    "steps": 12,
                    "use_cfg": True,
                    "cfg": 4,
                    "prompt": "detail [PROMPT]",
                    "negative_prompt": "detail negative",
                }],
            },
        )
        source = Image.new("RGB", (256, 192), "black")
        detections = [
            Detection((20, 20, 100, 110), 0.9, "face"),
            Detection((150, 40, 210, 110), 0.8, "face"),
        ]

        class Runtime:
            last_generation_metrics = {"sampling": {"seconds": 1.0}}

            def __init__(self):
                self.calls = []

            def refine_batch(self, **kwargs):
                self.calls.append(kwargs)
                kwargs["on_step"](1, 6, None)
                kwargs["on_step"](6, 6, None)
                self.last_generation_metrics = {
                    **self.last_generation_metrics,
                    "refinement.sampling": {"seconds": 0.1 * len(self.calls)},
                }
                return [Image.new("RGB", kwargs["images"][0].size, "white")]

        class Control:
            def checkpoint(self, *_args):
                pass

            def active_elapsed(self, _started_at):
                return 0

            def total_paused(self):
                return 0

        runtime = Runtime()
        inference_server.jobs["anima-detail"] = {"runtime_metrics": {"sampling": {"seconds": 1.0}}}
        try:
            with (
                patch.object(inference_server, "resolve_detector_model", return_value=Path("face.pt")),
                patch.object(inference_server, "run_detector", return_value=detections),
                patch.object(
                    inference_server.AutoPipelineForInpainting,
                    "from_pipe",
                    side_effect=AssertionError("Diffusers inpaint pipeline used"),
                ),
                patch.object(
                    inference_server,
                    "prepare_prompt_conditioning",
                    side_effect=AssertionError("CLIP conditioning used"),
                ),
            ):
                result, diagnostics, warning = inference_server.apply_adetailer(
                    source,
                    runtime,
                    "anima",
                    request,
                    "anima-detail",
                    Control(),
                    0,
                    schedule_latent_preview=Mock(side_effect=AssertionError("latent preview used")),
                    image_seed=100,
                )
            self.assertIsNone(warning)
            self.assertEqual(len(runtime.calls), 2)
            self.assertEqual([call["generators"][0].initial_seed() for call in runtime.calls], [101, 102])
            self.assertTrue(all(call["prompt"] == "detail parent prompt" for call in runtime.calls))
            self.assertTrue(all(call["negative_prompt"] == "detail negative" for call in runtime.calls))
            self.assertTrue(all(call["steps"] == 12 and call["denoise"] == 0.5 and call["cfg"] == 4 for call in runtime.calls))
            self.assertTrue(all(call["sampler"] == "euler" and call["scheduler"] == "simple" for call in runtime.calls))
            self.assertTrue(all(call["guidance"] == "pag" for call in runtime.calls))
            self.assertTrue(all(call["pag_scale"] == 0.7 for call in runtime.calls))
            self.assertTrue(all(call["pag_applied_layers"] == "all" for call in runtime.calls))
            for call in runtime.calls:
                crop = call["images"][0]
                mask = call["masks"][0]
                self.assertEqual(crop.mode, "RGB")
                self.assertEqual(mask.size, crop.size)
                self.assertLessEqual(max(crop.size), 1024)
                self.assertEqual(crop.width % 32, 0)
                self.assertEqual(crop.height % 32, 0)
                self.assertIsNotNone(mask.getbbox())
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
            self.assertNotEqual(result.getpixel((60, 60)), (0, 0, 0))
            # The stage records one block per pass; a single-unit run has one.
            self.assertEqual(len(diagnostics["units"]), 1)
            self.assertEqual(diagnostics["units"][0]["unit"], 1)
            self.assertEqual(diagnostics["detector"], "face.pt")
            self.assertTrue(all("crop_box" in item for item in diagnostics["units"][0]["detections"]))
            self.assertIn("sampling", inference_server.jobs["anima-detail"]["runtime_metrics"])
            self.assertIn(
                "adetailer.region_1.refinement.sampling",
                inference_server.jobs["anima-detail"]["runtime_metrics"],
            )
            self.assertIn(
                "adetailer.region_2.refinement.sampling",
                inference_server.jobs["anima-detail"]["runtime_metrics"],
            )
        finally:
            inference_server.jobs.pop("anima-detail", None)

    def test_official_yolo_catalog_contains_safe_unique_filenames(self):
        models = builtin_yolo_models()
        names = [model["name"] for model in models]
        self.assertEqual(names, [
            "face_yolov8n.pt",
            "face_yolov8s.pt",
            "hand_yolov8n.pt",
            "person_yolov8n-seg.pt",
            "person_yolov8s-seg.pt",
        ])
        self.assertEqual(len(names), len(set(names)))

    def test_generation_failure_log_excludes_prompt_text(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="private prompt text", negative_prompt="private negative text",
            width=512, height=512, steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
        )
        with tempfile.TemporaryDirectory() as temporary:
            original_directory = inference_server.LOG_DIRECTORY
            inference_server.LOG_DIRECTORY = Path(temporary)
            try:
                try:
                    raise RuntimeError("sample generation failure")
                except RuntimeError as error:
                    write_generation_failure_log("a" * 32, request, error)
                logs = list(Path(temporary).glob("*.log"))
                self.assertEqual(len(logs), 1)
                content = logs[0].read_text(encoding="utf-8")
                self.assertIn("sample generation failure", content)
                self.assertNotIn("private prompt text", content)
                self.assertNotIn("private negative text", content)
            finally:
                inference_server.LOG_DIRECTORY = original_directory

    def test_memory_failure_log_includes_performance_and_memory_diagnostics(self):
        request = GenerateInput(
            engine="iL", checkpoint="model.safetensors", prompt="private prompt text",
            width=1024, height=1024, steps=20, cfg=7, denoise=1, seed=1,
            sampler="euler", scheduler="simple",
        )
        with tempfile.TemporaryDirectory() as temporary:
            original_directory = inference_server.LOG_DIRECTORY
            original_settings = inference_server.performance_settings.copy()
            inference_server.LOG_DIRECTORY = Path(temporary)
            inference_server.performance_settings.update({
                "memory_mode": "low_vram",
                "vae_mode": "tiled",
                "staged_vae_decode": True,
            })
            try:
                try:
                    raise RuntimeError("CUDA out of memory while decoding VAE")
                except RuntimeError as error:
                    write_generation_failure_log("b" * 32, request, error)
                logs = list(Path(temporary).glob("*-generation-memory-failure-*.log"))
                self.assertEqual(len(logs), 1)
                content = logs[0].read_text(encoding="utf-8")
                self.assertIn('"out_of_memory": true', content)
                self.assertIn('"memory_mode": "low_vram"', content)
                self.assertIn('"vae_mode": "tiled"', content)
                self.assertIn('"staged_vae_decode": true', content)
                self.assertIn('"cuda_memory"', content)
                self.assertIn('"system_memory"', content)
                self.assertNotIn("private prompt text", content)
            finally:
                inference_server.performance_settings.clear()
                inference_server.performance_settings.update(original_settings)
                inference_server.LOG_DIRECTORY = original_directory

    def test_model_hash_is_written_to_png_metadata_when_available(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
        )
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            original_hash = inference_server.loaded_checkpoint_hash
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            inference_server.loaded_checkpoint_hash = "a" * 64
            try:
                output = inference_server.save_image(Image.new("RGB", (8, 8)), "job", request, 1.0, 1.0, 0.0)
                with Image.open(output) as saved:
                    self.assertEqual(saved.info["model_sha256"], "a" * 64)
                    self.assertIn('"checkpoint_sha256": "' + "a" * 64 + '"', saved.info["parameters"])
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output
                inference_server.loaded_checkpoint_hash = original_hash

    def test_save_image_removes_temporary_and_final_files_when_commit_is_cancelled(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
        )
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                def cancel():
                    raise inference_server.GenerationCancelled()

                with self.assertRaises(inference_server.GenerationCancelled):
                    inference_server.save_image(
                        Image.new("RGB", (8, 8)), "job", request, 1.0, 1.0, 0.0,
                        before_commit=cancel,
                    )
                self.assertEqual([path for path in Path(temporary).rglob("*") if path.is_file()], [])
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_transparent_save_history_and_copy_preserve_alpha(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="subject, ({Transparent background})",
            width=512, height=512, steps=20, cfg=7, denoise=1, seed=1,
            sampler="euler", scheduler="simple",
        )
        image = Image.new("RGBA", (8, 8), (80, 120, 180, 255))
        alpha = Image.new("L", (8, 8), 0)
        alpha.putpixel((4, 4), 255)
        image.putalpha(alpha)
        result = {"status": "complete", "method": "algorithm", "transparent_ratio": 0.98}
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                output = inference_server.save_image(
                    image, "job", request, 1.0, 1.0, 0.0,
                    background_removal_result=result,
                    conditioning_prompt="subject, isolated subject, simple plain background",
                )
                with Image.open(output) as saved:
                    self.assertEqual(saved.mode, "RGBA")
                    self.assertEqual(saved.getchannel("A").getextrema(), (0, 255))
                record = inference_server.history_file_record(output)
                self.assertTrue(record["transparent_background"])
                copied = inference_server.copy_history_asset({"asset_id": record["id"]})
                with Image.open(io.BytesIO(copied.body)) as copied_image:
                    self.assertEqual(copied_image.mode, "RGBA")
                    self.assertEqual(copied_image.getchannel("A").getextrema(), (0, 255))
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_history_folders_include_old_png_files_when_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            folder = Path(temporary) / "2026-07-22"
            folder.mkdir()
            image_path = folder / "old.png"
            Image.new("RGB", (8, 8)).save(image_path)
            old_time = inference_server.HISTORY_STARTED_AT - 60
            image_path.touch()
            os.utime(image_path, (old_time, old_time))
            try:
                folders = inference_server.list_history_folders()
                self.assertEqual(len(folders), 1)
                self.assertEqual(folders[0]["label"], "2026-07-22")
                self.assertEqual(inference_server.list_history_cards(), [])
                cards = inference_server.list_history_cards(folder, session_only=False)
                self.assertEqual(len(cards), 1)
                self.assertEqual(cards[0]["preview"]["name"], "old.png")
                self.assertEqual(inference_server.history_folder_path(folders[0]["id"]), folder.resolve())
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_history_directory_listing_supports_nested_mixed_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            original_preview = inference_server.PREVIEW_DIRECTORY
            root = Path(temporary)
            inference_server.OUTPUT_DIRECTORY = root
            inference_server.PREVIEW_DIRECTORY = root / ".previews"
            nested = root / "自定义" / "level-two"
            nested.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(root / "root.png")
            Image.new("RGB", (8, 8)).save(root / "自定义" / "mixed.png")
            Image.new("RGB", (8, 8)).save(nested / "deep.png")
            try:
                root_listing = inference_server.history_directory_listing()
                self.assertEqual(root_listing["image_count"], 1)
                self.assertEqual(root_listing["folders"][0]["name"], "自定义")
                self.assertEqual(root_listing["folders"][0]["count"], 1)
                self.assertEqual(root_listing["folders"][0]["folder_count"], 1)
                custom = inference_server.history_folder_path(root_listing["folders"][0]["id"])
                custom_listing = inference_server.history_directory_listing(custom)
                self.assertEqual(custom_listing["parent_id"], root_listing["id"])
                self.assertEqual(custom_listing["image_count"], 1)
                self.assertEqual(custom_listing["folders"][0]["name"], "level-two")
                cards = inference_server.list_history_cards(custom, session_only=False)
                self.assertEqual(cards[0]["preview"]["name"], "mixed.png")
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output
                inference_server.PREVIEW_DIRECTORY = original_preview

    def test_saved_manual_collage_layout_is_available_in_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            source_directory = Path(temporary) / "2026-07-24"
            source_directory.mkdir()
            first = source_directory / "first.png"
            second = source_directory / "second.png"
            Image.new("RGB", (8, 8), "red").save(first)
            Image.new("RGB", (8, 8), "blue").save(second)
            first_id = inference_server.history_asset_token(first)
            second_id = inference_server.history_asset_token(second)
            data = "data:image/png;base64," + base64.b64encode(first.read_bytes()).decode("ascii")
            layout = {"version": 1, "layers": [
                {"assetId": first_id, "url": f"/api/inference/history/assets/{first_id}", "name": "first.png", "x": 0, "y": 0, "scale": 1},
                {"assetId": second_id, "url": f"/api/inference/history/assets/{second_id}", "name": "second.png", "x": 4, "y": 2, "scale": .25},
            ]}
            try:
                payload = inference_server.CollageInput(image_data=data, name="manual.png", manual_layout=layout)
                saved = inference_server.save_collage(payload)
                record = inference_server.history_file_record(inference_server.history_asset_path(saved["id"]))
                self.assertEqual(record["manual_layout"], layout)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_animated_collage_preserves_gif_frames_and_saves_gif(self):
        frames = [Image.new("RGBA", (8, 8), "red"), Image.new("RGBA", (8, 8), "blue")]
        buffer = io.BytesIO()
        frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=[40, 80], loop=0, disposal=2)
        image_data = "data:image/gif;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        request = inference_server.AnimatedCollageInput(
            layers=[{"url": image_data, "x": 1, "y": 2, "width": 8, "height": 8}],
            width=12,
            height=12,
        )
        rendered, durations = inference_server.animated_collage_frames(request)
        self.assertEqual(len(rendered), 2)
        self.assertEqual(durations, [100, 100])
        self.assertEqual(rendered[0].getpixel((2, 3))[:3], (255, 0, 0))
        self.assertEqual(rendered[1].getpixel((2, 3))[:3], (0, 0, 255))
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            output = io.BytesIO()
            rendered[0].save(output, format="GIF", save_all=True, append_images=rendered[1:], duration=durations, loop=0, disposal=2)
            try:
                saved = inference_server.save_collage(inference_server.CollageInput(
                    image_data="data:image/gif;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
                    name="animated.gif",
                ))
                path = inference_server.history_asset_path(saved["id"])
                with Image.open(path) as image:
                    self.assertEqual(path.suffix, ".gif")
                    self.assertEqual(image.n_frames, 2)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output


class ADetailerStepAccountingTests(unittest.TestCase):
    """The job-wide step counter, which reported `STEP 43 / 33` until both halves were fixed."""

    def request(self, **overrides):
        fields = dict(
            engine="SD", checkpoint="model.safetensors", prompt="test", width=512, height=512,
            steps=30, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
        )
        fields.update(overrides)
        return GenerateInput(**fields)

    def anima_request(self, **overrides):
        return self.request(
            engine="Anima", checkpoint=None, diffusion_model="d.safetensors",
            text_encoder="t.safetensors", vae="v.safetensors", preview_enabled=False,
            sampler="euler", scheduler="simple", **overrides,
        )

    def test_a_pass_is_counted_the_way_the_family_actually_runs_it(self):
        unit = ADetailerUnitInput(detector="face_yolov8n.pt", denoise=0.1)
        # Diffusers inpainting runs int(steps * strength) updates.
        self.assertEqual(inference_server.adetailer_effective_steps(unit, self.request(), "sd"), 3)
        # Native Anima refinement keeps the last `steps + 1` sigmas of a longer schedule, so it runs
        # every requested step whatever the denoise. Multiplying here is what let the counter pass
        # its own total: the pass reported 3 and then delivered 30 callbacks.
        self.assertEqual(inference_server.adetailer_effective_steps(unit, self.anima_request(), "anima"), 30)

    def test_an_independent_step_count_replaces_the_inherited_one_in_both_families(self):
        unit = ADetailerUnitInput(detector="face_yolov8n.pt", denoise=0.5, use_steps=True, steps=40)
        self.assertEqual(inference_server.adetailer_effective_steps(unit, self.request(), "sd"), 20)
        self.assertEqual(inference_server.adetailer_effective_steps(unit, self.anima_request(), "anima"), 40)

    def test_the_counter_can_never_exceed_the_total_it_reports(self):
        # `apply_adetailer_unit` counts `base + region * effective + step` against a total of
        # `base + regions * effective`, so the invariant is exactly that the last step of the last
        # region lands on the total — for every family, and with the base pass present or absent.
        unit = ADetailerUnitInput(detector="face_yolov8n.pt", denoise=0.1)
        cases = [
            (self.request(), "sd"),
            (self.anima_request(), "anima"),
            (self.request(source_image=SOURCE, denoise=0.4), "sd"),
            (self.request(source_image=SOURCE, denoise=0.4, postprocess_only=True,
                          adetailer={"enabled": True, "units": [{"detector": "face_yolov8n.pt"}]}), "sd"),
        ]
        for request, family in cases:
            base = inference_server.base_sampling_steps(request, family)
            effective = inference_server.adetailer_effective_steps(unit, request, family)
            regions = 3
            total = base + regions * effective
            last = base + (regions - 1) * effective + effective
            self.assertEqual(last, total, f"{family} base={base} effective={effective}")
            self.assertGreaterEqual(effective, 1, family)

    def test_post_processing_counts_no_base_steps_because_none_were_run(self):
        # The offset is what the base pass performed, not what was configured. Counting 30 phantom
        # steps is how a post-processing run opened its ADetailer stage at step 30 of 33.
        request = self.request(
            source_image=SOURCE, denoise=0.4, postprocess_only=True,
            adetailer={"enabled": True, "units": [{"detector": "face_yolov8n.pt"}]},
        )
        self.assertEqual(inference_server.base_sampling_steps(request, "sd"), 0)
        self.assertEqual(inference_server.base_sampling_steps(self.request(), "sd"), 30)


if __name__ == "__main__":
    unittest.main()
