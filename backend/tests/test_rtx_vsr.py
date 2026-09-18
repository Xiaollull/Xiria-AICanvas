import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from backend import inference_server, rtx_vsr
from backend.inference_server import GenerateInput


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


class RtxVsrTests(unittest.TestCase):
    def test_target_size_uses_ties_to_even_and_enforces_limits(self):
        self.assertEqual(rtx_vsr.target_size((10, 14), 2), (16, 32))
        self.assertEqual(rtx_vsr.target_size((512, 512), 1.04), (528, 528))
        self.assertEqual(rtx_vsr.target_size((512, 512), 1), (512, 512))
        self.assertEqual(rtx_vsr.target_size((4096, 2048), 2), (8192, 4096))
        with self.assertRaisesRegex(ValueError, "between"):
            rtx_vsr.target_size((512, 512), 0.99)
        with self.assertRaisesRegex(ValueError, "safe"):
            rtx_vsr.target_size((4100, 1024), 2)
        with self.assertRaisesRegex(ValueError, "safe"):
            rtx_vsr.target_size((3000, 3000), 2)

    def test_worker_result_requires_matching_protocol(self):
        self.assertEqual(rtx_vsr._parse_result('noise\n{"protocol":1,"available":true}\n')["available"], True)
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            rtx_vsr._parse_result('{"protocol":2,"available":true}')
        with self.assertRaisesRegex(RuntimeError, "invalid output"):
            rtx_vsr._parse_result("not json")

    def test_status_starts_one_nonblocking_probe(self):
        started = []

        class FakeThread:
            def __init__(self, target, **_kwargs):
                self.target = target
                self.started = False

            def start(self):
                self.started = True
                started.append(self)

            def is_alive(self):
                return self.started

        original_cache = rtx_vsr._status_cache
        original_thread = rtx_vsr._probe_thread
        rtx_vsr._status_cache = None
        rtx_vsr._probe_thread = None
        try:
            with patch.object(rtx_vsr.threading, "Thread", FakeThread):
                before = time.monotonic()
                first = rtx_vsr.status()
                second = rtx_vsr.status()
            self.assertLess(time.monotonic() - before, 0.1)
            self.assertTrue(first["probing"])
            self.assertTrue(second["probing"])
            self.assertEqual(len(started), 1)
        finally:
            rtx_vsr._status_cache = original_cache
            rtx_vsr._probe_thread = original_thread

    def test_unexpected_probe_error_clears_probing_state(self):
        original_cache = rtx_vsr._status_cache
        original_thread = rtx_vsr._probe_thread
        rtx_vsr._status_cache = {"probing": True}
        rtx_vsr._probe_thread = object()
        try:
            with patch.object(rtx_vsr, "_base_status", side_effect=MemoryError("probe allocation failed")):
                rtx_vsr._run_probe()
            self.assertIsNone(rtx_vsr._probe_thread)
            self.assertFalse(rtx_vsr._status_cache["probing"])
            self.assertIn("probe allocation failed", rtx_vsr._status_cache["reason"])
        finally:
            rtx_vsr._status_cache = original_cache
            rtx_vsr._probe_thread = original_thread

    def test_failed_pipeline_park_discards_the_partial_cache(self):
        class FailingComponent(inference_server.torch.nn.Module):
            def to(self, *_args, **_kwargs):
                raise RuntimeError("component transfer failed")

        pipeline = SimpleNamespace(
            components={"unet": FailingComponent()},
            vae=SimpleNamespace(to=lambda *_args, **_kwargs: None),
            remove_all_hooks=lambda: None,
        )
        with patch.object(inference_server, "loaded_pipeline", pipeline), \
                patch.object(inference_server, "clear_pipeline") as clear:
            with self.assertRaisesRegex(RuntimeError, "component transfer failed"):
                inference_server.park_pipeline_for_vae(pipeline)
        clear.assert_called_once()

    def test_failed_hook_removal_discards_the_partial_cache(self):
        pipeline = SimpleNamespace(
            remove_all_hooks=lambda: (_ for _ in ()).throw(RuntimeError("hook removal failed")),
            components={},
            vae=SimpleNamespace(to=lambda *_args, **_kwargs: None),
        )
        with patch.object(inference_server, "loaded_pipeline", pipeline), \
                patch.object(inference_server, "clear_pipeline") as clear:
            with self.assertRaisesRegex(RuntimeError, "hook removal failed"):
                inference_server.park_pipeline_for_vae(pipeline)
        clear.assert_called_once()

    def test_anima_external_stage_parks_runtime_without_generic_park_state(self):
        runtime = SimpleNamespace(to_cpu=Mock())
        with (
            patch.object(inference_server, "pipeline_cpu_parked", False),
            patch.object(inference_server.torch.cuda, "empty_cache") as empty_cache,
            patch.object(
                inference_server,
                "park_pipeline_for_vae",
                side_effect=AssertionError("generic Diffusers parking used"),
            ),
        ):
            inference_server.park_pipeline_for_external_stage(runtime, "anima")
            self.assertFalse(inference_server.pipeline_cpu_parked)
        runtime.to_cpu.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_failed_anima_external_stage_park_clears_authoritative_cache(self):
        runtime = SimpleNamespace(to_cpu=Mock(side_effect=RuntimeError("runtime transfer failed")))
        with (
            patch.object(inference_server, "loaded_pipeline", runtime),
            patch.object(inference_server, "clear_pipeline") as clear,
        ):
            with self.assertRaisesRegex(RuntimeError, "runtime transfer failed"):
                inference_server.park_pipeline_for_external_stage(runtime, "anima")
        clear.assert_called_once_with()

    def test_schema_accepts_any_chain_order_and_rejects_legacy_fields(self):
        resample = {"type": "resample", "enabled": True, "upscaler": "upscaler.pth"}
        adetailer = {"type": "adetailer", "enabled": True, "units": [{"detector": "face.pt"}]}
        rtx = {"type": "rtx", "enabled": True, "scale": 2}
        for chain in (
            [resample, adetailer, rtx], [resample, rtx, adetailer], [adetailer, resample, rtx],
            [adetailer, rtx, resample], [rtx, resample, adetailer], [rtx, adetailer, resample],
        ):
            request = generation_request(postprocess_chain=chain)
            self.assertEqual(inference_server.postprocess_stage_types(request), [entry["type"] for entry in chain])
        # Two of a single-instance stage, and the fields the old three-block model carried.
        with self.assertRaisesRegex(ValidationError, "at most one RTX"):
            generation_request(postprocess_chain=[rtx, rtx])
        with self.assertRaisesRegex(ValidationError, "at most one ADetailer"):
            generation_request(postprocess_chain=[adetailer, adetailer])
        for legacy in ({"hires": {"enabled": True}}, {"postprocess_order": ["hires", "adetailer", "rtx"]},
                       {"rtx": {"enabled": True}}, {"adetailer": {"enabled": True}}):
            with self.subTest(legacy=legacy), self.assertRaises(ValidationError):
                generation_request(**legacy)
        with self.assertRaises(ValidationError):
            generation_request(postprocess_chain=[{"type": "unknown"}])
        with self.assertRaises(ValidationError):
            generation_request(unknown_feature=True)

    def test_enabled_subsets_preserve_the_chain_order(self):
        request = generation_request(postprocess_chain=[
            {"type": "adetailer", "enabled": False},
            {"type": "rtx", "enabled": True, "scale": 2, "quality": "ultra"},
            {"type": "resample", "enabled": True, "upscaler": "upscaler.pth"},
        ])
        self.assertEqual(inference_server.postprocess_stage_types(request), ["rtx", "resample"])

    def test_memory_workload_tracks_rtx_before_a_resample_unit_only(self):
        unit = {"type": "resample", "enabled": True, "upscaler": "upscaler.pth", "scale": 2}
        rtx = {"type": "rtx", "enabled": True, "scale": 2}
        before = generation_request(postprocess_chain=[rtx, unit])
        after = generation_request(postprocess_chain=[unit, rtx])
        self.assertEqual(inference_server.generation_memory_workload(before, "sd")[:2], (2048, 2048))
        self.assertEqual(inference_server.generation_memory_workload(after, "sd")[:2], (1024, 1024))

    def test_schema_rejects_unsafe_intermediate_and_final_rtx_chains(self):
        unit = {"type": "resample", "enabled": True, "upscaler": "upscaler.pth", "scale": 2}
        rtx = {"type": "rtx", "enabled": True, "scale": 2}
        for chain in ([unit, rtx], [rtx, unit]):
            with self.subTest(chain=[entry["type"] for entry in chain]), self.assertRaises(ValidationError):
                generation_request(width=2048, height=2048, postprocess_chain=chain)

    def test_unavailable_rtx_job_is_rejected_before_enqueue(self):
        request = generation_request(postprocess_chain=[{"type": "rtx", "enabled": True, "scale": 2}])
        with patch.object(rtx_vsr, "status", return_value={"available": False, "probing": False, "reason": "runtime missing"}):
            with self.assertRaises(HTTPException) as raised:
                inference_server.create_job(request)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("runtime missing", raised.exception.detail)

    def test_optional_probe_does_not_block_a_non_rtx_job(self):
        request = generation_request()
        with inference_server.jobs_lock:
            inference_server.jobs.clear()
            inference_server.job_controls.clear()
        try:
            with patch.object(rtx_vsr, "status", side_effect=AssertionError("non-RTX jobs must not query RTX status")), \
                    patch.object(inference_server.executor, "submit") as submit:
                job = inference_server.create_job(request)
            self.assertEqual(job["status"], "queued")
            submit.assert_called_once()
        finally:
            with inference_server.jobs_lock:
                inference_server.jobs.clear()
                inference_server.job_controls.clear()

    def test_apply_rtx_parks_pipeline_and_records_diagnostics(self):
        request = generation_request(postprocess_chain=[{"type": "rtx", "enabled": True, "scale": 2, "quality": "ultra"}])
        image = Image.new("RGB", (64, 64), "red")

        class Control:
            cancelled = False

            def checkpoint(self, *_args):
                pass

            def raise_if_cancelled(self):
                pass

            def active_elapsed(self, _started_at):
                return 0

        output = Image.new("RGB", (128, 128), "blue")
        diagnostics = {"provider": "NVIDIA RTX VSR"}
        with patch.object(inference_server, "pipeline_cpu_parked", False), \
                patch.object(inference_server, "park_pipeline_for_external_stage") as park, \
                patch.object(rtx_vsr, "upscale_image", return_value=(output, diagnostics)) as upscale:
            result, result_diagnostics = inference_server.apply_rtx_vsr(image, object(), request, "missing-job", Control(), 0)
        park.assert_called_once()
        upscale.assert_called_once()
        self.assertEqual(result.size, (128, 128))
        self.assertEqual(result_diagnostics, diagnostics)

    def test_anima_rtx_uses_native_external_parking_without_diffusers_api(self):
        request = GenerateInput(
            engine="Anima",
            diffusion_model="diffusion.safetensors",
            text_encoder="text.safetensors",
            vae="vae.safetensors",
            prompt="test",
            width=512,
            height=512,
            steps=20,
            cfg=7,
            denoise=1,
            seed=1,
            sampler="euler",
            scheduler="simple",
            postprocess_chain=[{"type": "rtx", "enabled": True, "scale": 2}],
        )
        image = Image.new("RGB", (64, 64), "red")
        runtime = SimpleNamespace(to_cpu=Mock())

        class Control:
            def checkpoint(self, *_args):
                pass

            def raise_if_cancelled(self):
                pass

            def active_elapsed(self, _started_at):
                return 0

        with (
            patch.object(inference_server, "pipeline_cpu_parked", False),
            patch.object(inference_server, "park_pipeline_for_external_stage") as park,
            patch.object(
                inference_server,
                "derived_sampling_pipeline",
                side_effect=AssertionError("Diffusers pipeline API used"),
            ),
            patch.object(rtx_vsr, "upscale_image", return_value=(Image.new("RGB", (128, 128)), {"provider": "RTX"})),
        ):
            result, _diagnostics = inference_server.apply_rtx_vsr(
                image, runtime, request, "missing-job", Control(), 0
            )
        park.assert_called_once_with(runtime, "anima")
        self.assertEqual(result.size, (128, 128))

    def test_png_and_history_record_rtx_and_final_dimensions(self):
        request = generation_request(postprocess_chain=[
            {"type": "rtx", "enabled": True, "scale": 2, "quality": "ultra"},
            {"type": "resample", "enabled": True, "upscaler": "upscaler.pth", "scale": 1.5, "denoise": 0},
        ])
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            try:
                output = inference_server.save_image(
                    Image.new("RGB", (1024, 768)),
                    "job",
                    request,
                    1,
                    1,
                    0,
                    rtx_result={"provider": "NVIDIA RTX VSR", "scale": 2},
                )
                with Image.open(output) as saved:
                    parameters = json.loads(saved.info["parameters"])
                record = inference_server.history_file_record(output)
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output
        self.assertEqual([entry["type"] for entry in parameters["postprocess_chain"]], ["rtx", "resample"])
        self.assertEqual(parameters["rtx"]["scale"], 2)
        self.assertEqual(record["width"], 1024)
        self.assertEqual(record["height"], 768)
        self.assertEqual(record["rtx"]["scale"], 2)


if __name__ == "__main__":
    unittest.main()
