import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend import inference_server

from backend.performance_settings import (
    DEFAULT_PERFORMANCE_SETTINGS,
    file_sha256,
    memory_mode_for_family,
    enable_sequential_batch_forward,
    vae_decode_tile_count,
    normalize_performance_settings,
    nvfp4_capabilities,
    read_performance_settings,
    write_performance_settings,
)


class FakeCuda:
    def __init__(self, available=True, capability=(10, 0)):
        self.available = available
        self.capability = capability

    def is_available(self):
        return self.available

    def get_device_capability(self):
        return self.capability


class FakeTorch:
    float4_e2m1fn_x2 = object()

    def __init__(self, capability=(10, 0)):
        self.cuda = FakeCuda(capability=capability)


class PerformanceSettingsTests(unittest.TestCase):
    def test_invalid_values_fall_back_without_losing_valid_values(self):
        settings = normalize_performance_settings({
            "memory_mode": "high_vram",
            "attention_backend": "legacy-doggettx",
            "compute_dtype": "bf16",
            "keep_model_cached": False,
            "allow_shared_memory": False,
            "calculate_model_hash": True,
            "staged_vae_decode": True,
            "vram_limit_gb": 6.75,
        })
        self.assertEqual(settings["memory_mode"], "high_vram")
        self.assertEqual(settings["attention_backend"], "auto")
        self.assertEqual(settings["compute_dtype"], "bf16")
        self.assertFalse(settings["keep_model_cached"])
        self.assertFalse(settings["allow_shared_memory"])
        self.assertTrue(settings["calculate_model_hash"])
        self.assertTrue(settings["staged_vae_decode"])
        self.assertEqual(settings["vram_limit_gb"], 6.8)

    def test_vram_wall_defaults_to_auto_and_rejects_nonfinite_values(self):
        self.assertEqual(normalize_performance_settings({})["vram_limit_gb"], 0.0)
        self.assertEqual(normalize_performance_settings({"vram_limit_gb": "8"})["vram_limit_gb"], 0.0)
        self.assertEqual(normalize_performance_settings({"vram_limit_gb": float("nan")})["vram_limit_gb"], 0.0)

    def test_performance_payload_exposes_hardware_vram_wall_contract(self):
        properties = SimpleNamespace(total_memory=8 * 1024**3)
        original = inference_server.performance_settings.copy()
        try:
            inference_server.performance_settings.update({"vram_limit_gb": 0.0, "allow_shared_memory": True})
            with (
                patch.object(inference_server.torch.cuda, "is_available", return_value=True),
                patch.object(inference_server.torch.cuda, "get_device_capability", return_value=(8, 9)),
                patch.object(inference_server.torch.cuda, "get_device_properties", return_value=properties),
                patch.object(inference_server.torch.cuda, "get_device_name", return_value="RTX test"),
                patch.object(inference_server.torch.cuda, "is_bf16_supported", return_value=True),
            ):
                payload = inference_server.performance_payload()
        finally:
            inference_server.performance_settings.clear()
            inference_server.performance_settings.update(original)

        wall = payload["capabilities"]["vram_limit"]
        self.assertEqual(wall["minimum_bytes"], 2 * 1024**3)
        self.assertEqual(wall["maximum_bytes"], 8 * 1024**3 - 600 * 1024**2)
        self.assertEqual(wall["effective_bytes"], 8 * 1024**3)
        self.assertTrue(wall["automatic"])

    def test_ultra_low_mode_locks_peak_memory_options(self):
        settings = normalize_performance_settings({
            "memory_mode": "ultra_low_vram",
            "attention_backend": "xformers",
            "compute_dtype": "bf16",
            "vae_mode": "full",
            "keep_model_cached": True,
            "allow_shared_memory": True,
            "calculate_model_hash": True,
        })
        self.assertEqual(settings["attention_backend"], "sliced")
        self.assertEqual(settings["compute_dtype"], "fp16")
        self.assertEqual(settings["vae_mode"], "tiled")
        self.assertEqual(settings["cuda_math"], "strict")
        self.assertFalse(settings["keep_model_cached"])
        self.assertFalse(settings["allow_shared_memory"])
        self.assertFalse(settings["calculate_model_hash"])
        self.assertTrue(settings["staged_vae_decode"])

    def test_settings_round_trip_to_project_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "performance.json"
            written = write_performance_settings(path, {"vae_mode": "tiled", "cuda_math": "strict"})
            self.assertEqual(read_performance_settings(path), written)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["vae_mode"], "tiled")

    def test_missing_file_uses_current_safe_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(read_performance_settings(Path(temporary) / "missing.json"), DEFAULT_PERFORMANCE_SETTINGS)

    def test_model_hash_reads_file_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            path.write_bytes(b"xiriai model")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"xiriai model").hexdigest())

    def test_large_model_profile_only_forces_component_scheduling_for_sdxl(self):
        self.assertEqual(memory_mode_for_family("sdxl_balanced", "sdxl"), "normal_vram")
        self.assertEqual(memory_mode_for_family("sdxl_balanced", "sd"), "auto")
        self.assertEqual(memory_mode_for_family("low_vram", "sdxl"), "low_vram")

    def test_sequential_batch_forward_splits_cfg_batch_and_reassembles_output(self):
        class Module:
            def __init__(self):
                self.batch_sizes = []

            def forward(self, sample, encoder_hidden_states=None, return_dict=False):
                self.batch_sizes.append(sample.shape[0])
                return (sample + encoder_hidden_states[:, :1, :1].reshape(-1, 1, 1, 1),)

        module = Module()
        enable_sequential_batch_forward(module)
        sample = torch.zeros((2, 1, 2, 2))
        conditioning = torch.tensor([[[1.0]], [[2.0]]])
        output = module.forward(sample, encoder_hidden_states=conditioning, return_dict=False)[0]
        self.assertEqual(module.batch_sizes, [1, 1])
        self.assertEqual(output[:, 0, 0, 0].tolist(), [1.0, 2.0])

    def test_sequential_batch_forward_can_be_reapplied_after_hook_rebuild(self):
        class Module:
            def forward(self, sample, return_dict=False):
                return (sample + 1,)

        module = Module()
        enable_sequential_batch_forward(module)
        first_wrapper = module.forward
        module.forward = module._xiriai_full_batch_forward
        enable_sequential_batch_forward(module)
        self.assertIsNot(module.forward, first_wrapper)
        self.assertEqual(module.forward(torch.zeros((2, 1)), return_dict=False)[0].shape[0], 2)

    def test_vae_tile_count_matches_overlapping_decode_grid(self):
        self.assertEqual(vae_decode_tile_count(128, 128, 64, 0.25), 9)
        self.assertEqual(vae_decode_tile_count(64, 64, 64, 0.25), 4)

    def test_staged_decode_parks_sampler_before_loading_vae_on_gpu(self):
        events = []

        class Component(torch.nn.Module):
            def to(self, device, **_kwargs):
                events.append(("unet", str(device)))
                return self

        class Latents:
            shape = (1, 4, 32, 32)
            dtype = torch.float16
            device = torch.device("cpu")

            def to(self, device=None, dtype=None):
                events.append(("latents", str(device or dtype)))
                return self

            def __truediv__(self, _value):
                return self

        class Decoder:
            def forward(self, value):
                events.append(("decode", "run"))
                return value

        class VAE:
            dtype = torch.float16
            config = SimpleNamespace(force_upcast=False, scaling_factor=1.0)
            tile_latent_min_size = 64
            tile_overlap_factor = 0.25

            def __init__(self):
                self.decoder = Decoder()

            def to(self, device=None, dtype=None):
                if dtype is not None:
                    self.dtype = dtype
                events.append(("vae", str(device or dtype)))
                return self

            def enable_tiling(self):
                pass

            def disable_tiling(self):
                pass

            def decode(self, value, return_dict=False):
                return (self.decoder.forward(value),)

        class Pipeline:
            def __init__(self):
                self.unet = Component()
                self.vae = VAE()
                self.components = {"unet": self.unet, "vae": self.vae}
                self.image_processor = SimpleNamespace(postprocess=lambda _value, output_type: [output_type])

            def remove_all_hooks(self):
                events.append(("hooks", "removed"))

        class Control:
            def checkpoint(self, _job_id, stage):
                events.append(("checkpoint", stage))

        previous_parked = inference_server.pipeline_cpu_parked
        try:
            result = inference_server.decode_staged_latents(Pipeline(), Latents(), "sd", "missing-job", Control(), False)
        finally:
            inference_server.pipeline_cpu_parked = previous_parked

        self.assertEqual(result, ["pil"])
        self.assertLess(events.index(("unet", "cpu")), events.index(("vae", "cuda")))
        self.assertLess(events.index(("vae", "cuda")), events.index(("decode", "run")))
        self.assertEqual([event for event in events if event[0] == "vae"][-1], ("vae", "cpu"))

    def test_the_optional_accelerators_are_off_by_default_and_survive_a_round_trip(self):
        self.assertEqual(DEFAULT_PERFORMANCE_SETTINGS["compile_transformer"], False)
        self.assertIn("sage", normalize_performance_settings({"attention_backend": "sage"})["attention_backend"])
        self.assertTrue(normalize_performance_settings({"compile_transformer": True})["compile_transformer"])
        # A non-boolean must not be coerced into "on"; it falls back to the default.
        self.assertFalse(normalize_performance_settings({"compile_transformer": "yes"})["compile_transformer"])
        self.assertEqual(normalize_performance_settings({"attention_backend": "sagey"})["attention_backend"], "auto")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.json"
            written = write_performance_settings(path, {"attention_backend": "sage", "compile_transformer": True})
            self.assertEqual(written["attention_backend"], "sage")
            self.assertTrue(read_performance_settings(path)["compile_transformer"])

    def test_the_settings_endpoint_accepts_every_setting_the_file_stores(self):
        # `PerformanceInput.model_dump()` is written straight to disk, so a setting the model does
        # not declare is a setting the UI can never turn on: it is dropped on the way in and
        # normalised back to its default on the way out. Both accelerators were exactly that.
        declared = set(inference_server.PerformanceInput.model_fields)
        self.assertEqual(set(DEFAULT_PERFORMANCE_SETTINGS) - declared, set())
        accepted = inference_server.PerformanceInput(attention_backend="sage", compile_transformer=True).model_dump()
        self.assertEqual(normalize_performance_settings(accepted)["attention_backend"], "sage")
        self.assertTrue(normalize_performance_settings(accepted)["compile_transformer"])

    def test_clearing_an_empty_pipeline_slot_costs_nothing(self):
        # A full collection plus `empty_cache` is real time on a process holding a torch allocator
        # pool, and `clear_pipeline` is reached with nothing loaded on several paths — model switching
        # most often. Paying for it there bought a round trip and freed nothing.
        original = inference_server.loaded_pipeline
        try:
            inference_server.loaded_pipeline = None
            with patch.object(inference_server.gc, "collect") as collect, \
                    patch.object(torch.cuda, "empty_cache") as empty_cache:
                inference_server.clear_pipeline()
            collect.assert_not_called()
            empty_cache.assert_not_called()

            inference_server.loaded_pipeline = SimpleNamespace(
                unload_lora_weights=lambda: None, remove_all_hooks=lambda: None
            )
            with patch.object(inference_server.gc, "collect") as collect, \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "empty_cache") as empty_cache:
                inference_server.clear_pipeline()
            collect.assert_called_once()
            empty_cache.assert_called_once()
            self.assertIsNone(inference_server.loaded_pipeline)
        finally:
            inference_server.loaded_pipeline = original

    def test_ultra_low_memory_refuses_compilation_because_it_holds_workspace(self):
        forced = normalize_performance_settings({"memory_mode": "ultra_low_vram", "compile_transformer": True})
        self.assertFalse(forced["compile_transformer"])
        self.assertEqual(forced["attention_backend"], "sliced")

    def test_nvfp4_requires_blackwell_and_a_quantization_runtime(self):
        available = lambda name: name == "torchao"
        self.assertTrue(nvfp4_capabilities(FakeTorch((10, 0)), available)["runtime_ready"])
        self.assertFalse(nvfp4_capabilities(FakeTorch((8, 9)), available)["hardware_supported"])
        self.assertFalse(nvfp4_capabilities(FakeTorch((10, 0)), lambda _name: False)["runtime_ready"])


if __name__ == "__main__":
    unittest.main()
