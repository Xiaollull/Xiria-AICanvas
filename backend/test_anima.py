import json
import copy
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from backend import inference_server
from backend.benchmark_lease import create_lease, validate_lease
from backend.anima_sampling import ANIMA_SAMPLERS, ANIMA_SCHEDULERS


def anima_request(**overrides):
    values = {
        "engine": "Anima",
        "diffusion_model": "anima/diffusion.safetensors",
        "text_encoder": "qwen/text-encoder.safetensors",
        "vae": "qwen/vae.safetensors",
        "prompt": "test subject",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 7,
        "denoise": 1,
        "seed": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "preview_enabled": False,
    }
    values.update(overrides)
    return inference_server.GenerateInput(**values)


class AnimaSchemaTests(unittest.TestCase):
    def test_engine_component_field_matrix_is_exact(self):
        request = anima_request()
        self.assertIsNone(request.checkpoint)
        self.assertEqual(request.diffusion_model, "anima/diffusion.safetensors")

        for missing in ("diffusion_model", "text_encoder", "vae"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValidationError, "requires"):
                anima_request(**{missing: None})
        with self.assertRaisesRegex(ValidationError, "forbids checkpoint"):
            anima_request(checkpoint="combined.safetensors")

        sd_values = {
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
        self.assertEqual(inference_server.GenerateInput(**sd_values).checkpoint, "model.safetensors")
        with self.assertRaisesRegex(ValidationError, "requires checkpoint"):
            inference_server.GenerateInput(**{**sd_values, "checkpoint": None})
        for split_field in ("diffusion_model", "text_encoder", "vae"):
            with self.subTest(split_field=split_field), self.assertRaisesRegex(ValidationError, "forbids"):
                inference_server.GenerateInput(**{**sd_values, split_field: "split.safetensors"})
        with self.assertRaises(ValidationError):
            anima_request(diffusion_model="x" * 501)

    def test_anima_capability_matrix_accepts_postprocessing_and_rejects_only_native_incompatibilities(self):
        rejected = (
            ({"sampler": "unknown"}, "sampler"),
            ({"scheduler": "unknown"}, "scheduler"),
            ({"preview_enabled": True}, "previews"),
        )
        for overrides, message in rejected:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValidationError, message):
                anima_request(**overrides)

        self.assertEqual(anima_request(sampler="dpmpp_2m", scheduler="karras").sampler, "dpmpp_2m")
        self.assertEqual(anima_request(guidance="cfg_zero_star").guidance, "cfg_zero_star")
        self.assertEqual(anima_request(guidance="pag").guidance, "pag")
        features = anima_request(
            loras=[{"path": "style.safetensors", "weight": 0.75}],
            hires={"enabled": True, "model": "upscaler.pth"},
            adetailer={"enabled": True, "units": [{"detector": "face.pt"}]},
            rtx={"enabled": True},
        )
        self.assertEqual(features.loras[0].weight, 0.75)
        self.assertTrue(features.hires.enabled)
        self.assertTrue(features.adetailer.enabled)
        self.assertTrue(features.rtx.enabled)
        transparent = anima_request(
            prompt="subject, ({Transparent background})",
            background_removal_model="model.onnx",
        )
        self.assertEqual(transparent.background_removal_model, "model.onnx")

    def test_anima_hires_rejects_native_refinement_edges_above_4096(self):
        self.assertEqual(
            anima_request(
                width=2048,
                height=1024,
                hires={"enabled": True, "model": "upscaler.pth", "scale": 2},
            ).hires.scale,
            2,
        )
        with self.assertRaisesRegex(ValidationError, "4096"):
            anima_request(
                width=2048,
                height=1024,
                hires={"enabled": True, "model": "upscaler.pth", "scale": 2.1},
            )
        with self.assertRaisesRegex(ValidationError, "4096"):
            anima_request(
                width=2048,
                height=1024,
                hires={"enabled": True, "model": "upscaler.pth", "scale": 2},
                rtx={"enabled": True, "scale": 2},
                postprocess_order=["rtx", "hires", "adetailer"],
            )

    def test_anima_runtime_performance_rejections_happen_before_resource_admission(self):
        request = anima_request()
        with patch.dict(inference_server.performance_settings, {"memory_mode": "ultra_low_vram"}):
            with patch.object(inference_server.rtx_vsr, "status") as rtx_status:
                with self.assertRaisesRegex(HTTPException, "ultra-low-memory"):
                    inference_server.create_job(request)
                rtx_status.assert_not_called()
        with patch.dict(inference_server.performance_settings, {"memory_mode": "auto", "staged_vae_decode": True}):
            with patch.object(inference_server.executor, "submit") as submit:
                job = inference_server.create_job(request)
            self.assertEqual(job["status"], "queued")
            submit.assert_called_once()
            inference_server.jobs.pop(job["id"], None)
            inference_server.job_controls.pop(job["id"], None)

    def test_anima_feature_request_keeps_generic_resource_admission(self):
        request = anima_request(
            hires={"enabled": True, "model": "upscaler.pth"},
            adetailer={"enabled": True, "units": [{"detector": "face.pt"}]},
            rtx={"enabled": True},
        )
        with (
            patch.dict(inference_server.performance_settings, {"memory_mode": "auto"}),
            patch.object(inference_server.rtx_vsr, "status", return_value={"available": True, "probing": False}),
            patch.object(inference_server, "adetailer_runtime_available", return_value=True),
            patch.object(inference_server, "resolve_detector_model", return_value=Path("face.pt")),
            patch.object(inference_server, "resolve_upscaler_model", return_value=(Path("upscaler.pth"), {})),
            patch.object(inference_server.executor, "submit") as submit,
        ):
            job = inference_server.create_job(request)
        try:
            self.assertEqual(job["status"], "queued")
            self.assertEqual(job["postprocess_stages"], ["hires", "adetailer", "rtx"])
            submit.assert_called_once()
        finally:
            inference_server.jobs.pop(job["id"], None)
            inference_server.job_controls.pop(job["id"], None)


class AnimaPathAndHealthTests(unittest.TestCase):
    def test_model_roots_are_explicit_for_every_engine(self):
        def resolved(_project_root, key, engine_key=None):
            suffix = f"-{engine_key}" if engine_key else ""
            return Path(f"C:/models/{key}{suffix}")

        with patch.object(inference_server, "resolve_model_directory", side_effect=resolved) as resolve:
            self.assertEqual(inference_server.model_roots("SD")[0], Path("C:/models/checkpoints-sd"))
            self.assertEqual(inference_server.model_roots("iL")[0], Path("C:/models/checkpoints-illustrious"))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                inference_server.model_roots("unknown")
            roots = inference_server.anima_model_roots()

        self.assertEqual(set(roots), {"diffusion_model", "text_encoder", "vae", "lora"})
        calls = [call.args[1:] for call in resolve.call_args_list]
        self.assertIn(("diffusion_models",), calls)
        self.assertIn(("text_encoders",), calls)
        self.assertIn(("vae",), calls)
        self.assertIn(("loras", "anima"), calls)

    def test_anima_assets_are_contained_safetensors_and_bundled_tokenizers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = {
                "diffusion_model": root / "diffusion",
                "text_encoder": root / "text",
                "vae": root / "vae",
                "lora": root / "loras",
            }
            for directory in roots.values():
                directory.mkdir()
            tokenizer_directory = root / "bundled"
            tokenizer_directory.mkdir()
            component = roots["diffusion_model"] / "nested" / "model.safetensors"
            component.parent.mkdir()
            component.write_bytes(b"weights")
            (roots["diffusion_model"] / "bad.ckpt").write_bytes(b"weights")
            lora = roots["lora"] / "nested" / "style.safetensors"
            lora.parent.mkdir()
            lora.write_bytes(b"lora")
            qwen = tokenizer_directory / "anima-qwen3-tokenizer.json"
            qwen_config = tokenizer_directory / "anima-qwen3-tokenizer-config.json"
            t5 = tokenizer_directory / "anima-t5-tokenizer.json"
            qwen.write_text("{}", encoding="utf-8")
            qwen_config.write_text("{}", encoding="utf-8")
            t5.write_text("{}", encoding="utf-8")

            self.assertEqual(
                inference_server.validate_anima_component(
                    "nested/model.safetensors", roots["diffusion_model"], "Anima diffusion"
                ),
                component,
            )
            with self.assertRaisesRegex(ValueError, "safetensors"):
                inference_server.validate_anima_component(
                    "bad.ckpt", roots["diffusion_model"], "Anima diffusion"
                )
            with self.assertRaises((ValueError, FileNotFoundError)):
                inference_server.validate_anima_component(
                    "../outside.safetensors", roots["diffusion_model"], "Anima diffusion"
                )
            self.assertEqual(inference_server.validate_anima_lora("nested/style.safetensors", roots["lora"]), lora)
            with self.assertRaisesRegex(ValueError, "safetensors"):
                bad_lora = roots["lora"] / "bad.pt"
                bad_lora.write_bytes(b"lora")
                inference_server.validate_anima_lora("bad.pt", roots["lora"])
            statuses = {
                "qwen": {"path": qwen, "installed": True, "reason": None},
                "qwen_config": {"path": qwen_config, "installed": True, "reason": None},
                "t5": {"path": t5, "installed": True, "reason": None},
            }
            with patch.object(inference_server, "anima_tokenizer_status", return_value=statuses):
                self.assertEqual(
                    inference_server.anima_tokenizer_sources(tokenizer_directory),
                    {"qwen": qwen, "qwen_config": qwen_config, "t5": t5},
                )

    def test_anima_tokenizer_integrity_rejects_placeholder_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for filename, _size, _sha256 in inference_server.ANIMA_TOKENIZER_ARTIFACTS.values():
                (root / filename).write_text("{}", encoding="utf-8")

            statuses = inference_server.anima_tokenizer_status(root)

            self.assertTrue(all(not status["installed"] for status in statuses.values()))
            with self.assertRaisesRegex(ValueError, "missing or corrupt"):
                inference_server.anima_tokenizer_sources(root)

    def test_bundled_anima_tokenizers_pass_exact_integrity(self):
        statuses = inference_server.anima_tokenizer_status(force_hash=True)
        self.assertTrue(all(status["installed"] for status in statuses.values()))
        self.assertEqual(
            {status["path"].parent for status in statuses.values()},
            {inference_server.ANIMA_TOKENIZER_DIRECTORY},
        )

    def test_anima_tokenizer_integrity_accepts_only_declared_line_ending_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            filename = "tokenizer.json"
            lf = b'{"line":"one"}\n'
            crlf = lf.replace(b"\n", b"\r\n")
            variants = {
                "qwen": (
                    (filename, len(lf), hashlib.sha256(lf).hexdigest()),
                    (filename, len(crlf), hashlib.sha256(crlf).hexdigest()),
                ),
            }
            artifacts = {"qwen": variants["qwen"][0]}
            with patch.object(inference_server, "ANIMA_TOKENIZER_ARTIFACTS", artifacts), patch.object(
                inference_server, "ANIMA_TOKENIZER_ARTIFACT_VARIANTS", variants
            ):
                (root / filename).write_bytes(lf)
                self.assertTrue(inference_server.anima_tokenizer_status(root, force_hash=True)["qwen"]["installed"])
                (root / filename).write_bytes(crlf)
                self.assertTrue(inference_server.anima_tokenizer_status(root, force_hash=True)["qwen"]["installed"])
                (root / filename).write_bytes(b'{"line":"two"}\n')
                status = inference_server.anima_tokenizer_status(root, force_hash=True)["qwen"]
                self.assertFalse(status["installed"])
                self.assertIn("SHA-256", status["reason"])

    def test_health_exposes_anima_and_guidance_capabilities(self):
        anima = inference_server.anima_health_fields()
        self.assertEqual(anima["samplers"], list(ANIMA_SAMPLERS))
        self.assertEqual(anima["schedulers"], list(ANIMA_SCHEDULERS))
        self.assertIn("runtime_ready", anima)
        self.assertIn("tokenizers_ready", anima)
        self.assertIn("bf16_ready", anima)
        self.assertEqual(
            anima["features"],
            {
                "pag": True,
                "cfg_zero_star": True,
                "hires": True,
                "adetailer": True,
                "rtx": True,
                "process_preview": False,
                "staged_vae_decode": False,
                "lora": True,
                "transparent_background": True,
            },
        )
        self.assertTrue(anima["features"]["hires"])
        self.assertIn("loras", anima["required_assets"])
        for path in anima["required_assets"].values():
            self.assertFalse(Path(path).is_absolute())

        with patch.object(inference_server, "anima_health_fields", return_value=anima):
            health = inference_server.health()
        self.assertEqual(health["protocol"], inference_server.INFERENCE_PROTOCOL)
        self.assertEqual(health["protocol"], 34)
        self.assertEqual(health["engines"]["Anima"], anima)
        self.assertEqual(health["guidance"]["pag"]["engines"], ["SD", "iL", "Anima"])
        self.assertEqual(
            health["guidance"]["pag"]["implementations"]["Anima"],
            "native_cosmos_identity_self_attention",
        )
        self.assertTrue(health["guidance"]["cfg_zero_star"]["available"])
        # Krea 2 joined Anima once its runtime landed: it is the second engine with a real
        # unconditional branch to rescale against.
        self.assertEqual(health["guidance"]["cfg_zero_star"]["engines"], ["Anima", "Krea2"])
        # Neither Flux generation ships CFG-Zero* and neither can ever gain it: guidance
        # distillation leaves nothing to rescale against, so nothing is merely planned.
        self.assertEqual(health["guidance"]["cfg_zero_star"]["planned_engines"], [])
        self.assertFalse(health["engines"]["Flux"]["features"]["cfg_zero_star"])
        self.assertIsNone(health["guidance"]["cfg_zero_star"]["reason"])


class AnimaCacheAndMetadataTests(unittest.TestCase):
    def test_anima_balanced_math_keeps_tf32_but_disables_restart_variant_cudnn_benchmark(self):
        original_setting = inference_server.performance_settings["cuda_math"]
        original_matmul_tf32 = inference_server.torch.backends.cuda.matmul.allow_tf32
        original_cudnn_tf32 = inference_server.torch.backends.cudnn.allow_tf32
        original_benchmark = inference_server.torch.backends.cudnn.benchmark
        try:
            inference_server.performance_settings["cuda_math"] = "balanced"
            inference_server.configure_anima_cuda_math()
            self.assertTrue(inference_server.torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(inference_server.torch.backends.cudnn.allow_tf32)
            self.assertFalse(inference_server.torch.backends.cudnn.benchmark)
        finally:
            inference_server.performance_settings["cuda_math"] = original_setting
            inference_server.torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            inference_server.torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            inference_server.torch.backends.cudnn.benchmark = original_benchmark

    def test_model_cache_uses_anima_diffusion_identity_to_release_composite(self):
        runtime = SimpleNamespace(close=Mock())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "diffusion"
            root.mkdir()
            original = (
                inference_server.loaded_pipeline,
                inference_server.loaded_checkpoint,
                inference_server.loaded_family,
                inference_server.loaded_engine,
                inference_server.loaded_model_assets,
                inference_server.loaded_model_revisions,
            )
            inference_server.loaded_pipeline = runtime
            inference_server.loaded_checkpoint = str(root / "anima.safetensors")
            inference_server.loaded_family = "anima"
            inference_server.loaded_engine = "Anima"
            inference_server.loaded_model_assets = {
                "diffusion_model": str(root / "anima.safetensors"),
                "text_encoder": str(root.parent / "text" / "qwen.safetensors"),
                "vae": str(root.parent / "vae" / "qwen.safetensors"),
            }
            inference_server.loaded_model_revisions = {}
            try:
                with patch.object(
                    inference_server,
                    "anima_model_roots",
                    return_value={
                        "diffusion_model": root,
                        "text_encoder": root.parent / "text",
                        "vae": root.parent / "vae",
                        "configs": root.parent / "configs",
                    },
                ):
                    retained = inference_server.unload_model_cache("Anima", "other.safetensors")
                    released = inference_server.unload_model_cache("Anima", "anima.safetensors")
                self.assertEqual(retained["status"], "retained")
                self.assertEqual(retained["loaded_checkpoint"], "anima.safetensors")
                self.assertEqual(released, {"status": "released", "model_cached": False})
                runtime.close.assert_called_once_with()
            finally:
                (
                    inference_server.loaded_pipeline,
                    inference_server.loaded_checkpoint,
                    inference_server.loaded_family,
                    inference_server.loaded_engine,
                    inference_server.loaded_model_assets,
                    inference_server.loaded_model_revisions,
                ) = original

    def test_model_cache_can_clear_the_authoritative_current_runtime_without_identity(self):
        runtime = SimpleNamespace(close=Mock())
        original = (
            inference_server.loaded_pipeline,
            inference_server.loaded_checkpoint,
            inference_server.loaded_family,
            inference_server.loaded_engine,
        )
        inference_server.loaded_pipeline = runtime
        inference_server.loaded_checkpoint = "C:/configured/diffusion/anima.safetensors"
        inference_server.loaded_family = "anima"
        inference_server.loaded_engine = "Anima"
        try:
            self.assertEqual(
                inference_server.unload_model_cache(),
                {"status": "released", "model_cached": False},
            )
            runtime.close.assert_called_once_with()
            self.assertIsNone(inference_server.loaded_pipeline)
        finally:
            inference_server.clear_pipeline()
            (
                inference_server.loaded_pipeline,
                inference_server.loaded_checkpoint,
                inference_server.loaded_family,
                inference_server.loaded_engine,
            ) = original

    def test_model_cache_rechecks_active_jobs_after_waiting_for_pipeline_lock(self):
        runtime = SimpleNamespace(close=Mock())
        original = (
            inference_server.loaded_pipeline,
            inference_server.loaded_checkpoint,
            inference_server.loaded_family,
            inference_server.loaded_engine,
        )
        inference_server.loaded_pipeline = runtime
        inference_server.loaded_checkpoint = "C:/configured/diffusion/anima.safetensors"
        inference_server.loaded_family = "anima"
        inference_server.loaded_engine = "Anima"
        started = threading.Event()
        errors = []

        def unload():
            started.set()
            try:
                inference_server.unload_model_cache()
            except Exception as error:
                errors.append(error)

        inference_server.pipeline_lock.acquire()
        thread = threading.Thread(target=unload)
        thread.start()
        try:
            self.assertTrue(started.wait(1))
            with inference_server.jobs_lock:
                inference_server.jobs["cache-race"] = {"status": "queued"}
        finally:
            inference_server.pipeline_lock.release()
        thread.join(2)
        try:
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], HTTPException)
            self.assertEqual(errors[0].status_code, 409)
            runtime.close.assert_not_called()
        finally:
            inference_server.jobs.pop("cache-race", None)
            inference_server.clear_pipeline()
            (
                inference_server.loaded_pipeline,
                inference_server.loaded_checkpoint,
                inference_server.loaded_family,
                inference_server.loaded_engine,
            ) = original

    def test_anima_restore_only_returns_runtime_to_cpu(self):
        runtime = SimpleNamespace(to_cpu=Mock())
        original = (
            inference_server.loaded_pipeline,
            inference_server.loaded_family,
            inference_server.active_memory_strategy,
        )
        inference_server.loaded_pipeline = runtime
        inference_server.loaded_family = "anima"
        inference_server.active_memory_strategy = {"mode": "normal_vram"}
        try:
            with patch.object(
                inference_server,
                "reconfigure_memory_strategy",
                side_effect=AssertionError("SD pipeline recovery hook used"),
            ):
                self.assertTrue(inference_server.restore_cached_pipeline_state())
            runtime.to_cpu.assert_called_once_with()
        finally:
            (
                inference_server.loaded_pipeline,
                inference_server.loaded_family,
                inference_server.active_memory_strategy,
            ) = original

    def test_metadata_uses_relative_composite_assets_and_diffusion_hash(self):
        request = anima_request(guidance="cfg_zero_star")
        with tempfile.TemporaryDirectory() as temporary:
            original_output = inference_server.OUTPUT_DIRECTORY
            original_hash = inference_server.loaded_checkpoint_hash
            inference_server.OUTPUT_DIRECTORY = Path(temporary)
            inference_server.loaded_checkpoint_hash = "a" * 64
            try:
                output = inference_server.save_image(
                    Image.new("RGB", (8, 8)), "job", request, 1.0, 1.0, 0.0
                )
                with Image.open(output) as saved:
                    parameters = json.loads(saved.info["parameters"])
                self.assertIsNone(parameters["checkpoint"])
                self.assertEqual(parameters["diffusion_model"], request.diffusion_model)
                self.assertEqual(parameters["text_encoder"], request.text_encoder)
                self.assertEqual(parameters["vae"], request.vae)
                self.assertEqual(parameters["flow_shift"], 3)
                self.assertEqual(parameters["checkpoint_sha256"], "a" * 64)
                self.assertEqual(parameters["diffusion_model_sha256"], "a" * 64)
                self.assertEqual(parameters["sampler"], "euler")
                self.assertEqual(parameters["scheduler"], "simple")
                self.assertEqual(parameters["guidance"]["type"], "cfg_zero_star")
            finally:
                inference_server.OUTPUT_DIRECTORY = original_output
                inference_server.loaded_checkpoint_hash = original_hash

    def test_composite_cache_reuses_only_all_three_exact_paths(self):
        class Runtime:
            def __init__(self):
                self.weight_sizes = {
                    "transformer": 40,
                    "text_encoder": 20,
                    "llm_adapter": 10,
                    "vae": 30,
                    "total": 100,
                }
                self.vae = SimpleNamespace(enable_tiling=Mock(), disable_tiling=Mock())
                self.to_cpu = Mock()
                self.close = Mock()

        strategy = {
            "mode": "normal_vram",
            "requested_mode": "auto",
            "label": "NORMAL_VRAM",
            "reason": "test",
            "total_gb": 8.0,
            "free_gb": 6.0,
            "weight_gb": 4.0,
            "inference_gb": 1.0,
            "reserved_gb": 1.0,
            "base_weight_bytes": 100,
            "base_largest_component_bytes": 40,
            "offload_mode": "model",
            "model_resident": False,
            "normal_available": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = []
            for name in ("diffusion.safetensors", "text.safetensors", "vae.safetensors", "vae-2.safetensors"):
                path = root / name
                path.write_bytes(b"weights")
                paths.append(path)
            tokenizer_sources = {"qwen": root / "qwen.json", "t5": root / "t5.json"}
            for path in tokenizer_sources.values():
                path.write_text("{}", encoding="utf-8")
            runtimes = [Runtime(), Runtime()]
            original_settings = inference_server.performance_settings.copy()
            inference_server.clear_pipeline()
            inference_server.performance_settings.update({"vae_mode": "full", "calculate_model_hash": False})
            try:
                with (
                    patch.object(inference_server.torch.cuda, "is_available", return_value=True),
                    patch.object(inference_server.torch.cuda, "is_bf16_supported", return_value=True),
                    patch.object(inference_server.torch.cuda, "empty_cache"),
                    patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()),
                    patch.object(
                        inference_server,
                        "anima_runtime_tokenizer_paths",
                        return_value=tokenizer_sources,
                    ),
                    patch.object(inference_server, "load_anima_runtime", side_effect=runtimes) as load_runtime,
                ):
                    first = inference_server.load_anima_pipeline(
                        paths[0], paths[1], paths[2], tokenizer_sources, 512, 512, 7, "job"
                    )
                    reused = inference_server.load_anima_pipeline(
                        paths[0], paths[1], paths[2], tokenizer_sources, 512, 512, 7, "job"
                    )
                    replaced = inference_server.load_anima_pipeline(
                        paths[0], paths[1], paths[3], tokenizer_sources, 512, 512, 7, "job"
                    )
                self.assertIs(first, runtimes[0])
                self.assertIs(reused, first)
                self.assertIs(replaced, runtimes[1])
                self.assertEqual(load_runtime.call_count, 2)
                # Comfy-style default: normal-VRAM returns components to RAM after the job.
                runtimes[0].to_cpu.assert_called_once_with()
                runtimes[0].close.assert_called_once()
                self.assertEqual(
                    inference_server.loaded_model_assets,
                    {
                        "diffusion_model": str(paths[0]),
                        "text_encoder": str(paths[1]),
                        "vae": str(paths[3]),
                        "loras": [],
                    },
                )
                self.assertEqual(inference_server.loaded_checkpoint, str(paths[0]))
            finally:
                inference_server.performance_settings.clear()
                inference_server.performance_settings.update(original_settings)
                inference_server.clear_pipeline()

    def test_composite_cache_identity_includes_ordered_lora_revision_and_multiplier(self):
        class Runtime:
            def __init__(self):
                self.weight_sizes = {
                    "transformer": 40,
                    "text_encoder": 20,
                    "llm_adapter": 10,
                    "vae": 30,
                    "total": 100,
                }
                self.vae = SimpleNamespace(enable_tiling=Mock(), disable_tiling=Mock())
                self.to_cpu = Mock()
                self.close = Mock()

        strategy = {
            "mode": "normal_vram",
            "requested_mode": "auto",
            "label": "NORMAL_VRAM",
            "reason": "test",
            "total_gb": 8.0,
            "free_gb": 6.0,
            "weight_gb": 4.0,
            "inference_gb": 1.0,
            "reserved_gb": 1.0,
            "base_weight_bytes": 100,
            "base_largest_component_bytes": 40,
            "offload_mode": "model",
            "model_resident": False,
            "normal_available": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assets = []
            for name in ("diffusion.safetensors", "text.safetensors", "vae.safetensors"):
                path = root / name
                path.write_bytes(b"weights")
                assets.append(path)
            first_lora = root / "first.safetensors"
            second_lora = root / "second.safetensors"
            first_lora.write_bytes(b"first")
            second_lora.write_bytes(b"second")
            tokenizer_sources = {"qwen": root / "qwen.json", "t5": root / "t5.json"}
            for path in tokenizer_sources.values():
                path.write_text("{}", encoding="utf-8")
            runtimes = [Runtime() for _ in range(6)]
            original_settings = inference_server.performance_settings.copy()
            inference_server.clear_pipeline()
            inference_server.performance_settings.update({"vae_mode": "full", "calculate_model_hash": False})
            try:
                with (
                    patch.object(inference_server.torch.cuda, "is_available", return_value=True),
                    patch.object(inference_server.torch.cuda, "is_bf16_supported", return_value=True),
                    patch.object(inference_server.torch.cuda, "empty_cache"),
                    patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()) as choose,
                    patch.object(inference_server, "anima_runtime_tokenizer_paths", return_value=tokenizer_sources),
                    patch.object(inference_server, "load_anima_runtime", side_effect=runtimes) as load_runtime,
                ):
                    ordered = [(first_lora, 0.5), (second_lora, 1.0)]
                    first = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job", loras=ordered
                    )
                    reused = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job", loras=ordered
                    )
                    zero_ignored = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job",
                        loras=[*ordered, (first_lora, 0.0)],
                    )
                    multiplier = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job",
                        loras=[(first_lora, 0.75), (second_lora, 1.0)],
                    )
                    reordered = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job",
                        loras=[(second_lora, 1.0), (first_lora, 0.75)],
                    )
                    first_lora.write_bytes(b"first-revised")
                    revised = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job",
                        loras=[(second_lora, 1.0), (first_lora, 0.75)],
                    )
                    empty = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job", loras=[]
                    )
                    fused_again = inference_server.load_anima_pipeline(
                        *assets, tokenizer_sources, 512, 512, 7, "job", loras=[(first_lora, 0.75)]
                    )
                self.assertIs(first, reused)
                self.assertIs(first, zero_ignored)
                self.assertEqual(
                    [multiplier, reordered, revised, empty, fused_again], runtimes[1:]
                )
                self.assertEqual(load_runtime.call_count, 6)
                self.assertEqual(load_runtime.call_args.kwargs["loras"], [(first_lora, 0.75)])
                self.assertEqual(inference_server.loaded_model_revisions["loras"][0][0], str(first_lora))
                self.assertTrue(all(call.args[5] == 0 for call in choose.call_args_list))
                self.assertEqual(inference_server.active_memory_strategy["adapter_source_bytes"], first_lora.stat().st_size)
            finally:
                inference_server.performance_settings.clear()
                inference_server.performance_settings.update(original_settings)
                inference_server.clear_pipeline()

    def test_low_vram_strategy_serializes_anima_sampling_batch(self):
        strategy = {
            "mode": "low_vram",
            "normal_available": True,
            "weight_gb": 5.0,
            "offload_mode": "sequential",
            "model_resident": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = {}
            for name in ("diffusion_model", "text_encoder", "vae"):
                path = Path(temporary) / f"{name}.safetensors"
                path.write_bytes(b"weights")
                paths[name] = path
            with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()):
                selected = inference_server.anima_memory_strategy(
                    paths["diffusion_model"], paths, 512, 512, 1.0, 4, "none"
                )
        self.assertEqual(selected["sampling_batch_size"], 1)
        self.assertEqual(selected["offload_mode"], "staged_transformer_group_offload_microbatch")
        self.assertTrue(selected["transformer_group_offload"])
        self.assertIn("单图串行", selected["reason"])

    def test_anima_strategy_rechecks_a_single_image_before_rejecting_a_large_batch(self):
        unsafe_batch = {
            "mode": "low_vram",
            "normal_available": False,
            "weight_gb": 5.0,
            "offload_mode": "sequential",
            "model_resident": False,
        }
        safe_single = {
            "mode": "normal_vram",
            "normal_available": True,
            "weight_gb": 5.0,
            "offload_mode": "model",
            "model_resident": False,
        }
        checkpoint = Path("C:/models/anima.safetensors")
        with patch.object(
            inference_server,
            "choose_memory_strategy",
            side_effect=[unsafe_batch, safe_single],
        ) as choose:
            selected = inference_server.choose_anima_memory_strategy(
                checkpoint, 1024, 1024, 1.0, 4, "none", 100, 80
            )
        self.assertEqual([call.args[6] for call in choose.call_args_list], [4, 1])
        self.assertEqual(selected["sampling_batch_size"], 1)
        self.assertEqual(selected["mode"], "low_vram")

    def test_anima_guidance_memory_budget_uses_one_physical_sequential_forward(self):
        strategy = {
            "mode": "normal_vram",
            "normal_available": True,
            "weight_gb": 5.0,
            "offload_mode": "model",
            "model_resident": False,
        }
        with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()) as choose:
            selected = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 7.0, 1, "pag", 100, 80
            )
        self.assertEqual([call.args[7] for call in choose.call_args_list], [1])

    def test_anima_admission_separates_logical_and_physical_guidance_copies(self):
        strategy = {
            "mode": "normal_vram", "normal_available": True, "weight_gb": 5.0,
            "offload_mode": "model", "model_resident": False,
            "free_bytes": 7_700 * 1024**2, "inference_bytes": 1_300 * 1024**2,
            "normal_required_bytes": 7_450 * 1024**2, "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()) as choose:
            selected = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1088, 1472, 5.0, 1, "none", 5 * 1024**3, 4 * 1024**3
            )
        self.assertEqual([call.args[7] for call in choose.call_args_list], [1, 2])
        self.assertEqual((selected["logical_guidance_copies"], selected["physical_forward_copies"]), (2, 1))
        self.assertEqual(selected["actual_physical_forward_copies"], 1)
        self.assertEqual(selected["admission"]["serial_resident_sampling_required_bytes"], strategy["normal_required_bytes"])
        self.assertFalse(selected["cfg_batch"])
        self.assertEqual(selected["offload_mode"], "staged_transformer_resident")
        self.assertIsNone(selected["admission"]["ambient_free_bytes"])
        json.dumps(selected["admission"])

    def test_anima_pag_is_logically_three_but_physically_one_and_batch_cfg_needs_headroom(self):
        strategy = {
            "mode": "normal_vram", "normal_available": True, "weight_gb": 5.0,
            "offload_mode": "model", "model_resident": False,
            "free_bytes": 32 * 1024**3, "inference_bytes": 1024**3,
            "normal_required_bytes": 6 * 1024**3, "reserved_bytes": 600 * 1024**2,
        }
        def choose_by_physical(*args, **_kwargs):
            physical_copies = args[7]
            return {
                **strategy,
                "inference_bytes": physical_copies * 1024**3,
                "normal_required_bytes": (5 + physical_copies) * 1024**3,
            }
        with patch.object(inference_server, "choose_memory_strategy", side_effect=choose_by_physical):
            pag = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 5.0, 1, "pag", 6 * 1024**3, 4 * 1024**3
            )
            pag_admission = dict(pag["admission"])
            pag_cfg_batch = pag["cfg_batch"]
            batched = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3
            )
        self.assertEqual((pag_admission["logical_guidance_copies"], pag_admission["physical_forward_copies"]), (3, 1))
        self.assertFalse(pag_cfg_batch)
        self.assertFalse(batched["cfg_batch"])
        self.assertGreater(batched["cfg_batch_required_bytes"], batched["admission"]["serial_resident_sampling_required_bytes"])
        self.assertEqual((batched["logical_guidance_copies"], batched["actual_physical_forward_copies"]), (2, 1))
        self.assertEqual(batched["physical_inference_bytes"], batched["admission"]["serial_resident_inference_bytes"])

    def test_anima_pag_scale_zero_excludes_the_skipped_perturbed_branch_from_logical_work(self):
        strategy = {
            "mode": "normal_vram", "normal_available": True, "weight_gb": 5.0,
            "offload_mode": "model", "model_resident": False, "free_bytes": 16 * 1024**3,
            "inference_bytes": 1024**3, "normal_required_bytes": 6 * 1024**3, "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", side_effect=lambda *_args, **_kwargs: strategy.copy()):
            cfg = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 5.0, 1, "pag", 6 * 1024**3, 4 * 1024**3,
                pag_scale=0.0,
            )
            cfg_copies = (cfg["logical_guidance_copies"], cfg["actual_physical_forward_copies"])
            no_cfg = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 1.0, 1, "pag", 6 * 1024**3, 4 * 1024**3,
                pag_scale=0.0,
            )
            no_cfg_copies = (no_cfg["logical_guidance_copies"], no_cfg["actual_physical_forward_copies"])
            pag = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 5.0, 1, "pag", 6 * 1024**3, 4 * 1024**3,
                pag_scale=0.3,
            )
            pag_copies = (pag["logical_guidance_copies"], pag["actual_physical_forward_copies"])
        self.assertEqual(cfg_copies, (2, 1))
        self.assertEqual(no_cfg_copies, (1, 1))
        self.assertEqual(pag_copies, (3, 1))
        with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()) as choose:
            inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 7.0, 1, "cfg_zero_star", 100, 80
            )
        self.assertEqual([call.args[7] for call in choose.call_args_list], [1, 2])

    def test_anima_cfg_batch_is_disabled_for_pag_guidance(self):
        strategy = {
            "mode": "normal_vram",
            "normal_available": True,
            "weight_gb": 5.0,
            "offload_mode": "model",
            "model_resident": False,
            "free_bytes": 20 * 1024**3,
            "inference_bytes": 1024**3,
            "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()):
            selected = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 5.0, 1, "pag", 6 * 1024**3, 4 * 1024**3
            )
        self.assertFalse(selected["cfg_batch"])
        self.assertFalse(selected["keep_transformer_resident"])

    def test_anima_strategy_uses_group_offload_when_full_transformer_is_over_budget(self):
        strategy = {
            "mode": "low_vram",
            "normal_available": False,
            "weight_gb": 5.0,
            "offload_mode": "sequential",
            "model_resident": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = {}
            for name in ("diffusion_model", "text_encoder", "vae"):
                path = Path(temporary) / f"{name}.safetensors"
                path.write_bytes(b"weights")
                paths[name] = path
            with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()):
                selected = inference_server.anima_memory_strategy(
                    paths["diffusion_model"], paths, 1024, 1024, 1.0, 1, "none"
                )
        self.assertTrue(selected["transformer_group_offload"])
        self.assertEqual(selected["offload_mode"], "staged_transformer_group_offload")

    def test_demonstrated_1088x1472_request_prefers_resident_transformer_when_budget_allows(self):
        strategy = {
            "mode": "normal_vram",
            "normal_available": True,
            "weight_gb": 5.2,
            "total_gb": 8.0,
            "free_bytes": 7_441_743_872,
            "inference_bytes": 1_355_284_480,
            "reserved_bytes": 629_145_600,
        }
        with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()) as choose:
            selected = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"),
                1088,
                1472,
                5.0,
                1,
                "none",
                5_628_022_758,
                4_182_137_856,
                lora_bytes=334_694_104,
            )
        self.assertEqual([call.args[6:8] for call in choose.call_args_list], [(1, 1), (1, 2)])
        self.assertFalse(selected["transformer_group_offload"])
        self.assertEqual(selected["transformer_blocks_per_group"], 0)
        self.assertEqual(selected["offload_mode"], "staged_transformer_resident")
        self.assertEqual(selected["sampling_batch_size"], 1)
        self.assertEqual(selected["adapter_source_bytes"], 334_694_104)
        self.assertFalse(selected["cfg_batch"])
        self.assertFalse(selected["keep_transformer_resident"])

    def test_large_gpu_high_vram_strategy_keeps_transformer_between_jobs(self):
        strategy = {
            "mode": "high_vram",
            "normal_available": True,
            "weight_gb": 5.2,
            "free_bytes": 20 * 1024**3,
            "inference_bytes": 1024**3,
            "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", return_value=strategy.copy()):
            selected = inference_server.choose_anima_memory_strategy(
                Path("C:/models/anima.safetensors"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3
            )
        self.assertTrue(selected["keep_transformer_resident"])
        self.assertFalse(selected["cfg_batch"])

    def test_group_offload_still_rejects_when_one_group_and_activations_do_not_fit(self):
        strategy = {
            "mode": "low_vram",
            "normal_available": False,
            "weight_gb": 5.2,
            "total_gb": 8.0,
            "free_bytes": 1_000,
            "inference_bytes": 2_000,
            "reserved_bytes": 1_000,
        }
        with patch.object(inference_server, "choose_memory_strategy", return_value=strategy):
            with self.assertRaisesRegex(RuntimeError, "Transformer"):
                inference_server.choose_anima_memory_strategy(
                    Path("C:/models/anima.safetensors"), 2048, 2048, 7.0, 1, "none", 100, 80
                )

    def test_group_cfg_batch_is_internal_gate_with_admission_and_pag_cfg_exclusions(self):
        strategy = {
            "mode": "low_vram", "normal_available": False, "weight_gb": 5.0,
            "offload_mode": "sequential", "model_resident": False,
            "free_bytes": 128 * 1024**3, "inference_bytes": 1024**3,
            "normal_required_bytes": 8 * 1024**3, "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", side_effect=lambda *_args, **_kwargs: strategy.copy()):
            with patch.object(inference_server, "anima_group_cfg_batch_requested", return_value=False):
                default = copy.deepcopy(inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3))
            with patch.object(inference_server, "anima_group_cfg_batch_requested", return_value=True):
                enabled = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3)
                pag = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "pag", 6 * 1024**3, 4 * 1024**3)
                cfg1 = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 1.0, 1, "none", 6 * 1024**3, 4 * 1024**3)
        self.assertFalse(default["cfg_batch"])
        self.assertFalse(default["admission"]["group_cfg_feature_requested"])
        self.assertTrue(enabled["cfg_batch"])
        self.assertTrue(enabled["admission"]["group_cfg_feature_admitted"])
        self.assertGreater(enabled["admission"]["group_cfg_required_bytes"], enabled["admission"]["group_sequential_required_bytes"])
        self.assertFalse(pag["cfg_batch"])
        self.assertFalse(cfg1["cfg_batch"])

    def test_resident_cfg_batch_probe_is_default_off_and_stage_isolated(self):
        strategy = {
            "mode": "normal_vram", "normal_available": True, "weight_gb": 5.0,
            "free_bytes": 32 * 1024**3, "inference_bytes": 1024**3,
            "normal_required_bytes": 8 * 1024**3, "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", side_effect=lambda *_a, **_k: strategy.copy()), \
             patch.object(inference_server, "anima_resident_cfg_batch_requested", return_value=False), \
             patch.object(inference_server.torch.cuda, "is_available", return_value=True):
            default = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3)
        with patch.object(inference_server, "choose_memory_strategy", side_effect=lambda *_a, **_k: strategy.copy()), \
             patch.object(inference_server, "anima_resident_cfg_batch_requested", return_value=True), \
             patch.object(inference_server.torch.cuda, "is_available", return_value=True), \
             patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_RESIDENT_CFG_BATCH": "1"}, clear=False):
            enabled = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3)
            pag = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "pag", 6 * 1024**3, 4 * 1024**3)
            cfg1 = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 1.0, 1, "none", 6 * 1024**3, 4 * 1024**3)
        self.assertFalse(default["cfg_batch"])
        self.assertFalse(default["admission"]["resident_cfg_batch_requested"])
        self.assertTrue(enabled["cfg_batch"])
        self.assertTrue(enabled["admission"]["resident_cfg_batch_requested"])
        self.assertTrue(enabled["admission"]["resident_cfg_batch_probe_eligible"])
        self.assertFalse(enabled["admission"]["resident_cfg_batch_admitted"])
        self.assertEqual(enabled["admission"]["resident_cfg_batch_admission_kind"], "experimental_estimate_not_safety_guarantee")
        self.assertFalse(pag["cfg_batch"])
        self.assertFalse(cfg1["cfg_batch"])

    def test_resident_cfg_batch_probe_fails_closed_with_insufficient_headroom(self):
        strategy = {
            "mode": "normal_vram", "normal_available": True, "weight_gb": 5.0,
            "free_bytes": 8 * 1024**3, "inference_bytes": 1024**3,
            "normal_required_bytes": 8 * 1024**3, "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", side_effect=lambda *_a, **_k: strategy.copy()), \
             patch.object(inference_server, "anima_resident_cfg_batch_requested", return_value=True), \
             patch.object(inference_server.torch.cuda, "is_available", return_value=True):
            selected = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3)
        admission = selected["admission"]
        self.assertFalse(selected["cfg_batch"])
        self.assertFalse(admission["resident_cfg_batch_probe_eligible"])
        self.assertGreater(admission["resident_cfg_batch_fragmentation_margin_bytes"], 0)

    def test_resident_cfg_probe_requires_two_flags_and_isolated_service_identity(self):
        isolated_output = Path(tempfile.gettempdir()) / "xirai-probe-output"
        isolated_state = Path(tempfile.gettempdir()) / "xirai-probe-state"
        base = {"XIRAI_BENCHMARK_CHILD": "", "XIRAI_ANIMA_RESIDENT_CFG_BATCH": ""}
        with patch.dict(inference_server.os.environ, base, clear=False):
            self.assertFalse(inference_server.anima_resident_cfg_batch_requested())
        with patch.dict(inference_server.os.environ, {**base, "XIRAI_BENCHMARK_CHILD": "1"}, clear=False):
            self.assertFalse(inference_server.anima_resident_cfg_batch_requested())
        with patch.dict(inference_server.os.environ, {**base, "XIRAI_ANIMA_RESIDENT_CFG_BATCH": "1"}, clear=False):
            self.assertFalse(inference_server.anima_resident_cfg_batch_requested())
        with patch.object(inference_server, "INFERENCE_PORT", 8719), \
             patch.object(inference_server, "OUTPUT_DIRECTORY", isolated_output), \
             patch.object(inference_server, "STATE_DIRECTORY", isolated_state), \
             patch.dict(inference_server.os.environ, {"XIRAI_BENCHMARK_CHILD": "1", "XIRAI_ANIMA_RESIDENT_CFG_BATCH": "1"}, clear=False):
            self.assertTrue(inference_server.anima_resident_cfg_batch_requested())

    def test_resident_cfg_force_requires_four_factor_isolated_identity_and_can_override_headroom(self):
        isolated_output = Path(tempfile.gettempdir()) / "xirai-force-output"
        isolated_state = Path(tempfile.gettempdir()) / "xirai-force-state"
        identity = {
            "XIRAI_BENCHMARK_CHILD": "1", "XIRAI_ANIMA_RESIDENT_CFG_BATCH": "1",
            "XIRAI_ANIMA_RESIDENT_CFG_BATCH_FORCE": "1", "XIRAI_BENCHMARK_PURPOSE": "resident_cfg_batch_probe",
        }
        with patch.object(inference_server, "INFERENCE_PORT", 8719), \
             patch.object(inference_server, "OUTPUT_DIRECTORY", isolated_output), \
             patch.object(inference_server, "STATE_DIRECTORY", isolated_state):
            for missing in identity:
                env = identity.copy(); env[missing] = ""
                with patch.dict(inference_server.os.environ, env, clear=False):
                    self.assertFalse(inference_server.anima_resident_cfg_batch_force_effective(), missing)
            with patch.dict(inference_server.os.environ, identity, clear=False), \
                 patch.object(inference_server, "benchmark_lease_validation", return_value={"valid": True, "reason": "valid", "expiry": None, "purpose": "resident_cfg_batch_probe"}):
                self.assertTrue(inference_server.anima_resident_cfg_batch_force_effective())
        strategy = {
            "mode": "normal_vram", "normal_available": True, "weight_gb": 5.0,
            "free_bytes": 8 * 1024**3, "inference_bytes": 1024**3,
            "normal_required_bytes": 8 * 1024**3, "reserved_bytes": 600 * 1024**2,
        }
        with patch.object(inference_server, "choose_memory_strategy", side_effect=lambda *_a, **_k: strategy.copy()), \
             patch.object(inference_server, "anima_resident_cfg_batch_requested", return_value=True), \
             patch.object(inference_server, "anima_resident_cfg_batch_force_requested", return_value=True), \
             patch.object(inference_server, "anima_resident_cfg_batch_force_effective", return_value=True), \
             patch.object(inference_server.torch.cuda, "is_available", return_value=True):
            selected = inference_server.choose_anima_memory_strategy(Path("C:/a"), 1024, 1024, 5.0, 1, "none", 6 * 1024**3, 4 * 1024**3)
        admission = selected["admission"]
        self.assertTrue(selected["cfg_batch"])
        self.assertFalse(admission["resident_cfg_batch_probe_eligible"])
        self.assertFalse(admission["resident_cfg_batch_admitted"])
        self.assertTrue(admission["resident_cfg_batch_force_requested"])
        self.assertTrue(admission["resident_cfg_batch_force_effective"])
        self.assertEqual(admission["resident_cfg_batch_admission_kind"], "forced_experimental_speculative_no_safety_guarantee")
        self.assertEqual(admission["resident_cfg_batch_gate_reason"], "forced_experimental_speculative_probe")
        json.dumps(admission)

    def test_forced_probe_lease_rejects_env_copy_and_validates_parent_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); state = root / "child-state"; output = root / "child-output"
            script = root / "state-cache" / "benchmark" / "lease-harness.py"
            script.parent.mkdir(parents=True, exist_ok=True); script.write_text("# harness", encoding="utf-8")
            python = root / ".venv" / "Scripts" / "python.exe"; python.parent.mkdir(parents=True, exist_ok=True); python.write_bytes(b"")
            now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            path, nonce = create_lease(state, output, "workspace-test", script, 60, parent_pid=42, parent_executable=python)
            inspector = lambda _pid: {"executable": str(python), "command_line": [str(python), str(script)]}
            valid = validate_lease(path, nonce, state, output, "workspace-test", root, parent_pid=42, now=now, inspector=inspector)
            self.assertEqual(valid["reason"], "valid"); self.assertTrue(valid["valid"]); json.dumps(valid)
            self.assertFalse(validate_lease(path, "wrong", state, output, "workspace-test", root, parent_pid=42, now=now, inspector=inspector)["valid"])
            self.assertFalse(validate_lease(path, nonce, state, output, "wrong", root, parent_pid=42, now=now, inspector=inspector)["valid"])
            self.assertFalse(validate_lease(path, nonce, state, output, "workspace-test", root, parent_pid=43, now=now, inspector=inspector)["valid"])
            self.assertFalse(validate_lease(path, nonce, state, output, "workspace-test", root, parent_pid=42, now=now, inspector=lambda _p: {"executable": "node.exe", "command_line": ["node"]})["valid"])
            expired_path, expired_nonce = create_lease(state, output, "workspace-test", script, 1, parent_pid=42, parent_executable=python)
            later = now + __import__("datetime").timedelta(seconds=2)
            self.assertFalse(validate_lease(expired_path, expired_nonce, state, output, "workspace-test", root, parent_pid=42, now=later, inspector=inspector)["valid"])

    def test_run_generation_uses_native_batch_api_without_sd_hooks(self):
        request = anima_request(
            guidance="pag",
            pag={"scale": 0.65, "applied_layers": "all"},
            images_per_batch=2,
            loras=[
                {"path": "styles/first.safetensors", "weight": 0.75},
                {"path": "second.safetensors", "weight": -0.25},
            ],
            hires={"enabled": True, "model": "upscaler.pth"},
            adetailer={"enabled": True, "units": [{"detector": "face.pt"}]},
            rtx={"enabled": True},
            postprocess_order=["rtx", "hires", "adetailer"],
        )

        class Runtime:
            generate_kwargs = None

            def token_diagnostics(self, text):
                return {
                    "qwen": {"token_count": len(text), "weighted_token_count": 0, "max_length": 512},
                    "t5": {"token_count": len(text), "weighted_token_count": 3, "max_length": 512},
                }

            def generate_batch(self, **kwargs):
                self.generate_kwargs = kwargs
                self.sampling_batch_size = kwargs["sampling_batch_size"]
                self.last_generation_metrics = {"sampling": {"seconds": 0.1}}
                kwargs["on_step"](1, kwargs["steps"], None)
                kwargs["on_step"](kwargs["steps"], kwargs["steps"], None)
                return [Image.new("RGB", (512, 512), "red"), Image.new("RGB", (512, 512), "blue")]

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = {
                "diffusion_model": root / "diffusion",
                "text_encoder": root / "text",
                "vae": root / "vae",
                "lora": root / "loras",
            }
            for directory in roots.values():
                directory.mkdir()
            tokenizer_directory = root / "bundled"
            tokenizer_directory.mkdir()
            for directory, relative in (
                (roots["diffusion_model"], request.diffusion_model),
                (roots["text_encoder"], request.text_encoder),
                (roots["vae"], request.vae),
            ):
                path = directory / relative
                path.parent.mkdir(parents=True)
                path.write_bytes(b"weights")
            for lora in request.loras:
                path = roots["lora"] / lora.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(lora.path.encode("utf-8"))
            for name in (
                "anima-qwen3-tokenizer.json",
                "anima-qwen3-tokenizer-config.json",
                "anima-t5-tokenizer.json",
            ):
                (tokenizer_directory / name).write_text("{}", encoding="utf-8")

            output = root / "outputs"
            output.mkdir()
            original_output = inference_server.OUTPUT_DIRECTORY
            original_settings = inference_server.performance_settings.copy()
            inference_server.OUTPUT_DIRECTORY = output
            inference_server.performance_settings.update({
                "memory_mode": "auto",
                "staged_vae_decode": False,
                "keep_model_cached": True,
            })
            inference_server.jobs["anima-job"] = {"status": "queued"}
            inference_server.job_controls["anima-job"] = inference_server.JobControl()
            stage_trace = []

            def rtx_stage(image, *_args, **_kwargs):
                stage_trace.append("rtx")
                return image, {"provider": "mock RTX"}

            def hires_stage(image, *_args, **_kwargs):
                stage_trace.append("hires")
                return image, {"model": "upscaler.pth", "effective_steps": 7}

            def detail_stage(image, *_args, **_kwargs):
                stage_trace.append("adetailer")
                return image, {"detector": "face.pt"}, None

            try:
                with (
                    patch.object(inference_server.torch.cuda, "is_available", return_value=True),
                    patch.object(inference_server, "anima_model_roots", return_value=roots),
                    patch.object(
                        inference_server,
                        "anima_tokenizer_sources",
                        return_value={
                            "qwen": tokenizer_directory / "anima-qwen3-tokenizer.json",
                            "qwen_config": tokenizer_directory / "anima-qwen3-tokenizer-config.json",
                            "t5": tokenizer_directory / "anima-t5-tokenizer.json",
                        },
                    ),
                    patch.object(inference_server, "load_anima_pipeline", return_value=runtime) as load_runtime,
                    patch.object(inference_server, "apply_rtx_vsr", side_effect=rtx_stage) as apply_rtx,
                    patch.object(inference_server, "apply_hires_fix", side_effect=hires_stage) as apply_hires,
                    patch.object(inference_server, "apply_adetailer", side_effect=detail_stage) as apply_detail,
                    patch.object(inference_server, "configure_scheduler", side_effect=AssertionError("SD scheduler hook used")),
                    patch.object(inference_server, "configure_loras", side_effect=AssertionError("LoRA hook used")),
                    patch.object(
                        inference_server,
                        "prepare_prompt_conditioning",
                        side_effect=AssertionError("CLIP prompt hook used"),
                    ),
                ):
                    inference_server.run_generation("anima-job", request)

                job = inference_server.jobs["anima-job"]
                self.assertEqual(job["status"], "complete")
                self.assertEqual(job["completed_images"], 2)
                self.assertEqual(job["prompt_tokens"], len(request.prompt))
                self.assertEqual(job["prompt_blocks"], 1)
                self.assertEqual(job["prompt_weighted_tokens"], 3)
                self.assertIn("Flow Matching uses shift 3", job["warning"])
                self.assertEqual(len(list(output.rglob("*.png"))), 2)
                self.assertEqual(runtime.sampling_batch_size, 2)
                self.assertEqual(runtime.generate_kwargs["guidance"], "pag")
                self.assertEqual(runtime.generate_kwargs["pag_scale"], 0.65)
                self.assertEqual(runtime.generate_kwargs["pag_applied_layers"], "all")
                self.assertEqual(job["runtime_metrics"], runtime.last_generation_metrics)
                load_runtime.assert_called_once()
                self.assertEqual(
                    load_runtime.call_args.kwargs["loras"],
                    [
                        (roots["lora"] / "styles/first.safetensors", 0.75),
                        (roots["lora"] / "second.safetensors", -0.25),
                    ],
                )
                self.assertEqual(
                    load_runtime.call_args.kwargs["lora_bytes"],
                    sum((roots["lora"] / lora.path).stat().st_size for lora in request.loras),
                )
                self.assertEqual(stage_trace, ["rtx", "hires", "adetailer"] * 2)
                self.assertEqual(apply_rtx.call_count, 2)
                self.assertEqual(apply_hires.call_count, 2)
                self.assertEqual(apply_detail.call_count, 2)
                with Image.open(next(output.rglob("*.png"))) as saved:
                    parameters = json.loads(saved.info["parameters"])
                self.assertEqual(
                    parameters["guidance"]["implementation"],
                    "native_cosmos_identity_self_attention",
                )
                self.assertEqual(parameters["guidance"]["resolved_layer_count"], 28)
                self.assertEqual(parameters["guidance"]["logical_prediction_branches"], 3)
                self.assertEqual(
                    parameters["loras"],
                    [
                        {"name": "first.safetensors", "weight": 0.75},
                        {"name": "second.safetensors", "weight": -0.25},
                    ],
                )
                self.assertEqual(parameters["postprocess_stages"], ["rtx", "hires", "adetailer"])
            finally:
                inference_server.jobs.pop("anima-job", None)
                inference_server.job_controls.pop("anima-job", None)
                inference_server.performance_settings.clear()
                inference_server.performance_settings.update(original_settings)
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_run_generation_resolves_one_independent_random_hires_seed_per_output(self):
        request = anima_request(
            seed=1015878324182247,
            images_per_batch=2,
            hires={"enabled": True, "model": "upscaler.pth", "mode": "random"},
            postprocess_order=["hires", "adetailer", "rtx"],
        )

        class Runtime:
            def token_diagnostics(self, text):
                return {
                    "qwen": {"token_count": len(text), "weighted_token_count": 0, "max_length": 512},
                    "t5": {"token_count": len(text), "weighted_token_count": 3, "max_length": 512},
                }

            def generate_batch(self, **kwargs):
                self.last_generation_metrics = {"sampling": {"seconds": 0.1}}
                kwargs["on_step"](kwargs["steps"], kwargs["steps"], None)
                return [Image.new("RGB", (512, 512), "red"), Image.new("RGB", (512, 512), "blue")]

        runtime = Runtime()
        resolved = [885289963651097, 9007199254740993]
        random_calls = []
        hires_seeds = []

        def random_uint64():
            random_calls.append(len(random_calls))
            return resolved[len(random_calls) - 1]

        def hires_stage(image, *_args, effective_hires_seed=None, **_kwargs):
            hires_seeds.append(effective_hires_seed)
            return image, {"model": "upscaler.pth", "effective_steps": 7, "hires_seed": str(effective_hires_seed)}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = {
                "diffusion_model": root / "diffusion",
                "text_encoder": root / "text",
                "vae": root / "vae",
                "lora": root / "loras",
            }
            for directory in roots.values():
                directory.mkdir()
            tokenizer_directory = root / "bundled"
            tokenizer_directory.mkdir()
            for directory, relative in (
                (roots["diffusion_model"], request.diffusion_model),
                (roots["text_encoder"], request.text_encoder),
                (roots["vae"], request.vae),
            ):
                path = directory / relative
                path.parent.mkdir(parents=True)
                path.write_bytes(b"weights")
            for name in (
                "anima-qwen3-tokenizer.json",
                "anima-qwen3-tokenizer-config.json",
                "anima-t5-tokenizer.json",
            ):
                (tokenizer_directory / name).write_text("{}", encoding="utf-8")

            output = root / "outputs"
            output.mkdir()
            original_output = inference_server.OUTPUT_DIRECTORY
            original_settings = inference_server.performance_settings.copy()
            inference_server.OUTPUT_DIRECTORY = output
            inference_server.performance_settings.update({
                "memory_mode": "auto",
                "staged_vae_decode": False,
                "keep_model_cached": True,
            })
            inference_server.jobs["seed-job"] = {"status": "queued"}
            inference_server.job_controls["seed-job"] = inference_server.JobControl()
            try:
                with (
                    patch.object(inference_server.torch.cuda, "is_available", return_value=True),
                    patch.object(inference_server, "anima_model_roots", return_value=roots),
                    patch.object(
                        inference_server,
                        "anima_tokenizer_sources",
                        return_value={
                            "qwen": tokenizer_directory / "anima-qwen3-tokenizer.json",
                            "qwen_config": tokenizer_directory / "anima-qwen3-tokenizer-config.json",
                            "t5": tokenizer_directory / "anima-t5-tokenizer.json",
                        },
                    ),
                    patch.object(inference_server, "load_anima_pipeline", return_value=runtime),
                    patch.object(inference_server, "secure_random_uint64", side_effect=random_uint64),
                    patch.object(inference_server, "apply_hires_fix", side_effect=hires_stage),
                ):
                    inference_server.run_generation("seed-job", request)

                job = inference_server.jobs["seed-job"]
                self.assertEqual(job["status"], "complete")
                self.assertEqual(job["completed_images"], 2)

                # One secure resolution per output: never per tile, per stage, or per batch.
                self.assertEqual(len(random_calls), 2)
                self.assertEqual(hires_seeds, resolved)
                self.assertEqual(len(set(hires_seeds)), 2)

                self.assertEqual(job["hires_seed_mode"], "random")
                self.assertEqual(job["base_seed"], job["seed"])
                self.assertEqual(job["hires_seed"], str(resolved[-1]))

                saved = sorted(output.rglob("*.png"))
                self.assertEqual(len(saved), 2)
                facts = []
                for path in saved:
                    with Image.open(path) as image:
                        facts.append(json.loads(image.info["parameters"]))
                self.assertEqual([item["hires_seed_mode"] for item in facts], ["random", "random"])
                self.assertEqual(sorted(item["hires_seed"] for item in facts), sorted(str(value) for value in resolved))
                self.assertEqual({item["base_seed"] for item in facts}, {item["seed"] for item in facts})
                self.assertEqual(len({item["base_seed"] for item in facts}), 2)
                for path, item in zip(saved, facts):
                    record = inference_server.history_file_record(path)
                    self.assertEqual(record["hires_seed_mode"], item["hires_seed_mode"])
                    self.assertEqual(record["hires_seed"], item["hires_seed"])
                    self.assertEqual(record["base_seed"], item["base_seed"])
            finally:
                inference_server.jobs.pop("seed-job", None)
                inference_server.job_controls.pop("seed-job", None)
                inference_server.performance_settings.clear()
                inference_server.performance_settings.update(original_settings)
                inference_server.OUTPUT_DIRECTORY = original_output

    def test_create_job_records_unresolved_hires_seed_facts(self):
        cases = (
            ({"enabled": True, "model": "upscaler.pth"}, "inherit"),
            ({"enabled": True, "model": "upscaler.pth", "mode": "fixed", "seed": "885289963651097"}, "fixed"),
            ({"enabled": True, "model": "upscaler.pth", "mode": "random"}, "random"),
            ({"enabled": False}, "inherit"),
        )
        for hires, mode in cases:
            with self.subTest(mode=mode, enabled=hires["enabled"]):
                request = anima_request(seed=18446744073709551615, hires=hires)
                with (
                    patch.dict(inference_server.performance_settings, {"memory_mode": "auto"}),
                    patch.object(inference_server, "resolve_upscaler_model", return_value=(Path("upscaler.pth"), {})),
                    patch.object(
                        inference_server,
                        "secure_random_uint64",
                        side_effect=AssertionError("Hires seed resolved before the job started"),
                    ),
                    patch.object(inference_server.executor, "submit") as submit,
                ):
                    job = inference_server.create_job(request)
                try:
                    self.assertEqual(job["seed"], "18446744073709551615")
                    self.assertEqual(job["base_seed"], "18446744073709551615")
                    self.assertEqual(job["hires_seed_mode"], mode)
                    self.assertIsNone(job["hires_seed"])
                    self.assertEqual(job["hires_enabled"], hires["enabled"])
                    submit.assert_called_once()
                finally:
                    inference_server.jobs.pop(job["id"], None)
                    inference_server.job_controls.pop(job["id"], None)

    def test_active_job_exposes_requested_anima_identity(self):
        request = anima_request()
        with patch.object(inference_server.executor, "submit"):
            job = inference_server.create_job(request)
        try:
            payload = inference_server.get_active_job()["job"]
            self.assertEqual(payload["id"], job["id"])
            self.assertEqual(payload["requested_engine"], "Anima")
            self.assertEqual(
                payload["requested_model_assets"],
                {
                    "diffusion_model": request.diffusion_model,
                    "text_encoder": request.text_encoder,
                    "vae": request.vae,
                },
            )
        finally:
            inference_server.jobs.pop(job["id"], None)
            inference_server.job_controls.pop(job["id"], None)


class AnimaAccelerationGateTests(unittest.TestCase):
    """Both accelerators are opt-in, and diagnostics report reality, not intent."""

    def setUp(self):
        # These gates read the live Performance settings as well as the
        # environment, so the settings have to be isolated too — otherwise the
        # suite passes or fails according to whatever the user last saved.
        self._saved_settings = dict(inference_server.performance_settings)
        inference_server.performance_settings.update({"attention_backend": "auto", "compile_transformer": False})

    def tearDown(self):
        inference_server.performance_settings.clear()
        inference_server.performance_settings.update(self._saved_settings)

    def _runtime(self, **overrides):
        state = {"compiled": False, "mode": None, "sage": False}

        def configure_compile(enabled=False, mode="default"):
            state.update(compiled=bool(enabled), mode=mode if enabled else None)
            return state["compiled"]

        def configure_sage(enabled=False):
            state["sage"] = bool(enabled)
            return state["sage"]

        runtime = SimpleNamespace(
            configure_transformer_compilation=configure_compile,
            configure_sage_attention=configure_sage,
            transformer_group_offload_enabled=False,
        )
        runtime.__dict__.update(overrides)
        return runtime, state

    def test_nothing_is_enabled_without_the_environment_switches(self):
        runtime, state = self._runtime()
        with patch.dict(inference_server.os.environ, {}, clear=False):
            inference_server.os.environ.pop("XIRAI_ANIMA_COMPILE", None)
            inference_server.os.environ.pop("XIRAI_ANIMA_SAGE_ATTENTION", None)
            record = inference_server.configure_anima_acceleration(runtime)
        self.assertEqual(
            {key: record[key] for key in ("compile_requested", "compile_active", "sage_requested", "sage_active")},
            {"compile_requested": False, "compile_active": False, "sage_requested": False, "sage_active": False},
        )
        # An un-asked-for accelerator is actively turned off, never merely skipped:
        # a cached runtime must not keep yesterday's switch.
        self.assertEqual(state, {"compiled": False, "mode": None, "sage": False})

    def test_both_switches_engage_and_are_reported(self):
        runtime, state = self._runtime()
        with patch.dict(
            inference_server.os.environ,
            {"XIRAI_ANIMA_COMPILE": "1", "XIRAI_ANIMA_SAGE_ATTENTION": "1", "XIRAI_ANIMA_COMPILE_MODE": "max-autotune"},
        ), patch.object(inference_server, "find_spec", return_value=object()):
            record = inference_server.configure_anima_acceleration(runtime)
        self.assertTrue(record["compile_active"])
        self.assertEqual(record["compile_mode"], "max-autotune")
        self.assertTrue(record["sage_active"])
        self.assertEqual(state, {"compiled": True, "mode": "max-autotune", "sage": True})

    def test_group_offload_refuses_compilation_by_name(self):
        runtime, state = self._runtime(transformer_group_offload_enabled=True)
        with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_COMPILE": "1"}):
            record = inference_server.configure_anima_acceleration(runtime)
        self.assertFalse(record["compile_active"])
        self.assertEqual(record["compile_unavailable_reason"], "group_offload_active")
        self.assertFalse(state["compiled"])

        runtime, _state = self._runtime()
        with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_COMPILE": "1"}):
            record = inference_server.configure_anima_acceleration(runtime, {"transformer_group_offload": True})
        self.assertEqual(record["compile_unavailable_reason"], "group_offload_active")

    def test_a_missing_package_or_a_raising_runtime_is_reported_not_raised(self):
        runtime, _state = self._runtime(configure_sage_attention=lambda enabled=False: False)
        with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_SAGE_ATTENTION": "1"}):
            record = inference_server.configure_anima_acceleration(runtime)
        self.assertFalse(record["sage_active"])
        self.assertEqual(record["sage_unavailable_reason"], "sageattention_not_installed")

        def explode(enabled=False, mode="default"):
            raise RuntimeError("inductor unavailable")

        runtime, _state = self._runtime(configure_transformer_compilation=explode)
        with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_COMPILE": "1"}), patch.object(
            inference_server, "find_spec", return_value=object()
        ):
            record = inference_server.configure_anima_acceleration(runtime)
        self.assertFalse(record["compile_active"])
        # The message matters, not just the type: the first real failure here was
        # an upstream UnicodeDecodeError that the bare type name did not locate.
        self.assertEqual(record["compile_unavailable_reason"], "RuntimeError: inductor unavailable")

    def test_an_unknown_compile_mode_is_rejected_rather_than_guessed(self):
        with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_COMPILE_MODE": "turbo"}):
            with self.assertRaisesRegex(RuntimeError, "XIRAI_ANIMA_COMPILE_MODE"):
                inference_server.anima_compile_mode()
        with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_COMPILE_MODE": "reduce-overhead"}):
            self.assertEqual(inference_server.anima_compile_mode(), "reduce-overhead")

    def test_the_performance_settings_drive_both_accelerators(self):
        """The configurator installs them; the Performance panel turns them on."""
        settings = inference_server.performance_settings
        settings["attention_backend"] = "sage"
        settings["compile_transformer"] = True
        with patch.dict(inference_server.os.environ, {}, clear=False):
            inference_server.os.environ.pop("XIRAI_ANIMA_COMPILE", None)
            inference_server.os.environ.pop("XIRAI_ANIMA_SAGE_ATTENTION", None)
            self.assertTrue(inference_server.anima_compile_requested())
            self.assertTrue(inference_server.anima_sage_attention_requested())

            settings["attention_backend"] = "auto"
            settings["compile_transformer"] = False
            self.assertFalse(inference_server.anima_compile_requested())
            self.assertFalse(inference_server.anima_sage_attention_requested())

        # The environment switches remain, so the benchmark harness can force
        # either one without writing the user's saved settings.
        with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_COMPILE": "1", "XIRAI_ANIMA_SAGE_ATTENTION": "1"}):
            self.assertTrue(inference_server.anima_compile_requested())
            self.assertTrue(inference_server.anima_sage_attention_requested())

    def test_sage_reports_the_kernel_that_ran_the_self_attention(self):
        runtime, _state = self._runtime()
        previous = inference_server.active_attention_backend
        try:
            inference_server.active_attention_backend = "PyTorch SDPA"
            with patch.dict(inference_server.os.environ, {"XIRAI_ANIMA_SAGE_ATTENTION": "1"}):
                record = inference_server.configure_anima_acceleration(runtime)
            self.assertTrue(record["sage_active"])
            self.assertEqual(inference_server.active_attention_backend, "SageAttention INT8")
        finally:
            inference_server.active_attention_backend = previous

    def test_the_job_reports_sage_rather_than_the_dispatch_it_falls_back_to(self):
        """A real run showed `PyTorch SDPA` while Sage ran the self-attention.

        `current_attention_backend` reads the runtime's Diffusers backend, which
        stays `native` because Sage is applied by the processor, not the dispatch.
        """
        previous = (inference_server.loaded_engine, inference_server.loaded_pipeline)
        try:
            inference_server.loaded_engine = "Anima"
            inference_server.loaded_pipeline = SimpleNamespace(attention_backend="native", sage_attention_enabled=True)
            self.assertEqual(inference_server.current_attention_backend(), "SageAttention INT8")
            inference_server.loaded_pipeline = SimpleNamespace(attention_backend="native", sage_attention_enabled=False)
            self.assertEqual(inference_server.current_attention_backend(), "PyTorch SDPA")
        finally:
            inference_server.loaded_engine, inference_server.loaded_pipeline = previous

    def test_the_job_payload_carries_what_actually_ran(self):
        strategy = {
            "mode": "normal_vram", "label": "NORMAL", "reason": "test",
            "offload_mode": "staged_transformer_resident", "model_resident": False,
            "acceleration": {"compile_active": True, "sage_active": False},
        }
        fields = inference_server.memory_job_fields(strategy)
        self.assertEqual(fields["acceleration"], {"compile_active": True, "sage_active": False})
        self.assertIsNone(inference_server.memory_job_fields({**strategy, "acceleration": None})["acceleration"])


if __name__ == "__main__":
    unittest.main()
