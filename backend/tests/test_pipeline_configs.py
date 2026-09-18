import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.pipeline_configs import FAMILY_REQUIRED_FILES, download_pipeline_configs, failure_reason, installed_pipeline_families, pipeline_config_complete


class PipelineConfigTests(unittest.TestCase):
    def test_sdxl_config_requires_both_text_encoders_and_tokenizers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in FAMILY_REQUIRED_FILES["sdxl"]:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

            self.assertTrue(pipeline_config_complete(root, "sdxl"))
            (root / "tokenizer_2" / "vocab.json").unlink()
            self.assertFalse(pipeline_config_complete(root, "sdxl"))

    def test_sd_config_does_not_require_sdxl_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in FAMILY_REQUIRED_FILES["sd"]:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

            self.assertTrue(pipeline_config_complete(root, "sd"))

    def test_only_installed_checkpoint_families_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "models" / "checkpoints" / "illustrious" / "nested" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            self.assertEqual(installed_pipeline_families(root), ["sdxl"])

    def test_configured_nested_checkpoint_roots_are_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models" / "model-paths.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"checkpoints":{"sd":"models/custom/sd/deep","illustrious":"models/custom/il/deep"}}', encoding="utf-8")
            checkpoint = root / "models" / "custom" / "sd" / "deep" / "folder" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertEqual(installed_pipeline_families(root), ["sd"])

    def test_fresh_install_prepares_all_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(installed_pipeline_families(temporary), ["sd", "sdxl"])

    def test_fresh_install_requires_nothing_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(installed_pipeline_families(temporary, False), [])

    def test_required_scope_still_covers_installed_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "models" / "checkpoints" / "illustrious" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            self.assertEqual(installed_pipeline_families(root, False), ["sdxl"])

    def test_failure_reason_reports_the_root_network_cause(self):
        try:
            try:
                raise ConnectionRefusedError(111, "Connection refused")
            except ConnectionRefusedError as cause:
                raise OSError("An error happened while trying to locate the files on the Hub") from cause
        except OSError as error:
            reason = failure_reason(error)

        self.assertEqual(reason, "OSError <- ConnectionRefusedError: [Errno 111] Connection refused")

    def test_download_failure_names_every_endpoint_and_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("backend.pipeline_configs.snapshot_download", side_effect=OSError("Connection refused")):
                with self.assertRaises(RuntimeError) as failure:
                    download_pipeline_configs(root / "cache", ["sd"])

            message = str(failure.exception)
            self.assertIn("https://huggingface.co => OSError: Connection refused", message)
            self.assertIn("https://hf-mirror.com => OSError: Connection refused", message)

    def test_download_falls_back_to_hf_mirror(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in FAMILY_REQUIRED_FILES["sdxl"]:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            with patch("backend.pipeline_configs.snapshot_download", side_effect=[OSError("official unavailable"), str(root)]) as download:
                result = download_pipeline_configs(root / "cache", ["sdxl"])

            self.assertEqual(result["sdxl"], root)
            self.assertEqual([call.kwargs["endpoint"] for call in download.call_args_list], ["https://huggingface.co", "https://hf-mirror.com"])

    def test_download_uses_existing_cache_when_all_endpoints_fail(self):
        # When the network is completely unavailable but a complete local cache
        # exists, download_pipeline_configs should skip the error and use it.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in FAMILY_REQUIRED_FILES["sd"]:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            def offline_hub(*_args, **keywords):
                # Only the local-cache lookup passes local_files_only, so this fake is
                # "every route is down, the cache is already complete".
                if keywords.get("local_files_only"):
                    return str(root)
                raise OSError("network unavailable")

            with patch("backend.pipeline_configs.snapshot_download", side_effect=offline_hub):
                result = download_pipeline_configs(root, ["sd"])

            self.assertEqual(result["sd"], root)


if __name__ == "__main__":
    unittest.main()
