import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import inference_server
from backend.shared_model_paths import shared_root_id


class SharedModelRuntimeTests(unittest.TestCase):
    """The inference side of shared folders: a `shared:` reference has to load."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        (self.project / "models").mkdir()
        self.shared = self.project / "elsewhere" / "Lora"
        (self.shared / "character").mkdir(parents=True)
        self.lora = self.shared / "character" / "hero.safetensors"
        self.lora.write_bytes(b"weights")
        self.root_id = shared_root_id(str(self.shared.resolve()))
        self.write_roots([{"id": self.root_id, "path": str(self.shared), "kind": "loras"}])

    def tearDown(self):
        self.temporary.cleanup()

    def write_roots(self, roots):
        manifest = self.project / "models" / "shared-paths.json"
        manifest.write_text(json.dumps({"version": 1, "roots": roots}), encoding="utf-8")

    def test_shared_and_local_references_both_resolve(self):
        local_root = self.project / "models" / "loras" / "sd"
        local_root.mkdir(parents=True)
        (local_root / "local.safetensors").write_bytes(b"weights")

        with patch.object(inference_server, "PROJECT_ROOT", self.project):
            shared = inference_server.resolve_model_path(
                f"shared:{self.root_id}/character/hero.safetensors", local_root, inference_server.LORA_EXTENSIONS, "LoRA"
            )
            self.assertEqual(shared, self.lora.resolve())
            local = inference_server.resolve_model_path(
                "local.safetensors", local_root, inference_server.LORA_EXTENSIONS, "LoRA"
            )
            self.assertEqual(local.name, "local.safetensors")

    def test_extension_allowlist_still_applies_to_shared_files(self):
        stray = self.shared / "character" / "notes.txt"
        stray.write_bytes(b"x")
        with patch.object(inference_server, "PROJECT_ROOT", self.project):
            with self.assertRaises(ValueError):
                inference_server.resolve_model_path(
                    f"shared:{self.root_id}/character/notes.txt", self.shared, inference_server.LORA_EXTENSIONS, "LoRA"
                )
            # Anima keeps its narrower allowlist even when the file is shared.
            legacy = self.shared / "character" / "old.pt"
            legacy.write_bytes(b"x")
            with self.assertRaises(ValueError):
                inference_server.resolve_model_path(
                    f"shared:{self.root_id}/character/old.pt", self.shared, inference_server.ANIMA_LORA_EXTENSIONS, "Anima LoRA"
                )

    def test_unregistered_roots_are_refused_even_when_the_file_exists(self):
        with patch.object(inference_server, "PROJECT_ROOT", self.project):
            self.write_roots([])
            with self.assertRaisesRegex(ValueError, "not registered"):
                inference_server.resolve_model_path(
                    f"shared:{self.root_id}/character/hero.safetensors", self.shared, inference_server.LORA_EXTENSIONS, "LoRA"
                )

    def test_loaded_paths_report_back_as_references_not_disk_layout(self):
        with patch.object(inference_server, "PROJECT_ROOT", self.project):
            self.assertEqual(
                inference_server.shared_reference_for_path(str(self.lora)),
                f"shared:{self.root_id}/character/hero.safetensors",
            )
            self.assertIsNone(inference_server.shared_reference_for_path(str(self.project / "models" / "x.safetensors")))
            self.assertIsNone(inference_server.shared_reference_for_path(""))

        # A shared reference is already the safe relative form of itself, so it
        # must survive the trip out to job records and PNG parameter blocks.
        reference = f"shared:{self.root_id}/character/hero.safetensors"
        self.assertEqual(inference_server.public_model_reference(reference), reference)
        self.assertEqual(inference_server.public_model_reference("sd/base.safetensors"), "sd/base.safetensors")
        self.assertEqual(inference_server.public_model_reference("C:\\absolute\\base.safetensors"), "base.safetensors")
        self.assertEqual(inference_server.public_model_reference("/absolute/base.safetensors"), "base.safetensors")
        self.assertEqual(inference_server.public_model_reference("..\\escape.safetensors"), "escape.safetensors")


if __name__ == "__main__":
    unittest.main()
