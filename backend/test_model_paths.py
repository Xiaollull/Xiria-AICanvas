import json
import tempfile
import unittest
from pathlib import Path

from backend.model_paths import load_model_paths, resolve_model_directory


class ModelPathTests(unittest.TestCase):
    def test_missing_manifest_uses_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(resolve_model_directory(root, "upscalers"), root / "models" / "upscalers")
            self.assertEqual(resolve_model_directory(root, "loras", "anima"), root / "models" / "loras" / "anima")

    def test_custom_unicode_and_nested_roots_are_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models" / "model-paths.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({
                "checkpoints": {"sd": "models/My Models/SD/deep"},
                "loras": {"anima": "models/自定义/Anima LoRA/多层"},
                "yolo": "models/tools/detectors/custom",
            }), encoding="utf-8")
            self.assertEqual(resolve_model_directory(root, "checkpoints", "sd"), root / "models" / "My Models" / "SD" / "deep")
            self.assertEqual(resolve_model_directory(root, "loras", "anima"), root / "models" / "自定义" / "Anima LoRA" / "多层")
            self.assertEqual(resolve_model_directory(root, "yolo"), root / "models" / "tools" / "detectors" / "custom")

    def test_invalid_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models" / "model-paths.json"
            manifest.parent.mkdir(parents=True)
            for invalid in ("", "models", "../outside", str(root / "absolute")):
                manifest.write_text(json.dumps({"upscalers": invalid}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    resolve_model_directory(root, "upscalers")

    def test_invalid_manifest_types_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models" / "model-paths.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"checkpoints":"models/checkpoints"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_paths(root)


if __name__ == "__main__":
    unittest.main()
