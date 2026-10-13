import json
import os
import tempfile
import unittest
from pathlib import Path
from backend.lora_file_identity import file_identity


class FileIdentityTests(unittest.TestCase):
    def test_real_identity_rename_copy_hardlink(self):
        base = os.environ.get("P10_EVIDENCE_DIR")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="p10-identity-", dir=base))
        source = root / "original.safetensors"
        source.write_bytes(b"identity fixture, never user weights")
        original = file_identity(source)
        self.assertIsNotNone(original, "Native filesystem identity unavailable on this test filesystem")
        moved = root / "renamed.safetensors"
        source.rename(moved)
        after = file_identity(moved)
        self.assertEqual(original["key"], after["key"])
        copied = root / "copy.safetensors"
        copied.write_bytes(moved.read_bytes())
        copy_id = file_identity(copied)
        self.assertNotEqual(after["key"], copy_id["key"])
        hardlink = root / "hardlink.safetensors"
        os.link(moved, hardlink)
        linked = file_identity(hardlink)
        self.assertEqual(after["key"], linked["key"])
        self.assertGreaterEqual(linked["links"], 2)
        (root / "identity-evidence.json").write_text(json.dumps({"platform": os.name, "original": original, "moved": after, "copy": copy_id, "hardlink": linked}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
