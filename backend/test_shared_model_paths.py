import json
import os
import tempfile
import unittest
from pathlib import Path

from backend.shared_model_paths import (
    SHARED_PATHS_CASE_INSENSITIVE,
    classify_directory_name,
    load_shared_roots,
    normalize_shared_root_path,
    normalize_shared_roots,
    parse_shared_ref,
    resolve_shared_model,
    shared_kind_directories,
    shared_model_reference,
    shared_root_id,
)


def write_config(root: Path, roots):
    manifest = root / "models" / "shared-paths.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"version": 1, "roots": roots}), encoding="utf-8")


class SharedReferenceTests(unittest.TestCase):
    def test_reference_round_trip_and_escapes(self):
        self.assertEqual(parse_shared_ref("shared:0123456789ab/a/b.safetensors"), ("0123456789ab", "a/b.safetensors"))
        self.assertEqual(shared_model_reference("shared:0123456789ab/a/b.safetensors"), "shared:0123456789ab/a/b.safetensors")
        for invalid in (
            "character/hero.safetensors",
            "shared:0123456789ab/../../secrets.env",
            "shared:0123456789ab/./x.safetensors",
            "shared:0123456789ab/",
            "shared:0123456789ab",
            "shared:/x.safetensors",
            "shared:XYZ456789abc/x.safetensors",
            "shared:0123456789/x.safetensors",
            None,
            42,
        ):
            self.assertIsNone(parse_shared_ref(invalid), invalid)

    def test_root_ids_match_the_node_twin(self):
        # Pinned against scripts/shared-model-paths.test.mjs. If these two ever
        # disagree, every mounted shared LoRA fails to resolve at generation.
        # The folding mode is passed explicitly so the vectors hold on either host.
        self.assertEqual(shared_root_id("F:\\AI\\Ai SD\\sd-webui-aki-v4.10\\models\\Lora", True), "95746e6dad38")
        self.assertEqual(shared_root_id("D:\\.XAIG\\XiriaCanvas AI\\models", True), "5dff1a102833")
        self.assertEqual(
            normalize_shared_root_path("F:\\AI\\Ai SD\\sd-webui-aki-v4.10\\models\\Lora"),
            "F:/AI/Ai SD/sd-webui-aki-v4.10/models/Lora",
        )
        self.assertEqual(normalize_shared_root_path("/srv/models/Lora//"), "/srv/models/Lora")
        for spelling in (
            "F:/AI/Ai SD/sd-webui-aki-v4.10/models/Lora",
            "f:\\ai\\ai sd\\sd-webui-aki-v4.10\\models\\lora\\",
            "F:\\AI\\Ai SD\\sd-webui-aki-v4.10\\models\\\\Lora",
        ):
            self.assertEqual(shared_root_id(spelling, True), "95746e6dad38", spelling)

    def test_case_only_decides_identity_where_the_filesystem_says_it_does(self):
        # On Linux these are two real directories; folding them would give both
        # the same id and the second registration would be dropped as a
        # duplicate. On Windows they are one folder typed two ways.
        self.assertNotEqual(shared_root_id("/srv/models/Lora", False), shared_root_id("/srv/models/lora", False))
        self.assertEqual(shared_root_id("F:\\Models\\Lora", True), shared_root_id("f:\\models\\lora", True))
        self.assertEqual(SHARED_PATHS_CASE_INSENSITIVE, os.name == "nt")
        # The default must track the host, and must match the Node twin's default.
        self.assertEqual(
            shared_root_id("/srv/models/Lora"),
            shared_root_id("/srv/models/Lora", SHARED_PATHS_CASE_INSENSITIVE),
        )

    def test_directory_names_classify_case_insensitively(self):
        self.assertEqual(classify_directory_name("Stable-diffusion"), "checkpoints")
        self.assertEqual(classify_directory_name("Lora"), "loras")
        self.assertEqual(classify_directory_name("loras"), "loras")
        self.assertEqual(classify_directory_name("random-folder"), "")


class SharedRootConfigTests(unittest.TestCase):
    def test_missing_config_is_the_default_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(load_shared_roots(Path(temporary)), [])

    def test_damaged_entries_are_dropped_and_ids_dedupe(self):
        roots = normalize_shared_roots({"roots": [
            {"id": "95746e6dad38", "path": "F:/AI/Lora", "enabled": False, "engines": ["SD"]},
            {"id": "95746e6dad38", "path": "F:/AI/Duplicate"},
            {"id": "not-an-id", "path": "F:/AI/Bad"},
            {"id": "5dff1a102833", "path": ""},
            {"id": "5dff1a102833", "path": "D:/models", "engines": []},
            "nonsense",
        ]})
        self.assertEqual([root["id"] for root in roots], ["95746e6dad38", "5dff1a102833"])
        self.assertFalse(roots[0]["enabled"])
        self.assertEqual(roots[0]["engines"], ["SD"])
        # An unset list is every engine, which now includes both Flux generations and Krea 2.
        self.assertEqual(roots[1]["engines"], ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"])

    def test_invalid_config_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models" / "shared-paths.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_shared_roots(root)
            manifest.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_shared_roots(root)


class SharedResolutionTests(unittest.TestCase):
    def test_shared_files_resolve_and_stay_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "elsewhere" / "Lora"
            (shared / "character").mkdir(parents=True)
            (shared / "character" / "hero.safetensors").write_bytes(b"x")
            (root / "secret").mkdir()
            (root / "secret" / "keys.safetensors").write_bytes(b"x")

            root_id = shared_root_id(str(shared.resolve()))
            write_config(root, [{"id": root_id, "path": str(shared), "kind": "loras", "engines": ["SD"]}])
            extensions = {".safetensors"}

            resolved = resolve_shared_model(f"shared:{root_id}/character/hero.safetensors", root, extensions)
            self.assertEqual(resolved.name, "hero.safetensors")

            for bad, pattern in (
                (f"shared:{root_id}/../secret/keys.safetensors", "malformed"),
                ("shared:ffffffffffff/character/hero.safetensors", "not registered"),
                (f"shared:{root_id}/character/hero.exe", "not supported"),
                (f"shared:{root_id}/character/missing.safetensors", "does not exist"),
                ("character/hero.safetensors", "malformed"),
            ):
                with self.assertRaisesRegex(ValueError, pattern):
                    resolve_shared_model(bad, root, extensions)

    def test_disabled_roots_refuse_to_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "Lora"
            shared.mkdir()
            (shared / "hero.safetensors").write_bytes(b"x")
            root_id = shared_root_id(str(shared.resolve()))
            write_config(root, [{"id": root_id, "path": str(shared), "kind": "loras", "enabled": False}])
            with self.assertRaisesRegex(ValueError, "disabled"):
                resolve_shared_model(f"shared:{root_id}/hero.safetensors", root, {".safetensors"})

    @unittest.skipIf(os.name == "nt", "symlink creation needs privileges on Windows")
    def test_symlinks_cannot_escape_the_registered_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "Lora"
            shared.mkdir()
            (root / "secret").mkdir()
            (root / "secret" / "keys.safetensors").write_bytes(b"x")
            (shared / "escape").symlink_to(root / "secret", target_is_directory=True)
            root_id = shared_root_id(str(shared.resolve()))
            write_config(root, [{"id": root_id, "path": str(shared), "kind": "loras"}])
            with self.assertRaisesRegex(ValueError, "outside its registered shared directory"):
                resolve_shared_model(f"shared:{root_id}/escape/keys.safetensors", root, {".safetensors"})

    def test_kind_directories_follow_the_folder_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "ComfyUI" / "models"
            (shared / "checkpoints").mkdir(parents=True)
            (shared / "loras").mkdir()
            auto = {"id": "0123456789ab", "path": str(shared), "kind": "auto", "enabled": True, "engines": ["SD"]}
            self.assertEqual([entry.name for entry in shared_kind_directories(auto, "loras")], ["loras"])
            self.assertEqual([entry.name for entry in shared_kind_directories(auto, "checkpoints")], ["checkpoints"])

            leaf = {"id": "0123456789ab", "path": str(shared / "loras"), "kind": "loras", "enabled": True, "engines": ["SD"]}
            self.assertEqual(shared_kind_directories(leaf, "loras"), [(shared / "loras").resolve()])
            self.assertEqual(shared_kind_directories(leaf, "checkpoints"), [])
            self.assertEqual(shared_kind_directories({**leaf, "path": str(root / "gone")}, "loras"), [])


if __name__ == "__main__":
    unittest.main()
