import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend import hires_artifacts
from backend.benchmark_lease import HIRES_ARTIFACT_PURPOSE, LEASE_PURPOSE, LEASE_PURPOSES


def isolated_environment(**overrides):
    values = {
        hires_artifacts.CAPTURE_ENV: "1",
        "XIRAI_BENCHMARK_CHILD": "1",
        "XIRAI_BENCHMARK_PURPOSE": HIRES_ARTIFACT_PURPOSE,
    }
    values.update(overrides)
    return {key: value for key, value in values.items() if value is not None}


def gate(root, environ, lease_validator, **overrides):
    values = {
        "environ": environ,
        "inference_port": 8736,
        "project_root": root,
        "state_root": root / "isolated-state",
        "output_root": root / "isolated-outputs",
        "workspace_id": "workspace",
        "lease_validator": lease_validator,
    }
    values.update(overrides)
    return hires_artifacts.capture_gate(**values)


class ArtifactGateTests(unittest.TestCase):
    def test_capture_is_off_by_default_and_the_null_capture_writes_nothing(self):
        self.assertFalse(hires_artifacts.capture_requested({}))
        self.assertFalse(hires_artifacts.capture_requested({hires_artifacts.CAPTURE_ENV: "0"}))
        self.assertFalse(hires_artifacts.capture_requested({hires_artifacts.CAPTURE_ENV: "true"}))
        self.assertTrue(hires_artifacts.capture_requested({hires_artifacts.CAPTURE_ENV: " 1 "}))

        capture = hires_artifacts.NULL_CAPTURE
        self.assertFalse(capture.enabled)
        with tempfile.TemporaryDirectory() as temporary:
            image = Image.new("RGB", (4, 4), "red")
            self.assertIsNone(capture.stage("base", image))
            self.assertIsNone(capture.tile(0, image))
            self.assertIsNone(capture.composite(0, image))
            self.assertIsNone(capture.finish(status="complete"))
            self.assertIsNone(capture.abort())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_gate_fails_closed_on_every_isolation_factor_before_touching_the_lease(self):
        calls = []

        def lease_validator():
            calls.append(1)
            return {"valid": True, "reason": "valid", "expiry": "2026-08-02T12:00:00Z", "purpose": HIRES_ARTIFACT_PURPOSE}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            failures = (
                ("capture_requested", {hires_artifacts.CAPTURE_ENV: "0"}, {}),
                ("benchmark_child", {"XIRAI_BENCHMARK_CHILD": None}, {}),
                ("purpose", {"XIRAI_BENCHMARK_PURPOSE": LEASE_PURPOSE}, {}),
                ("isolated_port", {}, {"inference_port": 8718}),
                ("isolated_state_root", {}, {"state_root": root / "state-cache"}),
                ("isolated_output_root", {}, {"output_root": root / "outputs"}),
                ("isolated_workspace", {}, {"workspace_id": "  "}),
            )
            for reason, environment, override in failures:
                with self.subTest(reason=reason):
                    result = gate(root, isolated_environment(**environment), lease_validator, **override)
                    self.assertFalse(result["enabled"])
                    self.assertEqual(result["reason"], reason)
                    self.assertFalse(result["lease_valid"])
                    self.assertIsNone(result["lease_expiry"])
            self.assertEqual(calls, [])

            invalid = gate(root, isolated_environment(), lambda: {"valid": False, "reason": "lease_expiry", "expiry": None, "purpose": None})
            self.assertFalse(invalid["enabled"])
            self.assertEqual(invalid["reason"], "lease_expiry")

            enabled = gate(root, isolated_environment(), lease_validator)
            self.assertTrue(enabled["enabled"])
            self.assertTrue(enabled["lease_valid"])
            self.assertEqual(enabled["purpose"], HIRES_ARTIFACT_PURPOSE)
            self.assertEqual(enabled["lease_expiry"], "2026-08-02T12:00:00Z")
            self.assertEqual(calls, [1])
            self.assertNotIn("nonce", json.dumps(enabled))

    def test_artifact_lease_purpose_is_separate_from_the_resident_probe_purpose(self):
        self.assertEqual(hires_artifacts.PURPOSE, "hires_quality_artifacts")
        self.assertNotEqual(HIRES_ARTIFACT_PURPOSE, LEASE_PURPOSE)
        self.assertEqual(LEASE_PURPOSES, {LEASE_PURPOSE, HIRES_ARTIFACT_PURPOSE})


class ArtifactRedactionTests(unittest.TestCase):
    def test_manifest_redaction_rejects_plaintext_secrets_and_absolute_paths(self):
        for rejected in (
            {"prompt": "text"}, {"negative_prompt": "text"}, {"nonce": "abc"}, {"lease_nonce": "abc"},
            {"command_line": ["python"]}, {"cmdline": []}, {"token": "abc"}, {"secret": "abc"},
            {"nested": {"Prompt": "text"}},
            {"stage": object()}, {1: "x"},
        ):
            with self.subTest(rejected=rejected), self.assertRaises(ValueError):
                hires_artifacts.redacted(rejected)

        # Both path flavours must be refused on both platforms: `Path.is_absolute()` alone would let
        # "/home/user/x" through on Windows and "D:\..." through on POSIX.
        for absolute in (
            "/home/user/x", "/", "//server/share",
            r"\home\user", "\\", r"\\server\share",
            "C:/x", r"D:\.XAIG\XiriaCanvas AI", "c:",
        ):
            with self.subTest(absolute=absolute):
                self.assertTrue(hires_artifacts._looks_absolute(absolute))
                with self.assertRaises(ValueError):
                    hires_artifacts.redacted({"path": absolute})

        for relative in ("tiles/00-restored.png", r"composites\00-after-composite.png", "manifest.json", "1:2", "", "a"):
            with self.subTest(relative=relative):
                self.assertFalse(hires_artifacts._looks_absolute(relative))

        allowed = {"engine": "Anima", "tile_plan": {"rows": 2, "cols": 2, "tiles": 4}, "paths": ["tiles/00-restored.png"], "denoise": 0.2, "fallback": None, "batch": True}
        self.assertEqual(hires_artifacts.redacted(allowed), allowed)

    def test_prompt_facts_expose_only_hash_and_length(self):
        facts = hires_artifacts.prompt_facts("positive text", "negative")
        self.assertEqual(set(facts), {"prompt_sha256", "prompt_length", "negative_prompt_sha256", "negative_prompt_length"})
        self.assertEqual(facts["prompt_length"], len("positive text"))
        self.assertEqual(facts["prompt_sha256"], hashlib.sha256(b"positive text").hexdigest())
        self.assertEqual(facts["negative_prompt_sha256"], hashlib.sha256(b"negative").hexdigest())
        self.assertNotIn("positive text", json.dumps(facts))
        empty = hires_artifacts.prompt_facts(None, None)
        self.assertEqual((empty["prompt_length"], empty["negative_prompt_length"]), (0, 0))
        self.assertEqual(empty["prompt_sha256"], hashlib.sha256(b"").hexdigest())

    def test_canonical_parameter_digest_is_stable_and_order_independent(self):
        first = hires_artifacts.canonical_parameter_digest({"steps": 12, "cfg": 5, "denoise": 0.2})
        second = hires_artifacts.canonical_parameter_digest({"denoise": 0.2, "cfg": 5, "steps": 12})
        self.assertEqual(first, second)
        self.assertNotEqual(first, hires_artifacts.canonical_parameter_digest({"steps": 30, "cfg": 5, "denoise": 0.2}))
        self.assertEqual(len(first), 64)
        with self.assertRaises(ValueError):
            hires_artifacts.canonical_parameter_digest({"prompt": "text"})


class ArtifactIoTests(unittest.TestCase):
    def capture_root(self, temporary):
        return Path(temporary) / "isolated-state" / "benchmark" / "hires-artifacts-run"

    def test_capture_writes_contained_atomic_artifacts_and_a_redacted_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.capture_root(temporary)
            capture = hires_artifacts.StageArtifactCapture(root)
            self.assertFalse(root.exists())

            base = capture.stage("base", Image.new("RGB", (1088, 1472), (10, 20, 30)))
            post = capture.stage("post_sr", Image.new("RGB", (2176, 2944), (40, 50, 60)))
            for index in range(4):
                capture.tile(index, Image.new("RGB", (1120, 1504), (index, 0, 0)))
                capture.composite(index, Image.new("RGB", (2176, 2944), (0, index, 0)))
            capture.stage("final", Image.new("RGB", (2176, 2944), (7, 7, 7)))

            manifest = capture.finish(
                status="complete", protocol=27, engine="Anima", execution_mode="usdu_tiled",
                base_seed="1015878324182247", hires_seed_mode="fixed", effective_hires_seed="885289963651097",
                base_dimensions=[1088, 1472], post_sr_dimensions=[2176, 2944],
                tile_plan={"rows": 2, "cols": 2, "tiles": 4, "core_tile": [1088, 1472], "processing_size": [1120, 1504], "padding": 32, "mask_blur": 8},
                canonical_parameter_sha256=hires_artifacts.canonical_parameter_digest({"steps": 12, "cfg": 5}),
                **hires_artifacts.prompt_facts("positive", "negative"),
            )

            self.assertEqual((base["width"], base["height"]), (1088, 1472))
            self.assertEqual(post["path"], hires_artifacts.POST_SR)
            self.assertEqual(len(manifest["artifacts"]), 11)
            self.assertEqual(manifest["schema_version"], hires_artifacts.SCHEMA_VERSION)
            self.assertEqual(manifest["purpose"], HIRES_ARTIFACT_PURPOSE)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(
                [item["path"] for item in manifest["artifacts"]],
                [
                    "base-decoded.png", "hires-post-sr.png",
                    "tiles/00-restored.png", "composites/00-after-composite.png",
                    "tiles/01-restored.png", "composites/01-after-composite.png",
                    "tiles/02-restored.png", "composites/02-after-composite.png",
                    "tiles/03-restored.png", "composites/03-after-composite.png",
                    "final-composited.png",
                ],
            )

            written = json.loads((root / hires_artifacts.MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(written, manifest)
            self.assertNotIn("positive", json.dumps(written))
            self.assertNotIn(str(root), json.dumps(written))
            for item in written["artifacts"]:
                path = root / item["path"]
                payload = path.read_bytes()
                self.assertEqual(item["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(item["bytes"], len(payload))
                self.assertEqual(item["pixel_mode"], "RGB")
                self.assertEqual(item["color_space"], hires_artifacts.COLOR_SPACE)
                with Image.open(path) as saved:
                    self.assertEqual((saved.width, saved.height), (item["width"], item["height"]))
            self.assertEqual([path.name for path in root.rglob(".*")], [])
            with self.assertRaises(RuntimeError):
                capture.finish(status="complete")

    def test_capture_preserves_pixels_and_records_the_rgb_output_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.capture_root(temporary)
            capture = hires_artifacts.StageArtifactCapture(root)
            source = Image.new("RGB", (8, 6))
            source.putpixel((0, 0), (1, 2, 3))
            source.putpixel((7, 5), (250, 251, 252))
            entry = capture.stage("base", source)
            self.assertEqual((entry["width"], entry["height"]), (8, 6))
            self.assertEqual(entry["source_pixel_mode"], "RGB")
            with Image.open(root / hires_artifacts.BASE_DECODED) as saved:
                self.assertEqual(saved.size, source.size)
                self.assertEqual(saved.convert("RGB").tobytes(), source.tobytes())

            rgba = capture.tile(0, Image.new("RGBA", (4, 4), (9, 9, 9, 255)))
            self.assertEqual(rgba["pixel_mode"], "RGB")
            self.assertEqual(rgba["source_pixel_mode"], "RGBA")
            capture.finish(status="complete")

    def test_capture_refuses_paths_outside_its_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.capture_root(temporary)
            capture = hires_artifacts.StageArtifactCapture(root)
            image = Image.new("RGB", (2, 2), "red")
            for relative in ("../escape.png", "tiles/../../escape.png", "/escape.png", "\\escape.png", ""):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    capture._record(relative, image)
            with self.assertRaises(ValueError):
                capture.stage("unknown", image)
            self.assertFalse(root.exists())
            self.assertEqual(capture.artifacts, [])

    def test_abort_removes_every_partial_artifact_it_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.capture_root(temporary)
            capture = hires_artifacts.StageArtifactCapture(root)
            capture.stage("base", Image.new("RGB", (4, 4), "red"))
            capture.tile(0, Image.new("RGB", (4, 4), "blue"))
            capture.composite(0, Image.new("RGB", (4, 4), "green"))
            self.assertTrue(root.is_dir())

            capture.abort()
            self.assertFalse(root.exists())
            self.assertEqual(capture.artifacts, [])
            self.assertFalse((root / hires_artifacts.MANIFEST).exists())
            with self.assertRaises(RuntimeError):
                capture.stage("final", Image.new("RGB", (4, 4), "red"))

    def test_abort_keeps_a_pre_existing_root_and_only_removes_its_own_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self.capture_root(temporary)
            root.mkdir(parents=True)
            keep = root / "unrelated.txt"
            keep.write_text("evidence", encoding="utf-8")
            capture = hires_artifacts.StageArtifactCapture(root)
            capture.stage("base", Image.new("RGB", (4, 4), "red"))
            capture.abort()
            self.assertTrue(root.is_dir())
            self.assertEqual(keep.read_text(encoding="utf-8"), "evidence")
            self.assertFalse((root / hires_artifacts.BASE_DECODED).exists())


if __name__ == "__main__":
    unittest.main()
