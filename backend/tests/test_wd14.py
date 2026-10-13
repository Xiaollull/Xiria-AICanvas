import io
import hashlib
import threading
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend import wd14
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi import HTTPException


class Input:
    name = "image"
    shape = [1, 8, 8, 3]


class Output:
    name = "tags"
    shape = [1, 7]


class Session:
    def get_inputs(self): return [Input()]
    def get_outputs(self): return [Output()]
    def run(self, names, arguments):
        import numpy as np
        assert arguments["image"].shape == (1, 8, 8, 3)
        return [np.array([[.1, .8, .05, .05, .36, .84, .86]], dtype=np.float32)]


class WD14Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Hermetic official-manifest stand-in: validate the fake transferred bytes
        # against a known digest, rather than weakening installed() to format-only.
        from backend.tagger_registry import Artifact
        content = {"model.onnx": b"\x08\x01", "selected_tags.csv": b"name,category\n" + b"tag,0\n" * 5}
        spec = {"repo": "SmilingWolf/test", "revision": "a" * 40,
                "artifacts": tuple(Artifact(name, name, len(data), hashlib.sha256(data).hexdigest()) for name, data in content.items())}
        manifest = patch.dict(wd14.WD_MODELS, {wd14.DEFAULT: spec})
        manifest.start()
        self.addCleanup(manifest.stop)

    def test_tag_preprocess_rating_threshold_and_sort(self):
        rgba = Image.new("RGBA", (8, 4), (255, 0, 0, 0))
        content = io.BytesIO()
        rgba.save(content, format="PNG")
        rows = [("safe", "9"), ("questionable", "9"), ("explicit", "9"), ("sensitive", "9"),
                ("one_tag", "0"), ("below_character", "4"), ("above_character", "4")]
        with patch.object(wd14, "_runtime", return_value=(Session(), rows, 8)):
            result = wd14.tag_image(content.getvalue(), wd14.DEFAULT, .35, .85, 2)
            self.assertEqual([tag["name"] for tag in result["tags"]], ["above character", "one tag"])
            self.assertAlmostEqual(result["rating"]["questionable"], .8)
            self.assertEqual(wd14.tag_image(content.getvalue(), wd14.DEFAULT, .35, .85, 2, True)["tags"][0]["name"], "above character")
        with self.assertRaises(ValueError):
            wd14.tag_image(b"x" * (wd14.MAX_BYTES + 1), wd14.DEFAULT, .35, .85, 20)

    def test_batch_directory_and_symlink_boundary(self):
        source = self.root / "photos"
        source.mkdir()
        (source / "a.png").write_bytes(b"png")
        (source / "a.txt").write_text("before", encoding="utf-8")
        self.assertEqual(wd14._images(source, False, self.root), [source / "a.png"])
        with self.assertRaises(ValueError):
            wd14._folder(str(Path(self.root.anchor)))
        with self.assertRaises(ValueError):
            wd14._folder(str(Path.home()))
        self.assertEqual(wd14.Batch(folder=str(source)).conflict, "ignore")
        for mode, expected in [("ignore", "before"), ("append", "before, hello"),
                               ("prepend", "hello, before"), ("copy", "hello")]:
            (source / "a.txt").write_text("before", encoding="utf-8")
            wd14._caption(source / "a.txt", "hello", mode, self.root)
            self.assertEqual((source / "a.txt").read_text(encoding="utf-8"), expected)
        link = source / "linked.txt"
        try:
            link.symlink_to(source / "a.txt")
        except OSError:
            pass  # Windows requires symlink permission on some installations.
        else:
            with self.assertRaises(ValueError):
                wd14._caption(link, "evil", "copy", self.root)
            self.assertEqual((source / "a.txt").read_text(encoding="utf-8"), "hello")
        directory_link = self.root / "junction"
        try:
            directory_link.symlink_to(source, target_is_directory=True)
        except OSError:
            pass
        else:
            with self.assertRaises(ValueError):
                wd14._folder(str(directory_link))

    def test_caption_rechecks_target_after_temp_before_replace(self):
        folder = self.root / "photos"
        folder.mkdir()
        target = folder / "a.txt"
        target.write_text("old")
        victim = self.root / "victim.txt"
        victim.write_text("outside")
        original = wd14._local_path
        calls = 0
        def swapped(path):
            nonlocal calls
            if Path(path) == target:
                calls += 1
                if calls == 2:
                    target.unlink()
                    target.symlink_to(victim)
            return original(path)
        try:
            with patch.object(wd14, "_local_path", side_effect=swapped):
                with self.assertRaises((ValueError, OSError)):
                    wd14._caption(target, "new", "copy")
        except OSError:  # Creating Windows symlinks can require additional privileges.
            self.skipTest("symlinks unavailable")
        finally:
            if target.is_symlink(): target.unlink()
        self.assertEqual(victim.read_text(), "outside")
        self.assertFalse(any(folder.glob(".wd14-*")))

    def test_batch_progress_cancel_and_failure(self):
        folder = self.root / "photos"
        folder.mkdir()
        (folder / "a.png").write_bytes(b"image")
        (folder / "b.png").write_bytes(b"image")
        (self.root / "outputs").mkdir()
        with patch.object(wd14, "OUTPUT_ROOT", self.root / "outputs"), patch.object(wd14, "_runtime"), \
             patch.object(wd14, "tag_image", return_value={"tags": [{"name": "cat"}]}):
            job = wd14.batch(wd14.Batch(folder=str(folder)))
            self._wait(job["id"])
            self.assertEqual(wd14.status(job["id"])["done"], 2)
            self.assertEqual((folder / "a.txt").read_text(), "cat")
            repeat = wd14.batch(wd14.Batch(folder=str(folder)))
            self._wait(repeat["id"])
            self.assertEqual(wd14.status(repeat["id"])["skipped"], 2)
            self.assertEqual((folder / "a.txt").read_text(), "cat")
            nested = folder / "sub"
            nested.mkdir()
            (nested / "nested.png").write_bytes(b"image")
            dest = self.root / "captions"
            dest.mkdir()
            mirrored = wd14.batch(wd14.Batch(folder=str(folder), output_folder=str(dest), recursive=True, conflict="copy"))
            self._wait(mirrored["id"])
            self.assertEqual(wd14.status(mirrored["id"])["done"], 3)
            self.assertEqual((dest / "sub" / "nested.txt").read_text(), "cat")
            with self.assertRaises(Exception):
                wd14.batch(wd14.Batch(folder=str(folder), output_folder=str(Path(self.root.anchor))))
            def interrupted(*args):
                wd14.cancel(current["id"])
                return {"tags": []}
            (folder / "a.txt").unlink()
            (folder / "b.txt").unlink()
            with patch.object(wd14, "tag_image", side_effect=interrupted):
                current = wd14.batch(wd14.Batch(folder=str(folder)))
                self._wait(current["id"])
                self.assertEqual(wd14.status(current["id"])["state"], "cancelled")
                self.assertFalse((folder / "a.txt").exists())

    def test_download_mocked_metadata_publish_and_cancel(self):
        weights = self.root / "weights"
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a"*40,
                                                         {"model.onnx": (2, "b"*64), "selected_tags.csv": (2, "c"*40)})), \
             patch.object(wd14, "_download_one", side_effect=lambda repo, commit, name, expected, target, job: target.write_bytes(b"\x08\x01" if name == "model.onnx" else b"name,category\n" + b"tag,0\n"*5)), \
             patch.object(wd14, "_tags", return_value=[("x", "0")]*5):
            job = wd14.download(wd14.DEFAULT)
            self._wait(job["id"])
            self.assertEqual(wd14.status(job["id"])["state"], "done")
            self.assertTrue((weights / wd14.DEFAULT / "model.onnx").is_file())
            self.assertTrue((weights / wd14.DEFAULT / "selected_tags.csv").is_file())
            self.assertTrue(wd14.installed(wd14.DEFAULT))
            self.assertFalse(list((weights / wd14.DEFAULT).glob(".wd14-*-old-*")))
            self.assertFalse(list((weights / wd14.DEFAULT).glob(".wd14-stage-*")))
            # Labels do not depend on the transient cache being present.
            self.assertFalse((self.root / ".cache").exists())
            self.assertTrue(wd14.installed(wd14.DEFAULT))

    def test_gitignore_matches_only_generated_wd14_runtime_names(self):
        repo = Path(__file__).resolve().parents[2]
        job_id = "0123456789abcdef0123456789abcdef"
        ignored = [
            "models/wd14/example/model.onnx",
            "models/wd14/example/selected_tags.csv",
            "models/wd14/example/.wd14-installing",
            "models/wd14/example/.wd14-stage-ab12_cd3/",
            f"models/wd14/example/.wd14-{job_id}-model.onnx.part",
            f"models/wd14/example/.wd14-{job_id}-selected_tags.csv.part",
            f"models/wd14/example/.wd14-{job_id}-old-model.onnx",
            f"models/wd14/example/.wd14-{job_id}-old-selected_tags.csv",
        ]
        for path in ignored:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "-v", "--no-index", "--", path],
                    cwd=repo, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        user_note = "models/wd14/example/.wd14-random-user-note.txt"
        result = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", "--", user_note],
            cwd=repo, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_cancel_before_publish_leaves_original_files_untouched(self):
        weights = self.root / "weights"
        previous = weights / wd14.DEFAULT
        previous.mkdir(parents=True)
        (previous / "model.onnx").write_bytes(b"old-invalid-model")
        (previous / "selected_tags.csv").write_bytes(b"old-tags")
        reached = threading.Event()
        release = threading.Event()
        def pause_tags(path):
            if str(path).endswith(".part"):
                reached.set()
                self.assertTrue(release.wait(5))
            return [("tag", "0")] * 5
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download), \
             patch.object(wd14, "_tags", side_effect=pause_tags):
            job = wd14.download(wd14.DEFAULT)
            self.assertTrue(reached.wait(5))
            self.assertEqual(wd14.cancel(job["id"]), {"ok": True})
            release.set()
            self._wait(job["id"])
            self.assertEqual(wd14.status(job["id"])["state"], "cancelled")
            self.assertFalse((previous / ".wd14-installing").exists())
            self.assertFalse(list(previous.glob(".wd14-stage-*")))
            self.assertEqual((previous / "model.onnx").read_bytes(), b"old-invalid-model")
            self.assertEqual((previous / "selected_tags.csv").read_bytes(), b"old-tags")

    def test_cancel_during_publish_is_refused_and_commits_done(self):
        weights = self.root / "weights"
        reached, release = threading.Event(), threading.Event()
        replace = wd14.os.replace
        def pause_csv(source, target):
            if Path(target).name == "selected_tags.csv":
                reached.set()
                self.assertTrue(release.wait(5))
            return replace(source, target)
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download), \
             patch.object(wd14, "_tags", return_value=[("tag", "0")] * 5), \
             patch.object(wd14.os, "replace", side_effect=pause_csv):
            job = wd14.download(wd14.DEFAULT)
            self.assertTrue(reached.wait(5))
            self.assertEqual(wd14.status(job["id"])["state"], "publishing")
            with self.assertRaises(HTTPException) as caught:
                wd14.cancel(job["id"])
            self.assertEqual(caught.exception.status_code, 409)
            release.set()
            self._wait(job["id"])
            self.assertEqual(wd14.status(job["id"])["state"], "done")
            self.assertTrue(wd14.installed(wd14.DEFAULT))
            self.assertEqual(wd14.cancel(job["id"]), {"ok": False, "state": "done"})

    def test_second_file_publish_failure_rolls_back_original_pair(self):
        weights = self.root / "weights"
        previous = weights / wd14.DEFAULT
        previous.mkdir(parents=True)
        (previous / "model.onnx").write_bytes(b"old-invalid-model")
        (previous / "selected_tags.csv").write_bytes(b"old-tags")
        replace = wd14.os.replace
        def fail_model_once(source, target):
            if Path(target).name == "model.onnx" and str(source).endswith(".part"):
                raise OSError("simulated model publication failure")
            return replace(source, target)
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download), \
             patch.object(wd14, "_tags", return_value=[("tag", "0")] * 5), \
             patch.object(wd14.os, "replace", side_effect=fail_model_once):
            job = wd14.download(wd14.DEFAULT)
            self._wait(job["id"])
            self.assertEqual(wd14.status(job["id"])["state"], "error")
            self.assertFalse(wd14.installed(wd14.DEFAULT))
            self.assertEqual((previous / "model.onnx").read_bytes(), b"old-invalid-model")
            self.assertEqual((previous / "selected_tags.csv").read_bytes(), b"old-tags")
            self.assertFalse((previous / ".wd14-installing").exists())
            self.assertFalse(list(previous.glob(".wd14-*-old-*")))
            # Retrying after a fully restored failure publishes a matched pair.
            with patch.object(wd14.os, "replace", replace):
                recovered = wd14.download(wd14.DEFAULT)
                self._wait(recovered["id"])
                self.assertEqual(wd14.status(recovered["id"])["state"], "done")
                self.assertTrue(wd14.installed(wd14.DEFAULT))

    def test_failed_rollback_keeps_marker_and_prevents_false_installation(self):
        weights = self.root / "weights"
        previous = weights / wd14.DEFAULT
        previous.mkdir(parents=True)
        (previous / "model.onnx").write_bytes(b"\x08\x01")
        (previous / "selected_tags.csv").write_bytes(b"invalid-old-csv")
        original_tags = wd14._tags
        replace = wd14.os.replace
        first_job_id = None
        def fail_model_and_rollback(source, target):
            if Path(target).name == "model.onnx" and str(source).endswith(".part"):
                raise OSError("simulated publication failure")
            if Path(target).name == "selected_tags.csv" and str(source).endswith(".old"):
                raise OSError("simulated rollback failure")
            return replace(source, target)
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download), \
             patch.object(wd14, "_tags", side_effect=lambda p: [("tag", "0")] * 5 if str(p).endswith(".part") else original_tags(p)), \
             patch.object(wd14.os, "replace", side_effect=fail_model_and_rollback):
            job = wd14.download(wd14.DEFAULT)
            first_job_id = job["id"]
            self._wait(job["id"])
            self.assertEqual(wd14.status(job["id"])["state"], "error")
            self.assertTrue((previous / ".wd14-installing").exists())
            self.assertFalse(wd14.installed(wd14.DEFAULT))
            self.assertEqual((previous / "model.onnx").read_bytes(), b"\x08\x01")
            stages = list(previous.glob(".wd14-stage-*"))
            self.assertEqual(len(stages), 1)
            orphan_backups = [stages[0] / "selected_tags.csv.old", stages[0] / "model.onnx.old"]
            self.assertTrue(all(path.is_file() for path in orphan_backups))
            user_note = previous / f".wd14-{first_job_id}-old-user-note.txt"
            user_note.write_text("user data", encoding="utf-8")
            with patch.object(wd14.os, "replace", side_effect=replace):
                recovered = wd14.download(wd14.DEFAULT)
                self._wait(recovered["id"])
                recovered_status = wd14.status(recovered["id"])
                self.assertEqual(recovered_status["state"], "error")
                self.assertIn("发现未完成安装，须手工核查/修复", recovered_status["error"])
            self.assertTrue((previous / ".wd14-installing").exists())
            self.assertTrue(all(path.is_file() for path in orphan_backups))
            self.assertEqual(user_note.read_text(encoding="utf-8"), "user data")

    def test_preexisting_marker_and_user_backups_fail_closed_without_reading_or_deleting(self):
        weights = self.root / "weights"
        previous = weights / wd14.DEFAULT
        previous.mkdir(parents=True)
        job_id = "a" * 32
        marker = previous / ".wd14-installing"
        marker.write_text(job_id, encoding="ascii")
        backups = [previous / f".wd14-{job_id}-old-{name}"
                   for name in ("selected_tags.csv", "model.onnx")]
        for index, backup in enumerate(backups):
            backup.write_bytes(f"user-owned-{index}".encode())
        original_read_text = Path.read_text
        def unreadable_marker(path, *args, **kwargs):
            if path == marker:
                raise PermissionError("simulated unreadable marker")
            return original_read_text(path, *args, **kwargs)
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download) as download_one, \
             patch.object(wd14, "_tags", return_value=[("tag", "0")] * 5), \
             patch.object(Path, "read_text", unreadable_marker):
            job = wd14.download(wd14.DEFAULT)
            self._wait(job["id"])
            status = wd14.status(job["id"])
        self.assertEqual(status["state"], "error")
        self.assertIn("发现未完成安装，须手工核查/修复", status["error"])
        self.assertEqual(download_one.call_count, 2)
        self.assertEqual(marker.read_text(encoding="ascii"), job_id)
        self.assertEqual([backup.read_bytes() for backup in backups], [b"user-owned-0", b"user-owned-1"])

    def test_public_job_id_part_collision_is_preserved_while_install_succeeds(self):
        weights = self.root / "weights"
        model_dir = weights / wd14.DEFAULT
        model_dir.mkdir(parents=True)
        job_id = "b" * 32
        reserved = model_dir / f".wd14-{job_id}-model.onnx.part"
        reserved.write_bytes(b"user-owned public-name collision")
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14.uuid, "uuid4", return_value=SimpleNamespace(hex=job_id)), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download), \
             patch.object(wd14, "_tags", return_value=[("tag", "0")] * 5):
            job = wd14.download(wd14.DEFAULT)
            self._wait(job["id"])
            status = wd14.status(job["id"])
        self.assertEqual(status["state"], "done")
        with patch.object(wd14, "ROOT", weights):
            self.assertTrue(wd14.installed(wd14.DEFAULT), status)
        self.assertEqual(reserved.read_bytes(), b"user-owned public-name collision")
        self.assertFalse(list(model_dir.glob(".wd14-stage-*")))

    def test_replaced_private_backup_is_not_unlinked_and_warns_after_success(self):
        weights = self.root / "weights"
        model_dir = weights / wd14.DEFAULT
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_bytes(b"old-onnx")
        (model_dir / "selected_tags.csv").write_bytes(b"old-csv")
        replace = wd14.os.replace
        swapped = self.root / "user-backup-replacement.bin"
        swapped.write_bytes(b"do not delete")
        swapped_path = None
        swapped_once = False
        def swap_backup_before_publish(source, target):
            nonlocal swapped_path, swapped_once
            source = Path(source)
            if not swapped_once and source.name == "selected_tags.csv.part":
                stage = source.parent
                swapped_path = stage / "model.onnx.old"
                swapped_path.unlink()
                swapped_path.write_bytes(b"user replacement after hardlink")
                swapped_once = True
            return replace(source, target)
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download), \
             patch.object(wd14, "_tags", return_value=[("tag", "0")] * 5), \
             patch.object(wd14.os, "replace", side_effect=swap_backup_before_publish):
            job = wd14.download(wd14.DEFAULT)
            self._wait(job["id"])
            status = wd14.status(job["id"])
        self.assertEqual(status["state"], "done")
        with patch.object(wd14, "ROOT", weights):
            self.assertTrue(wd14.installed(wd14.DEFAULT), status)
        self.assertIn("未删除替代文件", status["warning"])
        self.assertEqual(swapped_path.read_bytes(), b"user replacement after hardlink")
        self.assertTrue(swapped_path.parent.exists())

    def test_success_surfaces_manual_review_warning_for_old_backup_candidates(self):
        weights = self.root / "weights"
        model_dir = weights / wd14.DEFAULT
        model_dir.mkdir(parents=True)
        old = model_dir / (".wd14-" + "c" * 32 + "-old-model.onnx")
        old.write_bytes(b"unverified historical file")
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a" * 40,
                 {"model.onnx": (2, "b" * 64), "selected_tags.csv": (2, "c" * 40)})), \
             patch.object(wd14, "_download_one", side_effect=self._fake_download), \
             patch.object(wd14, "_tags", return_value=[("tag", "0")] * 5):
            job = wd14.download(wd14.DEFAULT)
            self._wait(job["id"])
            status = wd14.status(job["id"])
        self.assertEqual(status["state"], "done")
        self.assertIn("程序不会自动清理；请手工核查/修复", status["warning"])
        self.assertTrue(any("程序不会自动清理" in line for line in status["logs"]))
        self.assertEqual(old.read_bytes(), b"unverified historical file")

    @staticmethod
    def _fake_download(repo, commit, name, expected, target, job):
        target.write_bytes(b"\x08\x01" if name == "model.onnx" else b"name,category\n" + b"tag,0\n" * 5)

    def test_http_size_registry_and_cancelled_download_retry(self):
        app = FastAPI()
        app.include_router(wd14.router)
        client = TestClient(app, base_url="http://127.0.0.1:8718", client=("127.0.0.1", 50000),
                            headers={"Origin": "http://127.0.0.1:8718"})
        self.assertEqual(client.post("/api/inference/wd14/download/foreign").status_code, 422)
        self.assertEqual(client.post("/api/inference/wd14/tag", content=b"x" * (wd14.MAX_BYTES + 1)).status_code, 413)
        weights = self.root / "weights"
        def interrupted(repo, commit, name, expected, target, job):
            target.write_bytes(b"partial")
            wd14._record_stage_file(job, target)
            job["cancel"].set()
            raise ValueError("cancelled")
        with patch.object(wd14, "ROOT", weights), \
             patch.object(wd14, "_metadata", return_value=("SmilingWolf/test", "a"*40,
                                                          {"model.onnx": (7, "b"*64), "selected_tags.csv": (5, "c"*40)})), \
             patch.object(wd14, "_download_one", side_effect=interrupted):
            first = client.post(f"/api/inference/wd14/download/{wd14.DEFAULT}")
            self.assertEqual(first.status_code, 202)
            self._wait(first.json()["id"])
            self.assertEqual(client.get(f"/api/inference/wd14/jobs/{first.json()['id']}").json()["state"], "cancelled")
            self.assertFalse((weights / wd14.DEFAULT / "model.onnx").exists())
            self.assertFalse(any((weights / wd14.DEFAULT).iterdir()))
            retry = client.post(f"/api/inference/wd14/download/{wd14.DEFAULT}")
            self.assertEqual(retry.status_code, 202)
            self._wait(retry.json()["id"])

    def test_direct_python_origin_and_remote_access(self):
        app = FastAPI()
        app.include_router(wd14.router)
        def client(peer, origin=None):
            return TestClient(app, base_url="http://127.0.0.1:8718", client=(peer, 50001),
                              headers={"Origin": origin} if origin else {})
        for peer, origin in [("127.0.0.1", None), ("127.0.0.1", "http://evil.example"),
                             ("10.0.0.8", "http://127.0.0.1:8718")]:
            for endpoint in ("batch", f"download/{wd14.DEFAULT}", "jobs/any/cancel", "tag"):
                self.assertEqual(client(peer, origin).post("/api/inference/wd14/" + endpoint).status_code, 403)
        local = client("127.0.0.1", "http://127.0.0.1:8718")
        self.assertEqual(local.post("/api/inference/wd14/batch", json={"folder": str(self.root.parent)}).status_code, 422)
        with patch.dict("os.environ", {"XIRAI_WD14_PROXY_PROOF": "a" * 64}):
            spoofed = client("127.0.0.1", "http://evil.example")
            self.assertEqual(spoofed.post("/api/inference/wd14/batch", headers={"x-xirai-wd14-proxy-proof": "bad"}, json={"folder": str(self.root)}).status_code, 403)

    def test_download_stream_size_digest_and_redirect_policy(self):
        class Response:
            status_code = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): pass
            def iter_content(self, size): yield b"hello"
        class Redirect(Response):
            status_code = 302
            headers = {"Location": "https://127.0.0.1/private"}
        class Job(dict):
            pass
        from threading import Event
        job = Job(done=0, cancel=Event())
        destination = self.root / "model.part"
        with patch("huggingface_hub.hf_hub_url", return_value="https://huggingface.co/SmilingWolf/test/resolve/abc/model.onnx"), \
             patch("requests.Session.get", return_value=Response()):
            wd14._download_one("SmilingWolf/test", "a"*40, "model.onnx", (5, hashlib.sha256(b"hello").hexdigest()), destination, job)
            self.assertEqual(destination.read_bytes(), b"hello")
            self.assertEqual(job["done"], 5)
        destination.unlink()
        with patch("huggingface_hub.hf_hub_url", return_value="https://huggingface.co/SmilingWolf/test/resolve/abc/model.onnx"), \
             patch("requests.Session.get", return_value=Redirect()):
            with self.assertRaises(ValueError):
                wd14._download_one("SmilingWolf/test", "a"*40, "model.onnx", (5, "a"*64), destination, job)
            self.assertFalse(destination.exists())
        with patch("huggingface_hub.hf_hub_url", return_value="https://huggingface.co/SmilingWolf/test/resolve/abc/model.onnx"), \
             patch("requests.Session.get", return_value=Response()):
            with self.assertRaises(ValueError):
                wd14._download_one("SmilingWolf/test", "a"*40, "model.onnx", (4, "a"*64), destination, job)

    def _wait(self, job_id):
        wd14._jobs[job_id]["thread"].join(10)
        self.assertFalse(wd14._jobs[job_id]["thread"].is_alive())


if __name__ == "__main__":
    unittest.main()
