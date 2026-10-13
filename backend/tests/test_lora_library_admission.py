"""Actual job API admission with isolated files; executor mocked, never GPU."""
import hashlib
import json
import os
import struct
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.lora_file_identity import file_identity
from backend.lora_library_gate import library_admission, LibraryGateError


class LibraryAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import inference_server
        cls.server = inference_server

    def setUp(self):
        parent = os.environ.get("P10_EVIDENCE_DIR")
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="c-job-", dir=parent))
        self.state = self.root / "state"
        self.models = self.root / "models"
        self.models.mkdir()
        self.state.mkdir()
        self.file = self.models / "A.safetensors"
        header = json.dumps({"__metadata__": {}}).encode()
        self.file.write_bytes(struct.pack("<Q", len(header)) + header + b"original adapter bytes")
        self.id = str(uuid.uuid4())
        self.instance = {"instanceId": self.id, "engine": "SD", "rootId": "local-SD", "value": self.file.name,
                         "history": [], "status": "present", "contentRevision": 1,
                         "contentHash": hashlib.sha256(self.file.read_bytes()).hexdigest(), "fsIdentity": file_identity(self.file)}
        self.registry = {"schemaVersion": 1, "revision": 1, "instances": {self.id: self.instance}, "substitutions": []}
        (self.state / "lora-library").mkdir()
        self.write("registry.json", self.registry)
        self.write("publication.json", {"schemaVersion": 1, "phase": "published", "revision": 1})
        self.created = []
        self.stack = mock.patch.object(self.server, "STATE_DIRECTORY", self.state)
        self.stack.start()
        self.roots = mock.patch.object(self.server, "model_roots", return_value=(self.models, self.models))
        self.roots.start()

    def tearDown(self):
        self.stack.stop()
        self.roots.stop()
        with self.server.jobs_lock:
            for job in self.created:
                self.server.jobs.pop(job, None)
                self.server.job_controls.pop(job, None)

    def write(self, name, value):
        (self.state / "lora-library" / name).write_text(json.dumps(value), encoding="utf-8")

    def request(self, **overrides):
        return self.server.GenerateInput(**{**dict(engine="SD", checkpoint="checkpoint.safetensors", prompt="cat", width=512, height=512,
                                                   steps=4, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="normal"), **overrides})

    def http(self, context=None, revision="1"):
        return SimpleNamespace(headers={"x-xirai-lora-revision": revision,
                                        "x-xirai-lora-context": json.dumps(context or [])})

    def submit(self, request, http=None):
        with mock.patch.object(self.server.executor, "submit") as executor:
            response = self.server.create_job(request, http)
            self.created.append(response["id"])
            return response, executor.call_args.args

    def assert_refused(self, request, http=None):
        before = set(self.server.jobs)
        with mock.patch.object(self.server.executor, "submit") as executor:
            with self.assertRaises(self.server.HTTPException) as raised:
                self.server.create_job(request, http)
            self.assertEqual(raised.exception.status_code, 409)
            executor.assert_not_called()
        self.assertEqual(set(self.server.jobs), before)

    def test_actual_inline_preflight_is_validated_before_any_queued_record(self):
        # Same name is now different bytes. Incoming mounts are EMPTY: the old
        # implementation validated nothing, then added this adapter afterwards.
        self.file.write_bytes(self.file.read_bytes() + b"changed")
        self.assert_refused(self.request(prompt="raw <lora:A:0.8137>"), self.http())

    def test_exact_inline_and_manual_paths_record_current_identity_without_changing_weights_or_prompt_audit(self):
        response, call = self.submit(self.request(prompt="raw <lora:A.safetensors:0.8137>"), self.http())
        job = self.server.jobs[response["id"]]
        self.assertEqual(job["prompt"], "raw <lora:A.safetensors:0.8137>")
        self.assertEqual(job["lora_library"][0]["instanceId"], self.id)
        self.assertEqual(call[2].loras[0].weight, .8137)
        self.assertEqual(call[2].prompt, "raw ")
        self.assertEqual(call[3].prompt, job["prompt"])
        context = [{"instanceId": self.id, "contentHash": self.instance["contentHash"]}]
        response, call = self.submit(self.request(loras=[{"path": self.file.name, "weight": -.25}]), self.http(context))
        self.assertEqual(call[2].loras[0].weight, -.25)

    def test_byte_identical_registered_copies_need_exact_inline_reference(self):
        other = self.models / "other" / self.file.name
        other.parent.mkdir()
        other.write_bytes(self.file.read_bytes())
        other_id = str(uuid.uuid4())
        self.registry["instances"][other_id] = {**self.instance, "instanceId": other_id, "value": "other/A.safetensors", "fsIdentity": file_identity(other)}
        self.write("registry.json", self.registry)
        self.assert_refused(self.request(prompt="<lora:A:1>"), self.http())
        self.submit(self.request(prompt="<lora:other/A:1>"), self.http())

    def test_reused_old_path_stale_revision_and_foreign_identity_never_queue(self):
        old = self.models / "retired.bin"
        self.file.rename(old)
        self.file.write_bytes(old.read_bytes())
        context = [{"instanceId": self.id, "contentHash": self.instance["contentHash"]}]
        self.assert_refused(self.request(loras=[{"path": self.file.name, "weight": 1}]), self.http(context))
        self.assert_refused(self.request(loras=[{"path": self.file.name, "weight": 1}]), self.http(context, "0"))
        self.assert_refused(self.request(loras=[{"path": self.file.name, "weight": 1}]), self.http([{**context[0], "instanceId": str(uuid.uuid4())}]))

    def test_unregistered_engine_legacy_path_stays_compatible_after_another_engine_publication(self):
        self.instance["engine"] = "Anima"
        self.write("registry.json", self.registry)
        response, call = self.submit(self.request(loras=[{"path": self.file.name, "weight": .8137}]), self.http())
        self.assertEqual(call[2].loras[0].path, self.file.name)
        self.assertFalse(self.server.jobs[response["id"]]["lora_library"])

    def test_pending_or_corrupt_repair_blocks_related_jobs_but_unrelated_generation_still_admits(self):
        for manifest in [{"schemaVersion": 1, "phase": "preparing", "revision": 1}, {"schemaVersion": 999}, []]:
            self.write("publication.json", manifest)
            self.assert_refused(self.request(loras=[{"path": self.file.name, "weight": 1}]), self.http())
            self.submit(self.request())

    def test_published_manifest_without_receipt_and_corrupt_journal_are_closed_on_python_plane(self):
        operations = self.state / "lora-library" / "operations"
        operations.mkdir()
        operation = str(uuid.uuid4())
        record = {"schemaVersion": 1, "operationId": operation, "requestDigest": "a" * 64,
                  "baseRevision": 0, "revision": 1, "files": {}, "galleryPatches": [], "phase": "applied"}
        file = operations / (operation + ".json")
        file.write_text(json.dumps(record), encoding="utf-8")
        self.assert_refused(self.request(loras=[{"path": self.file.name, "weight": 1}]), self.http())
        file.write_text(json.dumps({**record, "phase": "published", "schemaVersion": 999}), encoding="utf-8")
        self.assert_refused(self.request(loras=[{"path": self.file.name, "weight": 1}]), self.http())

    def test_held_node_lease_blocks_complete_job_admission_without_mutating_any_snapshot(self):
        lock = self.state / ".lora-coordination.lock"
        lock.mkdir()
        (lock / "lease.json").write_text(json.dumps({"owner": "node", "pid": os.getpid(), "token": str(uuid.uuid4())}), encoding="utf-8")
        self.assert_refused(self.request(), self.http())
