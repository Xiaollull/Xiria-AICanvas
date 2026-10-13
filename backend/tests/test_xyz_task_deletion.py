"""Record-only deletion, real isolated store/HTTP, atomic session and concurrent ownership tests. No GPU."""
import copy
import hashlib
import json
import os
import tempfile
import threading
import subprocess
import unittest
import uuid
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend import lora_mix_search as mix
from backend import xyz_task_deletion as deletion
from backend.xyz_experiments import ExperimentError, ExperimentStore
from backend.tests.test_xyz_experiments import plan_for
from backend.tests.test_xyz_planner import baseline
from backend.tests import test_lora_mix_service as mix_fixture
from backend.tests import test_xyz_service as xyz_fixture


class TaskDeletionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="task-deletion-"))
        self.store = ExperimentStore(self.root / "xyz-plots")
        self.refs, self.exports, self.jobs, self.controls = set(), {}, {}, {}

    def create(self, *, engine="SD", origin=None, status="complete", store=None):
        store = store or self.store
        draft = baseline(engine)
        key, _ = store.create(submission_id=str(uuid.uuid4()), draft=draft, on_error="pause", plan=plan_for(draft), library_revision=0, requested_concurrency=1, origin=origin)
        state = store.record(key)["state"]
        state["status"] = status
        store.save(key)
        return key

    def round(self, session, number=1, **kwargs):
        return self.create(origin={"kind": "lora_mix_search", "schema_version": 1, "session_id": session, "round": number, "context_hash": "context", "candidates": []}, **kwargs)

    def preview(self, module="xyz", scope="all", ids=None):
        return deletion.preview(self.store, module, {"scope": scope, **({"ids": ids} if ids is not None else {})}, self.refs, self.exports, self.jobs, self.controls)

    def commit(self, value):
        payload = {"scope": value["scope"], "fingerprint": value["fingerprint"]}
        if value["scope"] == "selected":
            payload["ids"] = [item["id"] for item in value["tasks"]]
        return deletion.commit(self.store, value["module"], payload, self.refs, self.exports, self.jobs, self.controls)

    def test_all_over_fifty_six_engines_module_isolation_restart_and_files(self):
        keys = [self.create(engine=engine) for engine in ("SD", "iL", "Anima", "Flux", "Flux2", "Krea2") for _ in range(10)]
        session = uuid.uuid4().hex
        other = self.round(session)
        keep = self.root / "output.png"; keep.write_bytes(b"original")
        for key in keys:
            (self.store.root / key / "unknown-export.png").write_bytes(b"export nested in task directory")
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(self.store.list(plain_only=True)["total"], 60)
        self.assertEqual(len(self.store.list(plain_only=True)["experiments"]), 50)
        result = self.commit(self.preview())
        self.assertEqual((result["deleted"], result["failed"]), (60, 0))
        self.assertEqual(set(self.store.records), {other})
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        restart = ExperimentStore(self.store.root)
        restart.ensure_loaded()
        self.assertEqual(set(restart.records), {other})
        self.assertEqual(set(restart.deleted_records), set(keys))
        with self.assertRaises(ExperimentError):
            restart.record(keys[0])
        definition = restart.deleted_records[keys[0]]["definition"]
        with self.assertRaises(ExperimentError) as raised:
            restart.create(submission_id=definition["submission_id"], draft=definition["draft"], on_error="pause", plan=definition["plan"], library_revision=0, requested_concurrency=1)
        self.assertEqual(raised.exception.code, "deleted-submission")

    def test_selection_missing_wrong_module_and_replay(self):
        key = self.create(); sid = uuid.uuid4().hex; other = self.round(sid)
        value = self.preview(scope="selected", ids=[key, other, uuid.uuid4().hex])
        result = self.commit(value)
        self.assertEqual((result["deleted"], result["skipped"]), (1, 2))
        self.assertIn(other, self.store.records)
        with self.assertRaises(ExperimentError) as raised:
            self.commit(value)
        self.assertEqual(raised.exception.code, "delete-changed")

    def test_all_activity_and_reference_reasons(self):
        for status in ("queued", "running", "pausing", "paused", "cancelling"):
            key = self.create(status=status)
            self.assertFalse(self.preview(scope="selected", ids=[key])["eligible"])
        for kind in ("executor", "control", "export-ref", "export-state", "child", "running-cell", "decision", "foreign"):
            with self.subTest(kind=kind):
                key = self.create()
                state = self.store.record(key)["state"]
                if kind == "executor": self.refs.add(key)
                if kind == "control": self.controls[key] = object()
                if kind == "export-ref": self.exports[key] = {"experiment_id": key}
                if kind == "export-state": state["exports"] = [{"id": "ex", "status": "queued"}]
                if kind == "child": self.jobs[key + "child"] = {"xyz_parent": key, "status": "running"}
                if kind == "running-cell": next(iter(state["cells"].values()))["status"] = "running"
                if kind == "decision": state["decision"] = {"cell_id": "x"}
                if kind == "foreign": state["foreign"] = True
                value = self.preview(scope="selected", ids=[key])
                self.assertEqual(value["eligible"], 0)
                self.assertTrue(value["tasks"][0]["reason"])
                self.assertEqual(self.commit(value)["skipped"], 1)

    def test_search_whole_session_atomic_and_other_modules(self):
        sid, active = uuid.uuid4().hex, uuid.uuid4().hex
        rounds = [self.round(sid, n) for n in (1, 2, 3)]
        protected = [self.round(active, 1), self.round(active, 2, status="paused")]
        plain = self.create()
        value = self.preview("lora_mix")
        self.assertEqual((value["eligible"], value["skipped"]), (1, 1))
        result = self.commit(value)
        self.assertEqual((result["deleted"], result["skipped"]), (1, 1))
        self.assertTrue(set(rounds).isdisjoint(self.store.records))
        self.assertTrue(set(protected + [plain]).issubset(self.store.records))
        self.assertIn(sid, self.store.deleted_sessions)
        with self.assertRaises(ExperimentError) as raised:
            self.round(sid, 4)
        self.assertEqual(raised.exception.code, "deleted-session")
        fresh = ExperimentStore(self.store.root); fresh.ensure_loaded()
        self.assertEqual(mix.session_summaries(fresh.records)[0]["id"], active)

    def test_mix_all_over_fifty_six_engines_and_retained_tested_reserved_history(self):
        for engine in ("SD", "iL", "Anima", "Flux", "Flux2", "Krea2"):
            for _ in range(10): self.round(uuid.uuid4().hex, engine=engine)
        sid = uuid.uuid4().hex
        key = self.create(origin={"kind": "lora_mix_search", "schema_version": 1, "session_id": sid, "round": 1, "context_hash": "history", "candidates": [{"id": "C01", "key": "candidate-key", "weights": {"member": 5000}, "x_index": 0}]})
        for cell in self.store.record(key)["state"]["cells"].values(): cell["status"] = "complete"
        self.store.save(key)
        before = mix.used_candidates(self.store.candidate_records(), "SD", "history")
        self.assertEqual(before[0]["state"], "tested")
        self.assertEqual(self.commit(self.preview("lora_mix"))["deleted"], 61)
        fresh = ExperimentStore(self.store.root); fresh.ensure_loaded()
        self.assertEqual(mix.used_candidates(fresh.candidate_records(), "SD", "history"), before)
        self.assertFalse(fresh.records)

    def test_receipt_write_failure_has_no_half_session_and_partial_retry(self):
        sid = uuid.uuid4().hex
        rounds = [self.round(sid, n) for n in (1, 2)]
        plain = self.create()
        with mock.patch("backend.xyz_experiments.write_json_atomic", side_effect=OSError("synthetic disk refusal")):
            result = self.commit(self.preview("lora_mix"))
        self.assertEqual(result["failed"], 1)
        self.assertTrue(set(rounds).issubset(self.store.records))
        self.assertFalse(self.store.deleted_sessions)
        self.assertEqual(self.commit(self.preview("lora_mix"))["deleted"], 1)
        first, second = self.create(), self.create()
        real = self.store.tombstone
        def fail_one(module, key, ids):
            if key == second: raise OSError("second failed")
            return real(module, key, ids)
        with mock.patch.object(self.store, "tombstone", side_effect=fail_one):
            result = self.commit(self.preview(scope="selected", ids=[first, second]))
        self.assertEqual((result["deleted"], result["failed"]), (1, 1))
        self.assertEqual(self.commit(self.preview(scope="selected", ids=[second]))["deleted"], 1)
        self.assertIn(plain, self.store.records)

    def test_changed_membership_or_export_fails_before_writes(self):
        key = self.create()
        value = self.preview()
        self.exports["held"] = {"experiment_id": key}
        with self.assertRaises(ExperimentError): self.commit(value)
        self.assertIn(key, self.store.records)
        self.exports.clear()
        value = self.preview(); self.create()
        with self.assertRaises(ExperimentError): self.commit(value)
        self.assertFalse((self.store.root / "task-deletions.json").exists())

    def test_bad_ids_paths_and_extra_keys_refused(self):
        for payload in ({"scope": "selected", "ids": ["../outside"]}, {"scope": "all", "ids": []}, {"scope": "all", "path": "x"}, {"scope": "all", "engine": "SD"}, {"scope": "selected", "ids": ["0"*32]*2}):
            with self.subTest(payload=payload), self.assertRaises(ExperimentError): deletion.options(payload)

    def test_symlink_record_and_root_fail_closed(self):
        key = self.create()
        outside = self.root / "outside.json"; outside.write_bytes(b"outside")
        target = self.store.root / key / "state.json"
        backup = target.read_bytes(); target.unlink()
        try: target.symlink_to(outside)
        except OSError:
            target.write_bytes(backup); self.skipTest("Native symlink creation unavailable")
        self.assertEqual(self.preview()["eligible"], 0)
        self.assertEqual(outside.read_bytes(), b"outside")
        fresh = ExperimentStore(self.store.root); fresh.ensure_loaded()
        self.assertIn(key, fresh.unreadable)

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_native_windows_junction_root_escape_refused(self):
        self.create()
        link = self.root / "junction-root"
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(self.store.root)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            with self.assertRaises(ExperimentError) as raised: ExperimentStore(link).ensure_loaded()
            self.assertEqual(raised.exception.code, "unsafe-store")
        finally:
            link.rmdir()  # Only the test-owned junction, never its target.

    def test_corrupt_receipt_and_missing_archive_never_revive(self):
        key = self.create(); self.commit(self.preview())
        ledger = self.store.root / "task-deletions.json"
        good = ledger.read_bytes(); ledger.write_bytes(b"bad json")
        fresh = ExperimentStore(self.store.root)
        for _ in range(2):
            with self.assertRaises(ExperimentError): fresh.ensure_loaded()
        ledger.write_bytes(good)
        (self.store.root / key / "definition.json").write_bytes(b"bad archive")
        with self.assertRaises(ExperimentError): ExperimentStore(self.store.root).ensure_loaded()

    def test_unreadable_unknown_round_blocks_every_search_deletion(self):
        sid = uuid.uuid4().hex; self.round(sid)
        self.store.unreadable[uuid.uuid4().hex] = "bad round"
        self.assertEqual(self.preview("lora_mix")["eligible"], 0)

    def test_concurrent_double_confirm_one_publication(self):
        self.create(); value = self.preview(); results = []
        def run():
            with self.store.lock:
                try: results.append(self.commit(value)["deleted"])
                except ExperimentError: results.append("changed")
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(5)
        self.assertCountEqual(results, [1, "changed"])
        self.assertEqual(len(self.store.deletions["entries"]), 1)

    def test_second_process_view_cannot_republish_stale_receipt(self):
        key = self.create()
        second = ExperimentStore(self.store.root); second.ensure_loaded()
        self.commit(self.preview())
        with self.assertRaises(ExperimentError) as raised: second.record(key)
        self.assertEqual(raised.exception.code, "deletion-recovery")
        self.assertEqual(len(json.loads((self.store.root / "task-deletions.json").read_text())["entries"]), 1)

    def test_post_publication_error_is_reconciled_as_success(self):
        self.create()
        from backend.xyz_experiments import write_json_atomic
        def after_replace(path, value):
            write_json_atomic(path, value)
            raise OSError("synthetic post-replace error")
        with mock.patch("backend.xyz_experiments.write_json_atomic", side_effect=after_replace):
            self.assertEqual(self.commit(self.preview())["deleted"], 1)
        self.assertFalse(self.store.records)


class TaskDeletionHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import inference_server
        cls.server = inference_server

    def setUp(self):
        self.harness = mix_fixture.MixServiceTests(methodName="runTest")
        self.harness.server = self.server
        self.harness.setUp()
        app = FastAPI()
        app.router.routes.extend(route for route in self.server.app.routes if getattr(route, "path", "").startswith(("/api/inference/xyz-plots", "/api/inference/lora-mix", "/api/inference/history/assets")))
        self.client = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:9017")
        self.headers = {"origin": "http://127.0.0.1:9017"}

    def tearDown(self):
        self.client.close()
        self.harness.tearDown()

    def submit(self):
        return self.server.confirm_mix_round(self.harness.submission(), self.harness.http())

    def settle(self, key, status="interrupted"):
        store = self.server.xyz_store()
        store.record(key)["state"]["status"] = status
        store.save(key)
        with self.server.jobs_lock:
            self.server.job_controls.pop(key, None); self.server._xyz_executor_refs.discard(key); self.server.jobs[key]["status"] = status

    def test_actual_routes_local_gate_cancel_and_deleted_session_candidate_history(self):
        submitted = self.submit(); sid, key = submitted["session"]["id"], submitted["experiment"]["id"]
        self.settle(key)
        used_before = self.server.mix_used({"draft": self.harness.draft()})["used"]
        self.assertTrue(used_before)
        root = "/api/inference/lora-mix"
        value = self.client.get(root + "/tasks?engine=SD").json()
        self.assertEqual(value["tasks"][0]["id"], sid)
        for headers in ({}, {"origin": "http://evil.test:9017"}, {"origin": "http://127.0.0.1:9018"}):
            self.assertEqual(self.client.post(root + "/task-deletions", headers=headers, json={"scope": "all", "fingerprint": "0"*64}).status_code, 403)
        before = {str(path): path.read_bytes() for path in self.harness.state.rglob("*") if path.is_file()}
        preview = self.client.post(root + "/task-deletions/preview", headers=self.headers, json={"scope": "all"})
        self.assertEqual(preview.status_code, 200, preview.text)
        for path, content in before.items(): self.assertEqual(Path(path).read_bytes(), content)  # Cancelling here writes nothing.
        response = self.client.post(root + "/task-deletions", headers=self.headers, json={"scope": "all", "fingerprint": preview.json()["fingerprint"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["deleted"], 1)
        self.assertEqual(self.client.get(root + "/sessions").json()["sessions"], [])
        self.assertEqual(self.client.get(root + "/sessions/" + sid).status_code, 404)
        self.assertEqual(self.client.get("/api/inference/xyz-plots/" + key).status_code, 410)
        self.assertEqual(self.server.mix_used({"draft": self.harness.draft()})["used"], used_before)
        draft = self.harness.draft(session={"id": sid, "revision": 1})
        with self.assertRaises(self.server.HTTPException) as raised: self.server.confirm_mix_round(self.harness.submission(draft), self.harness.http())
        self.assertEqual(raised.exception.detail["code"], "deleted-session")
        self.server._xyz_store = None
        self.assertEqual(self.server.mix_used({"draft": self.harness.draft()})["used"], used_before)
        self.assertEqual(self.client.get(root + "/tasks").json()["total"], 0)

    def test_http_lan_and_proxy_proof(self):
        app = self.client.app
        with TestClient(app, client=("192.168.2.4", 50000), base_url="http://127.0.0.1:9017") as remote:
            self.assertEqual(remote.post("/api/inference/xyz-plots/task-deletions/preview", headers=self.headers, json={"scope": "all"}).status_code, 403)
        with mock.patch.dict(os.environ, {"XIRAI_WD14_PROXY_PROOF": "private-proof"}):
            response = self.client.post("/api/inference/xyz-plots/task-deletions/preview", json={"scope": "all"}, headers={"origin": "http://localhost:9020", "x-xirai-wd14-proxy-proof": "private-proof"})
            self.assertEqual(response.status_code, 200, response.text)

    def test_executor_drain_and_export_registration_after_delete(self):
        submitted = self.submit(); key = submitted["experiment"]["id"]
        store = self.server.xyz_store(); store.record(key)["state"]["status"] = "cancelled"; store.save(key)
        with self.server.jobs_lock: self.server.job_controls.pop(key, None); self.server.jobs[key]["status"] = "cancelled"
        self.assertEqual(self.server.module_tasks("lora_mix")["tasks"][0]["deletable"], False)
        self.server._xyz_executor_refs.discard(key)
        value = self.client.post("/api/inference/lora-mix/task-deletions/preview", headers=self.headers, json={"scope": "all"}).json()
        self.assertEqual(value["eligible"], 1)
        self.client.post("/api/inference/lora-mix/task-deletions", headers=self.headers, json={"scope": "all", "fingerprint": value["fingerprint"]})
        with mock.patch.object(self.server, "xyz_grid_estimate", return_value={"budget": {"ok": True}}):
            with self.assertRaises(self.server.HTTPException) as raised: self.server.start_xyz_export(key, {}, {})
        self.assertEqual(raised.exception.status_code, 410)

    def test_original_png_and_metadata_read_still_work_after_deletion_restart(self):
        helper = xyz_fixture.XYZServiceTests(methodName="runTest"); helper.server = self.server; helper.outputs = self.harness.outputs
        response = self.server.create_xyz_experiment(helper.payload())
        key = response["experiment"]["id"]
        fake, _calls = helper.fake_generation()
        with mock.patch.object(self.server, "run_generation", side_effect=fake): self.server.run_xyz_experiment(key)
        detail = self.server.get_xyz_experiment(key); asset = detail["cells"][0]["asset_id"]
        path = self.server.history_asset_path(asset); before = path.read_bytes()
        preview = self.client.post("/api/inference/xyz-plots/task-deletions/preview", headers=self.headers, json={"scope": "selected", "ids": [key]})
        self.assertEqual(preview.json()["eligible"], 1, preview.text)
        response = self.client.post("/api/inference/xyz-plots/task-deletions", headers=self.headers, json={"scope": "selected", "ids": [key], "fingerprint": preview.json()["fingerprint"]})
        self.assertEqual(response.json()["deleted"], 1, response.text)
        for restart in (False, True):
            if restart: self.server._xyz_store = None
            response = self.client.get("/api/inference/history/assets/" + asset)
            self.assertEqual(response.status_code, 200, response.text if response.status_code != 200 else "")
            self.assertEqual(response.content, before)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__": unittest.main()
