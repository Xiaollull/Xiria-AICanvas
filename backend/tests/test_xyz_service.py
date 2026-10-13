"""QPlan2/P1 Phase B service integration: routes, admission, single-slot orchestration.

Imports the real inference service against isolated state/output/model folders. The executor is
mocked and `run_generation` is replaced by a fake that writes a real file, so no GPU is touched;
real sampling evidence is outside this suite.
"""
import copy
import hashlib
import json
import struct
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, PngImagePlugin

from backend.lora_file_identity import file_identity
from backend.xyz_planner import CAPABILITIES, compile_xyz
from backend.tests.test_xyz_planner import baseline as engine_baseline


def axis(field, values, **extra):
    return dict(field=field, values=[dict(value=value, label=str(value)) for value in values], **extra)


class XYZServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import inference_server
        cls.server = inference_server

    def setUp(self):
        server = self.server
        self.root = Path(tempfile.mkdtemp(prefix="xyz-b-service-"))
        self.state, self.outputs, self.models = self.root / "state", self.root / "outputs", self.root / "models"
        for path in (self.state, self.outputs, self.models):
            path.mkdir()
        (self.models / "model.safetensors").write_bytes(b"checkpoint placeholder")
        (self.models / "other.safetensors").write_bytes(b"second checkpoint placeholder")
        self.lora = self.models / "A.safetensors"
        header = json.dumps({"__metadata__": {}}).encode()
        self.lora.write_bytes(struct.pack("<Q", len(header)) + header + b"adapter bytes")
        self.patches = [
            mock.patch.object(server, "STATE_DIRECTORY", self.state),
            mock.patch.object(server, "OUTPUT_DIRECTORY", self.outputs),
            mock.patch.object(server, "model_roots", return_value=(self.models, self.models)),
            mock.patch.object(server, "_xyz_store", None),
        ]
        for patch in self.patches:
            patch.start()
        self.submit = mock.patch.object(server.executor, "submit")
        self.executor = self.submit.start()
        self.draws = []
        self.rng = mock.patch.object(server, "secure_random_uint64", side_effect=lambda: self.draws.append(1) or (2**64 - len(self.draws)))
        self.rng.start()
        with server.jobs_lock:
            self.before = set(server.jobs)

    def tearDown(self):
        server = self.server
        self.rng.stop()
        self.submit.stop()
        for patch in reversed(self.patches):
            patch.stop()
        with server.jobs_lock:
            for job_id in set(server.jobs) - self.before:
                server.jobs.pop(job_id, None)
                server.job_controls.pop(job_id, None)

    # Fixtures ----------------------------------------------------------------------------------

    def draft(self, **overrides):
        parameters = dict(engine="SD", checkpoint="model.safetensors", prompt="cat, blue sky", negative_prompt="bad",
                          width=512, height=512, steps=4, cfg=7, seed="18446744073709551615", sampler="euler",
                          scheduler="normal", guidance="none", pag={"scale": 0.3, "applied_layers": "mid"}, postprocess_chain=[])
        draft = dict(schema_version=1, kind="xyz_draft", engine="SD",
                     baseline=dict(parameters=parameters, prompt_without_groups="cat, blue sky", group_prompt="", loras=[], seed_mode="fixed"),
                     axes={"x": axis("steps", [4, 8]), "y": axis("cfg", ["5", "7"]), "z": None}, repeat_count=1, requested_concurrency=1)
        for key, value in overrides.items():
            draft[key] = value
        return draft

    def payload(self, draft=None, submission=None, on_error="pause"):
        return {"schema_version": 1, "kind": "xyz_submit", "submission_id": submission or str(uuid.uuid4()),
                "draft": draft or self.draft(), "on_error": on_error}

    def http(self, revision="1"):
        return SimpleNamespace(headers={"x-xirai-lora-revision": revision})

    def registry(self):
        self.instance_id = str(uuid.uuid4())
        self.instance = {"instanceId": self.instance_id, "engine": "SD", "rootId": "local-SD", "value": self.lora.name,
                         "history": [], "status": "present", "contentRevision": 1,
                         "contentHash": hashlib.sha256(self.lora.read_bytes()).hexdigest(), "fsIdentity": file_identity(self.lora)}
        library = self.state / "lora-library"
        library.mkdir(exist_ok=True)
        (library / "registry.json").write_text(json.dumps({"schemaVersion": 1, "revision": 1, "instances": {self.instance_id: self.instance}, "substitutions": []}), encoding="utf-8")
        (library / "publication.json").write_text(json.dumps({"schemaVersion": 1, "phase": "published", "revision": 1}), encoding="utf-8")
        return {"value": self.lora.name, "weight": "0.5", "precision": 4, "identitySchema": 1, "instanceId": self.instance_id,
                "contentHash": self.instance["contentHash"], "contentRevision": 1}

    def submit_ok(self, payload=None, http=None):
        response = self.server.create_xyz_experiment(payload or self.payload(), http)
        return response["experiment"]["id"], response

    def refused(self, payload, status, http=None):
        before_jobs = set(self.server.jobs)
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.create_xyz_experiment(payload, http)
        self.assertEqual(raised.exception.status_code, status, raised.exception.detail)
        self.assertEqual(set(self.server.jobs), before_jobs)
        self.assertFalse((self.state / "xyz-plots" / uuid.UUID(payload["submission_id"]).hex).exists())
        return raised.exception.detail

    def fake_generation(self, outcomes=None, during=None, oom_above=None, batches=None):
        """Mimics run_generation's contract: it never raises and leaves a final record; one output
        per image, in image order, each with its own seed (the image_seeds override or request.seed)."""
        server, calls = self.server, []

        def run(job_id, request, audit_request=None, inline_lora_audit=None, *, image_seeds=None):
            job = server.jobs[job_id]
            calls.append((job["xyz_cell"], request))
            if batches is not None:
                batches.append((list(job["xyz_cells"]), request.images_per_batch, image_seeds))
            if during:
                during(job_id, job)
            if server.job_controls[job_id].cancelled:
                server.update_job(job_id, status="cancelled", stage="cancelled")
                return
            if oom_above is not None and request.images_per_batch > oom_above:
                server.update_job(job_id, status="error", failure_kind="oom", error="GPU memory exhausted. Try a smaller canvas.")
                return
            if (outcomes or {}).get((job["xyz_cell"], job["xyz_attempt"])) == "error":
                server.update_job(job_id, status="error", error="GPU memory exhausted. Try a smaller canvas.")
                return
            seeds = image_seeds if image_seeds is not None else [request.seed]
            assert len(seeds) == request.images_per_batch
            outputs = []
            for index, seed in enumerate(seeds):
                target = self.outputs / "cells" / f"{job_id}-{index}.png"
                target.parent.mkdir(exist_ok=True)
                cell_id = job['xyz_cells'][index]
                info = PngImagePlugin.PngInfo()
                info.add_text('parameters', json.dumps({'seed': str(seed), 'xyz_plot': {
                    'experiment_id': job['xyz_parent'], 'cell_id': cell_id,
                    'attempt': job['xyz_attempts'][cell_id], **job['xyz_asset_identity'][cell_id]}}))
                Image.new("RGB", (8, 8), (index * 40, 0, 0)).save(target, pnginfo=info)
                outputs.append({"index": index, "asset_id": server.history_asset_token(target), "output_name": target.name,
                                "output_relative": server.output_relative_path(target), "width": 8, "height": 8, "seed": str(seed)})
            server.update_job(job_id, status="complete", elapsed_seconds=1.5, sampling_batch_size=request.images_per_batch, outputs=outputs)
        return run, calls

    # Contract between the planner and the real strict request ----------------------------------

    def test_real_strict_validator_six_engines_and_inherited_resample_seed_strings(self):
        for engine in CAPABILITIES["engines"]:
            with self.subTest(engine=engine):
                draft = engine_baseline(engine)
                draft["baseline"]["seed_mode"] = "random"
                p = draft["baseline"]["parameters"]
                catalog = {(engine, key, value) for key, value in p.items() if key in CAPABILITIES["engines"][engine]["model_fields"]}
                plan = compile_xyz(draft, model_catalog=catalog, resolve_seeds=True, random_uint64=lambda: 7,
                                   validate_request=self.server.GenerateInput.model_validate)
                self.assertEqual(plan.errors, [])
                self.assertEqual(plan.cells[0].request["seed"], 7)
        draft = self.draft(inherit_postprocess=True)
        draft["baseline"]["parameters"]["postprocess_chain"] = [
            {"type": "resample", "enabled": True, "scale": 1.5, "denoise": 0.35, "seed_mode": "random"},
            {"type": "resample", "enabled": False, "scale": 2, "denoise": 0.5, "seed_mode": "inherit"},
        ]
        draws = iter([11, 12, 13, 14, 15])
        plan = compile_xyz(draft, model_catalog={("SD", "checkpoint", "model.safetensors")}, resolve_seeds=True,
                           random_uint64=lambda: next(draws), validate_request=self.server.GenerateInput.model_validate)
        self.assertEqual(plan.errors, [])
        stages = [cell.request["postprocess_chain"] for cell in plan.cells]
        self.assertTrue(all(isinstance(chain[0]["seed"], str) and chain[0]["seed_mode"] == "fixed" for chain in stages))
        self.assertEqual(len({chain[0]["seed"] for chain in stages}), 4)
        self.assertNotIn("seed", stages[0][1])
        self.assertEqual(stages[0][1]["seed_mode"], "inherit")
        request = self.server.GenerateInput.model_validate(plan.cells[0].request)
        self.assertEqual(self.server.resolve_chain_seeds(request, request.seed)[0], int(stages[0][0]["seed"]))

    # Submission ---------------------------------------------------------------------------------

    def test_submit_persists_takes_one_slot_and_replay_never_redraws(self):
        draft = self.draft()
        draft["baseline"]["seed_mode"] = "random"
        payload = self.payload(draft)
        experiment_id, response = self.submit_ok(payload)
        self.assertFalse(response["replayed"])
        self.assertEqual(len(self.draws), 1)
        self.executor.assert_called_once_with(self.server.run_xyz_experiment, experiment_id)
        self.assertTrue((self.state / "xyz-plots" / experiment_id / "definition.json").is_file())
        with self.server.jobs_lock:
            parent = self.server.jobs[experiment_id]
            self.assertIsInstance(self.server.job_controls[experiment_id], self.server.XYZJobControl)
        self.assertEqual((parent["kind"], parent["status"], parent["total_images"]), ("xyz_plot", "queued", 4))
        replay = self.server.create_xyz_experiment(copy.deepcopy(payload), None)
        self.assertTrue(replay["replayed"])
        self.assertEqual(len(self.draws), 1)
        self.executor.assert_called_once()
        changed = copy.deepcopy(payload)
        changed["draft"]["axes"]["x"]["values"][0]["value"] = 12
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.create_xyz_experiment(changed, None)
        self.assertEqual(raised.exception.status_code, 409)
        detail = self.server.get_xyz_experiment(experiment_id)
        self.assertEqual(len({cell["seed"] for cell in detail["cells"]}), 1)
        self.assertEqual(detail["cells"][0]["seed"], str(2**64 - 1))
        self.assertEqual(self.server.list_xyz_experiments(None)["total"], 1)
        self.assertEqual(self.server.list_xyz_experiments("iL")["total"], 0)
        # The queued parent closes the LoRA repair and GPU-tagger gates like any active job.
        self.assertTrue(self.server.tagger_generation_busy())
        lock = self.state / ".lora-coordination.lock"
        lock.mkdir()
        token = str(uuid.uuid4())
        (lock / "lease.json").write_text(json.dumps({"owner": "node", "token": token}), encoding="utf-8")
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={"origin": "http://127.0.0.1:7709", "host": "127.0.0.1:7709"})
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.lora_library_admission({"leaseToken": token}, request)
        self.assertEqual(raised.exception.status_code, 409)

    def test_refusals_create_nothing_and_draw_no_seed(self):
        draft = self.draft(requested_concurrency=11)
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.create_xyz_experiment(self.payload(draft), None)
        self.assertEqual(raised.exception.status_code, 422)
        draft = self.draft()
        draft["axes"]["x"] = axis("checkpoint", ["model.safetensors", "absent.safetensors"])
        detail = self.refused(self.payload(draft), 422)
        self.assertEqual(detail["errors"][0]["code"], "model-resource")
        self.assertEqual(detail["errors"][0]["coordinate"]["x_index"], 1)
        draft = self.draft(inherit_postprocess=True)
        draft["baseline"]["seed_mode"] = "random"
        draft["baseline"]["parameters"]["postprocess_chain"] = [{"type": "resample", "enabled": True, "upscaler": "xyz-missing-upscaler.pth", "scale": 2, "denoise": 0.3}]
        detail = self.refused(self.payload(draft), 422)
        self.assertEqual({issue["code"] for issue in detail["errors"]}, {"admission"})
        self.assertEqual(detail["error_count"], 4)
        preview = self.server.plan_xyz_experiment(draft, None)
        self.assertFalse(preview["ok"])
        self.assertEqual(preview["error_count"], 4)
        self.assertEqual(self.draws, [])
        preview = self.server.plan_xyz_experiment(self.draft(), None)
        self.assertTrue(preview["ok"])
        self.assertEqual((preview["total_cells"], preview["max_concurrency"]), (4, 10))
        self.assertEqual(preview["batches"]["count"], 4)
        self.assertEqual(self.server.xyz_store().list()["total"], 0)

    def test_lora_identity_is_admitted_at_submit_and_rechecked_at_execution(self):
        reference = self.registry()
        draft = self.draft()
        draft["baseline"]["loras"] = [reference]
        draft["axes"] = {"x": axis("lora_weight", ["0.5", "1"], target=self.instance_id), "y": None, "z": None}
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.create_xyz_experiment(self.payload(draft), self.http("0"))
        self.assertEqual(raised.exception.status_code, 409)
        experiment_id, _ = self.submit_ok(self.payload(draft), self.http("1"))
        cell = self.server.get_xyz_cell(experiment_id, "1:0:0:0")
        self.assertEqual(cell["request"]["loras"], [{"path": "A.safetensors", "weight": 1.0}])
        self.assertEqual(cell["lora_references"][0]["instanceId"], self.instance_id)
        # Bytes change after submit: the frozen identity no longer admits; nothing is generated.
        self.lora.write_bytes(self.lora.read_bytes() + b"changed")
        run, calls = self.fake_generation()
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
        self.assertEqual(calls, [])
        summary = self.server.xyz_store().summary(experiment_id)
        self.assertEqual((summary["status"], summary["decision"]["cell_id"]), ("paused", "0:0:0:0"))
        self.assertIn("LoRA target is missing, changed or identity-conflicted", summary["decision"]["error"])
        preview = self.server.plan_xyz_experiment(draft, self.http("1"))
        self.assertEqual({issue["code"] for issue in preview["errors"]}, {"admission"})

    # Execution ----------------------------------------------------------------------------------

    def test_cells_run_through_run_generation_in_order_and_stay_out_of_the_ordinary_queue(self):
        experiment_id, _ = self.submit_ok()
        observed = []

        def during(job_id, job):
            listing = self.server.list_jobs()
            ids = [entry["id"] for entry in listing["jobs"]]
            observed.append((job_id in ids, experiment_id in ids, self.server.get_active_job()["job"],
                             self.server.get_job(experiment_id)["status"]))
            self.assertEqual(job["xyz_parent"], experiment_id)
            parent_entry = next(entry for entry in listing["jobs"] if entry["id"] == experiment_id)
            self.assertEqual(parent_entry["kind"], "xyz_plot")
            self.assertEqual(parent_entry["xyz"]["total"], 4)

        run, calls = self.fake_generation(during=during)
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
        self.assertEqual([cell for cell, _ in calls], ["0:0:0:0", "1:0:0:0", "0:1:0:0", "1:1:0:0"])
        self.assertEqual([(request.steps, request.cfg) for _, request in calls], [(4, 5.0), (8, 5.0), (4, 7.0), (8, 7.0)])
        self.assertTrue(all(request.seed == 2**64 - 1 and request.images_per_batch == 1 for _, request in calls))
        self.assertEqual(observed[0][:3], (False, True, None))
        self.assertEqual(observed[0][3], "running")
        detail = self.server.get_xyz_experiment(experiment_id)
        self.assertEqual(detail["status"], "complete")
        self.assertTrue(all(cell["asset_url"].startswith("/api/inference/history/assets/") for cell in detail["cells"]))
        with self.server.jobs_lock:
            self.assertFalse([job for job in self.server.jobs.values() if job.get("xyz_parent") == experiment_id])
            self.assertNotIn(experiment_id, self.server.job_controls)
            parent = self.server.jobs[experiment_id]
        self.assertEqual((parent["status"], parent["progress"], parent["completed_images"]), ("complete", 100, 4))
        self.assertFalse(self.server.tagger_generation_busy())

    def test_stop_reaches_the_running_cell_and_generic_cancel_delegates(self):
        experiment_id, _ = self.submit_ok()

        def during(job_id, job):
            if job["xyz_cell"] == "1:0:0:0":
                self.assertEqual(self.server.xyz_control(experiment_id, "stop")["status"], "cancelling")
                self.assertTrue(self.server.job_controls[job_id].cancelled)

        run, calls = self.fake_generation(during=during)
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
        detail = self.server.get_xyz_experiment(experiment_id)
        self.assertEqual(detail["status"], "cancelled")
        self.assertEqual([cell["status"] for cell in detail["cells"]], ["complete", "cancelled", "cancelled", "cancelled"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.server.jobs[experiment_id]["status"], "cancelled")

        queued, _ = self.submit_ok()
        self.assertEqual(self.server.cancel_job(queued), {"status": "cancelled"})
        run, calls = self.fake_generation()
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(queued)
        self.assertEqual(calls, [])
        self.assertEqual(self.server.xyz_store().summary(queued)["counts"]["cancelled"], 4)

    def test_pause_before_start_then_resume_runs_and_stop_before_start_settles(self):
        experiment_id, _ = self.submit_ok()
        self.assertEqual(self.server.xyz_control(experiment_id, "pause")["status"], "pausing")
        self.assertEqual(self.server.xyz_control(experiment_id, "resume")["status"], "queued")
        run, calls = self.fake_generation()
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
        self.assertEqual(len(calls), 4)
        other, _ = self.submit_ok()
        self.server.xyz_control(other, "pause")
        self.assertEqual(self.server.xyz_control(other, "stop")["status"], "cancelled")
        run, calls = self.fake_generation()
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(other)
        self.assertEqual(calls, [])
        with self.server.jobs_lock:
            self.assertNotIn(other, self.server.job_controls)
            self.assertEqual(self.server.jobs[other]["status"], "cancelled")

    def test_parent_control_forwards_to_the_cell_without_missing_a_race(self):
        parent = self.server.XYZJobControl()
        parent.cancel()
        child = self.server.JobControl()
        parent.attach(child)
        self.assertTrue(child.cancelled)
        parent, child = self.server.XYZJobControl(), self.server.JobControl()
        parent.attach(child)
        parent.request_pause()
        self.assertTrue(child.paused)
        parent.resume()
        self.assertFalse(child.paused)
        parent.detach()
        parent.cancel()
        self.assertFalse(child.cancelled)

    def test_failed_cell_waits_for_a_decision_then_retry_requeues_the_same_seed(self):
        experiment_id, _ = self.submit_ok()
        run, calls = self.fake_generation(outcomes={("1:0:0:0", 1): "error"})
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
            parent = self.server.get_job(experiment_id)
            self.assertEqual(parent["status"], "paused")
            self.assertIn("等待选择", parent["phase"])
            self.assertTrue(self.server.tagger_generation_busy())
            with self.assertRaises(self.server.HTTPException) as raised:
                self.server.resume_job(experiment_id)
            self.assertEqual(raised.exception.status_code, 409)
            self.server.decide_xyz_experiment(experiment_id, {"action": "retry"})
            self.assertEqual(self.executor.call_count, 2)
            self.server.run_xyz_experiment(experiment_id)
        self.assertEqual([cell for cell, _ in calls], ["0:0:0:0", "1:0:0:0", "1:0:0:0", "0:1:0:0", "1:1:0:0"])
        self.assertEqual(calls[1][1].seed, calls[2][1].seed)
        cell = self.server.get_xyz_cell(experiment_id, "1:0:0:0")
        self.assertEqual(cell["attempts"], 2)
        self.assertEqual(self.server.xyz_store().summary(experiment_id)["status"], "complete")
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.continue_xyz_experiment(experiment_id, {})
        self.assertEqual(raised.exception.status_code, 409)

    def test_restart_recovery_then_explicit_continue(self):
        experiment_id, _ = self.submit_ok()
        state_path = self.state / "xyz-plots" / experiment_id / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "running"
        state["owner"] = {"pid": 1, "host": "elsewhere", "token": "previous-process"}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        with mock.patch.object(self.server, "_xyz_store", None):
            self.assertEqual(self.server.xyz_store().summary(experiment_id)["status"], "interrupted")
            self.server.continue_xyz_experiment(experiment_id, {"retry": "none"})
            self.assertEqual(self.server.xyz_store().summary(experiment_id)["status"], "queued")
        self.assertEqual(self.executor.call_count, 2)

    def test_seed_axis_runs_as_one_real_batch_with_frozen_per_cell_seeds(self):
        draft = self.draft(requested_concurrency=4)
        seeds = ["7", "18446744073709551615", "3", "123456789012345"]
        draft["axes"] = {"x": axis("seed", seeds), "y": None, "z": None}
        preview = self.server.plan_xyz_experiment(copy.deepcopy(draft), None)
        self.assertEqual((preview["batches"]["count"], preview["batches"]["sizes"]), (1, {"4": 1}))
        experiment_id, _ = self.submit_ok(self.payload(draft))
        batches = []
        run, calls = self.fake_generation(batches=batches)
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
        self.assertEqual(len(calls), 1)
        self.assertEqual(batches, [(["0:0:0:0", "1:0:0:0", "2:0:0:0", "3:0:0:0"], 4, [int(seed) for seed in seeds])])
        self.assertEqual(calls[0][1].batch_count, 1)
        detail = self.server.get_xyz_experiment(experiment_id)
        self.assertEqual([cell["seed"] for cell in detail["cells"]], seeds)
        self.assertEqual(len({cell["asset_url"] for cell in detail["cells"]}), 4)
        self.assertEqual([cell["batch"]["index"] for cell in detail["cells"]], [0, 1, 2, 3])
        self.assertEqual((detail["effective_concurrency"], detail["batch_log"][0]["size"]), (4, 4))
        for cell_id, seed in zip(["0:0:0:0", "1:0:0:0", "2:0:0:0", "3:0:0:0"], seeds):
            self.assertEqual(self.server.get_xyz_cell(experiment_id, cell_id)["output"]["seed"], seed)

    def test_incompatible_cells_stay_separate_and_oom_splits_through_the_real_hooks(self):
        draft = self.draft(requested_concurrency=4)
        draft["axes"] = {"x": axis("seed", ["1", "2", "3", "4"]), "y": axis("steps", [4, 8]), "z": None}
        experiment_id, _ = self.submit_ok(self.payload(draft))
        batches = []
        run, _calls = self.fake_generation(batches=batches, oom_above=2)
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
        self.assertEqual([size for _cells, size, _seeds in batches], [4, 2, 2, 4, 2, 2])
        self.assertEqual(batches[1][2], [1, 2])
        detail = self.server.get_xyz_experiment(experiment_id)
        self.assertEqual(detail["counts"]["complete"], 8)
        self.assertTrue(detail["batch_log"][0]["oom"])
        self.assertEqual(detail["cells"][0]["batch"]["size"], 2)

    def test_image_seeds_are_validated_before_any_model_work(self):
        request = self.server.GenerateInput(engine="SD", checkpoint="model.safetensors", prompt="cat", width=64, height=64,
                                            steps=1, cfg=1, denoise=1, seed=3, sampler="euler", scheduler="normal", images_per_batch=2)
        job_id = "xyz-seed-validation-job"
        with self.server.jobs_lock:
            self.server.jobs[job_id] = {"id": job_id, "status": "queued"}
            self.server.job_controls[job_id] = self.server.JobControl()
        with mock.patch.object(self.server, "write_generation_failure_log"), mock.patch.object(self.server, "load_pipeline") as load:
            self.server.run_generation(job_id, request, image_seeds=[1])
        load.assert_not_called()
        self.assertEqual(self.server.jobs[job_id]["status"], "error")
        self.assertIn("image_seeds", self.server.jobs[job_id]["error"])

    def completed_experiment(self):
        experiment_id, _ = self.submit_ok()
        run, _calls = self.fake_generation()
        with mock.patch.object(self.server, "run_generation", side_effect=run):
            self.server.run_xyz_experiment(experiment_id)
        return experiment_id

    def test_publish_mirrors_are_linear_and_detail_projects_post_validation_state(self):
        checked = []
        verify = self.server.xyz_verify_asset

        def track_verify(definition, planned, current):
            checked.append(planned["cell_id"])
            return verify(definition, planned, current)

        with mock.patch.object(self.server, "xyz_verify_asset", side_effect=track_verify):
            experiment_id, _ = self.submit_ok()
            run, _calls = self.fake_generation()
            with mock.patch.object(self.server, "run_generation", side_effect=run):
                self.server.run_xyz_experiment(experiment_id)

            # Submit/queue/progress mirrors must not hash every already-complete source.
            self.assertEqual(checked, [])
            store = self.server.xyz_store()
            record = store.record(experiment_id)
            output = record["state"]["cells"]["0:0:0:0"]["output"]
            path = self.outputs / output["relative"]
            with Image.open(path) as image:
                info = PngImagePlugin.PngInfo()
                info.add_text("parameters", image.info["parameters"])
            Image.new("RGB", (8, 8), "blue").save(path, pnginfo=info)  # only this fixture's blob
            forged = path.read_bytes()
            definition = (store.root / experiment_id / "definition.json").read_bytes()

            detail = self.server.get_xyz_experiment(experiment_id)
            self.assertEqual(len(checked), 4)
            self.assertCountEqual(checked, record["definition"]["order"])
            self.assertEqual((detail["counts"]["complete"], detail["counts"]["missing"]), (3, 1))
            cells = {cell["id"]: cell for cell in detail["cells"]}
            bad = cells["0:0:0:0"]
            self.assertEqual(bad["status"], "missing")
            self.assertNotIn("asset_url", bad)
            self.assertNotIn("thumbnail_url", bad)
            self.assertTrue(all(cells[cell_id]["status"] == "complete" and "asset_url" in cells[cell_id]
                                for cell_id in record["definition"]["order"][1:]))
            self.assertEqual(path.read_bytes(), forged)
            self.assertEqual((store.root / experiment_id / "definition.json").read_bytes(), definition)

    def start_export(self, experiment_id, options):
        submitted = []
        with mock.patch.object(self.server.xyz_export_executor, "submit", side_effect=lambda fn, *args: submitted.append((fn, args))):
            created = self.server.create_xyz_export(experiment_id, options)
        return created["export"], submitted[0]

    def test_grid_export_summary_names_the_lora_of_a_lora_axis(self):
        from backend import xyz_grid
        reference = self.registry()
        draft = self.draft()
        draft["baseline"]["loras"] = [reference]
        draft["axes"] = {"x": axis("steps", ["20", "24"]), "y": axis("lora_weight", ["0.5", "1"], target=self.instance_id), "z": None}
        experiment_id, _ = self.submit_ok(self.payload(draft), self.http("1"))
        subtitle = self.server.xyz_grid_snapshot(experiment_id)["subtitle"]
        self.assertIn("X：steps", subtitle)
        self.assertIn(f"Y：LoRA 权重 · {xyz_grid.lora_display_name(reference)}", subtitle)
        self.assertNotIn("lora_weight", subtitle)

    def test_whole_grid_export_publishes_one_png_and_manifest_through_the_routes(self):
        self.assertEqual({key: self.server.xyz_capabilities()[key] for key in ("phase", "export", "integration")}, {"phase": "E", "export": True, "integration": True})
        experiment_id = self.completed_experiment()
        estimate = self.server.estimate_xyz_export(experiment_id, {"mode": "scaled", "percent": 50, "arrangement": "grid"})
        self.assertTrue(estimate["budget"]["ok"])
        self.assertEqual(estimate["cells"], {"complete": 4})
        record, (run, args) = self.start_export(experiment_id, {"mode": "original"})
        self.assertEqual(record["status"], "queued")
        run(*args)
        record = self.server.get_xyz_export(experiment_id, record["id"])
        self.assertEqual(record["status"], "complete", record.get("error"))
        png = self.outputs / record["output"]["relative"]
        self.assertTrue(png.is_file() and png.parent.name == "XYZ" and png.name.startswith("XiriusA-XYZPlot-"), (record, sorted(str(p.relative_to(self.outputs)) for p in self.outputs.rglob("*"))))
        self.assertEqual(record["output"]["asset_url"], f"/api/inference/history/assets/{record['output']['asset_id']}")
        manifest = self.server.get_xyz_export_manifest(experiment_id, record["id"])
        self.assertEqual(Path(manifest.path), png.with_suffix(".json"))
        with Image.open(png) as image:
            self.assertEqual(json.loads(image.info["xyz_plot"])["experiment_id"], experiment_id)
            self.assertNotIn("parameters", image.info)
        detail = self.server.get_xyz_experiment(experiment_id)
        self.assertEqual(detail["exports"][-1]["id"], record["id"])
        self.assertTrue(detail["cells"][0]["thumbnail_url"].endswith("/thumbnail"))
        first = self.server.get_xyz_cell_thumbnail(experiment_id, "0:0:0:0")
        second = self.server.get_xyz_cell_thumbnail(experiment_id, "0:0:0:0")
        self.assertEqual(first.path, second.path)
        self.assertTrue(Path(first.path).is_relative_to(self.state / "xyz-plots" / experiment_id / "thumbs"))

    def test_completion_fingerprint_is_final_metadata_and_live_swap_blocks_every_critical_path(self):
        experiment_id = self.completed_experiment()
        store = self.server.xyz_store()
        cell = self.server.get_xyz_cell(experiment_id, '0:0:0:0')
        output = cell['output']
        path = self.outputs / output['relative']
        before = path.read_bytes()
        self.assertEqual(output['asset_fingerprint']['sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(cell['asset_verification']['assurance'], 'sha256')
        # Start an export BEFORE swapping: render must not trust its frozen path/dimensions.
        export, (run, args) = self.start_export(experiment_id, {'mode': 'original'})
        with Image.open(path) as image:
            info = PngImagePlugin.PngInfo()
            info.add_text('parameters', image.info['parameters'])
        Image.new('RGB', (8, 8), 'blue').save(path, pnginfo=info)  # metadata correct, pixels changed
        forged = path.read_bytes()
        run(*args)
        self.assertEqual(self.server.get_xyz_export(experiment_id, export['id'])['status'], 'error')
        self.assertFalse(list((self.outputs / 'XYZ').glob('*.png')))
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.get_history_asset(output['asset_id'])
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.server.get_xyz_cell(experiment_id, '0:0:0:0')['status'], 'missing')
        self.assertEqual(store.grid_snapshot(experiment_id, self.server.xyz_contained_output)['cells'][0]['relative'], None)
        with self.assertRaises(self.server.HTTPException):
            self.server.get_xyz_cell_thumbnail(experiment_id, '0:0:0:0')
        self.assertEqual(path.read_bytes(), forged, 'verification never rewrites the original')
        self.assertEqual(store.cell_detail(experiment_id, '0:0:0:0', str)['output']['asset_fingerprint'], output['asset_fingerprint'])
        self.server.continue_xyz_experiment(experiment_id, {'retry': ['0:0:0:0']})
        run_generation, _ = self.fake_generation()
        with mock.patch.object(self.server, 'run_generation', side_effect=run_generation):
            self.server.run_xyz_experiment(experiment_id)
        retried = self.server.get_xyz_cell(experiment_id, '0:0:0:0')
        self.assertEqual(retried['status'], 'complete')
        self.assertNotEqual(retried['output']['relative'], output['relative'])
        self.assertEqual(path.read_bytes(), forged)

    def test_recovery_rejects_plain_corrupt_wrong_cell_and_same_dimension_replacements(self):
        for kind in ('missing', 'corrupt', 'plain', 'wrong-cell', 'pixels'):
            with self.subTest(kind=kind):
                experiment_id = self.completed_experiment()
                cell = self.server.get_xyz_cell(experiment_id, '0:0:0:0')
                path = self.outputs / cell['output']['relative']
                definition = (self.state / 'xyz-plots' / experiment_id / 'definition.json').read_bytes()
                if kind == 'missing':
                    path.unlink()
                elif kind == 'corrupt':
                    path.write_bytes(b'broken PNG')
                else:
                    info = PngImagePlugin.PngInfo()
                    if kind != 'plain':
                        with Image.open(path) as image:
                            parameters = json.loads(image.info['parameters'])
                        if kind == 'wrong-cell': parameters['xyz_plot']['cell_id'] = 'other'
                        info.add_text('parameters', json.dumps(parameters))
                    Image.new('RGB', (8, 8), 'green').save(path, pnginfo=info)
                with mock.patch.object(self.server, '_xyz_store', None):
                    recovered = self.server.get_xyz_cell(experiment_id, '0:0:0:0')
                    self.assertEqual(recovered['status'], 'missing')
                    self.assertEqual(recovered['asset_verification']['status'], 'needs_repair')
                    self.assertNotIn('asset_url', recovered['output'])
                self.assertEqual((self.state / 'xyz-plots' / experiment_id / 'definition.json').read_bytes(), definition)

    def test_legacy_metadata_only_is_explicit_and_never_backfills_a_digest(self):
        experiment_id = self.completed_experiment()
        store = self.server.xyz_store()
        current = store.record(experiment_id)['state']['cells']['0:0:0:0']
        current['output'].pop('asset_fingerprint')
        path = self.outputs / current['output']['relative']
        with Image.open(path) as image:
            parameters = json.loads(image.info['parameters'])
        for field in ('coordinate', 'request_sha256', 'definition_digest'):
            parameters['xyz_plot'].pop(field)
        info = PngImagePlugin.PngInfo(); info.add_text('parameters', json.dumps(parameters))
        Image.new('RGB', (8, 8), 'red').save(path, pnginfo=info)
        original = path.read_bytes()
        store.save(experiment_id)
        with mock.patch.object(self.server, '_xyz_store', None):
            recovered = self.server.get_xyz_cell(experiment_id, '0:0:0:0')
            self.assertEqual(recovered['status'], 'complete')
            self.assertEqual(recovered['asset_verification']['assurance'], 'legacy_metadata_only')
            self.assertIn('不能证明像素', recovered['asset_verification']['limitation'])
            self.assertNotIn('asset_fingerprint', recovered['output'])
            export, (run, args) = self.start_export(experiment_id, {'mode': 'scaled', 'percent': 50})
            run(*args)
            self.assertEqual(self.server.get_xyz_export(experiment_id, export['id'])['status'], 'complete')
        self.assertEqual(path.read_bytes(), original)

    def test_export_refusals_never_change_the_request_and_cancel_publishes_nothing(self):
        experiment_id = self.completed_experiment()
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.create_xyz_export(experiment_id, {"mode": "original", "percent": 50})
        self.assertEqual(raised.exception.status_code, 422)
        from backend import xyz_grid
        with mock.patch.object(xyz_grid, "EXPORT_MAX_PIXELS", 100):
            with self.assertRaises(self.server.HTTPException) as raised:
                self.server.create_xyz_export(experiment_id, {"mode": "original"})
        self.assertEqual((raised.exception.status_code, raised.exception.detail["code"]), (422, "budget"))
        self.assertIn("suggestions", raised.exception.detail)
        self.assertEqual(self.server.get_xyz_experiment(experiment_id)["exports"], [])
        record, (run, args) = self.start_export(experiment_id, {"mode": "scaled", "percent": 50})
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.create_xyz_export(experiment_id, {"mode": "original"})
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self.server.cancel_xyz_export(experiment_id, record["id"]), {"status": "cancelling"})
        run(*args)
        self.assertEqual(self.server.get_xyz_export(experiment_id, record["id"])["status"], "cancelled")
        folder = self.outputs / "XYZ"
        self.assertFalse(folder.exists() and any(folder.iterdir()), sorted(str(p.relative_to(self.outputs)) for p in self.outputs.rglob("*")))
        pending, _ = self.submit_ok()
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.estimate_xyz_export(pending, {})
        self.assertEqual(raised.exception.detail["code"], "no-complete")

    def test_saved_cell_png_names_its_experiment_and_ordinary_png_does_not(self):
        request = self.server.GenerateInput(engine="SD", checkpoint="model.safetensors", prompt="cat", width=64, height=64,
                                            steps=1, cfg=1, denoise=1, seed=3, sampler="euler", scheduler="normal")
        batch = {"xyz_parent": "b" * 32, "xyz_cell": "0:0:0:0", "xyz_cells": ["0:0:0:0", "1:0:0:0"], "xyz_attempt": 1, "xyz_attempts": {"0:0:0:0": 1, "1:0:0:0": 3}}
        with self.server.jobs_lock:
            self.server.jobs["xyz-batch-job"] = {"id": "xyz-batch-job", "status": "running", **batch}
        path = self.server.save_image(Image.new("RGB", (64, 64)), "xyz-batch-job", request, 1.0, 1.0, 0.0, image_index=2)
        with Image.open(path) as image:
            self.assertEqual(json.loads(image.info["parameters"])["xyz_plot"],
                             {"experiment_id": "b" * 32, "cell_id": "1:0:0:0", "attempt": 3, "batch_size": 2, "batch_index": 1})
        for job_id, extra in [("xyz-cell-job", {"xyz_parent": "a" * 32, "xyz_cell": "1:0:0:0", "xyz_attempt": 2}), ("plain-job", {})]:
            with self.server.jobs_lock:
                self.server.jobs[job_id] = {"id": job_id, "status": "running", **extra}
            path = self.server.save_image(Image.new("RGB", (64, 64)), job_id, request, 1.0, 1.0, 0.0)
            with Image.open(path) as image:
                parameters = json.loads(image.info["parameters"])
            if extra:
                self.assertEqual(parameters["xyz_plot"], {"experiment_id": "a" * 32, "cell_id": "1:0:0:0", "attempt": 2})
            else:
                self.assertNotIn("xyz_plot", parameters)


if __name__ == "__main__":
    unittest.main()
