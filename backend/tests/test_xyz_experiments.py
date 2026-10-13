"""QPlan2/P1 Phase B experiment owner: persistence, recovery and the sequential runner.

Pure: no torch, no inference service. Hooks are fakes that record what the runner asked for.
"""
import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from backend.xyz_experiments import (CellRefused, ExperimentError, ExperimentRunner, ExperimentStopped,
                                     ExperimentStore, HISTORY_LIMIT, cell_prompt_parts, experiment_id_for,
                                     process_owner, project_batches, submission_digest)
from backend.xyz_planner import compile_xyz, plan_record
from backend.tests.test_xyz_planner import axis, baseline, compile as compile_plan


def plan_for(draft):
    calls = iter(range(1, 10_000))
    return plan_record(compile_plan(draft, resolve_seeds=True, random_uint64=lambda: next(calls), validate_request=lambda _: None))


class FakeHooks:
    """Batch-contract fake. `oom_above`: a batch larger than this fails with an OOM result;
    `sampling`: reported sampling batch size (runtime micro-batching)."""

    def __init__(self, store, outcomes=None, stop_at=None, interrupt_at=None, refuse=None, oom_above=None, sampling=None):
        self.store, self.outcomes = store, outcomes or {}
        self.stop_at, self.interrupt_at, self.refuse = stop_at, interrupt_at, refuse or set()
        self.oom_above, self.sampling = oom_above, sampling
        self.ran, self.batches, self.published, self.finished_calls, self.persisted_running = [], [], 0, 0, []
        self.stopped = False

    def checkpoint(self, experiment_id, phase):
        if self.stopped:
            raise ExperimentStopped(interrupted=False)

    def admit(self, experiment_id, cell_ids):
        if any(cell_id in self.refuse for cell_id in cell_ids):
            raise CellRefused("LoRA bytes changed since the confirmed identity")
        return {"cells": list(cell_ids)}

    def run(self, experiment_id, cell_ids, attempts, admitted):
        # The runner must have persisted `running` for every member BEFORE executing the batch.
        on_disk = json.loads((self.store.root / experiment_id / "state.json").read_text(encoding="utf-8"))
        self.persisted_running.extend(on_disk["cells"][cell_id]["status"] for cell_id in cell_ids)
        self.batches.append(list(cell_ids))
        self.ran.extend((cell_id, attempts[cell_id]) for cell_id in cell_ids)
        sampling = self.sampling or len(cell_ids)
        if self.stop_at in cell_ids:
            self.stopped = True
            return {"status": "cancelled", "job_id": "job-stop", "interrupted": False}
        if self.interrupt_at in cell_ids:
            return {"status": "cancelled", "job_id": "job-interrupt", "interrupted": True}
        if self.oom_above is not None and len(cell_ids) > self.oom_above:
            return {"status": "error", "job_id": "job-oom", "error": "GPU memory exhausted", "oom": True, "sampling_batch_size": sampling}
        if any(self.outcomes.get((cell_id, attempts[cell_id]), self.outcomes.get(cell_id)) == "error" for cell_id in cell_ids):
            return {"status": "error", "job_id": f"job-{cell_ids[0]}", "error": "GPU memory exhausted. Try a smaller canvas.", "sampling_batch_size": sampling}
        return {"status": "complete", "job_id": f"job-{cell_ids[0]}-{attempts[cell_ids[0]]}", "sampling_batch_size": sampling, "outputs": {
            cell_id: {"relative": f"xyz/{cell_id.replace(':', '-')}-{attempts[cell_id]}.png", "asset_id": f"asset-{cell_id}-{attempts[cell_id]}",
                      "name": f"{cell_id}.png", "width": 512, "height": 512, "seed": "1"} for cell_id in cell_ids}}

    def publish(self, experiment_id):
        self.published += 1

    def finished(self, experiment_id):
        self.finished_calls += 1


class ExperimentStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="xyz-b-store-"))
        self.store = ExperimentStore(self.root / "xyz-plots")
        draft = baseline(); draft['axes'] = dict(x=axis('steps', ['10', '20']), y=axis('cfg', ['1', '2']))
        self.draft = draft

    def create(self, store=None, draft=None, on_error="pause", submission=None, concurrency=1):
        draft = draft or self.draft
        store = store or self.store
        return store.create(submission_id=submission or str(uuid.uuid4()), draft=draft, on_error=on_error,
                            plan=plan_for(draft), library_revision=0, requested_concurrency=concurrency)

    def test_submission_id_is_the_experiment_and_replay_is_idempotent(self):
        submission = str(uuid.uuid4())
        experiment_id, created = self.create(submission=submission)
        self.assertTrue(created)
        self.assertEqual(experiment_id, uuid.UUID(submission).hex)
        definition = (self.store.root / experiment_id / "definition.json").read_bytes()
        self.assertEqual(self.create(submission=submission), (experiment_id, False))
        other = copy.deepcopy(self.draft); other['axes']['x']['values'][0]['value'] = '30'
        with self.assertRaises(ExperimentError) as raised:
            self.create(draft=other, submission=submission)
        self.assertEqual(raised.exception.code, "submission-conflict")
        self.assertEqual((self.store.root / experiment_id / "definition.json").read_bytes(), definition)
        for bad in ["not-a-uuid", str(uuid.uuid4()).replace("-", ""), " " + str(uuid.uuid4())[1:]]:
            with self.assertRaises(ExperimentError):
                experiment_id_for(bad)

    def test_definition_is_frozen_with_string_seeds_and_order(self):
        experiment_id, _ = self.create()
        definition = json.loads((self.store.root / experiment_id / "definition.json").read_text(encoding="utf-8"))
        self.assertEqual(definition["order"], ["0:0:0:0", "1:0:0:0", "0:1:0:0", "1:1:0:0"])
        self.assertEqual({cell["request"]["seed"] for cell in definition["plan"]["cells"]}, {"18446744073709551615"})
        self.assertEqual(definition["axes"]["x"], {"field": "steps", "labels": ["v0", "v1"]})
        self.assertEqual(definition["digest"], submission_digest(self.draft, "pause"))
        self.assertEqual(definition["execution"], {"policy": "sequential", "order": "z_sample_y_x", "max_concurrency": 10})

    def test_sequential_run_persists_running_before_each_cell_and_definition_never_changes(self):
        experiment_id, _ = self.create()
        before = (self.store.root / experiment_id / "definition.json").read_bytes()
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual([cell for cell, _ in hooks.ran], ["0:0:0:0", "1:0:0:0", "0:1:0:0", "1:1:0:0"])
        self.assertEqual(hooks.persisted_running, ["running"] * 4)
        summary = self.store.summary(experiment_id)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["counts"]["complete"], 4)
        self.assertEqual(hooks.finished_calls, 1)
        self.assertEqual((self.store.root / experiment_id / "definition.json").read_bytes(), before)
        detail = self.store.detail(experiment_id, lambda asset: f"/a/{asset}")
        self.assertEqual(detail["cells"][1]["asset_url"], "/a/asset-1:0:0:0-1")
        self.assertNotIn(str(self.root), json.dumps(detail))

    def test_list_does_not_rehash_but_summary_and_detail_still_check_once(self):
        verified = []
        store = ExperimentStore(self.root / "xyz-integrity", verify_asset=lambda _definition, planned, _current: (
            verified.append(planned["cell_id"]) or {"status": "verified", "assurance": "sha256"}
        ))
        experiment_id, _ = self.create(store=store)
        ExperimentRunner(store, FakeHooks(store)).run(experiment_id)

        self.assertEqual(store.summary(experiment_id)["counts"]["complete"], 4)
        self.assertEqual(len(verified), 4, "an ordinary per-experiment summary still validates its assets")
        verified.clear()
        self.assertEqual(store.list()["total"], 1)
        self.assertEqual(verified, [], "the polled collection must not hash every completed PNG")

        verified.clear()
        detail = store.detail(experiment_id, str)
        self.assertEqual(detail["counts"]["complete"], 4)
        self.assertEqual(len(verified), 4, "one detailed read validates each completed cell once")

    def test_failed_cell_pauses_by_default_then_retry_skip_and_stop(self):
        experiment_id, _ = self.create()
        hooks = FakeHooks(self.store, outcomes={("1:0:0:0", 1): "error"})
        runner = ExperimentRunner(self.store, hooks)
        runner.run(experiment_id)
        summary = self.store.summary(experiment_id)
        self.assertEqual(summary["status"], "paused")
        self.assertEqual(summary["decision"]["cell_id"], "1:0:0:0")
        self.assertEqual(summary["counts"]["complete"], 1)
        with self.assertRaises(ExperimentError):
            self.store.continue_experiment(experiment_id)
        self.assertTrue(self.store.decide(experiment_id, "retry"))
        runner.run(experiment_id)
        # Same frozen request: attempt 2 of the same cell, then the rest; no other cell re-runs.
        self.assertEqual(hooks.ran[2], ("1:0:0:0", 2))
        self.assertEqual(self.store.summary(experiment_id)["status"], "complete")
        cell = self.store.cell_detail(experiment_id, "1:0:0:0", str)
        self.assertEqual([entry["status"] for entry in cell["history"]], ["error", "complete"])

        other, _ = self.create()
        hooks = FakeHooks(self.store, outcomes={"0:0:0:0": "error"})
        ExperimentRunner(self.store, hooks).run(other)
        self.assertTrue(self.store.decide(other, "skip"))
        ExperimentRunner(self.store, hooks).run(other)
        counts = self.store.summary(other)["counts"]
        self.assertEqual((counts["skipped"], counts["complete"]), (1, 3))

        third, _ = self.create()
        ExperimentRunner(self.store, FakeHooks(self.store, outcomes={"1:0:0:0": "error"})).run(third)
        self.assertFalse(self.store.decide(third, "stop"))
        summary = self.store.summary(third)
        self.assertEqual(summary["status"], "cancelled")
        self.assertEqual((summary["counts"]["complete"], summary["counts"]["error"], summary["counts"]["cancelled"]), (1, 1, 2))
        with self.assertRaises(ExperimentError):
            self.store.decide(third, "retry")

    def test_continue_on_error_policy_runs_every_cell(self):
        experiment_id, _ = self.create(on_error="continue")
        hooks = FakeHooks(self.store, outcomes={"0:0:0:0": "error", "1:1:0:0": "error"})
        ExperimentRunner(self.store, hooks).run(experiment_id)
        summary = self.store.summary(experiment_id)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual((summary["counts"]["complete"], summary["counts"]["error"]), (2, 2))
        self.store.continue_experiment(experiment_id, "failed")
        ExperimentRunner(self.store, FakeHooks(self.store)).run(experiment_id)
        self.assertEqual(self.store.summary(experiment_id)["counts"]["complete"], 4)

    def test_execution_admission_refusal_is_a_cell_error_with_reason(self):
        experiment_id, _ = self.create()
        hooks = FakeHooks(self.store, refuse={"0:0:0:0"})
        ExperimentRunner(self.store, hooks).run(experiment_id)
        summary = self.store.summary(experiment_id)
        self.assertEqual(summary["decision"]["cell_id"], "0:0:0:0")
        self.assertIn("LoRA bytes changed", summary["decision"]["error"])
        self.assertEqual(hooks.ran, [])

    def test_global_stop_keeps_completed_originals_and_cancels_the_rest(self):
        experiment_id, _ = self.create()
        hooks = FakeHooks(self.store, stop_at="0:1:0:0")
        ExperimentRunner(self.store, hooks).run(experiment_id)
        summary = self.store.summary(experiment_id)
        self.assertEqual(summary["status"], "cancelled")
        self.assertEqual((summary["counts"]["complete"], summary["counts"]["cancelled"]), (2, 2))
        detail = self.store.detail(experiment_id, str)
        self.assertEqual([cell["status"] for cell in detail["cells"]], ["complete", "complete", "cancelled", "cancelled"])
        # Explicit continuation runs only the unfinished cells; completed originals are untouched.
        self.store.continue_experiment(experiment_id)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual([cell for cell, _ in hooks.ran], ["0:1:0:0", "1:1:0:0"])
        self.assertEqual(self.store.cell_detail(experiment_id, "0:0:0:0", str)["output"]["asset_id"], "asset-0:0:0:0-1")

    def test_complete_cells_are_immutable_and_only_retryable_states_requeue(self):
        experiment_id, _ = self.create(on_error="continue")
        ExperimentRunner(self.store, FakeHooks(self.store, outcomes={"0:0:0:0": "error"})).run(experiment_id)
        with self.assertRaises(ExperimentError) as raised:
            self.store.continue_experiment(experiment_id, ["1:0:0:0"])
        self.assertEqual(raised.exception.code, "immutable-cell")
        with self.assertRaises(ExperimentError) as raised:
            self.store.continue_experiment(experiment_id)
        self.assertEqual(raised.exception.code, "nothing-to-continue")
        self.store.continue_experiment(experiment_id, ["0:0:0:0"])
        self.assertEqual(self.store.summary(experiment_id)["status"], "queued")

    def test_shutdown_interrupt_keeps_cells_pending_for_explicit_continue(self):
        experiment_id, _ = self.create()
        ExperimentRunner(self.store, FakeHooks(self.store, interrupt_at="1:0:0:0")).run(experiment_id)
        summary = self.store.summary(experiment_id)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual((summary["counts"]["complete"], summary["counts"]["pending"]), (1, 3))
        self.store.continue_experiment(experiment_id)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(hooks.ran[0], ("1:0:0:0", 2))

    def test_restart_marks_interruption_checks_assets_and_never_resumes_by_itself(self):
        experiment_id, _ = self.create()
        ExperimentRunner(self.store, FakeHooks(self.store, outcomes={"1:0:0:0": "error"})).run(experiment_id)
        queued, _ = self.create()
        state_path = self.store.root / queued / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "running"; state["cells"]["0:0:0:0"]["status"] = "running"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        present = {"xyz/0-0-0-0-1.png"}
        restarted = ExperimentStore(self.store.root, owner={**process_owner(), "token": "next"},
                                    asset_exists=lambda relative: relative in present)
        summary = restarted.summary(queued)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["counts"]["pending"], 4)
        first = restarted.summary(experiment_id)
        # A decision-paused experiment is interrupted too; its failed cell stays failed, not retried.
        self.assertEqual(first["status"], "interrupted")
        self.assertEqual(first["counts"]["error"], 1)
        self.assertEqual(restarted.cell_detail(experiment_id, "0:0:0:0", str)["status"], "complete")
        self.assertEqual(json.loads((self.store.root / queued / "state.json").read_text(encoding="utf-8"))["status"], "interrupted")

        missing_root = Path(tempfile.mkdtemp(prefix="xyz-b-missing-")) / "xyz-plots"
        store = ExperimentStore(missing_root)
        done, _ = self.create(store=store)
        ExperimentRunner(store, FakeHooks(store)).run(done)
        reread = ExperimentStore(missing_root, owner={**process_owner(), "token": "x"}, asset_exists=lambda relative: False)
        self.assertEqual(reread.summary(done)["counts"]["missing"], 4)
        reread.continue_experiment(done, "failed")
        self.assertEqual(reread.summary(done)["counts"]["pending"], 4)

    def test_live_foreign_owner_is_never_stolen(self):
        experiment_id, _ = self.create()
        state_path = self.store.root / experiment_id / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")); state["status"] = "running"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        before = state_path.read_bytes()
        other = ExperimentStore(self.store.root, owner={**process_owner(), "token": "other"}, owner_alive=lambda owner: True)
        summary = other.summary(experiment_id)
        self.assertEqual((summary["status"], summary["foreign"]), ("running", True))
        self.assertEqual(state_path.read_bytes(), before)
        with self.assertRaises(ExperimentError) as raised:
            other.continue_experiment(experiment_id)
        self.assertEqual(raised.exception.code, "foreign-owner")

    def test_unreadable_and_future_records_are_preserved_raw(self):
        good, _ = self.create()
        broken = self.store.root / ("f" * 32)
        broken.mkdir()
        (broken / "definition.json").write_text("{not json", encoding="utf-8")
        future = self.store.root / ("e" * 32)
        future.mkdir()
        definition = json.loads((self.store.root / good / "definition.json").read_text(encoding="utf-8"))
        state = json.loads((self.store.root / good / "state.json").read_text(encoding="utf-8"))
        (future / "definition.json").write_text(json.dumps({**definition, "experiment_id": "e" * 32, "schema_version": 2}), encoding="utf-8")
        (future / "state.json").write_text(json.dumps({**state, "experiment_id": "e" * 32}), encoding="utf-8")
        before = {path: path.read_bytes() for path in [broken / "definition.json", future / "definition.json"]}
        reread = ExperimentStore(self.store.root, owner={**process_owner(), "token": "y"})
        listing = reread.list()
        self.assertEqual({item["id"] for item in listing["unreadable"]}, {"f" * 32, "e" * 32})
        self.assertEqual([item["id"] for item in listing["experiments"]], [good])
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)
        with self.assertRaises(ExperimentError) as raised:
            reread.detail("f" * 32, str)
        self.assertEqual(raised.exception.code, "unreadable")

    def test_attempt_history_is_bounded(self):
        experiment_id, _ = self.create(draft=baseline())
        for attempt in range(HISTORY_LIMIT + 5):
            ExperimentRunner(self.store, FakeHooks(self.store, outcomes={"0:0:0:0": "error"})).run(experiment_id)
            self.store.decide(experiment_id, "retry")
        cell = self.store.cell_detail(experiment_id, "0:0:0:0", str)
        self.assertEqual(len(cell["history"]), HISTORY_LIMIT)
        self.assertEqual(cell["attempts"], HISTORY_LIMIT + 5)

    def test_pause_requested_before_the_slot_is_kept_not_dropped(self):
        experiment_id, _ = self.create()
        state = self.store.record(experiment_id)["state"]
        state["status"] = "pausing"
        self.store.save(experiment_id)
        hooks = FakeHooks(self.store)
        seen = []
        hooks.checkpoint = lambda _id, _phase: seen.append(self.store.summary(experiment_id)["status"])
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(seen[0], "pausing")
        self.assertEqual(len(hooks.ran), 4)
        self.assertEqual(self.store.summary(experiment_id)["status"], "complete")

    def test_stop_racing_the_slot_settles_as_cancelled(self):
        experiment_id, _ = self.create()
        state = self.store.record(experiment_id)["state"]
        state["status"] = "cancelling"
        self.store.save(experiment_id)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(hooks.ran, [])
        summary = self.store.summary(experiment_id)
        self.assertEqual((summary["status"], summary["counts"]["cancelled"]), ("cancelled", 4))

    def test_stop_while_queued_never_runs_a_cell(self):
        experiment_id, _ = self.create()
        self.store.stop_queued(experiment_id)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(hooks.ran, [])
        self.assertEqual(hooks.finished_calls, 1)
        self.assertEqual(self.store.summary(experiment_id)["counts"]["cancelled"], 4)


class BatchingTests(unittest.TestCase):
    """Phase C: compatible cells share one real batch; everything else stays separate and explained."""

    def setUp(self):
        self.store = ExperimentStore(Path(tempfile.mkdtemp(prefix="xyz-c-store-")) / "xyz-plots")

    def create(self, draft, concurrency, on_error="pause"):
        return self.store.create(submission_id=str(uuid.uuid4()), draft=draft, on_error=on_error, plan=plan_for(draft),
                                 library_revision=0, requested_concurrency=concurrency)[0]

    def seed_draft(self, seeds=("1", "18446744073709551615", "7", "123456789012345"), steps=("10", "20")):
        draft = baseline(); draft["axes"] = dict(x=axis("seed", list(seeds)), y=axis("steps", list(steps)))
        return draft

    def test_seed_cells_share_one_batch_and_other_fields_split_with_reasons(self):
        experiment_id = self.create(self.seed_draft(), 4)
        before = (self.store.root / experiment_id / "definition.json").read_bytes()
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(hooks.batches, [["0:0:0:0", "1:0:0:0", "2:0:0:0", "3:0:0:0"], ["0:1:0:0", "1:1:0:0", "2:1:0:0", "3:1:0:0"]])
        detail = self.store.detail(experiment_id, str)
        self.assertEqual(detail["counts"]["complete"], 8)
        self.assertEqual([cell["batch"]["index"] for cell in detail["cells"]], [0, 1, 2, 3] * 2)
        self.assertEqual(detail["batch_log"][0]["effective"], 4)
        self.assertIn("仅种子不同", detail["batch_log"][0]["reason"])
        # Concurrency never rewrites the frozen definition or adds cells.
        self.assertEqual((self.store.root / experiment_id / "definition.json").read_bytes(), before)
        self.assertEqual(detail["total_cells"], 8)

        uneven = self.create(self.seed_draft(), 3)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(uneven)
        self.assertEqual([len(batch) for batch in hooks.batches], [3, 1, 3, 1])
        log = self.store.detail(uneven, str)["batch_log"]
        self.assertIn("步数", log[1]["reason"])
        self.assertEqual(log[1]["effective"], 1)

    def test_repeat_samples_batch_across_samples_without_reordering_coordinates(self):
        draft = baseline(); draft["axes"] = dict(x=axis("cfg", ["5", "7"])); draft["repeat_count"] = 3
        draft["baseline"]["seed_mode"] = "random"
        experiment_id = self.create(draft, 10)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(hooks.batches, [["0:0:0:0", "0:0:0:1", "0:0:0:2"], ["1:0:0:0", "1:0:0:1", "1:0:0:2"]])
        self.assertIn("只剩 3 个待生成格", self.store.detail(experiment_id, str)["batch_log"][1]["reason"])
        self.assertEqual([cell["id"] for cell in self.store.detail(experiment_id, str)["cells"]][:3], ["0:0:0:0", "1:0:0:0", "0:0:0:1"])

    def test_oom_splits_in_halves_with_same_seeds_then_fails_one_cell_by_policy(self):
        experiment_id = self.create(self.seed_draft(steps=("10",)), 4)
        hooks = FakeHooks(self.store, oom_above=2)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual([len(batch) for batch in hooks.batches], [4, 2, 2])
        detail = self.store.detail(experiment_id, str)
        self.assertEqual(detail["counts"]["complete"], 4)
        self.assertEqual([entry["oom"] for entry in detail["batch_log"]], [True, False, False])
        self.assertIn("由 4 张拆分重试", detail["batch_log"][1]["reason"])
        cell = self.store.cell_detail(experiment_id, "0:0:0:0", str)
        self.assertEqual([entry["status"] for entry in cell["history"]], ["oom-split", "complete"])
        self.assertEqual(cell["attempts"], 2)

        failing = self.create(self.seed_draft(steps=("10",)), 4)
        hooks = FakeHooks(self.store, oom_above=0)
        ExperimentRunner(self.store, hooks).run(failing)
        self.assertEqual([len(batch) for batch in hooks.batches], [4, 2, 1])
        summary = self.store.summary(failing)
        self.assertEqual((summary["status"], summary["decision"]["cell_ids"]), ("paused", ["0:0:0:0"]))

    def test_runtime_micro_batch_is_reported_as_actual_concurrency(self):
        experiment_id = self.create(self.seed_draft(steps=("10",)), 4)
        ExperimentRunner(self.store, FakeHooks(self.store, sampling=1)).run(experiment_id)
        summary = self.store.summary(experiment_id)
        self.assertEqual((summary["requested_concurrency"], summary["effective_concurrency"]), (4, 1))
        self.assertIn("每批 1 张采样", summary["concurrency_reason"])
        # X10: a micro-batched run is never advertised as N, not even before a correction.
        self.assertTrue(summary["concurrency_reason"].startswith("请求 4 / 实际 1："), summary["concurrency_reason"])
        self.assertNotIn("实际 4", summary["concurrency_reason"])
        self.assertEqual(self.store.cell_detail(experiment_id, "2:0:0:0", str)["output"]["batch"]["effective"], 1)

    def test_refused_compatible_batch_never_counts_planned_members_as_sampling(self):
        experiment_id = self.create(self.seed_draft(steps=('10',)), 4)
        hooks = FakeHooks(self.store, refuse={'0:0:0:0'})
        ExperimentRunner(self.store, hooks).run(experiment_id)
        detail = self.store.detail(experiment_id, str)
        batch = detail['batch_log'][0]
        self.assertEqual((batch['size'], batch['effective'], batch['sampling_batch_size'], batch['sampling_status']), (4, 0, None, 'not_started'))
        self.assertEqual(detail['effective_concurrency'], 0)
        self.assertIn('采样尚未开始', batch['reason'])
        self.assertEqual(hooks.ran, [])

    def test_queued_stopped_and_unmarked_execution_never_invent_an_actual_batch(self):
        experiment_id = self.create(self.seed_draft(steps=('10',)), 4)
        self.assertIsNone(self.store.summary(experiment_id)['effective_concurrency'])
        self.store.stop_queued(experiment_id)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(hooks.ran, [])
        self.assertIsNone(self.store.summary(experiment_id)['effective_concurrency'])
        for outcome in ({'status': 'error', 'error': 'before sampling'}, {'status': 'cancelled'},
                        {'status': 'error', 'error': 'partial failure', 'sampling_batch_size': 2},
                        {'status': 'error', 'oom': True, 'sampling_batch_size': 4}):
            with self.subTest(outcome=outcome):
                experiment_id = self.create(self.seed_draft(steps=('10',)), 4)
                hooks = FakeHooks(self.store)
                hooks.run = lambda *args: dict(outcome)
                ExperimentRunner(self.store, hooks).run(experiment_id)
                batch = self.store.detail(experiment_id, str)['batch_log'][0]
                self.assertEqual(batch['effective'], outcome.get('sampling_batch_size'))
                self.assertEqual(batch['sampling_status'], 'reported' if outcome.get('sampling_batch_size') else 'unknown')

    def test_failed_batch_decision_covers_every_member(self):
        experiment_id = self.create(self.seed_draft(steps=("10",)), 4)
        runner = ExperimentRunner(self.store, FakeHooks(self.store, outcomes={("1:0:0:0", 1): "error"}))
        runner.run(experiment_id)
        summary = self.store.summary(experiment_id)
        self.assertEqual(summary["decision"]["cell_ids"], ["0:0:0:0", "1:0:0:0", "2:0:0:0", "3:0:0:0"])
        self.assertEqual(summary["counts"]["error"], 4)
        self.store.decide(experiment_id, "retry")
        self.assertEqual(self.store.summary(experiment_id)["counts"]["pending"], 4)
        runner.run(experiment_id)
        self.assertEqual(self.store.summary(experiment_id)["counts"]["complete"], 4)

    def test_group_by_resources_reduces_switches_but_keeps_coordinates(self):
        draft = baseline(); draft["axes"] = dict(x=axis("checkpoint", ["a.safetensors", "b.safetensors"]), y=axis("steps", ["10", "20"]))
        catalog = {("SD", "checkpoint", "a.safetensors"), ("SD", "checkpoint", "b.safetensors")}
        def make(group):
            value = copy.deepcopy(draft); value["group_by_resources"] = group
            plan = plan_record(compile_xyz(value, model_catalog=catalog, resolve_seeds=True, random_uint64=lambda: 1, validate_request=lambda _: None))
            return self.store.create(submission_id=str(uuid.uuid4()), draft=value, on_error="pause", plan=plan, library_revision=0, requested_concurrency=1)[0]
        logical, grouped = FakeHooks(self.store), FakeHooks(self.store)
        ExperimentRunner(self.store, logical).run(make(False))
        ExperimentRunner(self.store, grouped).run(make(True))
        self.assertEqual([cell for cell, _ in logical.ran], ["0:0:0:0", "1:0:0:0", "0:1:0:0", "1:1:0:0"])
        self.assertEqual([cell for cell, _ in grouped.ran], ["0:0:0:0", "0:1:0:0", "1:0:0:0", "1:1:0:0"])

    def test_preview_projection_is_the_runner_grouping(self):
        draft = self.seed_draft()
        projected = project_batches(plan_for(draft)["cells"], 3)
        experiment_id = self.create(draft, 3)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual([batch["cells"] for batch in projected], hooks.batches)

    def test_continuation_may_set_a_new_limit_but_never_an_invalid_one(self):
        experiment_id = self.create(self.seed_draft(steps=("10",)), 1)
        ExperimentRunner(self.store, FakeHooks(self.store, stop_at="1:0:0:0")).run(experiment_id)
        with self.assertRaises(ExperimentError):
            self.store.continue_experiment(experiment_id, "none", 11)
        self.store.continue_experiment(experiment_id, "none", 4)
        hooks = FakeHooks(self.store)
        ExperimentRunner(self.store, hooks).run(experiment_id)
        self.assertEqual(hooks.batches, [["1:0:0:0", "2:0:0:0", "3:0:0:0"]])
        self.assertEqual(self.store.summary(experiment_id)["requested_concurrency"], 4)


class CellPromptPartsTests(unittest.TestCase):
    """Phase E: what Apply-to-Basic may write into the prompt box without doubling combination words."""

    def parts(self, draft):
        return [cell_prompt_parts(draft, cell) for cell in plan_for(draft)["cells"]]

    def test_baseline_group_words_lead_every_cell_and_are_split_off(self):
        draft = baseline(); draft['axes'] = dict(x=axis('steps', ['10', '20']))
        for part in self.parts(draft):
            self.assertEqual((part["removed"], part["lead"], part["lead_source"], part["rest"]), (True, "group words", "groups", "cat, blue sky"))
            self.assertEqual(part["prompt"], "group words\n\ncat, blue sky")

    def test_a_profile_that_includes_its_preset_text_leads_with_that_text_instead(self):
        draft = baseline()
        draft['axes'] = dict(x=axis('lora_profile', [dict(mode='full', members=[]), dict(mode='full', members=[], preset_prompt='new words', includePresetText=True)]))
        first, second = self.parts(draft)
        self.assertEqual((first["lead"], first["lead_source"], first["rest"]), ("group words", "groups", "cat, blue sky"))
        self.assertEqual((second["prompt"], second["lead"], second["lead_source"], second["rest"]), ("new words\n\ncat, blue sky", "new words", "profile", "cat, blue sky"))

    def test_a_prompt_that_does_not_start_with_the_words_is_returned_whole(self):
        replaced = baseline(); replaced['axes'] = dict(x=axis('prompt', ['dog in snow']))
        edited = baseline(); edited['axes'] = dict(x=axis('prompt_replace', [dict(find='group', replace='band', expected_matches=1)]))
        bare = baseline(); bare['baseline']['group_prompt'] = ''; bare['baseline']['parameters']['prompt'] = 'cat, blue sky'
        for draft, prompt in [(replaced, 'dog in snow'), (edited, 'band words\n\ncat, blue sky'), (bare, 'cat, blue sky')]:
            [part] = self.parts(draft)
            self.assertEqual((part["removed"], part["lead"], part["rest"], part["prompt"]), (False, "", prompt, prompt))
            self.assertEqual(part["expected_lead"], "" if draft is bare else "group words")

    def test_cell_detail_carries_the_split(self):
        store = ExperimentStore(Path(tempfile.mkdtemp(prefix="xyz-e-parts-")) / "xyz-plots")
        draft = baseline(); draft['axes'] = dict(x=axis('steps', ['10']))
        experiment_id, _ = store.create(submission_id=str(uuid.uuid4()), draft=draft, on_error="pause", plan=plan_for(draft), library_revision=0, requested_concurrency=1)
        detail = store.cell_detail(experiment_id, "0:0:0:0", str)
        self.assertEqual((detail["prompt_parts"]["rest"], detail["prompt_parts"]["negative_prompt"]), ("cat, blue sky", "bad"))


if __name__ == "__main__":
    unittest.main()
