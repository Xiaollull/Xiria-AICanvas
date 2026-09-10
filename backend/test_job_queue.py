"""Submitting while a job runs queues the new one instead of refusing it.

The worker pool has always been a single thread, so jobs already ran one at a time; what stopped a
second submission was a 409 in `create_job`, which meant the user had to sit and watch a run finish
before setting up the next one. Removing it turns that pool into a real queue, and this file pins
what the queue then has to guarantee: submissions are numbered from one per session, they keep the
parameters captured at submit time, a job cancelled while still waiting never loads a model, and
the list reports what happened to every job the session has seen.

The registry is manipulated directly rather than through a running server, which is how
`test_anima` already tests the same lock: the behaviour under test is bookkeeping, and standing up
an HTTP server would test FastAPI instead.
"""

import time
import unittest
from unittest.mock import patch

from backend import inference_server
from backend.inference_server import (
    JOB_ACTIVE_STATUSES,
    JOB_FINISHED_STATUSES,
    job_queue_entry,
    public_job_view,
    trim_job_history,
)


def job_record(sequence, status, **overrides):
    record = {
        "id": f"job-{sequence}",
        "sequence": sequence,
        "status": status,
        "phase": "",
        "stage": "",
        "progress": 0,
        "outputs": [],
        "created_at": 1000.0 + sequence,
    }
    record.update(overrides)
    return record


class RegistryTestCase(unittest.TestCase):
    """Each test owns the registry: it is module state the real server shares between requests."""

    def setUp(self):
        self._jobs = dict(inference_server.jobs)
        self._controls = dict(inference_server.job_controls)
        inference_server.jobs.clear()
        inference_server.job_controls.clear()

    def tearDown(self):
        inference_server.jobs.clear()
        inference_server.jobs.update(self._jobs)
        inference_server.job_controls.clear()
        inference_server.job_controls.update(self._controls)

    def install(self, *records):
        for record in records:
            inference_server.jobs[record["id"]] = record


class QueueListingTests(RegistryTestCase):
    def test_the_list_is_oldest_first_and_counts_every_state(self):
        self.install(
            job_record(1, "complete"),
            job_record(2, "error", error="boom"),
            job_record(3, "cancelled"),
            job_record(4, "running"),
            job_record(5, "queued"),
            job_record(6, "queued"),
        )
        payload = inference_server.list_jobs()
        self.assertEqual([entry["sequence"] for entry in payload["jobs"]], [1, 2, 3, 4, 5, 6])
        self.assertEqual(payload["total"], 6)
        self.assertEqual(
            payload["counts"],
            {"queued": 2, "running": 1, "complete": 1, "error": 1, "cancelled": 1},
        )

    def test_a_waiting_job_is_distinguished_from_the_one_being_worked_on(self):
        self.install(job_record(1, "running"), job_record(2, "queued"))
        entries = {entry["sequence"]: entry for entry in inference_server.list_jobs()["jobs"]}
        self.assertFalse(entries[1]["waiting"])
        self.assertTrue(entries[1]["active"])
        self.assertTrue(entries[2]["waiting"])
        self.assertTrue(entries[2]["active"])
        # The second knows something is ahead of it; the first knows nothing is.
        self.assertFalse(entries[1]["running_ahead"])
        self.assertTrue(entries[2]["running_ahead"])

    def test_a_completed_row_carries_what_it_produced(self):
        outputs = [{"index": 0, "output_name": "XirAI-1.png", "image_url": "/x/0", "seed": "7"}]
        self.install(job_record(1, "complete", outputs=outputs, elapsed_seconds=12.5))
        entry = inference_server.list_jobs()["jobs"][0]
        self.assertEqual(entry["elapsed_seconds"], 12.5)
        self.assertEqual(entry["outputs"][0]["output_name"], "XirAI-1.png")
        self.assertEqual(entry["outputs"][0]["image_url"], "/x/0")

    def test_a_failed_row_carries_its_reason(self):
        self.install(job_record(1, "error", error="CUDA out of memory"))
        self.assertEqual(inference_server.list_jobs()["jobs"][0]["error"], "CUDA out of memory")

    def test_an_empty_session_lists_nothing(self):
        payload = inference_server.list_jobs()
        self.assertEqual(payload["jobs"], [])
        self.assertEqual(payload["total"], 0)


class ActiveJobTests(RegistryTestCase):
    def test_the_active_job_is_the_one_being_worked_on_not_the_newest(self):
        """A client recovering its progress view wants the job that is actually moving. Before the
        queue those were the same job; now the newest submission is usually still waiting."""
        self.install(job_record(1, "running"), job_record(2, "queued"), job_record(3, "queued"))
        self.assertEqual(inference_server.get_active_job()["job"]["sequence"], 1)

    def test_with_nothing_started_the_front_of_the_queue_is_reported(self):
        self.install(job_record(4, "queued"), job_record(5, "queued"))
        self.assertEqual(inference_server.get_active_job()["job"]["sequence"], 4)

    def test_finished_jobs_are_not_active(self):
        self.install(job_record(1, "complete"), job_record(2, "cancelled"), job_record(3, "error"))
        self.assertIsNone(inference_server.get_active_job()["job"])

    def test_the_status_sets_do_not_overlap(self):
        self.assertEqual(JOB_ACTIVE_STATUSES & JOB_FINISHED_STATUSES, frozenset())


class CancellationTests(RegistryTestCase):
    def test_cancelling_a_waiting_job_finishes_it_immediately(self):
        """Nothing has started, so there is no current step to stop after. Left as "cancelling" it
        would sit in "stopping" until the worker reached it, which behind a queue is minutes."""
        control = inference_server.JobControl()
        inference_server.job_controls["job-1"] = control
        self.install(job_record(1, "queued"))
        result = inference_server.cancel_job("job-1")
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(inference_server.jobs["job-1"]["status"], "cancelled")
        self.assertTrue(control.cancelled)

    def test_cancelling_a_running_job_still_stops_after_the_current_step(self):
        control = inference_server.JobControl()
        inference_server.job_controls["job-1"] = control
        self.install(job_record(1, "running"))
        self.assertEqual(inference_server.cancel_job("job-1")["status"], "cancelling")
        self.assertEqual(inference_server.jobs["job-1"]["status"], "cancelling")

    def test_a_job_cancelled_before_the_worker_reaches_it_loads_nothing(self):
        control = inference_server.JobControl()
        control.cancel()
        inference_server.job_controls["job-1"] = control
        self.install(job_record(1, "cancelling"))
        # `parse_prompt_directives` is the first thing the old entry point did. Reaching it would
        # mean the early exit had not fired.
        with patch.object(inference_server, "parse_prompt_directives") as parsed:
            inference_server.run_generation("job-1", object())
        parsed.assert_not_called()
        self.assertEqual(inference_server.jobs["job-1"]["status"], "cancelled")


class HistoryTrimTests(RegistryTestCase):
    def test_finished_jobs_are_trimmed_oldest_first(self):
        self.install(*(job_record(index, "complete") for index in range(1, 12)))
        with patch.object(inference_server, "JOB_HISTORY_LIMIT", 5):
            trim_job_history()
        remaining = sorted(job["sequence"] for job in inference_server.jobs.values())
        self.assertEqual(remaining, [7, 8, 9, 10, 11])

    def test_a_waiting_job_is_never_trimmed(self):
        """A queued job dropped from the registry would still run, with nothing to report to."""
        self.install(*(job_record(index, "complete") for index in range(1, 8)))
        self.install(job_record(0, "queued"), job_record(99, "running"))
        with patch.object(inference_server, "JOB_HISTORY_LIMIT", 2):
            trim_job_history()
        statuses = sorted(job["status"] for job in inference_server.jobs.values())
        self.assertEqual(statuses, ["complete", "complete", "queued", "running"])

    def test_trimming_forgets_the_stage_pictures_too(self):
        self.install(*(job_record(index, "complete") for index in range(1, 5)))
        removed = []
        with (
            patch.object(inference_server, "JOB_HISTORY_LIMIT", 1),
            patch.object(inference_server, "cleanup_job_stage_previews", removed.append),
        ):
            trim_job_history()
        self.assertEqual(removed, ["job-1", "job-2", "job-3"])


class PublicViewTests(RegistryTestCase):
    def test_local_paths_never_reach_a_client(self):
        job = job_record(1, "complete",
                         output_path="/tmp/a.png", output_paths=["/tmp/a.png"],
                         stage_preview_paths=["/tmp/s.jpg"],
                         stage_previews=[{"index": 0, "stage": "base", "url": "/api/x"}])
        view = public_job_view(job)
        for field in ("output_path", "output_paths", "stage_preview_paths"):
            self.assertNotIn(field, view)
        # The stage list itself is public: it carries urls, not filenames.
        self.assertEqual(view["stage_previews"][0]["url"], "/api/x")

    def test_a_queue_entry_reports_the_model_the_job_captured(self):
        job = job_record(1, "queued", requested_engine="Krea2", requested_checkpoint=None,
                         requested_model_assets={"diffusion_model": "krea2.safetensors"},
                         width=1536, height=1152)
        entry = job_queue_entry(job, running_seen=False)
        self.assertEqual(entry["engine"], "Krea2")
        self.assertEqual(entry["model_assets"]["diffusion_model"], "krea2.safetensors")
        self.assertEqual((entry["width"], entry["height"]), (1536, 1152))


class SequenceTests(unittest.TestCase):
    def test_numbering_counts_up_from_one_and_never_repeats(self):
        import itertools

        with patch.object(inference_server, "job_sequence", itertools.count(1)):
            drawn = [next(inference_server.job_sequence) for _ in range(4)]
        self.assertEqual(drawn, [1, 2, 3, 4])


class WithdrawnLatentPreviewTests(unittest.TestCase):
    """The latent process preview is gone; the stage previews it sat beside are not.

    These two moved together for a long time, so the risk in withdrawing one is quietly taking the
    other with it. The first test pins the request contract, the rest pin the machinery that
    publishes a picture after each stage finishes while the next stage is already running.
    """

    def test_the_request_no_longer_carries_a_preview_switch(self):
        self.assertNotIn("preview_enabled", inference_server.GenerateInput.model_fields)

    def test_a_request_that_still_asks_for_one_is_refused_rather_than_ignored(self):
        # `extra="forbid"` is what makes the removal real: a client built against the old contract
        # is told its field is gone instead of silently getting no previews.
        with self.assertRaises(Exception) as error:
            inference_server.GenerateInput(
                engine="Krea2", checkpoint=None, diffusion_model="k.safetensors",
                text_encoder="t.safetensors", vae="v.safetensors",
                prompt="x", negative_prompt="", width=1024, height=1024,
                steps=8, cfg=1.0, denoise=1.0, seed=1, sampler="euler", scheduler="simple",
                preview_enabled=True,
            )
        self.assertIn("preview_enabled", str(error.exception))

    def test_the_stage_preview_route_survived_the_removal(self):
        routes = {getattr(route, "path", "") for route in inference_server.app.routes}
        self.assertIn("/api/inference/jobs/{job_id}/stages/{index}", routes)
        self.assertNotIn("/api/inference/jobs/{job_id}/preview", routes)

    def test_the_stage_preview_writer_survived_the_removal(self):
        for name in ("save_stage_preview", "cleanup_job_stage_previews", "STAGE_PREVIEW_LABELS"):
            self.assertTrue(hasattr(inference_server, name), name)
        for name in ("save_latent_preview", "save_pil_preview", "cleanup_job_previews"):
            self.assertFalse(hasattr(inference_server, name), name)

    def test_internal_folders_stay_out_of_the_output_browser(self):
        # `.stages` was missing from the old single-folder guard, so the browser listed it.
        for folder in (inference_server.STAGE_PREVIEW_DIRECTORY, inference_server.LEGACY_PREVIEW_DIRECTORY):
            self.assertTrue(inference_server.is_internal_output_directory(folder), folder)
        self.assertFalse(inference_server.is_internal_output_directory(inference_server.OUTPUT_DIRECTORY))


if __name__ == "__main__":
    unittest.main()
