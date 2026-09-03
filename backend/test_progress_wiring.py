import unittest

from backend import inference_server
from backend.progress_console import ProgressConsole


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RecordingConsole(ProgressConsole):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lines = []
        self.lock_free = []

    def write(self, line):
        # The whole reason the line is composed under the lock and printed outside it: a sampler
        # callback is on this thread, and every other thread wanting the job record is behind it.
        acquired = inference_server.jobs_lock.acquire(blocking=False)
        self.lock_free.append(acquired)
        if acquired:
            inference_server.jobs_lock.release()
        self.lines.append(line.text)


class UpdateJobTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.console = RecordingConsole(clock=self.clock)
        self.replaced = inference_server.progress_console
        inference_server.progress_console = self.console
        with inference_server.jobs_lock:
            inference_server.jobs["progress-test"] = {"status": "running"}

    def tearDown(self):
        inference_server.progress_console = self.replaced
        with inference_server.jobs_lock:
            inference_server.jobs.pop("progress-test", None)

    def test_a_step_update_reaches_the_console_and_the_job_record(self):
        inference_server.update_job(
            "progress-test", phase="Batch 1/1 · Sampling 1 images", stage="base_sampling",
            stage_step=0, stage_total=22,
        )
        self.clock.now = 8.0
        inference_server.update_job("progress-test", stage_step=4, progress=30)

        self.assertEqual(len(self.console.lines), 2)
        self.assertIn("4/22", self.console.lines[1])
        self.assertIn("2.00s/it", self.console.lines[1])
        with inference_server.jobs_lock:
            self.assertEqual(inference_server.jobs["progress-test"]["stage_step"], 4)
            self.assertEqual(inference_server.jobs["progress-test"]["progress"], 30)

    def test_the_line_is_printed_after_the_lock_is_released(self):
        inference_server.update_job("progress-test", phase="Sampling", stage="base_sampling", stage_step=0, stage_total=20)
        self.assertEqual(self.console.lock_free, [True])

    def test_mounted_adapters_reach_the_console_and_the_job_record(self):
        inference_server.report_mounted_loras("progress-test", [
            (inference_server.PROJECT_ROOT / "package.json", 0.75),
            (inference_server.PROJECT_ROOT / "no-such-lora.safetensors", -0.5),
        ])
        self.assertEqual(len(self.console.lines), 1, "the stack is one entry, not one per adapter")
        self.assertIn("LoRA · 2 mounted", self.console.lines[0])
        self.assertIn("0.75 · package.json", self.console.lines[0])
        # A file whose size cannot be read is still mounted; only its size is missing.
        self.assertIn("-0.50 · no-such-lora.safetensors", self.console.lines[0])
        with inference_server.jobs_lock:
            mounted = inference_server.jobs["progress-test"]["mounted_loras"]
        self.assertEqual([item["name"] for item in mounted], ["package.json", "no-such-lora.safetensors"])
        self.assertIsNone(mounted[1]["bytes"])

    def test_a_job_without_adapters_writes_no_record(self):
        inference_server.report_mounted_loras("progress-test", [])
        self.assertEqual(self.console.lines, [])
        with inference_server.jobs_lock:
            self.assertNotIn("mounted_loras", inference_server.jobs["progress-test"])

    def test_an_update_with_nothing_to_report_prints_nothing(self):
        inference_server.update_job("progress-test", phase="Loading checkpoint", progress=4)
        self.assertEqual(self.console.lines, [])

    def test_an_update_for_a_job_that_is_gone_changes_nothing(self):
        inference_server.update_job("no-such-job", stage_step=1, stage_total=10)
        self.assertEqual(self.console.lines, [])
        with inference_server.jobs_lock:
            self.assertNotIn("no-such-job", inference_server.jobs)

    def test_the_console_ships_enabled_with_a_line_budget(self):
        # Off by default would mean the terminal stayed as empty as it was before.
        self.assertTrue(self.replaced.enabled)
        self.assertEqual(self.replaced.budget, 12)

    def test_the_shipped_console_can_take_a_memory_reading(self):
        # The model line is the one place a reading is taken; if it is not wired up the line still
        # prints, silently missing the numbers it exists to carry.
        self.assertIsNotNone(self.replaced._memory)
        snapshot = inference_server.model_memory_snapshot()
        self.assertIn("ram_total_mb", snapshot)


if __name__ == "__main__":
    unittest.main()
