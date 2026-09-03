import io
import unittest

from backend.progress_console import (
    Line,
    ProgressConsole,
    format_bar,
    format_clock,
    format_loras,
    format_model,
    format_progress,
    format_rate,
    format_size,
    format_summary,
    line_stride,
)


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


class Job(dict):
    """A job record and the updates merged into it, the way ``update_job`` sees them."""

    def apply(self, **updates):
        self.update(updates)
        return updates


def sampling_console(**kwargs):
    clock = FakeClock()
    return ProgressConsole(clock=clock, **kwargs), clock


def run_steps(console, job, clock, total, per_step=1.0, **opening):
    """Drive a whole run and return the lines it printed."""
    lines = []
    opened = console.observe("job-1", job, job.apply(stage_step=0, stage_total=total, **opening))
    if opened:
        lines.append(opened)
    for step in range(1, total + 1):
        clock.advance(per_step)
        line = console.observe("job-1", job, job.apply(stage_step=step))
        if line:
            lines.append(line)
    return lines


class FormattingTest(unittest.TestCase):
    def test_clock_grows_a_field_only_when_the_run_passes_an_hour(self):
        self.assertEqual(format_clock(0), "00:00")
        self.assertEqual(format_clock(86.4), "01:26")
        self.assertEqual(format_clock(3600), "1:00:00")

    def test_an_unknown_duration_reads_as_unknown_rather_than_zero(self):
        # The first line of a run has no rate yet, so it has no remaining time either. Printing
        # 00:00 there would claim the run was about to finish.
        for value in (None, -1, float("nan"), float("inf"), "soon"):
            self.assertEqual(format_clock(value), "--:--")

    def test_the_rate_stays_in_seconds_per_unit_however_fast_the_step_is(self):
        self.assertEqual(format_rate(2.9), "2.90s/it")
        self.assertEqual(format_rate(0.031), "0.03s/it")
        self.assertEqual(format_rate(50.87, "tile"), "50.87s/tile")
        self.assertEqual(format_rate(None), "?s/it")

    def test_the_bar_is_solid_blocks_against_blank_space(self):
        # Blocks read as one filled bar at a glance; the hashes and dashes they replaced read as a
        # row of separate marks, which is what made a screen of them so dense.
        self.assertEqual(format_bar(0, width=10), " " * 10)
        self.assertEqual(format_bar(0.5, width=10), "█████     ")
        self.assertEqual(format_bar(1, width=10), "█" * 10)
        # 99.9% rounds to a full bar; truncating keeps "full" meaning "done".
        self.assertEqual(format_bar(0.999, width=10), "█████████ ")
        self.assertEqual(format_bar(4, width=10), "█" * 10)

    def test_the_bar_survives_the_code_pages_this_ships_to(self):
        bar = format_bar(0.5, width=4)
        for encoding in ("utf-8", "gbk", "cp437"):
            self.assertEqual(bar.encode(encoding).decode(encoding), bar, encoding)

    def test_sizes_read_in_the_unit_that_suits_them(self):
        self.assertEqual(format_size(6_940_000_000), "6.46 GB")
        self.assertEqual(format_size(241 * 1024**2), "241 MB")
        self.assertEqual(format_size(None), "?")

    def test_a_progress_line_carries_step_elapsed_and_rate(self):
        line = format_progress("Batch 1/2 · Sampling 4 images", 20, 30, elapsed=57.4, rate=2.9)
        self.assertIn("20/30", line)
        self.assertIn("[00:57<00:29, 2.90s/it]", line)
        self.assertIn(" 66%|", line)

    def test_every_field_before_the_brackets_holds_one_column(self):
        # The bar has to sit in the same columns whatever wrote the line: a 30-step sampling run, a
        # 56-tile upscale and a 15-step tile all put their bar and their bracket in one place, so
        # the eye follows the fill rather than the line shifting under it.
        lines = [
            format_progress("Sampling", 8, 150, elapsed=10, rate=1.25),
            format_progress("Sampling", 148, 150, elapsed=185, rate=1.25),
            format_progress("Hires.fix · Upscaling 2176 x 2816", 3, 56, elapsed=1, rate=0.07, unit="tile"),
            format_progress("Hires.fix · Tile 2/4", 15, 15, elapsed=52, rate=3.5),
            format_progress("Batch 1/1 · Sampling 1 images", 0, 30),
        ]
        self.assertEqual(len({line.index("|") for line in lines}), 1, "the bar opens in one column")
        self.assertEqual(len({line.index("[") for line in lines}), 1, "and the brackets follow in one more")
        self.assertIn("  8/150", lines[0])
        self.assertIn("148/150", lines[1])

    def test_the_job_counter_is_appended_only_when_there_is_one(self):
        self.assertNotIn("job", format_progress("Sampling", 20, 30, elapsed=57.4, rate=2.9))
        with_job = format_progress(
            "face_yolov8n · Inpainting 1/2", 8, 12, elapsed=23.0, rate=2.87,
            overall=(46, 54), job_elapsed=135.0,
        )
        self.assertTrue(with_job.endswith("job 46/54, 02:15"), with_job)

    def test_the_model_line_states_the_wait_the_size_and_the_memory(self):
        line = format_model(
            "Model ready · NORMAL_VRAM", seconds=12.43, weight_bytes=6_940_000_000,
            memory={"vram_used_mb": 7384, "vram_total_mb": 8188, "ram_used_mb": 11576, "ram_total_mb": 32662},
        )
        self.assertIn("loaded in 12.4s", line)
        self.assertIn("weights 6.46 GB", line)
        self.assertIn("VRAM 7.21/8.00 GB", line)
        self.assertIn("RAM 11.30/31.90 GB", line)

    def test_a_reused_model_says_so_and_survives_missing_readings(self):
        line = format_model("Reusing loaded model", reused=True, seconds=0.2, memory={})
        self.assertIn("reused in 0.2s", line)
        self.assertNotIn("VRAM", line, "a reading that could not be taken is left out, not zeroed")

    def test_mounted_loras_are_grouped_under_one_heading(self):
        block = format_loras([
            {"name": "kazutake-hazano_v2_epoch28.safetensors", "weight": 0.75, "bytes": 152_000_000},
            {"name": "ashima_v4.safetensors", "weight": 0.25, "bytes": 220_000_000},
        ])
        heading, second = block.split("\n")
        self.assertTrue(heading.startswith("LoRA · 2 mounted"), heading)
        self.assertIn("0.75 · kazutake-hazano_v2_epoch28.safetensors · 145 MB", heading)
        # The continuation is indented into the phase column, so the stack reads as one entry.
        self.assertEqual(second.index("0.25"), heading.index("0.75"))

    def test_lora_weights_line_up_including_negative_ones(self):
        block = format_loras([
            {"name": "a.safetensors", "weight": -0.5, "bytes": None},
            {"name": "b.safetensors", "weight": 1.0, "bytes": None},
        ])
        first, second = block.split("\n")
        self.assertIn("-0.50 · a.safetensors", first)
        self.assertIn(" 1.00 · b.safetensors", second)
        self.assertNotIn("MB", block, "a size that could not be read is left out, not printed as zero")

    def test_nothing_mounted_prints_nothing(self):
        self.assertIsNone(format_loras([]))
        self.assertIsNone(format_loras(None))

    def test_the_summary_states_what_the_run_cost_and_what_it_produced(self):
        summary = format_summary("complete", images=4, steps=30, elapsed=86.4, output="ComfyUI_0002.png")
        self.assertIn("Complete", summary)
        self.assertIn("4 images", summary)
        self.assertIn("01:26 elapsed", summary)
        self.assertIn("2.88s/it average", summary)
        self.assertTrue(summary.endswith("ComfyUI_0002.png"), summary)
        self.assertIn("1 image ", format_summary("complete", images=1, steps=8, elapsed=10))

    def test_a_failure_summary_carries_the_reason(self):
        summary = format_summary("error", elapsed=12.0, error="CUDA out of memory\n  while allocating")
        self.assertIn("Failed", summary)
        self.assertIn("CUDA out of memory while allocating", summary)
        self.assertIn("Cancelled", format_summary("cancelled", images=0, steps=30, elapsed=4.0))

    def test_the_stride_spends_the_budget_across_the_whole_run(self):
        # One line is spent announcing the run, so the rest of the budget covers the steps.
        self.assertEqual(line_stride(30, 12), 3)
        self.assertEqual(line_stride(1000, 12), 91)
        self.assertEqual(line_stride(8, 12), 1, "a run shorter than the budget prints every step")
        self.assertEqual(line_stride(0, 12), 1)


class RunLengthTest(unittest.TestCase):
    def test_a_sampling_run_stays_inside_its_line_budget_at_any_length(self):
        for total in (1, 7, 15, 30, 60, 150, 1000, 4321):
            console, clock = sampling_console()
            lines = run_steps(console, Job(), clock, total, phase="Sampling", stage="base_sampling")
            self.assertLessEqual(len(lines), 12, f"{total} steps printed {len(lines)} lines")
            self.assertIn(f"{total}/{total}", lines[-1].text, "the finished run is always shown")

    def test_the_lines_are_spread_evenly_rather_than_bunched(self):
        console, clock = sampling_console()
        lines = run_steps(console, Job(), clock, 30, phase="Sampling", stage="base_sampling")
        steps = [int(line.text.split("|")[2].split("/")[0]) for line in lines]
        self.assertEqual(steps, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])

    def test_a_smaller_budget_thins_the_run_further(self):
        console, clock = sampling_console(budget=4)
        lines = run_steps(console, Job(), clock, 30, phase="Sampling", stage="base_sampling")
        self.assertEqual(len(lines), 4)

    def test_every_line_of_a_sampling_run_stands_on_its_own(self):
        console, clock = sampling_console()
        for line in run_steps(console, Job(), clock, 30, phase="Sampling", stage="base_sampling"):
            self.assertTrue(line.commit, "a sampling run is a history, so no line is drawn over")


class TiledTest(unittest.TestCase):
    def tiles(self, tile_count=4, steps=15, per_step=1.0):
        console, clock = sampling_console()
        job = Job()
        total = tile_count * steps
        console.observe("job-1", job, job.apply(
            phase="Hires.fix · Tile 1/4", stage="hires_sampling", stage_step=0, stage_total=total,
            stage_unit_index=1, stage_unit_total=tile_count, stage_unit_step=0, stage_unit_steps=steps,
        ))
        lines = []
        for index in range(tile_count):
            for step in range(1, steps + 1):
                clock.advance(per_step)
                line = console.observe("job-1", job, job.apply(
                    phase=f"Hires.fix · Tile {index + 1}/{tile_count}",
                    stage_step=index * steps + step,
                    stage_unit_index=index + 1, stage_unit_step=step, stage_unit_steps=steps,
                ))
                if line:
                    lines.append(line)
        return lines

    def test_four_tiles_commit_exactly_four_lines(self):
        # The tile is the unit a reader waits on, so the finished output has one line per tile —
        # everything in between is the same line being redrawn.
        committed = [line for line in self.tiles() if line.commit]
        self.assertEqual(len(committed), 4)
        for index, line in enumerate(committed, start=1):
            self.assertIn(f"Tile {index}/4", line.text)
            self.assertIn("15/15", line.text)
            self.assertIn("100%", line.text)

    def test_a_tile_redraws_over_itself_until_it_finishes(self):
        lines = self.tiles(tile_count=2, steps=5)
        self.assertEqual([line.commit for line in lines], [False] * 4 + [True] + [False] * 4 + [True])

    def test_each_tile_counts_its_own_steps_not_the_stages(self):
        # Tiles after the first are first seen at step 1, never 0. Deriving the tile-local count
        # from where the run started leaves every later tile one step short of its total, so its
        # committing line never arrives and the tiles run together on one line.
        lines = self.tiles(tile_count=3, steps=4)
        fourth_tile_steps = [line.text for line in lines if "Tile 3/3" in line.text]
        self.assertIn(" 1/4", fourth_tile_steps[0])
        self.assertIn(" 4/4", fourth_tile_steps[-1])
        self.assertTrue(lines[-1].commit)

    def test_each_tile_is_timed_on_its_own(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(
            phase="Hires.fix · Tile 1/2", stage="hires_sampling", stage_step=0, stage_total=20,
            stage_unit_index=1, stage_unit_total=2, stage_unit_step=0, stage_unit_steps=10,
        ))
        for step in range(1, 11):
            clock.advance(1.0)
            console.observe("job-1", job, job.apply(stage_step=step, stage_unit_step=step))
        # The second tile runs at four seconds a step; carrying the first tile's rate into it would
        # describe neither.
        last = None
        for step in range(1, 11):
            clock.advance(4.0)
            last = console.observe("job-1", job, job.apply(
                phase="Hires.fix · Tile 2/2", stage_step=10 + step,
                stage_unit_index=2, stage_unit_step=step, stage_unit_steps=10,
            )) or last
        self.assertIn("4.00s/it", last.text)

    def test_a_stage_counted_in_tiles_leaves_a_single_line(self):
        # Upscaling counts tiles directly, and there can be fifty-six of them a fraction of a
        # second apart. One line per tile buried the log, so the whole stage redraws one line and
        # commits it once, the same way a single tile does.
        for stage, unit in (("hires_upscale", "tile"), ("vae_decode", "tile")):
            console, clock = sampling_console()
            lines = run_steps(console, Job(), clock, 56, per_step=0.07, phase="Upscaling", stage=stage)
            committed = [line for line in lines if line.commit]
            self.assertEqual(len(committed), 1, f"{stage} left {len(committed)} lines behind")
            self.assertIn("56/56", committed[0].text, stage)
            self.assertIn(f"s/{unit}]", committed[0].text, stage)

    def test_a_fast_stage_redraws_no_faster_than_it_can_be_read(self):
        # Fifty-six tiles at 0.07s each would be fifty-six writes, and fifty-six console entries,
        # inside four seconds; the redraw floor turns that into a handful.
        console, clock = sampling_console()
        lines = run_steps(console, Job(), clock, 56, per_step=0.07, phase="Upscaling", stage="hires_upscale")
        self.assertLess(len(lines), 25, f"{len(lines)} redraws for a four-second stage")


class ObservationTest(unittest.TestCase):
    def test_a_run_announces_itself_before_it_has_a_rate_to_report(self):
        console, _ = sampling_console()
        job = Job()
        line = console.observe("job-1", job, job.apply(
            phase="Batch 1/1 · Sampling 1 images", stage="base_sampling", stage_step=0, stage_total=30,
        ))
        self.assertIn("0/30", line.text)
        self.assertIn("?s/it", line.text)
        self.assertIn("--:--", line.text)

    def test_the_rate_is_measured_across_the_run_not_taken_from_the_job(self):
        console, clock = sampling_console()
        job = Job()
        # The job clock has been running since the model started loading; the run's has not.
        job.update(elapsed_seconds=120.0)
        console.observe("job-1", job, job.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=11))
        clock.advance(9.0)
        line = console.observe("job-1", job, job.apply(stage_step=3, elapsed_seconds=129.0))
        self.assertIn("3/11", line.text)
        self.assertIn("3.00s/it", line.text)
        self.assertIn("[00:09<00:24", line.text)

    def test_the_same_step_reported_twice_prints_once(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=4))
        clock.advance(1.0)
        self.assertIsNotNone(console.observe("job-1", job, job.apply(stage_step=2)))
        clock.advance(0.1)
        self.assertIsNone(console.observe("job-1", job, job.apply(stage_step=2, progress=51)))

    def test_each_stage_times_itself(self):
        # Hires.fix runs slower than base sampling; carrying the base rate into it would describe
        # neither stage. A new stage restarts the clock and the step count.
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=20))
        clock.advance(20.0)
        console.observe("job-1", job, job.apply(stage_step=20))
        clock.advance(1.0)
        console.observe("job-1", job, job.apply(phase="Hires.fix · Sampling", stage="hires_sampling", stage_step=0, stage_total=10))
        clock.advance(30.0)
        line = console.observe("job-1", job, job.apply(stage_step=5))
        self.assertIn("5/10", line.text)
        self.assertIn("6.00s/it", line.text)

    def test_each_adetailer_region_is_its_own_run(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(
            phase="face_yolov8n · Inpainting 1/2", stage="adetailer_inpaint", stage_step=0, stage_total=12,
        ))
        clock.advance(12.0)
        console.observe("job-1", job, job.apply(stage_step=12))
        clock.advance(2.0)
        # The second region reuses the stage but not the phase, and its counter starts over.
        console.observe("job-1", job, job.apply(phase="face_yolov8n · Inpainting 2/2", stage_step=0))
        clock.advance(2.0)
        line = console.observe("job-1", job, job.apply(stage_step=2))
        self.assertIn("Inpainting 2/2", line.text)
        self.assertIn("2/12", line.text)
        self.assertIn("1.00s/it", line.text)

    def test_a_stage_that_omits_the_phase_keeps_the_one_it_opened_with(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(phase="VAE tiled decode", stage="vae_decode", stage_step=0, stage_total=4))
        clock.advance(3.0)
        line = console.observe("job-1", job, job.apply(stage_step=1, progress=95))
        self.assertIn("VAE tiled decode", line.text)

    def test_the_job_counter_rides_along_when_the_call_site_sets_it(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(
            phase="face_yolov8n · Inpainting 1/2", stage="adetailer_inpaint", stage_step=0, stage_total=11,
        ))
        clock.advance(6.0)
        line = console.observe("job-1", job, job.apply(stage_step=2, step=32, total_steps=54, elapsed_seconds=135.0))
        self.assertTrue(line.text.endswith("job 32/54, 02:15"), line.text)

    def test_a_stale_job_counter_is_not_quoted(self):
        # Hires.fix tiles never touch the job-wide step count, which still holds the base sampling
        # total. Reprinting it beside the tiles would show a finished run next to a running one.
        console, clock = sampling_console()
        job = Job(step=30, total_steps=30)
        console.observe("job-1", job, job.apply(
            phase="Hires.fix · Upscaling", stage="hires_upscale", stage_step=0, stage_total=4,
        ))
        clock.advance(4.0)
        line = console.observe("job-1", job, job.apply(stage_step=1))
        self.assertNotIn("30/30", line.text)
        self.assertNotIn("job", line.text)

    def test_a_stale_job_clock_is_not_quoted_either(self):
        console, clock = sampling_console()
        job = Job(elapsed_seconds=96.0)
        console.observe("job-1", job, job.apply(phase="VAE tiled decode", stage="vae_decode", stage_step=0, stage_total=4))
        clock.advance(3.0)
        line = console.observe("job-1", job, job.apply(stage_step=1))
        self.assertNotIn("01:36", line.text)

    def test_paused_seconds_are_taken_off_the_rate(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(
            phase="Sampling", stage="base_sampling", stage_step=0, stage_total=11, paused_seconds=0.0,
        ))
        clock.advance(2.0)
        console.observe("job-1", job, job.apply(stage_step=1))
        clock.advance(38.0)  # thirty-six of those seconds were spent paused
        line = console.observe("job-1", job, job.apply(stage_step=2, paused_seconds=36.0))
        self.assertIn("2.00s/it", line.text)

    def test_a_pause_banked_before_this_run_does_not_credit_it(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(
            phase="Hires.fix · Sampling", stage="hires_sampling", stage_step=0, stage_total=11, paused_seconds=60.0,
        ))
        clock.advance(4.0)
        line = console.observe("job-1", job, job.apply(stage_step=2, paused_seconds=60.0))
        self.assertIn("2.00s/it", line.text)

    def test_an_update_that_moves_no_counter_prints_nothing(self):
        console, _ = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=10))
        self.assertIsNone(console.observe("job-1", job, job.apply(progress=42)))
        self.assertIsNone(console.observe("job-1", job, job.apply(status="paused", phase="Paused")))
        self.assertIsNone(console.observe("job-1", job, job.apply(preview_url="/api/preview/1")))

    def test_a_stage_without_a_total_is_not_charted(self):
        # Model offload reports stage_step=0, stage_total=0: it has phases, not steps.
        console, _ = sampling_console()
        job = Job()
        self.assertIsNone(console.observe("job-1", job, job.apply(
            phase="Moving sampler to system memory", stage="sampler_offload", stage_step=0, stage_total=0,
        )))

    def test_a_terminal_status_closes_the_job_with_a_summary(self):
        console, clock = sampling_console()
        job = Job()
        console.observe("job-1", job, job.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=30))
        clock.advance(86.4)
        console.observe("job-1", job, job.apply(stage_step=30, step=30, total_steps=30))
        line = console.observe("job-1", job, job.apply(
            status="complete", phase="Complete", stage="complete", completed_images=4,
            elapsed_seconds=86.4, output_name="ComfyUI_0007.png",
        ))
        self.assertIn("4 images", line.text)
        self.assertIn("ComfyUI_0007.png", line.text)
        self.assertTrue(line.commit)

    def test_a_new_job_does_not_inherit_the_previous_run(self):
        console, clock = sampling_console()
        first = Job()
        console.observe("job-1", first, first.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=11))
        clock.advance(50.0)
        console.observe("job-1", first, first.apply(stage_step=5))
        second = Job()
        clock.advance(1.0)
        console.observe("job-2", second, second.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=11))
        clock.advance(2.0)
        line = console.observe("job-2", second, second.apply(stage_step=2))
        self.assertIn("1.00s/it", line.text)

    def test_disabling_the_console_silences_it_without_touching_the_job(self):
        console, clock = sampling_console(enabled=False)
        job = Job()
        self.assertIsNone(console.observe("job-1", job, job.apply(
            phase="Sampling", stage="base_sampling", stage_step=0, stage_total=10,
        )))
        clock.advance(1.0)
        self.assertIsNone(console.observe("job-1", job, job.apply(stage_step=1)))
        self.assertEqual(job["stage_step"], 1)


class ModelLoadTest(unittest.TestCase):
    def console(self, memory=None):
        clock = FakeClock()
        return ProgressConsole(clock=clock, memory=memory), clock

    def test_a_loaded_model_reports_the_wait_and_the_memory_it_took(self):
        readings = {"vram_used_mb": 7384, "vram_total_mb": 8188, "ram_used_mb": 11576, "ram_total_mb": 32662}
        console, clock = self.console(memory=lambda: readings)
        job = Job()
        console.observe("job-1", job, job.apply(phase="Preparing model", stage="model_load", status="running"))
        clock.advance(12.4)
        line = console.observe("job-1", job, job.apply(
            phase="Model ready · NORMAL_VRAM", model_reused=False, model_weight_bytes=6_940_000_000,
        ))
        self.assertIn("loaded in 12.4s", line.text)
        self.assertIn("weights 6.46 GB", line.text)
        self.assertIn("VRAM 7.21/8.00 GB", line.text)

    def test_the_wait_is_measured_from_when_the_load_stage_opened(self):
        console, clock = self.console()
        job = Job()
        clock.advance(500.0)  # the process has been up a while; the load has not started
        console.observe("job-1", job, job.apply(phase="Preparing model", stage="model_load", status="running"))
        clock.advance(3.0)
        line = console.observe("job-1", job, job.apply(phase="Model ready", model_reused=False))
        self.assertIn("loaded in 3.0s", line.text)

    def test_a_reload_is_timed_again_rather_than_from_the_first_load(self):
        console, clock = self.console()
        job = Job()
        console.observe("job-1", job, job.apply(phase="Preparing model", stage="model_load", status="running"))
        clock.advance(4.0)
        console.observe("job-1", job, job.apply(phase="Model ready", model_reused=False))
        clock.advance(300.0)
        console.observe("job-1", job, job.apply(phase="Sampling", stage="base_sampling", stage_step=0, stage_total=10))
        console.observe("job-1", job, job.apply(phase="Preparing model", stage="model_load"))
        clock.advance(2.0)
        line = console.observe("job-1", job, job.apply(phase="Model ready", model_reused=False))
        self.assertIn("loaded in 2.0s", line.text)

    def test_adapters_are_announced_straight_after_the_model_they_sit_on(self):
        console, clock = self.console()
        job = Job()
        console.observe("job-1", job, job.apply(phase="Preparing model", stage="model_load", status="running"))
        clock.advance(6.3)
        model = console.observe("job-1", job, job.apply(phase="Anima ready", model_reused=False))
        loras = console.observe("job-1", job, job.apply(mounted_loras=[
            {"name": "ashima_v4.safetensors", "weight": 0.25, "bytes": 220_000_000},
        ]))
        self.assertIn("loaded in 6.3s", model.text)
        self.assertIn("LoRA · 1 mounted", loras.text)
        self.assertTrue(loras.commit)

    def test_a_job_with_no_adapters_says_nothing_about_them(self):
        console, _ = self.console()
        job = Job()
        self.assertIsNone(console.observe("job-1", job, job.apply(mounted_loras=[])))

    def test_a_memory_reading_that_fails_costs_the_reading_not_the_line(self):
        def broken():
            raise RuntimeError("no CUDA context")

        console, clock = self.console(memory=broken)
        job = Job()
        console.observe("job-1", job, job.apply(phase="Preparing model", stage="model_load", status="running"))
        clock.advance(1.0)
        line = console.observe("job-1", job, job.apply(phase="Model ready", model_reused=True))
        self.assertIn("reused in 1.0s", line.text)


class DrawerParityTest(unittest.TestCase):
    """The in-app console parses these lines back; ``src/console-progress.js`` holds the reader.

    Nothing enforces the contract between the two languages at run time — a line the reader cannot
    match simply renders as raw text — so the exact shape is pinned on both sides. The literals
    below appear verbatim in ``scripts/console-progress.test.mjs``.
    """

    def test_a_sampling_line(self):
        self.assertEqual(
            format_progress("Batch 1/2 · Sampling 2 images", 20, 30, elapsed=56.2, rate=2.81, job_elapsed=68.2),
            "Batch 1/2 · Sampling 2 images          66%|█████████████       |     20/30 [00:56<00:28, 2.81s/it]  job 01:08",
        )

    def test_an_opening_line(self):
        self.assertEqual(
            format_progress("Batch 1/2 · Sampling 2 images", 0, 30, elapsed=0.0),
            "Batch 1/2 · Sampling 2 images           0%|                    |      0/30 [00:00<--:--, ?s/it]",
        )

    def test_an_adetailer_line_carrying_the_job_counter(self):
        self.assertEqual(
            format_progress(
                "face_yolov8n · Inpainting 1/2", 3, 12,
                elapsed=7.2, rate=2.40, overall=(33, 54), job_elapsed=159.6,
            ),
            "face_yolov8n · Inpainting 1/2          25%|█████               |      3/12 [00:07<00:21, 2.40s/it]  job 33/54, 02:39",
        )

    def test_a_tiled_line(self):
        self.assertEqual(
            format_progress("Hires.fix · Upscaling 2176 x 2816", 2, 4, elapsed=25.4, rate=12.70, unit="tile"),
            "Hires.fix · Upscaling 2176 x 2816      50%|██████████          |       2/4 [00:25<00:25, 12.70s/tile]",
        )

    def test_a_line_from_a_run_measured_in_hours(self):
        self.assertEqual(
            format_progress("Sampling", 900, 1800, elapsed=3725, rate=4.14),
            "Sampling                               50%|██████████          |  900/1800 [1:02:05<1:02:05, 4.14s/it]",
        )


class WriteTest(unittest.TestCase):
    def test_every_line_opens_at_column_zero_and_is_flushed(self):
        # The leading return is what the dev server keys on to forward a chunk without its
        # "[inference] " prefix. Emitting it only on redraws would indent every other progress
        # line by twelve characters and break the single column the bars are meant to share.
        stream = io.StringIO()
        flushes = []
        stream.flush = lambda: flushes.append(True)
        ProgressConsole(stream=stream).write(Line("Sampling  50%"))
        self.assertEqual(stream.getvalue(), "\rSampling  50%\n")
        self.assertEqual(len(flushes), 1)

    def test_an_uncommitted_line_is_redrawn_over_rather_than_stacked(self):
        stream = io.StringIO()
        console = ProgressConsole(stream=stream)
        console.write(Line("Tile 1/4  20%", commit=False))
        console.write(Line("Tile 1/4  40%", commit=False))
        console.write(Line("Tile 1/4 100%", commit=True))
        self.assertEqual(stream.getvalue(), "\rTile 1/4  20%\rTile 1/4  40%\rTile 1/4 100%\n")

    def test_a_shorter_replacement_wipes_what_the_longer_line_left_behind(self):
        # Without the padding the tail of the previous line stays on screen and the numbers read
        # as a splice of two different updates.
        stream = io.StringIO()
        console = ProgressConsole(stream=stream)
        console.write(Line("Tile 1/4 ......... 100%", commit=False))
        console.write(Line("Tile 2/4 0%", commit=True))
        self.assertEqual(stream.getvalue(), "\rTile 1/4 ......... 100%\rTile 2/4 0%            \n")

    def test_the_next_committed_line_starts_clean_after_one_was_redrawn(self):
        stream = io.StringIO()
        console = ProgressConsole(stream=stream)
        console.write(Line("Tile 1/4  20%", commit=False))
        console.write(Line("Tile 1/4 100%", commit=True))
        console.write(Line("Complete", commit=True))
        self.assertTrue(stream.getvalue().endswith("Tile 1/4 100%\n\rComplete\n"), stream.getvalue())
        self.assertNotIn("           \n", stream.getvalue(), "a committed line leaves no padding to wipe")

    def test_a_character_the_code_page_cannot_encode_does_not_stop_the_run(self):
        class NarrowStream:
            encoding = "ascii"

            def __init__(self):
                self.written = []

            def write(self, text):
                text.encode(self.encoding)  # raises the way a cp1252 console would
                self.written.append(text)

            def flush(self):
                pass

        stream = NarrowStream()
        ProgressConsole(stream=stream).write(Line("Batch 1/2 · Sampling"))
        self.assertEqual(stream.written, ["\rBatch 1/2 ? Sampling\n"])

    def test_a_closed_console_does_not_take_the_generation_with_it(self):
        stream = io.StringIO()
        stream.close()
        ProgressConsole(stream=stream).write(Line("Sampling"))  # must not raise


if __name__ == "__main__":
    unittest.main()
