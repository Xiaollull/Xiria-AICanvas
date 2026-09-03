"""Sampling progress for the two consoles.

The web UI has always had the step counter — it polls the job record and draws a bar. The consoles
had nothing: ``set_progress_bar_config(disable=True)`` turns off the diffusers tqdm bar on every
pipeline, so a running generation printed no output at all between "job accepted" and the saved
file. Watching a long batch from the terminal meant watching a blank window.

Both consoles read the same bytes. The dev-server plugin pipes the backend's stdout to its own
stdout (the window ``Start-XirAI`` opens) and, in the same handler, appends it to the buffer that
``/api/console`` serves to the in-app drawer. So progress printed here reaches both, and reaches
them saying exactly the same thing.

How much it prints is the whole design. A step is 0.05s on a small SDXL run and 3s on a large one,
so a line per step is either a useless trickle or a wall of text that buries everything else in the
log. Each kind of work is therefore given a shape that fits it:

* **A sampling run gets a fixed budget of lines** — twelve — spread evenly across however many
  steps it has. Thirty steps print every third; a thousand print every ninety-first. The reader
  sees the same amount of output either way, and it is a *history*: the lines stay, so the rate
  early in a run can be compared with the rate late in it.
* **Tiled work gets one line per tile, redrawn in place.** Four tiles produce exactly four lines,
  each rewriting itself as its steps complete and committed with a newline when the tile finishes.
  A tile is the unit a reader waits on, so a tile is the unit that gets a line.
* **Loading a model gets one line, once it is loaded**, carrying what it cost: how long it took,
  how large the weights are, and what the device and system memory look like now that they are
  resident.

The bar is drawn with full blocks against spaces rather than ``#`` against ``-``: at a glance the
filled run reads as one solid bar instead of a row of separate marks. ``U+2588`` encodes in UTF-8,
GBK and CP437, which covers the consoles this ships to; :meth:`ProgressConsole.write` degrades
rather than raising if it ever meets one that cannot.
"""

import math
import sys
import time

# Stages whose steps are image tiles, not sampler iterations. Labelling a 50-second tile `s/it`
# would invite the reader to compare it with a sampling step — and one tile deserves one line,
# so these are never thinned out the way a sampling run is.
STAGE_UNITS = {"vae_decode": "tile", "hires_upscale": "tile"}
DEFAULT_UNIT = "it"

TERMINAL_STATUSES = {"complete": "Complete", "cancelled": "Cancelled", "error": "Failed"}

# The most lines one sampling run may print, however many steps it has.
MAX_RUN_LINES = 12
# An in-place line is cheap to print but not free to poll, so a tile redraws at most this often.
REDRAW_MIN_SECONDS = 0.2

BAR_WIDTH = 20
BAR_FILLED = "█"
BAR_EMPTY = " "
PHASE_WIDTH = 36
# Wide enough for a four-digit total, so the bracket after it holds one column across every stage.
COUNTER_WIDTH = 9
UNKNOWN_CLOCK = "--:--"


class Line:
    """One console line, and whether it is finished with.

    ``commit`` false means the line will be redrawn over: it is written with a carriage return and
    no newline, so the next redraw replaces it rather than stacking under it.
    """

    __slots__ = ("text", "commit")

    def __init__(self, text, commit=True):
        self.text = text
        self.commit = commit

    def __eq__(self, other):
        return isinstance(other, Line) and (self.text, self.commit) == (other.text, other.commit)

    def __repr__(self):
        return f"Line({self.text!r}, commit={self.commit})"


def _count(value):
    """An integer step count, or ``None`` for anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return int(value)


def _seconds(value, fallback=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value) if math.isfinite(value) else fallback


def format_clock(seconds):
    """``MM:SS`` while under an hour, ``H:MM:SS`` past it — the tqdm reading everyone knows."""
    seconds = _seconds(seconds)
    if seconds is None or seconds < 0:
        return UNKNOWN_CLOCK
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}" if hours else f"{minutes:02d}:{whole_seconds:02d}"


def format_rate(seconds_per_unit, unit=DEFAULT_UNIT):
    """Always seconds-per-unit, never its reciprocal.

    tqdm flips to ``it/s`` once a step takes under a second, which means a log can report the same
    stage in two different units depending on the hardware it ran on. One unit throughout stays
    comparable across machines.
    """
    seconds_per_unit = _seconds(seconds_per_unit)
    if seconds_per_unit is None or seconds_per_unit <= 0:
        return f"?s/{unit}"
    return f"{seconds_per_unit:.2f}s/{unit}"


def format_size(byte_count):
    byte_count = _seconds(byte_count)
    if byte_count is None or byte_count < 0:
        return "?"
    if byte_count >= 1024**3:
        return f"{byte_count / 1024**3:.2f} GB"
    return f"{byte_count / 1024**2:.0f} MB"


def format_bar(fraction, width=BAR_WIDTH):
    fraction = min(1.0, max(0.0, _seconds(fraction, 0.0)))
    # Truncated rather than rounded: a full bar means finished, so 99.9% must still show a gap.
    filled = width if fraction >= 1 else int(fraction * width)
    return BAR_FILLED * filled + BAR_EMPTY * (width - filled)


def format_phase(phase, width=PHASE_WIDTH):
    """One fixed-width column, so the bars line up when the lines are read as a block."""
    text = " ".join(str(phase or "Working").split())
    if len(text) > width:
        text = text[: width - 3] + "..."
    return text.ljust(width)


def format_progress(phase, step, total, *, elapsed=None, rate=None, unit=DEFAULT_UNIT, overall=None, job_elapsed=None):
    fraction = step / total if total > 0 else 0.0
    percent = 100 if step >= total else int(fraction * 100)
    # A finished run has nothing left to wait for, whether or not its rate was ever measured.
    remaining = 0.0 if step >= total else (total - step) * rate if rate is not None else None
    # Every field before the brackets is fixed-width, so the bar occupies the same columns on every
    # line whatever stage wrote it and however many steps that stage has. The counter is padded as
    # a group rather than per number: a 30-step run and a 56-tile one otherwise put the bracket in
    # different places, and the eye reads that as the whole line shifting.
    counter = f"{step:>{len(str(total))}}/{total}".rjust(COUNTER_WIDTH)
    line = (
        f"{format_phase(phase)}  {percent:>3}%|{format_bar(fraction)}| {counter} "
        f"[{format_clock(elapsed)}<{format_clock(remaining)}, {format_rate(rate, unit)}]"
    )
    tail = []
    if overall:
        tail.append(f"{overall[0]}/{overall[1]}")
    if job_elapsed is not None:
        tail.append(format_clock(job_elapsed))
    return f"{line}  job {', '.join(tail)}" if tail else line


def format_model(label, *, reused=False, seconds=None, weight_bytes=None, memory=None):
    """What a model cost to have ready: the wait, its size, and the memory it now occupies."""
    memory = memory or {}
    parts = ["reused" if reused else "loaded"]
    if seconds is not None:
        parts[0] += f" in {seconds:.1f}s"
    if weight_bytes:
        parts.append(f"weights {format_size(weight_bytes)}")
    for name, used, total in (
        ("VRAM", memory.get("vram_used_mb"), memory.get("vram_total_mb")),
        ("RAM", memory.get("ram_used_mb"), memory.get("ram_total_mb")),
    ):
        if used is None:
            continue
        gigabytes = f"{used / 1024:.2f}"
        parts.append(f"{name} {gigabytes}/{total / 1024:.2f} GB" if total else f"{name} {gigabytes} GB")
    return f"{format_phase(label)}  {' · '.join(parts)}"


def format_loras(mounted):
    """What was mounted on top of the model, one adapter to a line.

    Returned as a single block rather than a line each: the adapters are one fact about the run,
    and the console groups them under a heading with the rest indented into the phase column, so a
    stack of nine reads as one entry instead of nine unrelated ones.
    """
    if not mounted:
        return None
    heading = format_phase(f"LoRA · {len(mounted)} mounted")
    rows = []
    for adapter in mounted:
        weight = _seconds(adapter.get("weight"))
        parts = [f"{weight:>5.2f}" if weight is not None else "    ?", str(adapter.get("name") or "?")]
        if adapter.get("bytes"):
            parts.append(format_size(adapter["bytes"]))
        rows.append(" · ".join(parts))
    indent = " " * (PHASE_WIDTH + 2)
    return f"{heading}  {rows[0]}" + "".join(f"\n{indent}{row}" for row in rows[1:])


def format_summary(status, *, images=None, steps=None, elapsed=None, error=None, output=None):
    parts = []
    if images:
        parts.append(f"{images} image{'' if images == 1 else 's'}")
    if steps:
        parts.append(f"{steps} steps")
    if elapsed is not None:
        parts.append(f"{format_clock(elapsed)} elapsed")
        if steps:
            parts.append(f"{format_rate(elapsed / steps)} average")
    if output:
        parts.append(str(output))
    if error:
        parts.append(" ".join(str(error).split())[:160])
    label = TERMINAL_STATUSES.get(status, status)
    return f"{format_phase(label)}  {' · '.join(parts)}" if parts else format_phase(label).rstrip()


def line_stride(total, budget=MAX_RUN_LINES):
    """How many steps to advance between printed lines to stay inside the budget.

    The opening line costs one of the budgeted lines, so the remaining steps are divided among the
    rest. A run shorter than the budget prints every step and the stride is 1.
    """
    if total <= 0 or budget <= 1:
        return 1
    return max(1, math.ceil(total / (budget - 1)))


class ProgressConsole:
    """Turns the stream of job updates into console lines.

    ``observe`` keeps the timing state and returns the :class:`Line` to print, or ``None``. Writing
    is a separate call so the caller can do it outside the jobs lock.
    """

    def __init__(self, *, stream=None, budget=MAX_RUN_LINES, clock=time.monotonic, enabled=True, memory=None):
        self._stream = stream
        self._clock = clock
        self._memory = memory
        self.budget = max(2, int(budget))
        self.enabled = enabled
        self._job_id = None
        self._run = None
        self._model_started = None
        self._open_width = None

    # ---- observation -------------------------------------------------------------------

    def observe(self, job_id, job, updates):
        if not self.enabled:
            return None
        if job_id != self._job_id:
            self._job_id, self._run, self._model_started = job_id, None, None

        status = updates.get("status")
        if status in TERMINAL_STATUSES:
            self._run = None
            return Line(format_summary(
                status,
                images=_count(job.get("completed_images")),
                steps=_count(job.get("total_steps")),
                elapsed=_seconds(job.get("elapsed_seconds")),
                error=job.get("error") if status == "error" else None,
                output=job.get("output_name") if status == "complete" else None,
            ))

        model_line = self._observe_model(job, updates)
        if model_line is not None:
            return model_line

        # Reported where they are mounted, which is after the model is resident — so the adapters
        # read as something added to the model just announced, not as a stage of their own.
        if "mounted_loras" in updates:
            block = format_loras(updates.get("mounted_loras"))
            return None if block is None else Line(block)

        # Only an update that moved a counter can advance a bar. Everything else — a phase rename,
        # a pause, a preview URL — passes through untouched.
        if "stage_step" not in updates and "step" not in updates:
            return None
        step, total = _count(job.get("stage_step")), _count(job.get("stage_total"))
        if step is None or total is None or total <= 0 or step < 0:
            return None
        return self._observe_step(job, updates, min(step, total), total)

    def _observe_model(self, job, updates):
        if job.get("stage") == "model_load" and self._model_started is None:
            self._model_started = self._clock()
        if "model_reused" not in updates:
            return None
        started, self._model_started = self._model_started, None
        memory = {}
        if self._memory is not None:
            try:
                memory = self._memory() or {}
            except Exception:
                # A memory reading is a nicety; failing to take one must not cost the line.
                memory = {}
        return Line(format_model(
            job.get("phase"),
            reused=bool(updates.get("model_reused")),
            seconds=None if started is None else max(0.0, self._clock() - started),
            weight_bytes=_seconds(job.get("model_weight_bytes")),
            memory=memory,
        ))

    def _observe_step(self, job, updates, step, total):
        now = self._clock()
        paused = _seconds(job.get("paused_seconds"), 0.0)
        stage = job.get("stage") or ""
        unit = _count(job.get("stage_unit_index"))
        unit_step = _count(job.get("stage_unit_step"))
        unit_steps = _count(job.get("stage_unit_steps"))

        # A tile counts its own steps, and it is the tile's own count that has to drive the bar.
        # Deriving it from where the run happened to start goes wrong the moment a tile is first
        # seen at step 1 rather than 0 — every tile after the first — and the run then finishes a
        # step short of its total, so the line that commits it is never reached.
        per_tile = unit is not None and unit_steps is not None and unit_steps > 0 and unit_step is not None
        local_step, local_total = (min(unit_step, unit_steps), unit_steps) if per_tile else (step, total)

        # A run is identified by its counter, not by its label: the tiled Hires.fix path rewrote
        # its phase on every step, and keying on that would start a new run each time. Where the
        # stage reports tiles, the tile index joins the key so each tile is timed on its own.
        key = (stage, total, unit)
        run = self._run
        if run is None or run["key"] != key or local_step < run["local"]:
            run = self._run = {
                "key": key, "origin": local_step, "local": local_step, "started": now, "paused": paused,
                "reported": None, "drawn": None,
                # Anything counted in tiles redraws one line rather than printing per tile: a
                # 4-tile refinement and a 56-tile upscale both leave a single line behind. Only a
                # sampling run spends the line budget, which is what bounds it to twelve.
                "inplace": per_tile or stage in STAGE_UNITS,
                "stride": line_stride(local_total, self.budget),
            }
        run["local"] = max(run["local"], local_step)
        if local_step == run["reported"]:
            return None

        complete = local_step >= local_total
        if not self._due(run, local_step, complete, run["inplace"], now):
            return None
        run["reported"], run["drawn"] = local_step, now

        # A pause stops the clock rather than inflating the rate: only the paused seconds banked
        # since this run began are subtracted, so an earlier stage's pause does not credit this one.
        elapsed = max(0.0, (now - run["started"]) - max(0.0, paused - run["paused"]))
        # Divided by the steps this run actually timed. A stage first seen part-way through — a
        # resumed sampler, a stage whose opening update was missed — has not measured the steps
        # that came before it, and charging its elapsed time against them would halve the rate.
        measured = local_step - run["origin"]
        rate = elapsed / measured if measured > 0 and elapsed > 0 else None

        # The job-wide figures are quoted only when this update set them. Left to the merged record
        # they would be whatever an earlier stage wrote — a finished step count beside a running
        # bar, or a job clock frozen where the last stage that bothered to set it ended.
        job_step, job_total = _count(job.get("step")), _count(job.get("total_steps"))
        overall = (
            (job_step, job_total)
            if "step" in updates and "total_steps" in updates and job_step is not None and job_total is not None
            and (job_step, job_total) != (local_step, local_total)
            else None
        )
        text = format_progress(
            job.get("phase"), local_step, local_total,
            elapsed=elapsed, rate=rate, unit=STAGE_UNITS.get(stage, DEFAULT_UNIT),
            overall=overall,
            job_elapsed=_seconds(job.get("elapsed_seconds")) if "elapsed_seconds" in updates else None,
        )
        # The last step commits the line that has been redrawing itself, so whatever comes next
        # starts under it rather than over it.
        return Line(text, commit=complete or not run["inplace"])

    def _due(self, run, step, complete, inplace, now):
        if complete:
            return True
        if inplace:
            # Redrawing costs a line only once, but it still costs a write and a console entry per
            # update, and a 56-tile upscale can turn those over faster than anyone can read them.
            return run["drawn"] is None or now - run["drawn"] >= REDRAW_MIN_SECONDS
        # The opening line is worth one of the budgeted lines: it announces the stage before the
        # first stride has elapsed, which on a long run is otherwise a silent minute.
        return step == 0 or step % run["stride"] == 0

    # ---- output ------------------------------------------------------------------------

    def write(self, line):
        """Print a line, and never let the console be the reason a generation fails.

        This runs inside the sampler callback. A console that has been closed, or that cannot
        encode a character in the active code page, is a cosmetic problem; raising here would
        abort the run that the line was describing.
        """
        stream = self._stream if self._stream is not None else sys.stdout
        text, commit = (line.text, line.commit) if isinstance(line, Line) else (str(line), True)
        # Every line opens with a carriage return, not only the ones being redrawn over. It costs
        # nothing at the start of a fresh line, and it is what the dev server keys on to forward a
        # chunk unprefixed — so progress lines all begin at column zero and their bars share one
        # column, instead of the redrawn ones sitting twelve characters left of the rest.
        padding = max(0, (self._open_width or 0) - len(text))
        rendered = f"\r{text}{' ' * padding}"
        self._open_width = None if commit else len(text)
        if commit:
            rendered += "\n"
        try:
            try:
                stream.write(rendered)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "ascii"
                stream.write(rendered.encode(encoding, "replace").decode(encoding, "replace"))
            stream.flush()
        except (OSError, ValueError, AttributeError):
            pass
