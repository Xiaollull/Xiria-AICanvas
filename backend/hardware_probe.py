"""Telemetry for the hardware monitor panel.

Every number the panel drew used to come from whichever source answered first, and two of them
ended up meaning different things on different machines:

* "VRAM used" was ``torch.cuda.memory_allocated()`` — live PyTorch tensors in this process only.
  That excludes the CUDA context, the cuBLAS/cuDNN workspaces, and every block the caching
  allocator is holding but not currently handing out, so it under-reports by a gigabyte or more.
  Then, on any machine with ``nvidia-smi`` on PATH, the same field was silently overwritten with
  the device-wide figure, which counts every other process on the card. One field, one bar, two
  definitions separated by gigabytes.
* "RAM used" was ``psutil.virtual_memory().used`` while the percentage printed beside it came
  from ``.percent``, which is derived from ``.available``. Those are the same quantity on Windows
  and different quantities on Linux, where ``used`` excludes the page cache that ``available``
  counts as reclaimable — so the figure and the bar could disagree by several gigabytes on one
  screen.

This module answers each question once, from the source that actually knows:

* device VRAM from ``cudaMemGetInfo`` via ``torch.cuda.mem_get_info()`` — driver truth, all
  processes, and no subprocess to spawn;
* this process's share from the allocator's own reserved/allocated counters, reported as its own
  fields instead of being folded into the device number;
* system RAM as ``total - available``, with the percentage derived from that same subtraction so
  the two can never contradict each other;
* ``nvidia-smi`` only for what CUDA does not expose at all — temperature, utilisation, power and
  fan — sampled on a background cadence so that a slow probe costs freshness (which the payload
  reports) instead of stretching the reply.
"""

import os
import shutil
import subprocess
import threading
import time

MIB = 1024**2
GIB = 1024**3

SMI_QUERY = "index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,fan.speed"

# `nvidia-smi` prints these instead of a number for a sensor the board does not have — a fanless
# datacentre card, or power draw on a laptop GPU. They are absences, not zeroes.
SMI_BLANKS = {
    "",
    "n/a",
    "[n/a]",
    "not supported",
    "[not supported]",
    "unknown error",
    "[unknown error]",
    "insufficient permissions",
    "[insufficient permissions]",
}


def _numeric(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in SMI_BLANKS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalise_gpu_uuid(value):
    """Reduce a GPU UUID to a comparable form.

    `nvidia-smi` prints ``GPU-0f3a…`` while torch hands back a `uuid.UUID`, and MIG instances add
    their own prefix. Comparing the bare hex avoids matching on formatting.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    for prefix in ("gpu-", "mig-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.replace("-", "") or None


def parse_nvidia_smi(stdout, device_uuid=None, device_index=None):
    """Pick the row describing the GPU this process is actually running on.

    Matching on the UUID first is what makes this correct on a multi-GPU machine: both
    ``CUDA_VISIBLE_DEVICES`` and ``CUDA_DEVICE_ORDER=FASTEST_FIRST`` renumber devices, so torch's
    device 0 need not be `nvidia-smi`'s row 0. A single-GPU listing is unambiguous whatever the
    numbering; anything else with no match is reported as unknown rather than charting the wrong
    card's temperature next to this card's memory.
    """
    records = [
        [part.strip() for part in line.split(",")]
        for line in (stdout or "").splitlines()
        if line.strip()
    ]
    records = [record for record in records if len(record) >= 9]
    if not records:
        return None

    wanted_uuid = normalise_gpu_uuid(device_uuid)
    selected = None
    if wanted_uuid:
        selected = next((record for record in records if normalise_gpu_uuid(record[1]) == wanted_uuid), None)
    if selected is None and device_index is not None:
        wanted_index = str(device_index).strip()
        selected = next((record for record in records if record[0] == wanted_index), None)
    if selected is None and len(records) == 1:
        selected = records[0]
    if selected is None:
        return None

    reading = {
        "gpu_name": selected[2] or None,
        "gpu_index": _numeric(selected[0]),
        "gpu_uuid": normalise_gpu_uuid(selected[1]),
        "gpu_temp": _numeric(selected[3]),
        "gpu_util": _numeric(selected[4]),
        "power_w": _numeric(selected[7]),
        "fan_speed": _numeric(selected[8]),
        # Kept apart from the CUDA figures on purpose: this is only used when torch cannot report
        # the device itself, and the two are never averaged or silently swapped.
        "smi_vram_used_mb": _numeric(selected[5]),
        "smi_vram_total_mb": _numeric(selected[6]),
    }
    if reading["gpu_index"] is not None:
        reading["gpu_index"] = int(reading["gpu_index"])
    return reading


def _run_nvidia_smi(executable, timeout):
    return subprocess.run(
        [executable, f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        # A packaged launcher can run without a console. Without this flag every probe would flash
        # a window on Windows, twice a second, for as long as the panel stays open.
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
    )


def _locate_nvidia_smi():
    return os.environ.get("NVIDIA_SMI_PATH") or shutil.which("nvidia-smi")


def _spawn_daemon(target):
    threading.Thread(target=target, name="xirai-gpu-sampler", daemon=True).start()


class NvidiaSmiSampler:
    """Keeps `nvidia-smi` off the request path.

    The probe used to be spawned inline on every poll. A process spawn costs a few hundred
    milliseconds on Windows and considerably more while the GPU is saturated — which is precisely
    when the panel is being watched. The reply arrived late, so the two-second cadence the chart
    assumes was not the cadence the samples were taken at, and the x-axis quietly compressed under
    load. Sampling on a background thread separates the two: a slow probe now costs freshness,
    and the age returned alongside the reading says how much.

    The thread is started by the first read, backs off geometrically while the probe keeps
    failing, and retires itself once nothing has asked for a reading in `idle_timeout` seconds, so
    an unwatched server spawns nothing at all.
    """

    def __init__(
        self,
        interval=2.0,
        idle_timeout=30.0,
        timeout=6.0,
        runner=_run_nvidia_smi,
        locate=_locate_nvidia_smi,
        spawn=_spawn_daemon,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self._interval = interval
        self._idle_timeout = idle_timeout
        self._timeout = timeout
        self._runner = runner
        self._locate = locate
        self._spawn = spawn
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._running = False
        self._executable = None
        self._located = False
        self._reading = None
        self._reading_at = None
        self._failures = 0
        self._last_request = 0.0
        self._target = (None, None)

    @property
    def available(self):
        with self._lock:
            return bool(self._executable) if self._located else None

    def read(self, device_uuid=None, device_index=None):
        """Return ``(reading, age_seconds)``; waits on a probe only when no worker is refreshing."""
        with self._lock:
            self._target = (device_uuid, device_index)
            self._last_request = self._clock()
            if not self._located:
                self._located = True
                self._executable = self._locate()
            if not self._executable:
                return None, None
            # Nothing cached, or the worker has retired since the panel was last open and what is
            # cached predates the interval. Waiting out one probe beats opening the panel onto a
            # temperature from the last session.
            stale = self._reading_at is None or (self._clock() - self._reading_at) > self._interval
            probe_now = stale and not self._running
        if probe_now:
            self._probe()
        self._ensure_worker()
        with self._lock:
            if self._reading is None:
                return None, None
            return dict(self._reading), max(0.0, self._clock() - self._reading_at)

    def stop(self):
        with self._lock:
            self._last_request = self._clock() - self._idle_timeout - 1.0

    def _ensure_worker(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        self._spawn(self._loop)

    def _loop(self):
        try:
            while True:
                with self._lock:
                    delay = self._interval * min(8, 2**self._failures)
                self._sleep(delay)
                with self._lock:
                    if self._clock() - self._last_request > self._idle_timeout:
                        return
                self._probe()
        finally:
            with self._lock:
                self._running = False

    def _probe(self):
        with self._lock:
            executable = self._executable
            device_uuid, device_index = self._target
        started = self._clock()
        reading = None
        try:
            result = self._runner(executable, self._timeout)
            if getattr(result, "returncode", 1) == 0:
                reading = parse_nvidia_smi(result.stdout, device_uuid, device_index)
        except Exception:
            reading = None
        duration = self._clock() - started
        with self._lock:
            if reading is None:
                self._failures = min(self._failures + 1, 3)
                return
            self._failures = 0
            reading["probe_ms"] = round(duration * 1000, 1)
            self._reading = reading
            self._reading_at = self._clock()


class CpuSampler:
    """Owns the process-wide window that ``psutil.cpu_percent()`` measures over.

    ``cpu_percent(interval=None)`` reports the average since *its own previous call anywhere in
    the process*. So the first reading after the panel opens is always exactly 0.0 — charted as a
    genuine idle sample — and any other caller in between collapses the window to the gap between
    those two calls, which turns the next reading into noise. Keeping the timing here means a
    value is only ever returned for a window this sampler observed, and the priming call returns
    `None` instead of a fake zero.
    """

    def __init__(self, minimum_interval=0.25, clock=time.monotonic):
        self._minimum_interval = minimum_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._sampled_at = None
        self._value = None

    def read(self, psutil_module):
        with self._lock:
            now = self._clock()
            if self._sampled_at is None:
                psutil_module.cpu_percent(interval=None)
                self._sampled_at = now
                return None
            if now - self._sampled_at < self._minimum_interval:
                # Too short a window to divide by; the previous average is the honest answer.
                return self._value
            self._value = round(float(psutil_module.cpu_percent(interval=None)), 1)
            self._sampled_at = now
            return self._value


def read_gpu_memory(torch_module):
    """Device VRAM from the driver, with this process's share reported separately.

    ``mem_get_info`` is ``cudaMemGetInfo``: what the driver says is free and total on the device
    this process is bound to, counting every process on the card and the CUDA context itself. It
    is what `nvidia-smi` reports, without the spawn — and unlike the allocator counters it does
    not change meaning when another application takes memory. The allocator counters stay in the
    payload under their own names because "how much has the pipeline got hold of" is a different
    and equally real question; they are never added together.
    """
    if not getattr(torch_module, "cuda", None) or not torch_module.cuda.is_available():
        return None

    stats = {"vram_source": "cuda"}
    total = None
    free = None
    try:
        free, total = (int(value) for value in torch_module.cuda.mem_get_info())
    except Exception:
        free = None
    if total is None:
        try:
            total = int(torch_module.cuda.get_device_properties(0).total_memory)
        except Exception:
            total = None

    reserved = None
    allocated = None
    try:
        reserved = int(torch_module.cuda.memory_reserved())
        allocated = int(torch_module.cuda.memory_allocated())
    except Exception:
        pass

    if total:
        stats["vram_total_mb"] = round(total / MIB)
    if total and free is not None:
        used = max(0, total - free)
        stats["vram_used_mb"] = round(used / MIB)
        stats["vram_free_mb"] = round(free / MIB)
        stats["vram_percent"] = round(used / total * 100, 1)
    elif reserved is not None:
        # No driver query available: say so rather than passing an allocator number off as the
        # device total, which is the substitution this module exists to stop.
        stats["vram_source"] = "torch_allocator"
        stats["vram_used_mb"] = round(reserved / MIB)
        if total:
            stats["vram_free_mb"] = round(max(0, total - reserved) / MIB)
            stats["vram_percent"] = round(min(total, reserved) / total * 100, 1)
    if reserved is not None:
        stats["vram_process_mb"] = round(reserved / MIB)
    if allocated is not None:
        stats["vram_tensors_mb"] = round(allocated / MIB)
    return stats


def read_system_memory(psutil_module):
    """System RAM with the figure and the percentage derived from the same subtraction.

    ``available`` is the number both Windows' Task Manager and Linux's ``MemAvailable`` treat as
    "what a new allocation could actually get", and it is what ``psutil``'s own ``percent`` is
    computed from. Deriving used from it too is what stops the panel printing a total that
    disagrees with its own bar on Linux, where ``virtual_memory().used`` leaves out the reclaimable
    page cache.
    """
    memory = psutil_module.virtual_memory()
    total = int(memory.total)
    available = int(getattr(memory, "available", 0) or 0)
    used = max(0, total - available)
    stats = {
        "ram_total_mb": round(total / MIB),
        "ram_available_mb": round(available / MIB),
        "ram_used_mb": round(used / MIB),
        "ram_percent": round(used / total * 100, 1) if total else None,
        # Kept for the settings summary, but at two decimals: a tenth of a gigabyte is 100 MB, and
        # rounding to it turned the memory curve into a staircase.
        "ram_total_gb": round(total / GIB, 2),
        "ram_used_gb": round(used / GIB, 2),
    }
    try:
        swap = psutil_module.swap_memory()
        stats["swap_used_mb"] = round(int(swap.used) / MIB)
        stats["swap_total_mb"] = round(int(swap.total) / MIB)
    except Exception:
        pass
    return stats


def read_process_memory(psutil_module, process=None):
    """Resident memory held by the inference process and its workers.

    Under every memory mode below ``high_vram`` the weights live in host RAM between steps, and
    the RTX VSR path runs in a child process, so "how much of that 32 GB is this program" is the
    question the system section is really being asked — and nothing in the panel answered it.
    Shared pages count once per process, so a forked worker inflates the total slightly; on
    Windows, where workers are always spawned, the sum is exact.
    """
    try:
        process = process or psutil_module.Process()
        total = int(process.memory_info().rss)
    except Exception:
        return None
    workers = 0
    try:
        for child in process.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
                workers += 1
            except Exception:
                continue
    except Exception:
        pass
    return {"process_ram_mb": round(total / MIB), "process_workers": workers}


def probe(
    torch_module=None,
    psutil_module=None,
    smi=None,
    cpu=None,
    process=None,
    device_uuid=None,
    device_index=None,
    max_sensor_age=15.0,
):
    """Collect one consistent sample.

    Nothing here raises: a monitor that returns 503 because a sensor is missing is worse than one
    that omits the field. What it will not do is substitute — an absent reading stays absent so
    the chart can break the line instead of drawing a zero that looks like an idle GPU.
    """
    stats = {"sampled_at": time.time()}

    if torch_module is not None:
        gpu = read_gpu_memory(torch_module)
        if gpu:
            stats.update(gpu)

    if smi is not None:
        reading, age = smi.read(device_uuid, device_index)
        if reading and age is not None and age > max_sensor_age:
            # The probe has been failing long enough that the cached temperature and utilisation
            # describe a different moment. Reporting the gap is honest; charting the old numbers
            # into a live series would draw a flat line that looks like a measurement.
            stats["gpu_sensor_age_ms"] = round(age * 1000)
            stats["gpu_sensor_stale"] = True
            reading = None
        if reading:
            smi_used = reading.pop("smi_vram_used_mb", None)
            smi_total = reading.pop("smi_vram_total_mb", None)
            probe_ms = reading.pop("probe_ms", None)
            stats.update({key: value for key, value in reading.items() if value is not None})
            # Only used where torch has no CUDA device to ask about — a CPU-only build on a
            # machine that still has a driver. It never overwrites a driver figure.
            if "vram_total_mb" not in stats and smi_total:
                stats["vram_source"] = "nvidia_smi"
                stats["vram_total_mb"] = round(smi_total)
                if smi_used is not None:
                    stats["vram_used_mb"] = round(smi_used)
                    stats["vram_free_mb"] = round(max(0.0, smi_total - smi_used))
                    stats["vram_percent"] = round(smi_used / smi_total * 100, 1) if smi_total else None
            if age is not None:
                stats["gpu_sensor_age_ms"] = round(age * 1000)
            if probe_ms is not None:
                stats["gpu_sensor_probe_ms"] = probe_ms
            stats["gpu_sensor_source"] = "nvidia_smi"

    if psutil_module is not None:
        try:
            stats.update(read_system_memory(psutil_module))
        except Exception:
            pass
        if cpu is not None:
            try:
                value = cpu.read(psutil_module)
            except Exception:
                value = None
            if value is not None:
                stats["cpu_percent"] = value
        try:
            stats["cpu_cores"] = psutil_module.cpu_count(logical=True)
        except Exception:
            pass
        resident = read_process_memory(psutil_module, process)
        if resident:
            stats.update(resident)

    return stats
