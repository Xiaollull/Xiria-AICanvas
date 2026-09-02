import unittest
import uuid
from types import SimpleNamespace

from backend.hardware_probe import (
    GIB,
    MIB,
    CpuSampler,
    NvidiaSmiSampler,
    normalise_gpu_uuid,
    parse_nvidia_smi,
    probe,
    read_gpu_memory,
    read_process_memory,
    read_system_memory,
)


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeCuda:
    def __init__(self, available=True, free=None, total=None, reserved=0, allocated=0, driver_query=True):
        self._available = available
        self._free = free
        self._total = total
        self._reserved = reserved
        self._allocated = allocated
        self._driver_query = driver_query

    def is_available(self):
        return self._available

    def mem_get_info(self):
        if not self._driver_query:
            raise RuntimeError("no CUDA context on this device")
        return self._free, self._total

    def memory_reserved(self):
        return self._reserved

    def memory_allocated(self):
        return self._allocated

    def get_device_properties(self, _index):
        return SimpleNamespace(total_memory=self._total)


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


class FakeProcess:
    def __init__(self, rss, children=()):
        self._rss = rss
        self._children = list(children)

    def memory_info(self):
        if self._rss is None:
            raise RuntimeError("no such process")
        return SimpleNamespace(rss=self._rss)

    def children(self, recursive=False):
        del recursive
        return self._children


class FakePsutil:
    def __init__(self, total=8 * GIB, available=4 * GIB, used=None, percent=None, cpu_values=(), process=None, cores=16):
        self._total = total
        self._available = available
        self._used = total - available if used is None else used
        self._percent = percent
        self._cpu_values = list(cpu_values)
        self._process = process or FakeProcess(256 * MIB)
        self._cores = cores
        self.cpu_calls = 0

    def virtual_memory(self):
        percent = self._percent
        if percent is None:
            percent = (self._total - self._available) / self._total * 100
        return SimpleNamespace(total=self._total, available=self._available, used=self._used, percent=percent)

    def swap_memory(self):
        return SimpleNamespace(total=2 * GIB, used=512 * MIB)

    def cpu_percent(self, interval=None):
        del interval
        self.cpu_calls += 1
        return self._cpu_values.pop(0) if self._cpu_values else 0.0

    def cpu_count(self, logical=True):
        del logical
        return self._cores

    def Process(self, pid=None):  # noqa: N802 - mirrors psutil's own name
        del pid
        return self._process


class StubSmi:
    def __init__(self, reading, age=0.0):
        self._reading = reading
        self._age = age
        self.reads = []

    def read(self, device_uuid=None, device_index=None):
        self.reads.append((device_uuid, device_index))
        if self._reading is None:
            return None, None
        return dict(self._reading), self._age


TWO_GPUS = (
    "0, GPU-11111111-2222-3333-4444-555555555555, NVIDIA GeForce RTX 3090, 41, 3, 812, 24576, 32.10, 30\n"
    "1, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, NVIDIA GeForce RTX 4090, 63, 97, 21044, 24564, 411.02, 74\n"
)
ONE_GPU = "0, GPU-11111111-2222-3333-4444-555555555555, NVIDIA GeForce RTX 3090, 41, 3, 812, 24576, 32.10, 30\n"
ONE_GPU_BUSY = "0, GPU-11111111-2222-3333-4444-555555555555, NVIDIA GeForce RTX 3090, 76, 96, 19488, 24576, 344.87, 82\n"


class NvidiaSmiParsingTests(unittest.TestCase):
    def test_uuid_match_wins_over_row_order(self):
        # `CUDA_DEVICE_ORDER=FASTEST_FIRST` makes the 4090 torch's device 0 while nvidia-smi still
        # lists it second. Reading row 0 would chart the idle card's sensors.
        reading = parse_nvidia_smi(TWO_GPUS, device_uuid=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"))
        self.assertEqual(reading["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(reading["gpu_util"], 97.0)
        self.assertEqual(reading["gpu_temp"], 63.0)

    def test_driver_index_is_the_fallback_when_torch_reports_no_uuid(self):
        self.assertEqual(parse_nvidia_smi(TWO_GPUS, device_index="1")["gpu_temp"], 63.0)

    def test_an_unmatched_multi_gpu_listing_reports_nothing(self):
        self.assertIsNone(parse_nvidia_smi(TWO_GPUS, device_index="7"))

    def test_a_single_gpu_listing_is_unambiguous_whatever_the_numbering(self):
        self.assertEqual(parse_nvidia_smi(ONE_GPU, device_index="3")["gpu_util"], 3.0)

    def test_unsupported_sensors_are_absent_rather_than_zero(self):
        line = "0, GPU-11111111-2222-3333-4444-555555555555, NVIDIA A100 80GB PCIe, 44, 12, 900, 81920, [N/A], [Not Supported]"
        reading = parse_nvidia_smi(line)
        self.assertIsNone(reading["power_w"])
        self.assertIsNone(reading["fan_speed"])
        self.assertEqual(reading["gpu_util"], 12.0)

    def test_uuid_forms_compare_equal_across_torch_and_nvidia_smi(self):
        self.assertEqual(
            normalise_gpu_uuid("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            normalise_gpu_uuid(uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")),
        )


class GpuMemoryTests(unittest.TestCase):
    def test_the_device_figure_comes_from_the_driver_not_the_allocator(self):
        stats = read_gpu_memory(FakeTorch(FakeCuda(free=6 * GIB, total=24 * GIB, reserved=9 * GIB, allocated=7 * GIB)))
        self.assertEqual(stats["vram_source"], "cuda")
        self.assertEqual(stats["vram_used_mb"], 18 * 1024)
        self.assertEqual(stats["vram_free_mb"], 6 * 1024)
        self.assertEqual(stats["vram_percent"], 75.0)

    def test_this_process_share_is_reported_beside_the_device_never_instead_of_it(self):
        stats = read_gpu_memory(FakeTorch(FakeCuda(free=6 * GIB, total=24 * GIB, reserved=9 * GIB, allocated=7 * GIB)))
        self.assertEqual(stats["vram_process_mb"], 9 * 1024)
        self.assertEqual(stats["vram_tensors_mb"], 7 * 1024)
        self.assertNotEqual(stats["vram_used_mb"], stats["vram_process_mb"])

    def test_the_allocator_fallback_is_labelled_rather_than_passed_off_as_the_device(self):
        cuda = FakeCuda(total=24 * GIB, reserved=9 * GIB, allocated=7 * GIB, driver_query=False)
        stats = read_gpu_memory(FakeTorch(cuda))
        self.assertEqual(stats["vram_source"], "torch_allocator")
        self.assertEqual(stats["vram_used_mb"], 9 * 1024)
        self.assertEqual(stats["vram_total_mb"], 24 * 1024)

    def test_no_cuda_device_reports_nothing_at_all(self):
        self.assertIsNone(read_gpu_memory(FakeTorch(FakeCuda(available=False))))


class SystemMemoryTests(unittest.TestCase):
    def test_used_and_percent_come_from_the_same_subtraction(self):
        # A Linux host: psutil's own `used` leaves out the page cache that `available` counts as
        # reclaimable, so `.used` and `.percent` describe different quantities. Printing both put
        # a figure and a bar that disagreed by 9 GB on the same panel.
        stats = read_system_memory(FakePsutil(total=32 * GIB, available=20 * GIB, used=3 * GIB, percent=37.5))
        self.assertEqual(stats["ram_used_mb"], 12 * 1024)
        self.assertEqual(stats["ram_available_mb"], 20 * 1024)
        self.assertEqual(stats["ram_percent"], 37.5)
        self.assertAlmostEqual(stats["ram_used_mb"] / stats["ram_total_mb"] * 100, stats["ram_percent"], places=3)

    def test_megabyte_precision_keeps_the_curve_off_the_staircase(self):
        stats = read_system_memory(FakePsutil(total=32 * GIB, available=32 * GIB - 250 * MIB))
        self.assertEqual(stats["ram_used_mb"], 250)
        self.assertEqual(stats["ram_used_gb"], 0.24)


class ProcessMemoryTests(unittest.TestCase):
    def test_workers_count_towards_what_this_program_is_holding(self):
        process = FakeProcess(3 * GIB, children=[FakeProcess(512 * MIB), FakeProcess(256 * MIB)])
        stats = read_process_memory(FakePsutil(process=process), process)
        self.assertEqual(stats["process_ram_mb"], 3 * 1024 + 768)
        self.assertEqual(stats["process_workers"], 2)

    def test_a_worker_that_exits_mid_scan_does_not_lose_the_reading(self):
        process = FakeProcess(2 * GIB, children=[FakeProcess(None), FakeProcess(128 * MIB)])
        stats = read_process_memory(FakePsutil(process=process), process)
        self.assertEqual(stats["process_ram_mb"], 2 * 1024 + 128)
        self.assertEqual(stats["process_workers"], 1)


class CpuSamplerTests(unittest.TestCase):
    def test_the_priming_reading_is_absent_rather_than_a_fabricated_zero(self):
        clock = FakeClock()
        psutil_module = FakePsutil(cpu_values=[0.0, 47.5])
        sampler = CpuSampler(clock=clock)
        self.assertIsNone(sampler.read(psutil_module))
        clock.now += 2.0
        self.assertEqual(sampler.read(psutil_module), 47.5)

    def test_a_window_too_short_to_divide_repeats_the_last_average(self):
        clock = FakeClock()
        psutil_module = FakePsutil(cpu_values=[0.0, 47.5, 99.9])
        sampler = CpuSampler(clock=clock)
        sampler.read(psutil_module)
        clock.now += 2.0
        self.assertEqual(sampler.read(psutil_module), 47.5)
        clock.now += 0.01
        self.assertEqual(sampler.read(psutil_module), 47.5)
        self.assertEqual(psutil_module.cpu_calls, 2)


class NvidiaSmiSamplerTests(unittest.TestCase):
    def build(self, results, *, clock, calls, sleeps, retire_after=1):
        holder = {}

        def runner(executable, timeout):
            del timeout
            calls.append(executable)
            return results.pop(0) if results else SimpleNamespace(returncode=1, stdout="")

        def sleep(delay):
            sleeps.append(delay)
            # `spawn` runs the loop inline here, so the loop has to see itself go idle to hand
            # control back. Retiring it through `stop()` rather than by winding the clock forward
            # keeps the ages the test asserts on equal to the time the test actually elapsed.
            if len(sleeps) >= retire_after:
                holder["sampler"].stop()

        holder["sampler"] = NvidiaSmiSampler(
            runner=runner,
            locate=lambda: "nvidia-smi",
            spawn=lambda target: target(),
            clock=clock,
            sleep=sleep,
        )
        return holder["sampler"]

    def test_the_first_read_probes_once_and_the_next_is_served_from_cache(self):
        clock, calls, sleeps = FakeClock(), [], []
        results = [SimpleNamespace(returncode=0, stdout=ONE_GPU)]
        sampler = self.build(results, clock=clock, calls=calls, sleeps=sleeps)
        reading, age = sampler.read(device_index="0")
        self.assertEqual(reading["gpu_util"], 3.0)
        self.assertEqual(age, 0.0)
        self.assertEqual(len(calls), 1)
        sampler.read(device_index="0")
        self.assertEqual(len(calls), 1)

    def test_a_missing_nvidia_smi_never_spawns_anything(self):
        calls = []

        def runner(executable, timeout):
            calls.append(executable)
            raise AssertionError("nvidia-smi must not be spawned when it is not installed")

        sampler = NvidiaSmiSampler(runner=runner, locate=lambda: None, spawn=lambda target: target())
        self.assertEqual(sampler.read(), (None, None))
        self.assertEqual(calls, [])
        self.assertIs(sampler.available, False)

    def test_a_failing_probe_backs_off_instead_of_hammering_the_driver(self):
        clock, calls, sleeps = FakeClock(), [], []
        sampler = self.build([], clock=clock, calls=calls, sleeps=sleeps, retire_after=3)
        self.assertEqual(sampler.read(), (None, None))
        self.assertEqual(sleeps, [4.0, 8.0, 16.0])
        self.assertEqual(len(calls), 3)

    def test_a_reading_younger_than_the_interval_is_served_from_cache_with_its_age(self):
        clock, calls, sleeps = FakeClock(), [], []
        results = [SimpleNamespace(returncode=0, stdout=ONE_GPU)]
        sampler = self.build(results, clock=clock, calls=calls, sleeps=sleeps)
        sampler.read(device_index="0")
        clock.now += 1.0
        _reading, age = sampler.read(device_index="0")
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(age, 1.0, places=3)

    def test_a_retired_worker_refreshes_instead_of_serving_the_last_session_reading(self):
        clock, calls, sleeps = FakeClock(), [], []
        results = [
            SimpleNamespace(returncode=0, stdout=ONE_GPU),
            SimpleNamespace(returncode=0, stdout=ONE_GPU_BUSY),
        ]
        sampler = self.build(results, clock=clock, calls=calls, sleeps=sleeps)
        sampler.read(device_index="0")
        # The panel was closed and the worker retired; reopening it must not show the temperature
        # the card had a minute ago.
        clock.now += 60.0
        reading, age = sampler.read(device_index="0")
        self.assertEqual(len(calls), 2)
        self.assertEqual(age, 0.0)
        self.assertEqual(reading["gpu_util"], 96.0)


class ProbeCompositionTests(unittest.TestCase):
    def sensors(self, **overrides):
        reading = {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_temp": 70.0,
            "gpu_util": 88.0,
            "power_w": None,
            "fan_speed": 74.0,
            "smi_vram_used_mb": 21044.0,
            "smi_vram_total_mb": 24564.0,
            "probe_ms": 312.5,
        }
        reading.update(overrides)
        return reading

    def test_nvidia_smi_never_overwrites_the_driver_vram_figure(self):
        torch_module = FakeTorch(FakeCuda(free=6 * GIB, total=24 * GIB, reserved=9 * GIB, allocated=7 * GIB))
        smi = StubSmi(self.sensors(), age=0.4)
        stats = probe(torch_module=torch_module, smi=smi)
        self.assertEqual(stats["vram_source"], "cuda")
        self.assertEqual(stats["vram_used_mb"], 18 * 1024)
        self.assertEqual(stats["gpu_util"], 88.0)
        self.assertEqual(stats["gpu_sensor_age_ms"], 400)
        self.assertEqual(stats["gpu_sensor_probe_ms"], 312.5)
        self.assertNotIn("smi_vram_used_mb", stats)

    def test_nvidia_smi_memory_is_used_only_where_torch_has_no_device(self):
        smi = StubSmi(self.sensors(), age=0.1)
        stats = probe(torch_module=FakeTorch(FakeCuda(available=False)), smi=smi)
        self.assertEqual(stats["vram_source"], "nvidia_smi")
        self.assertEqual(stats["vram_used_mb"], 21044)
        self.assertEqual(stats["vram_free_mb"], 3520)

    def test_an_unsupported_sensor_is_omitted_so_the_chart_can_break_the_line(self):
        smi = StubSmi(self.sensors(gpu_util=None), age=0.0)
        stats = probe(torch_module=FakeTorch(FakeCuda(available=False)), smi=smi)
        self.assertNotIn("gpu_util", stats)
        self.assertNotIn("power_w", stats)

    def test_a_sensor_reading_too_old_to_be_current_is_withheld_and_declared(self):
        # A probe that keeps failing leaves the last good reading in the cache. Charting a
        # temperature from a minute ago into a live series draws a flat line that reads as a
        # measurement; saying how old it is does not.
        smi = StubSmi(self.sensors(), age=42.0)
        stats = probe(torch_module=FakeTorch(FakeCuda(available=False)), smi=smi)
        self.assertTrue(stats["gpu_sensor_stale"])
        self.assertEqual(stats["gpu_sensor_age_ms"], 42_000)
        self.assertNotIn("gpu_util", stats)
        self.assertNotIn("gpu_temp", stats)
        self.assertNotIn("vram_used_mb", stats)

    def test_the_priming_cpu_sample_is_not_reported_as_an_idle_machine(self):
        clock = FakeClock()
        psutil_module = FakePsutil(total=32 * GIB, available=20 * GIB, cpu_values=[0.0, 63.0])
        cpu = CpuSampler(clock=clock)
        first = probe(psutil_module=psutil_module, cpu=cpu)
        self.assertNotIn("cpu_percent", first)
        self.assertEqual(first["ram_used_mb"], 12 * 1024)
        clock.now += 2.0
        second = probe(psutil_module=psutil_module, cpu=cpu)
        self.assertEqual(second["cpu_percent"], 63.0)

    def test_a_machine_with_no_sensors_reports_absence_not_zeroes(self):
        stats = probe(torch_module=FakeTorch(FakeCuda(available=False)))
        for key in ("gpu_util", "gpu_temp", "vram_used_mb", "cpu_percent", "ram_used_mb"):
            self.assertNotIn(key, stats)
        self.assertIn("sampled_at", stats)


if __name__ == "__main__":
    unittest.main()
