import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  appendHardwareSample,
  EMPTY_HARDWARE_HISTORY,
  formatMib,
  HARDWARE_HISTORY_SAMPLES,
  sensorAgeLabel,
  sparklineSegments,
  vramWallPercent,
} from "../src/hardware-monitor.js";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("a reading the backend could not take stays absent instead of becoming zero", () => {
  // The GPU sensors come from `nvidia-smi`; on a machine without it the old panel charted 0 and
  // the curve was indistinguishable from a genuinely idle card.
  const history = appendHardwareSample(EMPTY_HARDWARE_HISTORY, { cpu_percent: 12.5, ram_used_mb: 8192 });
  assert.deepEqual(history.gpu, [null]);
  assert.deepEqual(history.vram, [null]);
  assert.deepEqual(history.cpu, [12.5]);
  assert.deepEqual(history.ram, [8]);
});

test("a dropped poll records a hole rather than dragging every curve to the floor", () => {
  let history = appendHardwareSample(EMPTY_HARDWARE_HISTORY, { gpu_util: 90, cpu_percent: 40, vram_used_mb: 2048, ram_used_mb: 4096 });
  history = appendHardwareSample(history, null);
  assert.deepEqual(history.gpu, [90, null]);
  assert.deepEqual(history.cpu, [40, null]);
  assert.deepEqual(history.vram, [2, null]);
});

test("memory series are converted from MiB once, not rounded to a tenth of a gigabyte", () => {
  // `ram_used_gb` used to arrive already rounded to 0.1 GB — a 100 MB step that turned the memory
  // curve into a staircase. The MiB field keeps the resolution the samples actually have.
  const history = appendHardwareSample(EMPTY_HARDWARE_HISTORY, { ram_used_mb: 12_450, vram_used_mb: 18_500 });
  assert.equal(history.ram[0].toFixed(3), "12.158");
  assert.equal(history.vram[0].toFixed(3), "18.066");
});

test("history is bounded so an open panel cannot grow without limit", () => {
  let history = EMPTY_HARDWARE_HISTORY;
  for (let index = 0; index < HARDWARE_HISTORY_SAMPLES + 12; index += 1) {
    history = appendHardwareSample(history, { cpu_percent: index });
  }
  assert.equal(history.cpu.length, HARDWARE_HISTORY_SAMPLES);
  assert.equal(history.cpu.at(-1), HARDWARE_HISTORY_SAMPLES + 11);
});

test("the sparkline breaks across a gap instead of drawing through it", () => {
  const segments = sparklineSegments([10, 20, null, 40, 50], 100);
  assert.equal(segments.length, 2);
  assert.equal(segments[0], "0,90 25,80");
  assert.equal(segments[1], "75,60 100,50");
});

test("a lone sample between two gaps is still drawn", () => {
  const segments = sparklineSegments([null, 50, null], 100);
  assert.deepEqual(segments, ["50,50 50,50"]);
});

test("a sample above the ceiling is clamped into the chart rather than drawn outside it", () => {
  assert.deepEqual(sparklineSegments([150], 100), ["0,0 0,0"]);
  assert.deepEqual(sparklineSegments([-20], 100), ["0,100 0,100"]);
});

test("memory is formatted from MiB with a unit that matches the number", () => {
  assert.equal(formatMib(812), "812 MB");
  assert.equal(formatMib(18_432), "18.0 GB");
  assert.equal(formatMib(undefined), "--");
  assert.equal(formatMib(null), "--");
});

test("the VRAM wall is marked only when it actually caps the card", () => {
  assert.equal(vramWallPercent({ vram_limit_mb: 18_432, vram_total_mb: 24_576 }), 75);
  // Automatic management resolves the wall to the whole card; a mark at 100% says nothing.
  assert.equal(vramWallPercent({ vram_limit_mb: 24_576, vram_total_mb: 24_576 }), null);
  assert.equal(vramWallPercent({ vram_total_mb: 24_576 }), null);
});

test("sensor age is stated so a stale nvidia-smi reading is not passed off as live", () => {
  assert.equal(sensorAgeLabel({ gpu_sensor_age_ms: 400 }), "刚刚");
  assert.equal(sensorAgeLabel({ gpu_sensor_age_ms: 4500 }), "4.5 秒前");
  assert.equal(sensorAgeLabel({}), null);
});

test("the monitor polls by chaining rather than on a fixed interval", async () => {
  const source = await read("src/App.jsx");
  const effect = source.slice(source.indexOf("if (!hardwareMonitorOpen) return undefined;"));
  const body = effect.slice(0, effect.indexOf("}, [hardwareMonitorOpen]);"));
  // A reply slower than the period used to overlap the next request and land out of order, which
  // put samples on the chart at times they were not taken.
  assert.match(body, /setTimeout\(poll, HARDWARE_POLL_MS\)/);
  assert.doesNotMatch(body, /setInterval/);
  assert.match(body, /controller\.abort\(\)/);
});

test("the panel separates the whole card from this process", async () => {
  const source = await read("src/App.jsx");
  // One field used to mean "tensors this process holds" or "everything on the card" depending on
  // whether nvidia-smi was installed. Both are shown now, each labelled.
  assert.match(source, /显存 · 整卡（含其他程序）/);
  assert.match(source, /本程序 \{formatMib\(hardwareStats\?\.vram_process_mb\)\}/);
  assert.match(source, /本程序内存/);
  assert.match(source, /formatMib\(hardwareStats\?\.ram_used_mb\)/);
});
