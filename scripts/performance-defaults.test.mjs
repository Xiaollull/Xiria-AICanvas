import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  DEFAULT_PERFORMANCE,
  STAGED_DECODE_VRAM_CEILING_GB,
  performanceDefaultsFor,
  performancePresetsFor,
  stagedDecodeDefault,
} from "../src/performance-defaults.js";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("the staged decode ships on for cards of 16 GB and under", () => {
  // The sizes actually sold, read as the driver reports them: a marketed size comes back a little
  // short, which is why the ceiling carries half a gigabyte of tolerance.
  for (const vram of [4, 6, 7.99, 8, 11.99, 12, 15.92, 16]) {
    assert.equal(stagedDecodeDefault(vram), true, `${vram} GB starts with the staged decode on`);
  }
  // The next size up is 20 GB, so nothing larger can drift in under the ceiling.
  for (const vram of [19.5, 20, 23.6, 24, 48]) {
    assert.equal(stagedDecodeDefault(vram), false, `${vram} GB has room for both at once`);
  }
  assert.equal(STAGED_DECODE_VRAM_CEILING_GB, 16.5);
  // No card measured -- no CUDA device, or the panel has not heard back yet -- keeps the plain
  // table rather than guessing a small one.
  for (const unknown of [0, -1, NaN, null, undefined]) assert.equal(stagedDecodeDefault(unknown), false);
});

test("it is a default, not a lock: only that one entry follows the card", async () => {
  const small = performanceDefaultsFor(8);
  const large = performanceDefaultsFor(24);
  assert.equal(small.staged_vae_decode, true);
  assert.equal(large.staged_vae_decode, false);
  assert.deepEqual({ ...small, staged_vae_decode: false }, { ...large, staged_vae_decode: false });
  assert.deepEqual(large, DEFAULT_PERFORMANCE);
  // The toggle stays live either way -- nothing about the switch is disabled by the card.
  const app = await read("src/App.jsx");
  const rest = app.slice(app.indexOf("采样后卸载再解码"));
  const row = rest.slice(0, rest.indexOf("performance-toggle-row"));
  assert.match(row, /aria-checked=\{performanceConfig\.staged_vae_decode\}/);
  assert.doesNotMatch(row, /disabled=/, "on by default must not mean forced on");
  assert.match(row, /16 GB 及以下显卡默认开启/, "the panel says why it is already on");
});

test("the recommended cards follow the card only where they claim to", async () => {
  const small = performancePresetsFor(8);
  const large = performancePresetsFor(24);
  const settings = (presets, id) => presets.find((preset) => preset.id === id).settings;
  // "自适应推荐" is the shipped baseline, so it is the one that moves -- and moving it is also what
  // keeps exactly one card highlighted on a fresh install.
  assert.equal(settings(small, "auto").staged_vae_decode, true);
  assert.equal(settings(large, "auto").staged_vae_decode, false);
  // Every other card states its own intent and reads the same on any machine. "高速直通" keeps the
  // model resident precisely because it is for a card that can afford to.
  assert.equal(settings(small, "speed").staged_vae_decode, false);
  assert.deepEqual(settings(small, "speed"), settings(large, "speed"));
  for (const id of ["large-model", "balanced", "low-memory", "ultra-low", "large-canvas"]) {
    assert.equal(settings(small, id).staged_vae_decode, true, `${id} asks for the staged decode outright`);
    assert.deepEqual(settings(small, id), settings(large, id), `${id} does not depend on the card`);
  }
  assert.deepEqual(small.map((preset) => preset.id), large.map((preset) => preset.id));
});

test("the panel is built from the measured card, not from the constant", async () => {
  const app = await read("src/App.jsx");
  assert.match(app, /import \{ DEFAULT_PERFORMANCE, performanceDefaultsFor, performancePresetsFor \} from "\.\/performance-defaults\.js"/);
  assert.match(app, /const performancePresets = performancePresetsFor\(vramGb\);/);
  assert.match(app, /setPerformanceConfig\(\(current\) => \(\{ \.\.\.performanceDefaultsFor\(vramGb\), vram_limit_gb: current\.vram_limit_gb \}\)\)/,
    "恢复建议值 restores the recommendation for this machine");
  // The presets and the baseline live in one module now; a second copy in the page is how the two
  // drifted the last time this was edited.
  assert.doesNotMatch(app, /^const DEFAULT_PERFORMANCE = \{/m);
  assert.doesNotMatch(app, /^const performancePresets = \[/m);
});

test("the frontend baseline and the server's table stay in step", async () => {
  const python = await read("backend/performance_settings.py");
  const table = python.slice(python.indexOf("DEFAULT_PERFORMANCE_SETTINGS = {"), python.indexOf("SETTING_CHOICES"));
  // The MOSS encoder placement is sparse on the server: it stays out of the table so a settings file
  // written before MOSS reads back item for item, and its default is declared on its own.
  const sparse = { moss_text_encoder_placement: "MOSS_TEXT_ENCODER_PLACEMENT_DEFAULT" };
  for (const [key, value] of Object.entries(DEFAULT_PERFORMANCE)) {
    if (key === "vram_limit_gb") continue;
    const literal = typeof value === "boolean" ? (value ? "True" : "False") : `"${value}"`;
    if (sparse[key]) {
      assert.ok(!table.includes(`"${key}":`), `${key} stays out of the server table`);
      assert.ok(python.includes(`${sparse[key]} = ${literal}\n`), `${key} is ${literal} on both sides`);
      continue;
    }
    assert.ok(table.includes(`"${key}": ${literal},`), `${key} is ${literal} on both sides`);
  }
  // The rule itself is written twice -- the server decides what an untouched install runs with, the
  // panel decides what the restore button and the adaptive card offer -- so the number is pinned.
  assert.match(python, /STAGED_DECODE_VRAM_CEILING_GB = 16\.5/);
  assert.match(python, /def read_performance_settings\(path: Path, vram_bytes=None\)/);
  const server = await read("backend/inference_server.py");
  assert.match(server, /performance_settings = read_performance_settings\(PERFORMANCE_SETTINGS_FILE, installed_vram_bytes\(\)\)/,
    "the shipped default has to see the card, or an untouched install never gets it");
});
