// P8 AUTO · 动态决策: the label, the below-6-GB notice, and the manual tiers left exactly as they were.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DEFAULT_PERFORMANCE, performancePresetsFor } from "../src/performance-defaults.js";

const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");

test("AUTO is shown under its own name and the settings key stays `auto`", () => {
  assert.equal(DEFAULT_PERFORMANCE.memory_mode, "auto");
  assert.match(app, /\["auto", "AUTO · 动态决策", "[^"]+"\]/);
  assert.doesNotMatch(app, /\["auto", "动态预算"/);
  const auto = performancePresetsFor(8).find((preset) => preset.id === "auto");
  assert.equal(auto.label, "AUTO · 动态决策");
  assert.equal(auto.settings.memory_mode, "auto");
});

test("below the recommended 6 GB AUTO warns and stays selectable", () => {
  assert.match(app, /低于推荐配置，可能无法完成任务/);
  assert.match(app, /const autoBelowRecommended = performanceConfig\.memory_mode === "auto"/);
  assert.match(app, /auto_below_recommended \?\? \(vramGb > 0 && vramGb < 5\.5\)/);
  // What disables a memory-mode button is only ever an unavailable kernel or dtype; nothing about
  // the card's size can take AUTO away.
  const unavailable = app.match(/const unavailable = ([\s\S]*?);\n/)[1];
  assert.doesNotMatch(unavailable, /memory_mode/);
  assert.match(styles, /\.auto-vram-notice \{/);
});

test("the running path is reported through the label AUTO carries", () => {
  assert.match(app, /actualPerformanceValue\(performanceActive\.memory_label, inferenceHealth\?\.memory_label/);
});

test("every manual preset is item-by-item what it was", () => {
  const presets = Object.fromEntries(performancePresetsFor(8).map((preset) => [preset.id, preset.settings]));
  assert.deepEqual(presets["large-model"], { ...DEFAULT_PERFORMANCE, memory_mode: "sdxl_balanced", staged_vae_decode: true });
  assert.deepEqual(presets.speed, { ...DEFAULT_PERFORMANCE, memory_mode: "high_vram", vae_mode: "full" });
  assert.deepEqual(presets.balanced, { ...DEFAULT_PERFORMANCE, memory_mode: "normal_vram", vae_mode: "sliced", staged_vae_decode: true });
  assert.deepEqual(presets["low-memory"], { ...DEFAULT_PERFORMANCE, memory_mode: "low_vram", attention_backend: "sliced", vae_mode: "tiled", keep_model_cached: false, staged_vae_decode: true });
  assert.deepEqual(presets["ultra-low"], { ...DEFAULT_PERFORMANCE, memory_mode: "ultra_low_vram", attention_backend: "sliced", compute_dtype: "fp16", vae_mode: "tiled", cuda_math: "strict", keep_model_cached: false, allow_shared_memory: false, staged_vae_decode: true });
});
