import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  ENGINE_SETTINGS_SCHEMA_VERSION,
  SETTINGS_ENGINES,
  emptyEngineSettingsMap,
  engineSettingsDefaults,
  engineSettingsFor,
  normalizeEngineSettings,
  normalizeEngineSettingsMap,
  withEngineSettings,
} from "../src/engine-settings.js";

// Each engine owns its own parameters. The workspace used to hold a single set and rewrite parts of
// it on every engine change, so SD -> Krea2 -> SD did not return the SD workspace; it returned
// whatever had survived being pushed through Krea 2. These tests are about that isolation, and
// about the coercion that keeps a record valid for the engine it belongs to.

test("every ready engine has a record, and each starts from its own defaults", () => {
  const map = emptyEngineSettingsMap();
  assert.equal(map.schemaVersion, ENGINE_SETTINGS_SCHEMA_VERSION);
  assert.deepEqual(Object.keys(map.byEngine).sort(), [...SETTINGS_ENGINES].sort());

  // The split engines default to what their native runtimes ship; SD and iL keep the ComfyUI pair.
  assert.equal(engineSettingsDefaults("SD").sampler, "dpmpp_2m");
  assert.equal(engineSettingsDefaults("SD").scheduler, "karras");
  for (const engine of ["Anima", "Flux", "Flux2", "Krea2"]) {
    assert.equal(engineSettingsDefaults(engine).sampler, "euler", engine);
    assert.equal(engineSettingsDefaults(engine).scheduler, "simple", engine);
  }
});

test("writing one engine's record leaves every other engine untouched", () => {
  const start = emptyEngineSettingsMap();
  const after = withEngineSettings(start, "Krea2", { ...engineSettingsDefaults("Krea2"), steps: 44, size: { width: 1536, height: 1152 } });

  assert.equal(engineSettingsFor(after, "Krea2").steps, 44);
  assert.deepEqual(engineSettingsFor(after, "Krea2").size, { width: 1536, height: 1152 });
  for (const engine of SETTINGS_ENGINES.filter((name) => name !== "Krea2")) {
    assert.deepEqual(engineSettingsFor(after, engine), engineSettingsDefaults(engine), `${engine} moved`);
  }
  // The input map is not mutated, so a caller holding the previous value still sees it.
  assert.equal(engineSettingsFor(start, "Krea2").steps, engineSettingsDefaults("Krea2").steps);
});

test("a round trip through another engine brings the first one back unchanged", () => {
  // This is the reported bug in miniature: leave SD, work in Krea 2, come back.
  const sd = { ...engineSettingsDefaults("SD"), steps: 32, cfg: 8, sampler: "dpmpp_2m", checkpoint: "sd.safetensors", hires: { ...engineSettingsDefaults("SD").hires, enabled: true, scale: 2 } };
  let map = withEngineSettings(emptyEngineSettingsMap(), "SD", sd);
  const stored = engineSettingsFor(map, "SD");

  map = withEngineSettings(map, "Krea2", { ...engineSettingsDefaults("Krea2"), steps: 8, cfg: 1, diffusionModel: "krea2.safetensors" });
  assert.deepEqual(engineSettingsFor(map, "SD"), stored, "working in Krea 2 changed the SD record");
  assert.equal(engineSettingsFor(map, "SD").hires.enabled, true);
  assert.equal(engineSettingsFor(map, "Krea2").steps, 8);
});

test("a record only keeps the model pickers its engine actually has", () => {
  // Otherwise a stale value from the other kind of engine reappears as a phantom selection.
  const split = normalizeEngineSettings("Krea2", { checkpoint: "sd.safetensors", diffusionModel: "k.safetensors", vae: "v.safetensors" });
  assert.equal(split.checkpoint, "");
  assert.equal(split.diffusionModel, "k.safetensors");

  const single = normalizeEngineSettings("SD", { checkpoint: "sd.safetensors", diffusionModel: "k.safetensors", vae: "v.safetensors" });
  assert.equal(single.checkpoint, "sd.safetensors");
  assert.equal(single.diffusionModel, "");
  assert.equal(single.vae, "");

  // Only FLUX.1 carries a second text encoder.
  assert.equal(normalizeEngineSettings("Flux", { textEncoder2: "t5.safetensors" }).textEncoder2, "t5.safetensors");
  assert.equal(normalizeEngineSettings("Flux2", { textEncoder2: "t5.safetensors" }).textEncoder2, "");
});

test("guidance is coerced to what the engine can run", () => {
  // Distillation removes the unconditional branch, so there is nothing to guide against.
  for (const engine of ["Flux", "Flux2"]) {
    assert.equal(normalizeEngineSettings(engine, { guidance: "pag" }).guidance, "none", engine);
    assert.equal(normalizeEngineSettings(engine, { guidance: "cfg_zero_star" }).guidance, "none", engine);
  }
  // CFG-Zero* rescales a flow-matching step, which the epsilon-prediction engines do not have.
  assert.equal(normalizeEngineSettings("SD", { guidance: "cfg_zero_star" }).guidance, "none");
  assert.equal(normalizeEngineSettings("Anima", { guidance: "cfg_zero_star" }).guidance, "cfg_zero_star");
  // Krea 2 keeps CFG-Zero* but has no attention override for PAG.
  assert.equal(normalizeEngineSettings("Krea2", { guidance: "cfg_zero_star" }).guidance, "cfg_zero_star");
  assert.equal(normalizeEngineSettings("Krea2", { guidance: "pag" }).guidance, "none");
  assert.equal(normalizeEngineSettings("Anima", { guidance: "pag" }).guidance, "pag");
});

test("a pre-split workspace seeds every engine, not just the selected one", () => {
  // Anything else looks like data loss on the first launch after upgrading: the settings survive
  // under one engine and every other engine opens on defaults.
  const legacy = { steps: 33, cfg: 9, size: { width: 1216, height: 832 }, sampler: "euler_ancestral", guidance: "cfg_zero_star" };
  const { container, migrated } = normalizeEngineSettingsMap(undefined, { fieldMissing: true, legacy });
  assert.equal(migrated, true);
  for (const engine of SETTINGS_ENGINES) {
    const record = engineSettingsFor(container, engine);
    assert.equal(record.steps, 33, engine);
    assert.deepEqual(record.size, { width: 1216, height: 832 }, engine);
    assert.equal(record.sampler, "euler_ancestral", engine);
  }
  // Each copy is still coerced to its own engine, so the fold cannot plant an invalid request.
  assert.equal(engineSettingsFor(container, "Anima").guidance, "cfg_zero_star");
  assert.equal(engineSettingsFor(container, "SD").guidance, "none");
  assert.equal(engineSettingsFor(container, "Flux").guidance, "none");
});

test("a first run with nothing saved opens on defaults rather than failing", () => {
  const { container, migrated, warning } = normalizeEngineSettingsMap(undefined, { fieldMissing: true });
  assert.equal(migrated, false);
  assert.equal(warning, "");
  assert.deepEqual(engineSettingsFor(container, "SD"), engineSettingsDefaults("SD"));
});

test("a corrupt record is reset and reported, and never takes the others with it", () => {
  const map = { schemaVersion: 1, byEngine: { SD: "not a record", Krea2: { steps: 12 } } };
  const { container, rejected, warning } = normalizeEngineSettingsMap(map);
  assert.equal(rejected, 1);
  assert.match(warning, /1/);
  assert.deepEqual(engineSettingsFor(container, "SD"), engineSettingsDefaults("SD"));
  assert.equal(engineSettingsFor(container, "Krea2").steps, 12);
});

test("an engine missing from a saved map is an empty scope, not a corrupt file", () => {
  // A map written before an engine shipped cannot carry its record; treating that as corruption
  // would strand every engine that is in the file.
  const { container, rejected } = normalizeEngineSettingsMap({ schemaVersion: 1, byEngine: { SD: { steps: 21 } } });
  assert.equal(rejected, 0);
  assert.equal(engineSettingsFor(container, "SD").steps, 21);
  assert.deepEqual(engineSettingsFor(container, "Krea2"), engineSettingsDefaults("Krea2"));
});

test("out-of-range values are clamped rather than passed to the service", () => {
  const record = normalizeEngineSettings("SD", {
    steps: 9999, cfg: -4, denoise: 3, imagesPerBatch: 100, batchCount: 0,
    size: { width: 99999, height: 50 }, rtx: { scale: 42, quality: "insane" },
  });
  assert.equal(record.steps, 60);
  assert.equal(record.cfg, 1);
  assert.equal(record.denoise, 1);
  assert.equal(record.imagesPerBatch, 10);
  assert.equal(record.batchCount, 1);
  assert.equal(record.size.width, 2048);
  assert.equal(record.rtx.scale, 4);
  assert.equal(record.rtx.quality, "ultra");
  // Canvas dimensions stay on the 64-pixel grid the samplers require.
  assert.equal(record.size.height % 64, 0);
});

test("a post-processing order that lost a stage still runs it", () => {
  const record = normalizeEngineSettings("SD", { postprocessOrder: ["rtx", "rtx", "nonsense"] });
  assert.deepEqual(record.postprocessOrder, ["rtx", "hires", "adetailer"]);
});

test("an unknown engine cannot address the map", () => {
  const map = emptyEngineSettingsMap();
  assert.deepEqual(withEngineSettings(map, "NotAnEngine", { steps: 5 }), map);
  // Reading falls back to the first ready engine rather than returning undefined.
  assert.deepEqual(engineSettingsFor(map, "NotAnEngine"), engineSettingsDefaults(SETTINGS_ENGINES[0]));
});

test("the workspace swaps records on an engine change instead of editing one set", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const select = app.slice(app.indexOf("const selectModel = (nextModel)"), app.indexOf("const selectCheckpoint"));
  assert.match(select, /commitEngineSettings\(model, captureEngineSettings\(\)\)/,
    "the engine being left must have its parameters stored");
  assert.match(select, /applyEngineSettings\(restored\)/,
    "the engine being opened must have its parameters restored");
  // The old behaviour: coerce the shared values in place and clear the pickers outright.
  assert.doesNotMatch(select, /setCheckpoint\(""\)/);
  assert.doesNotMatch(select, /ANIMA_SAMPLERS\.includes\(current\)/);

  // Applying a Gallery card that names another engine is also a departure from the current one.
  assert.match(app, /if \(engineChanged\) commitEngineSettings\(model, captureEngineSettings\(\)\)/);
});

test("the map is persisted with the workspace and restored from it", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /engineSettingsByEngine: withEngineSettings\(engineSettingsByEngine, model, \{/,
    "the selected engine's live values must be folded into its record before saving");
  assert.match(app, /applyEngineSettings\(engineSettingsFor\(workspace\.engineSettingsByEngine, workspace\.model\)\)/);
  assert.match(app, /legacy: saved,/, "a pre-split document must still seed the engines");
});

test("a pre-split workspace's model files seed only the engine they were chosen under", () => {
  // Parameters are sensible everywhere; model files are not. Copying Anima's encoder into Krea 2's
  // record listed it there as a selection Krea 2 cannot load, and the image reader's Apply borrowed
  // it and was refused.
  const legacy = { model: "Anima", steps: 30, diffusionModel: "anima-base-v1.0.safetensors", textEncoder: "qwen_3_06b_base.safetensors", vae: "qwen_image_vae.safetensors" };
  const { container } = normalizeEngineSettingsMap(undefined, { fieldMissing: true, legacy });
  const anima = engineSettingsFor(container, "Anima");
  assert.equal(anima.diffusionModel, "anima-base-v1.0.safetensors");
  assert.equal(anima.textEncoder, "qwen_3_06b_base.safetensors");
  assert.equal(anima.vae, "qwen_image_vae.safetensors");
  for (const engine of ["Flux", "Flux2", "Krea2"]) {
    const record = engineSettingsFor(container, engine);
    assert.equal(record.steps, 30, `${engine} lost the shared parameters`);
    for (const field of ["diffusionModel", "textEncoder", "textEncoder2", "vae"]) {
      assert.equal(record[field], "", `${engine}.${field} was seeded with another engine's file`);
    }
  }
  // Same rule for the single-file engines: an Illustrious checkpoint is not an SD one.
  const fromIllustrious = normalizeEngineSettingsMap(undefined, { fieldMissing: true, legacy: { model: "iL", checkpoint: "wai.safetensors" } }).container;
  assert.equal(engineSettingsFor(fromIllustrious, "iL").checkpoint, "wai.safetensors");
  assert.equal(engineSettingsFor(fromIllustrious, "SD").checkpoint, "");
});
