// MOSS-Image as the browser sees it: the sampling block, its wire form, and the places the seventh
// engine reaches (engine records, Gallery cards, the image reader, image-to-image). The backend side
// of the same contract is `backend/tests/test_moss_integration.py`.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MOSS_DEFAULTS,
  MOSS_DIT_DEPTH,
  MOSS_SAMPLERS,
  MOSS_SAMPLING_DEFAULTS,
  MOSS_SCHEDULERS,
  describeMossSampling,
  formatSkipLayers,
  mossRequestFields,
  mossSamplerFor,
  mossSamplingFromPayload,
  mossSamplingPayload,
  mossSchedulerFor,
  normalizeMossSampling,
  parseSkipLayers,
} from "../src/moss-engine.js";
import { SETTINGS_ENGINES, engineGuidance, engineSettingsDefaults, isSplitEngine, normalizeEngineSettings } from "../src/engine-settings.js";
import { normalizedSettings } from "../src/gallery-core.js";
import { parseXiria } from "../src/image-metadata.js";
import { buildImageInfoApplyPlan } from "../src/image-info-apply.js";
import { imageToImageBlockers, imageToImageRequestBody } from "../src/image-to-image.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "../src/sampling-options.js";

const capabilities = JSON.parse(readFileSync(new URL("../src/xyz-capabilities.json", import.meta.url), "utf8"));

test("the official workflow is the default", () => {
  assert.deepEqual({ ...MOSS_DEFAULTS, size: { ...MOSS_DEFAULTS.size } }, {
    steps: 50, cfg: 3, sampler: "euler", scheduler: "sgm_uniform", negative: "", size: { width: 1024, height: 1024 },
  });
  assert.deepEqual(normalizeMossSampling(null), { shift: 5, uncondSkipLayers: [22], uncondSkipStart: 0, uncondSkipEnd: 1 });
  assert.deepEqual(mossSamplingPayload(null), { version: 1, shift: 5, uncond_skip_layers: [22], uncond_skip_start: 0, uncond_skip_end: 1 });
  assert.equal(MOSS_DIT_DEPTH, 42);
});

test("the sampler vocabulary matches the shared capability contract and the shared lists", () => {
  assert.deepEqual(capabilities.engines.MOSS.samplers, [...MOSS_SAMPLERS]);
  assert.deepEqual(capabilities.engines.MOSS.schedulers, [...MOSS_SCHEDULERS]);
  assert.deepEqual(capabilities.engines.MOSS.guidance, ["none"]);
  assert.equal(capabilities.engines.MOSS.max_edge, 4096);
  // Every MOSS name is one the shared selects already know, so a Gallery card never carries an
  // unknown label.
  for (const name of MOSS_SAMPLERS) assert.ok(SAMPLER_NAMES.includes(name), name);
  for (const name of MOSS_SCHEDULERS) assert.ok(SCHEDULER_NAMES.includes(name), name);
  assert.equal(mossSamplerFor("dpmpp_2m"), "euler");
  assert.equal(mossSamplerFor("heun"), "heun");
  assert.equal(mossSchedulerFor("align_your_steps"), "sgm_uniform");
  assert.equal(mossSchedulerFor("kl_optimal"), "kl_optimal");
});

test("skip-layer text is parsed strictly and reported, never silently trimmed", () => {
  assert.deepEqual(parseSkipLayers(""), { layers: [], error: "" });
  assert.deepEqual(parseSkipLayers(" 22, 24，30 、 7 "), { layers: [22, 24, 30, 7], error: "" });
  assert.match(parseSkipLayers("22, x").error, /不是区块编号/);
  assert.match(parseSkipLayers("42").error, /0～41/);
  assert.match(parseSkipLayers("-1").error, /不是区块编号/);
  assert.match(parseSkipLayers("3, 3").error, /重复/);
  assert.equal(formatSkipLayers([22, 24]), "22, 24");
});

test("a stored block is coerced to what the server accepts", () => {
  assert.deepEqual(normalizeMossSampling({ shift: 1000, uncondSkipLayers: [22, 22, 41, 42, -1, 1.5], uncondSkipStart: 0.8, uncondSkipEnd: 0.2 }),
    { shift: 100, uncondSkipLayers: [22, 41], uncondSkipStart: 0, uncondSkipEnd: 1 });
  assert.equal(normalizeMossSampling({ shift: 0 }).shift, 0.01);
  assert.equal(normalizeMossSampling({ shift: "abc" }).shift, 5);
  assert.deepEqual(normalizeMossSampling({ uncondSkipLayers: [] }).uncondSkipLayers, []);
  const round = mossSamplingFromPayload(mossSamplingPayload({ shift: 3.25, uncondSkipLayers: [10, 20], uncondSkipStart: 0.1, uncondSkipEnd: 0.6 }));
  assert.deepEqual(round, { shift: 3.25, uncondSkipLayers: [10, 20], uncondSkipStart: 0.1, uncondSkipEnd: 0.6 });
  // Standard SLG is never sent: the wire block has no such field.
  assert.equal("standard_slg" in mossSamplingPayload(null), false);
});

test("only a MOSS request carries the block", () => {
  assert.deepEqual(mossRequestFields("MOSS", null), { moss: mossSamplingPayload(null) });
  for (const engine of ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]) assert.deepEqual(mossRequestFields(engine, null), {});
  assert.match(describeMossSampling(null, 3), /负向分支跳过区块 22/);
  assert.match(describeMossSampling(null, 1), /CFG 为 1/);
  assert.match(describeMossSampling({ uncondSkipLayers: [] }, 3), /不跳层/);
});

test("engine records: MOSS is the seventh engine and the other six are unchanged", () => {
  assert.deepEqual(SETTINGS_ENGINES, ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2", "MOSS"]);
  assert.equal(isSplitEngine("MOSS"), true);
  const moss = engineSettingsDefaults("MOSS");
  assert.equal(moss.steps, 50);
  assert.equal(moss.cfg, 3);
  assert.equal(moss.negative, "");
  assert.equal(moss.scheduler, "sgm_uniform");
  assert.deepEqual(moss.mossSampling, normalizeMossSampling(null));
  for (const engine of ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]) {
    assert.equal("mossSampling" in engineSettingsDefaults(engine), false, engine);
    assert.equal("mossSampling" in normalizeEngineSettings(engine, { mossSampling: { shift: 2 } }), false, engine);
  }
  const stored = normalizeEngineSettings("MOSS", { sampler: "dpmpp_2m", scheduler: "karras", guidance: "pag", mossSampling: { shift: 2.5 } });
  assert.equal(stored.sampler, "euler");
  assert.equal(stored.scheduler, "karras");
  assert.equal(stored.guidance, "none");
  assert.equal(stored.mossSampling.shift, 2.5);
  for (const guidance of ["pag", "cfg_zero_star"]) assert.equal(engineGuidance("MOSS", guidance), "none");
});

test("a Gallery card keeps a MOSS block only on a MOSS card", () => {
  const card = normalizedSettings({ model: "MOSS", sampler: "dpmpp_2m", scheduler: "beta", guidance: "pag", mossSampling: { shift: 7, uncondSkipLayers: [3] } });
  assert.equal(card.model, "MOSS");
  assert.equal(card.sampler, "euler");
  assert.equal(card.scheduler, "beta");
  assert.equal(card.guidance, "none");
  assert.deepEqual(card.mossSampling, { shift: 7, uncondSkipLayers: [3], uncondSkipStart: 0, uncondSkipEnd: 1 });
  // A card silent about its scheduler takes the fallback's, as every engine's card does; an unknown
  // one becomes MOSS's default.
  const fresh = normalizedSettings({ model: "MOSS" });
  assert.equal(fresh.scheduler, "karras");
  assert.equal(normalizedSettings({ model: "MOSS", scheduler: "align_your_steps" }).scheduler, "sgm_uniform");
  assert.deepEqual(fresh.mossSampling, normalizeMossSampling(null));
  const krea = normalizedSettings({ model: "Krea2", mossSampling: { shift: 7 } });
  assert.equal("mossSampling" in krea, false);
});

const mossRecord = (overrides = {}) => ({
  job_id: "job-1",
  engine: "MOSS",
  checkpoint: null,
  width: 1024,
  height: 1024,
  output_width: 1024,
  output_height: 1024,
  steps: 50,
  cfg: 3,
  sampler: "euler",
  scheduler: "sgm_uniform",
  seed: "42",
  diffusion_model: "moss_image_sft_int8_convrot_attn.safetensors",
  text_encoder: "qwen3.5_9b_int8_convrot.safetensors",
  vae: "flux2-vae.safetensors",
  flow_shift: 4.5,
  moss: { version: 1, shift: 4.5, uncond_skip_layers: [22, 24], uncond_skip_start: 0, uncond_skip_end: 0.5, variant: "sft", variant_fingerprint: "b2f3abd8", qkv_layout: "fused", attention: "sdpa" },
  loras: [],
  ...overrides,
});

test("the image reader shows a MOSS picture's own block", () => {
  const info = parseXiria(mossRecord(), { prompt: "a lighthouse", negative_prompt: "" });
  const labels = Object.fromEntries(info.parameters.map((item) => [item.label, item.value]));
  assert.equal(String(labels.Shift), "4.5");
  assert.equal(labels["Uncond skip layers"], "22, 24 (0–0.5)");
  assert.equal(labels["MOSS variant"], "sft");
  assert.deepEqual(info.mossSampling, { version: 1, shift: 4.5, uncond_skip_layers: [22, 24], uncond_skip_start: 0, uncond_skip_end: 0.5 });
  // An unknown block version is shown raw nowhere and applied never.
  assert.equal(parseXiria(mossRecord({ moss: { version: 2, shift: 9 } })).mossSampling, null);
  assert.equal(parseXiria(mossRecord({ engine: "Krea2" })).mossSampling, null);
});

test("applying a MOSS picture writes its block on a MOSS text-to-image target only", () => {
  const info = { status: "ok", source: "xiria", ...parseXiria(mossRecord(), { prompt: "a lighthouse" }) };
  const onMoss = buildImageInfoApplyPlan(info, ["sampling"], "t2i", { engine: "MOSS" });
  assert.deepEqual(onMoss.overlay.mossSampling, { shift: 4.5, uncondSkipLayers: [22, 24], uncondSkipStart: 0, uncondSkipEnd: 0.5 });
  const onKrea = buildImageInfoApplyPlan(info, ["sampling"], "t2i", { engine: "Krea2" });
  assert.equal("mossSampling" in onKrea.overlay, false);
  assert.ok(onKrea.skipped.some((item) => item.includes("MOSS 采样设置")));
  const onImagePage = buildImageInfoApplyPlan(info, ["sampling"], "i2i", { engine: "MOSS" });
  assert.equal("mossSampling" in onImagePage.imageToImage, false);
});

test("image-to-image refuses MOSS before anything else, and the other engines are untouched", () => {
  const settings = { positive: "a lighthouse", mode: "img2img" };
  const source = { dataUrl: "data:image/png;base64,AAAA", width: 1024, height: 1024 };
  assert.match(imageToImageBlockers({ source: null, settings, engine: "MOSS", imageToImageAvailable: false }), /MOSS 本版本暂不支持图生图/);
  assert.match(imageToImageBlockers({ source, settings, engine: "MOSS", imageToImageAvailable: false }), /MOSS 本版本暂不支持图生图/);
  assert.equal(imageToImageBlockers({ source: null, settings, engine: "Krea2" }), "请先选择一张来源图片");
  // Were a MOSS request ever built, it would mount components, not a checkpoint.
  const body = imageToImageRequestBody({ engine: "MOSS", diffusionModel: "d", textEncoder: "t", vae: "v", checkpoint: "c", source, settings });
  assert.deepEqual([body.diffusion_model, body.text_encoder, body.vae, body.checkpoint], ["d", "t", "v", undefined]);
});

test("a MOSS Hires card offers only the samplers and schedulers MOSS runs", () => {
  // The server refuses any other override on a MOSS resample unit, so the card must not offer one.
  const app = readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const panel = app.slice(app.indexOf("<ResampleChainPanel"), app.indexOf("baseSize={size}", app.indexOf("<ResampleChainPanel")));
  assert.match(panel, /samplers=\{model === "Anima" \? ANIMA_SAMPLERS : isMoss \? MOSS_SAMPLERS : samplerNames\}/);
  assert.match(panel, /schedulers=\{model === "Anima" \? ANIMA_SCHEDULERS : isMoss \? MOSS_SCHEDULERS : schedulerNames\}/);
});
