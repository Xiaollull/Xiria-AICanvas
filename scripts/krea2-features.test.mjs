import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  DISTILLED_GUIDANCE_ENGINES,
  NATIVE_STEP_ENGINES,
  adetailerPayload,
  adetailerUnitSteps,
} from "../src/adetailer-units.js";
import { hiresEffectiveSteps } from "../src/hires-settings.js";
import { imageToImageRequestBody } from "../src/image-to-image.js";
import { READY_LORA_ENGINES, emptyMountedLoraMap, normalizeMountedLoraMap } from "../src/lora-model-scope.js";
import { SHARED_LORA_ENGINES, normalizeSharedEngines } from "../src/shared-model-refs.js";

const readSource = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

function sourceBetween(source, start, end) {
  const startIndex = source.indexOf(start);
  assert.notEqual(startIndex, -1, `missing source anchor: ${start}`);
  const endIndex = source.indexOf(end, startIndex + start.length);
  assert.notEqual(endIndex, -1, `missing source anchor: ${end}`);
  return source.slice(startIndex, endIndex);
}

test("the Krea2 entry point is open on the main page and mounts three components", async () => {
  const app = await readSource("src/App.jsx");

  // "即将支持" is the disabled-engine caption; Krea2 must no longer carry it.
  assert.doesNotMatch(app, /\{ name: "Krea2",[^}]*ready: false \}/);
  assert.doesNotMatch(app, /\{ name: "Krea2", detail: "即将支持"/);
  assert.match(app, /\{ name: "Krea2", detail: "[^"]+", ready: true \}/);
  assert.match(app, /const SPLIT_MODEL_ENGINES = \["Anima", "Flux", "Flux2", "Krea2"\]/);
  assert.match(app, /const READY_ENGINES = \["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"\]/);
  assert.match(app, /const KREA2_SAMPLERS = samplerNames/);
  assert.match(app, /const KREA2_SCHEDULERS = schedulerNames/);

  const picker = sourceBetween(app, "            {isSplitModel ? <div className={`checkpoint-picker split-model-picker", "            {isSplitModel && !engineAllowsLora");
  // Krea 2's whole conditioning is one language model, so it shares FLUX.2's label and never
  // opens the second encoder row.
  assert.match(picker, /isFlux2 \|\| isKrea2 \? "文本编码器 · 大语言模型"/);
  assert.doesNotMatch(picker, /isKrea2 \? \[\["textEncoder2"/);
});

test("a Krea2 generation keeps its unconditional branch, unlike either Flux", async () => {
  const app = await readSource("src/App.jsx");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");

  assert.match(generate, /const krea2Generation = model === "Krea2"/);
  // Native by sampler vocabulary and by payload shape, but not by the distilled rule.
  assert.match(generate, /nativeGeneration = animaGeneration \|\| fluxGeneration \|\| flux2Generation \|\| krea2Generation/);
  assert.match(generate, /: nativeGeneration \? \{ diffusion_model: diffusionModel, text_encoder: textEncoder, vae \} : \{ checkpoint \}/);
  assert.deepEqual(DISTILLED_GUIDANCE_ENGINES, ["Flux", "Flux2"]);

  const selectModel = sourceBetween(app, "  const selectModel = (nextModel)", "  const selectCheckpoint = (nextCheckpoint)");
  // PAG has no attention override here, so it is cleared; CFG-Zero* survives the switch because
  // Krea 2 has a real unconditional branch to rescale against.
  assert.match(selectModel, /\} else if \(nextModel === "Krea2"\) \{[\s\S]*?currentGuidance === "pag" \? "none" : currentGuidance[\s\S]*?setProcessPreview\(false\);/);
  assert.doesNotMatch(selectModel, /nextModel === "Krea2"\) \{[\s\S]*?setGuidance\("none"\);/);
});

test("the image-to-image request mounts one Krea2 encoder and keeps the negative prompt", () => {
  const source = { dataUrl: "data:image/png;base64,AAAA", name: "a.png", width: 1024, height: 1024 };
  const settings = { positive: "a lantern", negative: "blurry", steps: 20, cfg: 4.0, denoise: 0.6 };
  const body = imageToImageRequestBody({
    engine: "Krea2",
    checkpoint: "",
    diffusionModel: "krea2_raw_bf16.safetensors",
    textEncoder: "qwen3vl_4b_bf16.safetensors",
    vae: "qwen_image_vae.safetensors",
    source,
    settings,
    seed: "7",
  });

  assert.equal(body.engine, "Krea2");
  assert.equal(body.diffusion_model, "krea2_raw_bf16.safetensors");
  assert.equal(body.text_encoder, "qwen3vl_4b_bf16.safetensors");
  assert.equal(body.text_encoder_2, undefined);
  assert.equal(body.vae, "qwen_image_vae.safetensors");
  assert.equal(body.checkpoint, undefined);
  // This is the difference from both Flux engines: the negative prompt reaches the wire.
  assert.equal(body.negative_prompt, "blurry");
  assert.equal(body.preview_enabled, false);
  // USDU tiling stays Anima's.
  assert.equal(body.hires.execution_mode, "full_frame");
});

test("a Krea2 request keeps per-unit ADetailer negative prompts", () => {
  const stage = {
    enabled: true,
    units: [{ enabled: true, detector: "face_yolov8n.pt", prompt: "a face", negativePrompt: "blurry" }],
  };
  assert.equal(adetailerPayload(stage, "Krea2").units[0].negative_prompt, "blurry");
  assert.equal(adetailerPayload(stage, "Flux2").units[0].negative_prompt, "");
});

test("Krea2 refinement steps are counted like the other native engines", () => {
  assert.deepEqual(NATIVE_STEP_ENGINES, ["Anima", "Flux", "Flux2", "Krea2"]);
  const unit = { useSteps: true, steps: 20, denoise: 0.2, detector: "face_yolov8n.pt" };
  assert.equal(adetailerUnitSteps(unit, 30, "Krea2"), 20);
  assert.equal(adetailerUnitSteps(unit, 30, "SD"), 4);
  assert.equal(hiresEffectiveSteps({ steps: 12, denoise: 0.05 }, "Krea2"), 12);
});

test("Krea2 has its own LoRA scope without invalidating libraries saved before it existed", () => {
  assert.deepEqual(READY_LORA_ENGINES, ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]);
  assert.deepEqual(emptyMountedLoraMap().byEngine.Krea2, []);

  // A v2 library written when Flux2 was the newest engine has no `Krea2` key. Treating that as a
  // corrupt file would stop autosave and strand the user's existing mounts.
  const legacy = normalizeMountedLoraMap({
    schemaVersion: 2,
    byEngine: { SD: [{ value: "a.safetensors", weight: 1 }], iL: [], Anima: [], Flux: [], Flux2: [] },
  }, { activeEngine: "SD" });
  assert.equal(legacy.fatal, false);
  assert.deepEqual(legacy.container.byEngine.Krea2, []);
  assert.equal(legacy.container.byEngine.SD.length, 1);
});

test("a shared LoRA root saved before Krea2 existed expands rather than excluding it", () => {
  assert.deepEqual(SHARED_LORA_ENGINES, ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]);
  assert.deepEqual(normalizeSharedEngines(["SD", "iL", "Anima", "Flux", "Flux2"]), [...SHARED_LORA_ENGINES]);
  // A deliberately narrowed list is still respected.
  assert.deepEqual(normalizeSharedEngines(["SD", "Krea2"]), ["SD", "Krea2"]);
});

test("the control plane lists Krea2 components and its own LoRA root", async () => {
  const vite = await readSource("vite.config.js");
  assert.match(vite, /import \{ discoverKrea2Models \} from "\.\/scripts\/krea2-models\.mjs"/);
  assert.match(vite, /if \(!isLoraRequest && engine === "Krea2"\)/);
  assert.match(vite, /engine: "Krea2",\s+model_type: "split"/);
  // Three assets, and no fourth: a `text_encoder_2` entry here would offer a slot the backend
  // refuses.
  const branch = sourceBetween(vite, 'if (!isLoraRequest && engine === "Krea2") {', "    const directory = await getConfiguredDirectory(");
  assert.doesNotMatch(branch, /text_encoder_2/);
  // Qwen3-VL reads through the Qwen2.5 table, which is the resource Anima already bundles.
  assert.match(branch, /qwen_tokenizer:/);
  assert.match(vite, /const nativeEngines = new Set\(\["Anima", "Flux", "Flux2", "Krea2"\]\)/);
  assert.match(vite, /const loraEnginePathKeys = \{ SD: "sd", iL: "illustrious", Anima: "anima", Flux: "flux", Flux2: "flux2", Krea2: "krea2" \}/);

  const modelPaths = await readSource("scripts/model-paths.mjs");
  assert.match(modelPaths, /krea2: "models\/loras\/krea2"/);
});

test("the image-to-image page opens its split picker for Krea2 and leaves the negative prompt live", async () => {
  const page = await readSource("src/ImageToImagePage.jsx");
  assert.match(page, /\["Anima", "Flux", "Flux2", "Krea2"\]\.includes\(engine\.name\)/);
  // The negative-prompt lock is driven by the distilled list, which Krea 2 is not on.
  assert.match(page, /const pageAllowsNegativePrompt = !DISTILLED_GUIDANCE_ENGINES\.includes\(engine\.name\)/);
});
