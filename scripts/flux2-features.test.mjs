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

test("the Flux2 entry point is open on the main page and mounts three components", async () => {
  const app = await readSource("src/App.jsx");

  assert.match(app, /\{ name: "Flux2", detail: "FLUX\.2 大模型引导", ready: true \}/);
  // "即将支持" is the disabled-engine caption; Flux2 must no longer carry it.
  assert.doesNotMatch(app, /\{ name: "Flux2",[^}]*ready: false \}/);
  assert.match(app, /const SPLIT_MODEL_ENGINES = \["Anima", "Flux", "Flux2", "Krea2"\]/);
  assert.match(app, /const READY_ENGINES = \["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"\]/);

  const picker = sourceBetween(app, "            {isSplitModel ? <div className={`checkpoint-picker split-model-picker", "            {isSplitModel && !engineAllowsLora");
  // The second encoder row stays a FLUX.1 slot: FLUX.2's whole conditioning is one language model.
  assert.match(picker, /\.\.\.\(isFlux \? \[\["textEncoder2", "文本编码器 · T5-XXL"/);
  assert.doesNotMatch(picker, /isFlux2 \? \[\["textEncoder2"/);
});

test("a Flux2 generation is guidance distilled: no negative branch, no enhancement, no preview", async () => {
  const app = await readSource("src/App.jsx");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");

  assert.match(generate, /const flux2Generation = model === "Flux2"/);
  assert.match(generate, /const distilledGeneration = DISTILLED_GUIDANCE_ENGINES\.includes\(model\)/);
  // Flux2 mounts the three-component payload, not the four-component one.
  assert.match(generate, /: nativeGeneration \? \{ diffusion_model: diffusionModel, text_encoder: textEncoder, vae \} : \{ checkpoint \}/);
  assert.match(generate, /negative_prompt: distilledGeneration \? "" : negative\.trim\(\)/);
  assert.match(generate, /guidance: distilledGeneration \? "none" : guidance/);

  const selectModel = sourceBetween(app, "  const selectModel = (nextModel)", "  const selectCheckpoint = (nextCheckpoint)");
  assert.match(selectModel, /\} else if \(nextModel === "Flux2"\) \{[\s\S]*?setGuidance\("none"\);[\s\S]*?setProcessPreview\(false\);/);

  // The typed negative prompt survives the switch; only the request drops it.
  assert.match(app, /const isDistilledGuidance = DISTILLED_GUIDANCE_ENGINES\.includes\(model\)/);
  assert.doesNotMatch(app, /isFlux2 && setNegative\(""\)/);
});

test("the image-to-image request mounts one Flux2 encoder and drops the negative prompt", () => {
  const source = { dataUrl: "data:image/png;base64,AAAA", name: "a.png", width: 1024, height: 1024 };
  const settings = { positive: "a lantern", negative: "blurry", steps: 20, cfg: 4.0, denoise: 0.6 };
  const body = imageToImageRequestBody({
    engine: "Flux2",
    checkpoint: "",
    diffusionModel: "flux2-dev.safetensors",
    textEncoder: "flux2-text-encoder.safetensors",
    vae: "flux2-vae.safetensors",
    source,
    settings,
    seed: "7",
  });

  assert.equal(body.engine, "Flux2");
  assert.equal(body.diffusion_model, "flux2-dev.safetensors");
  assert.equal(body.text_encoder, "flux2-text-encoder.safetensors");
  assert.equal(body.text_encoder_2, undefined);
  assert.equal(body.vae, "flux2-vae.safetensors");
  assert.equal(body.checkpoint, undefined);
  assert.equal(body.negative_prompt, "");
  assert.equal(body.preview_enabled, false);
  assert.equal(body.guidance, "none");
  // USDU tiling stays Anima's.
  assert.equal(body.hires.execution_mode, "full_frame");
});

test("a Flux2 request drops per-unit ADetailer negative prompts without erasing them", () => {
  const stage = {
    enabled: true,
    units: [{ enabled: true, detector: "face_yolov8n.pt", prompt: "a face", negativePrompt: "blurry" }],
  };
  assert.equal(adetailerPayload(stage, "Flux2").units[0].negative_prompt, "");
  assert.equal(adetailerPayload(stage, "Flux2").units[0].prompt, "a face");
  assert.equal(adetailerPayload(stage, "Anima").units[0].negative_prompt, "blurry");
  assert.deepEqual(DISTILLED_GUIDANCE_ENGINES, ["Flux", "Flux2"]);
});

test("Flux2 refinement steps are counted like the other native engines", () => {
  assert.deepEqual(NATIVE_STEP_ENGINES, ["Anima", "Flux", "Flux2", "Krea2"]);
  const unit = { useSteps: true, steps: 20, denoise: 0.2, detector: "face_yolov8n.pt" };
  assert.equal(adetailerUnitSteps(unit, 30, "Flux2"), 20);
  assert.equal(adetailerUnitSteps(unit, 30, "SD"), 4);
  assert.equal(hiresEffectiveSteps({ steps: 12, denoise: 0.05 }, "Flux2"), 12);
});

test("Flux2 has its own LoRA scope without invalidating libraries saved before it existed", () => {
  assert.deepEqual(READY_LORA_ENGINES, ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]);
  assert.deepEqual(emptyMountedLoraMap().byEngine.Flux2, []);

  // A v2 library written when Flux was the newest engine has no `Flux2` key. Treating that as a
  // corrupt file would stop autosave and strand the user's existing mounts.
  const legacy = normalizeMountedLoraMap({
    schemaVersion: 2,
    byEngine: { SD: [{ value: "a.safetensors", weight: 1 }], iL: [], Anima: [], Flux: [] },
  }, { activeEngine: "SD" });
  assert.equal(legacy.fatal, false);
  assert.deepEqual(legacy.container.byEngine.Flux2, []);
  assert.equal(legacy.container.byEngine.SD.length, 1);
});

test("a shared LoRA root saved before Flux2 existed expands rather than excluding it", () => {
  assert.deepEqual(SHARED_LORA_ENGINES, ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]);
  // The stored list is exactly "every engine there was", so it means "not configured" and has to
  // grow with the vocabulary instead of silently hiding the folder from the new engine. Every
  // roster this field has ever shipped with reads that way, including the one Krea 2 replaced.
  assert.deepEqual(normalizeSharedEngines(["SD", "iL", "Anima", "Flux"]), [...SHARED_LORA_ENGINES]);
  assert.deepEqual(normalizeSharedEngines(["SD", "iL", "Anima", "Flux", "Flux2"]), [...SHARED_LORA_ENGINES]);
  // A deliberately narrowed list is still respected.
  assert.deepEqual(normalizeSharedEngines(["SD"]), ["SD"]);
});

test("the control plane lists Flux2 components and its own LoRA root", async () => {
  const vite = await readSource("vite.config.js");
  assert.match(vite, /import \{ discoverFlux2Models \} from "\.\/scripts\/flux2-models\.mjs"/);
  assert.match(vite, /if \(!isLoraRequest && engine === "Flux2"\)/);
  assert.match(vite, /engine: "Flux2",\s+model_type: "split"/);
  // Three assets, and no fourth: a `text_encoder_2` entry here would offer a slot the backend
  // refuses.
  const branch = sourceBetween(vite, 'if (!isLoraRequest && engine === "Flux2") {', "    const directory = await getConfiguredDirectory(");
  assert.doesNotMatch(branch, /text_encoder_2/);
  assert.match(vite, /const loraEnginePathKeys = \{ SD: "sd", iL: "illustrious", Anima: "anima", Flux: "flux", Flux2: "flux2", Krea2: "krea2" \}/);

  const modelPaths = await readSource("scripts/model-paths.mjs");
  assert.match(modelPaths, /flux2: "models\/loras\/flux2"/);
});

test("the image-to-image page opens its split picker and negative-prompt lock for Flux2", async () => {
  const page = await readSource("src/ImageToImagePage.jsx");
  assert.match(page, /\["Anima", "Flux", "Flux2", "Krea2"\]\.includes\(engine\.name\)/);
  assert.match(page, /const distilledPageLabel = engine\.name === "Flux2" \? "FLUX\.2" : "FLUX\.1"/);
  assert.match(page, /disabled=\{running \|\| !pageAllowsNegativePrompt\}/);
});
