import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  DISTILLED_GUIDANCE_ENGINES,
  adetailerPayload,
  adetailerStageIssue,
  adetailerUnitSteps,
} from "../src/adetailer-units.js";
import { hiresEffectiveSteps } from "../src/hires-settings.js";
import { imageToImageRequestBody } from "../src/image-to-image.js";
import {
  READY_LORA_ENGINES,
  emptyMountedLoraMap,
  engineScopeKey,
  normalizeMountedLoraMap,
} from "../src/lora-model-scope.js";

const readSource = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

function sourceBetween(source, start, end) {
  const startIndex = source.indexOf(start);
  assert.notEqual(startIndex, -1, `missing source anchor: ${start}`);
  const endIndex = source.indexOf(end, startIndex + start.length);
  assert.notEqual(endIndex, -1, `missing source anchor: ${end}`);
  return source.slice(startIndex, endIndex);
}

test("the Flux entry point is open on the main page and mounts four components", async () => {
  const app = await readSource("src/App.jsx");

  assert.match(app, /\{ name: "Flux", detail: "FLUX\.1 蒸馏引导", ready: true \}/);
  // "即将支持" is the disabled-engine caption; Flux must no longer carry it in the engine table.
  assert.doesNotMatch(app, /\{ name: "Flux",[^}]*ready: false \}/);
  assert.match(app, /const SPLIT_MODEL_ENGINES = \["Anima", "Flux", "Flux2", "Krea2"\]/);
  assert.match(app, /const READY_ENGINES = \["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"\]/);

  const picker = sourceBetween(app, "            {isSplitModel ? <div className={`checkpoint-picker split-model-picker", "            {isSplitModel && !engineAllowsLora");
  assert.match(picker, /\["textEncoder", isFlux \? "文本编码器 · CLIP-L" : isFlux2 \|\| isKrea2 \? "文本编码器 · 大语言模型" : "文本编码器"/);
  assert.match(picker, /\.\.\.\(isFlux \? \[\["textEncoder2", "文本编码器 · T5-XXL"/);
  // The second encoder is a Flux slot, not a fourth Anima component.
  assert.doesNotMatch(picker, /\["textEncoder2", "文本编码器 · T5-XXL", textEncoder2, textEncoders2, textEncoderDirectory, textEncoder2Missing\],\n/);
});

test("a Flux generation is guidance distilled: no negative branch, no enhancement, no preview", async () => {
  const app = await readSource("src/App.jsx");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");

  assert.match(generate, /const fluxGeneration = model === "Flux"/);
  assert.match(generate, /const nativeGeneration = animaGeneration \|\| fluxGeneration \|\| flux2Generation/);
  assert.match(generate, /\.\.\.\(fluxGeneration\s*\n\s*\? \{ diffusion_model: diffusionModel, text_encoder: textEncoder, text_encoder_2: textEncoder2, vae \}/);
  assert.match(generate, /negative_prompt: distilledGeneration \? "" : negative\.trim\(\)/);
  assert.match(generate, /guidance: distilledGeneration \? "none" : guidance/);
  assert.match(generate, /preview_enabled: nativeGeneration \? false : processPreview/);

  const selectModel = sourceBetween(app, "  const selectModel = (nextModel)", "  const selectCheckpoint = (nextCheckpoint)");
  assert.match(selectModel, /\} else if \(nextModel === "Flux"\) \{[\s\S]*?setGuidance\("none"\);[\s\S]*?setProcessPreview\(false\);/);

  // The typed negative prompt survives the switch; only the request drops it.
  assert.match(app, /const engineAllowsNegativePrompt = !isDistilledGuidance/);
  assert.match(app, /disabled=\{!engineAllowsNegativePrompt\}/);
  assert.doesNotMatch(app, /isFlux && setNegative\(""\)/);
});

test("the image-to-image request carries both Flux encoders and drops the negative prompt", () => {
  const source = { dataUrl: "data:image/png;base64,AAAA", name: "a.png", width: 1024, height: 1024 };
  const settings = { positive: "a lantern", negative: "blurry", steps: 20, cfg: 3.5, denoise: 0.6 };
  const body = imageToImageRequestBody({
    engine: "Flux",
    checkpoint: "",
    diffusionModel: "flux1-dev.safetensors",
    textEncoder: "clip_l.safetensors",
    textEncoder2: "t5xxl_fp16.safetensors",
    vae: "ae.safetensors",
    source,
    settings,
    seed: "7",
  });

  assert.equal(body.engine, "Flux");
  assert.equal(body.diffusion_model, "flux1-dev.safetensors");
  assert.equal(body.text_encoder, "clip_l.safetensors");
  assert.equal(body.text_encoder_2, "t5xxl_fp16.safetensors");
  assert.equal(body.vae, "ae.safetensors");
  assert.equal(body.checkpoint, undefined);
  assert.equal(body.negative_prompt, "");
  assert.equal(body.preview_enabled, false);
  assert.equal(body.guidance, "none");
  // USDU tiling stays Anima's; a Flux request must never ask for it.
  assert.equal(body.hires.execution_mode, "full_frame");

  const anima = imageToImageRequestBody({
    engine: "Anima",
    diffusionModel: "anima.safetensors",
    textEncoder: "qwen.safetensors",
    vae: "qwen-vae.safetensors",
    source,
    settings,
    seed: "7",
  });
  assert.equal(anima.text_encoder_2, undefined);
  assert.equal(anima.negative_prompt, "blurry");
});

test("a Flux request drops per-unit ADetailer negative prompts without erasing them", () => {
  const stage = {
    enabled: true,
    units: [{ enabled: true, detector: "face_yolov8n.pt", prompt: "a face", negativePrompt: "blurry" }],
  };
  assert.equal(adetailerPayload(stage, "Flux").units[0].negative_prompt, "");
  assert.equal(adetailerPayload(stage, "Anima").units[0].negative_prompt, "blurry");
  assert.equal(adetailerPayload(stage, "SD").units[0].negative_prompt, "blurry");
  // The positive prompt is untouched: only the branch that cannot be encoded is dropped.
  assert.equal(adetailerPayload(stage, "Flux").units[0].prompt, "a face");
  assert.deepEqual(DISTILLED_GUIDANCE_ENGINES, ["Flux", "Flux2"]);
});

test("Flux refinement steps are counted like Anima's, not like Diffusers'", () => {
  const unit = { useSteps: true, steps: 20, denoise: 0.2, detector: "face_yolov8n.pt" };
  assert.equal(adetailerUnitSteps(unit, 30, "Flux"), 20);
  assert.equal(adetailerUnitSteps(unit, 30, "Anima"), 20);
  assert.equal(adetailerUnitSteps(unit, 30, "SD"), 4);

  const hires = { steps: 12, denoise: 0.05 };
  assert.equal(hiresEffectiveSteps(hires, "Flux"), 12);
  assert.equal(hiresEffectiveSteps(hires, "SD"), 0);

  // A low-denoise unit that Diffusers would round to zero steps is a legal Flux configuration, so
  // the run blocker must not refuse it.
  const faint = { enabled: true, useSteps: true, steps: 12, denoise: 0.05, detector: "face_yolov8n.pt" };
  const stage = { enabled: true, units: [faint] };
  assert.equal(adetailerStageIssue(stage, 30, () => true, "Flux"), "");
  assert.equal(adetailerStageIssue(stage, 30, () => true, "Anima"), "");
  assert.notEqual(adetailerStageIssue(stage, 30, () => true, "SD"), "");
});

test("Flux has its own LoRA scope without invalidating libraries saved before it existed", () => {
  assert.deepEqual(READY_LORA_ENGINES, ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]);
  assert.equal(engineScopeKey("Flux"), "Flux");
  assert.deepEqual(emptyMountedLoraMap().byEngine.Flux, []);

  // A v2 library written before the Flux scope shipped has no `Flux` key. Treating that as a
  // corrupt file would stop autosave and strand the user's existing mounts.
  const legacy = normalizeMountedLoraMap({
    schemaVersion: 2,
    byEngine: { SD: [{ value: "a.safetensors", weight: 1 }], iL: [], Anima: [] },
  }, { activeEngine: "SD" });
  assert.equal(legacy.fatal, false);
  assert.deepEqual(legacy.container.byEngine.Flux, []);
  assert.equal(legacy.container.byEngine.SD.length, 1);

  // A malformed list for an established engine is still a corrupt library.
  assert.equal(normalizeMountedLoraMap({
    schemaVersion: 2,
    byEngine: { SD: "not a list", iL: [], Anima: [] },
  }, { activeEngine: "SD" }).fatal, true);
});

test("the control plane lists Flux components and its own LoRA root", async () => {
  const vite = await readSource("vite.config.js");
  assert.match(vite, /import \{ discoverFluxModels \} from "\.\/scripts\/flux-models\.mjs"/);
  assert.match(vite, /if \(!isLoraRequest && engine === "Flux"\)/);
  assert.match(vite, /text_encoder_2: \{ directory: relativeDirectory\(textEncoderDirectory\), models: discovered\.text_encoder_2 \}/);
  assert.match(vite, /const nativeEngines = new Set\(\["Anima", "Flux", "Flux2", "Krea2"\]\)/);

  const modelPaths = await readSource("scripts/model-paths.mjs");
  assert.match(modelPaths, /flux: "models\/loras\/flux"/);
});
