// P6 workspace contracts: CLIP skip, per-engine prompts, the canvas bound, output templates, LoRA
// folders, unambiguous model matching and recorded post-processing.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { clipSkipPayload, clipSkipRange, normalizeClipSkip } from "../src/clip-skip.js";
import {
  CANVAS_BOUND_PRESETS,
  DEFAULT_CANVAS_MAX_EDGE,
  canvasBoundConflicts,
  canvasEdgeIssue,
  effectiveCanvasMaxEdge,
  normalizeCanvasMaxEdge,
  snapCanvasMaxEdge,
} from "../src/canvas-limit.js";
import { DEFAULT_OUTPUT_TEMPLATE, normalizeStoredOutputTemplate, outputTemplatePayload } from "../src/output-template.js";
import {
  ENGINE_SETTINGS_SCHEMA_VERSION,
  SETTINGS_ENGINES,
  engineSettingsFor,
  normalizeEngineSettingsMap,
  withEngineSettings,
} from "../src/engine-settings.js";
import { imageToImageBlockers, imageToImageRequestBody, outputSize } from "../src/image-to-image.js";
import {
  interpretImageMetadata,
  matchModelName,
  modelDisplayName,
  parseA1111,
  parseLoraHashes,
  publicMatch,
  settleAmbiguousMatch,
  stripImageInfoMatches,
} from "../src/image-metadata.js";
import { buildImageInfoApplyPlan, imageInfoApplyFields, imageInfoClipSkip, loraUnappliedReason, matchLabel } from "../src/image-info-apply.js";
import { buildPostprocessChainPlan, imageInfoPostprocessCandidate, resolveDetectorName, resolveUpscalerName } from "../src/image-info-postprocess.js";
import { loraCategoryLocation, loraFileLabel, loraFolderView, normalizeLoraFolderPath } from "../src/lora-folders.js";
import { normalizedSettings } from "../src/gallery-core.js";
import { readPngMetadataChunks } from "./png-text-chunks.mjs";

// The reference infotext from README/BUG/00039-3576420342.png (A1111/Forge, Hires + three ADetailer units).
const REFERENCE_INFOTEXT = [
  "<lora:shou_ad_xl_half_v1:1>,<lora:matcha:0.4>,<lora:MomoSweetSpaceILL:0.4>,1girl, solo",
  "Negative prompt: bad hands",
  "Steps: 27, Sampler: Euler a, Schedule type: Automatic, CFG scale: 5, Seed: 3576420342, Size: 768x1280, Model: waiIllustriousSDXL_v140, Denoising strength: 0.25, Clip skip: 2, "
    + "ADetailer model: face_yolov8n.pt, ADetailer confidence: 0.3, ADetailer dilate erode: 4, ADetailer mask blur: 4, ADetailer denoising strength: 0.4, ADetailer inpaint only masked: True, ADetailer inpaint padding: 32, ADetailer ControlNet model: Passthrough, ADetailer ControlNet guidance start: 0.1, "
    + "ADetailer model 2nd: face_yolov8s.pt, ADetailer confidence 2nd: 0.3, ADetailer dilate erode 2nd: 4, ADetailer mask blur 2nd: 4, ADetailer denoising strength 2nd: 0.4, ADetailer inpaint only masked 2nd: True, ADetailer inpaint padding 2nd: 32, "
    + "ADetailer model 3rd: hand_yolov8n.pt, ADetailer confidence 3rd: 0.3, ADetailer dilate erode 3rd: 4, ADetailer mask blur 3rd: 4, ADetailer denoising strength 3rd: 0.1, ADetailer inpaint only masked 3rd: True, ADetailer inpaint padding 3rd: 32, "
    + "ADetailer version: 26.2.0, Hires Module 1: Use same choices, Hires CFG Scale: 5, Hires upscale: 2, Hires upscaler: R-ESRGAN 4x+ Anime6B, "
    + "Lora hashes: \"shou_ad_xl_half_v1: a0f6d73f81ef, matcha: cd00cd91342f, MomoSweetSpaceILL: 7cc957fa361c\", Version: f2.0.1v1.10.1-previous-669-gdfdcbab6",
].join("\n");

function catalogEntry(relative, engine, { origin = "local", value = relative, engines = [] } = {}) {
  const path = relative.split("\\").join("/").toLowerCase();
  const base = path.split("/").pop();
  return { origin, path, relative, display: modelDisplayName(relative), base, stem: base.replace(/\.[^.]+$/, ""), label: engine, value, engine, engines };
}

const LORA_CATALOG = [
  catalogEntry("style/Style_九V6.0/matcha.safetensors", "iL"),
  catalogEntry("style/Style_九V6.0/shou_ad_xl_half_v1.safetensors", "iL"),
  catalogEntry("style/Style_九V7.0/shou_ad_xl_half_v1.safetensors", "iL"),
  catalogEntry("MomoSweetSpaceILL.safetensors", "iL"),
  catalogEntry("style/Style_九V6.0/MomoSweetSpaceILL.safetensors", "iL"),
  catalogEntry("style/CUTE2712_Illustrious_v1.0.safetensors", "iL"),
  catalogEntry("style/matcha.safetensors", "SD"),
  catalogEntry("anime/matcha.safetensors", "", { origin: "shared", value: "shared:abc/anime/matcha.safetensors", engines: ["iL", "SD"] }),
];

test("CLIP skip ranges: SD 1–12 default 1, iL 2–12 default 2, none for native engines", () => {
  assert.deepEqual(clipSkipRange("SD"), { minimum: 1, maximum: 12, default: 1 });
  assert.deepEqual(clipSkipRange("iL"), { minimum: 2, maximum: 12, default: 2 });
  for (const engine of ["Anima", "Flux", "Flux2", "Krea2"]) {
    assert.equal(clipSkipRange(engine), null);
    assert.equal(normalizeClipSkip(engine, 3), null);
    assert.deepEqual(clipSkipPayload(engine, 3), {});
  }
  assert.equal(normalizeClipSkip("iL", 1), 2);
  assert.equal(normalizeClipSkip("iL", null), 2);
  assert.equal(normalizeClipSkip("SD", undefined), 1);
  assert.equal(normalizeClipSkip("SD", 99), 12);
  assert.equal(normalizeClipSkip("SD", true), 1);
  assert.deepEqual(clipSkipPayload("iL", null), { clip_skip: 2 });
  assert.deepEqual(clipSkipPayload("SD", 3), { clip_skip: 3 });
});

test("canvas bound: default 2048, presets, 64 steps, square bound over a rectangular canvas", () => {
  assert.equal(DEFAULT_CANVAS_MAX_EDGE, 2048);
  for (const preset of [1024, 2048, 3072, 4096, 6144, 8192]) assert.ok(CANVAS_BOUND_PRESETS.includes(preset));
  assert.deepEqual(normalizeCanvasMaxEdge(undefined), { value: 2048, warning: "" });
  assert.equal(normalizeCanvasMaxEdge(4096).value, 4096);
  for (const damaged of [512, 9000, 3000, "big", true, 2050]) {
    const result = normalizeCanvasMaxEdge(damaged);
    assert.equal(result.value, 2048);
    assert.match(result.warning, /已恢复默认/);
  }
  assert.equal(snapCanvasMaxEdge(3000), 3008);
  assert.equal(snapCanvasMaxEdge(100), 1024);
  assert.equal(snapCanvasMaxEdge(99999), 8192);
  assert.equal(canvasEdgeIssue({ width: 1024, height: 4096 }, { maxEdge: 4096, engine: "iL" }), "");
  assert.match(canvasEdgeIssue({ width: 1024, height: 4096 }, { maxEdge: 2048, engine: "iL" }), /超出当前画布上限 2048/);
  assert.match(canvasEdgeIssue({ width: 4160, height: 1024 }, { maxEdge: 8192, engine: "Anima" }), /不能超过 4096/);
  assert.match(canvasEdgeIssue({ width: 1000, height: 1024 }, { maxEdge: 2048 }), /64 的倍数/);
  assert.equal(effectiveCanvasMaxEdge(8192, "Krea2"), 4096);
  assert.equal(effectiveCanvasMaxEdge(8192, "SD"), 8192);
  assert.deepEqual(canvasBoundConflicts(2048, [{ label: "t2i", width: 2560, height: 1024 }, { label: "i2i", width: 1024, height: 1024 }]).map((item) => item.label), ["t2i"]);
});

test("output template: the default is sent as nothing, a custom one as trimmed text", () => {
  assert.equal(outputTemplatePayload(DEFAULT_OUTPUT_TEMPLATE), null);
  assert.equal(outputTemplatePayload(""), null);
  assert.equal(outputTemplatePayload("  Test/image  "), "Test/image");
  assert.deepEqual(normalizeStoredOutputTemplate(undefined), { value: DEFAULT_OUTPUT_TEMPLATE, warning: "" });
  assert.match(normalizeStoredOutputTemplate(42).warning, /恢复默认/);
});

test("engine settings schema 3: a schema-2 shared prompt is copied into all six engines once", () => {
  assert.equal(ENGINE_SETTINGS_SCHEMA_VERSION, 3);
  const legacyMap = { schemaVersion: 2, byEngine: { SD: { steps: 20 }, iL: { steps: 30, sampler: "euler" } } };
  const migrated = normalizeEngineSettingsMap(legacyMap, {
    legacyPrompts: { positive: "shared positive", negative: "shared negative", activePromptPresets: { positive: ["p1"], negative: [] } },
  });
  assert.equal(migrated.migrated, true);
  for (const engine of SETTINGS_ENGINES) {
    const record = migrated.container.byEngine[engine];
    assert.equal(record.positive, "shared positive", engine);
    assert.equal(record.negative, "shared negative", engine);
    assert.deepEqual(record.activePromptPresets, { positive: ["p1"], negative: [] });
  }
  assert.equal(migrated.container.byEngine.SD.steps, 20);
  assert.equal(migrated.container.byEngine.iL.clipSkip, 2);
  assert.equal(migrated.container.byEngine.SD.clipSkip, 1);
  assert.equal(migrated.container.byEngine.Anima.clipSkip, null);
  // Written back as schema 3, a second read does not copy the top-level prompt over the engines again.
  const edited = withEngineSettings(migrated.container, "iL", { ...engineSettingsFor(migrated.container, "iL"), positive: "only iL" });
  assert.equal(edited.schemaVersion, 3);
  const reread = normalizeEngineSettingsMap(JSON.parse(JSON.stringify(edited)), { legacyPrompts: { positive: "stale top level" } });
  assert.equal(reread.container.byEngine.iL.positive, "only iL");
  assert.equal(reread.container.byEngine.SD.positive, "shared positive");
  assert.equal(reread.migrated, false);
});

test("engine settings keep an image-page CLIP skip per engine and never leak one engine's prompt into another", () => {
  let map = normalizeEngineSettingsMap({ schemaVersion: 3, byEngine: {} }).container;
  map = withEngineSettings(map, "SD", { ...engineSettingsFor(map, "SD"), positive: "sd prompt", clipSkip: 2, imageToImage: { ...engineSettingsFor(map, "SD").imageToImage, clipSkip: 4 } });
  map = withEngineSettings(map, "Flux", { ...engineSettingsFor(map, "Flux"), positive: "flux prompt" });
  assert.equal(engineSettingsFor(map, "SD").positive, "sd prompt");
  assert.equal(engineSettingsFor(map, "SD").clipSkip, 2);
  assert.equal(engineSettingsFor(map, "SD").imageToImage.clipSkip, 4);
  assert.equal(engineSettingsFor(map, "Flux").positive, "flux prompt");
  assert.equal(engineSettingsFor(map, "Flux").imageToImage.clipSkip, null);
  assert.equal(engineSettingsFor(map, "iL").imageToImage.clipSkip, 2);
  // A cleared prompt is a value, not a missing field.
  map = withEngineSettings(map, "SD", { ...engineSettingsFor(map, "SD"), positive: "" });
  assert.equal(engineSettingsFor(map, "SD").positive, "");
});

test("image-to-image: the bound fits the source, the request carries CLIP skip and the template", () => {
  const source = { dataUrl: "data:image/png;base64,AA==", width: 3000, height: 2000, name: "a.png" };
  assert.deepEqual(outputSize(source, { sizeMode: "source" }), { width: 2048, height: 1344 });
  assert.deepEqual(outputSize(source, { sizeMode: "source" }, { maxEdge: 4096 }), { width: 3008, height: 1984 });
  const body = imageToImageRequestBody({ engine: "iL", checkpoint: "c.safetensors", source, settings: { positive: "x", clipSkip: 3, sizeMode: "source" }, seed: "1", canvasMaxEdge: 4096, outputTemplate: "Test/image" });
  assert.equal(body.clip_skip, 3);
  assert.equal(body.output_template, "Test/image");
  assert.equal(body.width, 3008);
  const native = imageToImageRequestBody({ engine: "Anima", diffusionModel: "d", textEncoder: "t", vae: "v", source, settings: { positive: "x", clipSkip: 3 }, seed: "1" });
  assert.equal("clip_skip" in native, false);
  assert.equal("output_template" in native, false);
  const blocked = imageToImageBlockers({ source, settings: { positive: "x", sizeMode: "custom", width: 3072, height: 1024 }, engine: "iL", canvasMaxEdge: 2048 });
  assert.match(blocked, /超出当前画布上限 2048/);
});

test("gallery cards keep the run's CLIP skip and never carry workspace preferences", () => {
  const saved = normalizedSettings({
    model: "iL", clipSkip: 3, canvasMaxEdge: 4096, outputTemplate: "Test/image",
    activePromptPresets: { positive: ["preset-a"], negative: [] },
  });
  assert.equal(saved.clipSkip, 3);
  for (const key of ["canvasMaxEdge", "outputTemplate", "activePromptPresets"]) assert.equal(key in saved, false, key);
  // Stored as JSON and read back, the card still says 3.
  assert.equal(normalizedSettings(JSON.parse(JSON.stringify(saved))).clipSkip, 3);
  // A card written before P6 ran the family default: it keeps saying nothing rather than adopting
  // whatever CLIP skip the workspace it is read against happens to hold.
  assert.equal(normalizedSettings({ model: "iL" }, { model: "iL", clipSkip: 5 }).clipSkip, null);
});

test("model matching: exact, unique suffix, filename, stem — and ambiguity instead of a first hit", () => {
  const exact = matchModelName("style/Style_九V6.0/matcha.safetensors", LORA_CATALOG);
  assert.equal(exact.status, "local");
  assert.equal(exact.precision, "exact");
  assert.equal(exact.display, "Style_九V6.0\\matcha.safetensors");
  assert.equal(exact.match.value, "style/Style_九V6.0/matcha.safetensors");
  const suffix = matchModelName("Style_九V6.0\\matcha.safetensors", LORA_CATALOG);
  assert.equal(suffix.precision, "suffix");
  assert.equal(suffix.match.value, "style/Style_九V6.0/matcha.safetensors");
  const longer = matchModelName("D:/sd-webui/models/Lora/style/Style_九V6.0/matcha.safetensors", LORA_CATALOG);
  assert.equal(longer.status, "local");
  assert.equal(longer.precision, "suffix");
  const filename = matchModelName("CUTE2712_Illustrious_v1.0.safetensors", LORA_CATALOG);
  assert.equal(filename.precision, "filename");
  const stem = matchModelName("CUTE2712_Illustrious_v1.0", LORA_CATALOG);
  assert.equal(stem.precision, "name");
  assert.equal(stem.display, "style\\CUTE2712_Illustrious_v1.0.safetensors");
  // Same name in two folders, at the root and in a folder, across engines and in a shared root.
  const duplicate = matchModelName("shou_ad_xl_half_v1", LORA_CATALOG);
  assert.equal(duplicate.status, "ambiguous");
  assert.equal(duplicate.count, 2);
  assert.deepEqual(duplicate.candidates.map((item) => item.display).sort(), ["Style_九V6.0\\shou_ad_xl_half_v1.safetensors", "Style_九V7.0\\shou_ad_xl_half_v1.safetensors"]);
  assert.equal(matchModelName("MomoSweetSpaceILL", LORA_CATALOG).status, "ambiguous");
  const crossEngine = matchModelName("matcha", LORA_CATALOG);
  assert.equal(crossEngine.status, "ambiguous");
  assert.equal(crossEngine.count, 3);
  // The picture's engine narrows a cross-engine name; a shared root enabled for that engine stays a candidate.
  assert.equal(matchModelName("matcha", LORA_CATALOG, { engine: "SD" }).status, "ambiguous");
  const onlyLocalSd = matchModelName("matcha", LORA_CATALOG.filter((entry) => entry.origin === "local"), { engine: "SD" });
  assert.equal(onlyLocalSd.status, "local");
  assert.equal(onlyLocalSd.precision, "name+engine");
  assert.equal(onlyLocalSd.match.value, "style/matcha.safetensors");
  const sharedExact = matchModelName("shared:abc/anime/matcha.safetensors", LORA_CATALOG);
  assert.equal(sharedExact.status, "shared");
  // A tag-only alias names no file and stays missing with its recorded name.
  const alias = matchModelName("my-alias-from-metadata", LORA_CATALOG);
  assert.deepEqual([alias.status, alias.name], ["missing", "my-alias-from-metadata"]);
  assert.equal(publicMatch(duplicate).pending, undefined);
});

test("an ambiguous match settles only on exactly one hash hit", () => {
  const ambiguous = matchModelName("shou_ad_xl_half_v1", LORA_CATALOG);
  const one = settleAmbiguousMatch(ambiguous, [ambiguous.pending[1]]);
  assert.equal(one.status, "local");
  assert.equal(one.precision, "name+hash");
  assert.equal(one.match.value, "style/Style_九V7.0/shou_ad_xl_half_v1.safetensors");
  const none = settleAmbiguousMatch(ambiguous, []);
  assert.equal(none.status, "ambiguous");
  assert.equal(none.hashChecked, true);
  assert.equal(none.pending, undefined);
});

test("A1111 lora hashes attach to the entries they name and display names keep the real folder", () => {
  const hashes = parseLoraHashes("shou_ad_xl_half_v1: a0f6d73f81ef, Amashiro Natsuki [style]Remake-Illus: 2b1f833e3d34, CUTE2712_Illustrious_v1.0: 2628e4f39ba5");
  assert.equal(hashes.get("amashiro natsuki [style]remake-illus"), "2b1f833e3d34");
  assert.equal(hashes.get("cute2712_illustrious_v1.0"), "2628e4f39ba5");
  const parsed = parseA1111(REFERENCE_INFOTEXT);
  assert.deepEqual(parsed.loras.map((item) => [item.name, item.weight, item.hash]), [
    ["shou_ad_xl_half_v1", 1, "a0f6d73f81ef"],
    ["matcha", 0.4, "cd00cd91342f"],
    ["MomoSweetSpaceILL", 0.4, "7cc957fa361c"],
  ]);
  assert.equal(modelDisplayName("matcha.safetensors"), "matcha.safetensors");
  assert.equal(modelDisplayName("style\\Style_九V6.0\\matcha.safetensors"), "Style_九V6.0\\matcha.safetensors");
  assert.equal(stripImageInfoMatches({ loras: [{ name: "a", hash: "abc", match: {} }] }).loras[0].hash, "abc");
});

test("this project's PNG records keep the relative LoRA ID, CLIP skip and the chain", () => {
  const info = interpretImageMetadata({
    prompt: "cat",
    negative_prompt: "",
    parameters: JSON.stringify({
      job_id: "j", engine: "iL", clip_skip: 3, width: 512, height: 512, output_width: 512, output_height: 512,
      loras: [{ name: "matcha.safetensors", path: "style/Style_九V6.0/matcha.safetensors", weight: 0.4 }],
      postprocess_chain: [{ type: "resample", enabled: true, upscaler: "RealESRGAN_x4plus_anime_6B.pth", scale: 2, resize_algorithm: "lanczos", denoise: 0.3, steps: null, cfg: null, seed_mode: "inherit", tile_size: 192, tile_overlap: 16, execution_mode: "full_frame", padding: 32, mask_blur: 8, uniform_tiles: true, tiled_decode: true, exact_output: false }],
    }),
  });
  assert.equal(info.loras[0].path, "style/Style_九V6.0/matcha.safetensors");
  assert.ok(info.parameters.some((item) => item.label === "Clip skip" && item.value === "3"));
  const candidate = imageInfoPostprocessCandidate(info);
  assert.equal(candidate.order, "recorded");
  assert.equal(candidate.stages[0].upscaler, "RealESRGAN_x4plus_anime_6B.pth");
});

test("the reference picture becomes an auditable Hires then three-unit ADetailer plan", () => {
  const info = { status: "ok", ...parseA1111(REFERENCE_INFOTEXT) };
  const candidate = imageInfoPostprocessCandidate(info);
  assert.equal(candidate.order, "assumed");
  assert.deepEqual(candidate.stages.map((stage) => stage.type), ["resample", "adetailer"]);
  const hires = candidate.stages[0];
  assert.deepEqual([hires.scale, hires.denoise, hires.steps, hires.cfg, hires.upscalerName], [2, 0.25, null, 5, "R-ESRGAN 4x+ Anime6B"]);
  const units = candidate.stages[1].units;
  assert.deepEqual(units.map((unit) => [unit.detectorName, unit.confidence, unit.dilateErode, unit.maskBlur, unit.denoise, unit.padding]), [
    ["face_yolov8n.pt", 0.3, 4, 4, 0.4, 32],
    ["face_yolov8s.pt", 0.3, 4, 4, 0.4, 32],
    ["hand_yolov8n.pt", 0.3, 4, 4, 0.1, 32],
  ]);
  assert.deepEqual(candidate.issues, []);
  const upscalers = [{ id: "RealESRGAN_x4plus_anime_6B.pth", name: "RealESRGAN_x4plus_anime_6B.pth", label: "RealESRGAN_x4plus_anime_6B", compatible: true }, { id: "RealESRGAN_x4plus.pth", name: "RealESRGAN_x4plus.pth", label: "RealESRGAN_x4plus", compatible: true }];
  const detectors = ["face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt"].map((name) => ({ name, value: name }));
  const plan = buildPostprocessChainPlan(candidate, { engine: "iL", upscalers, detectors, baseSteps: 27, size: { width: 768, height: 1280 } });
  assert.equal(plan.ready, true, plan.unapplied.join(";"));
  const enabled = plan.chain.filter((stage) => stage.enabled);
  assert.deepEqual(enabled.map((stage) => stage.type), ["resample", "adetailer"]);
  assert.equal(enabled[0].upscaler, "RealESRGAN_x4plus_anime_6B.pth");
  assert.equal(enabled[0].latentAlgorithm, null);
  assert.deepEqual(enabled[1].units.filter((unit) => unit.enabled).map((unit) => unit.detector), ["face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt"]);
  const reordered = buildPostprocessChainPlan(candidate, { engine: "iL", upscalers, detectors, baseSteps: 27, size: { width: 768, height: 1280 }, order: "ade-first" });
  assert.deepEqual(reordered.chain.filter((stage) => stage.enabled).map((stage) => stage.type), ["adetailer", "resample"]);
  // Nothing installed: nothing is enabled empty, both stages are itemised.
  const bare = buildPostprocessChainPlan(candidate, { engine: "iL", upscalers: [], detectors: [], baseSteps: 27 });
  assert.equal(bare.ready, false);
  assert.ok(bare.unapplied.some((line) => /放大器 R-ESRGAN 4x\+ Anime6B 未安装/.test(line)));
  assert.ok(bare.unapplied.some((line) => /face_yolov8n\.pt 未安装/.test(line)));
  // Only one detector installed: the other units are itemised and the one that can run is kept.
  const partial = buildPostprocessChainPlan(candidate, { engine: "iL", upscalers, detectors: detectors.slice(2), baseSteps: 27, size: { width: 768, height: 1280 } });
  assert.equal(partial.ready, true);
  assert.equal(partial.unapplied.length, 2);
});

test("unsupported recorded post-processing is reported, never fabricated", () => {
  const sevenUnits = Array.from({ length: 7 }, (_, index) => `ADetailer model${index ? ` ${index + 1}${["st", "nd", "rd"][index] || "th"}` : ""}: face_yolov8n.pt`).join(", ");
  const info = { status: "ok", ...parseA1111(`cat\nSteps: 20, Size: 512x512, ${sevenUnits}, ControlNet 0: \"Module: canny\", Refiner: sdxl_refiner`) };
  const candidate = imageInfoPostprocessCandidate(info);
  assert.ok(candidate.issues.some((line) => /超出 6 个单元/.test(line)));
  assert.ok(candidate.issues.some((line) => /ControlNet/.test(line)));
  assert.ok(candidate.issues.some((line) => /Refiner/.test(line)));
  const hiresWithoutDenoise = imageInfoPostprocessCandidate({ status: "ok", ...parseA1111("cat\nSteps: 20, Size: 512x512, Hires upscale: 2, Hires upscaler: Latent") });
  const plan = buildPostprocessChainPlan(hiresWithoutDenoise, { engine: "iL", baseSteps: 20 });
  assert.equal(plan.ready, false);
  assert.ok(plan.unapplied.some((line) => /Hires 未应用/.test(line)));
  const comfy = imageInfoPostprocessCandidate({ status: "ok", source: "comfyui", flags: [{ id: "hires", enabled: true }] });
  assert.match(comfy.issues[0], /ComfyUI/);
  assert.deepEqual(resolveUpscalerName("Latent (nearest-exact)"), { kind: "latent", latentAlgorithm: "latent-nearest-exact" });
  assert.equal(resolveUpscalerName("4x-UltraSharp", [{ id: "4x-UltraSharp.pth", label: "4x-UltraSharp" }]).upscaler, "4x-UltraSharp.pth");
  assert.equal(resolveUpscalerName("SwinIR 4x", []).kind, "missing");
  assert.equal(resolveDetectorName("face_yolov8n.pt", [{ name: "face_yolov8n.pt", value: "face_yolov8n.pt" }]), "face_yolov8n.pt");
  assert.equal(resolveDetectorName("None", []), "");
});

test("image-info apply: only unique target-engine LoRAs mount, the rest are itemised; CLIP skip is effective", () => {
  const info = {
    status: "ok", source: "a1111", positive: "cat", negative: "", parameters: [{ label: "Clip skip", value: "1" }, { label: "Size", value: "3072x1024" }],
    checkpoint: "wai", checkpointMatch: { status: "local", name: "wai", match: { value: "wai.safetensors", engine: "iL", display: "wai.safetensors" } },
    loras: [
      { name: "matcha", weight: 0.4, match: { status: "local", name: "matcha", display: "Style_九V6.0\\matcha.safetensors", match: { value: "style/Style_九V6.0/matcha.safetensors", engine: "iL" } } },
      { name: "shou", weight: 1, match: { status: "ambiguous", name: "shou", count: 2, candidates: [] } },
      { name: "sdonly", weight: 1, match: { status: "local", name: "sdonly", match: { value: "x.safetensors", engine: "SD" } } },
      { name: "gone", weight: 1, match: { status: "missing", name: "gone" } },
    ],
    flags: [],
  };
  const fields = imageInfoApplyFields(info);
  const loraField = fields.find((field) => field.id === "loras");
  assert.equal(loraField.entries[0].label, "Style_九V6.0\\matcha.safetensors");
  assert.equal(loraField.entries[1].status, "ambiguous");
  const plan = buildImageInfoApplyPlan(info, ["loras", "sampling", "canvas", "model"], "t2i", { engine: "SD", maximumEdge: 2048 });
  assert.equal(plan.targetEngine, "iL");
  assert.deepEqual(plan.overlay.loras.map((item) => item.value), ["style/Style_九V6.0/matcha.safetensors"]);
  assert.deepEqual(plan.missing.map((item) => item.name), ["shou", "sdonly", "gone"]);
  assert.match(plan.missing[0].reason, /有歧义/);
  assert.match(plan.missing[1].reason, /属于 SD 引擎/);
  assert.equal(plan.overlay.clipSkip, 2);
  assert.ok(plan.notes.some((line) => /CLIP skip 1 → 2/.test(line)));
  assert.deepEqual(plan.overlay.size, { width: 2048, height: 1024 });
  assert.ok(plan.notes.some((line) => /超出当前画布上限 2048/.test(line)));
  const raised = buildImageInfoApplyPlan(info, ["canvas"], "t2i", { engine: "iL", maximumEdge: 4096 });
  assert.deepEqual(raised.overlay.size, { width: 3072, height: 1024 });
  assert.deepEqual(imageInfoClipSkip(3, "Anima"), { value: null, note: "CLIP skip 3（Anima 不使用 CLIP skip）" });
  assert.equal(loraUnappliedReason({ status: "local", match: { engine: "iL" } }, "iL"), "");
  assert.equal(matchLabel({ status: "missing", name: "raw" }, "raw"), "raw");
});

test("LoRA folders: folders first, then the files directly inside, with search keeping location", () => {
  const category = {
    id: "style",
    label: "风格",
    models: [
      { name: "Style_九V6.0/matcha.safetensors", value: "style/Style_九V6.0/matcha.safetensors" },
      { name: "Style_九V6.0/shou.safetensors", value: "style/Style_九V6.0/shou.safetensors" },
      { name: "Style_九V6.0/deeper/inner.safetensors", value: "style/Style_九V6.0/deeper/inner.safetensors" },
      { name: "Style_九V7.0/shou.safetensors", value: "style/Style_九V7.0/shou.safetensors" },
      { name: "Deyui.safetensors", value: "style/Deyui.safetensors" },
    ],
  };
  const root = loraFolderView(category, "");
  assert.deepEqual(root.folders.map((folder) => [folder.name, folder.count]), [["Style_九V6.0", 3], ["Style_九V7.0", 1]]);
  assert.deepEqual(root.files.map(loraFileLabel), ["Deyui.safetensors"]);
  assert.equal(root.parent, null);
  const inside = loraFolderView(category, "Style_九V6.0");
  assert.deepEqual(inside.folders.map((folder) => folder.name), ["deeper"]);
  assert.deepEqual(inside.files.map(loraFileLabel), ["matcha.safetensors", "shou.safetensors"]);
  assert.equal(inside.parent, "");
  assert.deepEqual(inside.crumbs, [{ name: "Style_九V6.0", path: "Style_九V6.0" }]);
  const deeper = loraFolderView(category, "Style_九V6.0/deeper");
  assert.equal(deeper.parent, "Style_九V6.0");
  assert.deepEqual(deeper.files.map((item) => item.value), ["style/Style_九V6.0/deeper/inner.safetensors"]);
  // A search at the root lists the folders holding matches rather than floating the files up.
  const searched = loraFolderView(category, "", "shou");
  assert.deepEqual(searched.files, []);
  assert.deepEqual(searched.folders.map((folder) => [folder.name, folder.matches]), [["Style_九V6.0", 1], ["Style_九V7.0", 1]]);
  const vanished = loraFolderView(category, "Style_九V6.0/removed");
  assert.equal(vanished.path, "Style_九V6.0");
  assert.equal(vanished.missing, true);
  assert.equal(normalizeLoraFolderPath("a/../b"), "");
  assert.equal(normalizeLoraFolderPath("a\\b\\"), "a/b");
  assert.deepEqual(loraCategoryLocation("style/Style_九V6.0", [{ id: "style", directory: "style" }]), { categoryId: "style", folder: "Style_九V6.0" });
  assert.deepEqual(loraCategoryLocation("custom/x", [{ id: "style", directory: "style" }]), { categoryId: "folder:custom", folder: "x" });
});

test("the reference PNG itself reads into the same plan", async () => {
  const buffer = await readFile(new URL("../../README/BUG/00039-3576420342.png", import.meta.url)).catch(() => null);
  if (!buffer) return;
  const info = interpretImageMetadata(readPngMetadataChunks(buffer));
  const candidate = imageInfoPostprocessCandidate(info);
  assert.deepEqual(candidate.stages.map((stage) => stage.type), ["resample", "adetailer"]);
  assert.equal(candidate.stages[1].units.length, 3);
  assert.equal(info.loras.length, 7);
  assert.ok(info.loras.every((item) => /^[0-9a-f]{12}$/.test(item.hash)));
});
