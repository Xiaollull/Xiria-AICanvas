// A picture added to a gallery card fills the card from its own generation record.
import assert from "node:assert/strict";
import test from "node:test";
import { galleryCardFromImageInfo, imageInfoHasGenerationData, imageInfoRecordedSeed } from "../src/gallery-image-info.js";
import { buildImageInfoApplyPlan, promptTagLoraStems } from "../src/image-info-apply.js";
import { interpretImageMetadata, parseA1111 } from "../src/image-metadata.js";
import { normalizedSettings } from "../src/gallery-core.js";
import { createResampleUnit, normalizeResampleChain } from "../src/resample-chain.js";

// The reference infotext from README/BUG/00039-3576420342.png (Forge; Hires, then three ADetailer units).
const REFERENCE_INFOTEXT = [
  "<lora:shou_ad_xl_half_v1:1>,<lora:matcha:0.4>,<lora:MomoSweetSpaceILL:0.4>,1girl, solo",
  "Negative prompt: bad hands",
  "Steps: 27, Sampler: Euler a, Schedule type: Automatic, CFG scale: 5, Seed: 3576420342, Size: 768x1280, Model: waiIllustriousSDXL_v140, Denoising strength: 0.25, Clip skip: 2, "
    + "ADetailer model: face_yolov8n.pt, ADetailer confidence: 0.3, ADetailer denoising strength: 0.4, "
    + "ADetailer model 2nd: face_yolov8s.pt, ADetailer confidence 2nd: 0.3, ADetailer denoising strength 2nd: 0.4, "
    + "ADetailer model 3rd: hand_yolov8n.pt, ADetailer confidence 3rd: 0.3, ADetailer denoising strength 3rd: 0.1, "
    + "Hires upscale: 2, Hires upscaler: R-ESRGAN 4x+ Anime6B, "
    + "Lora hashes: \"shou_ad_xl_half_v1: a0f6d73f81ef, matcha: cd00cd91342f, MomoSweetSpaceILL: 7cc957fa361c\"",
].join("\n");

const UPSCALERS = [{ id: "RealESRGAN_x4plus_anime_6B.pth", name: "RealESRGAN_x4plus_anime_6B.pth", label: "RealESRGAN_x4plus_anime_6B", compatible: true }];
const DETECTORS = ["face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt"].map((name) => ({ name, value: name }));

function local(value, engine, display = value) {
  return { status: "local", name: value, display, match: { value, engine, display } };
}

// What the control plane's read route answers for the reference picture on a machine that has its
// checkpoint and LoRAs: the parsed record plus one verdict per model.
function referenceRecord() {
  const parsed = parseA1111(REFERENCE_INFOTEXT);
  return {
    status: "ok",
    sourceLabel: "SD-WebUI / A1111",
    ...parsed,
    checkpointMatch: local("waiIllustriousSDXL_v140.safetensors", "iL"),
    loras: parsed.loras.map((item) => ({ ...item, match: local(`style/Style_九V6.0/${item.name}.safetensors`, "iL", `Style_九V6.0\\${item.name}.safetensors`) })),
  };
}

// A card that starts from a different engine and a workspace chain the picture never used.
function workspaceCard() {
  return normalizedSettings({
    model: "SD",
    checkpoint: "sd15.safetensors",
    positive: "old prompt",
    loras: [{ value: "old.safetensors", name: "old", weight: 1, enabled: true }],
    clipSkip: 1,
    postprocessChain: normalizeResampleChain([createResampleUnit({ scale: 1.5, denoise: 0.4 }, { engine: "SD" })], { engine: "SD" }),
  });
}

test("a Forge picture fills the card: model, prompt with its LoRA tags, sampling, canvas and the chain", () => {
  const result = galleryCardFromImageInfo(workspaceCard(), referenceRecord(), {
    canvasMaxEdge: 2048, upscalers: UPSCALERS, detectors: DETECTORS,
  });
  const card = result.settings;
  assert.equal(result.engineChanged, true);
  assert.equal(card.model, "iL");
  assert.equal(card.checkpoint, "waiIllustriousSDXL_v140.safetensors");
  assert.match(card.positive, /^<lora:shou_ad_xl_half_v1:1>,<lora:matcha:0\.4>/);
  assert.equal(card.negative, "bad hands");
  assert.deepEqual([card.steps, card.cfg, card.seed, card.seedMode, card.clipSkip], [27, 5, "3576420342", "fixed", 2]);
  assert.deepEqual(card.size, { width: 768, height: 1280 });
  // The prompt's own tags load all three LoRAs; mounting them too would be refused at generation.
  assert.deepEqual(card.loras, []);
  assert.equal(result.promptLoras.length, 3);
  assert.ok(result.notes.some((line) => /3 个 LoRA 由提示词中的 <lora:…> 标签加载/.test(line)));
  const enabled = card.postprocessChain.filter((stage) => stage.enabled);
  assert.deepEqual(enabled.map((stage) => stage.type), ["resample", "adetailer"]);
  assert.equal(enabled[0].upscaler, "RealESRGAN_x4plus_anime_6B.pth");
  assert.deepEqual(enabled[1].units.filter((unit) => unit.enabled).map((unit) => unit.detector), DETECTORS.map((item) => item.value));
  assert.deepEqual(result.filled, ["提示词", "采样参数", "画布", "底模", "后处理链"]);
  assert.equal(result.seed, "3576420342");
  assert.deepEqual([result.missing, result.unapplied], [[], []]);
});

test("what the card cannot take is itemised, and only chain stages that can run here are written", () => {
  const record = referenceRecord();
  record.checkpointMatch = { status: "missing", name: "waiIllustriousSDXL_v140" };
  const before = workspaceCard();
  const result = galleryCardFromImageInfo(before, record, { canvasMaxEdge: 2048, upscalers: [], detectors: DETECTORS });
  // No installed checkpoint: the engine and the model stay; nothing is invented.
  assert.equal(result.settings.model, "SD");
  assert.equal(result.settings.checkpoint, "sd15.safetensors");
  assert.deepEqual(result.missing.map((item) => [item.kind, item.reasonKind]), [["checkpoint", "missing"]]);
  // The Hires upscaler is not installed: that stage is left out with its reason, while the
  // ADetailer stage, whose detectors are installed, is written — the image reader's own rule.
  assert.ok(result.unapplied.some((line) => /放大器 R-ESRGAN 4x\+ Anime6B 未安装/.test(line)));
  assert.deepEqual(result.settings.postprocessChain.filter((stage) => stage.enabled).map((stage) => stage.type), ["adetailer"]);
  // With nothing installed at all, no stage is enabled empty.
  const bare = galleryCardFromImageInfo(before, record, { canvasMaxEdge: 2048, upscalers: [], detectors: [], detectorRuntimeReady: false });
  assert.equal(bare.settings.postprocessChain.some((stage) => stage.enabled), false);
  assert.equal(bare.filled.includes("后处理链"), false);
});

test("a picture without post-processing or LoRAs leaves the card with none", () => {
  const info = interpretImageMetadata({
    prompt: "cat, masterpiece",
    negative_prompt: "lowres",
    parameters: JSON.stringify({ job_id: "j", engine: "SD", checkpoint: "sd15.safetensors", steps: 20, cfg: 7, sampler: "euler", scheduler: "normal", seed: "42", width: 512, height: 768, output_width: 512, output_height: 768, loras: [] }),
  });
  const result = galleryCardFromImageInfo(workspaceCard(), { ...info, checkpointMatch: local("sd15.safetensors", "SD") }, { canvasMaxEdge: 2048 });
  const card = result.settings;
  assert.equal(result.engineChanged, false);
  assert.deepEqual([card.positive, card.negative, card.sampler, card.scheduler, card.seed], ["cat, masterpiece", "lowres", "euler", "normal", "42"]);
  assert.deepEqual(card.size, { width: 512, height: 768 });
  // Not recorded: the family default ran, so the card says nothing either.
  assert.equal(card.clipSkip, null);
  assert.deepEqual(card.loras, []);
  assert.deepEqual(card.postprocessChain.map((stage) => [stage.type, stage.enabled]), [["adetailer", false], ["rtx", false]]);
});

test("a mounted LoRA that is not in the prompt is mounted on the card, with the card's own shape", () => {
  const info = interpretImageMetadata({
    prompt: "cat",
    negative_prompt: "",
    parameters: JSON.stringify({ job_id: "j", engine: "iL", checkpoint: "wai.safetensors", steps: 20, seed: "7", width: 832, height: 1216, output_width: 832, output_height: 1216, loras: [{ name: "matcha", path: "style/Style_九V6.0/matcha.safetensors", weight: 0.45 }] }),
  });
  const record = {
    ...info,
    checkpointMatch: local("wai.safetensors", "iL"),
    loras: info.loras.map((item) => ({ ...item, match: local(item.path, "iL") })),
  };
  const result = galleryCardFromImageInfo(workspaceCard(), record, { canvasMaxEdge: 2048 });
  assert.deepEqual(result.settings.loras, [{ value: "style/Style_九V6.0/matcha.safetensors", name: "matcha", category: "style", weight: 0.45, precision: 2, enabled: true }]);
  assert.deepEqual(result.filled.slice(-1), ["LoRA 1"]);
});

test("a Hires picture of this program fills its sampled canvas, not the finished size, with its chain", () => {
  const info = interpretImageMetadata({
    prompt: "1girl",
    negative_prompt: "",
    parameters: JSON.stringify({
      job_id: "j", engine: "iL", checkpoint: "wai.safetensors", steps: 12, cfg: 5.5, sampler: "euler", scheduler: "normal", seed: "246813579", clip_skip: 3,
      width: 768, height: 768, output_width: 1152, output_height: 1152, loras: [],
      postprocess_chain: [
        { type: "resample", enabled: true, upscaler: null, latent_algorithm: "latent", scale: 1.5, resize_algorithm: "lanczos", denoise: 0.5, steps: 8, cfg: null, seed_mode: "inherit", tile_size: 192, tile_overlap: 16, execution_mode: "full_frame", padding: 32, mask_blur: 8, uniform_tiles: true, tiled_decode: true, exact_output: false },
        { type: "adetailer", enabled: true, units: [{ detector: "face_yolov8n.pt", confidence: 0.3, max_detections: 3, mask_min_ratio: 0, mask_max_ratio: 1, dilate_erode: 4, mask_blur: 4, padding: 32, denoise: 0.4, use_steps: false, steps: 28, use_cfg: false, cfg: 7, prompt: "", negative_prompt: "" }] },
        { type: "rtx", enabled: false, scale: 2, quality: "ultra" },
      ],
    }),
  });
  const result = galleryCardFromImageInfo(workspaceCard(), { ...info, checkpointMatch: local("wai.safetensors", "iL") }, { canvasMaxEdge: 2048, detectors: DETECTORS });
  assert.deepEqual(result.settings.size, { width: 768, height: 768 });
  assert.equal(result.settings.clipSkip, 3);
  const enabled = result.settings.postprocessChain.filter((stage) => stage.enabled);
  assert.deepEqual(enabled.map((stage) => stage.type), ["resample", "adetailer"]);
  assert.deepEqual([enabled[0].latentAlgorithm, enabled[0].scale, enabled[0].steps], ["latent", 1.5, 8]);
});

test("records without generation data fill nothing; seeds are read as decimal strings", () => {
  assert.equal(galleryCardFromImageInfo(workspaceCard(), { status: "unsupported" }, {}), null);
  assert.equal(galleryCardFromImageInfo(workspaceCard(), { status: "ok", parameters: [], loras: [], flags: [] }, {}), null);
  assert.equal(imageInfoHasGenerationData({ status: "empty" }), false);
  assert.equal(imageInfoRecordedSeed(referenceRecord()), "3576420342");
  assert.equal(imageInfoRecordedSeed({ status: "ok", parameters: [{ label: "Seed", value: "-1" }] }), "");
});

test("the image reader keeps a tagged LoRA with the prompt and mounts it only when the prompt stays behind", () => {
  const record = referenceRecord();
  assert.deepEqual([...promptTagLoraStems("<lora:style/Matcha.safetensors:0.4> <LYCO:momo:1>")], ["matcha", "momo"]);
  const withPrompt = buildImageInfoApplyPlan(record, ["prompts", "loras"], "t2i", { engine: "iL" });
  assert.equal(withPrompt.overlay.loras, undefined);
  assert.equal(withPrompt.groups.includes("loras"), false);
  assert.equal(withPrompt.promptLoras.length, 3);
  const withoutPrompt = buildImageInfoApplyPlan(record, ["loras"], "t2i", { engine: "iL" });
  assert.equal(withoutPrompt.overlay.loras.length, 3);
  assert.deepEqual(withoutPrompt.promptLoras, []);
});
