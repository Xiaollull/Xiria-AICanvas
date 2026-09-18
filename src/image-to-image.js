// Image-to-image: the source picture, the canvas it is resampled onto, and the job body that
// carries both.
//
// Two decisions shape this module.
//
// First, `denoise` is not a new parameter. `GenerateInput.denoise` has existed since long before
// this page, and the inference server answered any value below 1.0 with "reserved for the upcoming
// image-to-image pipeline and is ignored for text-to-image". This is that pipeline, so the field
// finally means what it always said it would, and text-to-image keeps ignoring it. Nothing about
// the text-to-image contract moves.
//
// Second, the source picture is deliberately *not* part of persisted workspace state. Every other
// setting on the page is remembered in `workspace.imageToImage`, but a picture is a multi-megabyte
// data URL and `ui-state.json` is rewritten on every slider tick; storing it there would quietly
// turn a settings file into an image store and make each autosave proportional to the picture. The
// picture lives for the session only, and the page says so rather than pretending otherwise.

import { formatFileSize } from "./format-size.js";
import { DISTILLED_GUIDANCE_ENGINES, NATIVE_STEP_ENGINES, adetailerStageIssue } from "./adetailer-units.js";
import {
  chainStage,
  enabledChainStages,
  migrateResampleChain,
  normalizeResampleChain,
  resampleChainIssue,
  resampleChainPayload,
  resampleChainPlan,
  resampleUnits,
} from "./resample-chain.js";
import { transparentBackgroundEnabled, transparentBackgroundSubject } from "./transparent-background.js";

export const IMAGE_TO_IMAGE_SCHEMA_VERSION = 1;

// The two things this page can be asked to do. `transform` is the original contract: resample the
// source onto a canvas and sample it with the prompt. `postprocess` skips that pass entirely and
// runs the enabled enhancement stages on the picture as supplied — same model, same LoRAs, same
// prompt, same order controls, no regeneration. It is a mode rather than a separate page because
// every control except the base sampling pass is shared between the two.
export const IMAGE_TO_IMAGE_MODES = Object.freeze([
  { id: "transform", label: "图生图", detail: "以来源图为底重新采样" },
  { id: "postprocess", label: "后处理", detail: "不重绘，只按顺序执行增强" },
]);

// What the server will accept as a post-processing source. It keeps the picture at its own size, so
// the bound is the stage envelope (8192 edge / 32MP) rather than the 2048 sampling canvas.
export const POSTPROCESS_SOURCE_MIN_EDGE = 64;
export const POSTPROCESS_SOURCE_MAX_EDGE = 8192;
export const POSTPROCESS_SOURCE_MAX_PIXELS = 32 * 1024 * 1024;

// A source larger than this is refused before it is read, not after: `FileReader` would otherwise
// hold the file, a base64 copy of it and the decoded bitmap at once.
export const SOURCE_IMAGE_MAX_BYTES = 32 * 1024 * 1024;
export const SOURCE_IMAGE_TYPES = Object.freeze(["image/png", "image/jpeg", "image/webp"]);
export const SOURCE_IMAGE_ACCEPT = SOURCE_IMAGE_TYPES.join(",");

// The same bounds `GenerateInput` enforces. Duplicated here so a request is shaped correctly before
// it is sent rather than rejected after; the server remains the authority.
export const MINIMUM_EDGE = 64;
export const MAXIMUM_EDGE = 2048;
export const EDGE_STEP = 64;

export const MINIMUM_DENOISE = 0.05;
export const MINIMUM_SCALE = 0.25;
export const MAXIMUM_SCALE = 4;

// How the source pixels are mapped onto a canvas with a different aspect ratio. The backend does
// the resampling; these ids are the contract between the two.
export const RESIZE_MODES = Object.freeze([
  { id: "cover", label: "填充裁切", detail: "铺满画布，裁掉超出的部分" },
  { id: "contain", label: "完整留边", detail: "完整放入画布，空白处补边" },
  { id: "stretch", label: "拉伸变形", detail: "直接拉伸到画布比例" },
]);

export const SIZE_MODES = Object.freeze([
  { id: "source", label: "跟随原图", detail: "对齐到 64 的倍数" },
  { id: "scale", label: "按比例", detail: "在原图尺寸上缩放" },
  { id: "custom", label: "自定义", detail: "手动指定输出尺寸" },
]);

export const IMAGE_TO_IMAGE_DEFAULTS = Object.freeze({
  mode: "transform",
  positive: "",
  negative: "",
  // 0.6 keeps the composition of the source while giving the prompt real room. 1.0 would discard
  // the picture entirely, which is text-to-image and already has a page.
  denoise: 0.6,
  steps: 28,
  cfg: 6.5,
  sampler: "",
  scheduler: "",
  seed: "847291",
  seedMode: "random",
  imagesPerBatch: 1,
  batchCount: 1,
  sizeMode: "source",
  scale: 1,
  width: 1024,
  height: 1024,
  resizeMode: "cover",
  // One switched-off resample unit, the same as the workspace starts with, so the card is on the
  // page to be switched on rather than something to discover an "add" button for. An empty array is
  // a different thing and stays empty: it is what deleting every unit leaves behind.
  postprocessChain: migrateResampleChain({}),
});

function numberInRange(value, fallback, minimum, maximum) {
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(maximum, Math.max(minimum, numeric));
}

function pickFromCatalog(value, catalog, fallback) {
  if (!Array.isArray(catalog) || catalog.length === 0) return typeof value === "string" ? value : fallback;
  if (typeof value === "string" && catalog.includes(value)) return value;
  return catalog.includes(fallback) ? fallback : catalog[0];
}

export function normalizeSourceSeed(value) {
  const digits = String(value ?? "").replace(/\D/g, "") || "0";
  const parsed = BigInt(digits);
  const maximum = 0xFFFFFFFFFFFFFFFFn;
  return (parsed > maximum ? maximum : parsed).toString();
}

// Never throws. A damaged stored block costs the remembered settings, never the page — the same
// rule the rest of the workspace follows.
export function normalizeImageToImageSettings(value, { samplers = [], schedulers = [], defaultSampler = "", defaultScheduler = "" } = {}) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  return {
    mode: IMAGE_TO_IMAGE_MODES.some((item) => item.id === source.mode) ? source.mode : IMAGE_TO_IMAGE_DEFAULTS.mode,
    positive: typeof source.positive === "string" ? source.positive : IMAGE_TO_IMAGE_DEFAULTS.positive,
    negative: typeof source.negative === "string" ? source.negative : IMAGE_TO_IMAGE_DEFAULTS.negative,
    denoise: Math.round(numberInRange(source.denoise, IMAGE_TO_IMAGE_DEFAULTS.denoise, MINIMUM_DENOISE, 1) * 100) / 100,
    steps: Math.round(numberInRange(source.steps, IMAGE_TO_IMAGE_DEFAULTS.steps, 1, 60)),
    cfg: Math.round(numberInRange(source.cfg, IMAGE_TO_IMAGE_DEFAULTS.cfg, 0, 30) * 10) / 10,
    sampler: pickFromCatalog(source.sampler, samplers, defaultSampler),
    scheduler: pickFromCatalog(source.scheduler, schedulers, defaultScheduler),
    seed: normalizeSourceSeed(source.seed ?? IMAGE_TO_IMAGE_DEFAULTS.seed),
    seedMode: ["random", "fixed", "increment", "decrement"].includes(source.seedMode) ? source.seedMode : IMAGE_TO_IMAGE_DEFAULTS.seedMode,
    imagesPerBatch: Math.round(numberInRange(source.imagesPerBatch, 1, 1, 10)),
    batchCount: Math.round(numberInRange(source.batchCount, 1, 1, 20)),
    sizeMode: SIZE_MODES.some((mode) => mode.id === source.sizeMode) ? source.sizeMode : IMAGE_TO_IMAGE_DEFAULTS.sizeMode,
    scale: Math.round(numberInRange(source.scale, 1, MINIMUM_SCALE, MAXIMUM_SCALE) * 100) / 100,
    width: snapEdge(source.width ?? IMAGE_TO_IMAGE_DEFAULTS.width),
    height: snapEdge(source.height ?? IMAGE_TO_IMAGE_DEFAULTS.height),
    resizeMode: RESIZE_MODES.some((mode) => mode.id === source.resizeMode) ? source.resizeMode : IMAGE_TO_IMAGE_DEFAULTS.resizeMode,
    // Either shape, read once: a record saved before the chain carries the three blocks, one
    // saved after carries the chain itself.
    postprocessChain: Array.isArray(source.postprocessChain)
      ? normalizeResampleChain(source.postprocessChain, { samplers, schedulers })
      : migrateResampleChain(source, { samplers, schedulers }),
  };
}

export function snapEdge(value) {
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(numeric) || numeric <= 0) return MINIMUM_EDGE;
  const snapped = Math.round(numeric / EDGE_STEP) * EDGE_STEP;
  return Math.min(MAXIMUM_EDGE, Math.max(MINIMUM_EDGE, snapped));
}

// Scale to fit, then snap — in that order. Clamping a 4000 px edge straight down to 2048 while
// leaving a 3000 px edge to be snapped on its own would silently reframe an oversized picture;
// fitting the long edge first keeps the aspect ratio, and the 64 px rounding stays the only
// distortion the geometry introduces.
export function fitSourceSize(width, height) {
  const sourceWidth = Number.isFinite(width) && width > 0 ? width : 0;
  const sourceHeight = Number.isFinite(height) && height > 0 ? height : 0;
  if (!sourceWidth || !sourceHeight) return { width: snapEdge(IMAGE_TO_IMAGE_DEFAULTS.width), height: snapEdge(IMAGE_TO_IMAGE_DEFAULTS.height) };
  const ratio = Math.min(1, MAXIMUM_EDGE / Math.max(sourceWidth, sourceHeight));
  return { width: snapEdge(sourceWidth * ratio), height: snapEdge(sourceHeight * ratio) };
}

export function outputSize(source, settings) {
  const config = normalizeImageToImageSettings(settings);
  // Post-processing never resamples, so the canvas is not a choice: it is whatever the file is. The
  // stage chain then grows it from there, which is what `resampleChainPlan` reports.
  if (config.mode === "postprocess") {
    if (!source || !source.width || !source.height) return { width: config.width, height: config.height };
    return { width: source.width, height: source.height };
  }
  if (config.sizeMode === "custom" || !source || !source.width || !source.height) {
    return { width: config.width, height: config.height };
  }
  const scale = config.sizeMode === "scale" ? config.scale : 1;
  return fitSourceSize(source.width * scale, source.height * scale);
}

// Why a picture cannot be post-processed as-is, or "" when it can.
export function postprocessSourceIssue(source) {
  if (!source || !source.width || !source.height) return "";
  if (source.width < POSTPROCESS_SOURCE_MIN_EDGE || source.height < POSTPROCESS_SOURCE_MIN_EDGE) {
    return `后处理来源图每条边至少 ${POSTPROCESS_SOURCE_MIN_EDGE} 像素`;
  }
  if (Math.max(source.width, source.height) > POSTPROCESS_SOURCE_MAX_EDGE || source.width * source.height > POSTPROCESS_SOURCE_MAX_PIXELS) {
    return "后处理来源图超出 8192 边 / 32MP 安全上限";
  }
  return "";
}

// True when the canvas does not share the source's aspect ratio, which is exactly when the resize
// mode becomes visible in the result. The page uses it to explain the mode instead of leaving it as
// a control with no observable effect.
export function resizeModeMatters(source, size) {
  if (!source || !source.width || !source.height || !size?.width || !size?.height) return false;
  return Math.abs(source.width / source.height - size.width / size.height) > 0.01;
}

export function validateSourceFile(file) {
  if (!file) return "请选择一张图片";
  if (file.type && !SOURCE_IMAGE_TYPES.includes(file.type)) return "仅支持 PNG、JPEG 与 WebP 图片";
  if (Number.isFinite(file.size) && file.size > SOURCE_IMAGE_MAX_BYTES) {
    return `图片体积超过 ${formatFileSize(SOURCE_IMAGE_MAX_BYTES)} 上限`;
  }
  return "";
}

export function sourceImageSummary(source) {
  if (!source) return "";
  const parts = [`${source.width} × ${source.height}`];
  if (Number.isFinite(source.size) && source.size > 0) parts.push(formatFileSize(source.size));
  const label = String(source.type || "").split("/")[1];
  if (label) parts.push(label.toUpperCase());
  return parts.join(" · ");
}

// Why the run button is off, in the order the user can act on: no picture, then no model, then a
// service that is not up, then a run already in flight.
export function imageToImageBlockers({ source, settings, engine, engineReady = true, serviceReady = true, postprocess = {}, backgroundRemoval = {} }) {
  const config = normalizeImageToImageSettings(settings);
  const postprocessOnly = config.mode === "postprocess";
  if (!source) return "请先选择一张来源图片";
  // The prompt is required in both modes: Hires and ADetailer redraw through the mounted model and
  // the server refuses an empty prompt, so post-processing without one is not a shortcut available
  // here — even a run whose only stage is RTX carries it.
  if (!config.positive.trim()) return "请填写正向提示词";
  if (transparentBackgroundEnabled(config.positive) && !transparentBackgroundSubject(config.positive)) return "透明背景标签之外还需要填写主体提示词";
  if (transparentBackgroundEnabled(config.positive) && !backgroundRemoval.model) return "请选择透明背景模型";
  if (transparentBackgroundEnabled(config.positive) && !backgroundRemoval.model.installed) return "所选透明背景模型尚未下载";
  if (transparentBackgroundEnabled(config.positive) && !backgroundRemoval.runtimeAvailable) return "ONNX Runtime 尚未就绪，请重新运行环境配置器";
  if (!engineReady) return "请先在文生图页面选择可用的模型";
  if (!serviceReady) return "推理服务尚未就绪";
  if (postprocessOnly) {
    const sourceIssue = postprocessSourceIssue(source);
    if (sourceIssue) return sourceIssue;
    if (!enabledChainStages(config.postprocessChain).length) {
      return "后处理模式需要至少启用一个重采样单元、ADetailer 或 RTX VSR";
    }
  }
  // The server refuses every post-processing stage in ultra-low memory mode, so this is a 422 the
  // page can explain instead of submitting into.
  if (postprocess.ultraLow && enabledChainStages(config.postprocessChain).length) {
    return "极限省存模式不支持任何后处理阶段";
  }
  const chainIssue = resampleChainIssue(config.postprocessChain, {
    baseSteps: config.steps,
    engine,
    nativeEngines: NATIVE_STEP_ENGINES,
    upscalers: postprocess.upscalers ?? null,
  });
  if (chainIssue) return chainIssue;
  const adetailerStage = chainStage(config.postprocessChain, "adetailer");
  if (resampleUnits(config.postprocessChain).some((unit) => unit.enabled && unit.upscaler) && postprocess.hiresReady === false) {
    return postprocess.hiresReason || "超分运行环境当前不可用";
  }
  if (adetailerStage?.enabled && postprocess.adetailerReady === false) return postprocess.adetailerReason || "ADetailer 当前不可用";
  // Which unit is at fault matters once there are several: "ADetailer is not
  // ready" does not say which card to open.
  if (adetailerStage?.enabled) {
    // The engine decides what a pass executes: native Anima runs every requested
    // step whatever the denoise, so the multiplied count would refuse a run the
    // server would have accepted.
    const issue = adetailerStageIssue(adetailerStage, config.steps, undefined, engine);
    if (issue) return issue;
  }
  if (chainStage(config.postprocessChain, "rtx")?.enabled && postprocess.rtxReady === false) return postprocess.rtxReason || "RTX VSR 当前不可用";
  if (postprocess.dimensions?.valid === false) return postprocess.dimensions.reason || "后处理目标尺寸不安全";
  return "";
}

export function imageToImageRequestBody({
  engine,
  checkpoint,
  diffusionModel,
  textEncoder,
  textEncoder2,
  vae,
  source,
  settings,
  seed,
  loras = [],
  // A run started from a gallery output pins that output's resample seeds; an ordinary run sends
  // each unit's own mode.
  seedOverrides = null,
  gallerySettings = null,
  backgroundRemovalModel = null,
  samplers = [],
  schedulers = [],
}) {
  const config = normalizeImageToImageSettings(settings, { samplers, schedulers });
  const size = outputSize(source, config);
  const anima = engine === "Anima";
  const flux = engine === "Flux";
  const flux2 = engine === "Flux2";
  const krea2 = engine === "Krea2";
  // Every native engine mounts components instead of a checkpoint and none streams a decoded
  // latent, so the facts the request depends on are "is this a split model", "does it mount a
  // second text encoder" and "is it guidance distilled".
  const splitModel = anima || flux || flux2 || krea2;
  const distilled = DISTILLED_GUIDANCE_ENGINES.includes(engine);
  const postprocessOnly = config.mode === "postprocess";
  return {
    postprocess_only: postprocessOnly,
    engine,
    ...(flux
      ? { diffusion_model: diffusionModel, text_encoder: textEncoder, text_encoder_2: textEncoder2, vae }
      : splitModel
        ? { diffusion_model: diffusionModel, text_encoder: textEncoder, vae }
        : { checkpoint }),
    prompt: config.positive.trim(),
    background_removal_model: transparentBackgroundEnabled(config.positive) ? backgroundRemovalModel : null,
    // Both Flux generations are guidance distilled and have no unconditional branch to encode a
    // negative prompt into.
    negative_prompt: distilled ? "" : config.negative.trim(),
    width: size.width,
    height: size.height,
    steps: config.steps,
    cfg: config.cfg,
    denoise: config.denoise,
    seed: normalizeSourceSeed(seed ?? config.seed),
    images_per_batch: config.imagesPerBatch,
    batch_count: config.batchCount,
    sampler: config.sampler,
    scheduler: config.scheduler,
    guidance: "none",
    source_image: {
      enabled: true,
      image_data: source?.dataUrl || "",
      resize_mode: config.resizeMode,
      name: source?.name || "",
    },
    postprocess_chain: resampleChainPayload(config.postprocessChain, { engine, seedOverrides }),
    loras: loras.filter((lora) => lora.enabled !== false).map((lora) => ({ path: lora.value, weight: lora.weight })),
    gallery_settings: gallerySettings,
  };
}

// The seed actually used for a run, and what the box should read afterwards. Same walk as the
// text-to-image page so the two behave identically for a user who moves between them.
export function nextImageToImageSeed(settings, usedSeed) {
  const config = normalizeImageToImageSettings(settings);
  const total = BigInt(config.imagesPerBatch * config.batchCount);
  const maximum = 0xFFFFFFFFFFFFFFFFn;
  const current = BigInt(normalizeSourceSeed(usedSeed));
  if (config.seedMode === "increment") return ((current + total) & maximum).toString();
  if (config.seedMode === "decrement") return ((current - (total % (maximum + 1n)) + maximum + 1n) & maximum).toString();
  return normalizeSourceSeed(usedSeed);
}
