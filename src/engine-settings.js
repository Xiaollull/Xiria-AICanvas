// One independent set of generation parameters per engine.
//
// The workspace used to hold a single set and rewrite parts of it whenever the engine changed:
// switching to Flux forced guidance to none, switching to Anima coerced the sampler into its
// vocabulary, and every switch cleared the model pickers outright. Going SD -> Krea2 -> SD did not
// bring the SD workspace back; it brought back whatever had survived being pushed through Krea 2.
//
// Here each engine owns its own record, and switching between them is a swap rather than an edit.
// A change made under one engine cannot reach another, so a Krea 2 canvas size, a Flux step count
// and an SD Hires chain coexist and stay put.
//
// Schema 2 gave each engine its own image-to-image block as well. The image page used to hold one
// shared set, so switching engines left behind a denoise, a sampler and a Hires chain chosen for a
// different engine -- normalized against the new engine's vocabulary only at submit time, which is
// too late to be visible. The prompt stays out of it there too, for the same reason it stays out
// here, and `mode` stays shared because it names which job the page is set up to run, not a
// parameter of the engine.
//
// Schema 3 moved the text-to-image prompt and its negative in as well, with the preset switches
// that compose onto them and the SD/iL CLIP skip. A prompt written for Flux's language model is
// not a prompt for an SD CLIP tower, and switching engines used to carry one into the other. The
// shared prompt a schema-2 workspace held is copied into all six engines once, on upgrade; after
// that each engine edits its own. The image page's prompts were already per engine.
//
// What is deliberately NOT in here: the preset *library* (one shared library of preset texts), the
// LoRA library (which has its own per-engine store in `lora-model-scope.js`), the search and
// category filters, and the panel open/closed states — they describe the window, not the run.

import { normalizeADetailerStage } from "./adetailer-units.js";
import { clipSkipRange, normalizeClipSkip } from "./clip-skip.js";
import { IMAGE_TO_IMAGE_DEFAULTS, normalizeImageToImageSettings } from "./image-to-image.js";
import { normalizeUint64Seed } from "./hires-settings.js";
import { migrateResampleChain } from "./resample-chain.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";

export { clipSkipRange, normalizeClipSkip };
export const ENGINE_SETTINGS_SCHEMA_VERSION = 3;
export const DEFAULT_POSITIVE_PROMPT = "a quiet observatory above the clouds, lone astronomer, warm light, cinematic composition";
export const DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, malformed hands, oversaturated, watermark";

function normalizeActivePresetIds(value) {
  const source = isPlainObject(value) ? value : {};
  const ids = (list) => (Array.isArray(list) ? [...new Set(list.filter((id) => typeof id === "string" && id))] : []);
  return { positive: ids(source.positive), negative: ids(source.negative) };
}
export const SETTINGS_ENGINES = ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"];
// Engines that mount separate component files instead of one checkpoint.
const SPLIT_ENGINES = new Set(["Anima", "Flux", "Flux2", "Krea2"]);
// Guidance distillation removes the unconditional branch, so there is nothing for either guidance
// enhancement to work against and the negative prompt has nothing to encode into.
const DISTILLED_ENGINES = new Set(["Flux", "Flux2"]);
// CFG-Zero* rescales the unconditional branch of a flow-matching sampler; the epsilon-prediction
// engines have no equivalent step to rescale.
const FLOW_MATCHING_ENGINES = new Set(["Anima", "Krea2"]);
// Only FLUX.1 carries a second text encoder.
const DUAL_ENCODER_ENGINES = new Set(["Flux"]);
const GUIDANCE_IDS = new Set(["none", "pag", "cfg_zero_star"]);
const SEED_MODES = new Set(["fixed", "random", "increment", "decrement"]);
export const PAG_DEFAULTS = Object.freeze({ scale: 0.3, appliedLayers: "mid" });

export function isSplitEngine(engine) { return SPLIT_ENGINES.has(engine); }
export function isDistilledEngine(engine) { return DISTILLED_ENGINES.has(engine); }

function isPlainObject(value) { return Boolean(value) && typeof value === "object" && !Array.isArray(value); }
function text(value) { return typeof value === "string" ? value : ""; }
function inRange(value, fallback, minimum, maximum) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(maximum, Math.max(minimum, numeric));
}

/**
 * What an engine starts with before anyone touches it.
 *
 * The split engines default to euler/simple because that is what their native runtimes ship; SD and
 * iL keep the dpmpp_2m/karras pair the ComfyUI samplers default to.
 */
// The split engines default to euler/simple because that is what their native runtimes ship; SD and
// iL keep the dpmpp_2m/karras pair the ComfyUI samplers default to. Both pages start an engine on
// the same pair, so this is the one place that decides it.
function engineDefaultSampler(engine) { return isSplitEngine(engine) ? "euler" : "dpmpp_2m"; }
function engineDefaultScheduler(engine) { return isSplitEngine(engine) ? "simple" : "karras"; }

/**
 * An engine's own image-to-image parameters. Every engine offers the same sampler and scheduler
 * names, so what differs is which pair a fresh record starts on and how a saved one is normalized.
 */
export function imageToImageDefaultsFor(engine, overrides = null) {
  const source = isPlainObject(overrides) ? overrides : {};
  const settings = normalizeImageToImageSettings(
    { ...IMAGE_TO_IMAGE_DEFAULTS, ...source },
    {
      samplers: SAMPLER_NAMES,
      schedulers: SCHEDULER_NAMES,
      defaultSampler: engineDefaultSampler(engine),
      defaultScheduler: engineDefaultScheduler(engine),
    },
  );
  // The image page's CLIP skip is its own, like the rest of its sampling block.
  return { ...settings, clipSkip: normalizeClipSkip(engine, source.clipSkip) };
}

export function engineSettingsDefaults(engine) {
  const split = isSplitEngine(engine);
  return {
    positive: DEFAULT_POSITIVE_PROMPT,
    negative: DEFAULT_NEGATIVE_PROMPT,
    activePromptPresets: { positive: [], negative: [] },
    clipSkip: normalizeClipSkip(engine, null),
    checkpoint: "",
    diffusionModel: "",
    textEncoder: "",
    textEncoder2: "",
    vae: "",
    steps: 28,
    cfg: 6.5,
    denoise: 1,
    imagesPerBatch: 1,
    batchCount: 1,
    seed: "847291",
    seedMode: "random",
    sampler: engineDefaultSampler(engine),
    scheduler: engineDefaultScheduler(engine),
    guidance: "none",
    pag: { ...PAG_DEFAULTS },
    size: { width: 1024, height: 1024 },
    postprocessChain: migrateResampleChain({}, { engine }),
    backgroundRemovalModel: "",
    imageToImage: imageToImageDefaultsFor(engine),
  };
}

/**
 * Which guidance an engine can actually run.
 *
 * Only the rules that are true of the engine itself are applied here. Anima's PAG additionally
 * depends on what the installed runtime reports, which this store cannot see, so that check stays
 * where the health payload is.
 */
function resolveGuidance(engine, value) {
  const requested = GUIDANCE_IDS.has(value) ? value : "none";
  if (DISTILLED_ENGINES.has(engine)) return "none";
  if (requested === "cfg_zero_star" && !FLOW_MATCHING_ENGINES.has(engine)) return "none";
  // Krea 2 keeps its unconditional branch, so CFG-Zero* survives, but this runtime installs no
  // attention override for its single-stream blocks and cannot do PAG.
  if (requested === "pag" && engine === "Krea2") return "none";
  return requested;
}

/** The guidance `engine` will actually run for a requested one; shared with the Gallery normaliser. */
export function engineGuidance(engine, value) {
  return resolveGuidance(engine, value);
}

function normalizeSeedText(value, fallback) {
  const digits = String(value ?? "").replace(/\D/g, "");
  return digits ? normalizeUint64Seed(digits, fallback) : fallback;
}

// The hard sampling ceiling. What the user may pick is the configurable canvas bound, checked by the
// page against the canvas it has; a stored canvas is never cut down to a lower bound silently.
export const MAX_CANVAS_EDGE = 8192;

function normalizeSize(value, fallback) {
  const source = isPlainObject(value) ? value : {};
  const dimension = (input, previous) => Math.round(inRange(input, previous, 0, MAX_CANVAS_EDGE) / 64) * 64;
  return { width: dimension(source.width, fallback.width), height: dimension(source.height, fallback.height) };
}

/** One engine's record, coerced to what that engine can actually accept. */
export function normalizeEngineSettings(engine, value, fallback = null) {
  const defaults = fallback ? { ...engineSettingsDefaults(engine), ...fallback } : engineSettingsDefaults(engine);
  const source = isPlainObject(value) ? value : {};
  const split = isSplitEngine(engine);
  return {
    // An empty prompt is a real value — someone cleared the box — so only a missing or non-text
    // one falls back.
    positive: typeof source.positive === "string" ? source.positive : defaults.positive,
    negative: typeof source.negative === "string" ? source.negative : defaults.negative,
    activePromptPresets: normalizeActivePresetIds(isPlainObject(source.activePromptPresets) ? source.activePromptPresets : defaults.activePromptPresets),
    clipSkip: normalizeClipSkip(engine, source.clipSkip ?? defaults.clipSkip),
    // A checkpoint means nothing to a split engine and component files mean nothing to a
    // single-file one, so each engine stores only the pickers it actually has. Keeping both would
    // let a stale value from the other kind reappear as a phantom selection.
    checkpoint: split ? "" : text(source.checkpoint) || defaults.checkpoint,
    diffusionModel: split ? text(source.diffusionModel) || defaults.diffusionModel : "",
    textEncoder: split ? text(source.textEncoder) || defaults.textEncoder : "",
    textEncoder2: DUAL_ENCODER_ENGINES.has(engine) ? text(source.textEncoder2) || defaults.textEncoder2 : "",
    vae: split ? text(source.vae) || defaults.vae : "",
    steps: Math.round(inRange(source.steps, defaults.steps, 1, 60)),
    cfg: inRange(source.cfg, defaults.cfg, 1, 15),
    denoise: inRange(source.denoise, defaults.denoise, 0, 1),
    imagesPerBatch: Math.round(inRange(source.imagesPerBatch, defaults.imagesPerBatch, 1, 10)),
    batchCount: Math.round(inRange(source.batchCount, defaults.batchCount, 1, 20)),
    seed: normalizeSeedText(source.seed, defaults.seed),
    seedMode: SEED_MODES.has(source.seedMode) ? source.seedMode : defaults.seedMode,
    sampler: SAMPLER_NAMES.includes(source.sampler) ? source.sampler : defaults.sampler,
    scheduler: SCHEDULER_NAMES.includes(source.scheduler) ? source.scheduler : defaults.scheduler,
    guidance: resolveGuidance(engine, source.guidance ?? defaults.guidance),
    pag: {
      scale: Math.round(inRange(source.pag?.scale, defaults.pag.scale, 0, 5) * 100) / 100,
      appliedLayers: ["mid", "all"].includes(source.pag?.appliedLayers) ? source.pag.appliedLayers : defaults.pag.appliedLayers,
    },
    size: normalizeSize(source.size, defaults.size),
    // A record saved before the chain carries the three blocks and an order; one saved after
    // carries the chain. `migrateResampleChain` reads either, so an engine opened for the first
    // time since the upgrade finds the run it was left set up for.
    postprocessChain: migrateResampleChain(
      Array.isArray(source.postprocessChain) || isPlainObject(source.hires) || isPlainObject(source.rtx) || isPlainObject(source.adetailer)
        ? source
        : defaults,
      { engine, samplers: SAMPLER_NAMES, schedulers: SCHEDULER_NAMES },
    ),
    backgroundRemovalModel: text(source.backgroundRemovalModel) || defaults.backgroundRemovalModel,
    // A record saved under schema 1 has no image-to-image block; it starts from this engine's
    // defaults rather than from whatever the other engines happen to hold.
    imageToImage: imageToImageDefaultsFor(engine, isPlainObject(source.imageToImage) ? source.imageToImage : defaults.imageToImage),
  };
}

export function emptyEngineSettingsMap() {
  return {
    schemaVersion: ENGINE_SETTINGS_SCHEMA_VERSION,
    byEngine: Object.fromEntries(SETTINGS_ENGINES.map((engine) => [engine, engineSettingsDefaults(engine)])),
  };
}

/**
 * Reads a saved map, and folds a pre-split workspace into one when there is no map yet.
 *
 * The fold seeds every engine's parameters from the single set that used to be shared, rather than
 * seeding only the engine that happened to be selected. Anything else would look like data loss on the first
 * launch after upgrading: the parameters the user had set would survive under one engine and every
 * other engine would open on defaults. Each copy is coerced to what that engine accepts, so an SD
 * sampler landing in Flux's record becomes Flux's default rather than an invalid request.
 */
export function normalizeEngineSettingsMap(value, { fieldMissing = false, legacy = null, legacyImageToImage = null, legacyPrompts = null } = {}) {
  const container = emptyEngineSettingsMap();
  let rejected = 0;

  if (fieldMissing || !isPlainObject(value)) {
    if (isPlainObject(legacy)) {
      // Parameters seed every engine; model files seed only the engine they were chosen under. A
      // diffusion model, text encoder and VAE belong to one engine, and copying them everywhere put
      // Anima's Qwen3 0.6B encoder into Krea 2's record as a selection Krea 2 cannot load -- which
      // the image reader's Apply then borrowed and was refused for.
      const owner = SETTINGS_ENGINES.includes(legacy.model) ? legacy.model : null;
      const withoutModelFiles = { ...legacy, checkpoint: "", diffusionModel: "", textEncoder: "", textEncoder2: "", vae: "" };
      for (const engine of SETTINGS_ENGINES) {
        const record = engine === owner ? legacy : withoutModelFiles;
        container.byEngine[engine] = normalizeEngineSettings(engine, { ...record, imageToImage: legacyImageToImage ?? record.imageToImage });
      }
    }
    return { container, rejected: 0, migrated: isPlainObject(legacy), warning: "" };
  }

  const saved = isPlainObject(value.byEngine) ? value.byEngine : {};
  // A map written under schema 1 has no per-engine image block. The page held one shared set then,
  // so that set seeds every engine -- folded the same way the original single-set workspace was --
  // and each engine normalizes it against its own vocabulary.
  const foldImageBlock = Number(value.schemaVersion) < 2 && isPlainObject(legacyImageToImage);
  // A map written before schema 3 kept the text-to-image prompt beside it, shared by every engine.
  // That one prompt (and the preset switches composed onto it) is copied into each engine exactly
  // once; the schema number written back is what stops a second copy on the next read.
  const foldPrompts = Number(value.schemaVersion) < 3 && isPlainObject(legacyPrompts);
  const promptFields = foldPrompts ? {
    ...(typeof legacyPrompts.positive === "string" ? { positive: legacyPrompts.positive } : {}),
    ...(typeof legacyPrompts.negative === "string" ? { negative: legacyPrompts.negative } : {}),
    ...(isPlainObject(legacyPrompts.activePromptPresets) ? { activePromptPresets: legacyPrompts.activePromptPresets } : {}),
  } : {};
  for (const engine of SETTINGS_ENGINES) {
    const record = saved[engine];
    if (record !== undefined && !isPlainObject(record)) rejected += 1;
    // A missing engine is an empty scope rather than a corrupt file: a map written before an engine
    // shipped cannot carry its record, and treating that as corruption would strand the rest.
    const base = isPlainObject(record) ? record : null;
    container.byEngine[engine] = normalizeEngineSettings(engine, {
      ...base,
      ...(foldImageBlock ? { imageToImage: legacyImageToImage } : {}),
      ...promptFields,
    });
  }
  return {
    container,
    rejected,
    migrated: foldImageBlock || foldPrompts,
    warning: rejected ? `已重置 ${rejected} 个无效的引擎参数记录。` : "",
  };
}

export function engineSettingsFor(container, engine) {
  const target = SETTINGS_ENGINES.includes(engine) ? engine : SETTINGS_ENGINES[0];
  return normalizeEngineSettings(target, container?.byEngine?.[target]);
}

/** Returns a new map with one engine's record replaced; the others are untouched by construction. */
export function withEngineSettings(container, engine, settings) {
  if (!SETTINGS_ENGINES.includes(engine)) return container;
  const base = isPlainObject(container) && isPlainObject(container.byEngine)
    ? container
    : emptyEngineSettingsMap();
  return {
    schemaVersion: ENGINE_SETTINGS_SCHEMA_VERSION,
    byEngine: { ...base.byEngine, [engine]: normalizeEngineSettings(engine, settings) },
  };
}
