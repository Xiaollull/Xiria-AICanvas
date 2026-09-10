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
// What is deliberately NOT in here: the prompt and its negative, the preset library, the LoRA
// library (which has its own per-engine store in `lora-model-scope.js`), the search and category
// filters, and the panel open/closed states. Prompts are content rather than parameters -- trying
// the same prompt across engines is the normal reason to switch -- and the rest describe the
// window, not the run.

import { normalizeADetailerStage } from "./adetailer-units.js";
import { normalizeUint64Seed } from "./hires-settings.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";

export const ENGINE_SETTINGS_SCHEMA_VERSION = 1;
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
const RTX_QUALITIES = new Set(["low", "medium", "high", "ultra"]);
const POSTPROCESS_STAGES = ["hires", "adetailer", "rtx"];

export const PAG_DEFAULTS = Object.freeze({ scale: 0.3, appliedLayers: "mid" });
export const HIRES_DEFAULTS = Object.freeze({
  enabled: false,
  expanded: false,
  model: "",
  seedMode: "inherit",
  seed: "",
  scale: 1,
  denoise: 0.35,
  steps: 20,
  cfg: 7,
  tileSize: 192,
  tileOverlap: 16,
  executionMode: "full_frame",
  sampler: null,
  scheduler: null,
  tileWidth: "auto",
  tileHeight: "auto",
  padding: 32,
  maskBlur: 8,
  seamMode: "none",
  uniformTiles: true,
  tiledDecode: true,
});
export const RTX_DEFAULTS = Object.freeze({ enabled: false, expanded: false, scale: 2, quality: "ultra" });
export const ADETAILER_DEFAULTS = Object.freeze(normalizeADetailerStage({ enabled: false, expanded: false }));

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
export function engineSettingsDefaults(engine) {
  const split = isSplitEngine(engine);
  return {
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
    sampler: split ? "euler" : "dpmpp_2m",
    scheduler: split ? "simple" : "karras",
    guidance: "none",
    pag: { ...PAG_DEFAULTS },
    size: { width: 1024, height: 1024 },
    hires: { ...HIRES_DEFAULTS },
    adetailer: normalizeADetailerStage({ enabled: false, expanded: false }),
    rtx: { ...RTX_DEFAULTS },
    postprocessOrder: [...POSTPROCESS_STAGES],
    backgroundRemovalModel: "",
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

function normalizeSize(value, fallback) {
  const source = isPlainObject(value) ? value : {};
  const dimension = (input, previous) => Math.round(inRange(input, previous, 0, 2048) / 64) * 64;
  return { width: dimension(source.width, fallback.width), height: dimension(source.height, fallback.height) };
}

function normalizePostprocessOrder(value) {
  const source = Array.isArray(value) ? value.filter((stage) => POSTPROCESS_STAGES.includes(stage)) : [];
  const unique = [...new Set(source)];
  // A saved order that lost a stage still runs the missing one, in its default position, rather
  // than silently dropping it from the pipeline.
  return [...unique, ...POSTPROCESS_STAGES.filter((stage) => !unique.includes(stage))];
}

/** One engine's record, coerced to what that engine can actually accept. */
export function normalizeEngineSettings(engine, value, fallback = null) {
  const defaults = fallback ? { ...engineSettingsDefaults(engine), ...fallback } : engineSettingsDefaults(engine);
  const source = isPlainObject(value) ? value : {};
  const split = isSplitEngine(engine);
  return {
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
    hires: { ...HIRES_DEFAULTS, ...defaults.hires, ...(isPlainObject(source.hires) ? source.hires : {}) },
    adetailer: normalizeADetailerStage({ ...defaults.adetailer, ...(isPlainObject(source.adetailer) ? source.adetailer : {}) }),
    rtx: {
      ...RTX_DEFAULTS,
      ...defaults.rtx,
      ...(isPlainObject(source.rtx) ? source.rtx : {}),
      quality: RTX_QUALITIES.has(source.rtx?.quality) ? source.rtx.quality : defaults.rtx.quality,
      scale: inRange(source.rtx?.scale, defaults.rtx.scale, 1, 4),
    },
    postprocessOrder: normalizePostprocessOrder(source.postprocessOrder ?? defaults.postprocessOrder),
    backgroundRemovalModel: text(source.backgroundRemovalModel) || defaults.backgroundRemovalModel,
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
export function normalizeEngineSettingsMap(value, { fieldMissing = false, legacy = null } = {}) {
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
        container.byEngine[engine] = normalizeEngineSettings(engine, engine === owner ? legacy : withoutModelFiles);
      }
    }
    return { container, rejected: 0, migrated: isPlainObject(legacy), warning: "" };
  }

  const saved = isPlainObject(value.byEngine) ? value.byEngine : {};
  for (const engine of SETTINGS_ENGINES) {
    const record = saved[engine];
    if (record !== undefined && !isPlainObject(record)) rejected += 1;
    // A missing engine is an empty scope rather than a corrupt file: a map written before an engine
    // shipped cannot carry its record, and treating that as corruption would strand the rest.
    container.byEngine[engine] = normalizeEngineSettings(engine, isPlainObject(record) ? record : null);
  }
  return {
    container,
    rejected,
    migrated: false,
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
