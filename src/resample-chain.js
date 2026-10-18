// The post-processing chain: an ordered list of stages, each of which is either a resample unit,
// the ADetailer stage or the RTX VSR stage.
//
// It replaces the three fixed blocks and the separate order list. Those two could describe a run
// with one Hires pass; they cannot describe a run with three resample units, because an order of
// stage *names* has nothing to say about which unit runs third. So the order is the list.
//
// A resample unit is three optional parts — model upscale, algorithmic resize, diffusion redraw —
// and every combination of them is a real configuration:
//
//   model + scale>1 + denoise>0   the second pass Hires.fix always was
//   model + scale>1 + denoise=0   enlarge with the model, do not redraw
//   no model + scale>1 + denoise=0   a plain image resize
//   no model + scale>1 + denoise>0   resize, then redraw
//   scale=1 + denoise>0           a detail pass at the same size
//
// Sizes here are exact: `round(edge × scale)`, the number ComfyUI's `ImageScaleBy` computes, with
// no rounding up to 64. The engine's own latent grid is applied separately and reported separately,
// so a chain can always say both what was asked for and what will actually be sampled.

import { normalizeADetailerStage, activeADetailerUnits, adetailerPayload, adetailerSummary } from "./adetailer-units.js";
import { DIFFUSERS_SCHEDULE_STEP_LIMIT, diffusersScheduleTotal } from "./diffusers-schedule.js";
import { normalizeUint64Seed, supportsUsduTiled } from "./hires-settings.js";
import { bankersRound, MAX_RTX_OUTPUT_EDGE, MAX_RTX_OUTPUT_PIXELS, RTX_TARGET_LIMIT_REASON } from "./postprocessing.js";

export const RESAMPLE_UNIT_LIMIT = 8;
export { DIFFUSERS_SCHEDULE_STEP_LIMIT, diffusersScheduleTotal };
export const STAGE_TYPES = ["resample", "adetailer", "rtx"];
// The backend keeps latent and pixel resampling as two different contracts. Keep the two lists
// separate here as well: a model-less unit can choose either family, but the request can never
// accidentally send a latent id as `resize_algorithm` (or vice versa).
export const LATENT_ALGORITHMS = Object.freeze([
  { id: "latent", label: "Latent", detail: "潜空间双线性" },
  { id: "latent-antialias", label: "Latent · Antialiased", detail: "潜空间双线性抗锯齿" },
  { id: "latent-bicubic", label: "Latent · Bicubic", detail: "潜空间双三次" },
  { id: "latent-bicubic-antialias", label: "Latent · Bicubic antialiased", detail: "潜空间双三次抗锯齿" },
  { id: "latent-nearest", label: "Latent · Nearest", detail: "潜空间最近邻" },
  { id: "latent-nearest-exact", label: "Latent · Nearest-exact", detail: "潜空间精确最近邻" },
]);
export const LATENT_ALGORITHM_IDS = LATENT_ALGORITHMS.map((item) => item.id);

// ComfyUI's `ImageScale` method list, plus Forge's pixel-space `Nearest`. The ids are the
// contract with the backend, the labels deliberately distinguish Forge PIL `nearest` from the
// older Torch `nearest-exact` id.
export const RESIZE_ALGORITHMS = Object.freeze([
  { id: "lanczos", label: "Lanczos", detail: "锐利，放大首选" },
  { id: "nearest", label: "Nearest · Forge PIL", detail: "Forge PIL 硬边像素" },
  { id: "nearest-exact", label: "Nearest-exact · Torch", detail: "Torch 精确最近邻（保留旧 id）" },
  { id: "bilinear", label: "Bilinear", detail: "最柔和" },
  { id: "bicubic", label: "Bicubic", detail: "平滑，过渡自然" },
  { id: "area", label: "Area", detail: "缩小时保留均值" },
]);
export const RESIZE_ALGORITHM_IDS = RESIZE_ALGORITHMS.map((item) => item.id);

export function isRelativeUpscalerId(value) {
  if (typeof value !== "string" || !value.trim()) return false;
  const normalized = value.replaceAll("\\", "/");
  return normalized === value
    && !normalized.startsWith("/")
    && !/^[A-Za-z]:\//.test(normalized)
    && !normalized.split("/").some((part) => !part || part === "." || part === "..");
}

// What each engine's latent grid accepts as a redraw canvas, mirroring the backend table. A unit
// that does not redraw is not held to any of them.
const ENGINE_PIXEL_ALIGNMENT = { SD: 8, iL: 8, Anima: 32, Flux: 16, Flux2: 16, Krea2: 16, MOSS: 16 };
const ENGINE_MAX_REFINEMENT_EDGE = { SD: 8192, iL: 8192, Anima: 4096, Flux: 4096, Flux2: 4096, Krea2: 4096, MOSS: 4096 };

export const RESAMPLE_UNIT_DEFAULTS = Object.freeze({
  type: "resample",
  enabled: false,
  expanded: false,
  name: "",
  // "" is "no model": a unit is useful without one, and the picker carries an explicit entry for it.
  upscaler: "",
  // null is the compatibility-safe pixel/default state. Only createResampleUnit(..., { engine })
  // for a user-added SD/iL unit opts into the latent default.
  latentAlgorithm: null,
  scale: 1.5,
  resizeAlgorithm: "lanczos",
  denoise: 0.35,
  // null inherits the request's own. A unit added without being configured then runs the settings
  // the picture was sampled with rather than a second set of defaults nobody chose.
  steps: null,
  cfg: null,
  sampler: null,
  scheduler: null,
  seedMode: "inherit",
  seed: "",
  tileSize: 192,
  tileOverlap: 16,
  executionMode: "full_frame",
  tileWidth: "auto",
  tileHeight: "auto",
  padding: 32,
  maskBlur: 8,
  seamMode: "none",
  uniformTiles: true,
  tiledDecode: true,
  exactOutput: false,
});

export const RTX_STAGE_DEFAULTS = Object.freeze({ type: "rtx", enabled: false, expanded: false, scale: 2, quality: "ultra" });
const RTX_QUALITY_LEVELS = ["low", "medium", "high", "ultra"];

let sequence = 0;
/**
 * A React key for a stage the user just added. Never sent: the server identifies a stage by its
 * position in the chain.
 */
export function nextStageId(prefix = "stage") {
  sequence += 1;
  return `${prefix}-${Date.now().toString(36)}-${sequence.toString(36)}`;
}

/**
 * The key a stage read from storage gets when it has none of its own.
 *
 * Derived from its type and its position rather than from a counter, so normalising the same record
 * twice produces the same chain. Two default records are then equal to each other, which is what
 * the per-engine store compares them as.
 */
function storedStageId(type, index) {
  return `${type}-${index + 1}`;
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function inRange(value, fallback, minimum, maximum) {
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(maximum, Math.max(minimum, numeric));
}

function optionalNumber(value, minimum, maximum, { integer = false } = {}) {
  if (value === null || value === undefined || value === "") return null;
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value));
  if (!Number.isFinite(numeric)) return null;
  const bounded = Math.min(maximum, Math.max(minimum, numeric));
  return integer ? Math.round(bounded) : bounded;
}

function engineSupportsLatent(engine) {
  return engine === "SD" || engine === "iL";
}

export function createResampleUnit(overrides = {}, { engine = "" } = {}) {
  const source = isPlainObject(overrides) ? overrides : {};
  const newEngineDefault = engineSupportsLatent(engine)
    && source.latentAlgorithm === undefined
    && source.resizeAlgorithm === undefined
    && !source.upscaler;
  return {
    ...RESAMPLE_UNIT_DEFAULTS,
    ...(newEngineDefault ? { latentAlgorithm: "latent" } : {}),
    id: nextStageId("resample"),
    enabled: true,
    expanded: true,
    ...source,
  };
}

export function createAdetailerStage(overrides = {}) {
  return normalizeAdetailerChainStage({ enabled: false, expanded: false, ...overrides });
}

export function createRtxStage(overrides = {}) {
  return normalizeRtxStage(overrides);
}

export function normalizeResampleUnit(value, { samplers = [], schedulers = [], engine = "SD", index = 0 } = {}) {
  const source = isPlainObject(value) ? value : {};
  const tileSize = Math.round(inRange(source.tileSize, 192, 32, 2048));
  const tileOverlap = Math.min(Math.round(inRange(source.tileOverlap, 16, 0, 512)), Math.floor(tileSize / 2));
  // The seed mode is kept as chosen even while its number is empty: this runs on every render, so
  // repairing "fixed with no number" here would undo the mode before a number could be typed. The
  // empty number is a run blocker instead, and the payload falls back to `inherit`.
  const seedMode = ["inherit", "fixed", "random"].includes(source.seedMode) ? source.seedMode : "inherit";
  // Keep the user's mode even while an engine cannot execute it. An engine switch must not
  // silently rewrite an enabled unit (the issue is reported below) or destroy a disabled unit's
  // metadata. The request is blocked for an enabled incompatible unit before it can reach the
  // backend; disabled units are intentionally carried through untouched.
  const executionMode = source.executionMode === "usdu_tiled" ? "usdu_tiled" : "full_frame";
  const scale = Math.round(inRange(source.scale, RESAMPLE_UNIT_DEFAULTS.scale, 1, 4) * 10) / 10;
  const denoise = Math.round(inRange(source.denoise, RESAMPLE_UNIT_DEFAULTS.denoise, 0, 1) * 100) / 100;
  return {
    ...RESAMPLE_UNIT_DEFAULTS,
    id: typeof source.id === "string" && source.id ? source.id : storedStageId("resample", index),
    enabled: source.enabled === true,
    expanded: source.expanded === true,
    name: typeof source.name === "string" ? source.name.slice(0, 80) : "",
    upscaler: typeof source.upscaler === "string" ? source.upscaler : "",
    // Missing is intentionally null. Normalising a saved record must never turn it into the new
    // user-add default, otherwise opening an old Hires/workspace record changes its run.
    latentAlgorithm: LATENT_ALGORITHM_IDS.includes(source.latentAlgorithm) ? source.latentAlgorithm : null,
    scale,
    resizeAlgorithm: RESIZE_ALGORITHM_IDS.includes(source.resizeAlgorithm) ? source.resizeAlgorithm : "lanczos",
    denoise,
    steps: optionalNumber(source.steps, 1, 100, { integer: true }),
    cfg: optionalNumber(source.cfg, 0, 30),
    sampler: samplers.includes(source.sampler) ? source.sampler : null,
    scheduler: schedulers.includes(source.scheduler) ? source.scheduler : null,
    seedMode,
    seed: seedMode === "fixed" ? normalizeUint64Seed(source.seed) ?? "" : "",
    tileSize,
    tileOverlap,
    executionMode,
    padding: Math.round(inRange(source.padding, 32, 0, 256)),
    maskBlur: Math.round(inRange(source.maskBlur, 8, 0, 64)),
    uniformTiles: source.uniformTiles !== false,
    tiledDecode: source.tiledDecode !== false,
    exactOutput: source.exactOutput === true,
  };
}

function normalizeRtxStage(value) {
  const source = isPlainObject(value) ? value : {};
  return {
    ...RTX_STAGE_DEFAULTS,
    id: typeof source.id === "string" && source.id ? source.id : storedStageId("rtx", 0),
    enabled: source.enabled === true,
    expanded: source.expanded === true,
    scale: Math.round(inRange(source.scale, 2, 1, 4) * 100) / 100,
    quality: RTX_QUALITY_LEVELS.includes(source.quality) ? source.quality : "ultra",
  };
}

function normalizeAdetailerChainStage(value) {
  const source = isPlainObject(value) ? value : {};
  return {
    ...normalizeADetailerStage(source),
    type: "adetailer",
    id: typeof source.id === "string" && source.id ? source.id : storedStageId("adetailer", 0),
    expanded: source.expanded === true,
  };
}

/**
 * A chain read from anywhere — a workspace file, a gallery card, a PNG parameter block — coerced to
 * what this build can run.
 *
 * Never throws, and always returns a chain that holds exactly one ADetailer stage and one RTX
 * stage: both are single-instance, and a saved chain that lost one still offers it, switched off,
 * rather than dropping a control the user can no longer reach.
 */
export function normalizeResampleChain(value, options = {}) {
  const source = Array.isArray(value) ? value : [];
  const stages = [];
  let resampleCount = 0;
  let sawAdetailer = false;
  let sawRtx = false;
  for (const entry of source) {
    if (!isPlainObject(entry)) continue;
    if (entry.type === "resample") {
      if (resampleCount >= RESAMPLE_UNIT_LIMIT) continue;
      stages.push(normalizeResampleUnit(entry, { ...options, index: resampleCount }));
      resampleCount += 1;
    } else if (entry.type === "adetailer" && !sawAdetailer) {
      sawAdetailer = true;
      stages.push(normalizeAdetailerChainStage(entry));
    } else if (entry.type === "rtx" && !sawRtx) {
      sawRtx = true;
      stages.push(normalizeRtxStage(entry));
    }
  }
  if (!sawAdetailer) stages.push(createAdetailerStage());
  if (!sawRtx) stages.push(createRtxStage());
  return stages;
}

/**
 * The chain a pre-chain record describes.
 *
 * Everything written before this feature holds `hires`, `adetailer`, `rtx` and `postprocessOrder`.
 * The Hires block becomes one resample unit field for field, the other two stages keep their own
 * settings, and the old order decides the order — so opening a saved workspace produces the run it
 * always produced. Idempotent: a record that already carries a chain is normalised, not migrated,
 * so re-reading it can never insert the unit twice.
 */
export function migrateResampleChain(record, options = {}) {
  const source = isPlainObject(record) ? record : {};
  if (Array.isArray(source.postprocessChain)) return normalizeResampleChain(source.postprocessChain, options);
  const freshDefaults = !isPlainObject(source.hires) && !isPlainObject(source.adetailer) && !isPlainObject(source.rtx);
  const hires = isPlainObject(source.hires) ? source.hires : {};
  const order = Array.isArray(source.postprocessOrder) && source.postprocessOrder.length
    ? source.postprocessOrder
    : ["hires", "adetailer", "rtx"];
  const built = {
    hires: normalizeResampleUnit({
      ...hires,
      // The old block called the model `model` and always had one; the unit calls it `upscaler` and
      // may have none.
      upscaler: typeof hires.model === "string" ? hires.model : "",
      // An old Hires pass always redrew, so a stored `denoise` of 0 was never a resize-only unit.
      denoise: Number.isFinite(Number(hires.denoise)) && Number(hires.denoise) > 0 ? hires.denoise : 0.35,
      steps: hires.steps ?? null,
      cfg: hires.cfg ?? null,
      name: "",
      ...(freshDefaults && engineSupportsLatent(options.engine) ? { latentAlgorithm: "latent" } : {}),
    }, options),
    adetailer: normalizeAdetailerChainStage(source.adetailer),
    rtx: normalizeRtxStage(source.rtx),
  };
  const seen = new Set();
  const chain = [];
  for (const stage of [...order, "hires", "adetailer", "rtx"]) {
    if (seen.has(stage) || !built[stage]) continue;
    seen.add(stage);
    chain.push(built[stage]);
  }
  return chain;
}

export function resampleUnits(chain) {
  return (Array.isArray(chain) ? chain : []).filter((stage) => stage?.type === "resample");
}

export function chainStage(chain, type) {
  return (Array.isArray(chain) ? chain : []).find((stage) => stage?.type === type) || null;
}

export function enabledChainStages(chain) {
  return (Array.isArray(chain) ? chain : []).filter((stage) => stage?.enabled === true);
}

/** Which card a unit is, 1-based and counting the switched-off ones — the number the user sees. */
export function resampleUnitNumber(chain, unit) {
  return resampleUnits(chain).findIndex((entry) => entry.id === unit.id) + 1;
}

export function resampleUnitTitle(chain, unit) {
  const number = resampleUnitNumber(chain, unit);
  return unit.name?.trim() ? `重采样 ${number} · ${unit.name.trim()}` : `重采样 ${number}`;
}

/** What a unit actually does, in the words of the behaviour matrix. */
export function resampleUnitSummary(unit) {
  if (!unit.enabled) return "关闭";
  const parts = [`${unit.scale.toFixed(1)}×`];
  parts.push(unit.upscaler ? "超分模型" : "算法缩放");
  parts.push(resampleAlgorithmLabel(unit));
  parts.push(unit.denoise > 0 ? `重绘 ${unit.denoise.toFixed(2)}` : "不重绘");
  return parts.join(" · ");
}

export function resampleAlgorithmLabel(unit) {
  if (unit?.latentAlgorithm) return LATENT_ALGORITHMS.find((item) => item.id === unit.latentAlgorithm)?.label || unit.latentAlgorithm;
  return RESIZE_ALGORITHMS.find((item) => item.id === unit?.resizeAlgorithm)?.label || unit?.resizeAlgorithm || "Lanczos";
}

export function resampleAlgorithmSelectionUpdate(value) {
  return LATENT_ALGORITHM_IDS.includes(value)
    ? { latentAlgorithm: value, upscaler: "" }
    : { resizeAlgorithm: value, latentAlgorithm: null };
}

export function resampleModelSelectionUpdate(value) {
  return value ? { upscaler: value, latentAlgorithm: null } : { upscaler: "" };
}

export function enginePixelAlignment(engine) {
  return ENGINE_PIXEL_ALIGNMENT[engine] || 8;
}

export function engineMaxRefinementEdge(engine) {
  return ENGINE_MAX_REFINEMENT_EDGE[engine] || 8192;
}

function alignEdge(edge, alignment) {
  return Math.max(alignment, Math.round(edge / alignment) * alignment);
}

/** `round(edge × scale)` — ComfyUI's `ImageScaleBy`, half-to-even like Python's `round`. */
export function scaledSize(size, scale) {
  return {
    width: Math.max(1, bankersRound(size.width * scale)),
    height: Math.max(1, bankersRound(size.height * scale)),
  };
}

/** The three sizes one unit deals in, mirroring the backend's `resample_unit_sizes`. */
export function resampleUnitSizes(size, unit, engine) {
  const target = scaledSize(size, unit.scale);
  if (!(unit.denoise > 0)) return { target, redraw: target, output: target };
  const alignment = enginePixelAlignment(engine);
  const redraw = { width: alignEdge(target.width, alignment), height: alignEdge(target.height, alignment) };
  return { target, redraw, output: unit.exactOutput ? target : redraw };
}

function rtxTargetSize(size, scale) {
  return {
    width: Math.max(8, bankersRound(Math.trunc(size.width * scale) / 8) * 8),
    height: Math.max(8, bankersRound(Math.trunc(size.height * scale) / 8) * 8),
  };
}

function withinEnvelope(size) {
  return Number.isFinite(size.width) && Number.isFinite(size.height)
    && Math.max(size.width, size.height) <= MAX_RTX_OUTPUT_EDGE
    && size.width * size.height <= MAX_RTX_OUTPUT_PIXELS;
}

/**
 * Walk the enabled chain from `size`, reporting every stage's input and output.
 *
 * The same walk the backend makes, so the sizes the cards show are the sizes the run will produce —
 * including the moment an engine's grid moves a unit off its exact target, which is reported rather
 * than hidden.
 */
export function resampleChainPlan(size, chain, engine) {
  let current = { width: size?.width || 0, height: size?.height || 0 };
  const trace = [];
  let valid = true;
  let reason = "";
  const units = resampleUnits(chain);

  for (const stage of enabledChainStages(chain)) {
    const input = { ...current };
    const entry = { type: stage.type, id: stage.id, input, unitNumber: 0 };
    if (stage.type === "resample") {
      const sizes = resampleUnitSizes(current, stage, engine);
      current = sizes.output;
      Object.assign(entry, {
        unitNumber: units.findIndex((unit) => unit.id === stage.id) + 1,
        target: sizes.target,
        redraw: sizes.redraw,
        output: sizes.output,
        redraws: stage.denoise > 0,
        // The one fact a user cannot work out from the numbers on the card.
        rounded: sizes.target.width !== sizes.redraw.width || sizes.target.height !== sizes.redraw.height,
      });
    } else if (stage.type === "rtx") {
      current = rtxTargetSize(current, stage.scale);
      entry.output = current;
    } else {
      entry.output = current;
    }
    const overEnvelope = !withinEnvelope(current);
    const overEngine = stage.type === "resample" && entry.redraws
      && Math.max(entry.redraw.width, entry.redraw.height) > engineMaxRefinementEdge(engine);
    entry.valid = !overEnvelope && !overEngine;
    entry.reason = overEngine
      ? `重采样 ${entry.unitNumber} 的重绘画布超出 ${engine} 的 ${engineMaxRefinementEdge(engine)} 上限`
      : overEnvelope ? RTX_TARGET_LIMIT_REASON : "";
    if (!entry.valid && valid) {
      valid = false;
      reason = entry.reason;
    }
    trace.push(entry);
  }

  return { ...current, trace, valid, reason };
}

/** `1024 → 1536 → 1920`, the one line that says what the whole chain does to the picture. */
export function chainSizeLine(size, chain, engine) {
  const plan = resampleChainPlan(size, chain, engine);
  const steps = [`${size.width} × ${size.height}`];
  for (const entry of plan.trace) steps.push(`${entry.output.width} × ${entry.output.height}`);
  return steps.join(" → ");
}

export function chainSummary(chain) {
  const enabled = enabledChainStages(chain);
  if (!enabled.length) return "未启用任何阶段";
  return enabled.map((stage) => {
    if (stage.type === "resample") return `${stage.scale.toFixed(1)}×`;
    if (stage.type === "adetailer") return adetailerSummary(stage);
    return `RTX ${stage.scale}×`;
  }).join(" · ");
}

/**
 * How many denoise updates one unit's redraw performs, which is what "有效步数" promises.
 *
 * Every redraw executes the requested count. Denoise selects the schedule's starting noise; SD/iL
 * expand the Diffusers schedule separately so it no longer shrinks this value.
 */
export function resampleUnitEffectiveSteps(unit, baseSteps) {
  if (!(unit.denoise > 0)) return 0;
  const steps = Number(unit.steps ?? baseSteps ?? 0);
  return Number.isFinite(steps) ? Math.floor(steps) : 0;
}

/** Why the chain cannot run, in the order the user can act on it, or "" when it can. */
export function resampleChainIssue(chain, { baseSteps, engine, nativeEngines = [], upscalers = null, size = null } = {}) {
  for (const unit of resampleUnits(chain)) {
    const number = resampleUnitNumber(chain, unit);
    // Disabled units remain in the payload and metadata, but have no execution semantics until
    // they are enabled again. This also keeps an old/engine-switched workspace editable.
    if (!unit.enabled) continue;
    const algorithmIssue = resampleUnitAlgorithmIssue(unit, engine);
    if (algorithmIssue) return `重采样 ${number}：${algorithmIssue}`;
    if (unit.upscaler && !isRelativeUpscalerId(unit.upscaler)) return `重采样 ${number}：超分模型必须是模型目录内的相对 ID`;
    if (unit.upscaler && upscalers != null) {
      const models = Array.isArray(upscalers) ? upscalers : Array.isArray(upscalers.models) ? upscalers.models : [];
      const model = models.find((item) => item.id === unit.upscaler || item.local_id === unit.upscaler);
      if (!model) return `重采样 ${number}：超分模型不可用：“${unit.upscaler}”不在当前可用目录中`;
      if (model.compatible === false) return `重采样 ${number}：超分模型不可用：“${model.label || unit.upscaler}”不兼容：${model.reason || "当前运行时不支持"}`;
    }
    if (unit.seedMode === "fixed" && !unit.seed) return `重采样 ${number} 的固定 Seed 必须是 0 ～ 18446744073709551615`;
    if (unit.tileOverlap > Math.floor(unit.tileSize / 2)) return `重采样 ${number} 的分块重叠不能超过分块尺寸的一半`;
    const effectiveSteps = resampleUnitEffectiveSteps(unit, baseSteps);
    if (unit.denoise > 0 && effectiveSteps < 1) {
      return `重采样 ${number} 的有效步数至少需要 1 步`;
    }
    if (unit.denoise > 0 && !nativeEngines.includes(engine)
      && diffusersScheduleTotal(effectiveSteps, unit.denoise) === null) {
      return `重采样 ${number} 的重绘强度过低：调度总步数不能超过 ${DIFFUSERS_SCHEDULE_STEP_LIMIT}，请提高重绘强度或降低步数`;
    }
  }
  if (size) {
    const plan = resampleChainPlan(size, chain, engine);
    if (!plan.valid) return plan.reason;
  }
  return "";
}

/** The three user-facing latent constraints, kept separate so panels and request blockers agree. */
export function resampleUnitAlgorithmIssue(unit, engine) {
  if (!unit) return "";
  if (unit.latentAlgorithm) {
    if (!engineSupportsLatent(engine)) return "潜空间算法仅支持 SD / iL 引擎";
    if (unit.executionMode !== "full_frame") return "潜空间算法仅支持整图重绘（full_frame）";
    if (unit.upscaler) return "潜空间算法不能与超分模型同时使用";
  }
  if (unit.executionMode === "usdu_tiled" && !supportsUsduTiled(engine)) return "USDU 分块重绘仅支持 Anima / Krea2 引擎";
  if (unit.executionMode === "usdu_tiled" && !(unit.denoise > 0)) return "USDU 分块重绘需要大于 0 的重绘强度";
  return "";
}

/** The `postprocess_chain` a request carries: stage order, without the ids the browser needs. */
export function resampleChainPayload(chain, { engine, seedOverrides = null } = {}) {
  return (Array.isArray(chain) ? chain : []).map((stage) => {
    if (stage.type === "adetailer") return { type: "adetailer", ...adetailerPayload(stage, engine) };
    if (stage.type === "rtx") return { type: "rtx", enabled: stage.enabled === true, scale: stage.scale, quality: stage.quality };
    // A run started from the gallery can pin each unit's seed to the one an output recorded; an
    // ordinary run sends the unit's own mode.
    const override = isPlainObject(seedOverrides) ? seedOverrides[stage.id] : null;
    const seedMode = override?.seedMode ?? stage.seedMode;
    const seed = override?.seed ?? stage.seed;
    const normalizedSeed = normalizeUint64Seed(seed);
    const fixed = seedMode === "fixed" && normalizedSeed !== null;
    const preserveDisabledFixedSeed = stage.enabled !== true && seedMode === "fixed";
    return {
      type: "resample",
      enabled: stage.enabled === true,
      name: stage.name || "",
      upscaler: stage.upscaler || null,
      ...(stage.latentAlgorithm ? { latent_algorithm: stage.latentAlgorithm } : {}),
      scale: stage.scale,
      resize_algorithm: stage.resizeAlgorithm,
      denoise: stage.denoise,
      steps: stage.steps ?? null,
      cfg: stage.cfg ?? null,
      sampler: stage.sampler || null,
      scheduler: stage.scheduler || null,
      seed_mode: fixed || preserveDisabledFixedSeed ? "fixed" : seedMode === "random" ? "random" : "inherit",
      ...(fixed ? { seed: normalizedSeed } : preserveDisabledFixedSeed ? { seed: seed ?? "" } : {}),
      tile_size: stage.tileSize,
      tile_overlap: stage.tileOverlap,
      // Do not coerce a disabled unit's execution mode during an engine switch. The backend skips
      // disabled stages, while metadata still needs to describe the complete saved unit.
      execution_mode: stage.executionMode === "usdu_tiled" ? "usdu_tiled" : "full_frame",
      tile_width: "auto",
      tile_height: "auto",
      padding: stage.padding,
      mask_blur: stage.maskBlur,
      seam_mode: "none",
      uniform_tiles: stage.uniformTiles !== false,
      tiled_decode: stage.tiledDecode !== false,
      exact_output: stage.exactOutput === true,
    };
  });
}

/** Add, remove and reorder. Each returns a new chain; none mutates the one it was given. */
export function addResampleUnit(chain, overrides = {}, options = {}) {
  if (resampleUnits(chain).length >= RESAMPLE_UNIT_LIMIT) return chain;
  const unit = createResampleUnit(overrides, options);
  const lastResample = chain.map((stage) => stage.type).lastIndexOf("resample");
  if (lastResample < 0) return [unit, ...chain];
  return [...chain.slice(0, lastResample + 1), unit, ...chain.slice(lastResample + 1)];
}

export function removeChainStage(chain, id) {
  const stage = chain.find((entry) => entry.id === id);
  // ADetailer and RTX are single-instance and always present: removing one would take a control off
  // the page with no way back. Only resample units are added and removed.
  if (!stage || stage.type !== "resample") return chain;
  return chain.filter((entry) => entry.id !== id);
}

export function moveChainStage(chain, id, offset) {
  const index = chain.findIndex((entry) => entry.id === id);
  const target = index + offset;
  if (index < 0 || target < 0 || target >= chain.length) return chain;
  const next = [...chain];
  const [stage] = next.splice(index, 1);
  next.splice(target, 0, stage);
  return next;
}

export function updateChainStage(chain, id, updates) {
  return chain.map((stage) => (stage.id === id ? { ...stage, ...updates } : stage));
}

export { activeADetailerUnits };
