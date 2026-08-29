// ADetailer as a sequence of passes rather than a single one.
//
// A detector answers exactly one question — "where are the faces" — and a
// picture usually has more than one thing worth repairing. Running a face model
// and a hand model in the same render is two detections, two masks and two sets
// of denoise / steps / prompt, because the settings that fix a face are the
// wrong settings for a hand. So a *unit* is the whole configuration, and the
// stage is an ordered list of them: each unit sees the picture the one before it
// produced, which is also why order is the user's to decide.
//
// Both generate surfaces and the gallery read this one shape, so the
// text-to-image page and the image-to-image page cannot drift apart.

export const ADETAILER_UNIT_LIMIT = 6;

export const ADETAILER_UNIT_DEFAULTS = Object.freeze({
  enabled: true,
  detector: "",
  confidence: 0.3,
  maxDetections: 3,
  maskMinRatio: 0,
  maskMaxRatio: 1,
  dilateErode: 4,
  maskBlur: 4,
  padding: 32,
  denoise: 0.4,
  useSteps: false,
  steps: 28,
  useCfg: false,
  cfg: 7,
  prompt: "",
  negativePrompt: "",
});

// Ids exist so a control belongs to a unit rather than to a position: a unit's
// slot number and its settings must not be able to come apart, and a half-typed
// number field must not be handed to a different unit.
//
// Normalising must be idempotent — the same stored settings have to normalise to
// the same object, or every equality check downstream reports a change that did
// not happen — so a *stored* unit keeps its id and a unit that has none is given
// the first free positional one.
function bounded(value, fallback, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function text(value, limit) {
  return typeof value === "string" ? value.slice(0, limit) : "";
}

export function normalizeADetailerUnit(source) {
  const entry = source && typeof source === "object" ? source : {};
  const first = bounded(entry.maskMinRatio, ADETAILER_UNIT_DEFAULTS.maskMinRatio, 0, 1);
  const second = bounded(entry.maskMaxRatio, ADETAILER_UNIT_DEFAULTS.maskMaxRatio, 0, 1);
  return {
    id: typeof entry.id === "string" ? entry.id : "",
    enabled: entry.enabled !== false,
    detector: text(entry.detector, 500),
    confidence: bounded(entry.confidence, ADETAILER_UNIT_DEFAULTS.confidence, 0.05, 1),
    maxDetections: Math.round(bounded(entry.maxDetections, ADETAILER_UNIT_DEFAULTS.maxDetections, 1, 8)),
    // A range that arrives inverted is a range, not an error: order it.
    maskMinRatio: Math.min(first, second),
    maskMaxRatio: Math.max(first, second),
    dilateErode: Math.round(bounded(entry.dilateErode, ADETAILER_UNIT_DEFAULTS.dilateErode, -128, 128)),
    maskBlur: Math.round(bounded(entry.maskBlur, ADETAILER_UNIT_DEFAULTS.maskBlur, 0, 64)),
    padding: Math.round(bounded(entry.padding, ADETAILER_UNIT_DEFAULTS.padding, 0, 256)),
    denoise: bounded(entry.denoise, ADETAILER_UNIT_DEFAULTS.denoise, 0.05, 1),
    useSteps: entry.useSteps === true,
    steps: Math.round(bounded(entry.steps, ADETAILER_UNIT_DEFAULTS.steps, 1, 100)),
    useCfg: entry.useCfg === true,
    cfg: bounded(entry.cfg, ADETAILER_UNIT_DEFAULTS.cfg, 0, 30),
    prompt: text(entry.prompt, 8000),
    negativePrompt: text(entry.negativePrompt, 8000),
  };
}

export function normalizeADetailerUnits(stage) {
  const source = stage && typeof stage === "object" ? stage : {};
  const raw = Array.isArray(source.units) && source.units.length
    ? source.units
    // Before the stage could run more than once it *was* the unit, so a stored
    // configuration from then is unit one. Its `enabled` belongs to the stage,
    // not to the unit, or migrating a switched-off stage would produce a unit
    // that stays switched off after the user turns the stage back on.
    : [{ ...source, enabled: true }];
  const entries = raw.slice(0, ADETAILER_UNIT_LIMIT);
  // Every slot exists from the start, so a second detail pass is a switch to turn
  // on rather than a unit to create — there is nothing to add and nothing to
  // delete. Slots past the stored ones arrive switched off, because enabling all
  // six by default would quietly buy six inpaint passes on every render.
  while (entries.length < ADETAILER_UNIT_LIMIT) entries.push({ ...ADETAILER_UNIT_DEFAULTS, enabled: false });
  // Two stored lists merged into one can repeat an id, and a duplicate React key
  // would hand one unit's draft input state to another.
  const taken = new Set();
  let position = 0;
  const firstFreeId = () => {
    do { position += 1; } while (taken.has(`ade-${position}`));
    return `ade-${position}`;
  };
  const units = entries.map(normalizeADetailerUnit);
  for (const unit of units) {
    if (unit.id && !taken.has(unit.id)) taken.add(unit.id);
  }
  return units.map((unit) => {
    if (unit.id && taken.has(unit.id) && units.filter((other) => other.id === unit.id).length === 1) return unit;
    const id = firstFreeId();
    taken.add(id);
    return { ...unit, id };
  });
}

export function normalizeADetailerStage(stage) {
  const source = stage && typeof stage === "object" ? stage : {};
  return {
    enabled: source.enabled === true,
    expanded: source.expanded === true,
    units: normalizeADetailerUnits(source),
  };
}

export function activeADetailerUnits(stage) {
  return normalizeADetailerUnits(stage).filter((unit) => unit.enabled);
}

// Diffusers runs `int(steps × denoise)` updates on an inpaint pass, so a low
// denoise with the inherited step count can round to nothing.
//
// The native engines do not. Their refinement follows Comfy — build a longer schedule,
// keep the last `steps + 1` sigmas — so it executes **every** requested step
// whatever the denoise. Multiplying there understated the pass, which is what let
// the job's step counter climb past its own total. The backend makes the same
// split in `adetailer_effective_steps`; the two must agree, because this number is
// both what the page promises and what the run blocker is decided on.
export const NATIVE_STEP_ENGINES = ["Anima", "Flux", "Flux2", "Krea2"];

export function adetailerUnitSteps(unit, baseSteps, engine) {
  const steps = Number((unit?.useSteps ? unit.steps : baseSteps) || 0);
  if (NATIVE_STEP_ENGINES.includes(engine)) return Math.floor(steps);
  return Math.floor(steps * Number(unit?.denoise || 0));
}

export function adetailerUnitLabel(index, total) {
  return total > 1 ? `ADetailer ${index + 1}` : "ADetailer";
}

// One unit is on screen at a time and the rest are pages behind it, so the page
// count is simply how many units there are — nothing declares it. The open page
// is an id rather than a position, for the same reason the controls are: a page
// is a unit, and the section is keyed by that id so switching pages replaces the
// controls instead of pointing the same ones at different settings.
export function adetailerPageIndex(units, activeId) {
  const list = Array.isArray(units) ? units : [];
  const index = list.findIndex((unit) => unit?.id === activeId);
  return index >= 0 ? index : 0;
}

export function adetailerStepUnitId(units, activeId, step) {
  const list = Array.isArray(units) ? units : [];
  if (!list.length) return "";
  const index = Math.min(list.length - 1, Math.max(0, adetailerPageIndex(list, activeId) + step));
  return list[index].id;
}

// Every unit that will actually run has to name an available detector and buy at
// least one step. Reporting which unit is at fault matters once there are
// several: "ADetailer is not ready" does not say which page to open.
//
// The number quoted is the unit's *slot*, which is what the pager shows. Counting
// only the enabled ones would call slot 4 "ADetailer 2" whenever the slots in
// between are switched off, sending the user to the wrong page.
export function adetailerStageIssue(stage, baseSteps, isDetectorAvailable, engine) {
  const units = normalizeADetailerUnits(stage);
  if (!units.some((unit) => unit.enabled)) return "至少需要启用一个 ADetailer 单元";
  for (const [index, unit] of units.entries()) {
    if (!unit.enabled) continue;
    const label = adetailerUnitLabel(index, units.length);
    if (!unit.detector) return `${label} 尚未选择检测模型`;
    if (typeof isDetectorAvailable === "function" && !isDetectorAvailable(unit.detector)) return `${label} 的检测模型不可用`;
    if (adetailerUnitSteps(unit, baseSteps, engine) < 1) return `${label} 的有效步数至少需要 1 步`;
  }
  return "";
}

// Engines with no unconditional branch: a negative prompt has nothing to be encoded into, so the
// server refuses one rather than accepting it and discarding it. The editor keeps whatever the user
// typed — only the request drops it — which is the same rule the main negative prompt follows.
export const DISTILLED_GUIDANCE_ENGINES = ["Flux", "Flux2"];

// Only the units that will run are sent: the wire carries the run plan, so a
// unit switched off in the editor has no representation on the other side.
export function adetailerUnitsPayload(stage, engine) {
  const carriesNegativePrompt = !DISTILLED_GUIDANCE_ENGINES.includes(engine);
  return activeADetailerUnits(stage).map((unit) => ({
    detector: unit.detector,
    confidence: unit.confidence,
    max_detections: unit.maxDetections,
    mask_min_ratio: unit.maskMinRatio,
    mask_max_ratio: unit.maskMaxRatio,
    dilate_erode: unit.dilateErode,
    mask_blur: unit.maskBlur,
    padding: unit.padding,
    denoise: unit.denoise,
    use_steps: unit.useSteps,
    steps: unit.steps,
    use_cfg: unit.useCfg,
    cfg: unit.cfg,
    prompt: unit.prompt.trim(),
    negative_prompt: carriesNegativePrompt ? unit.negativePrompt.trim() : "",
  }));
}

// A switched-off stage runs nothing, so nothing of it belongs on the wire. The editor always holds
// a first unit — every slot exists from the start — and that unit begins with no detector chosen,
// so sending the units of a disabled stage handed the request validator a unit that cannot run and
// lost the whole generation to a stage the user never turned on.
export function adetailerPayload(stage, engine) {
  const enabled = stage?.enabled === true;
  return { enabled, units: enabled ? adetailerUnitsPayload(stage, engine) : [] };
}

function detectorName(detector) {
  return detector ? String(detector).split("/").pop() : "未选择模型";
}

export function adetailerSummary(stage) {
  const units = activeADetailerUnits(stage);
  if (!units.length) return "未启用单元";
  if (units.length === 1) return `${detectorName(units[0].detector)} · ${units[0].confidence.toFixed(2)}`;
  return `${detectorName(units[0].detector)} 等 ${units.length} 个模型依次执行`;
}

// The steps the whole stage will add, which is what a progress estimate and a
// memory admission both need.
export function adetailerTotalSteps(stage, baseSteps, engine) {
  return activeADetailerUnits(stage).reduce((total, unit) => total + Math.max(0, adetailerUnitSteps(unit, baseSteps, engine)), 0);
}
