// MOSS-Image (the seventh engine) as the browser sees it: defaults, the versioned `moss` sampling
// block, and the sampler vocabulary. Pure — App, the engine-settings store, XYZ and the image reader
// all read the same rules, and `backend/inference_server.py::MossSamplingInput` is the authority
// they mirror (`backend/tests/test_moss_integration.py` and `scripts/moss-engine.test.mjs` pin both).

export const MOSS_ENGINE = "MOSS";
export const MOSS_DIT_DEPTH = 42;
// ComfyUI's exact k-diffusion updates only; every other KSampler name is refused, not approximated.
export const MOSS_SAMPLERS = Object.freeze(["euler", "euler_ancestral", "heun"]);
export const MOSS_SCHEDULERS = Object.freeze([
  "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal",
]);
export const MOSS_SHIFT_RANGE = Object.freeze({ min: 0.01, max: 100 });
// The official workflow: 1024 x 1024, 50 steps, CFG 3, shift 5, euler / sgm_uniform, the
// unconditional forward skipping block 22 for the whole schedule, and an empty negative prompt.
export const MOSS_DEFAULTS = Object.freeze({
  steps: 50,
  cfg: 3,
  sampler: "euler",
  scheduler: "sgm_uniform",
  negative: "",
  size: Object.freeze({ width: 1024, height: 1024 }),
});
export const MOSS_SAMPLING_DEFAULTS = Object.freeze({
  shift: 5,
  uncondSkipLayers: Object.freeze([22]),
  uncondSkipStart: 0,
  uncondSkipEnd: 1,
});

export function isMossEngine(engine) {
  return engine === MOSS_ENGINE;
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function finite(value, fallback, minimum, maximum) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(maximum, Math.max(minimum, number));
}

/** Parse "22, 24" into unique in-range block indices; anything else is reported, never dropped. */
export function parseSkipLayers(text) {
  const source = String(text ?? "").trim();
  if (!source) return { layers: [], error: "" };
  const parts = source.split(/[\s,，、]+/).filter(Boolean);
  const layers = [];
  for (const part of parts) {
    if (!/^\d+$/.test(part)) return { layers: [], error: `“${part}”不是区块编号` };
    const index = Number(part);
    if (index >= MOSS_DIT_DEPTH) return { layers: [], error: `区块编号需在 0～${MOSS_DIT_DEPTH - 1} 之间` };
    if (layers.includes(index)) return { layers: [], error: `区块 ${index} 重复` };
    layers.push(index);
  }
  return { layers, error: "" };
}

export function formatSkipLayers(layers) {
  return (Array.isArray(layers) ? layers : []).join(", ");
}

/** A stored or typed MOSS sampling block, coerced to what the server accepts. */
export function normalizeMossSampling(value) {
  const source = isPlainObject(value) ? value : {};
  const layers = Array.isArray(source.uncondSkipLayers)
    ? source.uncondSkipLayers.filter((layer, index, all) => Number.isInteger(layer) && layer >= 0 && layer < MOSS_DIT_DEPTH && all.indexOf(layer) === index)
    : [...MOSS_SAMPLING_DEFAULTS.uncondSkipLayers];
  let start = finite(source.uncondSkipStart, MOSS_SAMPLING_DEFAULTS.uncondSkipStart, 0, 1);
  let end = finite(source.uncondSkipEnd, MOSS_SAMPLING_DEFAULTS.uncondSkipEnd, 0, 1);
  if (start > end) [start, end] = [MOSS_SAMPLING_DEFAULTS.uncondSkipStart, MOSS_SAMPLING_DEFAULTS.uncondSkipEnd];
  return {
    shift: Math.round(finite(source.shift, MOSS_SAMPLING_DEFAULTS.shift, MOSS_SHIFT_RANGE.min, MOSS_SHIFT_RANGE.max) * 1000) / 1000,
    uncondSkipLayers: layers,
    uncondSkipStart: start,
    uncondSkipEnd: end,
  };
}

/** The wire block (`GenerateInput.moss`, version 1). Standard SLG is never sent: it is not offered. */
export function mossSamplingPayload(value) {
  const settings = normalizeMossSampling(value);
  return {
    version: 1,
    shift: settings.shift,
    uncond_skip_layers: [...settings.uncondSkipLayers],
    uncond_skip_start: settings.uncondSkipStart,
    uncond_skip_end: settings.uncondSkipEnd,
  };
}

/** The wire block read back (a job, a PNG record or an XYZ baseline) into the editor's shape. */
export function mossSamplingFromPayload(value) {
  if (!isPlainObject(value)) return normalizeMossSampling(null);
  return normalizeMossSampling({
    shift: value.shift,
    uncondSkipLayers: value.uncond_skip_layers,
    uncondSkipStart: value.uncond_skip_start,
    uncondSkipEnd: value.uncond_skip_end,
  });
}

/** The request fragment a MOSS job adds; other engines add nothing. */
export function mossRequestFields(engine, value) {
  return isMossEngine(engine) ? { moss: mossSamplingPayload(value) } : {};
}

export function mossSamplerFor(sampler) {
  return MOSS_SAMPLERS.includes(sampler) ? sampler : MOSS_DEFAULTS.sampler;
}

export function mossSchedulerFor(scheduler) {
  return MOSS_SCHEDULERS.includes(scheduler) ? scheduler : MOSS_DEFAULTS.scheduler;
}

/** One-line description of what the block does, for the sampling panel and job details. */
export function describeMossSampling(value, cfg) {
  const settings = normalizeMossSampling(value);
  const skip = Number(cfg) === 1
    ? "CFG 为 1：无负向分支，跳层不生效"
    : settings.uncondSkipLayers.length
      ? `负向分支跳过区块 ${formatSkipLayers(settings.uncondSkipLayers)}（${settings.uncondSkipStart}～${settings.uncondSkipEnd}）`
      : "负向分支不跳层";
  return `Shift ${settings.shift} · ${skip}`;
}
