// Both native engines execute every requested refinement step, so the list of which engines those
// are is shared with the ADetailer accounting rather than repeated here.
import { NATIVE_STEP_ENGINES } from "./adetailer-units.js";

export const MAX_UINT64_SEED = 18446744073709551615n;

export function secureRandomUint64Seed(cryptoSource = globalThis.crypto) {
  if (!cryptoSource?.getRandomValues) throw new Error("Secure random seed generation is unavailable");
  const words = cryptoSource.getRandomValues(new Uint32Array(2));
  return ((BigInt(words[0]) << 32n) | BigInt(words[1])).toString();
}

export function normalizeUint64Seed(value, fallback = null) {
  let text;
  if (typeof value === "string") text = value;
  else if (typeof value === "bigint") text = value.toString();
  else if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) text = String(value);
  else return fallback;
  if (!/^[0-9]+$/.test(text)) return fallback;
  try {
    const parsed = BigInt(text);
    return parsed <= MAX_UINT64_SEED ? parsed.toString() : fallback;
  } catch {
    return fallback;
  }
}

export function normalizeHiresSeed(seedMode, seed) {
  if (seedMode === "fixed") {
    const normalized = normalizeUint64Seed(seed);
    return normalized === null ? { seedMode: "inherit", seed: "" } : { seedMode: "fixed", seed: normalized };
  }
  return seedMode === "random"
    ? { seedMode: "random", seed: "" }
    : { seedMode: "inherit", seed: "" };
}

export function generationHiresSeedSettings(hires) {
  return hires?.seedMode === "fixed"
    ? { seedMode: "fixed", seed: normalizeUint64Seed(hires.seed, "0") }
    : normalizeHiresSeed(hires?.seedMode, null);
}

// How many denoise updates the Hires refinement pass actually performs, which is
// what "有效步数" promises and what the run blocker is decided on.
//
// The same family split the backend makes in `_apply_hires_fix_stages`: Diffusers
// image-to-image runs `int(steps × denoise)` of them, while the native engines follow
// Comfy — a longer schedule with the last `steps + 1` sigmas kept — and so run
// every requested step whatever the denoise. Multiplying there refused a
// configuration the run would have executed happily.
export function hiresEffectiveSteps(hires, engine) {
  const steps = Number(hires?.steps || 0);
  if (NATIVE_STEP_ENGINES.includes(engine)) return Math.floor(steps);
  return Math.floor(steps * Number(hires?.denoise || 0));
}

export function hiresSeedPayload(hiresSeed) {
  const seedMode = hiresSeed?.seedMode;
  if (seedMode !== "fixed") return { mode: seedMode === "random" ? "random" : "inherit" };
  const seed = normalizeUint64Seed(hiresSeed.seed);
  return seed === null ? { mode: "inherit" } : { mode: "fixed", seed };
}

export function resolvedGalleryOutputHiresSeed(output, fallbackHires) {
  const outputMode = ["inherit", "fixed", "random"].includes(output?.hires_seed_mode)
    ? output.hires_seed_mode
    : fallbackHires?.seedMode;
  const effective = normalizeUint64Seed(output?.hires_seed);
  if (outputMode === "inherit") return { seedMode: "inherit", seed: "" };
  if (effective !== null) return { seedMode: "fixed", seed: effective };
  return normalizeHiresSeed(outputMode, fallbackHires?.seed);
}

export function galleryOutputSeedSettings(normalizedSettings, output, selectedItems, combined = false) {
  const items = Array.isArray(selectedItems) ? selectedItems : [];
  const hiresSeed = resolvedGalleryOutputHiresSeed(output, normalizedSettings.hires);
  const selectedHiresSeeds = items.map((item) => resolvedGalleryOutputHiresSeed(item, normalizedSettings.hires));
  return {
    ...normalizedSettings,
    seed: normalizeUint64Seed(output?.base_seed ?? output?.seed, normalizedSettings.seed),
    seedMode: "fixed",
    hires: { ...normalizedSettings.hires, ...hiresSeed },
    imagesPerBatch: combined ? items.length : 1,
    batchCount: 1,
    imageSeeds: items.map((item) => normalizeUint64Seed(item.base_seed ?? item.seed, "")),
    imageHiresSeedModes: selectedHiresSeeds.map((item) => item.seedMode),
    imageHiresSeeds: selectedHiresSeeds.map((item) => item.seed),
  };
}

export function normalizeGalleryHires(model, sourceHires, baseHires, defaults, { sourceKind = "persisted_card" } = {}) {
  const source = sourceHires && typeof sourceHires === "object" ? sourceHires : {};
  const base = baseHires && typeof baseHires === "object" ? baseHires : {};
  const { samplers = [], schedulers = [], ...fallback } = defaults && typeof defaults === "object" ? defaults : {};
  const hasSource = (key) => Object.prototype.hasOwnProperty.call(source, key);
  const value = (key, fallbackValue) => hasSource(key) ? source[key] : Object.prototype.hasOwnProperty.call(base, key) ? base[key] : fallbackValue;
  const tileDimension = (input) => input === "auto" || input === undefined || input === null ? "auto" : Number.isInteger(Number(input)) && Number(input) > 0 ? Number(input) : "auto";
  const number = (key, fallbackValue, minimum, maximum) => {
    const input = Number(value(key, fallbackValue));
    return Math.max(minimum, Math.min(maximum, Math.round(Number.isFinite(input) ? input : fallbackValue)));
  };
  const hasBase = (key) => Object.prototype.hasOwnProperty.call(base, key);
  const explicitExecutionMode = hasSource("executionMode")
    ? source.executionMode
    : sourceKind === "workspace_inheritance" && hasBase("executionMode") ? base.executionMode : undefined;
  const executionMode = model === "Anima" && explicitExecutionMode !== "full_frame" ? "usdu_tiled" : "full_frame";
  const hiresSeed = normalizeHiresSeed(value("seedMode", "inherit"), value("seed", null));
  return {
    ...fallback,
    ...base,
    ...source,
    executionMode,
    ...hiresSeed,
    sampler: samplers.includes(value("sampler")) ? value("sampler") : null,
    scheduler: schedulers.includes(value("scheduler")) ? value("scheduler") : null,
    tileWidth: tileDimension(value("tileWidth", "auto")),
    tileHeight: tileDimension(value("tileHeight", "auto")),
    padding: number("padding", 32, 0, 256),
    maskBlur: number("maskBlur", 8, 0, 64),
    seamMode: "none",
    uniformTiles: value("uniformTiles", true) !== false,
    tiledDecode: value("tiledDecode", true) !== false,
  };
}
