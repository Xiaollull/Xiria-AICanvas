// Both native engines execute every requested refinement step, so the list of which engines those
// are is shared with the ADetailer accounting rather than repeated here.
import { NATIVE_STEP_ENGINES } from "./adetailer-units.js";

export const MAX_UINT64_SEED = 18446744073709551615n;

// The engines whose Hires stage can refine tile by tile. Both prepare their conditioning and their
// sigma schedule once and hand the same pair to every tile; the Flux engines are guidance distilled
// and expose no such surface, so their Hires stays full frame. Kept here rather than written out at
// each of the seven places that ask, so the answer cannot drift between the workspace, the gallery
// and image-to-image.
export const USDU_TILED_ENGINES = ["Anima", "Krea2"];

export function supportsUsduTiled(model) {
  return USDU_TILED_ENGINES.includes(model);
}

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

/**
 * The seed a saved output's first redrawing unit actually ran with.
 *
 * An output records `hires_seed_mode` and `hires_seed` — the names the single Hires block used, and
 * still the first unit's — so reproducing that picture means pinning the unit to the number the run
 * resolved, not repeating the mode that produced a fresh one.
 */
export function resolvedGalleryOutputHiresSeed(output, fallbackUnit) {
  const outputMode = ["inherit", "fixed", "random"].includes(output?.hires_seed_mode)
    ? output.hires_seed_mode
    : fallbackUnit?.seedMode;
  const effective = normalizeUint64Seed(output?.hires_seed);
  if (outputMode === "inherit") return { seedMode: "inherit", seed: "" };
  if (effective !== null) return { seedMode: "fixed", seed: effective };
  return normalizeHiresSeed(outputMode, fallbackUnit?.seed);
}

export function galleryOutputSeedSettings(normalizedSettings, output, selectedItems, combined = false, firstUnit = null) {
  const items = Array.isArray(selectedItems) ? selectedItems : [];
  const selectedHiresSeeds = items.map((item) => resolvedGalleryOutputHiresSeed(item, firstUnit));
  const unitSeed = resolvedGalleryOutputHiresSeed(output, firstUnit);
  return {
    ...normalizedSettings,
    seed: normalizeUint64Seed(output?.base_seed ?? output?.seed, normalizedSettings.seed),
    seedMode: "fixed",
    // Which unit this belongs to is decided where the chain is in hand; here it is one pair.
    resampleSeed: unitSeed,
    imagesPerBatch: combined ? items.length : 1,
    batchCount: 1,
    imageSeeds: items.map((item) => normalizeUint64Seed(item.base_seed ?? item.seed, "")),
    imageHiresSeedModes: selectedHiresSeeds.map((item) => item.seedMode),
    imageHiresSeeds: selectedHiresSeeds.map((item) => item.seed),
  };
}
