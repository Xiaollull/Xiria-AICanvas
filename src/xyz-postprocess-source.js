import { RESAMPLE_UNIT_DEFAULTS, RTX_STAGE_DEFAULTS } from "./resample-chain.js";
import { ADETAILER_UNIT_DEFAULTS, ADETAILER_UNIT_LIMIT } from "./adetailer-units.js";
import capabilities from "./xyz-capabilities.json" with { type: "json" };

const object = (v) => v && typeof v === "object" && !Array.isArray(v);
const clone = (v) => structuredClone(v);
const resampleFields = { latentAlgorithm: "latent_algorithm", resizeAlgorithm: "resize_algorithm", seedMode: "seed_mode", tileSize: "tile_size", tileOverlap: "tile_overlap", executionMode: "execution_mode", tileWidth: "tile_width", tileHeight: "tile_height", maskBlur: "mask_blur", seamMode: "seam_mode", uniformTiles: "uniform_tiles", tiledDecode: "tiled_decode", exactOutput: "exact_output" };
const adeFields = { maxDetections: "max_detections", maskMinRatio: "mask_min_ratio", maskMaxRatio: "mask_max_ratio", dilateErode: "dilate_erode", maskBlur: "mask_blur", useSteps: "use_steps", useCfg: "use_cfg", negativePrompt: "negative_prompt" };
const resampleKeys = ["enabled", "name", "upscaler", "scale", "denoise", "steps", "cfg", "sampler", "scheduler", "padding", ...Object.keys(resampleFields)];
const adeKeys = Object.keys(ADETAILER_UNIT_DEFAULTS).filter((key) => key !== "enabled");
const rename = (value, fields, keys) => Object.fromEntries(keys.map((key) => [fields[key] || key, value[key]]).filter(([, v]) => v !== undefined));
function fromWire(value, fields) { return Object.fromEntries(Object.entries(value).map(([key, v]) => [Object.keys(fields).find((name) => fields[name] === key) || key, v])); }
const unitWire = (unit, engine) => ({ ...Object.fromEntries(Object.entries(unit).filter(([key]) => !["id", "enabled", ...adeKeys].includes(key))), ...rename(unit, adeFields, adeKeys), ...(capabilities.engines[engine]?.negative_prompt === false ? { negative_prompt: "" } : {}) });
function mergeChanges(wire, before, after) {
  const result = clone(wire);
  for (const key of new Set([...Object.keys(before), ...Object.keys(after)])) {
    if (JSON.stringify(before[key]) === JSON.stringify(after[key])) continue;
    if (Object.hasOwn(after, key)) result[key] = clone(after[key]); else delete result[key];
  }
  return result;
}
function stageWire(stage, engine) {
  if (stage.type === "resample") return { type: "resample", ...rename(stage, resampleFields, resampleKeys), upscaler: stage.upscaler || null, ...(stage.seedMode === "fixed" ? { seed: String(stage.seed ?? "") } : {}) };
  if (stage.type === "rtx") return { type: "rtx", enabled: stage.enabled, scale: stage.scale, quality: stage.quality };
  return { type: "adetailer", enabled: stage.enabled, units: stage.units.filter((unit) => unit.enabled).map((unit) => unitWire(unit, engine)) };
}
export function validXYZPostprocessSource(source) {
  return object(source) && source.schema_version === 1 && Array.isArray(source.chain) && source.chain.every((stage) => object(stage) && typeof stage.id === "string" && (stage.legacy === true || ["resample", "adetailer", "rtx"].includes(stage.type) && (stage.type !== "adetailer" || Array.isArray(stage.units) && stage.units.every((unit) => object(unit) && typeof unit.id === "string")))) && new Set(source.chain.map((s) => s.id)).size === source.chain.length;
}
export function captureXYZPostprocessSource(baseline, chain) {
  return { ...baseline, postprocess_source: { schema_version: 1, chain: clone(chain) } };
}
// Legacy wire is never normalized or written on open. Positional UI ids are deterministic.
export function xyzPostprocessSource(draft) {
  if (draft.baseline.postprocess_source) return clone(draft.baseline.postprocess_source);
  const chain = draft.baseline.parameters.postprocess_chain.map((wire, index) => {
    if (wire.type === "resample") return { ...RESAMPLE_UNIT_DEFAULTS, ...fromWire(wire, resampleFields), id: `xyz-resample-${index + 1}`, expanded: false };
    if (wire.type === "rtx") return { ...RTX_STAGE_DEFAULTS, ...wire, id: `xyz-rtx-${index + 1}`, expanded: false };
    if (wire.type === "adetailer" && Array.isArray(wire.units)) return { ...wire, id: `xyz-adetailer-${index + 1}`, expanded: false, units: Array.from({ length: Math.max(wire.units.length, ADETAILER_UNIT_LIMIT) }, (_, i) => ({ ...ADETAILER_UNIT_DEFAULTS, ...(wire.units[i] ? fromWire(wire.units[i], adeFields) : { enabled: false }), id: `ade-${i + 1}` })) };
    return { ...wire, id: `xyz-legacy-${index + 1}`, legacy: true };
  });
  return { schema_version: 1, chain };
}
export function editXYZPostprocess(draft, chain) {
  const previous = xyzPostprocessSource(draft).chain, wires = draft.baseline.parameters.postprocess_chain;
  const next = clone(draft);
  next.baseline.parameters.postprocess_chain = chain.map((stage) => {
    const index = previous.findIndex((s) => s.id === stage.id), before = previous[index], wire = wires[index];
    if (!before || !wire) return stageWire(stage, draft.engine);
    if (stage.legacy) return clone(wire);
    const oldProjection = stageWire(before, draft.engine), projection = stageWire(stage, draft.engine);
    // Only fields explicitly edited change, including deleting a now-inherited seed. Unknown/raw
    // wire fields stay visible to authoritative validation instead of disappearing in an adapter.
    const result = mergeChanges(wire, oldProjection, projection);
    if (stage.type === "adetailer" && (JSON.stringify(oldProjection.units) !== JSON.stringify(projection.units) || stage.enabled && !wire.units?.length) && Array.isArray(wire.units)) {
      const previousUnits = before.units.filter((unit) => unit.enabled);
      result.units = stage.units.filter((unit) => unit.enabled).map((unit) => {
        const i = previousUnits.findIndex((u) => u.id === unit.id);
        return i >= 0 && wire.units[i] ? mergeChanges(wire.units[i], unitWire(previousUnits[i], draft.engine), unitWire(unit, draft.engine)) : unitWire(unit, draft.engine);
      });
    }
    return result;
  });
  next.baseline.postprocess_source = { schema_version: 1, chain: clone(chain) };
  return next;
}
