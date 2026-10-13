import capabilities from "./xyz-capabilities.json" with { type: "json" };
import { composeGenerationPrompt, togglePromptPresetId } from "./prompt-presets.js";
import { composeGroupPrompt } from "./lora-groups.js";

/** Optional draft-only provenance. Old strings are actual text: never infer/remove a prefix. */
export function xyzPromptSource(draft) {
  if (!draft?.baseline) return null;
  return draft.baseline.prompt_source || { schema_version: 1, mode: "legacy_actual",
    positive: draft.baseline.parameters.prompt, negative: draft.baseline.parameters.negative_prompt,
    active: { positive: [], negative: [] }, records: [], group_ids: [] };
}
export function validXYZPromptSource(source) {
  return source && source.schema_version === 1 && ["raw", "legacy_actual"].includes(source.mode)
    && typeof source.positive === "string" && typeof source.negative === "string"
    && ["positive", "negative"].every((type) => Array.isArray(source.active?.[type]) && source.active[type].every((id) => typeof id === "string"))
    && Array.isArray(source.records) && source.records.every((record) => record && typeof record.id === "string" && typeof record.content === "string")
    && Array.isArray(source.group_ids) && source.group_ids.every((id) => typeof id === "string");
}
function recompose(baseline, source) {
  const next = structuredClone(baseline);
  next.prompt_source = structuredClone(source);
  const groups = next.group_snapshots.filter((group) => source.group_ids.includes(group.id));
  const groupPrompt = composeGroupPrompt(groups, "");
  const compose = (type, prefix = "") => source.mode === "legacy_actual" && !prefix && !source.active[type].length
    ? source[type] : composeGenerationPrompt({ groupPrompt: prefix, records: source.records, activeIds: source.active[type], type, prompt: source[type] });
  next.prompt_without_groups = compose("positive");
  next.group_prompt = groupPrompt;
  next.parameters.prompt = compose("positive", groupPrompt);
  // Unsupported negatives remain visible in provenance. Preserve legacy actual bytes (no migration
  // guess); new captures send the ordinary native-engine empty negative branch.
  if (capabilities.engines[next.parameters.engine].negative_prompt) next.parameters.negative_prompt = compose("negative");
  return next;
}
export function captureXYZPromptSource(baseline, { positive, negative, active = {}, records = [] }) {
  const ids = { positive: [...(active.positive || [])], negative: [...(active.negative || [])] };
  return recompose(baseline, { schema_version: 1, mode: "raw", positive, negative, active: ids,
    records: structuredClone(records.filter((record) => ids[record.type]?.includes(record.id))),
    group_ids: baseline.group_snapshots.filter((group) => group.enabled).map((group) => group.id) });
}
export function patchXYZPrompt(draft, type, text) {
  const next = structuredClone(draft), source = structuredClone(xyzPromptSource(draft));
  source[type] = text;
  next.baseline = recompose(next.baseline, source);
  return next;
}
export function toggleXYZPreset(draft, record, records) {
  const next = structuredClone(draft), source = structuredClone(xyzPromptSource(draft));
  source.active[record.type] = togglePromptPresetId(source.active[record.type], record.id);
  source.records = structuredClone(records.filter((item) => source.active[item.type]?.includes(item.id)));
  next.baseline = recompose(next.baseline, source);
  return next;
}
export function refreshXYZPromptLibrary(draft, records) {
  if (!draft.baseline.prompt_source) return draft; // no render/mount-time legacy migration
  const next = structuredClone(draft), source = next.baseline.prompt_source;
  for (const type of ["positive", "negative"]) source.active[type] = source.active[type].filter((id) => records.some((record) => record.id === id && record.type === type));
  source.records = structuredClone(records.filter((record) => source.active[record.type]?.includes(record.id)));
  next.baseline = recompose(next.baseline, source);
  return next;
}
export function recomposeXYZGroups(draft, addedIds = []) {
  // A legacy actual string keeps old frozen group text. Only an explicit new group enable opts
  // its own prefix in; no guessed substring removal when dropping legacy groups.
  const next = structuredClone(draft), source = structuredClone(xyzPromptSource(draft));
  source.group_ids = [...new Set([...source.group_ids, ...addedIds])];
  next.baseline = recompose(next.baseline, source);
  return next;
}
export function xyzWireDraft(draft) {
  const next = projectXYZEditor(draft);
  if (next?.baseline) { delete next.baseline.prompt_source; delete next.baseline.postprocess_source; }
  return next;
}
import { projectXYZEditor } from './xyz-editor-source.js';
