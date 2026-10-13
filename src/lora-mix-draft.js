// QPlan2/P2: the LoRA mix-search draft. It is deliberately XYZ-shaped (`baseline`, repeat/seed fields,
// empty axes) so the mother sidebar, global LoRA manager, prompt deck, canvas and post-process panels work on
// it unchanged, while its own versioned container keeps it independent of Basic and XYZ drafts.
import { inspectXYZDrafts, XYZ_CAPABILITIES } from "./xyz-draft.js";
import { xyzWireDraft } from "./xyz-prompt-source.js";
import { defaultMaxDisplacement } from "./lora-mix-rules.js";

export const LORA_MIX_SCHEMA = 1;
export const BUDGET_PRESETS = ["16", "32", "64", "128"];
export const DEFAULT_BUDGET = "32";
const clone = (value) => structuredClone(value);
const object = (value) => value && typeof value === "object" && !Array.isArray(value);

export const emptyLoraMixDrafts = () => ({ schema_version: LORA_MIX_SCHEMA, by_engine: {} });

export function defaultMix() {
  return {
    version: LORA_MIX_SCHEMA, members: [], max_displacement: defaultMaxDisplacement(null), budget: { preset: DEFAULT_BUDGET, custom: "" },
    include_baseline: true, sampling_seed: "1", cursor: 0, candidates: [], next_id: 1, proposed_key: "", favorites: [], session: { id: null, revision: 0 },
  };
}

/** Same baseline the Basic page would submit, no randomness; the search owns a private copy. */
export function createLoraMixDraft(baseline) {
  return { schema_version: LORA_MIX_SCHEMA, kind: "lora_mix_draft", engine: baseline.parameters.engine, baseline: clone(baseline),
    axes: { x: null, y: null, z: null }, repeat_count: 1, sample_seeds: null, requested_concurrency: 1, inherit_postprocess: false, group_by_resources: false, mix: defaultMix() };
}

const validMember = (member) => object(member) && object(member.ref) && typeof member.ref.value === "string" && object(member.rule);
const validCandidate = (candidate) => object(candidate) && typeof candidate.id === "string" && ["baseline", "sampled", "manual"].includes(candidate.source)
  && (candidate.pinned === undefined || typeof candidate.pinned === "boolean")
  && object(candidate.raw) && Object.values(candidate.raw).every((value) => typeof value === "string");
export function validLoraMix(mix) {
  return object(mix) && mix.version === LORA_MIX_SCHEMA && Array.isArray(mix.members) && mix.members.every(validMember)
    && typeof mix.max_displacement === "string" && object(mix.budget) && typeof mix.budget.preset === "string" && typeof mix.budget.custom === "string"
    && typeof mix.include_baseline === "boolean" && typeof mix.sampling_seed === "string" && Number.isInteger(mix.cursor) && mix.cursor >= 0
    && Array.isArray(mix.candidates) && mix.candidates.every(validCandidate) && Number.isInteger(mix.next_id) && typeof mix.proposed_key === "string"
    && (mix.favorites === undefined || Array.isArray(mix.favorites) && mix.favorites.every((item) => typeof item === "string"))
    && object(mix.session) && (mix.session.id === null || typeof mix.session.id === "string") && Number.isInteger(mix.session.revision);
}

/**
 * Never coerce, clamp or drop a bad or future draft. An unknown mix version or a malformed baseline keeps the raw
 * data and stops autosave, exactly like XYZ drafts; a search draft is never rebuilt from Basic automatically.
 */
export function inspectLoraMixDrafts(raw, selectedMode = "basic") {
  if (raw === undefined) return { container: emptyLoraMixDrafts(), raw: null, error: "", fatal: false };
  const fatal = () => ({ container: emptyLoraMixDrafts(), raw: clone(raw), error: "LoRA 混合搜索草稿版本或结构无法安全读取；保留原始数据，已暂停自动保存。不会从基础生图自动重建覆盖。", fatal: true });
  if (!object(raw) || raw.schema_version !== LORA_MIX_SCHEMA || !object(raw.by_engine)) return fatal();
  const probe = { schema_version: 1, by_engine: {} };
  for (const [engine, draft] of Object.entries(raw.by_engine)) {
    if (!object(draft) || draft.kind !== "lora_mix_draft" || draft.schema_version !== LORA_MIX_SCHEMA || !validLoraMix(draft.mix)) return fatal();
    probe.by_engine[engine] = { ...draft, kind: "xyz_draft", schema_version: 1 };
  }
  // The baseline is the XYZ baseline contract: validate it with the very same inspector.
  if (inspectXYZDrafts(probe, selectedMode).fatal) return fatal();
  return { container: clone(raw), raw: null, error: "", fatal: false };
}

export function withLoraMixDraft(container, engine, draft) {
  if (draft.engine !== engine || draft.kind !== "lora_mix_draft" || !XYZ_CAPABILITIES.engines[engine]) throw new Error("LoRA mix engine/draft mismatch");
  return { ...container, by_engine: { ...container.by_engine, [engine]: clone(draft) } };
}

/** The wire/recipe form: the XYZ wire projection (UI-only sources stripped) plus the search block. */
export function loraMixRecipe(draft) {
  return xyzWireDraft({ ...draft, kind: "xyz_draft" });
}
