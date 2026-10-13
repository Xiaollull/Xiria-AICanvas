// QPlan2/P2 Phase D: actions on a finished candidate. Every one of them is an explicit, previewable edit of the
// search DRAFT (or a new library group). None touches a source group, a global mount or an old round.
import { PREFERENCES, formatUnits, inspectRule, parseUnits } from "./lora-mix-rules.js";
import { mixIssues } from "./lora-mix-model.js";

export const GROUP_MEMBER_LIMIT = 16;
export const GROUP_NAME_LIMIT = 60;

/** Presentation only: 4x4 for 16, eight columns by four rows for 32, 8x8 for 64. Never an axis. */
export function recommendedColumns(count) {
  if (!Number.isInteger(count) || count < 1) return 1;
  if (count <= 4) return count;
  if (count === 16) return 4;
  if (count === 32 || count === 64) return 8;
  return Math.min(12, Math.ceil(Math.sqrt(count)));
}

const withMix = (draft, patch) => ({ ...draft, mix: { ...draft.mix, ...patch } });
const members = (draft) => draft.mix.members;

/** weights: { instanceId: units } of one frozen candidate. Returns the member rules that would change. */
export function adoptAsBase(draft, weights) {
  const changes = [], issues = [];
  members(draft).forEach((member, index) => {
    const rule = member.rule, name = member.ref.name || member.ref.value;
    if (rule.locked === true) return;
    const units = weights[member.ref.instanceId];
    if (!Number.isInteger(units)) { issues.push(`「${name}」不在这个候选中；不能设为基线`); return; }
    const verdict = inspectRule({ ...rule, base: formatUnits(units) });
    if (verdict.errors.base || verdict.errors.min || verdict.errors.max || verdict.errors.step) { issues.push(`「${name}」的当前范围无法容纳 ${formatUnits(units)}：${Object.values(verdict.errors)[0]}`); return; }
    changes.push({ index, name, from: rule.base, to: formatUnits(units) });
  });
  return { ok: issues.length === 0, issues, changes };
}

export function applyAdoptAsBase(draft, weights) {
  const plan = adoptAsBase(draft, weights);
  if (!plan.ok) return { ...plan, draft };
  const next = withMix(draft, { members: members(draft).map((member, index) => { const change = plan.changes.find((item) => item.index === index); return change ? { ...member, rule: { ...member.rule, base: change.to } } : member; }),
    candidates: [], proposed_key: "" });
  return { ...plan, draft: next };
}

/**
 * A narrower and/or finer follow-up around one candidate: base moves to its weights, the range halves around it
 * (clipped to the old range) and/or the step halves. Integer units throughout; the user previews it first.
 */
export function followUp(draft, weights, { narrow = true, finer = false } = {}) {
  const adopt = adoptAsBase(draft, weights);
  if (!adopt.ok) return { ok: false, issues: adopt.issues, changes: [] };
  const issues = [], changes = [];
  const nextMembers = members(draft).map((member, index) => {
    const rule = member.rule;
    if (rule.locked === true) return member;
    const base = weights[member.ref.instanceId], low = parseUnits(rule.min).units, high = parseUnits(rule.max).units, step = parseUnits(rule.step).units;
    const half = narrow ? Math.floor((high - low) / 4) : null;
    const next = { ...rule, base: formatUnits(base), min: narrow ? formatUnits(Math.max(low, base - half)) : rule.min, max: narrow ? formatUnits(Math.min(high, base + half)) : rule.max,
      step: finer ? formatUnits(Math.max(1, Math.floor(step / 2))) : rule.step };
    if (!PREFERENCES.includes(next.preference)) next.preference = "balanced";
    const verdict = inspectRule(next);
    if (Object.keys(verdict.errors).length) { issues.push(`「${member.ref.name || member.ref.value}」的后续规则无效：${Object.values(verdict.errors)[0]}`); return member; }
    changes.push({ index, name: member.ref.name || member.ref.value, before: { base: rule.base, min: rule.min, max: rule.max, step: rule.step }, after: { base: next.base, min: next.min, max: next.max, step: next.step } });
    return { ...member, rule: next };
  });
  if (issues.length) return { ok: false, issues, changes: [] };
  // The displacement bound follows the new reachable maximum only when it would otherwise be refused; the user sees this in the preview.
  let draftNext = withMix(draft, { members: nextMembers, candidates: [], proposed_key: "" });
  const report = mixIssues(draftNext);
  const bound = report.issues.find((issue) => issue.scope === "displacement");
  let displacement = { before: draft.mix.max_displacement, after: draft.mix.max_displacement };
  if (bound && report.analysis.valid) { displacement.after = formatUnits(report.analysis.dmax); draftNext = withMix(draftNext, { max_displacement: displacement.after }); }
  return { ok: true, issues: [], changes, displacement, draft: draftNext };
}

/** A new library group built from one candidate. Source groups are never edited; ids and names are fresh. */
export function groupFromCandidate(draft, candidate, { includeGlobals = false } = {}) {
  const tests = members(draft).map((member) => ({ value: member.ref.value, name: member.ref.name || member.ref.value, category: member.ref.category || "other",
    weight: Number(formatUnits(candidate.weights[member.ref.instanceId])), precision: 4, enabled: true,
    ...Object.fromEntries(["identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference"].filter((key) => member.ref[key] !== undefined).map((key) => [key, structuredClone(member.ref[key])])) }));
  const globals = includeGlobals ? draft.baseline.loras.map(({ groupId: _groupId, ...item }) => structuredClone(item)) : [];
  const all = [...globals, ...tests];
  if (all.length > GROUP_MEMBER_LIMIT) return { ok: false, message: `组合最多 ${GROUP_MEMBER_LIMIT} 个成员（当前 ${all.length}）；不会截断` };
  const paths = new Set(all.map((item) => item.value));
  if (paths.size !== all.length) return { ok: false, message: "组合中有重复路径；全局背景与测试成员不能重叠" };
  const label = `LoRA 混合 ${candidate.id}${includeGlobals ? "（含全局）" : "（仅测试成员）"}`.slice(0, GROUP_NAME_LIMIT);
  return { ok: true, name: label, members: all, presetPrompt: "", scope: includeGlobals ? "complete" : "tests" };
}

export const isFavorite = (draft, key) => (draft.mix.favorites || []).includes(key);
export function toggleFavorite(draft, key) {
  const list = draft.mix.favorites || [];
  return withMix(draft, { favorites: list.includes(key) ? list.filter((item) => item !== key) : [...list, key] });
}
