// QPlan2/P2: derived, side-effect-free view of a search draft. Everything here is recomputed from the raw
// draft on demand, so an edit can never leave a stale count, key or verdict behind, and nothing is clamped.
import { XYZ_CAPABILITIES } from "./xyz-draft.js";
import { xyzStrongLora } from "./xyz-sidebar-context.js";
import { effectiveLora } from "./xyz-group-resolution.js";
import { analyzeMembers, baselineVector, countSpace, displacementOf, formatUnits, inspectMaxDisplacement, inspectVector, parseUnits, vectorKey } from "./lora-mix-rules.js";

const ACTIVE_LIMIT = 16;
export const LORA_REF_FIELDS = ["value", "name", "category", "identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference"];

/** The strict reference of a test member, taken from a mount-like object (identity only; the weight is X2's). */
export function memberReference(item) {
  return Object.fromEntries(LORA_REF_FIELDS.filter((key) => item[key] !== undefined).map((key) => [key, structuredClone(item[key])]));
}

/** X1 transaction result -> members. Kept members keep their rule; a new member starts from the picked weight. */
export function applySelection(mix, picked) {
  const existing = new Map(mix.members.map((member) => [member.ref.instanceId, member]));
  const seen = new Set(), members = [];
  for (const item of picked) {
    if (seen.has(item.instanceId)) continue;
    seen.add(item.instanceId);
    const kept = existing.get(item.instanceId);
    members.push(kept ? { ref: memberReference({ ...kept.ref, ...item }), rule: kept.rule }
      : { ref: memberReference(item), rule: { locked: false, base: String(item.weight ?? "1"), min: "", max: "", preference: "balanced", step: "0.1" } });
  }
  return { ...mix, members };
}

/** Role conflicts between the frozen globals and the test members; never resolved automatically. */
export function roleConflicts(globals, members, rows) {
  const found = [];
  members.forEach((member, index) => {
    const row = rows?.[index], neverActive = row?.locked && row.parsed?.base === 0;
    for (const global of globals) {
      if (global.instanceId && global.instanceId === member.ref.instanceId) found.push({ kind: "instance", index, global });
      else if (global.value === member.ref.value) found.push({ kind: "path", index, global });
      else if (!neverActive && global.contentHash && global.contentHash === member.ref.contentHash && effectiveLora(global)) found.push({ kind: "content", index, global });
    }
  });
  return found;
}
const conflictText = { instance: "与全局 LoRA 是同一实例", path: "与全局 LoRA 路径相同", content: "与已启用的全局 LoRA 内容完全相同（不同副本）" };

export function budgetOf(draft, limits = XYZ_CAPABILITIES.limits) {
  const budget = draft.mix.budget, text = budget.preset === "custom" ? budget.custom : budget.preset, errors = [];
  const images = /^\d{1,7}$/.test(text.trim()) ? Number(text.trim()) : NaN;
  if (!Number.isInteger(images) || images < 1) errors.push("图片预算需要是正整数");
  const seeds = draft.repeat_count;
  if (!Number.isInteger(seeds) || seeds < 1 || seeds > 32) errors.push("重复样本必须为 1–32 整数");
  const slots = errors.length ? 0 : Math.floor(images / seeds);
  if (!errors.length && slots < 1) errors.push(`预算 ${images} 张少于样本数 ${seeds}；至少需要每个配置 1 组样本`);
  if (!errors.length && (slots > limits.axis_values || images > limits.total_cells)) errors.push(`超出现有执行上限（每轴 ${limits.axis_values} 项、总格数 ${limits.total_cells}）；不会截断样本`);
  return { images: Number.isInteger(images) ? images : null, seeds: Number.isInteger(seeds) ? seeds : null, slots, leftover: errors.length ? 0 : images - slots * seeds, errors };
}

/** The unlocked-member order used by every candidate vector. */
export function memberAnalysis(draft) {
  const analysis = analyzeMembers(draft.mix.members);
  return { ...analysis, ids: analysis.free.map((row) => draft.mix.members[row.index].ref.instanceId) };
}

/** Parse one preview row against the current rules. Raw text is kept; an invalid cell is an error, not a default. */
export function parseCandidate(candidate, analysis) {
  const errors = [], units = [];
  analysis.free.forEach((row, position) => {
    const id = analysis.ids[position], raw = candidate.raw[id];
    if (raw === undefined) { errors.push({ id, message: "缺少权重" }); units.push(null); return; }
    const value = parseUnits(raw);
    if (!value.ok) { errors.push({ id, message: value.message }); units.push(null); return; }
    if (row.sequence.indexOf(value.units) < 0) errors.push({ id, message: "不在该成员的候选序列中（不取整、不截断）" });
    units.push(value.units);
  });
  const complete = units.every((value) => value !== null);
  return { id: candidate.id, units: complete ? units : null, key: complete ? vectorKey(units) : null, errors };
}

export const candidateRaw = (analysis, units) => Object.fromEntries(analysis.ids.map((id, i) => [id, formatUnits(units[i])]));
export function newCandidate(mix, analysis, entry, source) {
  return { id: `C${String(mix.next_id).padStart(2, "0")}`, source, raw: candidateRaw(analysis, entry.units) };
}

/** Every verdict the run bar needs. Pure; call it on every render. */
export function mixIssues(draft, { limits = XYZ_CAPABILITIES.limits } = {}) {
  const mix = draft.mix, issues = [], add = (scope, message, extra = {}) => issues.push({ scope, message, ...extra });
  const analysis = memberAnalysis(draft), budget = budgetOf(draft, limits);
  if (!mix.members.length) add("members", "X1：至少选择 1 个测试 LoRA");
  mix.members.forEach((member, index) => {
    if (!xyzStrongLora(member.ref)) add("members", `「${member.ref.name || member.ref.value}」缺少强实例 / 内容修订，不能作为测试成员`, { member: index });
    for (const [field, message] of Object.entries(analysis.rows[index].errors)) add("rule", message, { member: index, field });
  });
  const ids = mix.members.map((member) => member.ref.instanceId);
  if (new Set(ids).size !== ids.length) add("members", "测试成员中有重复实例");
  for (const conflict of roleConflicts(draft.baseline.loras, mix.members, analysis.rows)) {
    add("role", `测试 LoRA「${mix.members[conflict.index].ref.name || mix.members[conflict.index].ref.value}」${conflictText[conflict.kind]}「${conflict.global.name || conflict.global.value}」；全局背景与测试成员不能重叠，请明确调整其中一侧`, { member: conflict.index, kind: conflict.kind });
  }
  let limit = null;
  // With every member locked there is nothing to displace: the bound is moot and the single profile is the baseline.
  if (analysis.valid && !analysis.free.length) limit = 0;
  else if (analysis.valid) {
    const verdict = inspectMaxDisplacement(mix.max_displacement, analysis.dmax);
    if (!verdict.ok) add("displacement", verdict.message); else limit = verdict.units;
  }
  for (const message of budget.errors) add("budget", message);
  if (!Number.isInteger(draft.requested_concurrency) || draft.requested_concurrency < 1 || draft.requested_concurrency > limits.requested_concurrency) add("budget", `请求并发必须为 1–${limits.requested_concurrency} 整数；不会自动调整`);
  const active = draft.baseline.loras.filter(effectiveLora).length;
  const seenKeys = new Map(), candidates = mix.candidates.map((candidate) => {
    const parsed = analysis.valid ? parseCandidate(candidate, analysis) : { id: candidate.id, units: null, key: null, errors: [] };
    const problems = parsed.errors.map((error) => ({ ...error }));
    let displacement = null;
    if (parsed.units) {
      const verdict = inspectVector(analysis.free, parsed.units, limit);
      displacement = displacementOf(analysis.free, parsed.units);
      if (!verdict.ok && verdict.reason === "displacement") problems.push({ message: verdict.message });
      if (seenKeys.has(parsed.key)) problems.push({ message: `与 ${seenKeys.get(parsed.key)} 完全重复` }); else seenKeys.set(parsed.key, candidate.id);
      const activeMembers = mix.members.filter((member, index) => {
        const row = analysis.rows[index], position = analysis.free.indexOf(row);
        return (row.locked ? row.parsed.base : parsed.units[position]) !== 0;
      }).length;
      if (active + activeMembers > ACTIVE_LIMIT) problems.push({ message: `全局 ${active} + 测试 ${activeMembers} 个启用且非零的 LoRA 超过 ${ACTIVE_LIMIT} 上限；不会截断` });
    }
    return { candidate, parsed, displacement, problems };
  });
  candidates.forEach((entry) => { if (entry.problems.length) add("candidate", `${entry.candidate.id}：${entry.problems.map((problem) => problem.message).join("；")}`, { candidate: entry.candidate.id }); });
  if (budget.slots && mix.candidates.length > budget.slots) add("budget", `候选 ${mix.candidates.length} 个超过预算可容纳的 ${budget.slots} 个配置（${budget.images} 张 ÷ ${budget.seeds} 样本）；请删除多余候选，不会自动截断`);
  if (analysis.valid && analysis.free.length && !mix.candidates.length) add("candidates", "尚无候选；请先生成候选预览");
  return { analysis, budget, limit, candidates, issues, active, expectedOutput: budget.seeds ? mix.candidates.length * budget.seeds : null,
    space: analysis.valid && limit !== null ? countSpace(analysis.free, limit) : null,
    valid: issues.length === 0 && (mix.candidates.length > 0 || (analysis.valid && analysis.free.length === 0)) };
}

/** The signature of everything a candidate depends on; a change marks the preview as generated under older rules. */
export function mixRulesKey(draft) {
  const mix = draft.mix;
  return JSON.stringify([draft.engine, mix.members.map((member) => [member.ref.instanceId, member.rule]), mix.max_displacement, mix.include_baseline]);
}

/** Candidate vectors as a map instanceId -> units, locked members at their base: the compile input. */
export function candidateWeights(draft, parsed, analysis) {
  const weights = {};
  draft.mix.members.forEach((member, index) => {
    const row = analysis.rows[index];
    weights[member.ref.instanceId] = row.locked ? row.parsed.base : parsed.units[analysis.free.indexOf(row)];
  });
  return weights;
}

export { baselineVector };
