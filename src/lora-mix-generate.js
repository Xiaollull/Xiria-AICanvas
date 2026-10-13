// QPlan2/P2 Phase B: the in-memory preview transactions. They produce a new draft, never a job, a
// reservation or a request, and they never touch anything but `draft.mix`.
import { SAMPLER_ALGORITHM, proposeCandidates } from "./lora-mix-sampler.js";
import { baselineVector, formatUnits, vectorKey } from "./lora-mix-rules.js";
import { candidateRaw, mixIssues, mixRulesKey, newCandidate } from "./lora-mix-model.js";

const BLOCKING = new Set(["members", "rule", "role", "displacement", "budget"]);
const withMix = (draft, patch) => ({ ...draft, mix: { ...draft.mix, ...patch } });

/**
 * Fill the preview from the rules. Pinned rows are kept (and spread the rest away from them); everything else
 * is replaced. `exclude` holds vector keys already tested or reserved in the same context.
 */
export function generateCandidates(draft, { exclude = new Set() } = {}) {
  const report = mixIssues(draft), blockers = report.issues.filter((issue) => BLOCKING.has(issue.scope));
  if (blockers.length) return { ok: false, issues: blockers };
  const { analysis, budget, limit } = report, mix = draft.mix;
  if (!analysis.free.length) return { ok: true, draft: withMix(draft, { candidates: [], cursor: mix.cursor + 1, proposed_key: mixRulesKey(draft) }), note: "全部成员已锁定：只有 1 个配置（固定基础权重）", allLocked: true, proposal: null };
  // Pinned rows survive a redraw; so does an unedited baseline row, so its id stays stable.
  const baseKey = vectorKey(baselineVector(analysis.free));
  const kept = report.candidates.filter((entry) => entry.candidate.pinned || (mix.include_baseline && entry.candidate.source === "baseline" && entry.parsed.key === baseKey));
  const validPinned = kept.filter((entry) => entry.parsed.units && !entry.problems.length), retained = kept.filter((entry) => !validPinned.includes(entry));
  const proposal = proposeCandidates({ free: analysis.free, limit, count: Math.max(0, budget.slots - retained.length), seed: mix.sampling_seed, cursor: mix.cursor,
    includeBaseline: mix.include_baseline, pinned: validPinned.map((entry) => entry.parsed.units), exclude });
  const byKey = new Map(validPinned.map((entry) => [entry.parsed.key, entry.candidate]));
  let working = mix;
  const rows = proposal.vectors.map((entry) => {
    if (entry.source === "pinned") return byKey.get(entry.key);
    const row = newCandidate(working, analysis, entry, entry.source);
    working = { ...working, next_id: working.next_id + 1 };
    return row;
  });
  const next = withMix(draft, { candidates: [...retained.map((entry) => entry.candidate), ...rows], next_id: working.next_id, cursor: mix.cursor + 1, proposed_key: mixRulesKey(draft) });
  const notes = [];
  if (proposal.shortfall > 0) notes.push(proposal.exhausted ? `候选空间已穷尽：只有 ${rows.length + retained.length} 个可用配置，不会用重复项补足` : `在有限工作量内只找到 ${rows.length + retained.length} 个配置；这不代表空间已穷尽`);
  return { ok: true, draft: next, proposal: { ...proposal, algorithm: SAMPLER_ALGORITHM }, note: notes.join("；"), allLocked: false };
}

export const clearCandidates = (draft) => withMix(draft, { candidates: [] });
export const removeCandidate = (draft, id) => withMix(draft, { candidates: draft.mix.candidates.filter((candidate) => candidate.id !== id) });
export const togglePin = (draft, id) => withMix(draft, { candidates: draft.mix.candidates.map((candidate) => candidate.id === id ? { ...candidate, pinned: !candidate.pinned } : candidate) });

/** A manual row starts from the baseline values so every cell is a real, visible number the user then edits. */
export function addManualCandidate(draft) {
  const report = mixIssues(draft);
  if (!report.analysis.valid) return { ok: false, issues: report.issues.filter((issue) => issue.scope === "rule" || issue.scope === "members") };
  const row = { id: `C${String(draft.mix.next_id).padStart(2, "0")}`, source: "manual", raw: candidateRaw(report.analysis, baselineVector(report.analysis.free)) };
  return { ok: true, draft: withMix(draft, { candidates: [...draft.mix.candidates, row], next_id: draft.mix.next_id + 1 }) };
}

/** Editing a cell stores exactly the typed text; it becomes a manual row (an edited sample is no longer the sampler's). */
export function editCandidateValue(draft, id, instanceId, text) {
  return withMix(draft, { candidates: draft.mix.candidates.map((candidate) => candidate.id !== id ? candidate
    : { ...candidate, source: candidate.source === "baseline" ? "baseline" : "manual", raw: { ...candidate.raw, [instanceId]: text } }) });
}

/** Explicit action: a new sampling seed is drawn only when the user asks (never while rendering). */
export function drawSamplingSeed(randomBytes = (bytes) => crypto.getRandomValues(bytes)) {
  const bytes = randomBytes(new Uint32Array(2));
  return ((BigInt(bytes[0]) << 32n) | BigInt(bytes[1])).toString();
}

export { formatUnits };
