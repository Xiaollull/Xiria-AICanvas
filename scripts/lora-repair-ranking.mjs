import { parseSharedRef } from "../src/shared-model-refs.js";

// P10.1 repair suggestions. Pure functions: a ranking reason is a preference
// the user reviews, never provenance proof and never a transaction type. The
// planner still receives only `relocate` (proven) or `replace` (explicit
// rebinding) and revalidates everything at commit.
export const REPAIR_BASIS = Object.freeze({ "proven-relocation": 4, "equal-content": 3, "name-location": 2, "different-content": 1 });
const PREFILL = new Set(["proven-relocation", "equal-content", "name-location"]);
const RELATION_RANK = { "same-folder": 3, "descendant-folder": 2, "ancestor-folder": 2, "same-category": 1, "same-folder-name": 1, "other-folder": 0 };
export const MAX_RANKED_CANDIDATES = 8;

export const repairRelative = (value) => parseSharedRef(value)?.relativePath || value;
/** Segment-aware folder containment (`style` never contains `style2/...`). */
export function withinPrefix(relative, prefix, insensitive = process.platform === "win32") {
  if (!prefix) return true;
  const item = insensitive ? relative.toLowerCase() : relative, base = insensitive ? prefix.toLowerCase() : prefix;
  return item === base || item.startsWith(`${base}/`);
}
/** Prefix-aware coverage (P10.2): a root listed with prefixes covers only those folders. */
export function scanCovers(scan, item, insensitive) {
  const root = scan.roots.find((entry) => entry.engine === item.engine && entry.rootId === item.rootId);
  return Boolean(root) && (root.prefixes || [""]).some((prefix) => withinPrefix(repairRelative(item.value), prefix, insensitive));
}
export const repairFolder = (relative) => relative.includes("/") ? relative.slice(0, relative.lastIndexOf("/")) : "";
export const repairFileName = (relative) => relative.slice(relative.lastIndexOf("/") + 1);

/** Display-only folder relationship. Case is folded because it only orders the
 * list; it never decides a tie or authorizes a selection. */
export function folderRelation(sourceRootId, sourceFolder, candidateRootId, candidateFolder) {
  const s = sourceFolder.toLowerCase(), c = candidateFolder.toLowerCase();
  if (s === c) return "same-folder";
  if (sourceRootId === candidateRootId) {
    if (!s || c.startsWith(`${s}/`)) return "descendant-folder";
    if (!c || s.startsWith(`${c}/`)) return "ancestor-folder";
  }
  if (s && c && s.split("/")[0] === c.split("/")[0]) return "same-category";
  if (s && c && s.split("/").at(-1) === c.split("/").at(-1)) return "same-folder-name";
  return "other-folder";
}

// Substantive criteria decide prefill and ties. Folder relation and names only
// order the display, so enumeration or alphabetical order never selects.
function substantive(entry) {
  return entry.basis === "name-location" ? [REPAIR_BASIS[entry.basis], entry.sameRoot ? 1 : 0, entry.support]
    : [REPAIR_BASIS[entry.basis], entry.existing ? 0 : 1, entry.sameRoot ? 1 : 0, entry.exactName ? 1 : 0, entry.support];
}
function compareKeys(a, b) {
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) if ((a[index] ?? 0) !== (b[index] ?? 0)) return (b[index] ?? 0) - (a[index] ?? 0);
  return 0;
}
const display = (entry) => [...substantive(entry), RELATION_RANK[entry.relation]];
function stable(a, b) { return compareKeys(display(a), display(b)) || (a.candidate.rootId < b.candidate.rootId ? -1 : a.candidate.rootId > b.candidate.rootId ? 1 : 0) || (a.candidate.value < b.candidate.value ? -1 : a.candidate.value > b.candidate.value ? 1 : 0); }

function reasons(entry) {
  const list = [];
  if (entry.basis === "proven-relocation") list.push("filesystem-identity", "same-content");
  else if (entry.basis === "equal-content") list.push("same-content");
  else if (entry.basis === "different-content") list.push("different-content");
  if (entry.exactName) list.push("same-name");
  // Null-SHA sources cannot tell a moved file from an older separate copy, so
  // `existing-copy` is a content-level reason only; R01/R02 then agree.
  if (entry.existing && entry.basis !== "name-location") list.push("existing-copy");
  list.push(entry.sameRoot ? "same-root" : "other-root", entry.relation);
  if (entry.support > 1) list.push("directory-group");
  if (entry.basis !== "proven-relocation") list.push("provenance-unconfirmed");
  return list;
}
function publicEntry(entry) {
  const { candidate } = entry;
  return { engine: candidate.engine, rootId: candidate.rootId, value: candidate.value, contentHash: candidate.contentHash, size: candidate.size,
    basis: entry.basis, existing: entry.existing, reasons: reasons(entry),
    ...(entry.support > 1 ? { group: { from: entry.sourceFolder, to: entry.folder, count: entry.support, of: entry.groupSize } } : {}) };
}

/**
 * Rank candidates for every repair row in one pass, so a coherent directory
 * change across several sources can support each of them and two sources can
 * never silently receive the same target.
 * - rows: [{ source, observed, provenTarget }] (unchanged sources excluded)
 * - claimedBy(observation): committed-baseline instance id at that path, or null
 */
export function rankRepairRows(rows, observations, { complete, claimedBy }) {
  const prepared = rows.map(({ source, observed, provenTarget }) => {
    const relative = repairRelative(source.value), name = repairFileName(relative), sourceFolder = repairFolder(relative);
    const entries = [];
    for (const candidate of observations) {
      if (candidate.engine !== source.engine || (candidate.rootId === source.rootId && candidate.value === source.value)) continue;
      const candidateRelative = candidate.relative || repairRelative(candidate.value);
      const exactName = repairFileName(candidateRelative) === name;
      let basis = null;
      if (source.contentHash) basis = candidate.contentHash === source.contentHash ? (candidate === provenTarget ? "proven-relocation" : "equal-content") : exactName ? "different-content" : null;
      else if (exactName) basis = "name-location";
      if (!basis) continue;
      const claimant = claimedBy(candidate);
      const folder = repairFolder(candidateRelative);
      entries.push({ candidate, basis, exactName, existing: Boolean(claimant && claimant !== source.instanceId), sameRoot: candidate.rootId === source.rootId,
        folder, sourceFolder, folderKey: `${candidate.rootId}\0${folder}`, relation: folderRelation(source.rootId, sourceFolder, candidate.rootId, folder), support: 0 });
    }
    return { source, observed, entries, groupKey: `${source.engine}\0${source.rootId}\0${sourceFolder}` };
  });
  // Group support: how many sources from the same old folder have a positive
  // counterpart in the candidate folder. A ranking reason, not proof.
  const groups = new Map();
  for (const row of prepared) {
    const group = groups.get(row.groupKey) || { size: 0, folders: new Map() };
    group.size += 1;
    for (const key of new Set(row.entries.filter((entry) => PREFILL.has(entry.basis)).map((entry) => entry.folderKey))) group.folders.set(key, (group.folders.get(key) || 0) + 1);
    groups.set(row.groupKey, group);
  }
  const ranked = prepared.map((row) => {
    const group = groups.get(row.groupKey);
    for (const entry of row.entries) { entry.support = group.folders.get(entry.folderKey) || 0; entry.groupSize = group.size; }
    row.entries.sort(stable);
    const top = row.entries[0];
    const ties = top ? row.entries.filter((entry) => compareKeys(substantive(entry), substantive(top)) === 0) : [];
    let suggestion = { state: "none" };
    if (top && PREFILL.has(top.basis)) suggestion = !complete || row.observed ? { state: "blocked", basis: top.basis }
      : ties.length > 1 ? { state: "tied", basis: top.basis, ties: ties.length } : { state: "suggested", basis: top.basis };
    return { ...row, top: suggestion.state === "suggested" ? top : null, suggestion };
  });
  // One target, one prefilled source. Strictly stronger evidence keeps it;
  // equal evidence leaves every competitor for the user. Never second-best.
  const byTarget = new Map();
  for (const row of ranked) if (row.top) {
    const key = `${row.top.candidate.rootId}\0${row.top.candidate.value}`;
    byTarget.set(key, [...(byTarget.get(key) || []), row]);
  }
  for (const competitors of byTarget.values()) {
    if (competitors.length < 2) continue;
    const best = Math.max(...competitors.map((row) => REPAIR_BASIS[row.top.basis]));
    const winners = competitors.filter((row) => REPAIR_BASIS[row.top.basis] === best);
    for (const row of competitors) {
      if (winners.length === 1 && row === winners[0]) continue;
      row.suggestion = { state: "contested", basis: row.top.basis, contestedBy: competitors.filter((other) => other !== row).map((other) => other.source.instanceId) };
      row.top = null;
    }
  }
  return ranked.map((row) => ({
    sourceId: row.source.instanceId,
    candidates: row.entries.slice(0, MAX_RANKED_CANDIDATES).map(publicEntry),
    candidateCount: row.entries.length,
    suggestion: row.suggestion,
    recommendation: row.top ? { value: row.top.candidate.value, rootId: row.top.candidate.rootId, kind: row.top.basis === "proven-relocation" ? "relocate" : "replace",
      basis: row.top.basis, reasons: reasons(row.top), ...(row.top.support > 1 ? { group: publicEntry(row.top).group } : {}) } : null,
  }));
}

/**
 * P10.2 lifecycle, classified server-side from fresh consumers, committed rules
 * and this scan's coverage (never from a client receipt or a timer):
 * - pending: a mount, group or Gallery card still uses the source
 * - settled: no consumer left and its replacement chain ends at an available target
 * - history: no consumer; the chain ends at a target that is itself gone (that
 *   target is the primary problem, this source is only its dependency)
 * - unused: missing with neither consumers nor a rule
 * - archived: explicitly confirmed deleted (tombstone)
 * - unobserved / incomplete: not enough evidence; never settled or deleted
 */
export function repairLifecycle(row, source, registry, scan) {
  if (source.retired) return { lifecycle: "archived" };
  if (row.status === "unobserved" || row.status === "incomplete") return { lifecycle: row.status };
  const impacts = row.impacts || {};
  if ((impacts.mounts || 0) + (impacts.groups?.length || 0) + (impacts.cards?.length || 0)) return { lifecycle: "pending" };
  let cursor = source, hash = source.contentHash, target = null;
  const seen = new Set([source.instanceId]);
  for (;;) {
    const rule = registry.substitutions.find((item) => item.engine === cursor.engine && item.sourceId === cursor.instanceId && item.sourceHash === hash);
    if (!rule) break;
    const next = registry.instances[rule.targetId];
    if (!next || seen.has(next.instanceId)) { target = null; break; }
    seen.add(next.instanceId); target = next; cursor = next; hash = rule.targetHash;
  }
  if (!target) return { lifecycle: "unused" };
  const effectiveTarget = { instanceId: target.instanceId, value: target.value, rootId: target.rootId };
  const observed = scan.observations.find((item) => item.engine === target.engine && item.rootId === target.rootId && item.value === target.value);
  const available = target.status === "present" && (!scanCovers(scan, target) || Boolean(observed && observed.contentHash === target.contentHash));
  return available ? { lifecycle: "settled", effectiveTarget } : { lifecycle: "history", dependsOn: target.instanceId, effectiveTarget };
}
