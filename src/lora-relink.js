// Pure draft rules shared by the main-window modal and the /lora repair panel.
// A server ranking `candidate` is never a transaction type: drafts carry only
// `relocate` (proven, still confirmed) or `replace` (explicit rebinding).
const draftTarget = (recommendation) => recommendation ? { value: recommendation.value, rootId: recommendation.rootId,
  kind: recommendation.kind === "relocate" ? "relocate" : "replace" } : null;
// A source that already has a committed replacement rule and no remaining
// consumer is settled: listed, but skipped unless the user selects it.
export const settledRelinkRow = (row) => Boolean(row.existingRule && row.impacts && !row.impacts.mounts && !row.impacts.groups?.length && !row.impacts.cards?.length);
// P10.2: the server classifies lifecycle. Only `pending` rows are editable repair
// rows; settled/history/unused/archived/unobserved/incomplete are listed apart
// and never counted as open problems. Rows from an older server fall back to
// the P10.1 settled rule.
export const relinkLifecycle = (row) => row.lifecycle || (settledRelinkRow(row) ? "settled" : "pending");
export function createRelinkDraft(scan, rows, revision) {
  const open = rows.filter((row) => row.status !== "unchanged");
  return { scanId: scan.scanId, complete: scan.complete, revision,
    rows: open.filter((row) => relinkLifecycle(row) === "pending").map((row) => ({ ...row,
      target: draftTarget(row.recommendation), chosenBy: "suggestion", skipped: false, scope: { all: true, history: true }, needsReview: false })),
    processed: open.filter((row) => relinkLifecycle(row) !== "pending") };
}
export function editRelinkRow(draft, sourceId, patch) {
  return { ...draft, rows: draft.rows.map((row) => row.sourceId === sourceId ? { ...row, directoryConflict: undefined, ...patch, sourceId: row.sourceId, chosenBy: "user" } : row) };
}
/** Restore only this row's suggested target. Scope and skip are user state. */
export function restoreRelinkSuggestion(draft, sourceId) {
  return { ...draft, rows: draft.rows.map((row) => row.sourceId === sourceId
    ? { ...row, target: draftTarget(row.recommendation), targetInput: undefined, directoryConflict: undefined, chosenBy: "suggestion" } : row) };
}
// Manual targets, skips and directory-rule choices are never overwritten by a
// later scan, ranking or notification.
const manual = (row) => row.chosenBy !== "suggestion";
export function mergeRelinkSuggestions(draft, incoming) {
  return { ...draft, rows: draft.rows.map((row) => {
    const next = incoming.rows.find((entry) => entry.sourceId === row.sourceId);
    if (!next) return { ...row, needsReview: true };
    if (manual(row)) return { ...row, recommendation: next.recommendation, candidates: next.candidates ?? row.candidates, suggestion: next.suggestion ?? row.suggestion,
      needsReview: row.needsReview || incoming.revision !== draft.revision || next.status !== row.status };
    return { ...next, scope: row.scope, skipped: row.skipped };
  }) };
}
export function selectedRelinkRows(draft) { return draft.rows.filter((row) => !row.skipped && row.target); }

/** Toolbar counts. "Unselected" = neither a target nor an explicit skip. */
export function relinkDraftSummary(draft) {
  const rows = draft?.rows || [];
  const count = (test) => rows.filter(test).length;
  return { sources: rows.length, selected: count((row) => !row.skipped && row.target), suggested: count((row) => !row.skipped && row.target && row.chosenBy === "suggestion"),
    manual: count((row) => !row.skipped && row.target && row.chosenBy !== "suggestion" && row.target.kind !== "delete"), unselected: count((row) => !row.skipped && !row.target),
    skipped: count((row) => row.skipped), tied: count((row) => !row.target && !row.skipped && ["tied", "contested"].includes(row.suggestion?.state)),
    review: count((row) => !row.skipped && row.needsReview), deletions: count((row) => !row.skipped && row.target?.kind === "delete"),
    processed: draft?.processed?.length || 0 };
}

/** A consumer item belongs to a source by strong identity, else by its path. */
export const relinkHit = (item, source) => item.instanceId ? item.instanceId === source.instanceId : item.value === source.value;

/** P10.2 confirmed deletion: drop only the source's members from the chosen
 * lists. Other members, weights, group names/notes and Prompt text stay. */
export function removeWorkspaceLoras(state, { source, scope }) {
  const next = structuredClone(state);
  const workspace = next.workspace || {}, patches = [];
  const strip = (list, address) => {
    const kept = (list || []).filter((item) => !relinkHit(item, source));
    if (kept.length !== (list || []).length) patches.push({ address, removed: (list || []).length - kept.length });
    return kept;
  };
  if (scope.all || scope.mounts) {
    const map = workspace.mountedLorasByEngine?.byEngine;
    if (map?.[source.engine]) map[source.engine] = strip(map[source.engine], ["workspace", "mountedLorasByEngine", "byEngine", source.engine]);
    if (workspace.model === source.engine && Array.isArray(workspace.loras)) workspace.loras = strip(workspace.loras, ["workspace", "loras"]);
  }
  (workspace.loraGroupsByEngine?.byEngine?.[source.engine] || []).forEach((group, index) => {
    if (scope.all || scope.groupIds?.includes(group.id)) group.members = strip(group.members, ["workspace", "loraGroupsByEngine", "byEngine", source.engine, index, "members"]);
  });
  return { state: next, patches };
}

/** Typed list patching: no recursion into unrelated text or run snapshots. */
export function patchLoraList(list, { source, target, kind }) {
  const changes = [];
  const items = (list || []).map((item, index) => {
    const hit = relinkHit(item, source);
    if (!hit) return item;
    const next = { ...item, value: target.value, identitySchema: 1, instanceId: target.instanceId,
      contentHash: target.contentHash, contentRevision: target.contentRevision,
      originalReference: { ...(item.originalReference || { value: item.value, ...(item.instanceId ? { instanceId: item.instanceId } : {}), ...(item.contentHash ? { contentHash: item.contentHash } : {}) }),
        lastBoundReference: { value: item.value, ...(item.instanceId ? { instanceId: item.instanceId } : {}) } } };
    const category = (target.relative || target.value).split("/")[0];
    next.category = ["style", "character", "concept", "other"].includes(category) ? category : "other";
    changes.push({ index, before: structuredClone(item), after: structuredClone(next), kind });
    return next;
  });
  const conflicts = items.flatMap((item, index) => items.some((other, at) => at !== index
    && (other.value === item.value || (item.instanceId && other.instanceId === item.instanceId))) ? [{ index, reason: "duplicate-target" }] : []);
  return { items, changes, conflicts };
}

export function patchWorkspaceLoras(state, selection) {
  const next = structuredClone(state);
  const workspace = next.workspace || {};
  const { source, scope } = selection;
  const patches = [], conflicts = [];
  const prompt = (owner, key, address) => {
    if (!scope.confirmTags || !["SD", "iL"].includes(source.engine) || typeof owner?.[key] !== "string") return;
    const parsed = rewriteExactLoraTags(owner[key], source.value, selection.target.value, { caseInsensitive: selection.caseInsensitive });
    if (!parsed.spans.length) return;
    const before = owner[key]; owner[key] = parsed.text;
    patches.push({ address: [...address, key], before, after: parsed.text, spans: parsed.spans, prompt: true });
  };
  const apply = (list, address) => {
    const result = patchLoraList(list, selection);
    patches.push(...result.changes.map((change) => ({ ...change, address })));
    conflicts.push(...result.conflicts.map((conflict) => ({ ...conflict, address })));
    return result.items;
  };
  if (scope.all || scope.mounts) {
    const map = workspace.mountedLorasByEngine?.byEngine;
    if (map?.[source.engine]) map[source.engine] = apply(map[source.engine], ["workspace", "mountedLorasByEngine", "byEngine", source.engine]);
    if (workspace.model === source.engine && Array.isArray(workspace.loras)) workspace.loras = apply(workspace.loras, ["workspace", "loras"]);
    if (workspace.model === source.engine) {
      for (const key of ["positive", "negative"]) prompt(workspace, key, ["workspace"]);
      for (const key of ["positive", "negative"]) prompt(workspace.imageToImage, key, ["workspace", "imageToImage"]);
    }
    const settings = workspace.engineSettingsByEngine?.byEngine?.[source.engine];
    for (const key of ["positive", "negative"]) prompt(settings, key, ["workspace", "engineSettingsByEngine", "byEngine", source.engine]);
  }
  const groups = workspace.loraGroupsByEngine?.byEngine?.[source.engine] || [];
  groups.forEach((group, index) => {
    if (scope.all || scope.groupIds?.includes(group.id)) {
      group.members = apply(group.members, ["workspace", "loraGroupsByEngine", "byEngine", source.engine, index, "members"]);
      prompt(group, "presetPrompt", ["workspace", "loraGroupsByEngine", "byEngine", source.engine, index]);
    }
  });
  for (const [index, preset] of (workspace.promptPresets?.records || []).entries()) {
    // Shared presets have no engine provenance. They are never included by
    // `all`; the user must explicitly confirm each shared preset attribution.
    if (scope.presetIds?.includes(preset.id)) prompt(preset, "content", ["workspace", "promptPresets", "records", index]);
  }
  return { state: next, patches, conflicts };
}

/** Locate exact tag-name spans; preserve argument bytes and every other word.
 * Extensionless folder-qualified IDs are traceable; bare `IDA` is not.
 */
export function rewriteExactLoraTags(text, sourceValue, targetValue, { caseInsensitive = false } = {}) {
  if (typeof text !== "string") return { text, spans: [] };
  const fold = (value) => caseInsensitive ? value.toLowerCase().replaceAll("\\", "/") : value;
  const withoutExtension = (value) => value.replace(/\.(?:safetensors|ckpt|pt|pth)$/i, "");
  const full = fold(sourceValue), short = sourceValue.includes("/") ? fold(withoutExtension(sourceValue)) : null;
  const numeric = /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?$/i;
  const spans = [];
  for (const match of text.matchAll(/<lora:([^<>]*)>/g)) {
    const parts = match[1].split(":");
    const raw = parts.shift(), value = raw.trim(), key = fold(value);
    const positional = parts.filter((item) => !item.includes("=")), named = parts.filter((item) => item.includes("="));
    if (positional.length > 2 || positional.some((item) => !numeric.test(item.trim())) || named.some((item) => !/^(?:te|unet)=/.test(item) || !numeric.test(item.slice(item.indexOf("=") + 1).trim()))) continue;
    const weights = Object.fromEntries(named.map((item) => item.split("=")));
    const te = Number(weights.te ?? positional[0] ?? 1), unet = Number(weights.unet ?? positional[1] ?? te);
    if (!Number.isFinite(te) || te < -5 || te > 5 || te !== unet) continue;
    if (key !== full && (!short || key !== short)) continue;
    const start = match.index + match[0].indexOf(":") + 1 + raw.indexOf(value);
    const end = start + value.length;
    const replacement = key === full ? targetValue : withoutExtension(targetValue);
    if (value !== replacement) spans.push({ start, end, before: value, after: replacement });
  }
  let result = text;
  for (const span of [...spans].reverse()) result = result.slice(0, span.start) + span.after + result.slice(span.end);
  return { text: result, spans };
}

/** Undo only owned entries. Later edits conflict, unrelated fields are retained. */
export function undoWorkspacePatches(state, patches) {
  const next = structuredClone(state), conflicts = [];
  for (const patch of [...patches].reverse()) {
    let list = next;
    for (const key of patch.address) list = list?.[key];
    if (!Array.isArray(list) || JSON.stringify(list[patch.index]) !== JSON.stringify(patch.after)) {
      conflicts.push({ address: patch.address, index: patch.index, reason: "edited-after-repair" });
      continue;
    }
    list[patch.index] = structuredClone(patch.before);
  }
  return { state: next, conflicts };
}
