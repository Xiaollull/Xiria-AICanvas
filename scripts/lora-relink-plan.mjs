import { randomUUID } from "node:crypto";
import { assertRegistry, libraryError, sameObservedInstance } from "./lora-library.mjs";
import { scanCovers } from "./lora-repair-ranking.mjs";
import { patchWorkspaceLoras, removeWorkspaceLoras, relinkHit } from "../src/lora-relink.js";

/** Plan typed associations only. No writes, no tensor moves, no prompt rewriting. */
export function planLoraRelink({ registry, scan, state, cards, galleryCards = [], selections, operationId = null, now = new Date().toISOString() }) {
  assertRegistry(registry);
  if (!scan.complete) throw libraryError("Incomplete scan cannot authorize repair");
  const nextRegistry = structuredClone(registry);
  let nextState = structuredClone(state), nextCards = structuredClone(cards);
  const workspacePatches = [], galleryPatches = [], cardPatches = [], conflicts = [], results = [];
  const selected = new Set();
  for (const selection of selections) {
    if (selection.skipped || !selection.target) continue;
    if (selected.has(selection.sourceId)) throw libraryError("Source selected more than once", 400);
    selected.add(selection.sourceId);
    const source = nextRegistry.instances[selection.sourceId];
    if (!source) throw libraryError("Source baseline no longer registered");
    if (selection.target.kind === "delete") {
      // P10.2 confirmed deletion: only an explicit user confirmation, only a
      // covered and actually absent source, only the chosen consumers.
      if (!scanCovers(scan, source)) throw libraryError("来源不在本次完整观察范围内，不能确认删除");
      const present = scan.observations.find((item) => item.engine === source.engine && item.rootId === source.rootId && item.value === source.value);
      if (present && sameObservedInstance(source, present)) throw libraryError("来源文件仍然存在，不能按已删除清理");
      const scope = selection.scope || {};
      const removed = removeWorkspaceLoras(nextState, { source, scope });
      nextState = removed.state;
      workspacePatches.push(...removed.patches);
      let count = removed.patches.reduce((sum, patch) => sum + patch.removed, 0);
      for (const card of galleryCards) {
        if (card.settings.model !== source.engine || !(scope.all || scope.galleryIds?.includes(card.id))) continue;
        const prior = galleryPatches.find((patch) => patch.cardId === card.id), input = prior?.after || card.settings;
        const kept = (input.loras || []).filter((item) => !relinkHit(item, source));
        if (kept.length === (input.loras || []).length) continue;
        count += (input.loras || []).length - kept.length;
        const after = { ...input, loras: kept };
        if (prior) prior.after = after; else galleryPatches.push({ cardId: card.id, before: card.settings, after });
      }
      if (scope.all) {
        // Minimal tombstone on the instance itself; its rules retire with it.
        nextRegistry.instances[source.instanceId] = { ...source, status: "missing", retired: { operationId, at: now, scope: "all" } };
        nextRegistry.substitutions = nextRegistry.substitutions.filter((rule) => !(rule.engine === source.engine && rule.sourceId === source.instanceId));
      }
      results.push({ sourceId: source.instanceId, targetId: null, kind: "delete", evidence: "confirmed-deleted", engine: source.engine, sourceValue: source.value, targetValue: null,
        sourceRootId: source.rootId, targetRootId: null, sourceHash: source.contentHash, targetHash: null, removed: count });
      continue;
    }
    const observation = scan.observations.find((item) => item.engine === source.engine && item.value === selection.target.value && item.rootId === selection.target.rootId);
    if (!observation) throw libraryError("Target is not in the current admitted engine inventory");
    const occupants = Object.values(nextRegistry.instances).filter((item) => item.status === "present" && item.value === observation.value && sameObservedInstance(item, observation));
    if (occupants.length > 1) throw libraryError("Conflicting target identities; review baseline");
    const oldTarget = occupants[0] || null;
    let target;
    const kind = selection.target.kind;
    if (kind === "relocate") {
      if (oldTarget && oldTarget.instanceId !== source.instanceId) throw libraryError("An existing copy is a replacement, not a relocation");
      if (source.contentHash !== observation.contentHash || !selection.confirmedOrigin) throw libraryError("Relocation requires verified equal bytes and explicit origin confirmation");
      target = { ...source, ...observation, instanceId: source.instanceId, status: "present", lastCompleteObservation: scan.scanId,
        history: [...new Set([...source.history, source.value])].filter((value) => value !== observation.value) };
      nextRegistry.instances[source.instanceId] = target;
    } else if (kind === "replace") {
      target = oldTarget ? { ...oldTarget, ...observation, contentRevision: oldTarget.contentRevision + (oldTarget.contentHash !== observation.contentHash ? 1 : 0), lastCompleteObservation: scan.scanId }
        : { ...observation, instanceId: observation.instanceId || randomUUID(), history: [], status: "present", contentRevision: 1, lastCompleteObservation: scan.scanId };
      if (target.instanceId === source.instanceId && target.contentHash === source.contentHash) continue;
      nextRegistry.instances[target.instanceId] = target;
      const sourceObservation = scan.observations.find((item) => item.engine === source.engine && item.rootId === source.rootId && item.value === source.value);
      const sourceCovered = scanCovers(scan, source);
      if (target.instanceId !== source.instanceId && sourceCovered && !sameObservedInstance(source, sourceObservation)) nextRegistry.instances[source.instanceId] = { ...source, status: "missing" };
      if (selection.scope.all && selection.scope.history) {
        // A replacement is a visible rule, never an identity/card merge.
        const rules = nextRegistry.substitutions;
        let cursor = target.instanceId, hash = target.contentHash;
        const seen = new Set([`${source.instanceId}:${source.contentHash}`]);
        while (cursor) {
          if (seen.has(`${cursor}:${hash}`)) throw libraryError("Replacement history would cycle");
          seen.add(`${cursor}:${hash}`);
          const rule = rules.find((rule) => rule.engine === source.engine && rule.sourceId === cursor && rule.sourceHash === hash);
          cursor = rule?.targetId; hash = rule?.targetHash;
        }
        nextRegistry.substitutions = rules.filter((rule) => !(rule.engine === source.engine && rule.sourceId === source.instanceId && rule.sourceHash === source.contentHash));
        nextRegistry.substitutions.push({ engine: source.engine, sourceId: source.instanceId, targetId: target.instanceId, kind: "replace", sourceHash: source.contentHash, targetHash: target.contentHash, tagsConfirmed: Boolean(selection.scope.confirmTags) });
      }
    } else throw libraryError("Choose relocation or explicit replacement", 400);
    const binding = { source, target, scope: selection.scope, kind, caseInsensitive: process.platform === "win32" };
    if (selection.scope.all && selection.scope.history && selection.scope.confirmTags) {
      const record = nextRegistry.instances[source.instanceId];
      record.confirmedTagRefs = [...new Set([...(record.confirmedTagRefs || []), source.value])];
    }
    const patched = patchWorkspaceLoras(nextState, binding);
    nextState = patched.state;
    workspacePatches.push(...patched.patches);
    conflicts.push(...patched.conflicts.map((conflict) => ({ ...conflict, sourceId: source.instanceId })));
    for (const card of galleryCards) {
      if (card.settings.model !== source.engine || !(selection.scope.all || selection.scope.galleryIds?.includes(card.id))) continue;
      const prior = galleryPatches.find((patch) => patch.cardId === card.id);
      const input = prior?.after || card.settings;
      const result = patchWorkspaceLoras({ workspace: { ...input, model: source.engine, loras: input.loras } }, { ...binding, scope: { mounts: true, confirmTags: selection.scope.confirmTags } });
      conflicts.push(...result.conflicts.map((conflict) => ({ ...conflict, galleryId: card.id, sourceId: source.instanceId })));
      if (result.patches.length) {
        const after = { ...input, loras: result.state.workspace.loras,
          ...(input.positive !== undefined ? { positive: result.state.workspace.positive } : {}),
          ...(input.negative !== undefined ? { negative: result.state.workspace.negative } : {}) };
        if (prior) prior.after = after;
        else galleryPatches.push({ cardId: card.id, before: card.settings, after });
      }
    }
    if (kind === "relocate" && nextCards?.byEngine?.[source.engine]?.loras) {
      const entries = nextCards.byEngine[source.engine].loras;
      const old = entries[source.value], occupied = entries[target.value];
      if (old && source.value !== target.value) {
        if (occupied) conflicts.push({ sourceId: source.instanceId, reason: "target-user-card-exists" });
        else {
          const after = { ...old, identitySchema: 1, instanceId: target.instanceId, contentHash: target.contentHash, contentRevision: target.contentRevision,
            originalReference: { ...(old.originalReference || { value: source.value, instanceId: source.instanceId }), lastBoundReference: { value: source.value, instanceId: source.instanceId } } };
          entries[target.value] = after;
          delete entries[source.value];
          nextCards.schemaVersion = 2;
          nextCards.byEngine[source.engine].byInstance ||= {};
          nextCards.byEngine[source.engine].byInstance[target.instanceId] = after;
          cardPatches.push({ engine: source.engine, from: source.value, to: target.value, before: old, after });
        }
      }
    }
    if (kind === "replace" && source.value === target.value && target.instanceId !== source.instanceId && (selection.scope.all || selection.scope.cards)) {
      const entries = nextCards?.byEngine?.[source.engine]?.loras;
      if (entries?.[source.value]) {
        const old = entries[source.value];
        entries[source.value] = { ...old, identitySchema: 1, instanceId: source.instanceId, ...(source.contentHash ? { contentHash: source.contentHash, contentRevision: source.contentRevision } : {}) };
        nextCards.schemaVersion = 2; nextCards.byEngine[source.engine].byInstance ||= {};
        nextCards.byEngine[source.engine].byInstance[source.instanceId] = entries[source.value];
      }
    }
    // Audit label derived from stored hashes only, never from a client ranking
    // reason: a null-SHA source stays an explicit rebinding (P10.1).
    const evidence = kind === "relocate" ? "relocation" : !source.contentHash ? "unverified-source" : source.contentHash === target.contentHash ? "equal-content" : "different-content";
    results.push({ sourceId: source.instanceId, targetId: target.instanceId, kind, evidence, engine: source.engine,
      sourceValue: source.value, targetValue: target.value, sourceRootId: source.rootId, targetRootId: target.rootId, sourceHash: source.contentHash, targetHash: target.contentHash });
  }
  if (results.length) nextRegistry.revision += 1;
  return { registry: nextRegistry, state: nextState, cards: nextCards, workspacePatches, cardPatches, galleryPatches, conflicts, results };
}
