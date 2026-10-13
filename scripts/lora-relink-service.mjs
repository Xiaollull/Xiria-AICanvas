import { randomUUID } from "node:crypto";
import path from "node:path";
import { scanLoraInventory, matchLoraInventory, registerInitialInventory, registerObservedInventory, registerLegacyReferences, readLoraRegistry, observeLoraFile, publicLoraInventory, libraryPaths, libraryError, sameObservedInstance } from "./lora-library.mjs";
import { LoraCoordinator, withLibraryLease, publishedLibraryVersion, readJsonIfPresent, documentDigest, coordinationPaths, recoverAbandonedLibraryLease } from "./lora-coordinator.mjs";
import { planLoraRelink } from "./lora-relink-plan.mjs";
import { resolveRepairDirectory, listRepairDirectories, inRepairScope, scanCovers, withinPrefix, fullRootScan } from "./lora-repair-scope.mjs";
import { repairLifecycle } from "./lora-repair-ranking.mjs";
import { emptyLoraCardStore, normalizeLoraCardStore } from "../src/lora-cards.js";
import { READY_LORA_ENGINES } from "../src/lora-model-scope.js";
import { resolveLibraryReference, INSTANCE_ID } from "../src/lora-identity.js";
import { parseSharedRef } from "../src/shared-model-refs.js";
import { assertWorkspaceIdentity } from "./lora-published-store.mjs";
import { inspectMountedLoras } from "../src/lora-state.js";

const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const fields = ["value", "instanceId", "contentHash", "contentRevision", "identitySchema", "originalReference", "category"];
function associationPatches(before, after, prefix = [], anchors = []) {
  const changes = [];
  const walk = (a, b, address, guards) => {
    if (equal(a, b)) return;
    if (a && b && typeof a.value === "string" && typeof b.value === "string") {
      const keys = fields.filter((key) => !equal(a[key], b[key]));
      changes.push({ address, anchors: guards, expectedInstanceId: b.instanceId, before: Object.fromEntries(keys.map((key) => [key, a[key]])), after: Object.fromEntries(keys.map((key) => [key, b[key]])), keys });
      return;
    }
    if (a && b && typeof a === "object" && typeof b === "object" && Array.isArray(a) === Array.isArray(b)) {
      if (Array.isArray(a) && a.length !== b.length) { changes.push({ address, anchors: guards, beforeValue: a, afterValue: b, scalar: true }); return; }
      for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
        const nextGuards = a.id && b.id && a.id === b.id ? [...guards, { address, id: a.id }] : guards;
        walk(a[key], b[key], [...address, Array.isArray(a) ? Number(key) : key], nextGuards);
      }
      return;
    }
    // Exact prompt-span edits can later add a field-owned patch; never recurse
    // arbitrary text to invent LoRA matches.
    changes.push({ address, anchors: guards, beforeValue: a, afterValue: b, scalar: true });
  };
  walk(before, after, prefix, anchors);
  return changes;
}
function at(document, address) { return address.reduce((value, key) => value?.[key], document); }
function inverse(document, patches) {
  const value = structuredClone(document), conflicts = [];
  for (const patch of [...patches].reverse()) {
    const current = at(value, patch.address);
    if (patch.anchors.some((guard) => at(value, guard.address)?.id !== guard.id)) { conflicts.push({ address: patch.address, reason: "owner-moved-or-edited" }); continue; }
    const valid = patch.scalar ? equal(current, patch.afterValue) : current && current.instanceId === patch.expectedInstanceId && patch.keys.every((key) => equal(current[key], patch.after[key]));
    if (!valid) { conflicts.push({ address: patch.address, reason: "association-edited-after-repair" }); continue; }
    if (patch.scalar) {
      const parent = at(value, patch.address.slice(0, -1)), key = patch.address.at(-1);
      if (patch.beforeValue === undefined) delete parent[key]; else parent[key] = structuredClone(patch.beforeValue);
    } else {
      for (const key of patch.keys) { if (!Object.hasOwn(patch.before, key)) delete current[key]; else current[key] = structuredClone(patch.before[key]); }
    }
  }
  return { value, conflicts };
}

/** Registered-root adapter + authoritative management operations. No startup
 * inventory migration, no implicit confirmation and no tensor writes.
 */
export class LoraRelinkService {
  constructor({ stateDirectory, gallery, getRoots, admitIdle, validatePath, onPublished = async () => {}, fault, projectRoot }) {
    Object.assign(this, { stateDirectory, gallery, getRoots, admitIdle, validatePath, onPublished, projectRoot });
    this.scans = new Map(); this.hashCache = new Map(); this.recovering = null;
    this.coordinator = new LoraCoordinator({ stateDirectory, gallery, admitIdle,
      validate: (request) => this.validate(request), fault });
  }
  async ensureRecovered() {
    const lease = await readJsonIfPresent(path.join(coordinationPaths(this.stateDirectory).lock, "lease.json"));
    if (lease?.pid) {
      try { process.kill(lease.pid, 0); }
      catch (error) { if (error.code === "ESRCH") await recoverAbandonedLibraryLease(this.stateDirectory); }
    }
    try { await publishedLibraryVersion(this.stateDirectory); return; }
    catch (error) {
      if (!this.recovering) this.recovering = this.coordinator.recover().finally(() => { this.recovering = null; });
      await this.recovering;
    }
  }
  async access(action) {
    await this.ensureRecovered();
    return withLibraryLease(this.stateDirectory, async (token) => action(token, await publishedLibraryVersion(this.stateDirectory)));
  }
  async snapshot(token) {
    const state = await readJsonIfPresent(path.join(this.stateDirectory, "ui-state.json"), {});
    assertWorkspaceIdentity(state);
    const cards = await readJsonIfPresent(path.join(this.stateDirectory, "lora-cards/cards.json"), emptyLoraCardStore());
    if (normalizeLoraCardStore(cards).fatal || normalizeLoraCardStore(cards).reset) throw libraryError("Unreadable card schema; repair disabled without resetting data", 503);
    const gallery = await this.gallery.list(token);
    if ((gallery.cards || []).some((card) => inspectMountedLoras(card.settings?.loras).fatal)) throw libraryError("Unrecognized Gallery identity; original settings retained", 503);
    return { state, cards, galleryCards: gallery.cards || [] };
  }
  references(snapshot, roots, registry) {
    const found = [];
    // A confirmed-deleted (retired) source's path-keyed user card is archived
    // presentation, not an active source to recollect (P10.2).
    const retired = (engine, value) => Boolean(registry) && Object.values(registry.instances).some((item) => item.engine === engine && item.value === value && item.retired)
      && !Object.values(registry.instances).some((item) => item.engine === engine && item.value === value && !item.retired && item.status === "present");
    const add = (engine, item) => {
      if (!READY_LORA_ENGINES.includes(engine) || typeof item?.value !== "string") return;
      const shared = parseSharedRef(item.value);
      const rootId = shared ? shared.rootId : `local-${engine}`;
      if (!roots.some((root) => root.engine === engine && root.id === rootId)) return;
      found.push({ ...item, engine, rootId });
    };
    const ws = snapshot.state.workspace || {};
    for (const engine of READY_LORA_ENGINES) {
      for (const item of ws.mountedLorasByEngine?.byEngine?.[engine] || []) add(engine, item);
      for (const group of ws.loraGroupsByEngine?.byEngine?.[engine] || []) for (const item of group.members || []) add(engine, item);
      for (const value of Object.keys(snapshot.cards.byEngine?.[engine]?.loras || {})) if (!retired(engine, value)) add(engine, { value });
    }
    for (const item of ws.loras || []) add(ws.model, item);
    for (const card of snapshot.galleryCards) for (const item of card.settings?.loras || []) add(card.settings.model, item);
    return found;
  }
  /** Normalized P10.2 scope. Blank source = the engines' local roots (filter
   * only); blank target = library-wide candidates. Server-authoritative. */
  async resolveScope({ engines, source, target }, roots) {
    const options = { roots, engines, projectRoot: this.projectRoot || process.cwd() };
    const resolved = { source: await resolveRepairDirectory(source, { ...options, role: "source" }), target: await resolveRepairDirectory(target, { ...options, role: "target" }) };
    resolved.sourceEntries = resolved.source?.entries || roots.filter((root) => engines.includes(root.engine) && root.id === `local-${root.engine}`).map((root) => ({ engine: root.engine, rootId: root.id, prefix: "", exists: true, label: "本地" }));
    return resolved;
  }
  /** Folder navigation and inline validation; never opens or hashes a file. */
  async scope({ engines = READY_LORA_ENGINES, source, target } = {}) {
    if (!Array.isArray(engines) || !engines.length || engines.some((engine) => !READY_LORA_ENGINES.includes(engine))) throw libraryError("Invalid scan engine scope", 400);
    // All roots are passed so a folder of another engine is named, never silently re-scoped.
    const roots = await this.getRoots();
    const directories = await listRepairDirectories(roots, engines);
    const check = async (input, role) => {
      try { return { ok: true, ...(await resolveRepairDirectory(input, { roots, engines, projectRoot: this.projectRoot || process.cwd(), role })) }; }
      catch (error) { if (error.statusCode !== 400) throw error; return { ok: false, error: error.message }; }
    };
    return { engines, directories, ...(source !== undefined ? { source: await check(source, "source") } : {}), ...(target !== undefined ? { target: await check(target, "target") } : {}) };
  }
  async scan({ engines = READY_LORA_ENGINES, rootIds, maxBytes = 16 * 1024 ** 3, source, target } = {}, signal) {
    if (!Array.isArray(engines) || !engines.length || engines.some((engine) => !READY_LORA_ENGINES.includes(engine))) throw libraryError("Invalid scan engine scope", 400);
    if (rootIds !== undefined && (!Array.isArray(rootIds) || !rootIds.every((id) => typeof id === "string"))) throw libraryError("Invalid root scope", 400);
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw libraryError("Invalid hash budget", 400);
    const allRoots = await this.getRoots(), admittedRoots = allRoots.filter((root) => engines.includes(root.engine));
    const scope = await this.resolveScope({ engines, source, target }, allRoots);
    let roots = admittedRoots.filter((root) => !rootIds || rootIds.includes(root.id));
    if (scope.target) {
      // Observe only the source and target folders (truthful, narrowed coverage).
      roots = roots.flatMap((root) => {
        const prefixes = [...scope.target.entries, ...scope.sourceEntries].filter((entry) => entry.engine === root.engine && entry.rootId === root.id).map((entry) => entry.prefix);
        if (!prefixes.length) return [];
        const minimal = [...new Set(prefixes)].filter((prefix, _index, all) => !all.some((other) => other !== prefix && withinPrefix(prefix, other)));
        return [minimal.includes("") ? root : { ...root, scopeDirectories: minimal, narrowed: true }];
      });
    }
    if (!roots.length) throw libraryError("No admitted roots in this scope", 400);
    const scan = await scanLoraInventory(roots, { signal, hashCache: this.hashCache, maxBytes: Math.min(32 * 1024 ** 3, Math.max(1, maxBytes)),
      fileIdentity: process.platform === "win32" ? (file) => this.gallery.identity(file) : undefined });
    scan.domain = { scoped: Boolean(scope.source || scope.target), fullRoots: fullRootScan(scan) && roots.length === admittedRoots.length,
      source: scope.sourceEntries.map(({ engine, rootId, prefix, exists, label }) => ({ engine, rootId, prefix, exists, label })),
      target: scope.target ? scope.target.entries.map(({ engine, rootId, prefix, label }) => ({ engine, rootId, prefix, label })) : null };
    return this.access(async (token, revision) => {
      const savedRegistry = await readLoraRegistry(this.stateDirectory);
      const snapshot = await this.snapshot(token);
      const initial = Object.keys(savedRegistry.instances).length === 0;
      // A narrowed scan never becomes a (partial) initial whole-library inventory.
      let registry = initial && scan.complete && scan.domain.fullRoots ? registerInitialInventory(savedRegistry, scan) : savedRegistry;
      registry = structuredClone(registry);
      if (scan.complete) for (const [id, record] of Object.entries(registry.instances)) {
        const observed = scan.observations.find((item) => item.value === record.value && sameObservedInstance(record, item));
        if (observed) registry.instances[id] = { ...record, ...observed, lastCompleteObservation: scan.scanId };
      }
      // Only references inside the observed coverage are staged; outside ones are unobserved, not missing.
      registry = registerLegacyReferences(registry, this.references(snapshot, admittedRoots, registry).filter((reference) => scanCovers(scan, reference)), { ...scan,
        roots: admittedRoots.map((root) => ({ engine: root.engine, rootId: root.id })), complete: scan.complete && scan.domain.fullRoots }).registry;
      for (const observation of scan.observations) {
        const matches = Object.values(registry.instances).filter((record) => record.value === observation.value && sameObservedInstance(record, observation));
        if (matches.length > 1) throw libraryError("Conflicting current identities; review inventory");
        observation.instanceId = matches[0]?.instanceId || randomUUID();
      }
      // "Existing copy" is judged against the last committed baseline, never
      // against the provisional registration this first scan just made.
      const allowed = scope.target ? (item) => inRepairScope(scope.target.entries, item) : undefined;
      const inSource = (item) => inRepairScope(scope.sourceEntries, item);
      const rows = matchLoraInventory(registry, scan, { baseline: savedRegistry, candidate: allowed }).filter((row) => inSource({ engine: row.engine, rootId: registry.instances[row.sourceId].rootId, value: row.sourceValue }));
      for (const source of Object.values(registry.instances)) {
        if (engines.includes(source.engine) && inSource(source) && !rows.some((row) => row.sourceId === source.instanceId)) rows.push({ sourceId: source.instanceId, engine: source.engine, sourceValue: source.value, status: "unobserved", confidence: "unresolved", candidates: [], candidateCount: 0, suggestion: { state: "none" }, recommendation: null });
      }
      for (const row of rows) {
        const source = registry.instances[row.sourceId];
        const rule = registry.substitutions.find((item) => item.engine === source.engine && item.sourceId === source.instanceId && item.sourceHash === source.contentHash);
        const target = rule && registry.instances[rule.targetId];
        Object.assign(row, { sourceContentHash: source.contentHash, rootId: source.rootId, impacts: this.impacts(snapshot, source),
          ...(target ? { existingRule: { targetId: target.instanceId, targetValue: target.value, targetRootId: target.rootId } } : {}),
          ...(source.retired ? { retired: { operationId: source.retired.operationId, at: source.retired.at } } : {}) });
        if (row.status !== "unchanged") Object.assign(row, repairLifecycle(row, source, registry, scan));
      }
      const session = { scan, registry, savedRegistry, snapshot, roots, revision, initial, scope, at: Date.now() };
      if (this.scans.size >= 8) this.scans.delete(this.scans.keys().next().value);
      this.scans.set(scan.scanId, session);
      const publicScan = publicLoraInventory(scan);
      publicScan.observations = publicScan.observations.map((item) => {
        const observed = scan.observations.find((row) => row.engine === item.engine && row.rootId === item.rootId && row.value === item.value);
        return { ...item, instanceId: observed.instanceId, ...(allowed && !allowed(item) ? { outsideTarget: true } : {}) };
      });
      return { ...publicScan, domain: scan.domain, revision, initial, rows };
    });
  }
  impacts(snapshot, source) {
    const hit = (item) => item.instanceId ? item.instanceId === source.instanceId : item.value === source.value;
    const ws = snapshot.state.workspace || {};
    const mounts = (ws.mountedLorasByEngine?.byEngine?.[source.engine] || (ws.model === source.engine ? ws.loras : []) || []).filter(hit).length;
    const groups = (ws.loraGroupsByEngine?.byEngine?.[source.engine] || []).filter((group) => (group.members || []).some(hit)).map(({ id, name, enabled }) => ({ id, name, enabled }));
    const cards = snapshot.galleryCards.filter((card) => card.settings.model === source.engine && (card.settings.loras || []).some(hit)).map(({ id, title }) => ({ id, title }));
    return { mounts, groups, cards };
  }
  session(scanId) {
    const session = this.scans.get(scanId);
    if (!session || Date.now() - session.at > 15 * 60 * 1000) throw libraryError("Scan expired; rescan without losing your choices");
    return session;
  }
  async plan(payload, token, revision) {
    const session = this.session(payload.scanId);
    if (!session.scan.complete || session.revision !== revision) throw libraryError("Scan baseline changed or incomplete; review the draft");
    const registry = await readLoraRegistry(this.stateDirectory);
    if (!equal(registry, session.savedRegistry)) throw libraryError("Source identity baseline changed; review the draft");
    if (!Array.isArray(payload.selections || []) || (payload.selections || []).length > 10000) throw libraryError("Invalid selection table", 400);
    for (const row of payload.selections || []) {
      if (!INSTANCE_ID.test(row?.sourceId || "") || !row.scope || typeof row.scope !== "object" || Array.isArray(row.scope)) throw libraryError("Invalid row/scope", 400);
      for (const key of ["groupIds", "galleryIds", "presetIds"]) if (row.scope[key] !== undefined && (!Array.isArray(row.scope[key]) || !row.scope[key].every((value) => typeof value === "string"))) throw libraryError("Invalid consumer scope", 400);
      const source = session.registry.instances[row.sourceId];
      if (row.target && !["relocate", "replace", "delete"].includes(row.target.kind)) throw libraryError("Choose relocation, explicit replacement or confirmed deletion", 400);
      if (row.target?.kind === "relocate" && source && !scanCovers(session.scan, source)) throw libraryError("Unobserved source cannot be certified as relocation; choose explicit replacement");
      // An explicit target folder bounds every choice, including typed references.
      if (!row.skipped && row.target && row.target.kind !== "delete" && source && session.scope?.target
        && !inRepairScope(session.scope.target.entries, { engine: source.engine, rootId: row.target.rootId, value: row.target.value })) throw libraryError("目标不在所选目标目录内；请更改目标目录并重新扫描", 400);
    }
    if (payload.registerBaseline && !session.scan.domain?.fullRoots) throw libraryError("完整观察基线需要覆盖全部根目录的扫描；请清空源目录与目标目录后重新扫描");
    const snapshot = await this.snapshot(token);
    const plan = planLoraRelink({ registry: session.registry, scan: session.scan, ...session.snapshot, selections: payload.selections || [], operationId: payload.operationId || null });
    const changes = associationPatches(session.snapshot.state, plan.state);
    // Only affected consumer lists must retain their scanned baseline. Unrelated
    // fields are rebuilt from the latest snapshot, never whole old-state restored.
    const currentPlan = planLoraRelink({ registry: session.registry, scan: session.scan, ...snapshot, selections: payload.selections || [], operationId: payload.operationId || null });
    if (payload.registerBaseline) currentPlan.registry = registerObservedInventory(currentPlan.registry, session.scan);
    for (const patch of plan.workspacePatches) {
      const oldList = at(session.snapshot.state, patch.address), latest = at(snapshot.state, patch.address);
      if (!equal(oldList, latest)) throw libraryError("Selected workspace consumer changed since scan");
    }
    for (const patch of plan.galleryPatches) {
      const before = session.snapshot.galleryCards.find((card) => card.id === patch.cardId)?.settings;
      const latest = snapshot.galleryCards.find((card) => card.id === patch.cardId)?.settings;
      if (!equal(before?.loras, latest?.loras) || before?.model !== latest?.model) throw libraryError("Selected Gallery consumer changed since scan");
      for (const key of ["positive", "negative"]) if (!equal(patch.before[key], patch.after[key]) && !equal(before?.[key], latest?.[key])) throw libraryError("Selected Gallery tag field changed since scan");
    }
    for (const patch of plan.cardPatches) {
      for (const value of [patch.from, patch.to]) {
        if (!equal(session.snapshot.cards.byEngine?.[patch.engine]?.loras?.[value], snapshot.cards.byEngine?.[patch.engine]?.loras?.[value])) throw libraryError("Selected user card changed since scan");
      }
      if (!equal(session.snapshot.cards.byEngine?.[patch.engine]?.byInstance?.[patch.after.instanceId], snapshot.cards.byEngine?.[patch.engine]?.byInstance?.[patch.after.instanceId])) throw libraryError("Selected instance card changed since scan");
    }
    for (const engine of READY_LORA_ENGINES) for (const kind of ["loras", "byInstance"]) {
      const before = session.snapshot.cards.byEngine?.[engine]?.[kind] || {}, after = plan.cards.byEngine?.[engine]?.[kind] || {};
      for (const key of new Set([...Object.keys(before), ...Object.keys(after)])) {
        if (!equal(before[key], after[key]) && !equal(before[key], snapshot.cards.byEngine?.[engine]?.[kind]?.[key])) throw libraryError("Selected user card changed since scan");
      }
    }
    const targets = (payload.selections || []).filter((row) => !row.skipped && row.target && row.target.kind !== "delete").map((row) => {
      const source = session.registry.instances[row.sourceId];
      if (!source) throw libraryError("Unknown source row", 400);
      const observation = session.scan.observations.find((item) => item.engine === source.engine && item.rootId === row.target.rootId && item.value === row.target.value);
      if (!observation) throw libraryError("Target is not an admitted scan observation");
      return observation;
    });
    if (payload.registerBaseline) targets.push(...session.scan.observations);
    const sources = (payload.selections || []).filter((row) => !row.skipped && row.target).map((row) => {
      const source = session.registry.instances[row.sourceId];
      const covered = scanCovers(session.scan, source);
      return { source, covered, observed: session.scan.observations.find((item) => item.engine === source.engine && item.rootId === source.rootId && item.value === source.value) || null };
    });
    const cardFieldsBefore = structuredClone(snapshot.cards), cardFieldsAfter = structuredClone(currentPlan.cards);
    for (const move of currentPlan.cardPatches) for (const document of [cardFieldsBefore, cardFieldsAfter]) {
      const scope = document.byEngine?.[move.engine];
      if (scope) { delete scope.loras[move.from]; delete scope.loras[move.to]; if (scope.byInstance) delete scope.byInstance[move.after.instanceId]; }
    }
    const ownership = {
      workspace: associationPatches(snapshot.state, currentPlan.state),
      cardMoves: currentPlan.cardPatches,
      cardFields: associationPatches(cardFieldsBefore, cardFieldsAfter).filter((patch) => patch.address[0] !== "schemaVersion"),
      gallery: currentPlan.galleryPatches.map((patch) => ({ cardId: patch.cardId, model: patch.before.model, patches: associationPatches(patch.before, patch.after) })),
      substitutions: [...new Map([...registry.substitutions, ...currentPlan.registry.substitutions].map((rule) => [`${rule.engine}:${rule.sourceId}:${rule.sourceHash}`, rule])).values()].map((rule) => {
        const match = (item) => item.engine === rule.engine && item.sourceId === rule.sourceId && item.sourceHash === rule.sourceHash;
        return { engine: rule.engine, sourceId: rule.sourceId, sourceHash: rule.sourceHash,
          before: registry.substitutions.find(match) || null, after: currentPlan.registry.substitutions.find(match) || null };
      }).filter((row) => !equal(row.before, row.after)),
      tagConfirmations: Object.entries(currentPlan.registry.instances).flatMap(([id, record]) => !equal(registry.instances[id]?.confirmedTagRefs, record.confirmedTagRefs)
        ? [{ sourceId: id, before: registry.instances[id]?.confirmedTagRefs, after: record.confirmedTagRefs }] : []),
      // Confirmed-deletion tombstones are owned like any other registry field.
      retirements: Object.entries(currentPlan.registry.instances).flatMap(([id, record]) => !equal(registry.instances[id]?.retired, record.retired)
        ? [{ sourceId: id, before: registry.instances[id]?.retired, after: record.retired }] : []),
    };
    if (!currentPlan.results.length && !payload.registerBaseline) return { noOp: true, conflicts: [], results: [] };
    return { ...currentPlan, snapshot, targets, sources, ownership, changes };
  }
  async preview(payload) {
    return this.access(async (token, revision) => {
      const plan = await this.plan(payload, token, revision);
      if (!plan.noOp) await this.validate({ validation: { targets: plan.targets, sources: plan.sources } });
      return { revision, noOp: Boolean(plan.noOp), conflicts: plan.conflicts, results: plan.results,
        affected: { workspace: plan.workspacePatches?.length || 0, cards: plan.cardPatches?.length || 0, gallery: plan.galleryPatches?.length || 0 } };
    });
  }
  async validate(request) {
    if (request.recovering && (!request.validation || (!["undo", "maintenance-reset"].includes(request.validation.type) && (!/^[0-9a-f]{64}$/.test(request.validation.clientDigest || "")
      || !Array.isArray(request.validation.targets) || !Array.isArray(request.validation.sources))))) throw libraryError("Unsupported/damaged service validation journal; evidence retained", 503);
    // Undo and controlled maintenance carry no file selections to revalidate.
    if (["undo", "maintenance-reset"].includes(request.validation?.type)) return;
    const roots = await this.getRoots();
    for (const expected of request.validation?.sources || []) {
      if (!expected.covered) continue;
      const source = expected.source;
      const root = roots.find((item) => item.engine === source.engine && item.id === source.rootId && item.enabled !== false);
      if (!root) throw libraryError("Source root admission changed; rescan");
      const relative = parseSharedRef(source.value)?.relativePath || source.value;
      let actual = null;
      try { actual = await observeLoraFile(root, relative, { force: true }); }
      catch (error) { if (error.code !== "ENOENT") throw libraryError("Source is inaccessible; rescan"); }
      if (Boolean(actual) !== Boolean(expected.observed) || (actual && (actual.contentHash !== expected.observed.contentHash || actual.fingerprint !== expected.observed.fingerprint))) throw libraryError("Source observation changed since scan; review the draft");
    }
    for (const expected of request.validation?.targets || []) {
      const root = roots.find((item) => item.engine === expected.engine && item.id === expected.rootId && item.enabled !== false);
      if (!root) throw libraryError("Selected target root disabled or changed");
      let actual;
      try { actual = await observeLoraFile(root, expected.relative, { force: true }); }
      catch { throw libraryError("Target renamed, removed or unreadable; review the draft"); }
      if (actual.contentHash !== expected.contentHash || actual.fingerprint !== expected.fingerprint) throw libraryError("Target renamed, removed or changed; review the draft");
    }
  }
  async commit(payload) {
    if (!INSTANCE_ID.test(payload.operationId || "")) throw libraryError("Valid idempotent operation id required", 400);
    const clientDigest = documentDigest({ scanId: payload.scanId, selections: payload.selections || [], registerBaseline: Boolean(payload.registerBaseline) });
    return this.access(async (token, revision) => {
      const old = await readJsonIfPresent(path.join(coordinationPaths(this.stateDirectory).operations, `${payload.operationId}.json`));
      if (old) {
        if (old.validation?.clientDigest !== clientDigest) throw libraryError("Operation id belongs to a different draft/scope");
        if (old.phase === "published") return { status: "published", operationId: old.operationId, revision: old.revision, replay: true };
        throw libraryError("Operation must complete recovery before retry");
      }
      const plan = await this.plan(payload, token, revision);
      if (plan.noOp) return { status: "no-op", operationId: null, revision };
      if (plan.conflicts.length) throw Object.assign(libraryError("Resolve binding/card conflicts before confirmation"), { conflicts: plan.conflicts });
      const registryBefore = await readJsonIfPresent(libraryPaths(this.stateDirectory).registry);
      const stateBefore = await readJsonIfPresent(path.join(this.stateDirectory, "ui-state.json"));
      const cardsBefore = await readJsonIfPresent(path.join(this.stateDirectory, "lora-cards/cards.json"));
      const files = { registry: { beforeDigest: documentDigest(registryBefore), after: plan.registry } };
      if (!equal(plan.snapshot.state, plan.state)) files.uiState = { beforeDigest: documentDigest(stateBefore), after: plan.state };
      if (!equal(plan.snapshot.cards, plan.cards)) files.cards = { beforeDigest: documentDigest(cardsBefore), after: plan.cards };
      const request = { operationId: payload.operationId, expectedRevision: revision, files, galleryPatches: plan.galleryPatches,
        validation: { clientDigest, targets: plan.targets, sources: plan.sources, ownership: plan.ownership } };
      const result = await this.coordinator.commit(request, { token });
      const cacheWarnings = await this.onPublished(plan, token);
      return { ...result, results: plan.results, ...(cacheWarnings?.length ? { cacheWarnings } : {}) };
    });
  }
  async undo({ operationId, undoOperationId }) {
    if (!INSTANCE_ID.test(operationId || "") || !INSTANCE_ID.test(undoOperationId || "")) throw libraryError("Valid operation/undo ids required", 400);
    return this.access(async (token, revision) => {
      const replay = await readJsonIfPresent(path.join(coordinationPaths(this.stateDirectory).operations, `${undoOperationId}.json`));
      if (replay) {
        if (replay.validation?.type !== "undo" || replay.validation.operationId !== operationId) throw libraryError("Undo id belongs to a different operation");
        if (replay.phase !== "published") throw libraryError("Undo recovery must finish before retry");
        return { status: "published", operationId: replay.operationId, revision: replay.revision, conflicts: replay.validation.conflicts || [], replay: true };
      }
      const file = path.join(coordinationPaths(this.stateDirectory).operations, `${operationId}.json`);
      const operation = await readJsonIfPresent(file);
      if (((await readLoraRegistry(this.stateDirectory)).maintenance || []).some((entry) => entry.supersedes.includes(operationId))) throw libraryError("该操作涉及的记录已被维护重置；仅保留审计，不能撤销");
      if (!operation || operation.phase !== "published" || !operation.validation?.ownership) throw libraryError("Operation has no published owned associations to undo");
      const owned = operation.validation.ownership, snapshot = await this.snapshot(token), conflicts = [];
      const ws = inverse(snapshot.state, owned.workspace);
      conflicts.push(...ws.conflicts);
      const cardUndo = inverse(snapshot.cards, owned.cardFields || []);
      conflicts.push(...cardUndo.conflicts.map((conflict) => ({ ...conflict, store: "cards" })));
      const cards = cardUndo.value;
      for (const move of owned.cardMoves) {
        const entries = cards.byEngine?.[move.engine]?.loras;
        const scope = cards.byEngine?.[move.engine], current = entries?.[move.to];
        const identityKeys = ["identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference"];
        if (!entries || entries[move.from] || !current || identityKeys.some((key) => !equal(current[key], move.after[key]))
          || (scope.byInstance?.[move.after.instanceId] && !equal(scope.byInstance[move.after.instanceId], current))) { conflicts.push({ card: move.to, reason: "card-association-edited" }); continue; }
        // Carry later user text/cover edits, not an old whole card snapshot.
        const restored = { ...entries[move.to] };
        for (const key of ["identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference"]) {
          if (Object.hasOwn(move.before, key)) restored[key] = structuredClone(move.before[key]); else delete restored[key];
        }
        entries[move.from] = restored; delete entries[move.to];
        const byInstance = cards.byEngine[move.engine].byInstance;
        if (byInstance) {
          delete byInstance[move.after.instanceId];
          if (restored.instanceId) byInstance[restored.instanceId] = restored;
        }
      }
      const galleryPatches = [];
      for (const ownership of owned.gallery) {
        const card = snapshot.galleryCards.find((item) => item.id === ownership.cardId);
        if (!card || card.settings.model !== ownership.model) { conflicts.push({ cardId: ownership.cardId, reason: "card-engine-or-owner-changed" }); continue; }
        const result = inverse(card.settings, ownership.patches);
        conflicts.push(...result.conflicts.map((conflict) => ({ ...conflict, cardId: card.id })));
        if (!equal(card.settings, result.value)) galleryPatches.push({ cardId: card.id, before: card.settings, after: result.value });
      }
      const registry = await readLoraRegistry(this.stateDirectory), nextRegistry = structuredClone(registry);
      for (const confirmation of owned.tagConfirmations || []) {
        const record = nextRegistry.instances[confirmation.sourceId];
        if (!record || !equal(record.confirmedTagRefs, confirmation.after)) { conflicts.push({ sourceId: confirmation.sourceId, reason: "tag-confirmation-edited" }); continue; }
        if (confirmation.before === undefined) delete record.confirmedTagRefs; else record.confirmedTagRefs = structuredClone(confirmation.before);
      }
      for (const retirement of owned.retirements || []) {
        const record = nextRegistry.instances[retirement.sourceId];
        if (!record || !equal(record.retired, retirement.after)) { conflicts.push({ sourceId: retirement.sourceId, reason: "retirement-edited" }); continue; }
        if (retirement.before === undefined) delete record.retired; else record.retired = structuredClone(retirement.before);
      }
      for (const rule of owned.substitutions) {
        const evidence = rule.after || rule.before;
        const match = (item) => item.sourceId === rule.sourceId && item.engine === (rule.engine || evidence?.engine) && item.sourceHash === (rule.sourceHash ?? evidence?.sourceHash);
        const now = registry.substitutions.find(match) || null;
        if (!equal(now, rule.after)) { conflicts.push({ sourceId: rule.sourceId, reason: "history-rule-edited" }); continue; }
        nextRegistry.substitutions = nextRegistry.substitutions.filter((item) => !match(item));
        if (rule.before) nextRegistry.substitutions.push(rule.before);
      }
      const files = {};
      if (!equal(snapshot.state, ws.value)) files.uiState = { beforeDigest: documentDigest(snapshot.state), after: ws.value };
      if (!equal(snapshot.cards, cards)) files.cards = { beforeDigest: documentDigest(snapshot.cards), after: cards };
      if (!equal(registry, nextRegistry)) { nextRegistry.revision += 1; files.registry = { beforeDigest: documentDigest(registry), after: nextRegistry }; }
      const result = await this.coordinator.commit({ operationId: undoOperationId || randomUUID(), expectedRevision: revision, files, galleryPatches,
        validation: { type: "undo", operationId, conflicts } }, { token });
      return { ...result, conflicts };
    });
  }
  async resolveXYZ(draft) {
    return this.access(async (token) => {
      const projection = await this.gallery.xyzReferences(draft, token);
      if (projection.error_count) throw libraryError(projection.errors.map((item) => item.message).join("; "), 422);
      const references = projection.references;
      const resolved = await this.resolve(draft.engine, references, { token });
      return { ...resolved, effectiveReferences: references };
    });
  }
  /** QPlan2/P2: the same authoritative resolution for a reference set the caller already projected (search rounds). */
  async resolveReferences(engine, references) {
    return this.access(async (token) => ({ ...(await this.resolve(engine, references, { token })), effectiveReferences: references }));
  }
  async resolve(engine, references, options = {}) {
    const apply = async (_token, revision) => {
      if (!READY_LORA_ENGINES.includes(engine) || !Array.isArray(references) || references.some((item) => !item || typeof item.value !== "string")) throw libraryError("Invalid engine/reference list", 400);
      let registry;
      try { registry = await readLoraRegistry(this.stateDirectory); }
      catch (error) {
        // An unused/broken management registry must not newly disable ordinary
        // path-only generation before the first confirmed publication.
        if (revision || references.some((item) => item.instanceId || item.contentHash || item.identitySchema || item.originalReference)) throw error;
        registry = { schemaVersion: 1, revision: 0, instances: {}, substitutions: [] };
      }
      const resolved = [];
      for (const reference of references) {
        if (reference.enabled === false) continue;
        let verdict = resolveLibraryReference(registry, engine, reference, { ...options, caseInsensitive: process.platform === "win32" });
        // A physical occupant can precede explicit inventory confirmation. Reads
        // must fail closed without registering it or changing any consumer.
        if (!reference.instanceId && verdict.status === "resolved") {
          const samePath = (a, b) => process.platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b;
          const origins = Object.values(registry.instances).filter((item) => item.engine === engine && (samePath(item.value, reference.value) || item.history.some((value) => samePath(value, reference.value))));
          const roots = await this.getRoots();
          for (const origin of origins) {
            const root = roots.find((item) => item.engine === engine && item.id === origin.rootId && item.enabled !== false);
            if (!root) continue;
            let actual;
            try { actual = await observeLoraFile(root, parseSharedRef(reference.value)?.relativePath || reference.value, { hashCache: this.hashCache,
              fileIdentity: process.platform === "win32" ? (file) => this.gallery.identity(file) : undefined }); }
            catch (error) { if (error.code === "ENOENT") continue; throw error; }
            const known = Object.values(registry.instances).filter((item) => item.status === "present" && samePath(item.value, reference.value) && sameObservedInstance(item, actual));
            if (!known.length && (!reference.contentHash || reference.contentHash === actual.contentHash)) {
              verdict = { status: "ambiguous-path", auditKind: "path-reuse", candidates: [...(verdict.candidates || []),
                { engine, rootId: actual.rootId, value: actual.value, contentHash: actual.contentHash, kind: "unconfirmed-current-occupant" }] };
              break;
            }
          }
        }
        const untouchedLegacy = !reference.instanceId && !reference.contentHash && verdict.status === "unresolved";
        if (!untouchedLegacy && verdict.status !== "resolved") throw Object.assign(libraryError(`LoRA reference is ${verdict.status}; choose an explicit target`), { resolution: verdict });
        const value = verdict.status === "resolved" ? verdict.value : reference.value;
        await this.validatePath(engine, value);
        if (verdict.status === "resolved") {
          const instance = registry.instances[verdict.instanceId];
          const roots = await this.getRoots();
          const root = roots.find((item) => item.engine === engine && item.id === instance.rootId && item.enabled !== false);
          if (!root) throw libraryError("LoRA root disabled; explicit repair required");
          const relative = parseSharedRef(value)?.relativePath || value;
          const actual = await observeLoraFile(root, relative, { hashCache: this.hashCache,
            fileIdentity: process.platform === "win32" ? (file) => this.gallery.identity(file) : undefined });
          if (!sameObservedInstance(instance, actual)) throw libraryError("LoRA instance or bytes changed; explicit repair required");
        }
        const origin = registry.instances[verdict.sourceId || reference.instanceId];
        const promptRewriteAllowed = Boolean(origin && (origin.confirmedTagRefs?.includes(reference.value)
          || registry.substitutions.some((rule) => rule.engine === engine && rule.sourceId === origin.instanceId && rule.tagsConfirmed && (!reference.contentHash || rule.sourceHash === reference.contentHash))));
        resolved.push({ ...reference, ...verdict, value, promptRewriteAllowed });
      }
      return { revision, references: resolved, loras: resolved.map((item) => ({ path: item.value, weight: item.weight })) };
    };
    return options.token ? apply(options.token, await publishedLibraryVersion(this.stateDirectory)) : this.access(apply);
  }
  async status() {
    return this.access(async (_token, revision) => {
      try { const registry = await readLoraRegistry(this.stateDirectory); return { revision, available: true, contractVersion: 1, instances: Object.keys(registry.instances).length }; }
      catch { return { revision, available: false, error: "Identity schema cannot be read; original data retained" }; }
    });
  }
}
