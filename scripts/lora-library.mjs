import { createHash, randomUUID } from "node:crypto";
import { lstat, open, readdir, readFile, realpath } from "node:fs/promises";
import path from "node:path";
import { READY_LORA_ENGINES } from "../src/lora-model-scope.js";
import { CONTENT_HASH, INSTANCE_ID } from "../src/lora-identity.js";
import { rankRepairRows, scanCovers } from "./lora-repair-ranking.mjs";

export const LIBRARY_SCHEMA = 1;
const engines = new Set(READY_LORA_ENGINES);
const native = new Set(["Anima", "Flux", "Flux2", "Krea2"]);
const legacyExtensions = new Set([".safetensors", ".ckpt", ".pt", ".pth"]);
export const libraryError = (message, statusCode = 409) => Object.assign(new Error(message), { statusCode });
export function emptyLoraRegistry() { return { schemaVersion: LIBRARY_SCHEMA, revision: 0, instances: {}, substitutions: [] }; }
export function libraryPaths(stateDirectory) {
  const directory = path.join(stateDirectory, "lora-library");
  return { directory, registry: path.join(directory, "registry.json"), operations: path.join(directory, "operations") };
}
export function assertRegistry(value) {
  if (!value || value.schemaVersion !== LIBRARY_SCHEMA || !Number.isSafeInteger(value.revision) || value.revision < 0
    || !value.instances || typeof value.instances !== "object" || Array.isArray(value.instances) || !Array.isArray(value.substitutions)) throw libraryError("Unsupported or damaged LoRA identity registry; original bytes retained", 503);
  for (const [id, instance] of Object.entries(value.instances)) {
    if (!instance || typeof instance !== "object" || !INSTANCE_ID.test(id) || instance.instanceId !== id || !engines.has(instance.engine) || !validRelative(instance.value)
      || typeof instance.rootId !== "string" || !/^[a-zA-Z0-9-]{1,64}$/.test(instance.rootId)
      || !((CONTENT_HASH.test(instance.contentHash) && Number.isSafeInteger(instance.contentRevision) && instance.contentRevision >= 1)
        || (instance.contentHash === null && instance.contentRevision === 0 && instance.status === "missing"))
      || !["present", "missing"].includes(instance.status) || !Array.isArray(instance.history)
      || !instance.history.every(validRelative)
      || (instance.confirmedTagRefs !== undefined && (!Array.isArray(instance.confirmedTagRefs) || !instance.confirmedTagRefs.every(validRelative)))
      || (instance.fsIdentity && (typeof instance.fsIdentity.key !== "string" || typeof instance.fsIdentity.creation !== "string"))
      || (instance.retired !== undefined && (!instance.retired || typeof instance.retired !== "object" || instance.status !== "missing" || typeof instance.retired.at !== "string"
        || !(instance.retired.operationId === null || INSTANCE_ID.test(instance.retired.operationId))))) throw libraryError("Invalid LoRA instance; repair disabled", 503);
  }
  // P10.2 maintenance audit: scoped resets and the operations they superseded.
  if (value.maintenance !== undefined && (!Array.isArray(value.maintenance) || !value.maintenance.every((entry) => entry && typeof entry === "object"
    && INSTANCE_ID.test(entry.operationId || "") && Array.isArray(entry.values) && Array.isArray(entry.supersedes) && entry.supersedes.every((id) => INSTANCE_ID.test(id))))) throw libraryError("Invalid LoRA maintenance audit; original data retained", 503);
  const seen = new Set();
  for (const rule of value.substitutions) {
    if (!rule || !engines.has(rule.engine) || rule.kind !== "replace" || !INSTANCE_ID.test(rule.sourceId || "") || !INSTANCE_ID.test(rule.targetId || "")
      || value.instances[rule.sourceId]?.engine !== rule.engine || value.instances[rule.targetId]?.engine !== rule.engine
      || !(rule.sourceHash === null || CONTENT_HASH.test(rule.sourceHash || "")) || !CONTENT_HASH.test(rule.targetHash || "")
      || (rule.tagsConfirmed !== undefined && typeof rule.tagsConfirmed !== "boolean")) throw libraryError("Invalid LoRA substitution; original data retained", 503);
    const key = `${rule.engine}:${rule.sourceId}:${rule.sourceHash}`;
    if (seen.has(key)) throw libraryError("Duplicate LoRA substitution; original data retained", 503);
    seen.add(key);
  }
  return value;
}
export async function readLoraRegistry(stateDirectory) {
  try { return assertRegistry(JSON.parse(await readFile(libraryPaths(stateDirectory).registry, "utf8"))); }
  catch (error) { if (error.code === "ENOENT") return emptyLoraRegistry(); throw error; }
}
export function validRelative(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 500 && !/[\x00-\x1f\\]/.test(value)
    && !value.startsWith("/") && !/^[a-z]:/i.test(value)
    && value.split("/").every((part) => part && part !== "." && part !== "..");
}
function contained(root, file) { const relative = path.relative(root, file); return relative && !relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative); }
export function fingerprint(stat) { return [stat.dev, stat.ino, stat.size, stat.mtimeNs, stat.ctimeNs, stat.birthtimeNs, stat.nlink].map(String).join(":"); }

/** Optional OS identity. Windows is explicitly unavailable without its native provider. */
export function statFileIdentity(stat, platform = process.platform) {
  if (platform === "win32" || stat.ino === 0n) return null;
  return { kind: "posix", key: `${stat.dev}:${stat.ino}`, links: Number(stat.nlink), creation: String(stat.birthtimeNs) };
}

export function sameObservedInstance(instance, observation, { content = true } = {}) {
  return Boolean(instance && observation && instance.engine === observation.engine && instance.rootId === observation.rootId
    && (!content || instance.contentHash === observation.contentHash)
    && (instance.fsIdentity ? observation.fsIdentity?.key === instance.fsIdentity.key && observation.fsIdentity?.creation === instance.fsIdentity.creation
      : instance.fingerprint && instance.fingerprint === observation.fingerprint));
}

/** Stream full bytes, never deserialize tensor/pickle data; detect ordinary concurrent changes. */
export async function observeLoraFile(root, relative, { signal, fileIdentity, hashCache, force = false, budget } = {}) {
  if (!engines.has(root.engine) || !validRelative(relative) || relative.startsWith("shared:")
    || !(native.has(root.engine) ? path.extname(relative).toLowerCase() === ".safetensors" : legacyExtensions.has(path.extname(relative).toLowerCase()))) throw libraryError("Invalid engine-relative LoRA target", 400);
  const rootPath = await realpath(root.directory);
  let candidate = rootPath;
  for (const part of relative.split("/")) {
    candidate = path.join(candidate, part);
    if ((await lstat(candidate)).isSymbolicLink()) throw libraryError("LoRA links are not admitted", 403);
  }
  const resolved = await realpath(candidate);
  if (!contained(rootPath, resolved)) throw libraryError("LoRA escapes registered root", 403);
  const handle = await open(resolved, "r");
  try {
    const before = await handle.stat({ bigint: true });
    if (!before.isFile()) throw libraryError("LoRA target is not a file", 400);
    const stamp = fingerprint(before);
    // Content cache may cross engine/root views of the SAME physical path.
    // Instance records remain separate random engine-scoped identities.
    const key = `${resolved}:${stamp}`;
    let contentHash = !force && hashCache?.get(key);
    if (!contentHash) {
      if (budget && before.size > BigInt(Math.max(0, budget.remaining))) throw libraryError("Hash byte budget exhausted", 413);
      if (budget) budget.remaining -= Number(before.size);
      const digest = createHash("sha256");
      const buffer = Buffer.alloc(1024 * 1024);
      let position = 0;
      for (;;) {
        signal?.throwIfAborted();
        const { bytesRead } = await handle.read(buffer, 0, buffer.length, position);
        if (!bytesRead) break;
        digest.update(buffer.subarray(0, bytesRead));
        position += bytesRead;
      }
      contentHash = digest.digest("hex");
    }
    const after = await handle.stat({ bigint: true });
    const named = await lstat(resolved, { bigint: true });
    if (stamp !== fingerprint(after) || stamp !== fingerprint(named)) throw libraryError("LoRA changed during observation");
    // A native identity provider can be an owner process per call (Windows).
    // The same resolved path with an unchanged full fingerprint (dev/ino/size/
    // times/links) is the same file object, so its identity is reused.
    const identityKey = `identity:${key}`;
    const fsIdentity = fileIdentity ? (!force && hashCache?.get(identityKey)) || await fileIdentity(resolved, after) : statFileIdentity(after);
    if (stamp !== fingerprint(await handle.stat({ bigint: true })) || stamp !== fingerprint(await lstat(resolved, { bigint: true }))) throw libraryError("LoRA changed during filesystem identity observation");
    if (hashCache) {
      while (hashCache.size >= 8192) hashCache.delete(hashCache.keys().next().value);
      hashCache.set(key, contentHash);
      if (fileIdentity && fsIdentity) hashCache.set(identityKey, fsIdentity);
    }
    return { engine: root.engine, rootId: root.id, relative,
      value: root.shared ? `shared:${root.id}/${relative}` : relative,
      contentHash, fingerprint: stamp, fsIdentity, size: Number(after.size) };
  } finally { await handle.close(); }
}

/** Explicit repair scan. It has no persistence/consumer/prune side effects. Roots are server-owned. */
export async function scanLoraInventory(roots, { signal, maxFiles = 10000, maxBytes = 512 * 1024 * 1024, hashCache = new Map(), fileIdentity } = {}) {
  const scan = { scanId: randomUUID(), complete: true, issues: [], observations: [], roots: [], metrics: { files: 0, hashedFiles: 0, hashedBytes: 0, cachedFiles: 0, elapsedMs: 0 } };
  const budget = { remaining: maxBytes }, started = performance.now();
  const issue = (root, relative, reason) => { scan.complete = false; scan.issues.push({ engine: root.engine, rootId: root.id, relative, reason }); };
  const seen = new Set();
  for (const root of roots) {
    if (!engines.has(root.engine) || typeof root.id !== "string" || !/^[a-zA-Z0-9-]{1,64}$/.test(root.id)
      || (root.shared && !/^[0-9a-f]{12}$/.test(root.id)) || !path.isAbsolute(root.directory)) throw libraryError("Invalid trusted root descriptor", 400);
    const rootKey = `${root.engine}:${root.id}`;
    if (seen.has(rootKey)) throw libraryError("Duplicate root descriptor", 400);
    seen.add(rootKey);
    // Label is presentation only; rootId stays the executable identity.
    // `prefixes` is the observed coverage (P10.2); `narrowed` marks a
    // user-scoped subset that must never pass as whole-root coverage.
    const entry = { engine: root.engine, rootId: root.id, ...(root.shared ? { shared: true } : {}),
      ...(typeof root.label === "string" && root.label ? { label: root.label.slice(0, 60) } : {}),
      prefixes: root.scopeDirectories || [""], ...(root.narrowed ? { narrowed: true } : {}) };
    scan.roots.push(entry);
    if (root.enabled === false) { issue(root, "", "root-disabled"); continue; }
    const walk = async (relative = "") => {
      if (signal?.aborted) { issue(root, relative, "cancelled"); return; }
      let entries;
      try { entries = await readdir(path.join(root.directory, relative), { withFileTypes: true }); }
      catch (error) { issue(root, relative, error.code || "unreadable"); return; }
      for (const entry of entries) {
        if (signal?.aborted) { issue(root, relative, "cancelled"); return; }
        const next = relative ? `${relative}/${entry.name}` : entry.name;
        if (entry.isSymbolicLink()) { issue(root, next, "link-not-scanned"); continue; }
        if (entry.isDirectory()) { await walk(next); continue; }
        if (!entry.isFile() || !(native.has(root.engine) ? path.extname(next).toLowerCase() === ".safetensors" : legacyExtensions.has(path.extname(next).toLowerCase()))) continue;
        if (scan.observations.length >= maxFiles) { issue(root, next, "file-budget"); return; }
        try {
          const before = budget.remaining, observation = await observeLoraFile(root, next, { signal, fileIdentity, hashCache, budget });
          scan.observations.push(observation);
          scan.metrics.files += 1;
          if (budget.remaining < before || !observation.size) { scan.metrics.hashedFiles += 1; scan.metrics.hashedBytes += before - budget.remaining; } else scan.metrics.cachedFiles += 1;
        }
        catch (error) { issue(root, next, signal?.aborted ? "cancelled" : error.code || `observation-${error.statusCode || "failed"}`); }
      }
    };
    for (const prefix of root.scopeDirectories || [""]) {
      if (prefix && !validRelative(prefix)) throw libraryError("Invalid trusted scan scope", 400);
      // A moved-away source folder is observed as absent, not a failed root.
      if (prefix && root.narrowed) {
        let info = null;
        try { info = await lstat(path.join(root.directory, ...prefix.split("/"))); } catch (error) { if (error.code !== "ENOENT") { issue(root, prefix, error.code || "unreadable"); continue; } }
        if (!info) { (entry.absent ||= []).push(prefix); continue; }
        if (info.isSymbolicLink()) { issue(root, prefix, "link-not-scanned"); continue; }
      }
      await walk(prefix);
    }
  }
  scan.metrics.elapsedMs = Math.round(performance.now() - started);
  return scan;
}

/** Only this projection may leave the server: no OS identities/fingerprints. */
export function publicLoraInventory(scan) {
  return { scanId: scan.scanId, complete: scan.complete, issues: scan.issues, roots: scan.roots, ...(scan.metrics ? { metrics: scan.metrics } : {}),
    observations: scan.observations.map(({ engine, rootId, value, contentHash, size }) => ({ engine, rootId, value, contentHash, size })) };
}

/** Bootstrap inventory only, never infer relocations or merge equal-content copies. */
export function registerInitialInventory(registry, scan) {
  assertRegistry(registry);
  if (!scan.complete || Object.keys(registry.instances).length) throw libraryError("Initial registration requires an empty registry and complete scan");
  const next = structuredClone(registry);
  for (const item of scan.observations) {
    const instanceId = randomUUID();
    next.instances[instanceId] = { ...item, instanceId, contentRevision: 1, history: [], status: "present", lastCompleteObservation: scan.scanId };
  }
  next.revision += 1;
  return next;
}

/** Complete, explicitly confirmed observation baseline. Never infer movement:
 * a new occupant has its own draft UUID even when bytes/OS IDs are duplicated.
 * Apply AFTER selected relocations so their destinations retain the source ID.
 */
export function registerObservedInventory(registry, scan) {
  assertRegistry(registry);
  if (!scan.complete) throw libraryError("Complete observation required for baseline registration");
  const next = structuredClone(registry);
  // Prefix-aware (P10.2): records outside a narrowed folder scan are not marked missing.
  const covered = (record) => scanCovers(scan, record);
  for (const [id, record] of Object.entries(next.instances)) {
    if (!covered(record)) continue;
    const observed = scan.observations.find((item) => item.value === record.value && sameObservedInstance(record, item));
    next.instances[id] = { ...record, status: observed ? "present" : "missing", lastCompleteObservation: scan.scanId };
  }
  for (const observation of scan.observations) {
    const matches = Object.values(next.instances).filter((record) => record.status === "present" && record.value === observation.value && sameObservedInstance(record, observation));
    if (matches.length > 1) throw libraryError("Conflicting current occupant identities; review baseline");
    const old = matches[0];
    const instanceId = old?.instanceId || observation.instanceId;
    if (!INSTANCE_ID.test(instanceId || "") || (!old && next.instances[instanceId])) throw libraryError("Observation draft identity conflict; rescan");
    next.instances[instanceId] = { ...(old || {}), ...observation, instanceId, contentRevision: old?.contentRevision || 1,
      history: old?.history || [], status: "present", lastCompleteObservation: scan.scanId };
  }
  next.revision += 1;
  return next;
}

/** Pure legacy staging. Missing paths get unresolved records, never a guessed
 * same-named/hash-claimed instance. Consumers are returned separately so only
 * explicit initialization/confirmation may persist these additive fields.
 */
export function registerLegacyReferences(registry, references, scan) {
  assertRegistry(registry);
  const next = structuredClone(registry), upgraded = [];
  for (const reference of references) {
    if (!engines.has(reference.engine) || !validRelative(reference.value) || typeof reference.rootId !== "string"
      || !scan.roots.some((root) => root.engine === reference.engine && root.rootId === reference.rootId)) throw libraryError("Legacy reference is not in the admitted engine/root scope", 400);
    let instance = reference.instanceId ? next.instances[reference.instanceId]
      : Object.values(next.instances).find((entry) => entry.engine === reference.engine && entry.rootId === reference.rootId && entry.value === reference.value && !entry.retired);
    if (reference.instanceId && (!instance || instance.engine !== reference.engine)) throw libraryError("Legacy identity conflict requires review");
    if (!instance) {
      const observed = scan.observations.find((entry) => entry.engine === reference.engine && entry.rootId === reference.rootId && entry.value === reference.value);
      const instanceId = randomUUID();
      instance = observed ? { ...observed, instanceId, contentRevision: 1, history: [], status: "present", lastCompleteObservation: scan.complete ? scan.scanId : null }
        : { engine: reference.engine, rootId: reference.rootId, value: reference.value, instanceId, contentHash: null,
          contentRevision: 0, history: [], fsIdentity: null, status: "missing", lastCompleteObservation: scan.complete ? scan.scanId : null, evidence: "legacy-unverified" };
      next.instances[instanceId] = instance;
    }
    const { contentHash: _untrustedHash, contentRevision: _untrustedRevision, ...original } = reference;
    upgraded.push({ ...original, identitySchema: 1, instanceId: instance.instanceId,
      ...(instance.contentHash ? { contentHash: instance.contentHash, contentRevision: instance.contentRevision } : {}) });
  }
  if (JSON.stringify(next.instances) !== JSON.stringify(registry.instances)) next.revision += 1;
  return { registry: next, references: upgraded };
}

/** `confidence` stays the provenance verdict. Ranked `candidates`,
 * `suggestion` and `recommendation` are editable preferences (P10.1): every
 * admitted observation is a candidate, including registered copies, and
 * "existing" is judged against the last COMMITTED baseline, never against a
 * first scan's provisional registration.
 */
export function matchLoraInventory(registry, scan, { baseline = registry, candidate: allowed = () => true } = {}) {
  assertRegistry(registry);
  const known = Object.values(registry.instances), committed = Object.values(baseline.instances);
  const covered = (item) => scanCovers(scan, item);
  // `allowed` is an explicit target-directory constraint; it limits candidates only.
  const pool = scan.observations.filter((observation) => allowed(observation));
  const at = (item) => scan.observations.find((observation) => observation.engine === item.engine && observation.rootId === item.rootId && observation.value === item.value);
  const unchanged = sameObservedInstance;
  // Indexed by path and memoized per observation: libraries can hold thousands of files.
  const pathKey = (item) => `${item.engine}\0${item.rootId}\0${item.value}`, byPath = new Map(), claims = new Map();
  for (const instance of committed) byPath.set(pathKey(instance), [...(byPath.get(pathKey(instance)) || []), instance]);
  const claimedBy = (observation) => {
    if (!claims.has(observation)) claims.set(observation, (byPath.get(pathKey(observation)) || []).find((instance) => sameObservedInstance(instance, observation))?.instanceId || null);
    return claims.get(observation);
  };
  const repair = [], rows = known.filter(covered).map((source) => {
    const observed = at(source);
    if (unchanged(source, observed)) return { sourceId: source.instanceId, engine: source.engine, sourceValue: source.value, status: "unchanged", candidates: [], recommendation: null };
    const matches = pool.filter((candidate) => candidate.engine === source.engine && source.contentHash && candidate.contentHash === source.contentHash
      && candidate !== observed && !claimedBy(candidate));
    const fsMatches = matches.filter((candidate) => source.fsIdentity && candidate.fsIdentity
      && source.fsIdentity.links === 1 && candidate.fsIdentity.links === 1
      && source.fsIdentity.key === candidate.fsIdentity.key && source.fsIdentity.creation === candidate.fsIdentity.creation
      && !known.some((other) => other.instanceId !== source.instanceId && other.engine === source.engine && other.fsIdentity?.key === candidate.fsIdentity.key)
      && scan.observations.filter((other) => other.engine === source.engine && other.fsIdentity?.key === candidate.fsIdentity.key).length === 1);
    const proven = scan.complete && !observed && source.lastCompleteObservation && fsMatches.length === 1;
    const row = { sourceId: source.instanceId, engine: source.engine, sourceValue: source.value,
      status: !scan.complete ? "incomplete" : observed ? "identity-conflict" : "missing",
      confidence: proven ? "filesystem-and-content" : matches.length === 1 ? "same-content-unproven" : matches.length > 1 ? "ambiguous-content" : "unresolved" };
    repair.push({ row, source, observed, provenTarget: proven ? fsMatches[0] : null });
    return row;
  });
  const ranked = rankRepairRows(repair, pool, { complete: scan.complete, claimedBy });
  repair.forEach(({ row }, index) => { const { sourceId: _same, ...ranking } = ranked[index]; Object.assign(row, ranking); });
  return rows;
}
