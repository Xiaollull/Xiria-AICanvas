import { inspectMountedLoras, normalizeMountedLoras, sameMountedLoras } from "./lora-state.js";

export const MOUNTED_LORAS_SCHEMA_VERSION = 2;
export const READY_LORA_ENGINES = ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"];
// The engines every v2 library on disk was written with. A library saved before an engine shipped
// cannot carry its list, so a missing one is an empty scope rather than a corrupt file — treating
// it as corruption would stop autosave and strand the user's existing mounts.
const ESTABLISHED_LORA_ENGINES = ["SD", "iL", "Anima"];
const ENGINES = new Set(READY_LORA_ENGINES);

function isPlainObject(value) { return Boolean(value) && typeof value === "object" && !Array.isArray(value); }
function validEngine(value) { return ENGINES.has(value) ? value : null; }
function cloneList(value) { return normalizeMountedLoras(value).map((item) => ({ ...item })); }
function v1KeyPayload(key) {
  if (typeof key !== "string" || !key.startsWith("xirai-lora-scope-v1.")) return null;
  try {
    const encoded = key.split(".").pop().replace(/-/g, "+").replace(/_/g, "/");
    const parsed = JSON.parse(atob(`${encoded}${"=".repeat((4 - encoded.length % 4) % 4)}`));
    const expectedAssets = parsed?.engine === "Anima" ? 3 : 1;
    if (!isPlainObject(parsed) || parsed.v !== 1 || !validEngine(parsed.engine) || !Array.isArray(parsed.assets) || parsed.assets.length !== expectedAssets || !parsed.assets.every((asset) => typeof asset === "string" && asset.length > 0 && asset.length <= 500 && !asset.includes("\0"))) return null;
    return parsed;
  } catch { return null; }
}

/** Only ready canonical engine enums can address the mounted library. */
export function engineScopeKey(value) { return validEngine(typeof value === "string" ? value : value?.model); }
export function emptyMountedLoraMap() {
  return { schemaVersion: MOUNTED_LORAS_SCHEMA_VERSION, byEngine: Object.fromEntries(READY_LORA_ENGINES.map((engine) => [engine, []])) };
}
function clonedMap(container) {
  const next = emptyMountedLoraMap();
  for (const engine of READY_LORA_ENGINES) next.byEngine[engine] = cloneList(container?.byEngine?.[engine]);
  return next;
}
function warningForRejected(count) { return count ? `已隔离 ${count} 个无效 LoRA 挂载库条目。` : ""; }

function v1Candidates(value, engine) {
  const candidates = Object.entries(value.byModel)
    .map(([key, loras]) => ({ key, loras: cloneList(loras), parsed: v1KeyPayload(key) }))
    // V1's reversible JSON contains engine; migration-only inspection never
    // exposes key/path data in metadata or warnings.
    .filter(({ parsed }) => parsed.engine === engine);
  return candidates.sort((a, b) => a.key.localeCompare(b.key));
}
function v1Winner(value, engine, activeEngine, activeModelIdentity, legacy) {
  if (engine === activeEngine && legacy.items.length) return legacy.items;
  const candidates = v1Candidates(value, engine);
  if (!candidates.length) return [];
  const exact = activeModelIdentity && candidates.find((candidate) => {
    const parsed = candidate.parsed;
    return parsed?.engine === activeModelIdentity.engine && JSON.stringify(parsed.assets) === JSON.stringify(activeModelIdentity.assets);
  });
  if (exact) return exact.loras;
  return (candidates.find((candidate) => candidate.loras.length) || candidates[0]).loras;
}

/**
 * Normalizes v2 data and performs the one-way v1 fold.  The winner rule is:
 * active engine uses a valid legacy mirror first; otherwise its exact saved
 * active v1 identity wins; otherwise lexicographically smallest nonempty v1
 * key; otherwise lexicographically smallest v1 key. Lists are never merged.
 */
export function normalizeMountedLoraMap(value, { fieldMissing = false, legacyLoras = [], activeEngine = null, activeModelIdentity = null } = {}) {
  const engine = engineScopeKey(activeEngine);
  const legacy = inspectMountedLoras(legacyLoras);
  if (fieldMissing) {
    const container = emptyMountedLoraMap();
    if (engine) container.byEngine[engine] = legacy.items;
    if (!engine && legacy.items.length) return { container, activeLoras: [], warning: "当前模型引擎无效，无法安全迁移旧 LoRA 列表；已停止自动保存以保护原始数据。", fatal: true, raw: undefined, migrated: false, rejected: legacy.rejected };
    return { container, activeLoras: engine ? container.byEngine[engine] : [], warning: warningForRejected(legacy.rejected), fatal: false, migrated: true, rejected: legacy.rejected };
  }
  if (!isPlainObject(value) || !Number.isInteger(value.schemaVersion)) return { container: emptyMountedLoraMap(), activeLoras: [], warning: "LoRA 挂载库格式损坏，已停止自动保存以保护原始数据。", fatal: true, raw: value, migrated: false, rejected: 0 };
  if (value.schemaVersion > MOUNTED_LORAS_SCHEMA_VERSION) return { container: emptyMountedLoraMap(), activeLoras: [], warning: "LoRA 挂载库来自较新版本，已停止自动保存以保护原始数据。", fatal: true, raw: value, migrated: false, rejected: 0 };
  if (value.schemaVersion === 1 && isPlainObject(value.byModel)) {
    const entries = Object.entries(value.byModel);
    // `{}` is a valid v1 empty library. Every nonempty v1 entry must be a
    // canonical, decodable v1 model key paired with an array; otherwise no
    // migration can safely choose a winner and raw data must survive intact.
    if (entries.some(([key, loras]) => !v1KeyPayload(key) || !Array.isArray(loras))) {
      return { container: emptyMountedLoraMap(), activeLoras: [], warning: "LoRA 挂载库格式损坏，已停止自动保存以保护原始数据。", fatal: true, raw: value, migrated: false, rejected: 0 };
    }
    const container = emptyMountedLoraMap();
    for (const target of READY_LORA_ENGINES) container.byEngine[target] = v1Winner(value, target, engine, activeModelIdentity, legacy);
    return { container, activeLoras: engine ? container.byEngine[engine] : [], warning: "LoRA 挂载库已按模型引擎迁移；未合并不同底模的列表。", fatal: false, migrated: true, rejected: legacy.rejected };
  }
  if (value.schemaVersion !== 2 || !isPlainObject(value.byEngine) || ESTABLISHED_LORA_ENGINES.some((engineName) => !Array.isArray(value.byEngine[engineName])) || READY_LORA_ENGINES.some((engineName) => value.byEngine[engineName] !== undefined && !Array.isArray(value.byEngine[engineName])) || Object.keys(value.byEngine).some((key) => !validEngine(key))) return { container: emptyMountedLoraMap(), activeLoras: [], warning: "LoRA 挂载库版本无效，已停止自动保存以保护原始数据。", fatal: true, raw: value, migrated: false, rejected: 0 };
  const container = emptyMountedLoraMap(); let rejected = 0;
  for (const key of Object.keys(value.byEngine)) {
    const inspected = inspectMountedLoras(value.byEngine[key]); rejected += inspected.rejected; container.byEngine[key] = inspected.items;
  }
  return { container, activeLoras: engine ? container.byEngine[engine] : [], warning: warningForRejected(rejected), fatal: false, migrated: false, rejected };
}
export function mountedLorasForScope(container, engine) { return engineScopeKey(engine) ? cloneList(container?.byEngine?.[engineScopeKey(engine)]) : []; }
export function frozenMountedLorasForScope(container, engine) { return mountedLorasForScope(container, engine); }
/** Semantic map equality keeps transient React/ref copies from becoming edits. */
export function sameMountedLoraMap(first, second) {
  return READY_LORA_ENGINES.every((engine) => sameMountedLoras(first?.byEngine?.[engine], second?.byEngine?.[engine]));
}
/**
 * Which files are mounted, ignoring how they are tuned.  A scan exists only to
 * prune entries whose file has left the directory, which depends on the set of
 * paths and on nothing else — not their order, weight, precision or enabled
 * flag.
 */
function mountedLoraIdentity(container) {
  return READY_LORA_ENGINES
    .map((engine) => normalizeMountedLoras(container?.byEngine?.[engine]).map((item) => item.value).sort().join(" "))
    .join("");
}
export function sameMountedLoraIdentity(first, second) {
  return mountedLoraIdentity(first) === mountedLoraIdentity(second);
}
/**
 * A scan generation changes only when the mounted *set* changes.
 *
 * This deliberately ignores weight, precision, enabled and order. Keying it on
 * full entry equality made every slider tick invalidate the scan, which
 * restarted the directory listing effect and tore the mounted list out from
 * under the pointer mid-drag.
 */
export function nextMountedLoraRevision(revision, previous, next) {
  return sameMountedLoraIdentity(previous, next) ? revision : revision + 1;
}
/** Scans are never allowed while a live/restored job or workspace lock exists. */
export function canStartMountedLoraScan({ uiStateReady = true, activeJobRecoveryPending = false, status = "idle", modelSwitching = false, workspaceLocked = false, shouldPersist = true } = {}) {
  return uiStateReady === true && activeJobRecoveryPending !== true && status !== "running" && modelSwitching !== true && workspaceLocked !== true && shouldPersist === true;
}
/**
 * Response admission is deliberately stricter than request admission.  A
 * directory listing is allowed to prune only the exact mounted-map generation
 * it observed before its request was sent.
 */
export function shouldApplyMountedLoraScan({ requestToken, latestToken, capturedRevision, latestRevision, responseEngine, activeEngine, activeJobRecoveryPending = false, status = "idle", modelSwitching = false, workspaceLocked = false, shouldPersist = true } = {}) {
  return canStartMountedLoraScan({ activeJobRecoveryPending, status, modelSwitching, workspaceLocked, shouldPersist })
    && Number.isInteger(requestToken) && requestToken === latestToken
    && Number.isInteger(capturedRevision) && Number.isInteger(latestRevision) && capturedRevision === latestRevision
    && engineScopeKey(responseEngine) === engineScopeKey(activeEngine) && Boolean(engineScopeKey(responseEngine));
}
/**
 * Pure scan completion.  It reads the current mounted map only after every
 * token/engine/lock/revision gate has admitted the response, allowing App to
 * make library, prune, broadcast and persistence effects conditional on the
 * same `applied` result.
 */
export function reconcileMountedLoraScan({ container, getCurrentContainer, categories, scanScopeKey, requestToken, latestToken, capturedRevision, latestRevision, responseEngine, activeEngine, activeJobRecoveryPending = false, status = "idle", modelSwitching = false, workspaceLocked = false, shouldPersist = true } = {}) {
  if (!shouldApplyMountedLoraScan({ requestToken, latestToken, capturedRevision, latestRevision, responseEngine, activeEngine, activeJobRecoveryPending, status, modelSwitching, workspaceLocked, shouldPersist })) {
    return { applied: false, changed: false, container, loras: null, categories: null };
  }
  const scopeKey = engineScopeKey(scanScopeKey);
  if (!scopeKey || scopeKey !== engineScopeKey(activeEngine)) {
    return { applied: false, changed: false, container, loras: null, categories: null };
  }
  // The App supplies a getter so a rejected response never even reads the
  // current mounted map in its completion path.
  const currentContainer = typeof getCurrentContainer === "function" ? getCurrentContainer() : container;
  const scannedCategories = Array.isArray(categories) ? categories : [];
  const available = new Set(scannedCategories.flatMap((category) => (category?.models || []).map((item) => item?.value)));
  const current = mountedLorasForScope(currentContainer, scopeKey);
  const loras = current.filter((item) => available.has(item.value));
  const changed = !sameMountedLoras(current, loras);
  return {
    applied: true,
    changed,
    container: changed ? withMountedLorasForScope(currentContainer, scopeKey, loras) : currentContainer,
    loras,
    categories: scannedCategories,
  };
}
export function galleryMountedLorasForTarget(container, { targetEngine, sourceLoras, applyLoras = false } = {}) { const engine = engineScopeKey(targetEngine); return engine ? { valid: true, loras: applyLoras ? cloneList(sourceLoras) : mountedLorasForScope(container, engine) } : { valid: false, loras: [] }; }
export function withMountedLorasForScope(container, engine, loras) { const target = engineScopeKey(engine); const next = clonedMap(container); if (target) next.byEngine[target] = cloneList(loras); return next; }
export function updateMountedLorasForScope(container, engine, updater) { const target = engineScopeKey(engine); const previous = mountedLorasForScope(container, target); if (!target || typeof updater !== "function") return { container: clonedMap(container), loras: previous, changed: false }; const loras = cloneList(updater(previous)); return { container: withMountedLorasForScope(container, target, loras), loras, changed: JSON.stringify(previous) !== JSON.stringify(loras) }; }
export function transitionMountedLoraScope(container, { sourceEngine = null, sourceLoras = [], targetEngine = null, accepted = true } = {}) {
  const original = clonedMap(container);
  const source = engineScopeKey(sourceEngine);
  if (!accepted) return { container: original, activeLoras: cloneList(sourceLoras), activeScopeKey: source, changed: false, scopeChanged: false };
  const saved = withMountedLorasForScope(original, source, sourceLoras);
  const target = engineScopeKey(targetEngine);
  return {
    container: saved,
    activeLoras: mountedLorasForScope(saved, target),
    activeScopeKey: target,
    changed: !sameMountedLoraMap(original, saved),
    scopeChanged: source !== target,
  };
}
export function applyMountedLoraSync(container, { engine, loras } = {}, { activeEngine = null, locked = false } = {}) { if (locked || !engineScopeKey(engine)) return { container: clonedMap(container), activeLoras: mountedLorasForScope(container, activeEngine), applied: false }; const next = withMountedLorasForScope(container, engine, loras); return { container: next, activeLoras: mountedLorasForScope(next, activeEngine), applied: true }; }
