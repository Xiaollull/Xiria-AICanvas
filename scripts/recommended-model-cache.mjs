import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, realpath, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

export const RECOMMENDED_MODEL_CACHE_SCHEMA = 2;
export const RECOMMENDED_REMOTE_JSON_MAX_BYTES = 2 * 1024 * 1024;
export const REMOTE_JSON_HARD_MAX_BYTES = 4 * 1024 * 1024;
export const RECOMMENDED_MAXIMUM_CONCURRENCY = 16;
export const RECOMMENDED_MAXIMUM_BACKOFF_MS = 10 * 60 * 1000;

const maximumCacheBytes = 4 * 1024 * 1024;
const maximumInstallations = 4096;
const maximumRemoteFamilies = 128;
const maximumRemoteModels = 4096;
const maximumRemoteFilesPerVersion = 256;
const cachedRoles = new Set(["checkpoint", "diffusion_model", "text_encoder", "lora", "vae", "yolo", "upscaler", "embedding", "config"]);
const digestPattern = /^[a-f0-9]{64}$/;

function plainObject(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype);
}

function hasOnlyKeys(value, keys) {
  return Object.keys(value).every((key) => keys.has(key));
}

function boundedString(value, maximum, { empty = false } = {}) {
  return typeof value === "string"
    && (empty || value.length > 0)
    && value.length <= maximum
    && !/[\u0000-\u001f\u007f]/.test(value);
}

function timestamp(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function fileTime(value) {
  return Number.isFinite(value) && value >= 0;
}

function digest(value) {
  const normalized = typeof value === "string" ? value.toLowerCase() : "";
  return digestPattern.test(normalized) ? normalized : "";
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (plainObject(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function recommendedCatalogFingerprint(catalog) {
  return createHash("sha256").update(stableJson(catalog)).digest("hex");
}

export function safeRecommendedRelativePath(value) {
  if (!boundedString(value, 1024) || value.includes("\\") || value.startsWith("/") || /^[a-z]:/i.test(value)) return "";
  const segments = value.split("/");
  if (!segments.length || segments.some((segment) => !segment || segment === "." || segment === "..")) return "";
  return segments.join("/");
}

function normalizedRemoteModel(value) {
  if (!plainObject(value) || !hasOnlyKeys(value, new Set(["modelId", "versionId", "fileId", "label", "filename", "size", "sha256", "baseModel"]))) return null;
  const sha256 = digest(value.sha256);
  if (!Number.isSafeInteger(value.modelId) || value.modelId <= 0
    || !Number.isSafeInteger(value.versionId) || value.versionId <= 0
    || !Number.isSafeInteger(value.fileId) || value.fileId <= 0
    || !boundedString(value.label, 500)
    || !boundedString(value.filename, 180)
    || !Number.isSafeInteger(value.size) || value.size <= 0
    || !boundedString(value.baseModel, 200)
    || !sha256) return null;
  return { modelId: value.modelId, versionId: value.versionId, fileId: value.fileId, label: value.label, filename: value.filename, size: value.size, sha256, baseModel: value.baseModel };
}

function normalizedRemoteSnapshot(value) {
  if (value === null) return null;
  if (!plainObject(value) || !hasOnlyKeys(value, new Set(["savedAt", "families"]))
    || !timestamp(value.savedAt) || !Array.isArray(value.families) || value.families.length > maximumRemoteFamilies) return null;
  const families = [];
  const familyIds = new Set();
  let modelCount = 0;
  for (const family of value.families) {
    if (!plainObject(family) || !hasOnlyKeys(family, new Set(["id", "fetchedAt", "complete", "models"]))
      || !boundedString(family.id, 160) || familyIds.has(family.id)
      || !timestamp(family.fetchedAt) || typeof family.complete !== "boolean" || !Array.isArray(family.models)) return null;
    const models = family.models.map(normalizedRemoteModel);
    modelCount += models.length;
    if (models.some((model) => !model) || modelCount > maximumRemoteModels
      || new Set(models.map((model) => model.versionId)).size !== models.length) return null;
    familyIds.add(family.id);
    families.push({ id: family.id, fetchedAt: family.fetchedAt, complete: family.complete, models });
  }
  return { savedAt: value.savedAt, families };
}

export function clampRecommendedRemoteState(value, {
  now = Date.now(),
  maximumBackoffMs = RECOMMENDED_MAXIMUM_BACKOFF_MS,
} = {}) {
  if (!plainObject(value) || !hasOnlyKeys(value, new Set(["lastAttemptAt", "lastSuccessAt", "failureCount", "nextAttemptAt"]))) return null;
  if (!timestamp(value.lastAttemptAt) || !timestamp(value.lastSuccessAt) || !timestamp(value.nextAttemptAt)
    || !Number.isSafeInteger(value.failureCount) || value.failureCount < 0 || value.failureCount > 1000) return null;
  const currentTime = Number.isSafeInteger(now) && now >= 0 ? now : Date.now();
  const lastAttemptAt = value.lastAttemptAt > currentTime ? 0 : value.lastAttemptAt;
  const lastSuccessAt = value.lastSuccessAt > currentTime ? 0 : value.lastSuccessAt;
  const nextAttemptAt = Math.min(value.nextAttemptAt, currentTime + Math.max(0, maximumBackoffMs));
  return {
    lastAttemptAt,
    lastSuccessAt,
    failureCount: value.failureCount,
    nextAttemptAt,
  };
}

function normalizedInstallation(value) {
  if (!plainObject(value) || !hasOnlyKeys(value, new Set(["sha256", "role", "relativePath", "size", "mtimeMs", "ctimeMs"]))) return null;
  const sha256 = digest(value.sha256);
  const relativePath = safeRecommendedRelativePath(value.relativePath);
  if (!sha256 || !cachedRoles.has(value.role) || !relativePath
    || !Number.isSafeInteger(value.size) || value.size <= 0
    || !fileTime(value.mtimeMs) || !fileTime(value.ctimeMs)) return null;
  return { sha256, role: value.role, relativePath, size: value.size, mtimeMs: value.mtimeMs, ctimeMs: value.ctimeMs };
}

export function validateRecommendedModelCache(value, expectedFingerprint, options = {}) {
  if (!plainObject(value) || !hasOnlyKeys(value, new Set(["schema", "catalogFingerprint", "remoteSnapshot", "remote", "installations"]))) return null;
  if (value.schema !== RECOMMENDED_MODEL_CACHE_SCHEMA || !digest(value.catalogFingerprint)
    || value.catalogFingerprint !== expectedFingerprint || !Array.isArray(value.installations)
    || value.installations.length > maximumInstallations) return null;
  const remoteSnapshot = normalizedRemoteSnapshot(value.remoteSnapshot);
  const remote = clampRecommendedRemoteState(value.remote, options);
  const installations = value.installations.map(normalizedInstallation);
  if ((value.remoteSnapshot !== null && !remoteSnapshot) || !remote || installations.some((entry) => !entry)
    || new Set(installations.map((entry) => entry.sha256)).size !== installations.length) return null;
  return {
    schema: RECOMMENDED_MODEL_CACHE_SCHEMA,
    catalogFingerprint: value.catalogFingerprint,
    remoteSnapshot,
    remote,
    installations,
  };
}

export function emptyRecommendedModelCache(catalogFingerprint) {
  if (!digest(catalogFingerprint)) throw new Error("A valid recommendation catalog fingerprint is required");
  return {
    schema: RECOMMENDED_MODEL_CACHE_SCHEMA,
    catalogFingerprint,
    remoteSnapshot: null,
    remote: { lastAttemptAt: 0, lastSuccessAt: 0, failureCount: 0, nextAttemptAt: 0 },
    installations: [],
  };
}

export async function readRecommendedModelCache(cachePath, expectedFingerprint, options = {}) {
  try {
    const file = await stat(cachePath);
    if (!file.isFile() || file.size <= 0 || file.size > maximumCacheBytes) return null;
    return validateRecommendedModelCache(JSON.parse(await readFile(cachePath, "utf8")), expectedFingerprint, options);
  } catch {
    return null;
  }
}

export async function writeRecommendedModelCache(cachePath, value) {
  const normalized = validateRecommendedModelCache(value, value?.catalogFingerprint);
  if (!normalized) throw new Error("Refusing to write an invalid recommended model cache");
  const serialized = `${JSON.stringify(normalized, null, 2)}\n`;
  if (Buffer.byteLength(serialized) > maximumCacheBytes) throw new Error("Recommended model cache is too large");
  await mkdir(path.dirname(cachePath), { recursive: true });
  const temporaryPath = path.join(path.dirname(cachePath), `.${path.basename(cachePath)}.${process.pid}.${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, serialized, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, cachePath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return normalized;
}

function pathIsInside(root, candidate) {
  const relative = path.relative(root, candidate);
  return Boolean(relative) && !relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative);
}

function civitaiPrimaryFile(version) {
  if (!Array.isArray(version?.files) || version.files.length > maximumRemoteFilesPerVersion) return null;
  return version.files.find((item) => item?.primary)
    || version.files.find((item) => item?.type === "Model")
    || version.files[0]
    || null;
}

function civitaiFileSize(file) {
  const bytes = Math.round(Number(file?.sizeKB || 0) * 1024);
  return Number.isSafeInteger(bytes) && bytes > 0 ? bytes : 0;
}

export function trustedCivitaiFamilyVersions(family, response) {
  if (!plainObject(family) || !boundedString(family.id, 160) || !Number.isSafeInteger(family.modelId) || family.modelId <= 0
    || !Array.isArray(family.baseModels) || !family.baseModels.length || !family.baseModels.every((item) => boundedString(item, 200))) {
    throw new Error("Invalid trusted Civitai family");
  }
  if (!plainObject(response) || Number(response.id) !== family.modelId || !Array.isArray(response.modelVersions)
    || response.modelVersions.length > maximumRemoteModels) {
    throw new Error("Civitai response does not belong to the trusted model family");
  }
  const allowed = new Set(family.baseModels);
  const records = [];
  for (const version of response.modelVersions) {
    if (!plainObject(version) || (version.modelId != null && Number(version.modelId) !== family.modelId)) {
      throw new Error("Civitai version does not belong to the trusted model family");
    }
    if (!allowed.has(version.baseModel)) continue;
    const versionId = Number(version.id);
    const file = civitaiPrimaryFile(version);
    const fileId = Number(file?.id);
    const sha256 = digest(file?.hashes?.SHA256 || file?.hashes?.sha256 || file?.sha256);
    const size = civitaiFileSize(file);
    const filename = typeof file?.name === "string" ? file.name : "";
    const label = typeof version.name === "string" && version.name ? version.name : `Version ${versionId}`;
    if (!Number.isSafeInteger(versionId) || versionId <= 0 || !Number.isSafeInteger(fileId) || fileId <= 0
      || !boundedString(filename, 180) || !boundedString(label, 500) || !sha256 || !size) continue;
    records.push({ modelId: family.modelId, versionId, fileId, label, filename, size, sha256, baseModel: version.baseModel });
  }
  if (!records.length) throw new Error("Civitai family has no fully bound downloadable versions");
  if (new Set(records.map((record) => record.versionId)).size !== records.length) throw new Error("Civitai family has duplicate versions");
  return records;
}

export async function raceTrustedCivitaiFamilyVersions(family, attempts) {
  if (!Array.isArray(attempts) || !attempts.length || attempts.some((attempt) => typeof attempt !== "function")) {
    throw new TypeError("Civitai family race requires response attempts");
  }
  return Promise.any(attempts.map(async (attempt) => trustedCivitaiFamilyVersions(family, await attempt())));
}

export function assertCivitaiArtifactForFamily(family, artifact) {
  if (!plainObject(family) || !plainObject(artifact) || artifact.familyId !== family.id
    || Number(artifact.modelId) !== Number(family.modelId)
    || !Number.isSafeInteger(artifact.versionId) || artifact.versionId <= 0
    || !Array.isArray(family.baseModels) || !family.baseModels.includes(artifact.baseModel)
    || artifact.id !== `${family.id}-${artifact.versionId}`) {
    throw new Error("Recommended Civitai artifact is outside its trusted family");
  }
  if (artifact.fileId == null) {
    const configured = Array.isArray(family.versions) && family.versions.some((version) => Number(version?.[0]) === artifact.versionId);
    if (!configured) throw new Error("Unbound Civitai artifact is not a configured fallback version");
    return artifact;
  }
  if (!Number.isSafeInteger(artifact.fileId) || artifact.fileId <= 0 || !boundedString(artifact.filename, 180)
    || !Number.isSafeInteger(artifact.size) || artifact.size <= 0 || !digest(artifact.sha256)) {
    throw new Error("Recommended Civitai artifact has an incomplete file binding");
  }
  return artifact;
}

export function assertCivitaiResolutionForArtifact(family, artifact, resolved) {
  assertCivitaiArtifactForFamily(family, artifact);
  if (!plainObject(resolved) || Number(resolved.modelId) !== Number(family.modelId)
    || Number(resolved.versionId) !== artifact.versionId
    || !family.baseModels.includes(resolved.baseModel)
    || !Number.isSafeInteger(resolved.fileId) || resolved.fileId <= 0
    || !boundedString(resolved.filename, 180)
    || !Number.isSafeInteger(resolved.size) || resolved.size <= 0
    || !digest(resolved.expectedSha256)) {
    throw new Error("Resolved Civitai file is outside the trusted recommendation family");
  }
  if (artifact.fileId != null && (resolved.fileId !== artifact.fileId
    || resolved.filename !== artifact.filename
    || (artifact.size && resolved.size !== artifact.size)
    || digest(resolved.expectedSha256) !== digest(artifact.sha256))) {
    throw new Error("Resolved Civitai file no longer matches the cached recommendation binding");
  }
  return resolved;
}

export function mergeResolvedCivitaiBinding(snapshot, family, artifact, resolved, actualSize, now = Date.now()) {
  assertCivitaiResolutionForArtifact(family, artifact, resolved);
  const model = normalizedRemoteModel({
    modelId: family.modelId,
    versionId: artifact.versionId,
    fileId: resolved.fileId,
    label: artifact.label,
    filename: resolved.filename,
    size: actualSize,
    sha256: resolved.expectedSha256,
    baseModel: resolved.baseModel,
  });
  if (!model) throw new Error("Resolved Civitai binding is incomplete");
  const families = (snapshot?.families || []).map((item) => ({ ...item, models: [...item.models] }));
  const familyIndex = families.findIndex((item) => item.id === family.id);
  const previous = familyIndex >= 0 ? families[familyIndex] : { id: family.id, fetchedAt: 0, complete: false, models: [] };
  const models = [...previous.models.filter((item) => item.versionId !== model.versionId), model];
  const modelCount = families.reduce((count, item) => count + item.models.length, 0) - previous.models.length + models.length;
  if (modelCount > maximumRemoteModels) throw new Error("Civitai snapshot has too many models");
  const updated = { ...previous, fetchedAt: now, models };
  if (familyIndex >= 0) families[familyIndex] = updated;
  else families.push(updated);
  if (families.length > maximumRemoteFamilies) throw new Error("Civitai snapshot has too many families");
  return { snapshot: { savedAt: now, families }, model };
}

export function mergeConcurrentCivitaiSnapshots({ baseSnapshot, currentSnapshot, refreshedSnapshot }) {
  const base = normalizedRemoteSnapshot(baseSnapshot);
  const current = normalizedRemoteSnapshot(currentSnapshot);
  const refreshed = normalizedRemoteSnapshot(refreshedSnapshot);
  if (!refreshed) throw new Error("Refreshed Civitai snapshot is invalid");
  if (!current) return refreshed;
  const baseModels = new Map((base?.families || []).flatMap((family) => family.models.map((model) => [`${family.id}\u0000${model.versionId}`, model])));
  const concurrent = [];
  for (const family of current.families) {
    for (const model of family.models) {
      const previous = baseModels.get(`${family.id}\u0000${model.versionId}`);
      if (!previous || stableJson(previous) !== stableJson(model)) concurrent.push({ family, model });
    }
  }
  if (!concurrent.length) return refreshed;
  const families = refreshed.families.map((family) => ({ ...family, models: [...family.models] }));
  for (const { family: currentFamily, model } of concurrent) {
    let target = families.find((family) => family.id === currentFamily.id);
    if (!target) {
      target = { id: currentFamily.id, fetchedAt: currentFamily.fetchedAt, complete: false, models: [] };
      families.push(target);
    }
    target.fetchedAt = Math.max(target.fetchedAt, currentFamily.fetchedAt);
    target.models = [...target.models.filter((item) => item.versionId !== model.versionId), model];
  }
  const merged = normalizedRemoteSnapshot({ savedAt: Math.max(refreshed.savedAt, current.savedAt), families });
  if (!merged) throw new Error("Merged Civitai snapshot exceeds its schema limits");
  return merged;
}

export function indexRecommendedInstallationCandidates(entries) {
  return new Map((entries || []).map((entry) => [entry.sha256, { ...entry }]));
}

export async function mapWithRecommendedConcurrency(values, worker, concurrency = RECOMMENDED_MAXIMUM_CONCURRENCY) {
  if (!Array.isArray(values) || typeof worker !== "function") throw new TypeError("A values array and worker are required");
  const limit = Math.max(1, Math.min(RECOMMENDED_MAXIMUM_CONCURRENCY, Math.floor(Number(concurrency) || 1), values.length || 1));
  const results = new Array(values.length);
  let nextIndex = 0;
  let firstError;
  const run = async () => {
    while (!firstError && nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      try {
        results[index] = await worker(values[index], index);
      } catch (error) {
        firstError ||= error;
      }
    }
  };
  await Promise.all(Array.from({ length: limit }, run));
  if (firstError) throw firstError;
  return results;
}

export async function validateCachedInstallation(entry, {
  rootForRole,
  statFile = stat,
  realpathFile = realpath,
} = {}) {
  const normalized = normalizedInstallation(entry);
  if (!normalized || typeof rootForRole !== "function") return null;
  try {
    const root = await realpathFile(await rootForRole(normalized.role));
    const candidate = path.resolve(root, ...normalized.relativePath.split("/"));
    if (!pathIsInside(root, candidate)) return null;
    const resolved = await realpathFile(candidate);
    if (!pathIsInside(root, resolved)) return null;
    const identity = await statFile(resolved);
    if (!identity.isFile() || identity.size !== normalized.size
      || identity.mtimeMs !== normalized.mtimeMs || identity.ctimeMs !== normalized.ctimeMs) return null;
    return { ...normalized, absolutePath: resolved };
  } catch {
    return null;
  }
}

export async function validateCachedInstallations(entries, options = {}) {
  const roots = new Map();
  const rootForRole = (role) => {
    if (!roots.has(role)) roots.set(role, Promise.resolve().then(() => options.rootForRole(role)));
    return roots.get(role);
  };
  const checked = await mapWithRecommendedConcurrency(entries || [], (entry) => validateCachedInstallation(entry, { ...options, rootForRole }), options.concurrency);
  const installed = new Map();
  const validEntries = [];
  for (const entry of checked) {
    if (!entry || installed.has(entry.sha256)) continue;
    installed.set(entry.sha256, entry);
    const { absolutePath: _absolutePath, ...persisted } = entry;
    validEntries.push(persisted);
  }
  return { entries: validEntries, installed };
}

function sameInstallationIdentity(first, second) {
  return Boolean(first && second
    && first.sha256 === second.sha256
    && first.role === second.role
    && first.relativePath === second.relativePath
    && first.size === second.size
    && first.mtimeMs === second.mtimeMs
    && first.ctimeMs === second.ctimeMs);
}

export async function trustedInstallationsForArtifacts(artifacts, {
  candidates,
  trusted,
  validateCandidate,
} = {}) {
  if (!(candidates instanceof Map) || !(trusted instanceof Map) || typeof validateCandidate !== "function") {
    throw new TypeError("Candidate and trusted maps plus a validator are required");
  }
  const installed = new Map();
  const invalidated = new Set();
  for (const artifact of artifacts || []) {
    const sha256 = digest(artifact?.sha256);
    const candidate = sha256 ? candidates.get(sha256) : null;
    if (!candidate) continue;
    const declaredIdentityMatches = candidate.sha256 === sha256
      && candidate.role === artifact.role
      && (!artifact.size || candidate.size === artifact.size);
    const valid = declaredIdentityMatches ? await validateCandidate(candidate) : null;
    if (!valid) {
      candidates.delete(sha256);
      trusted.delete(sha256);
      invalidated.add(sha256);
      continue;
    }
    candidates.set(sha256, valid);
    const trustedEntry = trusted.get(sha256);
    if (sameInstallationIdentity(trustedEntry, valid)) installed.set(sha256, valid.absolutePath || valid.relativePath);
  }
  return { installed, invalidated };
}

export async function verifyRecommendedFileAgainstCatalog({ file, role, artifactsByDigest, hashFile, createEntry }) {
  if (!file || !(artifactsByDigest instanceof Map) || typeof hashFile !== "function" || typeof createEntry !== "function") return null;
  const actualSha256 = digest(await hashFile(file));
  const artifact = actualSha256 ? artifactsByDigest.get(actualSha256) : null;
  if (!artifact || artifact.role !== role || artifact.size !== file.size) return null;
  return createEntry({ artifact, sha256: actualSha256, file, role });
}

export async function installationEntryForFile({ sha256, role, filePath, root, statFile = stat, realpathFile = realpath }) {
  const normalizedDigest = digest(sha256);
  if (!normalizedDigest || !cachedRoles.has(role)) return null;
  try {
    const resolvedRoot = await realpathFile(root);
    const resolvedFile = await realpathFile(filePath);
    if (!pathIsInside(resolvedRoot, resolvedFile)) return null;
    const identity = await statFile(resolvedFile);
    if (!identity.isFile() || !Number.isSafeInteger(identity.size) || identity.size <= 0) return null;
    const relativePath = path.relative(resolvedRoot, resolvedFile).split(path.sep).join("/");
    if (!safeRecommendedRelativePath(relativePath)) return null;
    return { sha256: normalizedDigest, role, relativePath, size: identity.size, mtimeMs: identity.mtimeMs, ctimeMs: identity.ctimeMs };
  } catch {
    return null;
  }
}

export function createSingleFlight(task) {
  if (typeof task !== "function") throw new TypeError("single-flight task must be a function");
  let active = null;
  const invoke = (...arguments_) => {
    if (active) return active;
    const request = Promise.resolve().then(() => task(...arguments_));
    active = request;
    const clear = () => { if (active === request) active = null; };
    request.then(clear, clear);
    return request;
  };
  Object.defineProperty(invoke, "active", { enumerable: true, get: () => Boolean(active) });
  return invoke;
}

export function createBackgroundSnapshot({
  initialSnapshot,
  initialState = {},
  refresh,
  persist = async () => {},
  now = Date.now,
  baseBackoffMs = 2000,
  maximumBackoffMs = RECOMMENDED_MAXIMUM_BACKOFF_MS,
  mergeConcurrent,
}) {
  if (typeof refresh !== "function") throw new TypeError("background snapshot refresh must be a function");
  let snapshot = initialSnapshot;
  let revision = 0;
  let state = clampRecommendedRemoteState({
    lastAttemptAt: timestamp(initialState.lastAttemptAt) ? initialState.lastAttemptAt : 0,
    lastSuccessAt: timestamp(initialState.lastSuccessAt) ? initialState.lastSuccessAt : 0,
    failureCount: Number.isSafeInteger(initialState.failureCount) && initialState.failureCount >= 0 ? initialState.failureCount : 0,
    nextAttemptAt: timestamp(initialState.nextAttemptAt) ? initialState.nextAttemptAt : 0,
  }, { now: now(), maximumBackoffMs });
  const safelyPersist = async () => {
    try {
      await persist(snapshot, { ...state });
    } catch {}
  };
  const run = createSingleFlight(async () => {
    const attemptedAt = now();
    const baseSnapshot = snapshot;
    const baseRevision = revision;
    state = { ...state, lastAttemptAt: attemptedAt };
    try {
      let nextSnapshot = await refresh(baseSnapshot);
      if (nextSnapshot === undefined || nextSnapshot === null) throw new Error("Background refresh returned no snapshot");
      if (revision !== baseRevision && typeof mergeConcurrent === "function") {
        nextSnapshot = await mergeConcurrent({ baseSnapshot, currentSnapshot: snapshot, refreshedSnapshot: nextSnapshot });
      }
      snapshot = nextSnapshot;
      revision += 1;
      state = { lastAttemptAt: attemptedAt, lastSuccessAt: now(), failureCount: 0, nextAttemptAt: 0 };
      await safelyPersist();
      return { started: true, ok: true, snapshot };
    } catch (error) {
      const failureCount = Math.min(1000, state.failureCount + 1);
      const delay = Math.min(maximumBackoffMs, baseBackoffMs * (2 ** Math.min(20, failureCount - 1)));
      state = { ...state, failureCount, nextAttemptAt: now() + delay };
      await safelyPersist();
      return { started: true, ok: false, error, snapshot };
    }
  });
  return {
    read() {
      state = clampRecommendedRemoteState(state, { now: now(), maximumBackoffMs });
      return { snapshot, state: { ...state }, refreshing: run.active };
    },
    refresh() {
      state = clampRecommendedRemoteState(state, { now: now(), maximumBackoffMs });
      if (!run.active && now() < state.nextAttemptAt) {
        return Promise.resolve({ started: false, ok: false, backoff: true, snapshot, nextAttemptAt: state.nextAttemptAt });
      }
      return run();
    },
    async replace(nextSnapshot, { successful = false } = {}) {
      if (nextSnapshot === undefined || nextSnapshot === null) throw new Error("Background snapshot replacement is empty");
      snapshot = nextSnapshot;
      revision += 1;
      if (successful) state = { ...state, lastSuccessAt: now(), failureCount: 0, nextAttemptAt: 0 };
      await safelyPersist();
      return snapshot;
    },
  };
}

async function readBoundedResponseBytes(response, maximumBytes, controller) {
  const lengthValue = response?.headers?.get?.("content-length");
  if (lengthValue != null && lengthValue !== "") {
    if (!/^\d+$/.test(lengthValue)) throw Object.assign(new Error("Remote JSON Content-Length is invalid"), { code: "ERESPONSESIZE" });
    if (Number(lengthValue) > maximumBytes) {
      controller.abort();
      throw Object.assign(new Error(`Remote JSON response exceeds ${maximumBytes} bytes`), { code: "ERESPONSESIZE" });
    }
  }
  if (!response?.body) throw new Error("Remote JSON response has no body");
  const chunks = [];
  let total = 0;
  const append = (value) => {
    const chunk = Buffer.from(value);
    total += chunk.length;
    if (total > maximumBytes) {
      controller.abort();
      throw Object.assign(new Error(`Remote JSON response exceeds ${maximumBytes} bytes`), { code: "ERESPONSESIZE" });
    }
    chunks.push(chunk);
  };
  if (typeof response.body.getReader === "function") {
    const reader = response.body.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        append(value);
      }
    } finally {
      reader.releaseLock?.();
    }
  } else if (typeof response.body[Symbol.asyncIterator] === "function") {
    for await (const chunk of response.body) append(chunk);
  } else {
    throw new Error("Remote JSON response body is not readable");
  }
  return Buffer.concat(chunks, total);
}

export async function fetchJsonWithDeadline(fetcher, url, options = {}, timeoutMs = 5000, maximumBytes = RECOMMENDED_REMOTE_JSON_MAX_BYTES) {
  if (typeof fetcher !== "function") throw new TypeError("fetcher must be a function");
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new TypeError("timeout must be positive");
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes <= 0 || maximumBytes > REMOTE_JSON_HARD_MAX_BYTES) throw new TypeError("remote JSON size limit is invalid");
  const controller = new AbortController();
  const externalSignal = options.signal;
  const abortFromExternal = () => controller.abort(externalSignal?.reason);
  if (externalSignal?.aborted) abortFromExternal();
  else externalSignal?.addEventListener("abort", abortFromExternal, { once: true });
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(Object.assign(new Error(`Remote JSON response exceeded ${timeoutMs} ms`), { name: "TimeoutError", code: "ETIMEDOUT" }));
    }, timeoutMs);
  });
  const request = Promise.resolve().then(async () => {
    const response = await fetcher(url, { ...options, signal: controller.signal });
    const bytes = await readBoundedResponseBytes(response, maximumBytes, controller);
    let text;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      throw new Error("Remote JSON response is not valid UTF-8");
    }
    const body = JSON.parse(text);
    return { response, body };
  });
  try {
    return await Promise.race([request, deadline]);
  } finally {
    clearTimeout(timer);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
}
