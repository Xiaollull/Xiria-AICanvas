import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  assertCivitaiArtifactForFamily,
  assertCivitaiResolutionForArtifact,
  clampRecommendedRemoteState,
  createBackgroundSnapshot,
  emptyRecommendedModelCache,
  fetchJsonWithDeadline,
  indexRecommendedInstallationCandidates,
  installationEntryForFile,
  mapWithRecommendedConcurrency,
  mergeConcurrentCivitaiSnapshots,
  mergeResolvedCivitaiBinding,
  readRecommendedModelCache,
  raceTrustedCivitaiFamilyVersions,
  recommendedCatalogFingerprint,
  safeRecommendedRelativePath,
  trustedCivitaiFamilyVersions,
  trustedInstallationsForArtifacts,
  validateCachedInstallation,
  validateCachedInstallations,
  validateRecommendedModelCache,
  verifyRecommendedFileAgainstCatalog,
  writeRecommendedModelCache,
} from "./recommended-model-cache.mjs";
import { filterPendingRecommendedArtifacts } from "./model-download-queue.mjs";

const digest = (character) => character.repeat(64);

function sourceBetween(source, start, end) {
  const startIndex = source.indexOf(start);
  assert.notEqual(startIndex, -1, `missing source anchor: ${start}`);
  const endIndex = source.indexOf(end, startIndex + start.length);
  assert.notEqual(endIndex, -1, `missing source anchor: ${end}`);
  return source.slice(startIndex, endIndex);
}

function remoteSnapshot() {
  return {
    savedAt: 100,
    families: [{
      id: "family",
      fetchedAt: 90,
      complete: true,
      models: [{ modelId: 34, versionId: 12, fileId: 56, label: "Version 12", filename: "model.safetensors", size: 5, sha256: digest("a"), baseModel: "Illustrious" }],
    }],
  };
}

test("recommended cache round-trips atomically and rejects corruption, schemas, fingerprints, and secret-shaped extras", async (context) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "xirai-recommended-cache-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const cachePath = path.join(directory, "state-cache", "recommended-model-cache.json");
  const fingerprint = recommendedCatalogFingerprint({ schema: 1, families: ["trusted"] });
  assert.equal(recommendedCatalogFingerprint({ b: 2, a: 1 }), recommendedCatalogFingerprint({ a: 1, b: 2 }));
  assert.notEqual(fingerprint, recommendedCatalogFingerprint({ schema: 1, families: ["changed"] }));
  const cache = {
    ...emptyRecommendedModelCache(fingerprint),
    remoteSnapshot: remoteSnapshot(),
    remote: { lastAttemptAt: 80, lastSuccessAt: 100, failureCount: 0, nextAttemptAt: 0 },
    installations: [{ sha256: digest("b"), role: "vae", relativePath: "nested/model.safetensors", size: 5, mtimeMs: 10, ctimeMs: 9 }],
  };

  await writeRecommendedModelCache(cachePath, cache);
  assert.deepEqual(await readRecommendedModelCache(cachePath, fingerprint), cache);
  const replaced = { ...cache, remote: { ...cache.remote, lastAttemptAt: 101 } };
  await writeRecommendedModelCache(cachePath, replaced);
  assert.deepEqual(await readRecommendedModelCache(cachePath, fingerprint), replaced);
  assert.deepEqual((await readdir(path.dirname(cachePath))).sort(), ["recommended-model-cache.json"]);

  await writeFile(cachePath, "{not-json", "utf8");
  assert.equal(await readRecommendedModelCache(cachePath, fingerprint), null);
  await writeFile(cachePath, JSON.stringify({ ...cache, schema: 1 }), "utf8");
  assert.equal(await readRecommendedModelCache(cachePath, fingerprint), null);
  await writeFile(cachePath, JSON.stringify(cache), "utf8");
  assert.equal(await readRecommendedModelCache(cachePath, digest("c")), null);

  const withSecret = { ...cache, apiKey: "must-never-be-written" };
  assert.equal(validateRecommendedModelCache(withSecret, fingerprint), null);
  await assert.rejects(writeRecommendedModelCache(cachePath, withSecret), /invalid recommended model cache/);
});

test("cached installation identity is accepted only inside its role root and invalidates after change or removal", async (context) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "xirai-recommended-identity-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const root = path.join(directory, "models", "vae");
  const model = path.join(root, "nested", "model.safetensors");
  await mkdir(path.dirname(model), { recursive: true });
  await writeFile(model, "model", "utf8");
  const entry = await installationEntryForFile({ sha256: digest("d"), role: "vae", filePath: model, root });
  assert.equal(entry.relativePath, "nested/model.safetensors");

  const rootForRole = async (role) => {
    assert.equal(role, "vae");
    return root;
  };
  let checked = await validateCachedInstallations([entry], { rootForRole });
  assert.equal(checked.installed.has(digest("d")), true);
  assert.equal(checked.entries.length, 1);

  await writeFile(model, "changed-size", "utf8");
  checked = await validateCachedInstallations([entry], { rootForRole });
  assert.equal(checked.installed.size, 0);
  await rm(model);
  assert.equal(await validateCachedInstallation(entry, { rootForRole }), null);
});

test("recommended cache paths reject lexical and resolved directory escapes", async () => {
  for (const unsafe of ["../model.safetensors", "nested/../model.safetensors", "/model.safetensors", "C:/model.safetensors", "nested\\model.safetensors", "./model.safetensors", "nested//model.safetensors"]) {
    assert.equal(safeRecommendedRelativePath(unsafe), "", unsafe);
  }
  assert.equal(safeRecommendedRelativePath("nested/model.safetensors"), "nested/model.safetensors");

  const root = path.resolve(os.tmpdir(), "recommended-root");
  const outside = path.resolve(os.tmpdir(), "outside-model.safetensors");
  const identity = { isFile: () => true, size: 5, mtimeMs: 2, ctimeMs: 1 };
  const escaped = await validateCachedInstallation({ sha256: digest("e"), role: "vae", relativePath: "nested/model.safetensors", size: 5, mtimeMs: 2, ctimeMs: 1 }, {
    rootForRole: async () => root,
    realpathFile: async (value) => path.resolve(value) === root ? root : outside,
    statFile: async () => identity,
  });
  assert.equal(escaped, null);
});

test("background catalog reads stay immediate and refresh work is single-flight", async () => {
  const oldSnapshot = { version: "cached" };
  let calls = 0;
  let release;
  let started;
  const began = new Promise((resolve) => { started = resolve; });
  const service = createBackgroundSnapshot({
    initialSnapshot: oldSnapshot,
    refresh: async () => {
      calls += 1;
      started();
      return new Promise((resolve) => { release = resolve; });
    },
  });

  const first = service.refresh();
  const second = service.refresh();
  await began;
  assert.equal(calls, 1);
  assert.equal(service.read().snapshot, oldSnapshot);
  assert.equal(service.read().refreshing, true);
  release({ version: "fresh" });
  const [firstResult, secondResult] = await Promise.all([first, second]);
  assert.equal(firstResult.ok, true);
  assert.equal(secondResult.ok, true);
  assert.deepEqual(service.read().snapshot, { version: "fresh" });
});

test("a remote response-body timeout retains the last snapshot and enters backoff", async () => {
  const oldSnapshot = remoteSnapshot();
  let clock = 1000;
  let persisted;
  const service = createBackgroundSnapshot({
    initialSnapshot: oldSnapshot,
    now: () => clock,
    baseBackoffMs: 25,
    refresh: async () => {
      const result = await fetchJsonWithDeadline(async (_url, options) => {
        assert.equal(options.signal instanceof AbortSignal, true);
        return {
          headers: { get: () => null },
          body: { getReader: () => ({ read: () => new Promise(() => {}), releaseLock() {} }) },
        };
      }, "https://example.invalid/catalog", {}, 15);
      return result.body;
    },
    persist: async (snapshot, state) => { persisted = { snapshot, state }; },
  });

  const result = await service.refresh();
  assert.equal(result.ok, false);
  assert.equal(result.error.code, "ETIMEDOUT");
  assert.equal(service.read().snapshot, oldSnapshot);
  assert.equal(service.read().state.failureCount, 1);
  assert.equal(service.read().state.nextAttemptAt, 1025);
  assert.equal(persisted.snapshot, oldSnapshot);
  clock = 1010;
  assert.equal((await service.refresh()).backoff, true);
});

test("disk installation entries are hints and cannot skip without process-trusted real-hash evidence", async () => {
  const artifact = { id: "trusted-artifact", role: "vae", size: 5, sha256: digest("a") };
  const diskEntry = { sha256: digest("a"), role: "vae", relativePath: "forged.safetensors", size: 5, mtimeMs: 2, ctimeMs: 1 };
  const candidates = indexRecommendedInstallationCandidates([diskEntry]);
  const trusted = new Map();
  const validateCandidate = async (entry) => ({ ...entry, absolutePath: "/models/forged.safetensors" });

  let result = await trustedInstallationsForArtifacts([artifact], { candidates, trusted, validateCandidate });
  assert.equal(result.installed.size, 0);
  assert.deepEqual(filterPendingRecommendedArtifacts([artifact], result.installed), [artifact]);

  const actual = await validateCandidate(diskEntry);
  trusted.set(digest("a"), actual);
  result = await trustedInstallationsForArtifacts([artifact], { candidates, trusted, validateCandidate });
  assert.equal(result.installed.has(digest("a")), true);

  const mismatchedSize = { ...artifact, size: 6 };
  result = await trustedInstallationsForArtifacts([mismatchedSize], { candidates, trusted, validateCandidate });
  assert.equal(result.installed.size, 0);
  assert.equal(candidates.has(digest("a")), false);
  assert.equal(trusted.has(digest("a")), false);
});

test("background verification ignores a candidate's claimed digest and uses the file's real hash", async () => {
  const expected = new Map([[digest("a"), { id: "a", role: "vae", size: 5, sha256: digest("a") }]]);
  const forged = { path: "/models/forged", size: 5, sha256: digest("a") };
  const rejected = await verifyRecommendedFileAgainstCatalog({
    file: forged,
    role: "vae",
    artifactsByDigest: expected,
    hashFile: async () => digest("b"),
    createEntry: async () => { throw new Error("must not create a trusted entry"); },
  });
  assert.equal(rejected, null);
  const accepted = await verifyRecommendedFileAgainstCatalog({
    file: forged,
    role: "vae",
    artifactsByDigest: expected,
    hashFile: async () => digest("a"),
    createEntry: async ({ sha256 }) => ({ sha256, verified: true }),
  });
  assert.deepEqual(accepted, { sha256: digest("a"), verified: true });
});

test("cold cache indexing performs no filesystem work and background pools never exceed sixteen", async () => {
  const entries = Array.from({ length: 4096 }, (_, index) => ({
    sha256: index.toString(16).padStart(64, "0"),
    role: "vae",
    relativePath: `${index}.safetensors`,
    size: index + 1,
    mtimeMs: 1,
    ctimeMs: 1,
  }));
  const indexed = indexRecommendedInstallationCandidates(entries);
  assert.equal(indexed.size, 4096);

  let active = 0;
  let maximumActive = 0;
  const values = Array.from({ length: 80 }, (_, index) => index);
  const results = await mapWithRecommendedConcurrency(values, async (value) => {
    active += 1;
    maximumActive = Math.max(maximumActive, active);
    await new Promise((resolve) => setTimeout(resolve, 1));
    active -= 1;
    return value * 2;
  }, 1000);
  assert.equal(maximumActive, 16);
  assert.deepEqual(results, values.map((value) => value * 2));

  active = 0;
  await assert.rejects(mapWithRecommendedConcurrency(values, async (value) => {
    active += 1;
    await new Promise((resolve) => setTimeout(resolve, value === 0 ? 1 : 3));
    active -= 1;
    if (value === 0) throw new Error("hash failed");
    return value;
  }, 16), /hash failed/);
  assert.equal(active, 0);
});

test("a real parent-model Civitai shape may omit child modelId while preserving the full trusted file binding", async () => {
  const family = { id: "family", modelId: 34, role: "checkpoint", baseModels: ["Illustrious"], versions: [[12, "Fallback"]] };
  const response = JSON.parse(await readFile(new URL("./civitai-model-response.fixture.json", import.meta.url), "utf8"));
  assert.equal(Object.hasOwn(response.modelVersions[0], "modelId"), false);
  const [record] = trustedCivitaiFamilyVersions(family, response);
  assert.deepEqual(record, { modelId: 34, versionId: 12, fileId: 56, label: "Bound", filename: "model.safetensors", size: 5, sha256: digest("a"), baseModel: "Illustrious" });
  const artifact = { id: "family-12", familyId: "family", modelId: 34, versionId: 12, fileId: 56, label: "Bound", role: "checkpoint", filename: "model.safetensors", size: 5, sha256: digest("a"), baseModel: "Illustrious" };
  assert.equal(assertCivitaiArtifactForFamily(family, artifact), artifact);
  assert.throws(() => assertCivitaiArtifactForFamily(family, { ...artifact, modelId: 99 }), /outside its trusted family/);
  const resolved = { modelId: 34, versionId: 12, fileId: 56, filename: "model.safetensors", size: 5, expectedSha256: digest("a"), baseModel: "Illustrious" };
  assert.equal(assertCivitaiResolutionForArtifact(family, artifact, resolved), resolved);
  assert.throws(() => trustedCivitaiFamilyVersions(family, { ...response, id: 99 }), /trusted model family/);
  assert.throws(() => trustedCivitaiFamilyVersions(family, { ...response, modelVersions: [{ ...response.modelVersions[0], modelId: 99 }] }), /does not belong/);
  assert.throws(() => trustedCivitaiFamilyVersions(family, { ...response, modelVersions: Array.from({ length: 4097 }, () => response.modelVersions[0]) }), /trusted model family/);
  assert.throws(() => assertCivitaiResolutionForArtifact(family, artifact, { ...resolved, fileId: 57 }), /no longer matches/);
  assert.throws(() => assertCivitaiResolutionForArtifact(family, { ...artifact, fileId: undefined, filename: "", size: 0, sha256: undefined }, { ...resolved, expectedSha256: "" }), /outside the trusted recommendation family/);
});

test("Civitai family racing rejects a fast invalid shape and waits for a slower trusted response", async () => {
  const family = { id: "family", modelId: 34, role: "checkpoint", baseModels: ["Illustrious"], versions: [[12, "Fallback"]] };
  const valid = JSON.parse(await readFile(new URL("./civitai-model-response.fixture.json", import.meta.url), "utf8"));
  let slowCompleted = false;
  const records = await raceTrustedCivitaiFamilyVersions(family, [
    async () => ({ ...valid, id: 999 }),
    async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
      slowCompleted = true;
      return valid;
    },
  ]);
  assert.equal(slowCompleted, true);
  assert.equal(records[0].modelId, 34);
  assert.equal(records[0].versionId, 12);
  assert.equal(records[0].fileId, 56);
});

test("a remote refresh merges a binding completed while its network request is in flight", async () => {
  const model = (versionId, fileId, marker) => ({
    modelId: 34,
    versionId,
    fileId,
    label: `Version ${versionId}`,
    filename: `model-${versionId}.safetensors`,
    size: versionId,
    sha256: digest(marker),
    baseModel: "Illustrious",
  });
  const base = { savedAt: 10, families: [{ id: "family", fetchedAt: 10, complete: true, models: [model(12, 112, "a")] }] };
  const refreshed = { savedAt: 30, families: [{ id: "family", fetchedAt: 30, complete: true, models: [model(13, 113, "b")] }] };
  const concurrentBinding = model(14, 114, "c");
  const withBinding = { savedAt: 20, families: [{ id: "family", fetchedAt: 20, complete: true, models: [model(12, 112, "a"), concurrentBinding] }] };
  let releaseRefresh;
  let markStarted;
  const refreshStarted = new Promise((resolve) => { markStarted = resolve; });
  const refreshGate = new Promise((resolve) => { releaseRefresh = resolve; });
  const persisted = [];
  const service = createBackgroundSnapshot({
    initialSnapshot: base,
    refresh: async () => {
      markStarted();
      await refreshGate;
      return refreshed;
    },
    mergeConcurrent: mergeConcurrentCivitaiSnapshots,
    persist: async (snapshot) => { persisted.push(snapshot); },
  });

  const refresh = service.refresh();
  await refreshStarted;
  await service.replace(withBinding);
  releaseRefresh();
  assert.equal((await refresh).ok, true);
  const final = service.read().snapshot;
  assert.deepEqual(final.families[0].models.map((item) => item.versionId).sort((a, b) => a - b), [13, 14]);
  assert.deepEqual(final.families[0].models.find((item) => item.versionId === 14), concurrentBinding);
  assert.deepEqual(persisted.at(-1), final);
});

test("a resolved SHA-less fallback becomes a persisted partial binding with actual identity", () => {
  const family = { id: "family", modelId: 34, role: "checkpoint", baseModels: ["Illustrious"], versions: [[12, "Fallback"]] };
  const fallback = { id: "family-12", familyId: "family", modelId: 34, versionId: 12, label: "Fallback", role: "checkpoint", filename: "", size: 0, baseModel: "Illustrious" };
  const resolved = { modelId: 34, versionId: 12, fileId: 56, filename: "model.safetensors", size: 5, expectedSha256: digest("a"), baseModel: "Illustrious" };
  const merged = mergeResolvedCivitaiBinding(null, family, fallback, resolved, 5, 1234);
  assert.deepEqual(merged.snapshot, {
    savedAt: 1234,
    families: [{ id: "family", fetchedAt: 1234, complete: false, models: [{ modelId: 34, versionId: 12, fileId: 56, label: "Fallback", filename: "model.safetensors", size: 5, sha256: digest("a"), baseModel: "Illustrious" }] }],
  });
  assert.equal(validateRecommendedModelCache({
    ...emptyRecommendedModelCache(digest("f")),
    remoteSnapshot: merged.snapshot,
  }, digest("f"))?.remoteSnapshot.families[0].models[0].sha256, digest("a"));
});

test("remote timestamps are clamped across future values and clock rollback", async (context) => {
  assert.deepEqual(clampRecommendedRemoteState({ lastAttemptAt: 5000, lastSuccessAt: 6000, failureCount: 3, nextAttemptAt: 999999 }, { now: 1000, maximumBackoffMs: 100 }), {
    lastAttemptAt: 0,
    lastSuccessAt: 0,
    failureCount: 3,
    nextAttemptAt: 1100,
  });
  const directory = await mkdtemp(path.join(os.tmpdir(), "xirai-recommended-time-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const cachePath = path.join(directory, "cache.json");
  const fingerprint = digest("f");
  await writeRecommendedModelCache(cachePath, {
    ...emptyRecommendedModelCache(fingerprint),
    remote: { lastAttemptAt: 5000, lastSuccessAt: 6000, failureCount: 2, nextAttemptAt: 999999 },
  });
  const read = await readRecommendedModelCache(cachePath, fingerprint, { now: 1000, maximumBackoffMs: 100 });
  assert.deepEqual(read.remote, { lastAttemptAt: 0, lastSuccessAt: 0, failureCount: 2, nextAttemptAt: 1100 });
});

test("remote JSON enforces Content-Length and streamed byte limits before parsing", async () => {
  await assert.rejects(fetchJsonWithDeadline(async () => ({
    headers: { get: () => "33" },
    body: new Response("{}").body,
  }), "https://example.invalid", {}, 100, 32), (error) => error.code === "ERESPONSESIZE");
  await assert.rejects(fetchJsonWithDeadline(async () => ({
    headers: { get: () => null },
    body: new Response(`{"value":"${"x".repeat(40)}"}`).body,
  }), "https://example.invalid", {}, 100, 32), (error) => error.code === "ERESPONSESIZE");
  const result = await fetchJsonWithDeadline(async () => new Response('{"ok":true}', { headers: { "Content-Length": "11" } }), "https://example.invalid", {}, 100, 32);
  assert.deepEqual(result.body, { ok: true });
});

test("recommended GET and POST use background refresh plus selected-cache O(1) admission", async () => {
  const vite = await readFile(new URL("../vite.config.js", import.meta.url), "utf8");
  const initialization = sourceBetween(vite, "async function initializeRecommendedModelRuntime", "function ensureRecommendedModelRuntime");
  const scanner = sourceBetween(vite, "async function recommendedArtifactInstallations", "const reviewRecommendedInstallations");
  const getHandler = sourceBetween(vite, "async function sendRecommendedCatalog", "function recommendedArtifactById");
  const postHandler = sourceBetween(vite, "async function downloadRecommendedModels", "async function runRecommendedModelDownloadBatch");
  const worker = sourceBetween(vite, "async function runRecommendedModelDownloadBatch", "async function downloadRecommendedYoloModels");

  assert.match(getHandler, /void recommendedRemoteCatalog\.refresh\(\)/);
  assert.match(getHandler, /triggerRecommendedInstallationReview\(\)/);
  assert.doesNotMatch(getHandler, /await recommendedArtifactInstallations|await recommendedRemoteCatalog\.refresh/);
  assert.match(initialization, /indexRecommendedInstallationCandidates\(base\.installations\)/);
  assert.match(initialization, /mergeConcurrent: mergeConcurrentCivitaiSnapshots/);
  assert.doesNotMatch(initialization, /validateCachedInstallations|realpath|Promise\.all\(base\.installations/);
  assert.match(scanner, /mapWithRecommendedConcurrency\(candidates/);
  assert.match(scanner, /verifyRecommendedFileAgainstCatalog/);
  assert.match(postHandler, /await recommendedInstallationsForArtifacts\(uniqueArtifacts\)/);
  assert.match(postHandler, /filterPendingRecommendedArtifacts\(uniqueArtifacts, installed/);
  assert.doesNotMatch(postHandler, /recommendedArtifactInstallations|findFilesBySize|cachedRecommendedFileHash/);
  assert.match(worker, /artifact\.provider === "Civitai" \? resolved\.expectedSha256/);
  assert.match(worker, /await cacheCompletedRecommendedInstallation/);
  assert.match(worker, /await bindCompletedCivitaiArtifact/);
  assert.match(worker, /triggerRecommendedInstallationReview\(\)/);
});

test("recommended UI requests are bounded and background installation review never locks download", async () => {
  const downloader = await readFile(new URL("../src/ModelDownloader.jsx", import.meta.url), "utf8");
  const request = sourceBetween(downloader, "  const requestRecommendedCatalog", "  const refreshRecommended =");
  const start = sourceBetween(downloader, "  const startRecommendedDownload", "  const retryFailedDownloads");
  const polling = sourceBetween(downloader, "  const syncModelDownloadJob", "  useEffect(() => {\n    const completionKey");
  const submission = sourceBetween(downloader, "  const runDownload", "  const startDownload");

  assert.match(request, /new AbortController\(\)/);
  assert.match(request, /signal: controller\.signal/);
  assert.match(request, /window\.setTimeout\(\(\) => \{ timedOut = true; controller\.abort\(\); \}, 4000\)/);
  assert.match(request, /const continuePolling = \(stillReviewing \|\| stillRefreshing\) && pollAttempt < 60/);
  assert.match(request, /checkingInstalled: continuePolling && stillReviewing/);
  assert.match(request, /reconcileRecommendedSelection\(nextFamilies/);
  assert.match(request, /families: nextFamilies \|\| current\.families/);
  assert.doesNotMatch(start, /recommended\.checkingInstalled|recommended\.installedChecked/);
  assert.match(start, /!selectedRecommendedModel/);
  assert.match(start, /civitai_key: credentials\.civitai/);
  assert.match(polling, /signal/);
  assert.match(polling, /downloadPollGate\.current\.commit/);
  assert.match(polling, /controller\.abort\(\)/);
  assert.match(submission, /downloadPollGate\.current\.invalidate\(\)/);
  assert.match(submission, /downloadPollController\.current\?\.abort\(\)/);
  assert.match(downloader, /disabled=\{submittingDownload \|\| !selectedRecommendedModel \|\| !pendingRecommendedArtifacts\.length/);
  assert.doesNotMatch(downloader, /disabled=\{submittingDownload \|\| recommended\.checkingInstalled/);
  assert.match(downloader, /安装状态正在后台复核，可立即提交下载/);
});
