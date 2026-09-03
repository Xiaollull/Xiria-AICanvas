import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createLoraPersistenceEpochGuard, runLoraPersistenceEpoch } from "../src/lora-persistence-epoch.js";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const deferred = () => { let resolve; let reject; const promise = new Promise((res, rej) => { resolve = res; reject = rej; }); return { promise, resolve, reject }; };

test("persistence epoch cancels a delayed GET when a lock arrives", async () => {
  const guard = createLoraPersistenceEpochGuard(); guard.mount();
  const pendingGet = deferred(); let putCalls = 0;
  const result = runLoraPersistenceEpoch({
    guard, epoch: guard.nextEpoch(), admission: () => ({ locked: false, syncReady: true }),
    get: () => pendingGet.promise, prepare: (state) => state,
    put: async () => { putCalls += 1; return { ok: true }; },
  });
  guard.invalidate(); // workspace-lora-lock arrives while GET is unresolved
  pendingGet.resolve({ state: {} });
  assert.deepEqual(await result, { written: false, stale: true });
  assert.equal(putCalls, 0);
});

test("persistence epoch cancels immediately before PUT and retries once after unlock", async () => {
  const guard = createLoraPersistenceEpochGuard(); guard.mount();
  const beforePut = deferred(); const prepareStarted = deferred(); let putCalls = 0;
  const stale = runLoraPersistenceEpoch({
    guard, epoch: guard.nextEpoch(), admission: () => ({ locked: false, syncReady: true }),
    get: async () => ({ state: {} }),
    prepare: async (state) => { prepareStarted.resolve(); await beforePut.promise; return state; },
    put: async () => { putCalls += 1; return { ok: true }; },
  });
  await prepareStarted.promise;
  guard.invalidate(); // lock transition immediately before PUT admission
  beforePut.resolve();
  assert.deepEqual(await stale, { written: false, stale: true });
  assert.equal(putCalls, 0);
  const fresh = await runLoraPersistenceEpoch({
    guard, epoch: guard.nextEpoch(), admission: () => ({ locked: false, syncReady: true }),
    get: async () => ({ state: {} }), prepare: (state) => state,
    put: async () => { putCalls += 1; return { ok: true }; },
  });
  assert.deepEqual(fresh, { written: true, stale: false });
  assert.equal(putCalls, 1);
});

test("LoRA asset page is lazy, opens in a new tab, and shares one sync channel", async () => {
  const [app, main, page, state, packageJson] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("src/main.jsx"),
    readSource("src/LoraManagerPage.jsx"),
    readSource("src/lora-state.js"),
    readSource("package.json").then(JSON.parse),
  ]);

  assert.match(app, /window\.open\("\/lora", "_blank", "noopener,noreferrer"\)/);
  assert.match(app, /lora-modal \$\{loraManagerMaximized \? "maximized" : ""\}/);
  assert.match(main, /lazy\(\(\) => import\("\.\/LoraManagerPage"\)\)/);
  assert.match(main, /const path = window\.location\.pathname/);
  assert.match(main, /path === "\/lora"/);
  assert.match(main, /path === "\/lora\/"/);
  assert.match(state, /LORA_SYNC_CHANNEL = "xirai-lora-workspace-v1"/);
  assert.match(app, /new BroadcastChannel\(LORA_SYNC_CHANNEL\)/);
  assert.match(page, /new BroadcastChannel\(LORA_SYNC_CHANNEL\)/);
  assert.match(page, /postMessage\(\{ type: "request-workspace-loras" \}\)/);
  assert.match(page, /matchMedia\?\.\("\(prefers-reduced-motion: reduce\)"\)\.matches/);
  // The entrance animation runs on the platform's own Web Animations API. It used to be GSAP,
  // whose licence the project's AGPL-3.0 cannot absorb, so no animation dependency is the point.
  assert.equal(packageJson.dependencies.gsap, undefined);
  assert.match(page, /from "\.\/entrance-animation\.js"/);
});

test("LoRA asset page persists only explicit scoped local edits without overwriting workspace state", async () => {
  const page = await readSource("src/LoraManagerPage.jsx");

  assert.match(page, /const ENGINES = \["SD", "iL", "Anima"\]/);
  assert.match(page, /useState\(\{ SD: null, iL: null, Anima: null \}\)/);
  assert.match(page, /ENGINES\.map\(async \(engine\) =>/);
  assert.match(page, /engineScopeKey\(engine\)/);
  assert.match(page, /normalizeMountedLoraMap\(workspace\.mountedLorasByEngine/);
  assert.match(page, /fieldMissing: !Object\.prototype\.hasOwnProperty\.call\(workspace, "mountedLorasByEngine"\)/);
  assert.match(page, /if \(!initializedRef\.current \|\| !localChangeRef\.current \|\| workspaceLocked \|\| !syncReady \|\| !canPersistRef\.current\) return undefined/);
  assert.match(page, /localChangeRef\.current = true;/);
  assert.match(page, /mountedLorasMapRef\.current = map;\s+lorasRef\.current = next;\s+setMountedLorasByEngine\(map\);\s+setLoras\(next\)/);
  assert.match(page, /payload\.engine \|\| payload\.scopeKey/);
  assert.match(page, /pendingScopedSyncRef\.current\.set\(payload\.engine \|\| payload\.scopeKey, normalizeMountedLoras\(payload\.loras\)\)/);
  assert.match(page, /for \(const \[pendingScopeKey, pendingLoras\] of pendingScopedSyncRef\.current\)/);
  assert.match(page, /mountedLorasByEngine: map/);
  assert.match(page, /loras: mountedLorasForScope\(map, serverScopeKey\)/);
  assert.match(page, /workspaceLocked \|\| !syncReady/);
  assert.doesNotMatch(page, /mountedLorasByModel/);
  assert.match(page, /if \(workspaceLockedRef\.current\) return false;/);
  assert.match(page, /requestToken !== libraryScanTokenRef\.current/);
  // A lock still cancels an in-flight scan here and an in-flight drag in the
  // shared mount panel, which ends its own gesture when `locked` reaches it as
  // a prop rather than relying on each host to remember the call.
  assert.match(page, /const applyWorkspaceLock = \(locked\) => \{[\s\S]*workspaceLockedRef\.current = locked;[\s\S]*libraryScanTokenRef\.current \+= 1;/);
  assert.match(page, /locked=\{loraDragLocked\}/);
  const panel = await readSource("src/LoraMountPanel.jsx");
  assert.match(panel, /if \(locked\) clearSession\(\);/);
  assert.match(page, /applyWorkspaceLock\(payload\.locked === true\);/);
  assert.match(page, /if \(!syncReadyRef\.current\) applyWorkspaceLock\(active\);/);
  assert.match(page, /if \(!syncReadyRef\.current\) applyWorkspaceLock\(false\);/);
  assert.match(page, /if \(!stateReady \|\| workspaceLocked\) return;/);
  assert.match(page, /disabled=\{refreshing \|\| workspaceLocked\}/);
  assert.match(page, /persistenceEpochRef\.current\.invalidate\(\);/);
  assert.match(page, /runLoraPersistenceEpoch\(\{/);
  assert.match(page, /signal \}\);/);
});

test("LoRA asset page uses fixed switchable views and portrait library cards", async () => {
  const [page, styles] = await Promise.all([
    readSource("src/LoraManagerPage.jsx"),
    readSource("src/styles.css"),
  ]);

  assert.match(page, /useState\("overview"\)/);
  assert.match(page, /className="lora-page-view-switch"/);
  assert.match(page, />挂载管理<\/button>/);
  assert.match(page, />资产概览<\/button>/);
  assert.ok(page.indexOf(">资产概览</button>") < page.indexOf(">挂载管理</button>"));
  assert.match(page, /activeView === "overview"/);
  assert.match(page, /lora-library-card lora-page-library-card/);
  assert.match(styles, /\.lora-page-shell \{ height: 100vh; height: 100dvh;[^}]*overflow: hidden;/);
  assert.match(styles, /\.lora-page-library-card \.lora-card-preview \{ aspect-ratio: 3 \/ 4; \}/);
  assert.match(styles, /\.lora-page-library-grid \{ grid-template-columns: repeat\(auto-fill, minmax\(176px, 205px\)\)/);
  assert.match(styles, /\.lora-page-view-switch button \{[^}]*font: 11px "DM Mono"/);
  assert.match(styles, /\.lora-page-library-card \.lora-card-copy > strong \{[^}]*font-size: 13px/);
});

test("LoRA cards open a shared layered metadata and trigger-word dialog", async () => {
  const [app, page, details, styles, vite] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("src/LoraManagerPage.jsx"),
    readSource("src/LoraDetailsDialog.jsx"),
    readSource("src/styles.css"),
    readSource("vite.config.js"),
  ]);
  assert.match(app, /className="lora-card-detail"/);
  assert.match(page, /className="lora-card-detail"/);
  assert.match(app, /<LoraDetailsDialog/);
  assert.match(page, /<LoraDetailsDialog/);
  assert.match(details, />查看详细信息|LoRA 详细信息/);
  assert.match(details, /触发词与特征组合/);
  assert.match(details, /风格 LoRA 只显示明确触发词/);
  assert.match(details, /自动审查摘要/);
  assert.match(details, /promptReview\?\.versionScope/);
  assert.match(details, /复制全部/);
  assert.match(styles, /\.lora-detail-backdrop \{ position: fixed; z-index: 340;/);
  assert.match(vite, /detailSchema = 1/);
  assert.match(vite, /reviewLoraPrompts/);
  assert.match(vite, /readLoraFileMetadata/);
});

test("LoRA metadata lookup reuses file-identity cache and keeps preview download off the detail critical path", async () => {
  const [app, page, vite] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("src/LoraManagerPage.jsx"),
    readSource("vite.config.js"),
  ]);
  assert.match(app, /JSON\.stringify\(\{ engine: model, path: lora\.value, refresh, category: categoryId \}\)/);
  assert.match(page, /JSON\.stringify\(\{ engine: model, path: item\.value, refresh, category: categoryId \}\)/);
  assert.match(app, /lookupLora\(item, true, loraDetail\.categoryId\)/);
  assert.match(page, /lookupLora\(item, true, detail\.categoryId\)/);
  assert.match(vite, /if \(!refresh && loraMetadataCacheValid\(cachedMetadata, fileStat, reviewKind\)\) return cachedMetadata/);
  assert.match(vite, /loraMetadataRequests\.has\(key\)/);
  assert.match(vite, /metadata\.previewUrl = previewCandidates\[0\]\?\.url \|\| null/);
  assert.match(vite, /if \(!metadata\?\.previewFile && metadata\?\.previewUrl\) metadata = await ensureLoraPreview\(modelPath, metadata, reviewKind\)/);
  assert.match(vite, /existingLoraPreviewFile/);
  assert.match(vite, /metadata\.previewSchema = 1/);
  assert.match(vite, /metadata\?\.status === "found" && !metadata\.previewFile && !metadata\.previewUrl\) metadata = await cacheLoraMetadata/);
  assert.match(vite, /preserveCachedRemote/);
  assert.match(vite, /reviewedLoraMetadata/);
  assert.match(vite, /cachedMetadata\.triggerReviewSchema !== TRIGGER_REVIEW_SCHEMA/);
  assert.match(vite, /metadata\.detailSchema === 1 && \(metadata\.triggerReviewSchema !== TRIGGER_REVIEW_SCHEMA \|\| metadata\.triggerReviewKind !== reviewKind \|\| !Object\.hasOwn\(metadata\.promptReview \|\| \{\}, "versionScopeKind"\)\)/);
  assert.doesNotMatch(vite, /version\.description = version\.description \|\| model\.description/);
  assert.match(app, /item\.metadata\?\.triggerReviewSchema !== 4/);
  assert.match(page, /item\.metadata\?\.triggerReviewSchema !== 4/);
});
