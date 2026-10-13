import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import crypto from "node:crypto"; import path from "node:path"; import os from "node:os"; import net from "node:net";
import { createServer } from "vite"; import react from "@vitejs/plugin-react";
import { LoraPublishedStore } from "./lora-published-store.mjs";
import { uiStateFixture } from "./ui-state-safety-fixture-data.mjs";

const ROOT = path.resolve(import.meta.dirname, ".."), EVIDENCE = process.env.UI_STATE_EVIDENCE_DIR || path.join(os.tmpdir(), "opencode", `ui-state-browser-${Date.now()}`);
await mkdir(EVIDENCE, { recursive: true }); const sha = b => crypto.createHash("sha256").update(b).digest("hex"), sleep = ms => new Promise(r => setTimeout(r, ms));
const source = await readFile(path.join(ROOT, "vite.config.js"), "utf8"), body = source.slice(source.indexOf("async function uiStateApi("), source.indexOf("\nfunction uiStateApiPlugin()"));
let current; const ledger = { scope: "executor self-test; actual React StrictMode App + actual loraFetch + extracted current UI middleware + real published lease/CAS store; synthetic Temp only", checks: [], visualChecks: [], accessibilityChecks: [] };
const middleware = async (req, res, next) => {
  if (new URL(req.url, "http://fixture").pathname !== "/api/ui-state") return next();
  const c = current, item = { method: req.method, revision: req.headers["x-xirai-lora-revision"] || null, version: req.headers["x-xirai-document-version"] || null }; c.requests.push(item);
  res.setHeader("Content-Type", "application/json");
  const end = res.end.bind(res); res.end = (...args) => { item.status = res.statusCode; return end(...args); };
  if (req.method === "GET") {
    if (c.mode === "visual-delay") await sleep(1300);
    if (c.mode === "network") { req.socket.destroy(); return; }
    if (c.mode === "timeout") return;
    if (c.mode === "http500") { res.statusCode = 500; res.end('{"error":"fixture read failure"}'); return; }
     if (c.mode === "late") { const mode = c.lateResult; await new Promise(r => (c.releases ||= []).push(r)); if (mode === "failed") { res.statusCode = 500; res.end('{}'); return; } }
    if (c.mode === "badjson" || c.mode.startsWith("shape:")) {
      const receipt = await c.store.read("uiState"); res.setHeader("X-XirAI-LoRA-Revision", receipt.revision); res.setHeader("X-XirAI-Document-Version", receipt.version);
      res.end(c.mode === "badjson" ? '{"state":' : c.mode.slice(6)); return;
    }
   } else if (req.method === "PUT" && c.putFault) {
     if (c.putFault === "network") { req.socket.destroy(); return; }
     if (c.putFault === "badack") { res.end('{}'); return; }
     res.statusCode = c.putFault; res.end(JSON.stringify({ error: "fixture save fault" })); return;
   }
   if (req.method === "PUT" && c.holdPut) { c.holdPut = false; await new Promise(r => c.releasePut = r); }
  const readJsonRequest = async request => { const chunks = []; for await (const chunk of request) chunks.push(chunk); return JSON.parse(Buffer.concat(chunks)); };
  const owner = c.mode === "eacces" ? { read: async () => { throw Object.assign(Error("fixture EACCES mock"), { code: "EACCES" }); } } : c.store;
  const api = new Function("loraPublishedStore", "publishedHeaders", "requestPublication", "readJsonRequest", "maximumUiStateBytes", "schedulePromptPresetPreviewPrune", "requireSameOrigin", `${body}; return uiStateApi;`)(owner,
    (response, receipt) => { response.setHeader("X-XirAI-LoRA-Revision", receipt.revision); response.setHeader("X-XirAI-Document-Version", receipt.version); },
    request => ({ revision: Number(request.headers["x-xirai-lora-revision"] || 0), version: request.headers["x-xirai-document-version"] }), readJsonRequest, 16 * 1024 * 1024, async () => {},
    request => { if (request.headers.origin && new URL(request.headers.origin).host !== request.headers.host) throw Error("fixture origin mismatch"); });
  await api(req, res, next);
};
class Page {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.errors = []; ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id) { const p = this.pending.get(m.id); if (p) { this.pending.delete(m.id); m.error ? p.reject(Error(m.error.message)) : p.resolve(m.result); } } else if (m.method === "Runtime.exceptionThrown") this.errors.push(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text); }; }
  send(method, params = {}) { const id = ++this.id; this.ws.send(JSON.stringify({ id, method, params })); return new Promise((r, j) => this.pending.set(id, { resolve: r, reject: j })); }
  async eval(expression) { const r = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result.value; }
   async wait(expression, ms = 16000) { for (let i = 0; i < ms / 100; i++) { if (await this.eval(`Boolean(${expression})`)) return; await sleep(100); } throw Error("Timeout: " + expression); }
  async click(text) { return this.eval(`(()=>{const b=[...document.querySelectorAll('button')].find(e=>e.textContent===${JSON.stringify(text)});if(!b)throw Error('missing button');b.click();})()`); }
  async fill(value) { await this.eval(`(()=>{const e=document.querySelector('.prompt-field textarea');e.focus();const setter=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;setter.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('input',{bubbles:true}));})()`); }
}
let edge, page, ws;
const vite = await createServer({ configFile: false, root: ROOT, cacheDir: path.join(EVIDENCE, "vite-cache"), appType: "mpa", logLevel: "error", plugins: [react(), { name: "isolated-ui-owner", configureServer(server) { server.middlewares.use((req, res, next) => { void middleware(req, res, next).catch(e => { res.statusCode = 500; res.end(JSON.stringify({ error: e.message })); }); }); } }], server: { host: "127.0.0.1", port: 0, hmr: false, open: false, watch: null }, optimizeDeps: { noDiscovery: true, include: ["react", "react-dom/client", "lucide-react"] } });
async function configure(label, mode, revision = 5, value = uiStateFixture(), query = "") {
  await page.send("Page.navigate", { url: "about:blank" }); await sleep(350);
  const dir = path.join(EVIDENCE, label); await mkdir(path.join(dir, "lora-library"), { recursive: true });
   if (value !== null) await writeFile(path.join(dir, "ui-state.json"), typeof value === "string" ? value : JSON.stringify(value, null, 2));
  await writeFile(path.join(dir, "lora-library/publication.json"), JSON.stringify({ schemaVersion: 1, revision, phase: "published" }));
  current = { label, mode, dir, revision, requests: [], store: new LoraPublishedStore({ stateDirectory: dir }) };
  try { current.before = await readFile(path.join(dir, "ui-state.json")); } catch (e) { if (e.code !== "ENOENT") throw e; current.before = null; }
   await page.send("Page.navigate", { url: `http://127.0.0.1:${vite.httpServer.address().port}/scripts/ui-state-safety.fixture.html${query}` });
  return current;
}
async function record(c, extra = {}) { let after; try { after = await readFile(path.join(c.dir, "ui-state.json")); } catch (e) { if (e.code !== "ENOENT") throw e; }
   const check = { label: c.label, mode: c.mode, beforeSha256: c.before ? sha(c.before) : null, afterSha256: after ? sha(after) : null, requests: structuredClone(c.requests), ...extra }; ledger.checks.push(check); return { check, after }; }
async function screenshot(name) { const { data } = await page.send('Page.captureScreenshot', { format: 'png' }); await writeFile(path.join(EVIDENCE, name + '.png'), Buffer.from(data, 'base64')); }
async function loaderGeometry() { return page.eval(`(()=>{const root=document.querySelector('.app-loader'), rect=e=>{const r=e.getBoundingClientRect();return {x:+r.x.toFixed(2),y:+r.y.toFixed(2),width:+r.width.toFixed(2),height:+r.height.toFixed(2),right:+r.right.toFixed(2),bottom:+r.bottom.toFixed(2)}}, emblem=root.querySelector('.loader-emblem'),logo=root.querySelector('.loader-logo'),brand=root.querySelector('.loader-brand'),brandName=brand.querySelector('span'),brandAI=brand.querySelector('strong'),status=root.querySelector('.loader-status'),note=root.querySelector('.app-state-loader-note'),es=getComputedStyle(emblem),ls=getComputedStyle(logo),ns=getComputedStyle(brandName),as=getComputedStyle(brandAI),ss=getComputedStyle(status),er=rect(emblem),br=rect(brand),sr=rect(status);return {state:root.dataset.uiState||'boot',role:root.getAttribute('role'),viewport:{width:innerWidth,height:innerHeight},emblem:er,logo:{...rect(logo),cssWidth:ls.width,cssHeight:ls.height},brand:br,status:sr,note:note?rect(note):null,typography:{brandName:ns.fontSize,brandNameWeight:ns.fontWeight,brandAI:as.fontSize,status:ss.fontSize,statusFamily:ss.fontFamily},gaps:{logoToBrand:+(br.y-er.bottom).toFixed(2),brandToStatus:+(sr.y-br.bottom).toFixed(2)},centerX:{emblem:+(er.x+er.width/2).toFixed(2),brand:+(br.x+br.width/2).toFixed(2),status:+(sr.x+sr.width/2).toFixed(2)},statusText:status.textContent.trim(),hasProgress:!!root.querySelector('.loader-track,.loader-readout'),content:rect(root.querySelector('.loader-content')),documentWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth}})()`); }
try {
  await vite.listen(); const probe = net.createServer(); await new Promise(r => probe.listen(0, "127.0.0.1", r)); const port = probe.address().port; await new Promise(r => probe.close(r));
  const profile = path.join(EVIDENCE, "isolated-edge-profile"); await mkdir(profile, { recursive: true });
  edge = spawn("C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--no-first-run", "--disable-extensions", "--disable-sync", "--disable-background-networking", "about:blank"], { stdio: "ignore", windowsHide: true });
  for (let i = 0; i < 100; i++) { try { if ((await fetch(`http://127.0.0.1:${port}/json/version`)).ok) break; } catch {} await sleep(100); }
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); ws = new WebSocket(targets.find(t => t.type === "page").webSocketDebuggerUrl); page = new Page(ws); await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
   await page.send("Page.enable"); await page.send("Runtime.enable"); await page.send("Network.enable"); await page.send("Network.setBlockedURLs", { urls: ["http://localhost:7709/*", "http://127.0.0.1:7709/*", "http://localhost:8718/*", "http://127.0.0.1:8718/*", "https://*"] });
   const visualViewports = [[1920,1080],[1440,900],[390,844],[390,430]];
   for (const [width, height] of visualViewports) {
     await page.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width <= 500 });
     if (width === 390 && height === 844) await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
     const c = await configure(`loader-visual-${width}x${height}`, "visual-delay", 5, uiStateFixture(), "?hold-catalog");
     await page.wait("document.querySelector('.app-shell[data-ui-state=loading] .app-state-loader .loader-emblem')"); await sleep(60);
     const read = await loaderGeometry();
     assert.equal(read.state, "loading"); assert.equal(read.role, "status"); assert.equal(read.hasProgress, false);
     assert.equal(read.statusText, "正在读取项目页面状态"); assert.ok(read.note && read.note.bottom <= height, "hydration note remains in the viewport");
     assert.ok(read.emblem.y >= 0 && read.status.bottom <= height, "hydration brand remains visible on short viewports");
     assert.ok(read.documentWidth <= read.clientWidth, "hydration screen has no horizontal overflow");
     assert.equal(c.requests.filter(r => r.method === "PUT").length, 0, "no write while the initial UI-state read is held");
     assert.equal(await page.eval("localStorage.getItem('xirai-assistant-prompt-snapshot-v1')===window.__localPromptBefore&&window.__localPromptWrites.length===0"), true);
     await screenshot(`loader-read-${width}x${height}`);
     await page.wait("document.querySelector('.app-shell[data-ui-state=loaded] .app-loader:not(.app-state-loader) .loader-emblem')");
     const boot = await loaderGeometry();
     assert.equal(boot.state, "boot"); assert.equal(boot.hasProgress, true);
     assert.deepEqual([read.emblem.width,read.emblem.height,read.logo.width,read.logo.height,read.logo.cssWidth,read.logo.cssHeight,read.typography.brandName,read.typography.brandNameWeight,read.typography.brandAI,read.typography.status,read.typography.statusFamily,read.gaps.logoToBrand,read.gaps.brandToStatus], [boot.emblem.width,boot.emblem.height,boot.logo.width,boot.logo.height,boot.logo.cssWidth,boot.logo.cssHeight,boot.typography.brandName,boot.typography.brandNameWeight,boot.typography.brandAI,boot.typography.status,boot.typography.statusFamily,boot.gaps.logoToBrand,boot.gaps.brandToStatus], `${width}x${height} hydration geometry matches the normal boot loader`);
     for (const stage of [read, boot]) for (const key of ["emblem","brand","status"]) assert.ok(Math.abs(stage.centerX[key] - width / 2) <= 1, `${width}x${height} ${key} stays centered`);
     assert.ok(boot.emblem.y >= 0 && boot.status.bottom <= height, "normal boot loader remains visible in viewport");
     if (width === 390 && height === 844) assert.equal(await page.eval("getComputedStyle(document.querySelector('.loader-orbit')).animationName"), "none", "reduced-motion disables the normal loader orbit");
     await screenshot(`loader-boot-${width}x${height}`);
     await page.eval("window.__releaseCatalog?.()");
     ledger.visualChecks.push({ viewport: `${width}x${height}`, read, boot, matchingGeometry: true, hydrationHadProgress: false, noPreHydrationWrites: true });
     if (width === 390 && height === 844) await page.send("Emulation.setEmulatedMedia", { features: [] });
     await page.send("Emulation.clearDeviceMetricsOverride");
   }
   await page.send("Emulation.setDeviceMetricsOverride", { width: 390, height: 430, deviceScaleFactor: 1, mobile: true });
   await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
   const mobileError = await configure("loader-error-mobile-short", "http500"); await page.wait("document.querySelector('.app-shell[data-ui-state=error] .app-state-loader-retry')");
   const errorLayout = await page.eval(`(()=>{const root=document.querySelector('.app-state-loader'),note=root.querySelector('.app-state-loader-note'),button=root.querySelector('.app-state-loader-retry'),n=note.getBoundingClientRect(),b=button.getBoundingClientRect(),s=getComputedStyle(note),orbits=[...root.querySelectorAll('.loader-orbit,.loader-motion-arcs,.loader-status i')].map(e=>getComputedStyle(e).animationName);return {role:root.getAttribute('role'),state:root.dataset.uiState,viewport:{width:innerWidth,height:innerHeight},note:{text:note.textContent,top:n.top,bottom:n.bottom,scrollHeight:note.scrollHeight,lineHeight:parseFloat(s.lineHeight),wrap:s.overflowWrap},button:{top:b.top,bottom:b.bottom,width:b.width,height:b.height,focused:document.activeElement===button},orbits,documentWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth,reduce:matchMedia('(prefers-reduced-motion: reduce)').matches,localPromptUnchanged:localStorage.getItem('xirai-assistant-prompt-snapshot-v1')===window.__localPromptBefore&&window.__localPromptWrites.length===0}})()`);
   assert.equal(errorLayout.state, "error"); assert.equal(errorLayout.role, "alert"); assert.equal(errorLayout.reduce, true);
   assert.ok(errorLayout.note.scrollHeight > errorLayout.note.lineHeight * 1.1, "long Chinese error message wraps within the mobile width");
   assert.equal(errorLayout.note.wrap, "anywhere"); assert.ok(errorLayout.note.top >= 0 && errorLayout.note.bottom <= 430);
   assert.equal(errorLayout.documentWidth <= errorLayout.clientWidth, true, "error text creates no horizontal overflow");
   assert.ok(errorLayout.button.height >= 44 && errorLayout.button.width >= 112 && errorLayout.button.top >= 0 && errorLayout.button.bottom <= 430, "retry has a visible keyboard/touch target on a short screen");
   assert.deepEqual(errorLayout.orbits, ["none","none","none","none","none"]); assert.equal(errorLayout.localPromptUnchanged, true);
   assert.equal(mobileError.requests.filter(r => r.method === "PUT").length, 0, "failed read does not issue a UI-state PUT");
   await screenshot("loader-error-mobile-short-reduced-motion");
   await page.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
   await page.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
   assert.equal(await page.eval("document.activeElement===document.querySelector('.app-state-loader-retry')"), true, "retry is keyboard reachable");
   await page.eval("window.__retryClicks=0;document.querySelector('.app-state-loader-retry').addEventListener('click',()=>window.__retryClicks++)");
   const readsBeforeRetry = mobileError.requests.filter(r => r.method === "GET").length; mobileError.mode = "valid";
   await page.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", text: "\r", unmodifiedText: "\r", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 });
   await page.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 });
   const afterKeyboard = await page.eval("({clicks:window.__retryClicks,state:document.querySelector('.app-shell')?.dataset.uiState,uiCalls:window.__uiCalls})");
   ledger.accessibilityChecks.push({ viewport: "390x430", errorRole: errorLayout.role, errorLayout, longMessageWrapped: true, noHorizontalOverflow: true, retryTarget: errorLayout.button, keyboardReachable: true, afterKeyboard });
   await page.wait("window.__retryClicks===1");
   await page.wait("document.querySelector('.app-shell[data-ui-state=loaded]')");
   assert.ok(mobileError.requests.filter(r => r.method === "GET").length > readsBeforeRetry, "Enter triggers the existing explicit retry read");
   ledger.accessibilityChecks[ledger.accessibilityChecks.length - 1] = { ...ledger.accessibilityChecks.at(-1), keyboardRetryReads: true, reducedMotionOrbits: errorLayout.orbits, zeroWritesBeforeRetry: true };
   await page.send("Emulation.setEmulatedMedia", { features: [] }); await page.send("Emulation.clearDeviceMetricsOverride");
   const failures = [["500-v5", "http500", 5], ["500-v0", "http500", 0], ["bad-json-v5", "badjson", 5], ["missing-state", 'shape:{}', 5], ["state-array", 'shape:{"state":[]}', 5], ["state-null", 'shape:{"state":null}', 5], ["api-array", 'shape:[]', 5], ["wrong-shape", 'shape:{"state":{"workspace":[]}}', 5], ["network-refused", "network", 5], ["network-timeout", "timeout", 5], ["disk-badJSON", "valid", 5, '{"broken":'], ["disk-null", "valid", 5, 'null'], ["disk-array", "valid", 5, '[]'], ["disk-EACCES-mock", "eacces", 5], ["future-engine-map", "valid", 5, { workspace: { engineSettingsByEngine: { schemaVersion: 99, byEngine: {} } } }], ["bad-engine-record", "valid", 5, { workspace: { engineSettingsByEngine: { schemaVersion: 3, byEngine: { Anima: [] } } } }]];
  for (const [label, mode, revision, value] of failures) {
     const c = await configure(label, mode, revision, value ?? uiStateFixture()); await page.wait("document.querySelector('[data-ui-state=error]')");
     assert.equal(await page.eval("(()=>{const b=document.querySelector('[data-ui-state=error] button');b.scrollIntoView();const r=b.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight&&b.disabled===false})()"),true,'read retry remains reachable in short browser viewport');
    await page.eval("window.dispatchEvent(new Event('pagehide'));window.dispatchEvent(new Event('focus'));document.body.click();0"); await sleep(750);
    const snapshotStable = await page.eval("localStorage.getItem('xirai-assistant-prompt-snapshot-v1')===window.__localPromptBefore&&window.__localPromptWrites.length===0");
    assert.equal(snapshotStable, true); assert.equal(c.requests.filter(r => r.method === "PUT").length, 0);
    assert.equal(await page.eval("document.querySelectorAll('textarea,.generation-mode-switch').length"), 0);
     const { check } = await record(c, { zeroUiWrites: true, localPromptSnapshotUnchanged: true, editingUnavailable: true }); assert.equal(check.beforeSha256, check.afterSha256);
     if (label === 'bad-json-v5') await screenshot('read-error');
  }
  // Repair/retry: same error page must load the real sentinel, not defaults.
  const retry = await configure("repair-retry", "http500"); await page.wait("document.querySelector('[data-ui-state=error]')"); retry.mode = "valid"; await page.click("重试读取");
  await page.wait("document.querySelector('.prompt-field textarea')?.value==='SENTINEL Anima positive'"); await sleep(1100);
  assert.equal(retry.requests.filter(r => r.method === "PUT").length, 1);
  const restored = JSON.parse((await record(retry, { retryLoadsSentinel: true, oneInitialSave: true })).after);
  const original = uiStateFixture();
  for (const engine of Object.keys(original.workspace.engineSettingsByEngine.byEngine)) {
    const a = original.workspace.engineSettingsByEngine.byEngine[engine], b = restored.workspace.engineSettingsByEngine.byEngine[engine];
     assert.equal(b.positive, a.positive); assert.deepEqual(b.size, a.size); assert.equal(b.seed, a.seed); assert.equal(b.imageToImage.positive, a.imageToImage.positive);
     assert.equal(b.negative, a.negative); assert.equal(b.imageToImage.negative, a.imageToImage.negative);
     assert.equal(b.imageToImage.width, a.imageToImage.width); assert.equal(b.imageToImage.height, a.imageToImage.height);
     assert.deepEqual(restored.workspace.mountedLorasByEngine.byEngine[engine], original.workspace.mountedLorasByEngine.byEngine[engine]);
    assert.deepEqual(restored.workspace.loraGroupsByEngine.byEngine[engine], original.workspace.loraGroupsByEngine.byEngine[engine]);
  }
   assert.deepEqual(restored.workspace.xyzByEngine, original.workspace.xyzByEngine); assert.deepEqual(restored.workspace.loraMixByEngine, original.workspace.loraMixByEngine);
   await screenshot('rich-retry-loaded');
   await page.fill("NEW LOCAL INPUT"); await sleep(700); assert.equal(JSON.parse(await readFile(path.join(retry.dir, "ui-state.json"))).workspace.positive, "NEW LOCAL INPUT");
   await page.send('Page.reload'); await page.wait("document.querySelector('.prompt-field textarea')?.value==='NEW LOCAL INPUT'"); await sleep(700);
   assert.deepEqual(await page.eval('window.__manualOpenCalls'), []);
   await page.eval("document.querySelector('.lora-title button').click();0");
   await page.wait("document.querySelector('[aria-label=\"在新标签页打开 LoRA 管理器\"]')");
   await page.eval("document.querySelector('[aria-label=\"在新标签页打开 LoRA 管理器\"]').click();0");
   assert.deepEqual(await page.eval('window.__manualOpenCalls'), [['/lora', '_blank', 'noopener,noreferrer']]);
   await record(retry, { label: 'rich-reload-manual-lora', reloadPreservedInput: true, manualLoraOnly: true });
   for (const fault of [409, 500, "network", "badack"]) {
    const c = await configure("put-fault-" + fault, "valid"); await page.wait("document.querySelector('.prompt-field textarea')?.value==='SENTINEL Anima positive'"); await sleep(1000);
     c.before = await readFile(path.join(c.dir, "ui-state.json")); c.putFault = fault; c.requests.length = 0; await page.eval('window.__uiCalls=[];0');
    await page.fill("UNSAVED INPUT " + fault); await page.wait("document.querySelector('[data-ui-save-error]')");
    await page.eval("window.dispatchEvent(new Event('pagehide'));0"); await sleep(700);
     const appPutCalls = await page.eval("window.__uiCalls.filter(r=>r.method==='PUT').length");
     assert.equal(appPutCalls, 1, "no automatic retry loop or pagehide overwrite after failure (Chromium may retry a disconnected HTTP request internally)");
    assert.equal(await page.eval("document.querySelector('.prompt-field textarea').value"), "UNSAVED INPUT " + fault);
      const { check } = await record(c, { visibleSaveError: true, inputPreserved: true, appPutCalls }); assert.equal(check.beforeSha256, check.afterSha256);
     if (fault === 409) await screenshot('save-conflict');
     // Retry cannot PUT if a fresh GET is unreadable; live text is not hydrated.
     c.mode = 'badjson'; const puts = c.requests.filter(r => r.method === 'PUT').length;
     await page.click('重试保存'); await sleep(450); assert.equal(c.requests.filter(r => r.method === 'PUT').length, puts);
     assert.equal(await page.eval("document.querySelector('.prompt-field textarea').value"), "UNSAVED INPUT " + fault);
     c.mode = 'valid';
     c.putFault = null; await page.click("重试保存"); await page.wait("!document.querySelector('[data-ui-save-error]')"); await sleep(200);
     assert.equal(JSON.parse(await readFile(path.join(c.dir, "ui-state.json"))).workspace.positive, "UNSAVED INPUT " + fault);
     await record(c, { label: c.label + '-validated-retry', verifiedRetrySavesInput: true });
  }
  // Confirmed absence from real ENOENT initializes; legal empty JSON and flat legacy restore.
   const absent = await configure("confirmed-absent", "valid", 5, null);
   await page.wait("document.querySelector('[data-ui-state=confirmed-absent]')"); await sleep(1000); assert.ok(absent.requests.some(r => r.method === "PUT" && r.status === 200)); await record(absent, { confirmedAbsentInitializes: true });
  for (const [label, value] of [["legal-empty", {}], ["legacy-flat", { workspace: { model: "Anima", positive: "LEGACY SENTINEL", negative: "LEGACY NEGATIVE", size: { width: 896, height: 1280 } } }]]) {
    const c = await configure(label, "valid", 0, value); await page.wait("document.querySelector('[data-ui-state=loaded]')"); await sleep(1000); assert.ok(c.requests.some(r => r.method === "PUT" && r.status === 200));
     if (label === "legacy-flat") assert.equal(await page.eval("document.querySelector('.prompt-field textarea').value"), "LEGACY SENTINEL"); await record(c, { legalInitialization: true });
   }
   for (const [key, value] of [['promptPresets', { schemaVersion: 99, raw: 'keep' }], ['mountedLorasByEngine', { schemaVersion: 99, raw: 'keep' }], ['loraGroupsByEngine', { schemaVersion: 99, raw: 'keep' }], ['xyzByEngine', { schema_version: 99, raw: 'keep' }], ['loraMixByEngine', { schema_version: 99, raw: 'keep' }]]) {
     const valueState = uiStateFixture(); valueState.workspace[key] = value;
     const c = await configure('fatal-raw-' + key, 'valid', 5, valueState); await page.wait("document.querySelector('[data-ui-state=loaded]')");
     await page.eval("window.dispatchEvent(new Event('pagehide'));0"); await sleep(700);
     assert.equal(c.requests.filter(r => r.method === 'PUT').length, 0);
     const { check } = await record(c, { fatalRawNoSave: true }); assert.equal(check.beforeSha256, check.afterSha256);
   }
   const late = await configure('late-old-get', 'late'); late.lateResult = 'failed'; await page.wait("document.querySelector('[data-ui-state=error]')");
   late.mode = 'valid'; await page.click('重试读取'); await page.wait("document.querySelector('.prompt-field textarea')?.value==='SENTINEL Anima positive'");
   await page.fill('LATEST AFTER RETRY'); for (const release of late.releases || []) release(); await sleep(800);
   assert.equal(await page.eval("document.querySelector('.prompt-field textarea').value"), 'LATEST AFTER RETRY'); await record(late, { oldGetIgnored: true });
   const racing = await configure('save-late-input', 'valid'); await page.wait("document.querySelector('.prompt-field textarea')?.value==='SENTINEL Anima positive'"); await sleep(900);
   racing.holdPut = true; await page.fill('FIRST INPUT'); for (let i=0;i<30&&!racing.releasePut;i++) await sleep(100); assert.ok(racing.releasePut);
   await page.fill('LATEST DURING SAVE'); await sleep(500); racing.releasePut(); await sleep(900);
   assert.equal(JSON.parse(await readFile(path.join(racing.dir, 'ui-state.json'))).workspace.positive, 'LATEST DURING SAVE'); await record(racing, { inFlightInputPreserved: true });
  ledger.browserErrors = page.errors; assert.deepEqual(page.errors, []);
   console.log(JSON.stringify({ evidence: EVIDENCE, checks: ledger.checks.length, visualChecks: ledger.visualChecks.length, accessibilityChecks: ledger.accessibilityChecks.length }));
} finally {
  await writeFile(path.join(EVIDENCE, "browser-ledger.json"), JSON.stringify(ledger, null, 2));
  try { if (page) await page.send("Browser.close"); } catch {} if (ws) ws.close(); if (edge && edge.exitCode == null) edge.kill(); await vite.close();
}
