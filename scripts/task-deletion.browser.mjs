import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import net from "node:net";
import { createHash } from "node:crypto";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { isTaskDeletionRequest, taskDeletionAdmission } from "./task-deletion-access.mjs";

const ROOT = path.resolve(import.meta.dirname, ".."), BASE = process.env.TASK_DELETION_EVIDENCE;
assert.ok(BASE && path.resolve(BASE).startsWith(path.join(process.env.LOCALAPPDATA, "Temp", "opencode") + path.sep));
const E = path.join(BASE, `browser-${Date.now()}-${process.pid}`);
await mkdir(path.join(E, "screenshots"), { recursive: true });
await writeFile(path.join(E, "OWNED-FIXTURE.json"), JSON.stringify({ pid: process.pid, root: E, noGPU: true, actualApp: true, actualHTTP: true }));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function port() { const s = net.createServer(); await new Promise((r) => s.listen(0, "127.0.0.1", r)); const p = s.address().port; await new Promise((r) => s.close(r)); return p; }
const backendPort = await port(), debugPort = await port(), target = `http://127.0.0.1:${backendPort}`;
let python, edge, page, holdId = null, releaseHeld = null, heldId = null, holdDelete = false, deleteHeld = false, loseDelete = false;
const requests = [], ledger = { scope: "Executor SELF-TEST: actual App + actual private Python HTTP, synthetic tasks, no GPU", checks: [] };
async function startPython() {
  python = spawn(process.env.P10_PYTHON || path.join(ROOT, ".venv", "Scripts", "python.exe"), ["-m", "scripts.task-deletion-service", "--root", E, "--port", String(backendPort)], { cwd: ROOT, env: { ...process.env, PYTHONUNBUFFERED: "1" }, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  let output = ""; python.stdout.on("data", (d) => output += d); python.stderr.on("data", (d) => output += d);
  for (let i = 0; i < 180; i++) { if (python.exitCode !== null) throw new Error(`Fixture exited: ${output}`); try { const r = await fetch(target + "/fixture-info"); if (r.ok) { await writeFile(path.join(E, `python-${python.pid}.log`), output); return r.json(); } } catch {} await sleep(150); }
  throw new Error(`Fixture readiness timed out: ${output}`);
}
async function stopPython() { if (!python || python.exitCode !== null) return; const child = python; await new Promise((resolve) => { child.once("exit", resolve); child.kill(); }); }
class Page {
  constructor(ws) { this.ws = ws; this.next = 0; this.pending = new Map(); this.errors = []; ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id) { const p = this.pending.get(m.id); if (p) { this.pending.delete(m.id); m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result); } } else if (m.method === "Runtime.exceptionThrown") this.errors.push(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text); }; }
  send(method, params = {}) { const id = ++this.next; this.ws.send(JSON.stringify({ id, method, params })); return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject })); }
  async eval(expression) { const r = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result.value; }
  async wait(expression) { for (let i = 0; i < 150; i++) { if (await this.eval(expression)) return; await sleep(80); } throw new Error(`Timeout: ${expression}`); }
  async click(expression) { const p = await this.eval(`(()=>{const e=${expression};if(!e)throw new Error('Missing element');e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2,h=document.elementFromPoint(x,y);if(!h||!e.contains(h))throw new Error('Covered '+e.className+' by '+h?.className);return{x,y}})()`); for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) await this.send("Input.dispatchMouseEvent", { type, ...p, button: "left", clickCount: 1 }); await sleep(160); }
  async viewport(width, height) { await this.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 500 }); await sleep(220); }
  async shot(name) { const { data } = await this.send("Page.captureScreenshot", { format: "png" }); await writeFile(path.join(E, "screenshots", name + ".png"), Buffer.from(data, "base64")); }
  async choose(label, text) { await this.click(`document.querySelector('button[aria-label=${JSON.stringify(label)}]')`); await this.click(`([...document.querySelectorAll('[role=option]')].find(e=>e.textContent.includes(${JSON.stringify(text)})&&!e.disabled))`); }
}
const byText = (selector, text) => `([...document.querySelectorAll(${JSON.stringify(selector)})].find(e=>e.textContent.includes(${JSON.stringify(text)})))`;
const taskButton = (text) => byText(".task-delete-manager button", text);
// The task list lives in a dialog opened from the results header; the modal blocks the page behind it.
const managerOpen = "Boolean(document.querySelector('.task-delete-manager'))";
async function openManager() { if (!(await page.eval(managerOpen))) { await page.click("document.querySelector('.task-delete-entry')"); await page.wait(managerOpen); } }
async function closeManager() { if (await page.eval(managerOpen)) { await page.wait("!document.querySelector('.task-delete-close')?.disabled"); await page.click("document.querySelector('.task-delete-close')"); await page.wait(`!${managerOpen}`); } }
const deleteCalls = () => requests.filter((item) => /\/task-deletions$/.test(item.url) && item.method === "POST").length;
const vite = await createServer({ configFile: false, root: ROOT, cacheDir: path.join(E, "vite-cache"), appType: "mpa", plugins: [react(), { name: "private-task-http", configureServer(v) { v.middlewares.use(async (req, res, next) => {
  if (!req.url.startsWith("/api/")) return next();
  if (!/^\/api\/inference\/(xyz-plots|lora-mix)(?:[/?]|$)/.test(req.url)) { res.statusCode = 403; res.end("Closed private fixture"); return; }
  if (req.method === "POST" && !isTaskDeletionRequest(req)) { res.statusCode = 403; res.end("GPU/work admission forbidden in fixture"); return; }
  const admission = taskDeletionAdmission(req);
  if (!admission.allowed) { res.statusCode = admission.status; res.end(JSON.stringify({ detail: "Trusted-local request required" })); return; }
  const chunks = []; for await (const chunk of req) chunks.push(chunk); const body = Buffer.concat(chunks);
  requests.push({ url: req.url, method: req.method, body: body.length ? JSON.parse(body.toString()) : null });
  try {
    const response = await fetch(target + req.url, { method: req.method, headers: { "content-type": "application/json", ...(req.headers.origin ? { origin: req.headers.origin } : {}), "x-xirai-wd14-proxy-proof": "task-deletion-private-proof" }, ...(body.length ? { body } : {}) });
    const bytes = Buffer.from(await response.arrayBuffer());
    if (holdId && req.method === "GET" && req.url.endsWith("/" + holdId)) { heldId = holdId; holdId = null; await new Promise((r) => releaseHeld = r); releaseHeld = null; }
    if (holdDelete && req.method === "POST" && /\/task-deletions$/.test(req.url)) { holdDelete = false; deleteHeld = true; await new Promise((r) => releaseHeld = r); releaseHeld = null; }
    if (loseDelete && req.method === "POST" && /\/task-deletions$/.test(req.url)) { loseDelete = false; res.statusCode = 503; res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify({ detail: "Private fixture: committed response lost" })); return; }
    res.statusCode = response.status; res.setHeader("Content-Type", response.headers.get("content-type") || "application/json"); res.end(bytes);
  } catch (error) { res.statusCode = 503; res.end(JSON.stringify({ detail: String(error) })); }
}); } }], logLevel: "error", css: { postcss: { plugins: [] } }, server: { host: "127.0.0.1", port: 0, hmr: false, ws: false, open: false, watch: null }, optimizeDeps: { noDiscovery: true, include: ["react", "react-dom/client", "lucide-react"] } });
try {
  const info = await startPython(); await vite.listen(); const origin = `http://127.0.0.1:${vite.httpServer.address().port}`;
  const protectedPaths = [path.join(info.outputs, "preserved-original.png"), path.join(info.outputs, "preserved-grid.png"), path.join(info.state, "preserved-gallery.json"), path.join(info.state, "lora-library", "registry.json")];
  const before = await Promise.all(protectedPaths.map(async (file) => [file, createHash("sha256").update(await readFile(file)).digest("hex")]));
  edge = spawn("C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", ["--headless=new", `--remote-debugging-port=${debugPort}`, `--user-data-dir=${path.join(E, "edge-profile")}`, "--no-first-run", "--disable-extensions", "--disable-sync", "--disable-background-networking", "about:blank"], { stdio: "ignore", windowsHide: true });
  for (let i = 0; i < 100; i++) { try { if ((await fetch(`http://127.0.0.1:${debugPort}/json/version`)).ok) break; } catch {} await sleep(100); }
  const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json(), ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
  page = new Page(ws); await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; }); await page.send("Page.enable"); await page.send("Runtime.enable");
  await page.viewport(1440, 900); await page.send("Page.navigate", { url: origin + "/scripts/task-deletion.fixture.html" }); await page.wait("Boolean(document.querySelector('.generation-mode-switch'))&&!document.querySelector('.app-loader')");
  await page.choose("生成模式", "XYZ"); await page.wait("Boolean(document.querySelector('.xyz-tabs'))"); await page.click(byText(".xyz-tabs button", "结果")); await page.wait("Boolean(document.querySelector('.task-delete-entry b'))");
  await page.shot("xyz-results-1440"); assert.equal(await page.eval("document.querySelector('.task-delete-entry b').textContent"), "11");
  await openManager(); await page.wait("document.querySelectorAll('.task-delete-list input').length===11");
  const draftGuard = await page.eval("JSON.stringify(window.__fixtureAppState().workspace.xyzByEngine?.by_engine.SD)");
  await page.shot("xyz-list-1440"); await page.viewport(390, 844); assert.ok(await page.eval("document.documentElement.scrollWidth<=innerWidth")); await page.shot("xyz-list-390");
  await page.click("document.querySelector('.task-delete-list input:not(:disabled)')"); await page.click("[...document.querySelectorAll('.task-delete-list input:not(:disabled)')].find(e=>!e.checked)");
  await page.click(taskButton("删除选中")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))");
  assert.ok((await page.eval("document.querySelector('.task-delete-dialog').textContent")).includes("共 2 个任务")); await page.shot("xyz-confirm-390"); await page.viewport(1440, 900); await page.shot("xyz-confirm-1440");
  const count = deleteCalls(); await page.click(byText(".task-delete-dialog button", "取消")); assert.equal(deleteCalls(), count); ledger.checks.push("selected confirmation cancel emits zero delete/write");
  await page.click(taskButton("删除选中")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))"); await page.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape", code: "Escape" }); await page.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Escape", code: "Escape" }); await page.wait("!document.querySelector('.task-delete-dialog')"); assert.equal(deleteCalls(), count);
  const chosen = requests.filter((r) => r.url.endsWith("/task-deletions/preview")).at(-1).body.ids;
  await fetch(target + "/fixture-failure", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: chosen[0] }) });
  await page.click(taskButton("删除选中")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))");
  await page.eval("(()=>{const b=[...document.querySelectorAll('.task-delete-dialog button')].find(e=>e.textContent.includes('确认删除'));b.click();b.click()})()");
  await page.wait("document.querySelector('.task-delete-receipt')?.textContent.includes('失败 1 个')"); assert.equal(deleteCalls(), count + 1); assert.ok(await page.eval("document.querySelector('.task-delete-receipt').textContent.includes('已删除 1 个')")); await page.shot("xyz-partial-retry-1440");
  await page.wait("[...document.querySelectorAll('.task-delete-manager button')].some(e=>e.textContent.includes('重试失败')&&!e.disabled)"); await page.click(taskButton("重试失败")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))"); await page.click(byText(".task-delete-dialog button", "确认删除")); await page.wait("document.querySelector('.task-delete-receipt')?.textContent.includes('失败 0 个')");
  ledger.checks.push("partial failure is itemized/retryable; synchronous double-confirm emits one POST");
  // Hold a real old detail response while deleting that task; its later arrival must not refill the panel.
  await closeManager(); await page.choose("选择 XYZ 实验", "已完成"); await page.wait("Boolean(document.querySelector('.xyz-frozen-summary'))");
  const currentId = await page.eval("document.querySelector('.xyz-frozen-summary small').textContent.match(/[0-9a-f]{32}/)[0]");
  holdId = currentId; await page.click(byText(".xyz-tabs button", "配置")); await page.click(byText(".xyz-tabs button", "结果"));
  for (let i = 0; i < 100 && heldId !== currentId; i++) await sleep(60); assert.equal(heldId, currentId);
  await openManager(); await page.wait("document.querySelectorAll('.task-delete-list input').length===9");
  await page.click(`document.querySelector('input[aria-label="勾选任务 ${currentId.slice(0, 8)}"]')`); await page.click(taskButton("删除选中")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))"); await page.click(byText(".task-delete-dialog button", "确认删除")); await page.wait("Boolean(document.querySelector('.task-delete-receipt'))");
  releaseHeld(); await sleep(500); assert.ok(!(await page.eval("document.querySelector('.xyz-frozen-summary small')?.textContent||''")).includes(currentId)); ledger.checks.push("late deleted detail cannot refill new selection");
  // The committed deletion's response arrives after switching to a different engine/selection.
  await page.wait("[...document.querySelectorAll('.task-delete-manager button')].some(e=>e.textContent.includes('删除全部')&&!e.disabled)");
  await page.click(taskButton("取消选择")); await page.click("document.querySelector('.task-delete-list input:not(:disabled)')"); await page.click(taskButton("删除选中")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))");
  holdDelete = true; await page.click(byText(".task-delete-dialog button", "确认删除")); for (let i = 0; i < 100 && !deleteHeld; i++) await sleep(50); assert.ok(deleteHeld);
  // The modal normally blocks this pointer path; force the engine prop change as a lifecycle negative,
  // not as a claim that a real user can click through the modal while the POST is inflight.
  await page.eval("[...document.querySelectorAll('.model-segment button')].find(e=>e.querySelector('.seg-name')?.textContent==='iL').click()"); await page.wait("document.querySelector('.model-segment .active .seg-name')?.textContent==='iL'&&Boolean(document.querySelector('.xyz-tabs'))");
  await page.click(byText(".xyz-tabs button", "结果")); await openManager(); await page.wait("document.querySelector('.task-delete-manager header')?.textContent.includes('iL')&&document.querySelectorAll('.task-delete-list input').length===10");
  await closeManager(); await page.choose("选择 XYZ 实验", "已完成"); await page.wait("document.querySelector('.xyz-frozen-summary')?.textContent.includes('iL')"); const newId = await page.eval("document.querySelector('.xyz-frozen-summary small').textContent.match(/[0-9a-f]{32}/)[0]");
  releaseHeld(); await sleep(450); assert.ok((await page.eval("document.querySelector('.xyz-frozen-summary small')?.textContent||''")).includes(newId)); await openManager(); await page.wait("document.querySelectorAll('.task-delete-list input').length===10"); await sleep(300); assert.equal(await page.eval("document.querySelectorAll('.task-delete-list input').length"), 10); await closeManager();
  ledger.checks.push("forced lifecycle engine change under inflight modal: actual App keyed remount ignores old committed response and keeps new selection/list (not modal click-through)");
  await page.click("[...document.querySelectorAll('.model-segment button')].find(e=>e.querySelector('.seg-name')?.textContent==='SD')"); await page.wait("document.querySelector('.model-segment .active .seg-name')?.textContent==='SD'&&Boolean(document.querySelector('.xyz-tabs'))"); await page.click(byText(".xyz-tabs button", "结果")); await openManager(); await page.wait("document.querySelector('.task-delete-manager header')?.textContent.includes('SD')"); await page.wait("[...document.querySelectorAll('.task-delete-manager button')].some(e=>e.textContent.includes('删除全部')&&!e.disabled)");
  await closeManager(); await page.choose("选择 XYZ 实验", "已完成"); await page.wait("Boolean(document.querySelector('.xyz-frozen-summary'))"); const lostId = await page.eval("document.querySelector('.xyz-frozen-summary small').textContent.match(/[0-9a-f]{32}/)[0]");
  await openManager(); await page.click(`document.querySelector('input[aria-label="勾选任务 ${lostId.slice(0, 8)}"]')`); await page.click(taskButton("删除选中")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))"); loseDelete = true;
  await page.click(byText(".task-delete-dialog button", "确认删除")); await page.wait("document.querySelector('.task-delete-manager')?.textContent.includes('删除结果未获确认')&&document.querySelectorAll('.task-delete-list input').length===6");
  await page.wait(`!document.querySelector('.xyz-frozen-summary small')?.textContent.includes('${lostId}')`); assert.ok(!await page.eval("Boolean(document.querySelector('.task-delete-receipt'))")); await page.shot("xyz-lost-response-1440");
  ledger.checks.push("committed POST response lost: explicit unknown result, no success receipt, fresh authoritative absence clears stale list/detail/selection");
  await page.wait("[...document.querySelectorAll('.task-delete-manager button')].some(e=>e.textContent.includes('删除全部')&&!e.disabled)");
  const allCount = (await (await fetch(target + "/api/inference/xyz-plots/tasks")).json()).total;
  await page.click(taskButton("删除全部")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))");
  const allText = await page.eval("document.querySelector('.task-delete-dialog').textContent"); assert.ok(allText.includes("所有引擎")); assert.ok(allText.includes(`共 ${allCount} 个任务`)); assert.ok(allCount > 50); assert.ok(allText.includes("保留 1 个"));
  await page.shot("xyz-all-six-engines-1440"); await page.viewport(390, 844); await page.shot("xyz-all-six-engines-390"); await page.click(byText(".task-delete-dialog button", "确认删除")); await page.wait("document.querySelectorAll('.task-delete-list input').length===1");
  assert.equal((await (await fetch(target + "/api/inference/lora-mix/sessions")).json()).sessions.length, 3);
  assert.equal(await page.eval("JSON.stringify(window.__fixtureAppState().workspace.xyzByEngine?.by_engine.SD)"), draftGuard); ledger.checks.push(`all ${allCount} tasks across six engines enumerated on server, ${allCount - 1} deleted/1 paused preserved, mix isolated, XYZ draft unchanged`);
  await page.viewport(1440, 900); await closeManager(); await page.choose("生成模式", "LoRA 混合搜索"); await page.wait("Boolean(document.querySelector('.mix-workspace .xyz-tabs'))"); await page.click(byText(".xyz-tabs button", "结果"));
  await page.wait("Boolean(document.querySelector('.mix-results-head .task-delete-entry b'))&&Boolean(document.querySelector('.mix-round'))"); await page.shot("mix-results-1440");
  await openManager(); await page.wait("document.querySelectorAll('.task-delete-list input').length===3"); await page.wait("Boolean(document.querySelector('.mix-round'))");
  const mixGuard = await page.eval("JSON.stringify(window.__fixtureAppState().workspace.loraMixByEngine)");
  await page.shot("mix-list-1440"); await page.viewport(390, 844); assert.ok(await page.eval("document.documentElement.scrollWidth<=innerWidth")); await page.shot("mix-list-390");
  await page.click(taskButton("全选")); assert.equal(await page.eval("document.querySelectorAll('.task-delete-list input:checked').length"), 2); await page.click(taskButton("取消选择")); assert.equal(await page.eval("document.querySelectorAll('.task-delete-list input:checked').length"), 0);
  await page.click(taskButton("删除全部")); await page.wait("Boolean(document.querySelector('.task-delete-dialog'))"); const text = await page.eval("document.querySelector('.task-delete-dialog').textContent"); assert.ok(text.includes("全部轮次")); assert.ok(text.includes("第 1 轮")); await page.shot("mix-all-confirm-390"); await page.viewport(1440, 900); await page.shot("mix-all-confirm-1440");
  await page.click(byText(".task-delete-dialog button", "确认删除")); await page.wait("document.querySelectorAll('.task-delete-list input').length===1"); assert.equal(await page.eval("JSON.stringify(window.__fixtureAppState().workspace.loraMixByEngine)"), mixGuard);
  const remaining = await (await fetch(target + "/api/inference/lora-mix/sessions")).json(); assert.equal(remaining.sessions.length, 1); assert.equal(remaining.sessions[0].rounds, 2); assert.equal((await (await fetch(target + "/api/inference/xyz-plots?origin=none")).json()).total, 1);
  ledger.checks.push("mix all removes whole 2-round sessions, active session all rounds preserved, other module and draft unchanged");
  await stopPython(); await startPython();
  assert.equal((await (await fetch(target + "/api/inference/xyz-plots?origin=none")).json()).total, 1); assert.equal((await (await fetch(target + "/api/inference/lora-mix/sessions")).json()).sessions.length, 1);
  for (const [file, sha] of before) assert.equal(createHash("sha256").update(await readFile(file)).digest("hex"), sha);
  ledger.checks.push("actual private Python process restart never resurrects deleted tasks; image/grid/Gallery/registry sentinel SHA unchanged");
  const gate = await fetch(origin + "/api/inference/xyz-plots/task-deletions/preview", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ scope: "all" }) }); assert.equal(gate.status, 403);
  assert.deepEqual(page.errors, []); ledger.status = "PASS"; ledger.requests = requests; ledger.protected = before; await writeFile(path.join(E, "BROWSER-LEDGER.json"), JSON.stringify(ledger, null, 2)); console.log(JSON.stringify({ status: ledger.status, checks: ledger.checks.length, evidence: E, screenshots: 14 }));
} catch (error) { ledger.status = "FAIL"; ledger.error = error.stack; ledger.requests = requests; ledger.browserErrors = page?.errors; await writeFile(path.join(E, "BROWSER-LEDGER.json"), JSON.stringify(ledger, null, 2)); throw error; }
finally { releaseHeld?.(); page?.ws.close(); if (edge && edge.exitCode === null) edge.kill(); await vite.close(); await stopPython(); }
