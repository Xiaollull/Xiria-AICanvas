// Gallery import/export, end to end in a headless browser: the real App and dialog, the real Node admission,
// and the real Python gallery routes on private state/output roots (scripts/gallery-transfer-service.py).
// No production state, output, model or GPU is touched. Screenshots land in the owned evidence folder.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import net from "node:net";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { galleryTransferAdmission, isGalleryTransferRequest } from "./gallery-transfer-access.mjs";

const ROOT = path.resolve(import.meta.dirname, ".."), BASE = process.env.GALLERY_TRANSFER_EVIDENCE;
assert.ok(BASE && path.resolve(BASE).startsWith(path.join(process.env.LOCALAPPDATA, "Temp", "opencode") + path.sep), "GALLERY_TRANSFER_EVIDENCE must be an owned folder under %LOCALAPPDATA%\\Temp\\opencode");
const E = path.join(BASE, `browser-${Date.now()}-${process.pid}`);
const DOWNLOADS = path.join(E, "downloads");
await mkdir(path.join(E, "screenshots"), { recursive: true });
await mkdir(DOWNLOADS, { recursive: true });
await writeFile(path.join(E, "OWNED-FIXTURE.json"), JSON.stringify({ pid: process.pid, root: E, noGPU: true, actualApp: true, actualHTTP: true }));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function port() { const server = net.createServer(); await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve)); const value = server.address().port; await new Promise((resolve) => server.close(resolve)); return value; }
const backendPort = await port(), debugPort = await port(), target = `http://127.0.0.1:${backendPort}`;
const checks = [];
let python, edge, page, proof;

async function startPython() {
  python = spawn(path.join(ROOT, ".venv", "Scripts", "python.exe"), [path.join(ROOT, "scripts", "gallery-transfer-service.py"), "--root", E, "--port", String(backendPort)], { cwd: ROOT, env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONPATH: ROOT }, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  let output = ""; python.stdout.on("data", (data) => output += data); python.stderr.on("data", (data) => output += data);
  for (let i = 0; i < 400; i++) {
    if (python.exitCode !== null) throw new Error(`Fixture exited: ${output}`);
    try { const response = await fetch(`${target}/fixture-info`); if (response.ok) return response.json(); } catch {}
    await sleep(150);
  }
  throw new Error(`Fixture readiness timed out: ${output}`);
}
async function stopChild(child) { if (!child || child.exitCode !== null) return; await new Promise((resolve) => { child.once("exit", resolve); child.kill(); }); }

class Page {
  constructor(ws) { this.ws = ws; this.next = 0; this.pending = new Map(); this.errors = []; ws.onmessage = (event) => { const message = JSON.parse(event.data); if (message.id) { const pending = this.pending.get(message.id); if (pending) { this.pending.delete(message.id); message.error ? pending.reject(new Error(message.error.message)) : pending.resolve(message.result); } } else if (message.method === "Runtime.exceptionThrown") this.errors.push(message.params.exceptionDetails?.exception?.description || message.params.exceptionDetails?.text); }; }
  send(method, params = {}) { const id = ++this.next; this.ws.send(JSON.stringify({ id, method, params })); return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject })); }
  async eval(expression) { const result = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }); if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text); return result.result.value; }
  async wait(expression, tries = 200) { for (let i = 0; i < tries; i++) { if (await this.eval(`Boolean(${expression})`)) return; await sleep(80); } throw new Error(`Timeout: ${expression}`); }
  async click(expression) {
    const point = await this.eval(`(()=>{const e=${expression};if(!e)throw new Error('Missing element');e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2,h=document.elementFromPoint(x,y);if(!h||!(e.contains(h)||h.contains(e)))throw new Error('Covered '+e.className+' by '+h?.className);return{x,y}})()`);
    for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) await this.send("Input.dispatchMouseEvent", { type, ...point, button: "left", clickCount: 1 });
    await sleep(160);
  }
  async type(selector, text) { await this.eval(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.focus();e.select();})()`); await this.send("Input.insertText", { text }); await sleep(120); }
  async key(key) { for (const type of ["keyDown", "keyUp"]) await this.send("Input.dispatchKeyEvent", { type, key, code: key }); await sleep(160); }
  async viewport(width, height) { await this.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 500 }); await sleep(260); }
  async shot(name) { const { data } = await this.send("Page.captureScreenshot", { format: "png" }); await writeFile(path.join(E, "screenshots", `${name}.png`), Buffer.from(data, "base64")); }
}
const byText = (selector, text) => `([...document.querySelectorAll(${JSON.stringify(selector)})].find(e=>e.textContent.includes(${JSON.stringify(text)})))`;
const dialog = "document.querySelector('.gallery-transfer-dialog')";
const receipt = "document.querySelector('.gallery-transfer-receipt')";
const fits = `(()=>{const d=document.querySelector('.gallery-transfer-dialog').getBoundingClientRect();return document.documentElement.scrollWidth<=innerWidth&&d.left>=0&&d.right<=innerWidth+0.5&&d.top>=0&&d.bottom<=innerHeight+0.5})()`;
async function waitForDownload(name) {
  for (let i = 0; i < 200; i++) {
    const files = await readdir(DOWNLOADS);
    if (files.includes(name) && !files.some((file) => file.endsWith(".crdownload"))) return path.join(DOWNLOADS, name);
    await sleep(100);
  }
  throw new Error(`Download did not arrive: ${name}`);
}

const vite = await createServer({ configFile: false, root: ROOT, cacheDir: path.join(E, "vite-cache"), appType: "mpa", plugins: [react(), { name: "private-gallery-http", configureServer(server) { server.middlewares.use(async (request, response, next) => {
  if (!request.url.startsWith("/api/")) return next();
  if (!request.url.startsWith("/api/inference/gallery")) { response.statusCode = 403; response.end("Closed private fixture"); return; }
  // The same admission the real Vite gate applies, then the proof only a transfer write carries.
  const admission = galleryTransferAdmission(request);
  if (!admission.allowed) { response.statusCode = admission.status; response.setHeader("Content-Type", "application/json"); response.end(JSON.stringify({ detail: admission.local ? "按路径导入只能在本机页面使用" : "画廊导入导出须从本服务的同源页面发起" })); return; }
  const chunks = []; for await (const chunk of request) chunks.push(chunk); const body = Buffer.concat(chunks);
  const headers = {};
  for (const key of ["content-type", "origin"]) if (request.headers[key]) headers[key] = request.headers[key];
  if (isGalleryTransferRequest(request)) headers["x-xirai-wd14-proxy-proof"] = proof;
  try {
    const answer = await fetch(target + request.url, { method: request.method, headers, ...(body.length ? { body } : {}) });
    response.statusCode = answer.status;
    for (const key of ["content-type", "content-disposition", "cache-control"]) if (answer.headers.get(key)) response.setHeader(key, answer.headers.get(key));
    response.end(Buffer.from(await answer.arrayBuffer()));
  } catch (error) { response.statusCode = 503; response.end(JSON.stringify({ detail: String(error) })); }
}); } }], logLevel: "error", server: { host: "127.0.0.1", port: 0, hmr: false, open: false, watch: null }, optimizeDeps: { noDiscovery: true, include: ["react", "react-dom/client", "lucide-react"] } });

try {
  const info = await startPython(); proof = info.proof;
  await vite.listen(); const origin = `http://127.0.0.1:${vite.httpServer.address().port}`;
  edge = spawn("C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", ["--headless=new", `--remote-debugging-port=${debugPort}`, `--user-data-dir=${path.join(E, "edge-profile")}`, "--no-first-run", "--disable-extensions", "--disable-sync", "--disable-background-networking", "about:blank"], { stdio: "ignore", windowsHide: true });
  for (let i = 0; i < 100; i++) { try { if ((await fetch(`http://127.0.0.1:${debugPort}/json/version`)).ok) break; } catch {} await sleep(100); }
  const version = await (await fetch(`http://127.0.0.1:${debugPort}/json/version`)).json();
  const browser = new Page(new WebSocket(version.webSocketDebuggerUrl)); await new Promise((resolve, reject) => { browser.ws.onopen = resolve; browser.ws.onerror = reject; });
  await browser.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: DOWNLOADS, eventsEnabled: true });
  const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json(), ws = new WebSocket(targets.find((item) => item.type === "page").webSocketDebuggerUrl);
  page = new Page(ws); await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; }); await page.send("Page.enable"); await page.send("Runtime.enable"); await page.send("DOM.enable");
  await page.viewport(1440, 900); await page.send("Page.navigate", { url: `${origin}/scripts/gallery-transfer.fixture.html` });
  await page.wait("document.querySelector('.generation-mode-switch')&&!document.querySelector('.app-loader')");
  await page.click(byText(".app-nav button, nav button", "画廊"));
  await page.wait("document.querySelector('.gallery-room-actions .gallery-transfer-entry')&&document.querySelectorAll('.gallery-room-library button').length===12");
  await sleep(700); await page.shot("01-gallery-toolbar-1440");
  const toolbar = await page.eval("(()=>{const a=document.querySelector('.gallery-room-actions'),b=[...a.children].map(e=>e.getBoundingClientRect());return{rows:new Set(b.map(r=>Math.round(r.top))).size,right:Math.max(...b.map(r=>r.right)),width:innerWidth}})()");
  assert.equal(toolbar.rows, 1, "the transfer button sits on the toolbar row"); checks.push("toolbar: 导入 / 导出 on one row at 1440");
  for (const [width, height] of [[1024, 768], [390, 844]]) {
    await page.viewport(width, height); await page.eval("document.querySelector('.gallery-room-tools').scrollIntoView({block:'center'})"); await sleep(300);
    assert.ok(await page.eval("document.querySelector('.gallery-transfer-entry').getBoundingClientRect().width>0&&document.documentElement.scrollWidth<=innerWidth"), `toolbar at ${width}`);
    await page.shot(`01-gallery-toolbar-${width}`);
  }
  await page.viewport(1440, 900); await page.eval("document.querySelector('.gallery-room').scrollTo(0,0)");

  // Export: choose two collections, download the ZIP, and find the folder under the private output root.
  await page.click("document.querySelector('.gallery-transfer-entry')"); await page.wait(dialog);
  assert.equal(await page.eval("document.querySelectorAll('.gallery-transfer-list li').length"), 11);
  assert.equal(await page.eval("document.querySelector('.gallery-transfer-dialog footer button.primary').disabled"), true, "nothing selected, nothing to export");
  await page.shot("02-export-empty-1440"); assert.ok(await page.eval(fits));
  await page.click(byText(".gallery-transfer-list label", "人像 portraits")); await page.click(byText(".gallery-transfer-list label", "landscapes"));
  assert.equal(await page.eval("document.querySelector('.gallery-transfer-dialog footer span').textContent"), "已选 2 个 · 3 卡片 · 3 图");
  await page.click(byText(".gallery-transfer-list label", "一个名字很长")); await page.shot("03-export-selected-1440");
  await page.click(byText(".gallery-transfer-list label", "一个名字很长"));
  await page.click("document.querySelector('.gallery-transfer-dialog footer button.primary')"); await page.wait(receipt);
  const exported = await page.eval("document.querySelector('.gallery-transfer-receipt code').textContent");
  const folder = exported.split("/").at(-1); assert.match(folder, /^XiriusA-Gallery-\d{8}-\d{6}$/);
  const zipPath = await waitForDownload(`${folder}.zip`);
  const info2 = JSON.parse(await readFile(path.join(info.outputs, folder, "gallery-info.json"), "utf8"));
  assert.deepEqual(info2.collections.map((item) => item.id), ["人像 portraits", "landscapes"]);
  assert.ok((await stat(zipPath)).size > 1000); await page.shot("04-export-receipt-1440");
  checks.push(`export: folder ${folder} under the private output root, browser download ${path.basename(zipPath)}`);

  // Import by typed path (quoted, as Windows "Copy as path" gives it) into a named folder.
  await page.click(byText(".gallery-transfer-tabs button", "导入")); await page.wait("document.querySelector('.gallery-transfer-drop')");
  await page.shot("05-import-empty-1440");
  await page.type(".gallery-transfer-body .gallery-field input", "D:\\Exports\\gallery.png");
  await page.wait("document.querySelector('.gallery-transfer-hint')?.textContent.includes('.zip')");
  assert.equal(await page.eval("document.querySelector('.gallery-transfer-dialog footer button.primary').disabled"), true); await page.shot("06-import-bad-path-1440");
  await page.type(".gallery-transfer-body .gallery-field input", `"${zipPath}"`);
  await page.type(".gallery-transfer-body .gallery-field:last-of-type input", "browser-import");
  await page.click("document.querySelector('.gallery-transfer-dialog footer button.primary')"); await page.wait(receipt);
  assert.ok((await page.eval(`${receipt}.textContent`)).includes("已导入 2 个收藏夹 · 3 张卡片 · 3 张图片"));
  await page.wait("[...document.querySelectorAll('.gallery-room-library button')].some(e=>e.textContent.includes('人像 portraits (2)'))");
  await stat(path.join(info.outputs, "browser-import", "gallery-info.json"));
  await stat(path.join(info.outputs, "browser-import", "人像 portraits", "001-02.jpg"));
  await page.shot("07-import-path-receipt-1440"); checks.push("import by quoted path: collections renamed '(2)', images under outputs/browser-import, gallery refreshed");

  // Import by choosing the file (click path of the drop box), with a blank folder: a time-named folder.
  await page.click(byText(".gallery-transfer-dialog footer button", "继续导入")); await page.wait("document.querySelector('.gallery-transfer-drop')");
  const { root } = await page.send("DOM.getDocument"); const { nodeId } = await page.send("DOM.querySelector", { nodeId: root.nodeId, selector: ".gallery-transfer-dialog input[type=file]" });
  await page.send("DOM.setFileInputFiles", { nodeId, files: [zipPath] });
  await page.wait("document.querySelector('.gallery-transfer-drop.has-file')");
  assert.equal(await page.eval("document.querySelector('.gallery-transfer-body .gallery-field input').disabled"), true, "a chosen file replaces the typed path");
  await page.shot("08-import-file-chosen-1440");
  await page.click("document.querySelector('.gallery-transfer-dialog footer button.primary')"); await page.wait(receipt);
  const timed = (await page.eval("document.querySelector('.gallery-transfer-receipt code').textContent")).split("/").at(-1);
  assert.match(timed, /^\d{8}-\d{6}$/); await stat(path.join(info.outputs, timed, "landscapes", "001-01.webp"));
  assert.ok((await page.eval(`${receipt}.textContent`)).includes("人像 portraits (3)")); checks.push(`import by chosen file: blank folder became outputs/${timed}`);

  // Drag and drop: a non-ZIP is refused in the page; a ZIP becomes the chosen file.
  await page.click(byText(".gallery-transfer-dialog footer button", "继续导入")); await page.wait("document.querySelector('.gallery-transfer-drop')");
  const drop = (name, type) => page.eval(`(async()=>{const box=document.querySelector('.gallery-transfer-drop');const bytes=await (await fetch(${JSON.stringify(`/api/inference/gallery/transfer/exports/${folder}.zip`)})).arrayBuffer();const data=new DataTransfer();data.items.add(new File([bytes],${JSON.stringify(name)},{type:${JSON.stringify(type)}}));for(const kind of ['dragenter','dragover','drop'])box.dispatchEvent(new DragEvent(kind,{bubbles:true,cancelable:true,dataTransfer:data}));return true})()`);
  await drop("notes.png", "image/png"); await page.wait("document.querySelector('.gallery-transfer-body .gallery-form-error')?.textContent.includes('不是 .zip')");
  await page.shot("09-import-drop-refused-1440");
  await drop("dragged.zip", "application/zip"); await page.wait("document.querySelector('.gallery-transfer-drop.has-file')?.textContent.includes('dragged.zip')");
  assert.equal(await page.eval("Boolean(document.querySelector('.gallery-transfer-body .gallery-form-error'))"), false); checks.push("drop: non-ZIP refused in page, ZIP accepted");
  await page.type(".gallery-transfer-body .gallery-field:last-of-type input", "browser-import");
  await page.click("document.querySelector('.gallery-transfer-dialog footer button.primary')");
  await page.wait("document.querySelector('.gallery-transfer-body .gallery-form-error')?.textContent.includes('已存在且不为空')");
  await page.shot("10-import-conflict-1440"); checks.push("import into a non-empty folder is refused with the server's message");

  // Phone width: both tabs fit, no horizontal scroll.
  await page.viewport(390, 844); assert.ok(await page.eval(fits)); await page.shot("11-import-390");
  await page.click(byText(".gallery-transfer-tabs button", "导出")); await page.wait("document.querySelector('.gallery-transfer-list')"); assert.ok(await page.eval(fits)); await page.shot("12-export-390");
  await page.viewport(1440, 900);
  await page.key("Escape"); await page.wait(`!${dialog}`); checks.push("390 px: both tabs fit; Escape closes");
  assert.deepEqual(page.errors, [], "no page exceptions");
  await writeFile(path.join(E, "ledger.json"), JSON.stringify({ scope: "Executor self-test: actual App + actual private Python gallery routes, synthetic collections, no GPU", checks }, null, 2));
  console.log(JSON.stringify({ evidence: E, checks }, null, 2));
} finally {
  await stopChild(edge); await stopChild(python); await vite.close();
}
