import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";

const ROOT = path.resolve(import.meta.dirname, "..");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const SETUP_MARKER = path.join(os.tmpdir(), "opencode", "xyz-sol-layout1-20261006");
await access(SETUP_MARKER); // Existing setup marker is read-only; this test writes only to its unique child.
const EVIDENCE = path.join(SETUP_MARKER, `runtime-guidance-${process.pid}`);
await mkdir(EVIDENCE, { recursive: false });
const SCREENSHOTS = path.join(EVIDENCE, "screenshots");
await mkdir(SCREENSHOTS, { recursive: false });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const expression = (selector) => `document.querySelector(${JSON.stringify(selector)})`;

class Page {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.errors = []; }
  static async open(port) {
    const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = pages.find((item) => item.type === "page");
    const client = new Page(new WebSocket(page.webSocketDebuggerUrl));
    await new Promise((resolve, reject) => { client.ws.onopen = resolve; client.ws.onerror = reject; });
    await client.send("Page.enable"); await client.send("Runtime.enable");
    return client;
  }
  send(method, params = {}) {
    const id = ++this.id; this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      if (this.id === id) this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.id && this.pending.has(message.id)) {
          const item = this.pending.get(message.id); this.pending.delete(message.id);
          message.error ? item.reject(new Error(message.error.message)) : item.resolve(message.result);
        } else if (message.method === "Runtime.exceptionThrown") this.errors.push(message.params.exceptionDetails?.exception?.description || message.params.exceptionDetails?.text);
      };
    });
  }
  async eval(source) {
    const result = await this.send("Runtime.evaluate", { expression: source, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }
  async waitFor(source, label = source) {
    const start = Date.now();
    while (Date.now() - start < 12000) { try { const value = await this.eval(source); if (value) return value; } catch {} await sleep(100); }
    throw new Error(`Timed out waiting for ${label}`);
  }
  async rect(source) { return this.eval(`(() => {const e=${source}; if(!e)return null; e.scrollIntoView({block:'nearest'}); const r=e.getBoundingClientRect(); return {x:r.left+r.width/2,y:r.top+r.height/2};})()`); }
  async click(source) {
    const p = await this.rect(source); assert.ok(p, `No visible target: ${source}`);
    for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) await this.send("Input.dispatchMouseEvent", { type, x: p.x, y: p.y, button: "left", clickCount: 1 });
    await sleep(120);
  }
  async key(key) {
    const code = { ArrowDown: 40, Escape: 27, Enter: 13 }[key];
    const params = { key, code: key, windowsVirtualKeyCode: code, nativeVirtualKeyCode: code };
    await this.send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...params });
    await this.send("Input.dispatchKeyEvent", { type: "keyUp", ...params });
    await sleep(100);
  }
  async shot(name) {
    const { data } = await this.send("Page.captureScreenshot", { format: "png" });
    await writeFile(path.join(SCREENSHOTS, name), Buffer.from(data, "base64"));
  }
}

const sourceFiles = ["src/App.jsx", "src/XYZPlotWorkspace.jsx", "src/XYZAxisEditor.jsx", "src/xyz-axis-editor.js", "scripts/xyz-ui-redesign.fixture.jsx", "scripts/xyz-runtime-guidance.test.mjs"];
const hashes = [];
for (const relative of sourceFiles) hashes.push(`${createHash("sha256").update(await readFile(path.join(ROOT, relative))).digest("hex")}  ${relative}`);
await writeFile(path.join(EVIDENCE, "source-under-test.sha256"), `${hashes.join("\n")}\n`);

await access(EDGE);
const vite = await createServer({ configFile: false, root: ROOT, cacheDir: path.join(EVIDENCE, "vite-cache"), appType: "mpa", plugins: [react()], logLevel: "error",
  server: { host: "127.0.0.1", port: 0, strictPort: false, hmr: false, open: false, watch: null },
  optimizeDeps: { noDiscovery: true, include: ["react", "react-dom/client", "lucide-react"] },
});
let edge = null, page = null;
const result = [];
const record = (name, pass, detail) => { result.push({ name, pass, detail }); assert.ok(pass, `${name}: ${JSON.stringify(detail)}`); };
try {
  await vite.listen();
  const address = vite.httpServer.address();
  const probe = net.createServer(); await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const cdpPort = probe.address().port; await new Promise((resolve) => probe.close(resolve));
  const profile = path.join(EVIDENCE, "edge-profile"); await mkdir(profile, { recursive: false });
  edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${cdpPort}`, `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check",
    "--disable-extensions", "--disable-sync", "--disable-background-networking", "--window-size=1440,900", "about:blank"], { stdio: "ignore", windowsHide: true });
  let ready = false;
  for (let i = 0; i < 80; i++) { try { if ((await fetch(`http://127.0.0.1:${cdpPort}/json/version`)).ok) { ready = true; break; } } catch {} await sleep(200); }
  assert.ok(ready, "owned Edge CDP did not become ready");
  page = await Page.open(cdpPort);
  await page.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await page.send("Page.navigate", { url: `http://127.0.0.1:${address.port}/scripts/xyz-ui-redesign.fixture.html` });
  await page.waitFor("Boolean(document.querySelector('.xyz-workspace .xyz-run-bar'))", "real XYZ workspace fixture");
  const initialGuidance = await page.eval(`({engine:window.__fixtureInspect().runtimeGuidance.engine,pag:window.__fixtureInspect().runtimeGuidance.modes.pag.status,blocker:Boolean(document.querySelector("[aria-label='运行时引导能力校验']"))})`);
  record("publisher-shaped SD implementation proves PAG available without health.engines.SD and leaves the workspace unblocked", initialGuidance.engine === "SD" && initialGuidance.pag === "available" && !initialGuidance.blocker, initialGuidance);
  await sleep(350); await page.shot("before-runtime-block.png");

  await page.eval(`window.__fixturePatchDraft("SD", {axes:{x:{field:"guidance",values:[{value:"pag",label:"saved PAG mode"}],range:null},y:{field:"pag.scale",values:[{value:"0.8",label:"saved PAG scale"}],range:null},z:null}})`);
  await page.eval(`(() => {const r=window.__fixtureInspect().runtimeGuidance; window.__fixtureSetRuntimeGuidance({...r,modes:{...r.modes,pag:{status:"unavailable",reason:"Fixture health says PAG unavailable"}}});})()`);
  await page.waitFor("Boolean(document.querySelector('[aria-label=\"运行时引导能力校验\"]'))", "runtime unavailable blocking message");

  await page.click(expression('button[aria-label="X guidance"]'));
  const guidanceOptions = await page.eval(`(() => [...document.querySelectorAll('[role="option"]')].map(e=>({text:e.textContent.trim(),disabled:e.disabled,reason:e.title})))()`);
  const disabledPag = guidanceOptions.find((item) => item.text.includes("PAG"));
  record("runtime-false mode is disabled with an explicit reason", Boolean(disabledPag?.disabled && disabledPag.reason.includes("Fixture health says PAG unavailable")), guidanceOptions);
  await page.key("ArrowDown"); await page.key("Escape");
  await page.click(expression('button[aria-label="X guidance"]'));
  await page.click(`(() => [...document.querySelectorAll('[role="option"]')].find(e=>e.textContent.includes("PAG")))()`);
  const preservedMode = await page.eval(`window.__fixtureInspect().draft.axes.x`);
  record("mouse and keyboard cannot replace a saved unsupported PAG mode", preservedMode.field === "guidance" && preservedMode.values[0].value === "pag", preservedMode);

  await page.click(expression('button[aria-label="Y 轴参数"]'));
  const fieldOption = await page.eval(`(() => {const e=[...document.querySelectorAll('[role="option"]')].find(item=>item.textContent.includes("PAG 强度")); return e&&{text:e.textContent.trim(),disabled:e.disabled,reason:e.title};})()`);
  record("runtime-false PAG scale field is disabled with an explicit reason", Boolean(fieldOption?.disabled && fieldOption.reason.includes("Fixture health says PAG unavailable")), fieldOption);
  await page.click(`(() => [...document.querySelectorAll('[role="option"]')].find(e=>e.textContent.includes("PAG 强度")))()`);
  const preservedScale = await page.eval(`window.__fixtureInspect().draft.axes.y`);
  record("saved PAG scale and its typed value remain unchanged", preservedScale.field === "pag.scale" && preservedScale.values[0].value === "0.8", preservedScale);
  const preflight = await page.eval(`({disabled:document.querySelector('.xyz-run-actions button')?.disabled,message:document.querySelector('[aria-label="运行时引导能力校验"]')?.textContent,footer:document.querySelector('.xyz-run-bar')?.textContent,plans:window.__xyzPlanBodies?.length||0})`);
  await page.click(expression(".xyz-run-actions button"));
  record("unsupported axes expose a visible preflight blocker and never claim a valid plan", preflight.disabled && preflight.message.includes("草稿原值保留") && preflight.footer.includes("不可用") && preflight.plans === 0, preflight);
  await page.shot("after-runtime-block.png");

  await page.eval(`window.__fixturePatchDraft("iL", {axes:{x:{field:"guidance",values:[{value:"pag",label:"saved iL PAG"}],range:null},y:null,z:null}})`);
  await page.eval(`(() => {const e=document.querySelector('select[aria-label="夹具引擎"]'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,'iL'); e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await page.waitFor("window.__fixtureInspect().engine === 'iL' && Boolean(document.querySelector('[aria-label=\"运行时引导能力校验\"]'))", "engine-scoped stale capability refusal");
  const stale = await page.eval(`({engine:window.__fixtureInspect().engine,snapshotEngine:window.__fixtureInspect().runtimeGuidance.engine,value:window.__fixtureInspect().draft.axes.x.values[0].value,issue:document.querySelector('[aria-label="运行时引导能力校验"]')?.textContent,planDisabled:document.querySelector('.xyz-run-actions button')?.disabled})`);
  record("late/old engine-tagged guidance cannot authorize the newly selected engine", stale.engine === "iL" && stale.snapshotEngine === "SD" && stale.value === "pag" && stale.issue.includes("不属于当前引擎") && stale.planDisabled, stale);

  await page.eval(`window.__fixturePatchDraft("Anima", {axes:{x:{field:"guidance",values:[{value:"cfg_zero_star",label:"saved CFG-Zero*"}],range:null},y:null,z:null}})`);
  await page.eval(`window.__fixtureSetRuntimeGuidance({engine:"Anima",modes:{none:{status:"available",reason:""},pag:{status:"available",reason:""},cfg_zero_star:{status:"unavailable",reason:"Fixture health says CFG-Zero* unavailable"}}})`);
  await page.eval(`(() => {const e=document.querySelector('select[aria-label="夹具引擎"]'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,'Anima'); e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await page.waitFor("window.__fixtureInspect().engine === 'Anima' && Boolean(document.querySelector('[aria-label=\"运行时引导能力校验\"]'))", "Anima CFG-Zero* runtime refusal");
  await page.click(expression('button[aria-label="X guidance"]'));
  const cfgOption = await page.eval(`(() => {const e=[...document.querySelectorAll('[role="option"]')].find(item=>item.textContent.includes("CFG-Zero*")); return e&&{text:e.textContent.trim(),disabled:e.disabled,reason:e.title};})()`);
  record("runtime-false CFG-Zero* mode is disabled with a current-engine reason", Boolean(cfgOption?.disabled && cfgOption.reason.includes("Fixture health says CFG-Zero* unavailable")), cfgOption);
  await page.click(`(() => [...document.querySelectorAll('[role="option"]')].find(e=>e.textContent.includes("CFG-Zero*")))()`);
  const cfgSaved = await page.eval(`({axis:window.__fixtureInspect().draft.axes.x,planDisabled:document.querySelector('.xyz-run-actions button')?.disabled,issue:document.querySelector('[aria-label="运行时引导能力校验"]')?.textContent})`);
  record("saved CFG-Zero* mode remains unchanged and blocks preflight", cfgSaved.axis.values[0].value === "cfg_zero_star" && cfgSaved.planDisabled && cfgSaved.issue.includes("当前运行时不可用"), cfgSaved);
  await page.shot("after-cfg-zero-star-block.png");

  record("controlled browser fixture has no runtime exceptions", page.errors.length === 0, { errors: page.errors });
  await writeFile(path.join(EVIDENCE, "ledger.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ evidence: EVIDENCE, result }, null, 2));
} finally {
  page?.ws.close();
  if (edge && edge.exitCode === null) edge.kill(); // Only the Edge process created above.
  await vite.close();
}
