import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";

const ROOT = path.resolve(import.meta.dirname, "..");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const stamp = `xyz-match-t2i-${new Date().toISOString().slice(0, 10).replaceAll("-", "")}-${Date.now()}-${process.pid}`;
const EVIDENCE = path.join(os.tmpdir(), "opencode", stamp);
const SHOTS = path.join(EVIDENCE, "screenshots");
await mkdir(SHOTS, { recursive: true });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const exprForText = (selector, text) => `(() => [...document.querySelectorAll(${JSON.stringify(selector)})].find((el) => el.textContent.includes(${JSON.stringify(text)})))()`;

class Page {
  static async open(port) {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const target = targets.find((item) => item.type === "page");
    const page = new Page(new WebSocket(target.webSocketDebuggerUrl));
    await new Promise((resolve, reject) => { page.ws.onopen = resolve; page.ws.onerror = reject; });
    await page.send("Page.enable"); await page.send("Runtime.enable");
    return page;
  }
  constructor(ws) {
    this.ws = ws; this.nextId = 0; this.pending = new Map(); this.errors = [];
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id); this.pending.delete(message.id);
        message.error ? reject(new Error(message.error.message)) : resolve(message.result);
      } else if (message.method === "Runtime.exceptionThrown") this.errors.push(message.params.exceptionDetails?.exception?.description || message.params.exceptionDetails?.text);
    };
  }
  send(method, params = {}) {
    const id = ++this.nextId;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  async eval(expression) {
    const result = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }
  async wait(expression, label = expression) {
    const start = Date.now();
    while (Date.now() - start < 15000) {
      if (await this.eval(expression).catch(() => false)) return;
      await sleep(100);
    }
    throw new Error(`Timed out: ${label}`);
  }
  async viewport(width, height) {
    await this.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 500 });
    await sleep(180);
  }
  async moveAway() {
    await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: 4, y: 4 });
    await sleep(100);
  }
  async click(expression) {
    const box = await this.eval(`(() => {const e=${expression};if(!e)return null;e.scrollIntoView({block:'nearest',inline:'nearest'});const r=e.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2,width:r.width,height:r.height};})()`);
    if (!box || box.width <= 0 || box.height <= 0) throw new Error(`Not visible: ${expression}`);
    for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) await this.send("Input.dispatchMouseEvent", { type, x: box.x, y: box.y, button: "left", clickCount: 1 });
    await sleep(120);
  }
  async key(key, modifiers = 0) {
    const codes = { Escape: 27, Tab: 9, Enter: 13, ArrowDown: 40, ArrowUp: 38, a: 65 };
    const base = { key, code: key.length === 1 ? `Key${key.toUpperCase()}` : key, windowsVirtualKeyCode: codes[key], nativeVirtualKeyCode: codes[key], modifiers };
    await this.send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...base });
    await this.send("Input.dispatchKeyEvent", { type: "keyUp", ...base });
    await sleep(100);
  }
  async fill(selector, value) {
    await this.click(`document.querySelector(${JSON.stringify(selector)})`);
    await this.key("a", 2);
    await this.send("Input.insertText", { text: String(value) });
    await sleep(120);
  }
  async choose(label, text) {
    await this.click(`document.querySelector(${JSON.stringify(`button[aria-label="${label}"]`)})`);
    await this.wait(`Boolean([...document.querySelectorAll('[role="option"]')].find((el)=>el.textContent.includes(${JSON.stringify(text)})&&!el.disabled))`, `${label} option ${text}`);
    await this.click(`([...document.querySelectorAll('[role="option"]')].find((el)=>el.textContent.includes(${JSON.stringify(text)})&&!el.disabled))`);
  }
  async shot(name) {
    const { data } = await this.send("Page.captureScreenshot", { format: "png" });
    await writeFile(path.join(SHOTS, name), Buffer.from(data, "base64"));
  }
}

const vite = await createServer({ configFile: false, root: ROOT, cacheDir: path.join(EVIDENCE, "vite-cache"), appType: "mpa",
  plugins: [react()], logLevel: "error", server: { host: "127.0.0.1", port: 0, strictPort: false, hmr: false, open: false, watch: null },
  optimizeDeps: { noDiscovery: true, include: ["react", "react-dom/client", "lucide-react"] },
});
let edge, page;
try {
  await vite.listen();
  const address = vite.httpServer.address();
  const probe = net.createServer(); await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const cdpPort = probe.address().port; await new Promise((resolve) => probe.close(resolve));
  const profile = path.join(EVIDENCE, "edge-profile"); await mkdir(profile);
  edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${cdpPort}`, `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--disable-background-networking", "--window-size=1440,900", "about:blank"], { stdio: "ignore", windowsHide: true });
  let ready = false;
  for (let i = 0; i < 80; i += 1) { try { if ((await fetch(`http://127.0.0.1:${cdpPort}/json/version`)).ok) { ready = true; break; } } catch {} await sleep(150); }
  assert.ok(ready, "owned Edge CDP endpoint is ready");
  page = await Page.open(cdpPort);
  await page.viewport(1440, 900);
  await page.send("Page.navigate", { url: `http://127.0.0.1:${address.port}/scripts/xyz-sol-app.fixture.html` });
  await page.wait("Boolean(document.querySelector('.generation-mode-switch'))&&!document.querySelector('.app-loader')", "real App fixture");
  await page.click(exprForText(".model-seg", "Anima"));
  await page.wait("Boolean(document.querySelector('.checkpoint-picker.split-model-picker'))", "T2I Anima split-model panel");
  await sleep(350);
  await page.shot("after-t2i-anima-1440x900.png");
  await page.moveAway();
  const captureParity = `(() => {
    const read = (selector, anchorSelector = null) => { const e=typeof selector==='string'?document.querySelector(selector):selector;if(!e)return null;const c=getComputedStyle(e),r=e.getBoundingClientRect(),a=anchorSelector?document.querySelector(anchorSelector)?.getBoundingClientRect():null;const n=(v)=>Math.round(v*100)/100;return {tag:e.tagName,className:e.className,rect:a?[n(r.x-a.x),n(r.y-a.y),n(r.width),n(r.height)]:[n(r.width),n(r.height)],display:c.display,gap:c.gap,margin:c.margin,padding:c.padding,border:c.border,borderRadius:c.borderRadius,backgroundColor:c.backgroundColor,color:c.color,font:c.font,fontSize:c.fontSize,fontFamily:c.fontFamily,minHeight:c.minHeight,height:c.height}; };
    const fields=[...document.querySelectorAll('.checkpoint-picker .split-model-field')].map((field)=>({label:field.querySelector(':scope > span:first-child')?.textContent.trim(),head:read(field.querySelector(':scope > span:first-child'),'.checkpoint-picker'),count:read(field.querySelector(':scope > span:first-child small'),'.checkpoint-picker'),select:read(field.querySelector('.workspace-select-trigger'),'.checkpoint-picker'),path:read(field.querySelector('.checkpoint-path'),'.checkpoint-picker')}));
    return {engine:document.querySelector('.model-seg.active .seg-name')?.textContent,model:read('.checkpoint-picker'),heading:read('.checkpoint-label','.checkpoint-picker'),fields,refresh:read('.checkpoint-picker .asset-refresh','.checkpoint-picker'),loraHeading:read('.lora-title','.checkpoint-picker'),loraTitle:read('.lora-title h2','.lora-title'),loraCount:read('.lora-count','.lora-title'),manage:read('.lora-title button','.lora-title'),table:read('.lora-summary-table','.checkpoint-picker'),row:read('.lora-summary-row','.lora-summary-table'),name:read('.lora-summary-name','.lora-summary-row'),weight:read('.lora-summary-row b','.lora-summary-row')};
  })()`;
  const basicMetrics = await page.eval(captureParity);
  const basicDisabledStyle = await page.eval(`(() => {const row=document.querySelector('.lora-summary-row'),originalClass=row.className;row.classList.add('disabled');const read=(e)=>{const c=getComputedStyle(e);return {color:c.color,opacity:c.opacity,textDecorationLine:c.textDecorationLine,font:c.font};};const result={row:read(row),name:read(row.querySelector('.lora-summary-name')),weight:read(row.querySelector('b'))};row.className=originalClass;return result;})()`);
  const basicBefore = await page.eval(`(() => {const w=window.__fixtureAppState().workspace;return {model:w.model,engineSettings:w.engineSettingsByEngine,mounts:w.mountedLorasByEngine,groups:w.loraGroupsByEngine,presets:w.promptPresets,active:w.activePromptPresets,canvas:w.canvasMaxEdge,output:w.outputTemplate};})()`);
  await page.viewport(1366, 768); await page.shot("after-t2i-anima-1366x768.png"); await page.moveAway();
  const basicMetrics1366 = await page.eval(captureParity);
  await page.viewport(1440, 900);
  await page.choose("生成模式", "XYZ");
  await page.wait("Boolean(document.querySelector('.xyz-sidebar-fields'))&&!document.querySelector('.xyz-sidebar-models [role=status]')", "XYZ sidebar with catalog");
  await sleep(350);
  await page.shot("after-xyz-anima-1440x900.png");
  await page.moveAway();
  const xyzMetrics = await page.eval(captureParity);
  const parityKeys = ["model", "heading", "fields", "refresh", "loraHeading", "loraTitle", "loraCount", "manage", "table", "row", "name", "weight"];
  const styleParity1440 = Object.fromEntries(parityKeys.map((key) => [key, JSON.stringify(basicMetrics[key]) === JSON.stringify(xyzMetrics[key])]));
  assert.ok(parityKeys.every((key) => styleParity1440[key]), `1440px T2I / XYZ computed styles and relative rectangles match: ${JSON.stringify(styleParity1440)}`);
  await page.viewport(1366, 768); await page.shot("after-xyz-anima-1366x768.png"); await page.moveAway();
  const xyzMetrics1366 = await page.eval(captureParity);
  const styleParity1366 = Object.fromEntries(parityKeys.map((key) => [key, JSON.stringify(basicMetrics1366[key]) === JSON.stringify(xyzMetrics1366[key])]));
  assert.ok(parityKeys.every((key) => styleParity1366[key]), `1366px T2I / XYZ computed styles and relative rectangles match: ${JSON.stringify(styleParity1366)}`);
  await page.viewport(1440, 900);
  assert.equal(basicMetrics.engine, "Anima"); assert.equal(xyzMetrics.engine, "Anima");
  assert.equal(await page.eval("document.querySelector('.xyz-sidebar-mounts h2')?.textContent"), "全局 LoRA");
  assert.equal(await page.eval("document.querySelectorAll('.xyz-sidebar-lora').length"), 0);

  // Catalog refresh uses the same two read-only engine adapters and leaves saved model values intact.
  const savedBeforeRefresh = await page.eval("structuredClone(window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.parameters)");
  await page.eval(`(() => {const base=window.__appFetch;window.__appFetch=async(url,method,body)=>{const response=await base(url,method,body);if(window.__dropAnimaModels&&url.startsWith('/api/models?engine=Anima')){const data=await response.json();data.assets.diffusion_model.models=[];return new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}})}return response};window.__dropAnimaModels=true;})()`);
  await page.click("document.querySelector('.xyz-sidebar-models .asset-refresh')");
  await page.wait("document.querySelector('.xyz-sidebar-models .split-model-field .workspace-select-trigger > span')?.textContent.includes('保存值')", "unavailable saved component display");
  const unavailable = await page.eval(`({value:window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.parameters.diffusion_model,label:document.querySelector('.xyz-sidebar-models .split-model-field .workspace-select-trigger > span')?.textContent,path:document.querySelector('.xyz-sidebar-models .split-model-field .checkpoint-path')?.textContent})`);
  assert.equal(unavailable.value, savedBeforeRefresh.diffusion_model);
  await page.eval("window.__dropAnimaModels=false");
  await page.click("document.querySelector('.xyz-sidebar-models .asset-refresh')");
  await page.wait("document.querySelector('.xyz-sidebar-models .split-model-field .workspace-select-trigger > span')?.textContent==='Anima diffusion'", "restored model catalog selection");

  // The XYZ manager is draft-owned. Exercise exact weight, enabled, reorder/remove and confirmed group copy.
  await page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");
  await page.wait("Boolean(document.querySelector('.xyz-lora-manager[role=dialog]'))", "XYZ-owned LoRA manager");
  await page.choose("XYZ 添加 LoRA", "Anima Portrait 1");
  await page.wait("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.loras.length===2", "XYZ manager add");
  await page.fill('[aria-label="XYZ LoRA 2 权重"]', "-0.1250"); await page.key("Tab");
  await page.wait("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.loras[1].weight==='-0.1250'", "four-place negative weight");
  await page.click('document.querySelector("[aria-label=\\"XYZ 启用 LoRA 2\\"]")');
  await page.wait("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.loras[1].enabled===false", "XYZ-only toggle");
  const xyzDisabledStyle = await page.eval(`(() => {const row=document.querySelector('.xyz-sidebar-mounts .lora-summary-row.disabled');const read=(e)=>{const c=getComputedStyle(e);return {color:c.color,opacity:c.opacity,textDecorationLine:c.textDecorationLine,font:c.font};};return {row:read(row),name:read(row.querySelector('.lora-summary-name')),weight:read(row.querySelector('b'))};})()`);
  assert.deepEqual(xyzDisabledStyle, basicDisabledStyle, "disabled summary row colors and typography match Basic");
  await page.click('document.querySelector("[aria-label=\\"XYZ LoRA 2 上移\\"]")');
  await page.wait("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.loras[0].value==='Anima-portrait-1.safetensors'", "XYZ-only reorder");
  await page.click('document.querySelector("[aria-label=\\"XYZ LoRA 2 移除\\"]")');
  await page.wait("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.loras.length===1", "XYZ-only remove");
  const beforeGroupCopy = await page.eval("structuredClone(window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.parameters.prompt)");
  await page.click("document.querySelector('.xyz-lora-manager-groups > summary')");
  await page.wait("document.querySelector('.xyz-lora-manager-groups')?.textContent.includes('Library Anima')", "fixture group snapshot is available");
  await page.click("([...document.querySelectorAll('.xyz-lora-manager-groups button')].find((button)=>button.textContent.includes('Library Anima')))");
  await page.wait("Boolean(document.querySelector('.xyz-group-copy-dialog[role=alertdialog]'))", "nested group snapshot confirmation");
  await page.key("Escape");
  await page.wait("!document.querySelector('.xyz-group-copy-dialog')", "nested confirmation Escape");
  assert.ok(await page.eval("Boolean(document.querySelector('.xyz-lora-manager[role=dialog]'))"), "Escape leaves the parent XYZ manager open");
  assert.ok(await page.eval("Boolean(document.activeElement?.closest('.xyz-lora-manager-groups'))"), "nested confirmation restores focus to its group row");
  await page.click("([...document.querySelectorAll('.xyz-lora-manager-groups button')].find((button)=>button.textContent.includes('Library Anima')))");
  await page.wait("Boolean(document.querySelector('.xyz-group-copy-dialog'))", "reopened group confirmation");
  await page.wait("Boolean(document.querySelector('.xyz-group-copy-dialog button.primary'))", "group verification finished");
  await page.click("document.querySelector('.xyz-group-copy-dialog button.primary')");
  await page.wait("!document.querySelector('.xyz-group-copy-dialog')", "confirmed group snapshot copy");
  await page.wait("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.group_snapshots.some((group)=>group.name==='Library Anima')", "confirmed XYZ snapshot saved");
  const managerDraft = await page.eval(`(() => {const d=window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima;return {mounts:d.baseline.loras,prompt:d.baseline.parameters.prompt,groups:d.baseline.group_snapshots};})()`);
  assert.equal(managerDraft.prompt, beforeGroupCopy);
  assert.equal(managerDraft.mounts.length, 1); assert.equal(managerDraft.mounts[0].contentRevision, 1);
  assert.ok(managerDraft.groups.some((group)=>group.name==='Library Anima'));
  await page.click("document.querySelector('.xyz-lora-manager-add .workspace-select-trigger')");
  await page.wait("Boolean(document.querySelector('.xyz-lora-manager-add .workspace-select.open'))", "manager catalog select open");
  await page.key("Escape");
  await page.wait("!document.querySelector('.xyz-lora-manager-add .workspace-select.open')", "catalog selector Escape");
  assert.ok(await page.eval("Boolean(document.querySelector('.xyz-lora-manager[role=dialog]'))"), "selector Escape leaves XYZ manager open");
  await page.key("Escape");
  await page.wait("!document.querySelector('.xyz-lora-manager')", "XYZ manager Escape close");
  assert.equal(await page.eval("document.activeElement?.textContent.trim()"), "管理", "manager close returns focus to the XYZ manage button");

  const basicAfterManager = await page.eval(`(() => {const w=window.__fixtureAppState().workspace;return {model:w.model,engineSettings:w.engineSettingsByEngine,mounts:w.mountedLorasByEngine,groups:w.loraGroupsByEngine,presets:w.promptPresets,active:w.activePromptPresets,canvas:w.canvasMaxEdge,output:w.outputTemplate};})()`);
  assert.deepEqual(basicAfterManager, basicBefore, "XYZ model refresh and LoRA edits cannot touch Basic / i2i / real groups");
  const loraApiWrites = await page.eval("window.__fixtureAppCalls.filter((call)=>call.url.startsWith('/api/loras')&&call.method!=='GET')");
  assert.equal(loraApiWrites.length, 0, "XYZ management adds no LoRA write endpoint");

  await page.viewport(390, 844);
  if (!(await page.eval("document.querySelector('.xyz-sidebar-content')?.classList.contains('expanded')"))) await page.click("document.querySelector('.xyz-sidebar-mobile-toggle')");
  await page.wait("document.querySelector('.xyz-sidebar-content')?.classList.contains('expanded')", "expanded mobile XYZ sidebar");
  await sleep(150); await page.shot("after-xyz-anima-390x844.png");
  const mobileSidebar = await page.eval("({width:innerWidth,body:document.body.scrollWidth,root:document.documentElement.scrollWidth,rail:document.querySelector('.xyz-sidebar-content')?.getBoundingClientRect().toJSON(),expanded:document.querySelector('.xyz-sidebar-content')?.classList.contains('expanded')})");
  assert.ok(mobileSidebar.body <= mobileSidebar.width && mobileSidebar.root <= mobileSidebar.width && mobileSidebar.expanded, `390px XYZ sidebar remains unclipped: ${JSON.stringify(mobileSidebar)}`);
  await page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");
  await page.wait("Boolean(document.querySelector('.xyz-lora-manager'))", "mobile XYZ manager");
  await page.shot("after-xyz-manager-390x844.png");
  const managerMobile = await page.eval(`(() => {const d=document.querySelector('.xyz-lora-manager'),b=d.getBoundingClientRect(),f=d.querySelector('.lora-modal-foot').getBoundingClientRect(),body=d.querySelector('.xyz-lora-manager-body').getBoundingClientRect();return {width:innerWidth,documentWidth:document.documentElement.scrollWidth,dialog:[b.x,b.y,b.width,b.height],body:[body.x,body.y,body.width,body.height],footer:[f.x,f.y,f.width,f.height],bodyScrollWidth:d.querySelector('.xyz-lora-manager-scroll').scrollWidth,bodyClientWidth:d.querySelector('.xyz-lora-manager-scroll').clientWidth,buttonRects:[...d.querySelectorAll('button')].map((e)=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.right,r.bottom]})};})()`);
  assert.ok(managerMobile.documentWidth <= managerMobile.width && managerMobile.footer[1] + managerMobile.footer[3] <= 844 && managerMobile.buttonRects.every((r)=>r[0]>=0&&r[2]<=390&&r[3]<=844), `390px manager controls/footer fit: ${JSON.stringify(managerMobile)}`);
  await page.viewport(390, 430); await page.shot("after-xyz-manager-keyboard-390x430.png");
  const keyboardManager = await page.eval(`(() => {const d=document.querySelector('.xyz-lora-manager').getBoundingClientRect(),f=document.querySelector('.xyz-lora-manager .lora-modal-foot').getBoundingClientRect();return {width:innerWidth,height:innerHeight,documentWidth:document.documentElement.scrollWidth,dialog:[d.x,d.y,d.width,d.height],footer:[f.x,f.y,f.width,f.height],finish:[...document.querySelectorAll('.xyz-lora-manager .lora-modal-foot button')].every((e)=>{const r=e.getBoundingClientRect();return r.y>=0&&r.bottom<=innerHeight&&r.left>=0&&r.right<=innerWidth})};})()`);
  assert.ok(keyboardManager.documentWidth <= 390 && keyboardManager.finish && keyboardManager.dialog[1] + keyboardManager.dialog[3] <= 430 && keyboardManager.footer[1] + keyboardManager.footer[3] <= 430, `soft-keyboard-height dialog keeps its footer visible: ${JSON.stringify(keyboardManager)}`);
  await page.key("Escape"); await page.wait("!document.querySelector('.xyz-lora-manager')", "mobile dialog close");
  const styleParity = { "1440x900": styleParity1440, "1366x768": styleParity1366 };
  const manifest = { stamp, evidence: EVIDENCE, basicMetrics, xyzMetrics, basicMetrics1366, xyzMetrics1366, styleParity, basicDisabledStyle, xyzDisabledStyle, unavailable, managerDraft, mobileSidebar, managerMobile, keyboardManager, browserErrors: page.errors,
    sourceSha256: createHash("sha256").update(await (await import("node:fs/promises")).readFile(path.join(ROOT, "src/XYZBaselineSidebar.jsx"))).digest("hex") };
  await writeFile(path.join(EVIDENCE, "comparison-and-manager-report.json"), JSON.stringify(manifest, null, 2));
  console.log(JSON.stringify({ evidence: EVIDENCE, screenshots: SHOTS, styleParity, unavailable, managerDraft, mobileSidebar, managerMobile, keyboardManager, browserErrors: page.errors }, null, 2));
} finally {
  try { page?.ws.close(); } catch {}
  if (edge && edge.exitCode === null) { edge.kill(); await Promise.race([new Promise((resolve) => edge.once("exit", resolve)), sleep(2500)]); }
  await vite.close();
}
