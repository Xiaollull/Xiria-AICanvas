import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";

const ROOT = path.resolve(import.meta.dirname, "..");
const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const EVIDENCE = await makeEvidenceRoot();
const SCREENSHOTS = path.join(EVIDENCE, "screenshots");
await mkdir(SCREENSHOTS, { recursive: true });
const ledger = [];
const record = (name, ok, detail = {}) => {
  ledger.push({ name, result: ok ? "PASS" : "FAIL", detail });
  console.log(`${ok ? "PASS" : "FAIL"} ${name} ${JSON.stringify(detail)}`);
  assert.ok(ok, `${name}: ${JSON.stringify(detail)}`);
};

async function makeEvidenceRoot() {
  const base = path.join(os.tmpdir(), "opencode", "xyz-sol-layout1-20261006");
  await access(base);
  const pathValue = path.join(base, `after-ui-${process.pid}`);
  await mkdir(pathValue, { recursive: false });
  return pathValue;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const clickExpression = (selector) => `document.querySelector(${JSON.stringify(selector)})`;
const clickTextExpression = (scope, text) => `(() => { const root = document.querySelector(${JSON.stringify(scope)}) || document; return [...root.querySelectorAll('button')].find((button) => button.textContent.includes(${JSON.stringify(text)})); })()`;

class Page {
  static async open(port) {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const target = list.find((item) => item.type === "page");
    const page = new Page(new WebSocket(target.webSocketDebuggerUrl));
    await new Promise((resolve, reject) => { page.ws.onopen = resolve; page.ws.onerror = reject; });
    await page.send("Page.enable"); await page.send("Runtime.enable"); await page.send("DOM.enable");
    return page;
  }
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.runtimeErrors = [];
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id); this.pending.delete(message.id);
        message.error ? reject(new Error(`${message.error.message} ${message.error.data || ""}`)) : resolve(message.result);
      } else if (message.method === "Runtime.exceptionThrown") this.runtimeErrors.push(message.params.exceptionDetails?.exception?.description || message.params.exceptionDetails?.text);
      else if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") this.runtimeErrors.push(message.params.args.map((arg) => arg.value ?? arg.description).join(" "));
    };
  }
  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  async eval(expression) {
    const result = await this.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }
  async waitFor(expression, { timeout = 12000, label = expression } = {}) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      try { const value = await this.eval(expression); if (value) return value; } catch {}
      await sleep(100);
    }
    throw new Error(`Timed out waiting for ${label}`);
  }
  async viewport(width, height) {
    await this.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 500 });
    await sleep(250);
  }
  async navigate(url) {
    await this.send("Page.navigate", { url });
    await this.waitFor("document.readyState === 'complete'", { label: "fixture document" });
    await this.waitFor("Boolean(document.querySelector('.xyz-workspace .xyz-run-bar'))", { label: "mounted XYZ component" });
    await sleep(500);
  }
  async rect(expression) {
    return this.eval(`(() => { const el = ${expression}; if (!el) return null; el.scrollIntoView({block:'nearest',inline:'nearest'}); const r=el.getBoundingClientRect(); return {x:r.left+r.width/2,y:r.top+r.height/2,w:r.width,h:r.height}; })()`);
  }
  async click(expression) {
    const box = await this.rect(expression);
    if (!box || !Number.isFinite(box.x) || !Number.isFinite(box.y)) throw new Error(`No visible element for ${expression}`);
    for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) await this.send("Input.dispatchMouseEvent", { type, x: box.x, y: box.y, button: "left", clickCount: 1 });
    await sleep(120);
  }
  async key(key, { control = false, shift = false } = {}) {
    const codes = { Enter: 13, Escape: 27, Tab: 9, ArrowDown: 40, ArrowUp: 38, Home: 36, End: 35, Backspace: 8, a: 65 };
    const base = { key, code: key.length === 1 ? `Key${key.toUpperCase()}` : key, windowsVirtualKeyCode: codes[key], nativeVirtualKeyCode: codes[key], modifiers: (control ? 2 : 0) | (shift ? 8 : 0) };
    await this.send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...base });
    await this.send("Input.dispatchKeyEvent", { type: "keyUp", ...base });
    await sleep(100);
  }
  async type(text) { await this.send("Input.insertText", { text }); await sleep(130); }
  async fill(selector, value) {
    await this.click(clickExpression(selector));
    await this.key("a", { control: true });
    await this.type(String(value));
  }
  async choose(ariaLabel, text) {
    await this.click(clickExpression(`button[aria-label=${JSON.stringify(ariaLabel)}]`));
    await this.waitFor(`Boolean([...document.querySelectorAll('[role="option"]')].find((item) => item.textContent.includes(${JSON.stringify(text)}) && !item.disabled))`, { label: `${ariaLabel} option ${text}` });
    await this.click(`(() => [...document.querySelectorAll('[role="option"]')].find((item) => item.textContent.includes(${JSON.stringify(text)}) && !item.disabled))()`);
    if (await this.eval("Boolean(document.querySelector('#xyz-axis-replace-title'))")) await this.clickText(".xyz-sync-dialog", "确认更改");
  }
  async chooseIndexed(ariaLabel, index, text) {
    await this.click(`document.querySelectorAll(${JSON.stringify(`button[aria-label=${JSON.stringify(ariaLabel)}]`)})[${index}]`);
    await this.waitFor(`Boolean([...document.querySelectorAll('[role="option"]')].find((item) => item.textContent.includes(${JSON.stringify(text)}) && !item.disabled))`, { label: `${ariaLabel}[${index}] option ${text}` });
    await this.click(`(() => [...document.querySelectorAll('[role="option"]')].find((item) => item.textContent.includes(${JSON.stringify(text)}) && !item.disabled))()`);
    if (await this.eval("Boolean(document.querySelector('#xyz-axis-replace-title'))")) await this.clickText(".xyz-sync-dialog", "确认更改");
  }
  async clickText(scope, text) {
    const expression = clickTextExpression(scope, text);
    await this.waitFor(`Boolean(${expression})`, { label: `button ${text}` });
    await this.click(expression);
  }
  async shot(name) {
    const { data } = await this.send("Page.captureScreenshot", { format: "png" });
    await writeFile(path.join(SCREENSHOTS, name), Buffer.from(data, "base64"));
  }
}

const snapshotFiles = ["src/XYZPlotWorkspace.jsx", "src/XYZAxisEditor.jsx", "src/xyz-axis-editor.js", "src/xyz-client.js", "src/App.jsx", "src/styles.css",
  "src/XYZBaselineSidebar.jsx", "src/xyz-sidebar-context.js", "src/use-xyz-catalog.js", "scripts/xyz-sol-layout.test.mjs",
  "scripts/xyz-sol-app.fixture.jsx", "scripts/xyz-sol-app.fixture.html", "backend/tests/test_xyz_fixed_canvas.py",
  "scripts/xyz-phase-b.test.mjs", "scripts/xyz-phase-c.test.mjs", "scripts/xyz-phase-e.test.mjs", "scripts/xyz-ui-redesign.test.mjs", "scripts/xyz-ui-redesign.fixture.html", "scripts/xyz-ui-redesign.fixture.jsx", "scripts/xyz-ui-redesign.browser.mjs",
  "docs/usage.html", "docs/zh/usage.html",
  ...["00-START-HERE.md", "02-FRONTEND-GENERATE.md", "09-TEST-VERIFICATION.md", "10-TASK-LOOKUP.md", "11-FILE-INVENTORY.md", "CHANGELOG.md"].map((name) => ({
    relative: `.Structure/${name}`, source: path.resolve(ROOT, "..", ".Structure", name),
  }))];
const hashes = [];
for (const file of snapshotFiles) {
  const relative = typeof file === "string" ? file : file.relative;
  const sourcePath = typeof file === "string" ? path.join(ROOT, relative) : file.source;
  const bytes = await readFile(sourcePath), target = path.join(EVIDENCE, "source-final", relative);
  await mkdir(path.dirname(target), { recursive: true }); await copyFile(sourcePath, target);
  hashes.push(`${createHash("sha256").update(bytes).digest("hex")}  ${relative}`);
}
await writeFile(path.join(EVIDENCE, "source-final.sha256"), `${hashes.join("\n")}\n`);
await writeFile(path.join(EVIDENCE, "baseline-reference.json"), JSON.stringify({
  note: "Fresh before-snapshot and four-viewport UI baseline are ../before and ../baseline-ui-four. The Phase E hashes below are historical only, not used for rollback.",
  phase_e_source_final: { "src/XYZPlotWorkspace.jsx": "2d9ba8aa9ce260c63f6dda495f11eb3822cff3eb854d47822d3c4d2edbc3f926", "src/App.jsx": "8f93b73c17c912bf3cc2ad215efce37ee38612e7f1a33e04b8fae2af2a7b0dc8", "src/styles.css": "7876c18824ecfff3f389cd52c0aa26c71612b6b689c080bc6f627b89e0bc57db" },
}, null, 2));

await access(EDGE);
const vite = await createServer({ configFile: false, root: ROOT, cacheDir: path.join(EVIDENCE, "vite-cache"), appType: "mpa", plugins: [react()], logLevel: "error",
  server: { host: "127.0.0.1", port: 0, strictPort: false, hmr: false, open: false, watch: null },
  optimizeDeps: { noDiscovery: true, include: ["react", "react-dom/client", "lucide-react"] },
});
let edge = null, page = null;
try {
  await vite.listen();
  const address = vite.httpServer.address(), url = `http://127.0.0.1:${address.port}/scripts/xyz-ui-redesign.fixture.html`;
  const portProbe = net.createServer();
  await new Promise((resolve) => portProbe.listen(0, "127.0.0.1", resolve));
  const cdpPort = portProbe.address().port;
  await new Promise((resolve) => portProbe.close(resolve));
  const profile = path.join(EVIDENCE, "edge-profile"); await mkdir(profile);
  edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${cdpPort}`, `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check",
    "--disable-extensions", "--disable-sync", "--disable-background-networking", "--window-size=1440,900", "about:blank"], { stdio: "ignore", windowsHide: true });
  let ready = false;
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try { const response = await fetch(`http://127.0.0.1:${cdpPort}/json/version`); if (response.ok) { ready = true; break; } } catch {}
    if (edge.exitCode !== null) throw new Error(`owned Edge exited early (${edge.exitCode})`);
    await sleep(200);
  }
  if (!ready) throw new Error("temporary Edge CDP endpoint did not start");
  page = await Page.open(cdpPort);
  await page.viewport(1440, 900);
  await page.navigate(url);
  await page.waitFor("window.__fixtureInspect && window.__fixtureInspect().engine === 'SD'", { label: "SD fixture state" });
  await page.waitFor("document.querySelector('.xyz-workspace-title')?.textContent.includes('SD')", { label: "SD workspace header" });

  for (const [width, height, name] of [[1920, 1080, "after-config-1920x1080.png"], [1440, 900, "after-config-1440x900.png"], [1366, 768, "after-config-1366x768.png"], [390, 844, "after-config-390x844.png"]]) {
    await page.viewport(width, height); await page.shot(name);
    const layout = await page.eval(`(() => { const bar=document.querySelector('.xyz-run-bar'), r=bar?.getBoundingClientRect(), port=document.querySelector('.xyz-workspace-scroll')?.getBoundingClientRect(); const visible=[...document.querySelectorAll('.xyz-axis-card input,.xyz-axis-card button')].filter(e=>{const q=e.getBoundingClientRect();return q.width>0&&q.height>0;}); const overlaps=visible.filter(e=>{const q=e.getBoundingClientRect(),top=Math.max(q.top,port?.top??q.top),bottom=Math.min(q.bottom,port?.bottom??q.bottom);return r&&q.left<r.right&&q.right>r.left&&top<bottom&&top<r.bottom&&bottom>r.top;}); return {width:innerWidth,body:document.body.scrollWidth,root:document.documentElement.scrollWidth,axisCards:document.querySelectorAll('.xyz-axis-card').length,barPosition:getComputedStyle(bar).position,barRect:r&&[Math.round(r.x),Math.round(r.y),Math.round(r.width),Math.round(r.height)],scrollRect:port&&[Math.round(port.x),Math.round(port.y),Math.round(port.width),Math.round(port.height)],overlapCount:overlaps.length,overlaps:overlaps.map(e=>({tag:e.tagName,text:(e.textContent||'').trim().slice(0,60),rect:(()=>{const q=e.getBoundingClientRect();return [Math.round(q.x),Math.round(q.y),Math.round(q.width),Math.round(q.height)]})()})),accent:getComputedStyle(document.documentElement).getPropertyValue('--accent-rgb').trim(),cols:getComputedStyle(document.querySelector('.xyz-axes')).gridTemplateColumns}; })()`);
    const runbarPosition = width <= 760 ? "static" : "sticky";
    record(`${width}x${height} layout has three axes, a docked runbar, no horizontal overflow or control overlap`, layout.axisCards === 3 && layout.body <= width && layout.root <= width && layout.barPosition === runbarPosition && layout.overlapCount === 0 && Boolean(layout.accent), layout);
  }
  await page.viewport(1440, 900);

  const basicBefore = await page.eval("window.__fixtureInspect().basic");
  record("basic controls remain independent and initial X/Y/Z entry points are compact", await page.eval("document.querySelectorAll('.xyz-axis-unused').length === 3 && !document.querySelector('.xyz-baseline').open && document.querySelector('.xyz-workspace-head')?.textContent.includes('新草稿')"), { basicBefore });
  await page.choose("X 轴参数", "种子");
  const seedField = await page.eval("({type:document.querySelector('[aria-label=\"X 第 1 项\"]')?.type,value:document.querySelector('[aria-label=\"X 第 1 项\"]')?.value,kind:typeof window.__fixtureInspect().draft.axes.x.values[0].value})");
  record("uint64 axis remains an exact text string and never passes through Number", seedField.type === "text" && seedField.value === "18446744073709551615" && seedField.kind === "string", seedField);

  await page.choose("X 轴参数", "正向文本");
  await page.fill('[aria-label="X 第 1 项"]', "red, blue, green");
  const prompt = await page.eval("window.__fixtureInspect().draft.axes.x.values[0].value");
  record("Prompt values preserve commas as one independent text item", prompt === "red, blue, green", { prompt });

  await page.choose("X 轴参数", "正向精确替换");
  await page.click("document.querySelector('.xyz-raw-json summary')");
  const rawSelector = 'textarea[aria-label="prompt_replace · 第 1 项高级 JSON"]';
  await page.fill(rawSelector, '{"find":"unfinished');
  await page.key("Tab");
  const invalidJSON = await page.eval(`({value:window.__fixtureInspect().draft.axes.x.values[0].value,rawText:document.querySelector(${JSON.stringify(rawSelector)})?.value,invalid:document.querySelector(${JSON.stringify(rawSelector)})?.getAttribute('aria-invalid'),warning:document.querySelector('.xyz-raw-json small')?.textContent,wrapped:JSON.stringify(window.__fixtureInspect().draft.axes.x.values[0].value).includes('invalid_raw')})`);
  record("malformed JSON repro stays byte-for-byte raw after real keyboard editing and blur, with its error state retained", invalidJSON.value === '{"find":"unfinished' && invalidJSON.rawText === invalidJSON.value && invalidJSON.invalid === "true" && invalidJSON.warning.includes("JSON 无效") && !invalidJSON.wrapped, invalidJSON);
  await page.shot('state-invalid-json.png');

  await page.choose("Y 轴参数", "PAG 强度");
  const pagInitial = await page.eval("({field:window.__fixtureInspect().draft.axes.y.field,value:window.__fixtureInspect().draft.axes.y.values[0].value})");
  record("selecting PAG scale from a null baseline PAG does not throw or invent a value", pagInitial.field === "pag.scale" && pagInitial.value === "", pagInitial);

  await page.choose("X 轴参数", "CFG");
  await page.click("document.querySelector('.xyz-range-toggle input')");
  await page.clickText(".xyz-sync-dialog", "确认更改");
  await page.fill('.xyz-axis-card .xyz-range label:nth-child(1) input', "0.1");
  await page.fill('.xyz-axis-card .xyz-range label:nth-child(2) input', "0.5");
  await page.fill('.xyz-axis-card .xyz-range label:nth-child(3) input', "0.2");
  const range = await page.eval("window.__fixtureInspect().draft.axes.x.range");
  record("decimal range editor sends exact finite start/end/step strings", JSON.stringify(range) === JSON.stringify({ start: "0.1", end: "0.5", step: "0.2" }), range);

  await page.choose("X 轴参数", "完整模型组件方案");
  const componentField = await page.eval("({value:window.__fixtureInspect().draft.axes.x.values[0].value.checkpoint,label:document.querySelector('button[aria-label=\"X checkpoint\"]')?.textContent})");
  record("model-component axis uses the engine's current App-provided asset choices", componentField.value === "fixture-model.safetensors" && componentField.label.includes("Fixture checkpoint"), componentField);

  await page.choose("X 轴参数", "采样器");
  await page.choose("X sampler", "heun");
  const sampler = await page.eval("window.__fixtureInspect().draft.axes.x.values[0].value");
  record("sampler is a constrained live choice, not a free JSON field", sampler === "heun", { sampler });

  await page.click("(() => [...document.querySelectorAll('.xyz-axis-unused')].find((axis)=>axis.querySelector('summary')?.textContent.trim().startsWith('Z'))?.querySelector('summary'))()");
  await page.choose("Z 轴参数", "LoRA 测试组合");
  await page.clickText(".xyz-profile-presets", "添加冻结组：Portrait profile");
  const profileDefault = await page.eval("window.__fixtureInspect().draft.axes.z.values.at(-1).value");
  record("frozen group preset starts with its exact member identity/order and preset text off", profileDefault.mode === "keep_background" && profileDefault.members[0].instanceId === "11111111-1111-4111-8111-111111111111" && profileDefault.includePresetText === false, profileDefault);
  await page.click("document.querySelectorAll('.xyz-profile-editor > label.xyz-check input[type=checkbox]')[1]");
  const presetOptIn = await page.eval("window.__fixtureInspect().draft.axes.z.values.at(-1).value");
  record("group preset words are a single explicit switch and not a default", presetOptIn.includePresetText === true && presetOptIn.preset_prompt === "soft light", { includePresetText: presetOptIn.includePresetText, preset_prompt: presetOptIn.preset_prompt });
  await page.shot('state-complex-profile.png');
  for(const [width,height] of [[1920,1080],[1440,900],[1366,768],[390,844]]) {
    await page.viewport(width,height);
    await page.rect('document.querySelector(".xyz-axis-card:last-child")');
    await page.shot(`state-complex-profile-${width}x${height}.png`);
    const complex=await page.eval(`(() => {const card=document.querySelector('.xyz-axis-card:last-child'),value=card.querySelector('.xyz-value-complex'),c=card.getBoundingClientRect(),v=value.getBoundingClientRect();return {width:innerWidth,body:document.body.scrollWidth,card:[c.x,c.y,c.width,c.height],value:[v.x,v.y,v.width,v.height],cardOverflow:getComputedStyle(card).overflowY,inside:v.left>=c.left&&v.right<=c.right,columns:getComputedStyle(document.querySelector('.xyz-axes')).gridTemplateColumns};})()`);
    record(`${width} complex LoRA profile uses full axis width without card-internal scrolling or body overflow`,complex.body<=width&&complex.inside&&complex.value[2]>=complex.card[2]-30&&complex.cardOverflow==='visible',complex);
  }
  await page.viewport(1440,900);
  await page.chooseIndexed("LoRA 方案", 1, "完全无 LoRA");
  const noLoras = await page.eval("window.__fixtureInspect().draft.axes.z.values.at(-1).value");
  record("no-test and complete-no-LoRA remain separate; selecting no-LoRA also disables unused group text", noLoras.mode === "no_loras" && noLoras.members.length === 0 && noLoras.test_targets.length === 0 && noLoras.includePresetText === false, noLoras);

  await page.choose("X 轴参数", "LoRA 实例权重");
  const targetLabel = await page.eval("document.querySelector('button[aria-label=\"X LoRA 目标实例\"]')?.textContent");
  await page.fill('[data-axis-value="x:0"] input[type="number"]', "0");
  const zeroWeight = await page.eval("({weight:window.__fixtureInspect().draft.axes.x.values[0].value,target:window.__fixtureInspect().draft.axes.x.target,button:document.querySelector('button[aria-label=\"X LoRA 目标实例\"]')?.textContent})");
  record("LoRA identity selector shows a human name, and weight zero stays a typed zero", targetLabel.includes("Portrait LoRA") && zeroWeight.weight === "0" && zeroWeight.target === "11111111-1111-4111-8111-111111111111", zeroWeight);

  const summaryControl = await page.eval("document.querySelector('.xyz-execution-settings')?.open === false && Boolean(document.querySelector('.xyz-repeat-control'))");
  await page.fill('.xyz-repeat-control input', "2");
  await page.click("document.querySelector('.xyz-execution-settings summary')");
  await page.fill('.xyz-execution-settings input[type="number"]', "3");
  const repeatConcurrency = await page.eval("({repeat:window.__fixtureInspect().draft.repeat_count,concurrency:window.__fixtureInspect().draft.requested_concurrency,count:document.querySelector('.xyz-run-bar strong')?.textContent})");
  record("repeat count and manual server-limited concurrency have separate labels/state and do not multiply together", summaryControl && repeatConcurrency.repeat === 2 && repeatConcurrency.concurrency === 3 && repeatConcurrency.count.includes("4 张") && !repeatConcurrency.count.includes("12 张"), repeatConcurrency);

  await page.eval(`window.__fixturePatchDraft("Flux2", {axes:{x:null,y:{field:"negative_prompt",values:[{value:"preserve unsupported" ,label:"future"}],range:null},z:null}})`);
  await page.eval(`(() => {const el=document.querySelector('select[aria-label="夹具引擎"]'); const setter=Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set; setter.call(el,'Flux2'); el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await page.waitFor("window.__fixtureInspect().engine === 'Flux2' && Boolean(document.querySelector('.xyz-axis-unsupported'))", { label: "switch to unsupported-axis engine" });
  const unsupported = await page.eval("({value:window.__fixtureInspect().draft.axes.y.values[0].value,notice:document.querySelector('.xyz-unsupported-note')?.textContent,field:document.querySelector('button[aria-label=\"Y 轴参数\"]')?.textContent})");
  record("engine switch hides unsupported negative-prompt capability but retains the saved raw axis", unsupported.value === "preserve unsupported" && unsupported.notice.includes("原始定义仍保留") && unsupported.field.includes("不支持"), unsupported);
  await page.shot('state-unsupported.png');
  await page.eval(`(() => {const el=document.querySelector('select[aria-label="夹具引擎"]'); const setter=Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set; setter.call(el,'SD'); el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  await page.waitFor("window.__fixtureInspect().engine === 'SD' && Boolean(document.querySelector('.xyz-workspace'))", { label: "restore SD draft" });
  const engineRestore = await page.eval("({weight:window.__fixtureInspect().draft.axes.x.values[0].value,repeat:window.__fixtureInspect().draft.repeat_count,basic:window.__fixtureInspect().basic})");
  record("engine switch restores the independent SD draft and does not mutate Basic", engineRestore.weight === "0" && engineRestore.repeat === 2 && JSON.stringify(engineRestore.basic) === JSON.stringify(basicBefore), engineRestore);

  await page.clickText(".xyz-head-actions", "同步基础基线");
  const syncDialog = await page.eval("({open:Boolean(document.querySelector('#xyz-sync-title')),button:document.activeElement?.textContent})");
  await page.key("Escape");
  const syncClosed = await page.eval("!document.querySelector('#xyz-sync-title') && document.activeElement?.textContent.includes('同步基础基线')");
  record("explicit Sync opens a selected-field dialog and Escape restores focus without applying", Boolean(syncDialog.open) && syncClosed, { syncDialog, syncClosed });

  await page.clickText(".xyz-run-actions", "服务端预检");
  await page.waitFor("Boolean(document.querySelector('.xyz-plan-summary'))", { label: "successful fixture preflight" });
  const capturedPlan = await page.eval("window.__xyzPlanBodies.at(-1)");
  record("typed form reaches the current plan transport as a draft-only payload; UI state is not nested into strict fields", capturedPlan?.axes?.x?.values?.[0]?.value === "0" && !JSON.stringify(capturedPlan).includes("invalid_raw") && !JSON.stringify(capturedPlan).includes("editorState"), { xField: capturedPlan?.axes?.x?.field, xValue: capturedPlan?.axes?.x?.values?.[0]?.value, keys: Object.keys(capturedPlan || {}) });
  const oldPlanSummary = await page.eval("document.querySelector('.xyz-plan-summary')?.textContent");
  await page.eval("window.__holdPlan=true");
  await page.clickText(".xyz-run-actions", "服务端预检");
  await page.waitFor("window.__releasePlan && window.__xyzPlanBodies.length === 2", { label: "held plan request" });
  await page.shot('state-plan-busy.png');
  await page.fill('[data-axis-value="x:0"] input[type="number"]', "0.5");
  await page.waitFor("document.querySelector('.xyz-plan-summary')?.textContent.includes('草稿已修改，结果已过期')", { label: "stale plan indicator" });
  await page.eval("window.__releasePlan(); window.__holdPlan=false");
  await sleep(350);
  const stalePlan = await page.eval("document.querySelector('.xyz-plan-summary')?.textContent");
  record("late plan response after an axis edit is ignored; only the prior preview remains marked stale", stalePlan.includes("草稿已修改，结果已过期") && stalePlan === oldPlanSummary.replace("服务端预检", "服务端预检 · 草稿已修改，结果已过期"), { old: oldPlanSummary, current: stalePlan });
  await page.shot('state-dirty.png');

  await page.eval("window.__holdSubmit=true");
  await page.clickText(".xyz-run-actions", "开始 XYZ");
  await page.waitFor("window.__releaseSubmit && window.__xyzSubmitBodies?.length === 1", { label: "held submit request" });
  await page.shot('state-submitting.png');
  await page.clickText(".xyz-run-actions", "正在提交").catch(() => {});
  const afterRapidClick = await page.eval("window.__xyzSubmitBodies.length");
  await page.fill('[data-axis-value="x:0"] input[type="number"]', "0.75");
  await page.eval("window.__releaseSubmit(); window.__holdSubmit=false");
  await page.waitFor("Boolean(document.querySelector('.xyz-submit-notice'))", { label: "late submission notice" });
  const lateSubmit = await page.eval("({calls:window.__xyzSubmitBodies.length,tab:document.querySelector('.xyz-tabs button[aria-pressed=true]')?.textContent,notice:document.querySelector('.xyz-submit-notice')?.textContent,draft:window.__fixtureInspect().draft.axes.x.values[0].value})");
  record("late submit cannot double-post or replace the edited draft; it offers explicit navigation to its frozen result", afterRapidClick === 1 && lateSubmit.calls === 1 && lateSubmit.tab.includes("配置") && lateSubmit.draft === "0.75" && lateSubmit.notice.includes("较早版本草稿"), { afterRapidClick, ...lateSubmit });

  await page.eval("window.__networkFailSubmit=2");
  await page.clickText(".xyz-run-actions", "开始 XYZ");
  await page.waitFor("Boolean(document.querySelector('.xyz-submit-error')?.textContent.includes('fixture network disconnect'))", { label: "first transport failure" });
  await page.clickText(".xyz-run-actions", "开始 XYZ");
  await page.waitFor("window.__xyzSubmitBodies.length === 3 && Boolean(document.querySelector('.xyz-submit-error')?.textContent.includes('fixture network disconnect'))", { label: "second transport failure" });
  const retryIds = await page.eval("window.__xyzSubmitBodies.slice(1,3).map((item)=>[item.submission_id,item.on_error])");
  if (!(await page.eval("document.querySelector('.xyz-execution-settings')?.open"))) await page.click("document.querySelector('.xyz-execution-settings summary')");
  await page.choose("XYZ 单格失败策略", "记录失败并继续");
  await page.clickText(".xyz-run-actions", "开始 XYZ");
  try { await page.waitFor("Boolean(document.querySelector('.xyz-results-context'))", { timeout: 5000, label: "results after changed policy" }); }
  catch (failure) {
    const debug = await page.eval("({tab:document.querySelector('.xyz-tabs')?.textContent,selected:document.querySelector('.xyz-tabs button[aria-pressed=true]')?.textContent,notice:document.querySelector('.xyz-submit-notice')?.textContent,error:document.querySelector('.xyz-submit-error')?.textContent,button:[...document.querySelectorAll('.xyz-run-actions button')].map((item)=>[item.textContent,item.disabled]),draft:window.__fixtureInspect().draft.axes.x,policies:window.__xyzSubmitBodies?.map((item)=>[item.submission_id,item.on_error]),calls:window.__xyzCalls.slice(-8)})");
    await writeFile(path.join(EVIDENCE, "changed-policy-debug.json"), JSON.stringify(debug, null, 2));
    console.error(`changed-policy debug ${JSON.stringify(debug)}`);
    throw failure;
  }
  const policyIds = await page.eval("window.__xyzSubmitBodies.slice(1,4).map((item)=>[item.submission_id,item.on_error])");
  record("identical transport retry reuses its id, but changing onError uses a new id and wire policy", retryIds[0][0] === retryIds[1][0] && policyIds[2][0] !== retryIds[1][0] && policyIds[2][1] === "continue", { retryIds, policyIds });

  await page.clickText(".xyz-tabs", "结果");
  await page.waitFor("Boolean(document.querySelector('.xyz-experiment-pick'))", { label: "results list" });
  const historicalLabel = await page.eval("new Date(1791234567*1000).toLocaleString()");
  await page.choose("选择 XYZ 实验", historicalLabel);
  await page.waitFor("document.querySelector('.xyz-experiment-head strong')?.textContent.includes('已暂停')", { label: "historical paused experiment" });
  await page.shot('state-paused-partial.png');
  const resume = await page.eval("[...document.querySelectorAll('.xyz-result-run-bar button')].find((item)=>item.textContent.includes('继续生成'))?.textContent");
  record("old results stay independent and a paused snapshot exposes an explicit resume control", Boolean(resume) && (await page.eval("window.__fixtureInspect().draft.axes.x.values[0].value")) === "0.75", { resume });
  await page.clickText(".xyz-result-run-bar", "继续生成");
  await page.waitFor("document.querySelector('.xyz-experiment-head strong')?.textContent.includes('生成中')", { label: "resume reflected in snapshot" });
  const resumeCall = await page.eval("window.__xyzCalls.findLast((call)=>call.url.endsWith('/resume'))?.url");
  record("resume acts on the selected historical experiment and refreshes its server snapshot", resumeCall?.endsWith("/old-exp/resume"), { resumeCall });

  await page.choose("选择 XYZ 实验", new Date(1792234567 * 1000).toLocaleString());
  await page.waitFor("document.querySelectorAll('.xyz-cell-actions').length === 2", { label: "grid first page actions" });
  const gridPager = await page.eval("document.querySelector('button[aria-label=\"选择 Z 子网格与重复样本\"]')?.textContent");
  record("results keep Z/sample paging and display fixture originals through the normal grid actions", Boolean(gridPager) && await page.eval("document.querySelectorAll('.xyz-grid th').length >= 3"), { gridPager });
  await page.choose("选择 Z 子网格与重复样本", "Z2");
  await page.waitFor("document.querySelector('button[aria-label=\"选择 Z 子网格与重复样本\"]')?.textContent.includes('Z2')", { label: "Z2/sample page selected" });
  const pageLabel = await page.eval("document.querySelector('button[aria-label=\"选择 Z 子网格与重复样本\"]')?.textContent");
  record("Z/sample picker changes the addressed page without modifying the experiment", pageLabel.includes("Z2"), { pageLabel });

  await page.click("document.querySelector('.xyz-cell-open')");
  const viewer = await page.eval("window.__viewerSource");
  record("Viewer receives the completed cell's original asset reference", viewer?.assetId?.startsWith("fixture-asset-"), viewer || {});
  await page.click("document.querySelector('button[aria-label$=\"加入画廊\"]')");
  await page.waitFor("Boolean(window.__gallerySource)", { label: "cell Gallery callback" });
  const gallery = await page.eval("({n:window.__gallerySource.outputs.length,settings:window.__gallerySource.settings})");
  record("Gallery receives exactly one cell using its frozen snapshot", gallery.n === 1 && gallery.settings !== undefined, { count: gallery.n });
  const applyTriggerLabel = await page.eval("[...document.querySelectorAll('.xyz-cell-actions button')].find((item)=>item.getAttribute('aria-label')?.endsWith('的参数应用到基础生图'))?.getAttribute('aria-label')");
  await page.click("document.querySelector('button[aria-label$=\"的参数应用到基础生图\"]')");
  await page.waitFor("Boolean(document.querySelector('.xyz-apply-dialog'))", { label: "apply dialog" });
  const applyStart = await page.eval("({groups:[...document.querySelectorAll('.xyz-apply-groups button')].map((button)=>[button.querySelector('strong')?.textContent,button.getAttribute('aria-checked')]),focus:document.activeElement?.getAttribute('aria-label')})");
  const applyWasCalled = await page.eval("Boolean(window.__applySource)");
  record("Apply modal begins with prompt only selected and does not invoke the Basic callback", applyStart.groups.length === 7 && applyStart.groups.filter(([,checked])=>checked === "true").length === 1 && applyStart.groups.find(([name])=>name === "提示词")?.[1] === "true" && !applyWasCalled, { ...applyStart, applyWasCalled });
  await page.key("Escape");
  const applyFocus = await page.eval("document.activeElement?.getAttribute('aria-label')");
  const applyClosed = await page.eval("!document.querySelector('.xyz-apply-dialog')");
  record("Escape closes Apply without applying and restores focus to its originating cell action", applyClosed && applyFocus === applyTriggerLabel, { closed: applyClosed, applyTriggerLabel, applyFocus });

  await page.clickText(".xyz-grid-tools", "导出整网格 PNG");
  await page.waitFor("Boolean(document.querySelector('.xyz-export-dialog'))", { label: "grid export dialog" });
  const defaultExport = await page.eval("({scope:document.querySelector('input[name=xyz-export-scope]:checked')?.parentElement?.textContent,mode:document.querySelector('input[name=xyz-export-mode]:checked')?.parentElement?.textContent,all:document.querySelector('input[name=xyz-export-scope]:checked')?.parentElement?.textContent})");
  record("export dialog defaults to whole experiment / one PNG and original-resolution mode", defaultExport.scope?.includes("整个实验") && defaultExport.mode?.includes("原始分辨率"), defaultExport);
  await page.click("document.querySelectorAll('input[name=xyz-export-mode]')[1]");
  await page.waitFor("document.querySelector('.xyz-export-estimate')?.textContent.includes('50%')", { label: "scaled default 50 percent estimate" });
  await page.fill('.xyz-export-percent input[type=number]', "100");
  await page.waitFor("document.querySelector('.xyz-export-estimate')?.textContent.includes('100%')", { label: "scaled 100 percent estimate" });
  const hundred = await page.eval("({percent:document.querySelector('.xyz-export-percent input[type=number]').value,disabled:document.querySelector('.xyz-export-dialog .xyz-actions button.primary')?.disabled,budget:document.querySelector('.xyz-export-estimate')?.textContent})");
  record("scaled 100% remains available but fixture memory budget blocks the export button", hundred.percent === "100" && hundred.disabled && hundred.budget.includes("超出预算"), hundred);
  await page.shot("after-export-budget-1440x900.png");
  await page.fill('.xyz-export-percent input[type=number]', "50");
  await page.waitFor("document.querySelector('.xyz-export-estimate')?.textContent.includes('1024 × 512 px')", { label: "scaled 50 percent estimate" });
  const fifty = await page.eval("({percent:document.querySelector('.xyz-export-percent input[type=number]').value,disabled:document.querySelector('.xyz-export-dialog .xyz-actions button.primary')?.disabled})");
  record("scaled 50% suggestion yields a valid budget-gated export", fifty.percent === "50" && !fifty.disabled, fifty);
  await page.fill('.xyz-export-percent input[type=number]', "0");
  const invalidPercent = await page.eval("({invalid:document.querySelector('.xyz-export-percent input[type=number]').getAttribute('aria-invalid'),alert:document.querySelector('.xyz-export-estimate [role=alert]')?.textContent,disabled:document.querySelector('.xyz-export-dialog .xyz-actions button.primary')?.disabled})");
  record("invalid manual percent is explicit and never reuses stale budget as a submit authorization", invalidPercent.invalid === "true" && invalidPercent.alert?.includes("1–100") && invalidPercent.disabled, invalidPercent);
  await page.fill('.xyz-export-percent input[type=number]', "50");
  await page.waitFor("document.querySelector('.xyz-export-estimate')?.textContent.includes('1024 × 512 px')", { label: "restore a valid scaled estimate" });
  await page.shot("after-export-valid-50-percent-1440x900.png");
  await page.click("document.querySelector('.xyz-export-dialog .xyz-actions button.primary')");
  await page.waitFor("Boolean(document.querySelector('.xyz-export-progress')?.textContent.includes('已完成'))", { label: "fixture export complete" });
  const exportRecord = await page.eval("({body:window.__xyzCalls.findLast((call)=>call.url.endsWith('/exports')&&call.method==='POST')?.body,links:[...document.querySelectorAll('.xyz-export-progress a')].map((item)=>item.textContent)})");
  record("valid scaled whole-grid export keeps one PNG plus the JSON manifest links", exportRecord.body?.scope === "all" && exportRecord.body.mode === "scaled" && exportRecord.body.percent === 50 && exportRecord.links.some((item)=>item.includes("JSON")) && exportRecord.links.length >= 2, exportRecord);
  await page.key("Escape");
  const exportFocus = await page.eval("document.activeElement?.textContent");
  record("export dialog Escape restores focus to its button and leaves the old experiment grid mounted", await page.eval("!document.querySelector('.xyz-export-dialog') && Boolean(document.querySelector('.xyz-grid'))") && exportFocus.includes("导出整网格 PNG"), { exportFocus });

  const finalLayout = await page.eval("({width:innerWidth,body:document.body.scrollWidth,root:document.documentElement.scrollWidth,errors:window.__xyzHostErrors,consoleErrors:window.__xyzConsoleErrors || []})");
  record("browser fixture ends without runtime exceptions or horizontal page overflow", finalLayout.width >= finalLayout.body && finalLayout.width >= finalLayout.root && finalLayout.errors.length === 0, finalLayout);

  for (const [width,height] of [[1920,1080],[1440,900],[1366,768],[390,844]]) {
    await page.viewport(width,height); await page.shot(`after-results-${width}x${height}.png`);
    const resultLayout = await page.eval(`(() => { const r=document.querySelector('.xyz-result-run-bar').getBoundingClientRect(), g=document.querySelector('.xyz-grid-scroll').getBoundingClientRect(); return {body:document.body.scrollWidth,width:innerWidth,bar:[r.x,r.y,r.width,r.height],grid:[g.x,g.y,g.width,g.height],gridOverflow:getComputedStyle(document.querySelector('.xyz-grid-scroll')).overflow,noStart:![...document.querySelectorAll('.xyz-workspace button')].some(b=>b.textContent==='开始 XYZ'),frozen:document.querySelector('.xyz-frozen-summary').textContent}; })()`);
    record(`${width} result controls selected experiment, grid scrolls independently, no draft start`, resultLayout.body<=width && resultLayout.noStart && resultLayout.gridOverflow==='auto' && resultLayout.frozen.includes('frozen-old-model'), resultLayout);
  }

  // Mount the real App with in-memory endpoints. Every fetch is intercepted; no formal service,
  // model, cache, Gallery or user-state request can escape this fixture.
  await page.viewport(1440,900);
  await page.send('Page.navigate',{url:`http://127.0.0.1:${address.port}/scripts/xyz-sol-app.fixture.html`});
  await page.waitFor("Boolean(document.querySelector('.generation-mode-switch')) && !document.querySelector('.app-loader')",{label:'real App fixture restored Basic'});
  await sleep(900);
  const basicSnapshot = await page.eval(`(() => {const w=window.__fixtureAppState().workspace;return {model:w.model,engineSettings:w.engineSettingsByEngine,mounts:w.mountedLorasByEngine,groups:w.loraGroupsByEngine,presets:w.promptPresets,active:w.activePromptPresets,canvas:w.canvasMaxEdge,output:w.outputTemplate};})()`);
  await writeFile(path.join(EVIDENCE,'basic-before.json'),JSON.stringify(basicSnapshot,null,2));
  await page.choose('生成模式','XYZ');
  await page.waitFor("Boolean(document.querySelector('.xyz-sidebar-fields')) && !document.querySelector('.xyz-sidebar-fields [role=status]')",{label:'real XYZ sidebar and catalogs'});
  await page.choose('XYZ 底模选择','SD checkpoint other');
  await page.choose('XYZ 添加背景 LoRA','SD Portrait 1');
  await page.click('document.querySelectorAll(".xyz-sidebar-lora summary")[1]');
  await page.fill('[aria-label="XYZ LoRA 2 权重"]','-0.1250'); await page.key('Tab');
   await page.click(clickExpression('[aria-label="XYZ LoRA 2 上移"]'));
  await page.fill('[aria-label="XYZ 基线步数"]','36');
  await page.fill('[aria-label="XYZ 基线CFG"]','5.25');
  await page.fill('[aria-label="XYZ 基线种子"]','18446744073709551615');
  await page.fill('[aria-label="XYZ 基线正向提示词"]','XYZ exact, long prompt — '.repeat(12));
  await page.click('document.querySelector(".xyz-sidebar-canvas summary")');
  await page.clickText('.xyz-sidebar-canvas .size-presets','9:16');
  const box = await page.rect('document.querySelector(".xyz-sidebar .size-grid")');
  // Surface pointer event + capture; move while held, release at exact generation size.
  const target={x:box.x-box.w/2+box.w*1024/2048,y:box.y+box.h/2-box.h*1408/2048};
  await page.send('Input.dispatchMouseEvent',{type:'mousePressed',...target,button:'left',clickCount:1});
  await page.send('Input.dispatchMouseEvent',{type:'mouseMoved',x:target.x+box.w/32,y:target.y,button:'left',buttons:1});
  await page.send('Input.dispatchMouseEvent',{type:'mouseMoved',...target,button:'left',buttons:1});
  await page.send('Input.dispatchMouseEvent',{type:'mouseReleased',...target,button:'left',clickCount:1});
  await page.click('document.querySelector(".xyz-sidebar .grid-handle-hitbox")');
  await page.key('ArrowUp'); await page.key('ArrowDown');
  await page.waitFor("document.querySelector('[aria-label=\"XYZ 生成高度\"]')?.value==='1408'",{label:'grid keyboard original height 1408'});
  await page.shot('app-canvas-drag-keyboard-1440x900.png');
  await page.choose('X 轴参数','步数'); await page.clickText('.xyz-values','添加独立项');
  await page.fill('[aria-label="X 第 2 项"]','24');
  await page.choose('Y 轴参数','CFG'); await page.clickText('.xyz-axis-card:nth-child(2) .xyz-values','添加独立项');
  await page.fill('[aria-label="Y 第 2 项"]','7');
  await page.clickText('.xyz-run-actions','服务端预检');
  await page.waitFor('window.__xyzPlanBodies?.length===1',{label:'true sidebar current preflight body'});
  const currentPlan = await page.eval('window.__xyzPlanBodies.at(-1)');
  record('real App sidebar edits reach XYZ current baseline with SizeGrid 1024x1408 and strong ordered mounts',currentPlan.baseline.parameters.width===1024 && currentPlan.baseline.parameters.height===1408 && currentPlan.baseline.parameters.checkpoint==='SD-checkpoint-other.safetensors' && currentPlan.baseline.parameters.steps===36 && currentPlan.baseline.parameters.cfg===5.25 && currentPlan.baseline.parameters.seed==='18446744073709551615' && currentPlan.baseline.loras[0].weight==='-0.1250' && currentPlan.baseline.loras[0].contentRevision===1,{baseline:currentPlan.baseline});
  await writeFile(path.join(EVIDENCE,'sidebar-plan-body.json'),JSON.stringify(currentPlan,null,2));
  // Confirm that field replacement is not an implicit destructive onChange.
  await page.click(clickExpression('button[aria-label="X 轴参数"]'));
  await page.click('([...document.querySelectorAll("[role=option]")].find(e=>e.textContent.includes("正向文本")))');
  const prior=await page.eval('window.__fixtureAppState().workspace.xyzByEngine.by_engine.SD.axes.x');
  await page.key('Escape');
  record('field replacement Escape preserves exact ordered values and returns focus',prior.field==='steps' && await page.eval('document.activeElement.getAttribute("aria-label")==="X 轴参数"') && !await page.eval('Boolean(document.querySelector("#xyz-axis-replace-title"))'),prior);
  await page.choose('X 轴参数','正向文本');
  await page.choose('X 轴参数','步数');
  await sleep(700);
  const restoredAxis = await page.eval('window.__fixtureAppState().workspace.xyzByEngine.by_engine.SD.axes.x');
  record('confirmed field switch then select-back restores previous ordered values without UI wire fields',JSON.stringify(restoredAxis)===JSON.stringify(prior),{restoredAxis});
  await page.choose('X 轴参数','LoRA 快照组合倍率');
  await page.fill('[aria-label="X 第 1 项"]','1');
  await page.clickText('.xyz-run-actions','服务端预检');
  await page.waitFor('window.__xyzPlanBodies.length===2',{label:'multiplier wire preflight'});
  const multiplierBody=await page.eval('window.__xyzPlanBodies.at(-1)');
  record('typed multiplier keeps snapshot only on axis, not strict AxisValue entries',Boolean(multiplierBody.axes.x.snapshot)&&Object.keys(multiplierBody.axes.x.values[0]).every(k=>['value','label'].includes(k)),{axisKeys:Object.keys(multiplierBody.axes.x),entryKeys:Object.keys(multiplierBody.axes.x.values[0])});
  await writeFile(path.join(EVIDENCE,'multiplier-plan-body.json'),JSON.stringify(multiplierBody,null,2));
  await page.choose('X 轴参数','步数');
  await page.eval('window.__holdSubmit=true');
  await page.clickText('.xyz-run-actions','开始 XYZ');
  await page.waitFor('Boolean(window.__releaseSubmit)',{label:'held size submission'});
  await page.fill('[aria-label="XYZ 生成宽度"]','512');
  await page.eval('window.__releaseSubmit();window.__holdSubmit=false');
  await page.waitFor('Boolean(window.__fixtureSubmitted)',{label:'frozen fixture detail'});
  const frozen = await page.eval('window.__fixtureSubmitted');
  record('in-flight canvas edit does not modify submitted 2x2 frozen cell dimensions (fixture metadata, not GPU PNG)',frozen.cells.length===4 && frozen.cells.every(c=>c.width===1024&&c.height===1408) && await page.eval(`document.querySelector('[aria-label="XYZ 生成宽度"]').value==="512"`),{cellCount:frozen.cells.length,dimensions:frozen.cells.map(c=>[c.width,c.height])});
  await writeFile(path.join(EVIDENCE,'sidebar-submit-body.json'),JSON.stringify(await page.eval('window.__xyzSubmitBodies.at(-1)'),null,2));
  await page.clickText('.xyz-submit-notice','查看该实验');
  await page.waitFor('Boolean(document.querySelector(".xyz-grid-size"))',{label:'frozen submitted result'});
  await page.choose('网格缩略图大小','大');
  record('screen cell size never writes generation baseline or frozen original dimensions',await page.eval('window.__fixtureSubmitted.cells.every(c=>c.width===1024&&c.height===1408)') && await page.eval(`document.querySelector('[aria-label="XYZ 生成宽度"]').value==="512"`),{});
  await page.clickText('.xyz-grid-tools','导出整网格 PNG');
  await page.click('document.querySelectorAll("input[name=xyz-export-mode]")[1]');
  await page.waitFor('document.querySelector(".xyz-export-estimate")?.textContent.includes("50%")');
  record('real App export scaling stays separate from original canvas and frozen cells',await page.eval(`document.querySelector('[aria-label="XYZ 生成宽度"]').value==='512' && window.__fixtureSubmitted.cells.every(c=>c.width===1024&&c.height===1408)`),{});
  await page.key('Escape');
  await page.clickText('.xyz-tabs','配置');
  await page.click('document.querySelector(".xyz-execution-settings summary")');
  await page.click('document.querySelectorAll(".xyz-execution-settings .xyz-check input")[1]');
  record('postprocess opt-in explains original vs final size without changing the base canvas',await page.eval('document.querySelector(".xyz-baseline-summary").textContent.includes("最终尺寸可能改变")') && await page.eval(`document.querySelector('[aria-label="XYZ 生成高度"]').value==='1408'`),{});
  await page.click('document.querySelectorAll(".xyz-execution-settings .xyz-check input")[1]');
  // Actual six-engine sidebar switches never invoke the Basic/i2i engine swap.
  for(const engine of ['iL','Anima','Flux','Flux2','Krea2','SD']) {
    await page.click(`([...document.querySelectorAll('.model-seg')].find(e=>e.querySelector('.seg-name')?.textContent===${JSON.stringify(engine)}))`);
    await page.waitFor(`document.querySelector('.xyz-workspace-title')?.textContent.includes(${JSON.stringify(engine)}) && !document.querySelector('.xyz-sidebar-models [role=status]')`,{label:`XYZ ${engine} draft/catalog`});
    await page.fill('[aria-label="XYZ 基线种子"]',engine==='SD'?'18446744073709551615':'7');
  }
  await sleep(700);
  await page.choose('生成模式','基础生图'); await sleep(700);
  const basicAfter = await page.eval(`(() => {const w=window.__fixtureAppState().workspace;return {model:w.model,engineSettings:w.engineSettingsByEngine,mounts:w.mountedLorasByEngine,groups:w.loraGroupsByEngine,presets:w.promptPresets,active:w.activePromptPresets,canvas:w.canvasMaxEdge,output:w.outputTemplate};})()`);
  record('real App Basic multi-engine / i2i / mounts / groups / presets / canvas / output are byte-identical after XYZ roundtrip',JSON.stringify(basicSnapshot)===JSON.stringify(basicAfter),{same:JSON.stringify(basicSnapshot)===JSON.stringify(basicAfter),basicSize:basicAfter.engineSettings.byEngine.SD.size});
  await writeFile(path.join(EVIDENCE,'basic-after.json'),JSON.stringify(basicAfter,null,2));
  await page.choose('生成模式','XYZ'); await page.waitFor('Boolean(document.querySelector(".xyz-sidebar-fields"))');
  record('mode roundtrip restores corresponding XYZ draft, not Basic values',await page.eval(`document.querySelector('[aria-label="XYZ 生成宽度"]').value==="512" && document.querySelector('[aria-label="XYZ 生成高度"]').value==="1408"`),{});
  if(!await page.eval('document.querySelector(".xyz-sidebar-canvas").open')) await page.click('document.querySelector(".xyz-sidebar-canvas summary")');
  await page.fill('[aria-label="XYZ 生成宽度"]','1024');
  await page.clickText('.xyz-run-actions','服务端预检');
  for(const [width,height] of [[1920,1080],[1440,900],[1366,768],[390,844]]) {
    await page.viewport(width,height); await page.shot(`app-config-${width}x${height}.png`);
    const rects=await page.eval(`(() => {const b=document.querySelector('.xyz-run-bar').getBoundingClientRect(),s=document.querySelector('.xyz-workspace-scroll').getBoundingClientRect();return {body:document.body.scrollWidth,width:innerWidth,bar:[b.x,b.y,b.width,b.height],scroll:[s.x,s.y,s.width,s.height],sidebarOverflow:getComputedStyle(document.querySelector('.xyz-sidebar .panel-scroll')).overflowY,overlap:b.top<s.bottom-1};})()`);
    record(`${width} real App populated config body / bar rect and source scroll owner`,rects.body<=width&&!rects.overlap,rects);
  }
  await page.viewport(390,430); await page.shot('app-softkeyboard-390x430.png');
  const keyboard=await page.eval('({width:innerWidth,body:document.body.scrollWidth,pageOverflow:getComputedStyle(document.querySelector(".workspace.xyz-mode")).overflowY,barPosition:getComputedStyle(document.querySelector(".xyz-run-bar")).position})');
  record('softkeyboard viewport makes page scroll rather than overlaying controls',keyboard.width>=keyboard.body&&keyboard.pageOverflow==='auto'&&keyboard.barPosition==='static',keyboard);
  await page.viewport(1440,900);
  await page.click(`([...document.querySelectorAll('.model-seg')].find(e=>e.querySelector('.seg-name')?.textContent==='Flux'))`);
  await page.waitFor("document.querySelector('.xyz-workspace-title')?.textContent.includes('Flux')");
  await page.choose('生成模式','基础生图');
  await page.eval('window.__fixtureQueueParent=true');
  await page.clickText('.queue-control','队列');
  await page.waitFor('Boolean(document.querySelector(".queue-item.xyz"))',{timeout:15000,label:'one XYZ queue parent'});
  await page.click('document.querySelector(".queue-item.xyz .queue-item-main")');
  await page.waitFor("document.querySelector('.xyz-workspace-title')?.textContent.includes('SD') && document.querySelector('.xyz-frozen-summary')?.textContent.includes('old-exp')",{label:'Basic SD queue opens SD frozen experiment, not remembered XYZ Flux'});
  record('XYZ parent queue navigation resolves same viewed engine rather than remembered draft engine',await page.eval("document.querySelector('.xyz-workspace-title').textContent.includes('SD') && window.__fixtureAppState().workspace.model==='SD'"),{});
  record('actual App fixture makes no ordinary generation/unload/cancel request',await page.eval('!window.__fixtureAppCalls.some(c=>(c.url==="/api/inference/jobs"&&c.method==="POST")||c.url.includes("model-cache")||c.url.includes("/cancel"))'),{});
  record('actual App fixture has no uncaught runtime exception',page.runtimeErrors.length===0 && await page.eval('window.__xyzHostErrors.length===0'),{errors:page.runtimeErrors});
  await writeFile(path.join(EVIDENCE, "ledger.json"), JSON.stringify(ledger, null, 2));
  await writeFile(path.join(EVIDENCE, "source-sha256.txt"), hashes.join("\n") + "\n");
} catch (error) {
  await writeFile(path.join(EVIDENCE, "failure.txt"), `${error.stack || error}\n`);
  await writeFile(path.join(EVIDENCE, "ledger.json"), JSON.stringify(ledger, null, 2));
  throw error;
} finally {
  try { page?.ws.close(); } catch {}
  if (edge && edge.exitCode === null) {
    edge.kill();
    await Promise.race([new Promise((resolve) => edge.once("exit", resolve)), sleep(2500)]);
  }
  await vite.close();
}

console.log(`UI_BROWSER_EVIDENCE=${EVIDENCE}`);
