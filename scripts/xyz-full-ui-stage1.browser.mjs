import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import net from "node:net";
import path from "node:path";
import os from "node:os";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";

const ROOT=path.resolve(import.meta.dirname,".."), EVIDENCE=path.join(os.tmpdir(),"opencode",`xyz-full-ui-stage1-browser-${Date.now()}-${process.pid}`);
await mkdir(path.join(EVIDENCE,"screenshots"),{recursive:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
class Page {
 constructor(ws){this.ws=ws;this.next=0;this.pending=new Map();this.errors=[];ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=this.pending.get(m.id);if(p){this.pending.delete(m.id);m.error?p.reject(new Error(m.error.message)):p.resolve(m.result);}}else if(m.method==="Runtime.exceptionThrown")this.errors.push(m.params.exceptionDetails?.exception?.description||m.params.exceptionDetails?.text);};}
 send(method,params={}){const id=++this.next;this.ws.send(JSON.stringify({id,method,params}));return new Promise((resolve,reject)=>this.pending.set(id,{resolve,reject}));}
 async eval(expression){const r=await this.send("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;}
 async wait(expression){for(let i=0;i<120;i++){if(await this.eval(expression))return;await sleep(100);}throw new Error(`Timeout: ${expression}`);}
 async viewport(width,height){await this.send("Emulation.setDeviceMetricsOverride",{width,height,deviceScaleFactor:1,mobile:width<500});await this.send("Input.dispatchMouseEvent",{type:"mouseMoved",x:4,y:4});await sleep(450);}
 async click(expression){const r=await this.eval(`(()=>{const e=${expression};if(!e)throw new Error('Missing element');e.scrollIntoView({block:'nearest'});const r=e.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2,hit=document.elementFromPoint(x,y);if(!hit||!e.contains(hit))throw new Error('Covered control: '+e.className+' by '+hit?.className);return {x,y,w:r.width,h:r.height};})()`);assert.ok(r.w>0&&r.h>0,expression);for(const type of ["mouseMoved","mousePressed","mouseReleased"])await this.send("Input.dispatchMouseEvent",{type,x:r.x,y:r.y,button:"left",clickCount:1});await sleep(150);}
 async key(key,modifiers=0){const codes={Escape:27,Tab:9,Enter:13,a:65};for(const type of ["rawKeyDown","keyUp"])await this.send("Input.dispatchKeyEvent",{type,key,code:key.length===1?`Key${key.toUpperCase()}`:key,windowsVirtualKeyCode:codes[key],modifiers});await sleep(100);}
 async fill(selector,value){await this.click(`document.querySelector(${JSON.stringify(selector)})`);await this.key("a",2);await this.send("Input.insertText",{text:String(value)});await sleep(150);}
 async shot(name){const {data}=await this.send("Page.captureScreenshot",{format:"png"});await writeFile(path.join(EVIDENCE,"screenshots",name+".png"),Buffer.from(data,"base64"));}
 async choose(label,text){await this.click(`document.querySelector(${JSON.stringify(`button[aria-label="${label}"]`)})`);await this.click(`([...document.querySelectorAll('[role=option]')].find(e=>e.textContent.includes(${JSON.stringify(text)})&&!e.disabled))`);}
}
const byText=(selector,text)=>`([...document.querySelectorAll(${JSON.stringify(selector)})].find(e=>e.textContent.includes(${JSON.stringify(text)})))`;
const managerMetric=`(()=>{const keys=['.lora-modal','.lora-modal-head','.lora-modal-actions','.lora-repair-entry','.lora-engine-bar','.lora-path-field','.lora-modal-body','.lora-categories','.lora-browser','.lora-browser-head','.lora-mount-toolbar','.lora-mounted-item','.mounted-item-head','.mounted-weight-field','.lora-weight-control','.lora-modal-foot'];const n=v=>Math.round(v*100)/100;const anchor=document.querySelector('.lora-modal').getBoundingClientRect();return Object.fromEntries(keys.map(k=>{const e=document.querySelector(k);if(!e)return[k,null];const c=getComputedStyle(e),r=e.getBoundingClientRect();return[k,{tag:e.tagName,rect:[n(r.x-anchor.x),n(r.y-anchor.y),n(r.width),n(r.height)],display:c.display,padding:c.padding,gap:c.gap,background:c.backgroundColor,border:c.border,borderRadius:c.borderRadius,font:c.font,gridTemplateRows:c.gridTemplateRows}];}));})()`;
const promptMetric=`(()=>{const root=document.querySelector('.i2i-prompt-deck');return ['.prompt-field','.prompt-field > div','.prompt-field textarea','.field-watermark','.preset-strip'].map(k=>{const e=root.querySelector(k);const c=getComputedStyle(e);return {tag:e.tagName,className:e.getAttribute('class'),padding:c.padding,margin:c.margin,border:c.border,borderRadius:c.borderRadius,font:c.font,background:c.backgroundColor,height:c.height,minHeight:c.minHeight,resize:c.resize};});})()`;
const privateState=`(()=>{const w=window.__fixtureAppState().workspace;return {model:w.model,engines:w.engineSettingsByEngine,mounts:w.mountedLorasByEngine,groups:w.loraGroupsByEngine,active:w.activePromptPresets};})()`;
let edge,page;const ledger={scope:"executor SELF-TEST; real App + fixture APIs; no GPU/service",checks:[]};
const vite=await createServer({configFile:false,root:ROOT,cacheDir:path.join(EVIDENCE,"vite-cache"),appType:"mpa",plugins:[react(),{name:"fixture-images",configureServer(server){server.middlewares.use((req,res,next)=>{if(req.url.startsWith('/api/lora-cards/image')||req.url.startsWith('/api/lora-preview')){res.setHeader('Content-Type','image/png');res.end(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jI4sAAAAASUVORK5CYII=','base64'));}else next();});}}],logLevel:"error",server:{host:"127.0.0.1",port:0,hmr:false,open:false,watch:null},optimizeDeps:{noDiscovery:true,include:["react","react-dom/client","lucide-react"]}});
try {
 await vite.listen();const probe=net.createServer();await new Promise(r=>probe.listen(0,"127.0.0.1",r));const port=probe.address().port;await new Promise(r=>probe.close(r));
 const profile=path.join(EVIDENCE,"edge-profile");await mkdir(profile);
 edge=spawn("C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",["--headless=new",`--remote-debugging-port=${port}`,`--user-data-dir=${profile}`,"--no-first-run","--disable-extensions","--disable-sync","--disable-background-networking","about:blank"],{stdio:"ignore",windowsHide:true});
 for(let i=0;i<80;i++){try{if((await fetch(`http://127.0.0.1:${port}/json/version`)).ok)break;}catch{}await sleep(150);}
 const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();const ws=new WebSocket(targets.find(t=>t.type==="page").webSocketDebuggerUrl);page=new Page(ws);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});await page.send("Page.enable");await page.send("Runtime.enable");
 await page.viewport(1440,900);await page.send("Page.navigate",{url:`http://127.0.0.1:${vite.httpServer.address().port}/scripts/xyz-full-ui-stage1.fixture.html`});
 await page.wait("Boolean(document.querySelector('.generation-mode-switch'))&&!document.querySelector('.app-loader')");
 await page.click(byText('.model-seg','Anima'));await page.wait("document.querySelector('.model-seg.active')?.textContent.includes('Anima')");await sleep(600);
 await page.click("document.querySelector('.lora-title button')");await page.wait("Boolean(document.querySelector('.lora-modal'))");await page.click(byText('.lora-categories button','已挂载'));
 const basic1440=await page.eval(managerMetric);await page.shot('basic-manager-mounted-1440');
 await page.viewport(390,844);const basic390=await page.eval(managerMetric);await page.shot('basic-manager-mounted-390');
 await page.key('Escape');await page.viewport(1440,900);
 // i2i owns a different page but intentionally opens the same App LoRA manager.
 await page.click(byText('.main-nav button','图生图'));await page.wait("Boolean(document.querySelector('.i2i-prompt-deck'))");
 const imagePrompt1440=await page.eval(promptMetric);await page.shot('i2i-prompt-1440');
 await page.viewport(390,844);const imagePrompt390=await page.eval(promptMetric);await page.shot('i2i-prompt-390');await page.viewport(1440,900);
 await page.click(byText('.lora-title button','管理'));await page.wait("Boolean(document.querySelector('.lora-modal'))");await page.click(byText('.lora-categories button','已挂载'));
 assert.deepEqual(await page.eval(managerMetric),basic1440,'i2i/Basic manager DOM geometry and styles');await page.shot('i2i-manager-1440');await page.key('Escape');
 await page.click(byText('.main-nav button','文生图'));await page.wait("Boolean(document.querySelector('.generation-mode-switch'))");await sleep(450);
 const before=await page.eval(privateState);await writeFile(path.join(EVIDENCE,'basic-i2i-groups-before.json'),JSON.stringify(before,null,2));
 await page.choose('生成模式','XYZ');await page.wait("Boolean(document.querySelector('.xyz-prompt-deck'))&&!document.querySelector('.xyz-sidebar-models [role=status]')");await sleep(400);
 assert.equal(await page.eval("document.querySelector('.xyz-sidebar-prompts')"),null);
 assert.deepEqual(await page.eval(promptMetric),imagePrompt1440,'XYZ / i2i prompt field styles at 1440');await page.shot('xyz-prompt-1440');
 await page.viewport(390,844);assert.deepEqual(await page.eval(promptMetric),imagePrompt390,'XYZ / i2i prompt field styles at 390');await page.shot('xyz-prompt-390');await page.viewport(1440,900);
 await page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");await page.wait("Boolean(document.querySelector('.xyz-owned-lora-manager'))");
 assert.deepEqual(await page.eval(managerMetric),basic1440,'XYZ / Basic whole manager DOM geometry and styles at 1440');await page.shot('xyz-manager-mounted-1440');
 await page.click("document.querySelector('.lora-modal > .lora-repair-entry button')");await page.wait("Boolean(document.querySelector('.lora-repair-panel'))");await page.key('Escape');await page.wait("!document.querySelector('.lora-repair-panel')");assert.ok(await page.eval("Boolean(document.querySelector('.xyz-owned-lora-manager'))"),'repair entry opens real top panel; Escape leaves manager intact (no scan/commit)');
 await page.viewport(390,844);assert.deepEqual(await page.eval(managerMetric),basic390,'XYZ / Basic whole manager DOM geometry and styles at 390');await page.shot('xyz-manager-mounted-390');
 await page.viewport(390,430);assert.ok(await page.eval("(()=>{const f=document.querySelector('.lora-modal-foot').getBoundingClientRect(),b=document.querySelector('.lora-browser').getBoundingClientRect();return f.bottom<=innerHeight&&f.top>=b.bottom-1&&b.height>0;})()"),'390x430 footer and body remain reachable');await page.shot('xyz-manager-390x430');
 await page.click("document.querySelector('.mounted-weight-input')");assert.equal(await page.eval("document.activeElement.className"),'mounted-weight-input','short-phone mounted weight is really hit-test reachable after scrolling');await page.shot('xyz-manager-390x430-weight-reachable');await page.key('Tab');await page.viewport(1440,900);
 ledger.checks.push('actual Basic/i2i/XYZ complete manager parity 1440 + 390; short-phone body/footer; prompt field parity 1440 + 390');
 await page.click(byText('.lora-categories button','人物'));await page.wait("document.querySelectorAll('.lora-library-card').length===5");assert.equal(await page.eval("document.querySelector('.lora-library-card strong').textContent"),'Custom Anima portrait');await page.shot('xyz-library-five-cards-1440');
 await page.click("document.querySelector('.lora-card-detail')");await page.wait("Boolean(document.querySelector('.lora-detail-backdrop'))");await page.key('Escape');assert.ok(await page.eval("Boolean(document.querySelector('.xyz-owned-lora-manager'))"),'details Esc closes top only');
 await page.click("document.querySelector('.lora-card-customize')");await page.wait("Boolean(document.querySelector('.lora-card-editor'))");await page.click("document.querySelector('.lora-card-insert')");await page.key('Escape');assert.ok(await page.eval("Boolean(document.querySelector('.xyz-owned-lora-manager'))"));await sleep(500);
 assert.ok(await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.prompt_source.positive.includes('card fixture words')"),'real card-editor insert appends only XYZ raw prompt');assert.deepEqual(await page.eval(privateState),before);
 await page.click(byText('.lora-categories button','其他'));await page.wait("Boolean(document.querySelector('.lora-folder-card'))");await page.click("document.querySelector('.lora-folder-card')");await page.wait("Boolean(document.querySelector('.lora-card-detail'))");await page.shot('xyz-folder-nested');
 await page.click(byText('.lora-categories button','共享'));await page.wait("Boolean(document.querySelector('.lora-modal-shared-folder'))");assert.equal(await page.eval("document.querySelector('.lora-modal-shared-folder button').getAttribute('aria-expanded')"),'false');await page.click("document.querySelector('.lora-modal-shared-folder button')");await page.wait("Boolean(document.querySelector('.lora-library-card'))");await page.shot('xyz-shared-root');
 await page.fill('.lora-path-field input','outside');await page.click("document.querySelector('.lora-path-field button')");await page.wait("document.querySelector('.xyz-owned-lora-manager')?.textContent.includes('outside registered roots')");
 await page.fill('.lora-path-field input','fixture/Anima');await page.click("document.querySelector('.lora-path-field button')");await page.wait("document.querySelector('.lora-categories button.active')?.textContent.includes('人物')");
 await page.click(byText('.lora-categories button','组合'));await page.wait("Boolean(document.querySelector('[aria-label=\"复制 Library Anima 到 XYZ\"]'))");await page.click("document.querySelector('[aria-label=\"复制 Library Anima 到 XYZ\"]')");await page.wait("Boolean(document.querySelector('.xyz-group-copy-dialog'))");await page.key('Escape');assert.ok(await page.eval("Boolean(document.querySelector('.xyz-owned-lora-manager'))"));
 await page.click("document.querySelector('.lora-group-create')");await page.wait("Boolean(document.querySelector('.lora-group-create-menu'))");await page.key('Escape');assert.equal(await page.eval("Boolean(document.querySelector('.lora-group-create-menu'))"),false);assert.ok(await page.eval("Boolean(document.querySelector('.xyz-owned-lora-manager'))"),'group menu Escape consumes only the transient menu');
 await page.click("document.querySelector('[aria-label=\"复制 Library Anima 到 XYZ\"]')");await page.wait("Boolean(document.querySelector('.xyz-group-copy-dialog .primary'))");// P2.1: the group's first member is the already mounted standalone instance, so the explicit action is the replace button.
 await page.click("document.querySelector('.xyz-group-copy-dialog .primary')");await page.wait("document.querySelectorAll('.lora-group-card').length===1");
 await page.click(byText('.lora-categories button','已挂载'));await page.wait("document.querySelectorAll('.lora-mounted-item').length===5");
 await page.fill('[aria-label="Anima Portrait 1 权重"]','-1.2345');await page.key('Tab');await sleep(500);
 assert.equal(await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.loras.find(e=>e.name==='Anima Portrait 1').weight"),-1.2345);
 await page.fill('[aria-label="Anima Portrait 2 权重"]','0');await page.key('Tab');await page.fill('[aria-label="Anima Portrait 3 权重"]','bad');await page.key('Tab');
 assert.equal(await page.eval("document.querySelector('[aria-label=\"Anima Portrait 3 权重\"]').value"),'bad');assert.equal(await page.eval("document.querySelector('[aria-label=\"Anima Portrait 3 权重\"]').getAttribute('aria-invalid')"),'true');
 await page.fill('[aria-label="Anima Portrait 3 权重"]','0.8137');await page.key('Tab');
 await page.click("document.querySelectorAll('.lora-mounted-item .mounted-toggle')[4]");await page.click("document.querySelector('.lora-preview-toggle')");await page.wait("document.querySelectorAll('.mounted-preview img').length===5");await page.shot('xyz-five-mounted-group-previews');
 // A native DragEvent session starts only on the real SVG handle and commits the private MIME.
 const orderBefore=await page.eval("[...document.querySelectorAll('.lora-mounted-item .mounted-item-info strong')].map(e=>e.textContent)");
 await page.eval(`(()=>{const rows=[...document.querySelectorAll('.lora-mounted-item')],transfer=new DataTransfer();rows[4].querySelector('[data-lora-drag-handle] svg').firstElementChild.dispatchEvent(new DragEvent('dragstart',{bubbles:true,dataTransfer:transfer}));rows[0].dispatchEvent(new DragEvent('dragover',{bubbles:true,cancelable:true,dataTransfer:transfer}));rows[0].dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:transfer}));})()`);await sleep(400);
 assert.equal(await page.eval("document.querySelector('.lora-mounted-item .mounted-item-info strong').textContent"),orderBefore[4]);
 await page.click("document.querySelector('.lora-mount-group-off')");await page.wait("document.querySelectorAll('.lora-mounted-item').length===0");await page.click(byText('.lora-categories button','组合'));
 await page.click("document.querySelector('.lora-group-card .mounted-toggle')");await page.click(byText('.lora-categories button','已挂载'));await page.wait("document.querySelectorAll('.lora-mounted-item').length===5");
 await page.click("document.querySelector('.lora-mounted-item .remove')");await page.wait("document.querySelectorAll('.lora-mounted-item').length===4");
 await page.click("document.querySelector('[aria-label=\"最大化 LoRA 管理器窗口\"]')");await page.wait("document.querySelector('.lora-modal').classList.contains('maximized')");await page.shot('xyz-maximized');await page.click("document.querySelector('[aria-label=\"还原 LoRA 管理器窗口\"]')");
 await page.eval("window.__externalCalls=[];window.open=(...args)=>{window.__externalCalls.push(args);return null;}");await page.click("document.querySelector('[aria-label=\"在新标签页打开 LoRA 管理器\"]')");assert.deepEqual(await page.eval('window.__externalCalls'),[['/lora','_blank','noopener,noreferrer']],'real external callback; intercepted fixture only, no standalone state writes');
 await page.click(byText('.lora-engine-switch button','FLUX.1'));await page.wait("document.querySelector('.lora-modal-head p')?.textContent.includes('Flux 引擎')");assert.equal(await page.eval("window.__fixtureAppState().workspace.model"),'Anima');
 await page.click(byText('.lora-categories button','人物'));await page.wait("document.querySelectorAll('.lora-library-card').length===5");await page.click("document.querySelectorAll('.lora-library-card')[1]");await page.click(byText('.lora-categories button','已挂载'));await page.wait("document.querySelectorAll('.lora-mounted-item').length===2");await page.fill('[aria-label="Flux Portrait 1 权重"]','-0.5');await page.key('Tab');await sleep(500);
 assert.equal(await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Flux.baseline.loras[1].weight"),-0.5);assert.equal(await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.loras.length"),4);
 await page.click(byText('.lora-engine-switch button','Anima'));await page.wait("document.querySelectorAll('.lora-mounted-item').length===4");
 await page.click("document.querySelector('[aria-label=\"刷新 LoRA 列表\"]')");await page.key('Escape');await page.wait("!document.querySelector('.lora-modal')");await sleep(600);
 assert.deepEqual(await page.eval(privateState),before,'XYZ manager operations and engine switches leave all Basic/i2i/groups/selections untouched');
 ledger.checks.push('folder + legal/refused location, shared roots, custom covers/titles, details + nested Esc, frozen group copy, 5 mounts, negative/zero/invalid weight, SVG-handle private drag, group off/on, remove, maximise, refresh, engine isolation');
 // Shared real preset window, with XYZ-owned selections and one-time composition.
 await page.click("document.querySelector('.xyz-prompt-deck .preset-strip-open')");await page.wait("Boolean(document.querySelector('.preset-window'))");await page.shot('xyz-preset-window');
 await page.click("document.querySelector('.preset-card')");await sleep(600);
 const selected=await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.prompt_source");assert.equal(selected.active.positive.length,1);
 const record=await page.eval("window.__fixtureAppState().workspace.promptPresets.records.find(r=>r.id===window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.prompt_source.active.positive[0])");
 await page.click("document.querySelector('[aria-label=\"关闭预设窗口\"]')");await page.fill('[aria-label="XYZ 基线正向提示词"]','XYZ exact raw');await sleep(550);
 const composed=await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.parameters.prompt");assert.equal(composed.split(record.content).length,2);assert.deepEqual(await page.eval(privateState),before,'XYZ preset selection and typed text do not save Basic/i2i selections');
 await page.click("document.querySelector('.xyz-prompt-deck .preset-strip-open')");await page.wait("Boolean(document.querySelector('.preset-window'))");
 await page.click(byText('.preset-group-actions button','新建预设'));await page.wait("Boolean(document.querySelector('.prompt-preset-editor'))");
 await page.fill('.prompt-preset-editor-fields > label input','Stage1 Shared');await page.fill('.prompt-preset-editor-content textarea','stage1 shared words');
 await page.shot('xyz-shared-preset-editor');await page.click("document.querySelector('.prompt-preset-editor button[type=submit]')");await page.wait("!document.querySelector('.prompt-preset-editor')");
 await page.click(byText('.preset-card-main','Stage1 Shared'));await sleep(550);
 const windowShape=`(()=>{const w=document.querySelector('.preset-window');return {sections:[...w.children].map(e=>[e.tagName,e.className]),cards:[...w.querySelectorAll('.preset-card strong')].map(e=>e.textContent),classes:[...w.querySelectorAll('.preset-group-actions button')].map(e=>e.className)};})()`;
 const xyzPresetShape=await page.eval(windowShape);await page.shot('xyz-shared-preset-selected');
 await page.click("document.querySelector('[aria-label=\"关闭预设窗口\"]')");await page.choose('生成模式','基础生图');
 await page.click("document.querySelector('.center-panel .preset-strip-open')");await page.wait("Boolean(document.querySelector('.preset-window'))");assert.deepEqual(await page.eval(windowShape),xyzPresetShape,'Basic and XYZ share the actual preset window library DOM');
 await page.shot('basic-shared-preset-window');await page.click("document.querySelector('[aria-label=\"编辑预设：Stage1 Shared\"]')");await page.wait("Boolean(document.querySelector('.prompt-preset-editor'))");
 await page.fill('.prompt-preset-editor-content textarea','edited shared words');await page.click("document.querySelector('.prompt-preset-editor button[type=submit]')");await page.wait("!document.querySelector('.prompt-preset-editor')");await sleep(550);
 assert.ok(await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.parameters.prompt.includes('edited shared words')"),'explicit Basic library edit updates XYZ selected content, not its raw text');
 await page.click("document.querySelector('[aria-label=\"关闭预设窗口\"]')");await page.click(byText('.main-nav button','图生图'));await page.wait("Boolean(document.querySelector('.i2i-prompt-deck'))");
 await page.click("document.querySelector('.i2i-prompt-deck .preset-strip-open')");await page.wait("Boolean(document.querySelector('.preset-window'))");assert.deepEqual(await page.eval(windowShape),xyzPresetShape,'i2i and XYZ share actual preset window shape and same edited library');await page.shot('i2i-shared-preset-window');
 assert.ok(await page.eval("document.querySelector('.preset-window').textContent.includes('edited shared words')"));
 await page.click("document.querySelector('[aria-label=\"关闭预设窗口\"]')");await page.click(byText('.main-nav button','文生图'));await page.choose('生成模式','XYZ');await page.wait("Boolean(document.querySelector('.xyz-prompt-deck'))");
 const composedFinal=await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.parameters.prompt");assert.equal(composedFinal.split('edited shared words').length,2);assert.equal(await page.eval("document.querySelector('[aria-label=\"XYZ 基线正向提示词\"]').value"),'XYZ exact raw');assert.deepEqual(await page.eval(privateState),before);
 await page.click(byText('.model-seg','Flux'));await page.wait("Boolean(document.querySelector('.xyz-prompt-deck .negative-field textarea[disabled]'))");assert.equal(await page.eval("document.querySelector('.xyz-prompt-deck .negative-field textarea').value"),'Basic exact negative');
 await page.click(byText('.model-seg','Anima'));await page.wait("document.querySelector('.xyz-prompt-deck .positive-field textarea').value==='XYZ exact raw'");
 ledger.checks.push('XYZ creates shared library card; Basic edits it; i2i sees the same library; three real preset-window DOM shapes equal; selection/raw remain private; native disabled negative retains text');
 await page.eval('window.__holdPlan=true');await page.click(byText('.xyz-run-actions button','服务端预检'));await page.wait('Boolean(window.__releasePlan)');
 await page.fill('[aria-label="XYZ 基线正向提示词"]','XYZ invalidates held plan');await page.eval('window.__holdPlan=false;window.__releasePlan()');await sleep(400);
 assert.equal(await page.eval("Boolean(document.querySelector('.xyz-plan-summary'))"),false,'late old prompt plan cannot restore validity');
 await page.fill('[aria-label="XYZ 基线正向提示词"]','XYZ exact raw');await page.click(byText('.xyz-run-actions button','服务端预检'));await page.wait("window.__xyzPlanBodies?.length===2");
 await page.eval('window.__networkFailSubmit=1');await page.click(byText('.xyz-run-actions button','开始 XYZ'));await page.wait("Boolean(document.querySelector('.xyz-submit-error'))");
 await page.eval('window.__holdSubmit=true');await page.click(byText('.xyz-run-actions button','开始 XYZ'));await page.wait('Boolean(window.__releaseSubmit)');
 assert.equal(await page.eval('window.__xyzSubmitBodies[0].submission_id'),await page.eval('window.__xyzSubmitBodies[1].submission_id'),'unchanged prompt transport retry reuses submission id');
 await page.fill('[aria-label="XYZ 基线正向提示词"]','XYZ newer independent draft');await page.eval('window.__holdSubmit=false;window.__releaseSubmit()');await page.wait("Boolean(document.querySelector('.xyz-submit-notice'))");
 assert.equal(await page.eval("document.querySelector('[aria-label=\"XYZ 基线正向提示词\"]').value"),'XYZ newer independent draft','late successful frozen submit never rolls the live prompt back');
 const bodies=await page.eval("({plan:window.__xyzPlanBodies.at(-1),submit:window.__xyzSubmitBodies.at(-1)})");assert.equal(bodies.submit.draft.baseline.parameters.prompt,composedFinal);assert.ok(!Object.hasOwn(bodies.submit.draft.baseline,'prompt_source'));assert.equal(bodies.plan.baseline.parameters.prompt,composedFinal);
 await writeFile(path.join(EVIDENCE,'wire-bodies.json'),JSON.stringify(bodies,null,2));
 assert.deepEqual(await page.eval(privateState),before);await writeFile(path.join(EVIDENCE,'basic-i2i-groups-after.json'),JSON.stringify(await page.eval(privateState),null,2));
 ledger.checks.push('real shared preset window selects only XYZ; raw text stays separate; once-composed plan/submit wire strips UI source; no ordinary generate request');
 ledger.checks.push('prompt mutation invalidates held plan; transport retry keeps id only for unchanged draft; held submit freezes actual text and cannot roll live newer raw prompt back');
 assert.equal(await page.eval("window.__fixtureAppCalls.filter(c=>c.url==='/api/inference/jobs'&&c.method==='POST').length"),0);
 await page.send('Page.navigate',{url:`http://127.0.0.1:${vite.httpServer.address().port}/scripts/xyz-full-ui-stage1.fixture.html?legacy`});await page.wait("Boolean(document.querySelector('.xyz-legacy-prompt-note'))&&!document.querySelector('.app-loader')");
 assert.equal(await page.eval("document.querySelector('[aria-label=\"XYZ 基线正向提示词\"]').value"),'  group preset\n\nlegacy already composed  ');
 await page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");await page.key('Escape');await sleep(500);assert.equal(await page.eval("window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline.parameters.prompt"),'  group preset\n\nlegacy already composed  ');await page.shot('xyz-legacy-actual-retained');
 ledger.checks.push('real legacy saved draft opens with exact actual bytes + explicit additive-preset notice, no render/manager-open-time word injection');
 assert.deepEqual(page.errors,[]);assert.deepEqual(await page.eval('window.__xyzHostErrors'),[]);
 ledger.metrics={basic1440,basic390,imagePrompt1440,imagePrompt390};ledger.status='PASS';
} catch(error){ledger.status='FAIL';ledger.error=error.stack;if(page){await page.shot('failure').catch(()=>{});ledger.debug=await page.eval("({text:document.querySelector('.lora-modal')?.textContent,path:document.querySelector('.lora-path-field input')?.value,errors:window.__xyzHostErrors,calls:window.__fixtureAppCalls?.slice(-8)})").catch(()=>null);}throw error;}
finally {
 await writeFile(path.join(EVIDENCE,'RESULT.json'),JSON.stringify(ledger,null,2));
 console.log(JSON.stringify({evidence:EVIDENCE,status:ledger.status,checks:ledger.checks,error:ledger.error},null,2));
 page?.ws.close();if(edge?.pid)spawn('taskkill',['/PID',String(edge.pid),'/T','/F'],{stdio:'ignore',windowsHide:true});await vite.close();
}
