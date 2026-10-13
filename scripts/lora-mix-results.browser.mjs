import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import net from "node:net";
import path from "node:path";
import os from "node:os";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";


const ROOT=path.resolve(import.meta.dirname,".."), EVIDENCE=path.join(os.tmpdir(),"opencode",`lora-mix-results-browser-${Date.now()}-${process.pid}`);
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

let edge,page;const ledger={scope:"executor SELF-TEST; real App + fixture APIs; no GPU/service",checks:[]};
const vite=await createServer({configFile:false,root:ROOT,cacheDir:path.join(EVIDENCE,"vite-cache"),appType:"mpa",plugins:[react(),{name:"fixture-images",configureServer(server){server.middlewares.use((req,res,next)=>{if(req.url.startsWith('/api/lora-cards/image')||req.url.startsWith('/api/lora-preview')){res.setHeader('Content-Type','image/png');res.end(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jI4sAAAAASUVORK5CYII=','base64'));}else next();});}}],logLevel:"error",server:{host:"127.0.0.1",port:0,hmr:false,open:false,watch:null},optimizeDeps:{noDiscovery:true,include:["react","react-dom/client","lucide-react"]}});
try {
 await vite.listen();const probe=net.createServer();await new Promise(r=>probe.listen(0,"127.0.0.1",r));const port=probe.address().port;await new Promise(r=>probe.close(r));
 const profile=path.join(EVIDENCE,"edge-profile");await mkdir(profile);
 edge=spawn("C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",["--headless=new",`--remote-debugging-port=${port}`,`--user-data-dir=${profile}`,"--no-first-run","--disable-extensions","--disable-sync","--disable-background-networking","about:blank"],{stdio:"ignore",windowsHide:true});
 for(let i=0;i<80;i++){try{if((await fetch(`http://127.0.0.1:${port}/json/version`)).ok)break;}catch{}await sleep(150);}
 const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();const ws=new WebSocket(targets.find(t=>t.type==="page").webSocketDebuggerUrl);page=new Page(ws);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});await page.send("Page.enable");await page.send("Runtime.enable");


 const check=(text)=>ledger.checks.push(text);
 const mixDraft=()=>page.eval("window.__fixtureAppState().workspace.loraMixByEngine?.by_engine?.Anima||null");
 const waitState=async(expr,label)=>{for(let i=0;i<80;i++){if(await page.eval(`(()=>{try{return Boolean(${expr});}catch{return false;}})()`))return;await sleep(150);}throw new Error('Timeout state '+label);};
 await page.viewport(1440,900);await page.send("Page.navigate",{url:`http://127.0.0.1:${vite.httpServer.address().port}/scripts/lora-mix.fixture.html`});
 await page.wait("Boolean(document.querySelector('.generation-mode-switch'))&&!document.querySelector('.app-loader')");
 await page.click(byText('.model-seg','Anima'));await page.wait("document.querySelector('.model-seg.active')?.textContent.includes('Anima')");await sleep(500);
 await page.choose('生成模式','LoRA 混合搜索');await page.wait("Boolean(document.querySelector('.mix-workspace'))");await sleep(300);
 const groupsBefore=await page.eval("JSON.stringify(window.__fixtureAppState().workspace.loraGroupsByEngine)");
 await page.click("document.querySelector('[aria-label=\"X1 测试 LoRA 选择\"]')");await page.wait("Boolean(document.querySelector('.xyz-owned-lora-manager'))");
 await page.click(byText('.xyz-owned-lora-manager .lora-categories button','人物'));await page.wait("document.querySelectorAll('.lora-library-card').length===5");
 for(const index of [1,2])await page.click(`document.querySelectorAll('.lora-library-card')[${index}]`);
 await page.click(byText('.xyz-owned-lora-manager .lora-modal-foot-actions .primary','确认轴选择'));await page.wait("document.querySelectorAll('.mix-rule').length===2");
 for(let i=0;i<2;i++){await page.fill(`#mix-rule-${i}-base`,'0.6');await page.fill(`#mix-rule-${i}-min`,'0.2');await page.fill(`#mix-rule-${i}-max`,'1');await page.fill(`#mix-rule-${i}-step`,'0.2');}
 await page.click(byText('.mix-radio','16'));
 await page.click(byText('.mix-candidate-actions button','生成候选预览'));await page.wait("document.querySelectorAll('.mix-candidate-table tbody tr').length===16");
 // ---- R1 server preflight then confirm (capabilities report a ready executor in this fixture)
 await page.wait("!document.querySelectorAll('.mix-run-bar button')[0].disabled");
 await page.click("document.querySelectorAll('.mix-run-bar button')[0]");await page.wait("Boolean(document.querySelector('.xyz-plan-summary'))");
 assert.ok((await page.eval("document.querySelector('.xyz-plan-summary').textContent")).includes('16 个配置 × 1 样本 = 16 张'));
 await page.click("document.querySelector('.mix-run-bar .primary')");await page.wait("document.querySelectorAll('.mix-grid .mix-tile').length===16");
 const confirmCall=await page.eval("window.__mixState.calls.find(c=>c.url==='/api/inference/lora-mix/sessions'&&c.method==='POST')");
 assert.equal(confirmCall.body.kind,'lora_mix_submit');assert.ok(/^[0-9a-f-]{36}$/.test(confirmCall.body.submission_id));assert.equal(confirmCall.body.draft.mix.candidates.length,16);
 assert.equal(confirmCall.body.draft.kind,'xyz_draft','the wire recipe is the XYZ projection');assert.equal(confirmCall.body.draft.baseline.prompt_source,undefined);
 await waitState("window.__fixtureAppState().workspace.loraMixByEngine.by_engine.Anima.mix.session.revision===1","session adopted");
 const first=await page.eval("document.querySelector('.mix-grid .mix-tile figcaption').textContent");
 for(const part of ['C01','基线','Δ0','已测试'])assert.ok(first.includes(part),part);
 assert.equal(await page.eval("getComputedStyle(document.querySelector('.mix-grid')).getPropertyValue('--mix-columns').trim()"),'4','16 candidates -> 4x4');
 await page.shot('results-round1-1440');
 check('M15/M19 confirm sends one lora_mix_submit with the XYZ wire recipe; the session id/revision is adopted; 16 tiles with id/baseline/displacement/state captions in a recommended 4x4 layout');
 // ---- R2 presentation controls, favorites, details
 await page.choose('候选网格列数','8 列');
 await page.wait("getComputedStyle(document.querySelector('.mix-grid')).getPropertyValue('--mix-columns').trim()==='8'");
 await page.click("document.querySelector('[aria-label=\"收藏 C03\"]')");await waitState("window.__fixtureAppState().workspace.loraMixByEngine.by_engine.Anima.mix.favorites.length===1","favorite saved");
 await page.click(byText('.mix-grid-tools label','只看收藏'));await page.wait("document.querySelectorAll('.mix-grid .mix-tile').length===1");
 await page.click(byText('.mix-grid-tools label','只看收藏'));await page.wait("document.querySelectorAll('.mix-grid .mix-tile').length===16");
 await page.click("[...document.querySelectorAll('.mix-tile')].find(t=>t.textContent.includes('C02')).querySelector('.mix-text-button')");
 await page.wait("Boolean(document.querySelector('.mix-details'))");
 const details=await page.eval("document.querySelector('.mix-details').textContent");
 for(const part of ['C02 · 配置详情','测试 LoRA','全局 LoRA','Anima Portrait 1','候选键'])assert.ok(details.includes(part),part);
 await page.shot('results-details-1440');await page.key('Escape');await page.wait("!document.querySelector('.mix-details')");
 check('M19/M20 columns are a presentation control; favorites persist by candidate key; the details dialog shows frozen test weights and the global background apart');
 // ---- R3 continue untested candidates: the next draw excludes every tested vector and says the space is used up honestly
 const round1=(await mixDraft()).mix.candidates.map(c=>JSON.stringify(c.raw));
 await page.click(byText('.mix-result-bar button','继续未测试候选'));await page.wait("document.querySelectorAll('.mix-candidate-table tbody tr').length===5");
 await waitState("window.__fixtureAppState().workspace.loraMixByEngine.by_engine.Anima.mix.candidates.length===5","continued draft saved");
 const next=(await mixDraft());
 assert.ok(next.mix.candidates.every(c=>!round1.includes(JSON.stringify(c.raw))),'no tested vector is proposed again');
 assert.ok((await page.eval("document.querySelector('.mix-config').textContent")).includes('候选空间已穷尽：只有 5 个可用配置'));
 await page.click("document.querySelector('.mix-run-bar .primary')");await page.wait("document.querySelectorAll('.mix-grid .mix-tile').length===5");
 await waitState("window.__fixtureAppState().workspace.loraMixByEngine.by_engine.Anima.mix.session.revision===2","round 2");
 await page.click("document.querySelector('button[aria-label=\"选择搜索回合\"]')");assert.equal(await page.eval("document.querySelectorAll('.workspace-select-menu [role=option]').length"),2);await page.key('Escape');
 check('M10/M14 continue draws only untested vectors of the same context, reports exhaustion of the 21-vector space instead of padding, and extends the same session');
 // ---- R4 explicit follow-up and group saving
 await page.click("document.querySelector('.mix-tile .mix-more-trigger')");
 await page.click(byText('.mix-more-menu button','更窄的后续'));await page.wait("Boolean(document.querySelector('#mix-follow-title'))");
 const plan=await page.eval("document.querySelector('#mix-follow-title').closest('section').textContent");
 assert.ok(plan.includes('锁定成员、全局 LoRA、旧回合与真实组合都不会改变'));
 await page.click(byText('.xyz-sync-dialog .primary','应用到搜索草稿'));await page.wait("!document.querySelector('#mix-follow-title')");
 const followed=await waitState("window.__fixtureAppState().workspace.loraMixByEngine.by_engine.Anima.mix.candidates.length===0","follow-up applied");
 const rules=(await mixDraft()).mix.members.map(m=>m.rule);assert.ok(rules.some(r=>r.min!=='0.2'||r.max!=='1'),'the range narrowed around the chosen vector');
 await page.click("document.querySelector('.mix-tile .mix-more-trigger')");
 await page.click(byText('.mix-more-menu button','保存为组合'));await page.wait("Boolean(document.querySelector('#mix-save-title'))");
 await page.click(byText('.xyz-sync-dialog label','完整混合'));
 await page.click(byText('.xyz-sync-dialog .primary','创建组合'));await page.wait("!document.querySelector('#mix-save-title')");
 await waitState("window.__fixtureAppState().workspace.loraGroupsByEngine.byEngine.Anima.length===JSON.parse("+JSON.stringify(groupsBefore)+").byEngine.Anima.length+1","group saved");
 const groups=await page.eval("window.__fixtureAppState().workspace.loraGroupsByEngine.byEngine.Anima");
 const saved=groups.at(-1);assert.match(saved.name,/含全局/);assert.equal(saved.members.length,3,'1 global + 2 tests');
 assert.deepEqual(groups.slice(0,-1),JSON.parse(groupsBefore).byEngine.Anima,'existing groups are untouched');
 check('M20 follow-up and adopt edits are previewed and apply only to the search draft; "complete mix" saves a NEW group (globals + tests); source groups unchanged');
 // ---- R5 single-image candidate-grid export carries the presentation columns
 await page.click(byText('.mix-grid-tools button','导出候选网格 PNG'));await page.wait("Boolean(document.querySelector('.xyz-export-dialog'))");
 await page.wait("document.querySelector('.xyz-export-estimate')?.textContent.includes('px')");
 // The seed lives in the summary now; the LoRA legend + numbered per-image weights are an explicit, default-off option.
 const lastEstimate="window.__mixState.calls.filter(c=>c.url.endsWith('/export/estimate')).at(-1)?.body";
 assert.equal(await page.eval(`${lastEstimate}.lora_weights`),false,'LoRA weights are off by default');
 assert.ok((await page.eval("document.querySelector('.xyz-export-dialog').textContent")).includes('编号 / 位移 / 状态'));
 await page.click(byText('.xyz-export-dialog label','显示 LoRA 与权重'));await page.wait(`${lastEstimate}?.lora_weights===true`);
 await page.click(byText('.xyz-export-dialog label','显示 LoRA 与权重'));await page.wait(`${lastEstimate}?.lora_weights===false`);
 await page.click(byText('.xyz-export-dialog .primary','导出 PNG'));await page.wait("document.querySelector('.xyz-export-progress')?.textContent.includes('已完成')");
 const exportCall=await page.eval("window.__mixState.calls.filter(c=>c.url.endsWith('/exports')).at(-1)");
 assert.ok(Number.isInteger(exportCall.body.columns)&&exportCall.body.columns>=1,'the column choice is sent to the search export route');
 assert.equal(exportCall.body.arrangement,'vertical');assert.equal(exportCall.body.lora_weights,false,'the export sends exactly what the checkbox shows');
 await page.shot('results-export-1440');
 // A short window makes the dialog scroll: it must use the thin dark XYZ scrollbar, not the light default.
 await page.viewport(1440,560);
 assert.ok(await page.eval("(()=>{const d=document.querySelector('.xyz-export-dialog'),s=getComputedStyle(d);return d.scrollHeight>d.clientHeight&&s.scrollbarWidth==='thin'&&s.scrollbarColor!=='auto'})()"),'the export dialog scrolls with the styled scrollbar');
 await page.shot('results-export-scroll-1440x560');await page.viewport(1440,900);await page.key('Escape');
 check('M22 the candidate-grid export uses the search route with explicit columns, the default-off LoRA weights option and the shared export dialog');
 // ---- R5b the XYZ workspace lists only its own experiments: search rounds never appear in XYZ results
 await page.choose('生成模式','XYZ 对比生成');await page.wait("Boolean(document.querySelector('.xyz-workspace:not(.mix-workspace) .xyz-tabs'))&&!document.querySelector('.mix-workspace')");
 await page.click(byText('.xyz-workspace:not(.mix-workspace) .xyz-tabs button','结果'));await page.wait("Boolean(document.querySelector('button[aria-label=\"选择 XYZ 实验\"]'))");
 await page.click("document.querySelector('button[aria-label=\"选择 XYZ 实验\"]')");await page.wait("document.querySelectorAll('.workspace-select-menu [role=option]').length>0");
 const xyzOptions=await page.eval("[...document.querySelectorAll('.workspace-select-menu [role=option]')].map(o=>o.textContent.trim())");await page.key('Escape');
 assert.equal(xyzOptions.length,1,`only the plain XYZ experiment is listed (${xyzOptions.join(' | ')})`);assert.ok(xyzOptions[0].includes('4/8'));
 assert.ok((await page.eval("window.__mixState.xyzLists.at(-1)")).includes('origin=none'),'the XYZ list asks the server for its own experiments only');
 assert.equal(await page.eval("document.querySelectorAll('.xyz-other-active button').length"),0,'search rounds are not offered as other active XYZ experiments');
 check('Isolation: after two search rounds the XYZ results list only XYZ experiments (origin=none + client filter)');
 // ---- R6 a queue row opens its own workspace by origin
 await page.choose('生成模式','基础生图');await page.wait("!document.querySelector('.mix-workspace')");
 await page.eval("window.__mixQueue=true;0");
 await page.click(byText('button','队列'));await page.wait("[...document.querySelectorAll('.queue-item')].some(e=>e.textContent.includes('LoRA 混合搜索 · 第 1 轮'))");
 await page.click("[...document.querySelectorAll('.queue-item')].find(e=>e.textContent.includes('LoRA 混合搜索')).querySelector('.queue-item-main')");
 await page.wait("Boolean(document.querySelector('.mix-results'))&&document.querySelectorAll('.mix-grid .mix-tile').length===16");
 check('M01 a queue parent with origin lora_mix_search opens the search results, not XYZ');
 // ---- R7 geometry
 for(const [w,h] of [[1440,900],[390,844]]){await page.viewport(w,h);await sleep(300);const m=await page.eval("(()=>{const r=document.querySelector('.mix-results');return {doc:document.scrollingElement.scrollWidth<=innerWidth+1,grid:document.querySelector('.mix-grid').getBoundingClientRect().width>0};})()");assert.ok(m.doc&&m.grid,`${w}x${h}`);
  // The results view owns its scroller: the last tile can be brought into view above the round footer.
  const reach=await page.eval("(async()=>{const main=document.querySelector('.mix-results-main'),tiles=document.querySelectorAll('.mix-grid .mix-tile'),last=tiles[tiles.length-1];last.scrollIntoView({block:'end'});await new Promise(r=>setTimeout(r,200));const t=last.getBoundingClientRect(),bar=document.querySelector('.mix-result-bar').getBoundingClientRect(),m=main.getBoundingClientRect();return {scrolls:main.scrollHeight>main.clientHeight,visible:t.bottom<=m.bottom+1&&t.bottom<=bar.top+1&&t.height>0};})()");
  assert.ok(reach.scrolls&&reach.visible,`${w}x${h}: the last candidate tile is reachable (${JSON.stringify(reach)})`);
  await page.shot(`results-${w}x${h}`);}
 assert.deepEqual(page.errors,[]);
 check('M23 results at 1440x900 and 390x844 without page-level horizontal scroll; the results view scrolls to its last tile above the round footer');
 ledger.status='PASS';
} catch(error){ledger.status='FAIL';ledger.error=error.stack;if(page){ledger.pageErrors=page.errors;await page.shot('failure').catch(()=>{});}}
finally {
 await writeFile(path.join(EVIDENCE,'RESULT.json'),JSON.stringify(ledger,null,2));
 console.log(JSON.stringify({evidence:EVIDENCE,status:ledger.status,checks:ledger.checks,error:ledger.error},null,2));
 page?.ws.close();if(edge?.pid)spawn('taskkill',['/PID',String(edge.pid),'/T','/F'],{stdio:'ignore',windowsHide:true});await vite.close();
}
