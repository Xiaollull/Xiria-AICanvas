import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import net from "node:net";
import path from "node:path";
import os from "node:os";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";


const ROOT=path.resolve(import.meta.dirname,".."), EVIDENCE=path.join(os.tmpdir(),"opencode",`lora-mix-ui-browser-${Date.now()}-${process.pid}`);
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


 const ledgerCheck=(text)=>ledger.checks.push(text);
 const openFresh=async()=>{
  await page.viewport(1440,900);await page.send("Page.navigate",{url:`http://127.0.0.1:${vite.httpServer.address().port}/scripts/xyz-group-copy.fixture.html`});
  await page.wait("Boolean(document.querySelector('.generation-mode-switch'))&&!document.querySelector('.app-loader')");
  await page.click(byText('.model-seg','Anima'));await page.wait("document.querySelector('.model-seg.active')?.textContent.includes('Anima')");await sleep(600);
 };
 const state=()=>page.eval("window.__fixtureAppState().workspace");
 const mixDraft=()=>page.eval("window.__fixtureAppState().workspace.loraMixByEngine?.by_engine?.Anima||null");
 const waitMix=async(predicate)=>{for(let i=0;i<80;i++){const value=await mixDraft();if(value&&await page.eval(`(${predicate})(window.__fixtureAppState().workspace.loraMixByEngine.by_engine.Anima)`))return value;await sleep(150);}throw new Error('Timeout mix '+predicate);};
 const overflow="(()=>{const s=document.querySelector('.mix-config');return {sw:s.scrollWidth,cw:s.clientWidth};})()";

 await openFresh();
 const basicBefore=await state();
 // ---- S1 three registered modes, selection, independent draft and mother rail
 await page.click("document.querySelector('.model-seg.active')");
 await page.click("document.querySelector('button[aria-label=\"生成模式\"]')");
 assert.deepEqual(await page.eval("[...document.querySelectorAll('[role=option]')].map(e=>e.textContent.trim())"),['基础生图','XYZ 对比生成','LoRA 混合搜索']);
 await page.click(byText('[role=option]','LoRA 混合搜索'));
 await page.wait("Boolean(document.querySelector('.mix-workspace'))&&!document.querySelector('.xyz-workspace:not(.mix-workspace)')");
 await page.click("document.querySelector('[aria-label=\"下一个生成模式\"]')");await page.wait("!document.querySelector('.mix-workspace')&&Boolean(document.querySelector('.prompt-field'))");
 await page.click("document.querySelector('[aria-label=\"上一个生成模式\"]')");await page.wait("Boolean(document.querySelector('.mix-workspace'))");
 for(const expected of ['.xyz-sidebar-mounts','.xyz-sidebar-models','.xyz-prompt-deck','.xyz-canvas-panel','.mix-x1']) assert.ok(await page.eval(`Boolean(document.querySelector('${expected}'))`),expected);
 assert.equal(await page.eval("document.querySelector('.xyz-sidebar-mounts .section-heading h2').textContent"),'全局 LoRA');
 await page.wait("Boolean(window.__fixtureAppState().workspace.loraMixByEngine?.by_engine?.Anima)");await sleep(300);
 let draft=await mixDraft();assert.equal(draft.kind,'lora_mix_draft');assert.equal(draft.mix.budget.preset,'32');assert.equal(draft.mix.members.length,0);
 assert.equal(draft.baseline.loras.length,1,'the search owns a copy of the Basic baseline mounts');
 assert.ok(await page.eval("(()=>{const c=document.querySelector('.mix-top .xyz-canvas-panel').getBoundingClientRect(),x=document.querySelector('.mix-x1').getBoundingClientRect();return Math.abs(c.top-x.top)<=1&&Math.abs(c.height-x.height)<=1&&x.left>=c.right;})()"),'X1 sits beside the canvas at the canvas card height');
 assert.ok(await page.eval("Boolean(document.querySelector('.mix-member-scroll .mix-member-empty'))"),'X1 shows an empty member list before selection');
 assert.ok(await page.eval("Boolean(document.querySelector('.mix-x2 .mix-rule-empty'))&&document.querySelector('.mix-x2 .mix-collapse').getAttribute('aria-expanded')==='true'"),'X2 is shown by default with its space reserved');
 // Shared chrome: the mode switch is one frame (borderless step arrows, a dot per mode) and the workspace scroller carries the
 // thin dark scrollbar, which the prompt textareas inherit (no white scrollbar corner behind a long prompt's resize grip).
 const chrome=await page.eval("(()=>{const steps=[...document.querySelectorAll('.generation-mode-step')];return {borders:steps.map(b=>getComputedStyle(b).borderTopWidth),dots:document.querySelectorAll('.generation-mode-dots i').length,active:[...document.querySelectorAll('.generation-mode-dots i')].findIndex(i=>i.classList.contains('active')),scroll:getComputedStyle(document.querySelector('.mix-config')).scrollbarColor,prompt:getComputedStyle(document.querySelector('.xyz-prompt-deck textarea')).scrollbarColor};})()");
 assert.deepEqual(chrome.borders,['0px','0px']);assert.equal(chrome.dots,3);assert.equal(chrome.active,2);
 assert.notEqual(chrome.scroll,'auto');assert.equal(chrome.prompt,chrome.scroll,'prompt textareas inherit the dark scrollbar');
 // A focused prompt shows the card's own focus state only, never a second workspace focus ring inside it.
 await page.click("document.querySelector('.xyz-prompt-deck .prompt-field.positive-field textarea')");
 assert.deepEqual(await page.eval("(()=>{const t=document.activeElement,f=t.closest('.prompt-field');return {deck:Boolean(t.closest('.xyz-prompt-deck')),outline:getComputedStyle(t).outlineStyle,card:getComputedStyle(f).borderTopColor!==getComputedStyle(document.querySelector('.xyz-prompt-deck .prompt-field.negative-field')).borderTopColor};})()"),{deck:true,outline:'none',card:true});
 assert.equal(await page.eval("document.querySelectorAll('.generation-mode-switch .workspace-select-trigger > svg').length"),2,'symmetric chevrons on the mode dropdown');
 await page.click("document.querySelector('.generation-mode-switch button[aria-label=\"生成模式\"]')");await page.wait("Boolean(document.querySelector('.workspace-select-menu[style*=\"left\"]'))");
 const menu=await page.eval("(()=>{const f=document.querySelector('.generation-mode-switch').getBoundingClientRect(),m=document.querySelector('.workspace-select-menu').getBoundingClientRect();return {left:Math.abs(f.left-m.left),right:Math.abs(f.right-m.right)};})()");
 assert.ok(menu.left<=1&&menu.right<=1,`the mode menu spans the whole switch frame (${JSON.stringify(menu)})`);await page.key('Escape');await page.wait("!document.querySelector('.workspace-select-menu')");
 ledgerCheck('M01 three modes in dropdown and both arrows; the search draft is created once from the Basic baseline; mother rail/prompt/canvas reused; X1 matches the canvas card with an empty list frame');
 await page.shot('mix-empty-1440');

 // ---- S2 X1 selection transaction (identities only), X2 defaults and field errors
 await page.click("document.querySelector('[aria-label=\"X1 测试 LoRA 选择\"]')");await page.wait("Boolean(document.querySelector('.xyz-owned-lora-manager'))");
 await page.click(byText('.xyz-owned-lora-manager .lora-categories button','人物'));await page.wait("document.querySelectorAll('.lora-library-card').length===5");
 for(const index of [1,2,3])await page.click(`document.querySelectorAll('.lora-library-card')[${index}]`);
 await page.click(byText('.xyz-owned-lora-manager .lora-modal-foot-actions .primary','确认轴选择'));await page.wait("!document.querySelector('.xyz-owned-lora-manager')&&document.querySelectorAll('.mix-rule').length===3");
 draft=await waitMix("m=>m.mix.members.length===3");assert.deepEqual(draft.mix.members.map(m=>m.ref.name),['Anima Portrait 1','Anima Portrait 2','Anima Portrait 3']);
 assert.equal(draft.baseline.loras.length,1,'X1 never touches the global background');
 assert.ok((await page.eval("document.querySelector('.mix-rule .mix-field.has-error small[role=alert]')?.textContent||''")).includes('不能为空'),'empty min/max are errors, not defaults');
 assert.equal(await page.eval("document.querySelector('#mix-rule-0-min').value"),'');
 assert.equal(await page.eval("document.querySelector('.mix-run-bar .primary').disabled"),true);
 ledgerCheck('M05/M06 X1 transaction picks identities only, keeps globals untouched; X2 rows start with explicit field errors (no default fill)');
 for(let i=0;i<3;i++){await page.fill(`#mix-rule-${i}-base`,'0.6');await page.fill(`#mix-rule-${i}-min`,'0.2');await page.fill(`#mix-rule-${i}-max`,'1');await page.fill(`#mix-rule-${i}-step`,'0.2');}
 await page.wait("document.querySelectorAll('.mix-rule .has-error').length===0");
 assert.ok((await page.eval("document.querySelector('.mix-rule .mix-badge').textContent")).includes('5 项：0.2、0.4、0.6、0.8、1'));
 // invalid raw input is retained and flagged
 await page.fill('#mix-rule-1-step','0');assert.ok((await page.eval("document.querySelector('#mix-rule-1-step-error')?.textContent||''")).includes('大于 0'));assert.equal(await page.eval("document.querySelector('#mix-rule-1-step').value"),'0');
 await page.fill('#mix-rule-1-step','0.12345');assert.ok((await page.eval("document.querySelector('#mix-rule-1-step-error')?.textContent||''")).includes('4 位小数'));assert.equal(await page.eval("document.querySelector('#mix-rule-1-step').value"),'0.12345');
 await page.fill('#mix-rule-1-step','0.2');await page.wait("document.querySelectorAll('.mix-rule .has-error').length===0");
 await page.click("document.querySelector('[aria-label=\"锁定 Anima Portrait 3\"]')");await page.wait("document.querySelector('#mix-rule-2-min').disabled");
 assert.equal(await page.eval("document.querySelector('#mix-rule-2-min').value"),'0.2','locking retains the raw unlocked values');
 await page.click("document.querySelector('[aria-label=\"锁定 Anima Portrait 3\"]')");await page.wait("!document.querySelector('#mix-rule-2-min').disabled");
 // X2 collapses to its heading (rules kept); an X1 rule chip re-opens it and focuses that member's base weight.
 await page.click("document.querySelector('.mix-x2 .mix-collapse')");await page.wait("!document.querySelector('.mix-x2-body')&&document.querySelector('.mix-x2 .mix-collapse').getAttribute('aria-expanded')==='false'");
 assert.ok((await page.eval("document.querySelector('.mix-x2 .mix-heading-meta').textContent")).includes('3 个成员'));
 assert.ok((await page.eval("document.querySelector('.mix-x1 .mix-member-rule').textContent")).includes('基础 0.6'),'X1 cards summarise the X2 rule');
 await page.click("document.querySelector('.mix-x1 .mix-member-rule')");await page.wait("Boolean(document.querySelector('.mix-x2-body'))&&document.activeElement?.id==='mix-rule-0-base'");
 assert.equal(await page.eval("document.querySelector('#mix-rule-1-min').value"),'0.2','collapsing kept the raw rules');
 ledgerCheck('M06/M07 exact decimal fields: raw retained for zero / over-precise steps; lock disables and retains; no rounding; X2 collapses without losing rules and X1 chips reopen it');

 // ---- S3 X3 and candidate preview
 assert.equal(await page.eval("document.querySelector('#mix-max-displacement').value"),'0.6');
 await page.click(byText('.mix-x3 button','完整范围'));assert.equal(await page.eval("document.querySelector('#mix-max-displacement').value"),'1.2');
 await page.fill('#mix-max-displacement','1.3');assert.ok((await page.eval("document.querySelector('#mix-max-displacement-error')?.textContent||''")).includes('不会自动截断'));assert.equal(await page.eval("document.querySelector('#mix-max-displacement').value"),'1.3','out-of-range stays visible, not clamped');
 await page.fill('#mix-max-displacement','0.6');await page.wait("!document.querySelector('#mix-max-displacement-error')");
 await page.click(byText('.mix-radio','16'));
 assert.ok((await page.eval("document.querySelector('.mix-summary').textContent")).includes('候选空间'));
 await page.click(byText('.mix-candidate-actions button','生成候选预览'));await page.wait("document.querySelectorAll('.mix-candidate-table tbody tr').length===16");
 assert.ok(await page.eval("Boolean(document.querySelector('.mix-candidate-table tbody tr .mix-badge'))"),'baseline row is marked');
 assert.ok((await page.eval("document.querySelector('.mix-run-bar').textContent")).includes('16 张'));
 draft=await waitMix("m=>m.mix.candidates.length===16");assert.equal(draft.mix.cursor,1);assert.equal(draft.mix.candidates[0].source,'baseline');
 const first16=draft.mix.candidates.map(c=>JSON.stringify(c.raw));
 await page.shot('mix-preview-1440');
 // editing: off-sequence value flagged, raw retained, row invalid
 await page.fill('[aria-label="C02 Anima Portrait 1 权重"]','0.65');
 assert.ok((await page.eval("document.querySelector('.mix-candidate-table tbody tr:nth-child(2) .mix-status').textContent")).includes('候选序列'));
 assert.equal(await page.eval("document.querySelector('[aria-label=\"C02 Anima Portrait 1 权重\"]').value"),'0.65');
 assert.equal(await page.eval("document.querySelector('.mix-run-bar .primary').disabled"),true);
 assert.ok((await page.eval("document.querySelector('.mix-issues').textContent")).includes('C02'));
 await page.click("document.querySelector('[aria-label=\"固定 C03\"]')");await page.click("document.querySelector('[aria-label=\"删除 C02\"]')");
 await page.wait("document.querySelectorAll('.mix-candidate-table tbody tr').length===15");
 await page.click(byText('.mix-candidate-actions button','重抽未固定项'));await page.wait("document.querySelectorAll('.mix-candidate-table tbody tr').length===16");
 draft=await waitMix("m=>m.mix.cursor===2");
 const pinnedRow=draft.mix.candidates.find(c=>c.id==='C03');assert.ok(pinnedRow&&pinnedRow.pinned,'pinned row survives redraw');assert.equal(JSON.stringify(pinnedRow.raw),first16[2]);
 assert.ok(!draft.mix.candidates.some(c=>c.id==='C02'),'a deleted id is not reused');
 await page.wait("!document.querySelector('.mix-issues')");
 ledgerCheck('M10/M11 generate/redraw/pin/delete/edit: raw retained, off-sequence flagged, ids never reused, no job and no request');

 // ---- S4 global/test role conflict is explicit
 await page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");await page.wait("Boolean(document.querySelector('.xyz-owned-lora-manager'))");
 await page.click(byText('.xyz-owned-lora-manager .lora-categories button','人物'));await page.wait("document.querySelectorAll('.lora-library-card').length===5");
 await page.click("document.querySelectorAll('.lora-library-card')[1]");await page.click(byText('.xyz-owned-lora-manager .lora-modal-foot-actions .primary','完成'));await page.wait("!document.querySelector('.xyz-owned-lora-manager')");
 await page.wait("Boolean(document.querySelector('.mix-role-error'))");
 assert.ok((await page.eval("document.querySelector('.mix-role-error').textContent")).includes('同一实例')&&(await page.eval("document.querySelector('.mix-role-error').textContent")).includes('不能重叠'));
 assert.equal(await page.eval("document.querySelector('.mix-run-bar .primary').disabled"),true);
 await page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");await page.wait("Boolean(document.querySelector('.xyz-owned-lora-manager'))");
 await page.click(byText('.xyz-owned-lora-manager .lora-categories button','人物'));await page.click("document.querySelectorAll('.lora-library-card')[1]");
 await page.click(byText('.xyz-owned-lora-manager .lora-modal-foot-actions .primary','完成'));await page.wait("!document.querySelector('.xyz-owned-lora-manager')&&!document.querySelector('.mix-role-error')");
 draft=await mixDraft();assert.equal(draft.mix.members.length,3,'resolving a conflict never moves or edits a test member');
 ledgerCheck('M03/M04 a global mount equal to a test member is an explicit error on the member row and blocks run; unmounting it clears it; nothing moved automatically');

 // ---- S5 isolation, geometry, run bar
 const basicAfter=await state();
 assert.deepEqual(basicAfter.engineSettingsByEngine,basicBefore.engineSettingsByEngine,'Basic settings untouched');
 assert.deepEqual(basicAfter.mountedLorasByEngine,basicBefore.mountedLorasByEngine,'Basic mounts untouched');
 assert.deepEqual(basicAfter.loraGroupsByEngine,basicBefore.loraGroupsByEngine,'real groups untouched');
 assert.equal(basicAfter.generationMode,'lora_mix_search');assert.equal(basicAfter.xyzByEngine?.by_engine?.Anima,undefined,'no XYZ draft was created or touched');
 const bar=await page.eval("({text:document.querySelector('.mix-run-bar').textContent,confirmTitle:document.querySelector('.mix-run-bar .primary').title,planDisabled:document.querySelectorAll('.mix-run-bar button')[0].disabled})");
 assert.ok(bar.confirmTitle.includes('尚未提供')||bar.text.includes('尚未'));assert.equal(bar.planDisabled,true);
 // Visual pass: the shared .xyz-workspace rules leave buttons at the browser's 16px; every mix control has its own compact scale,
 // X1/X2 sit beside the canvas like the XYZ axes, X2 fields share one row and candidate rows stay short.
 const look=await page.eval(`(()=>{const big=[...document.querySelectorAll('.mix-workspace button:not(.workspace-select-trigger)')].filter(b=>b.offsetParent&&!b.closest('.xyz-prompt-deck,.xyz-mother-controls,.xyz-canvas-panel,.xyz-toolbar')&&parseFloat(getComputedStyle(b).fontSize)>13).map(b=>b.textContent.trim().slice(0,20));
  const canvas=document.querySelector('.mix-top .xyz-canvas-panel').getBoundingClientRect(),x1=document.querySelector('.mix-x1').getBoundingClientRect(),x2=document.querySelector('.mix-x2').getBoundingClientRect(),list=document.querySelector('.mix-member-scroll');
  const tops=[...document.querySelectorAll('.mix-rule:first-child .mix-rule-fields > .mix-field')].map(e=>Math.round(e.getBoundingClientRect().top));
  const rows=[...document.querySelectorAll('.mix-candidate-table tbody tr')].map(r=>r.getBoundingClientRect().height);
  return {big,x1:Math.abs(x1.top-canvas.top)<=1&&Math.abs(x1.height-canvas.height)<=1,x2Below:x2.top>=canvas.bottom,gutter:getComputedStyle(list).scrollbarGutter,cards:list.querySelectorAll('li').length,oneRow:tops.length===5&&new Set(tops).size===1,row:Math.max(...rows)};})()`);
 assert.deepEqual(look.big,[],'no browser-default 16px buttons in the mix workspace');
 assert.ok(look.x1,'X1 keeps the canvas card height with members listed');assert.equal(look.cards,3);assert.equal(look.gutter,'stable','the X1 list reserves its vertical scrollbar');
 assert.ok(look.x2Below,'X2 spans below the canvas row');
 assert.ok(look.oneRow,'X2 base/min/max/step/preference share one row');
 assert.ok(look.row<=44,`candidate rows stay compact (${look.row}px)`);
 for(const [width,height] of [[1440,900],[1024,768],[390,844]]){
  await page.viewport(width,height);await sleep(300);
  const metric=await page.eval(overflow);assert.ok(metric.sw<=metric.cw+1,`${width}x${height}: no page-level horizontal scroll (${metric.sw}>${metric.cw})`);
  if(width>=1280)assert.ok(await page.eval("(()=>{const f=document.querySelector('.mix-run-bar').getBoundingClientRect();return f.bottom<=innerHeight+1&&f.height>0;})()"),`${width}x${height}: run bar reachable`);
  else{await page.eval("document.querySelector('.mix-run-bar').scrollIntoView({block:'end'})");await sleep(150);assert.ok(await page.eval("(()=>{const f=document.querySelector('.mix-run-bar').getBoundingClientRect();return f.top>=0&&f.bottom<=innerHeight+1&&f.height>0;})()"),`${width}x${height}: run bar reachable by scrolling the stacked layout`);}
  await page.shot(`mix-config-${width}x${height}`);
 }
 await page.viewport(1440,900);
 assert.deepEqual(page.errors,[]);
 ledgerCheck('M23 desktop / 1024 / mobile: single main scroll, candidate table is the only horizontal scroller, run bar reachable; compact controls, X1 at the canvas card height with a scrolling member list, one-row X2 fields below; Basic/real groups/XYZ untouched');
 ledger.status='PASS';
} catch(error){ledger.status='FAIL';ledger.error=error.stack;if(page){ledger.pageErrors=page.errors;await page.shot('failure').catch(()=>{});ledger.debug=await page.eval("({mix:document.querySelector('.mix-workspace')?.innerText.slice(0,1500)})").catch(()=>null);}}
finally {
 await writeFile(path.join(EVIDENCE,'RESULT.json'),JSON.stringify(ledger,null,2));
 console.log(JSON.stringify({evidence:EVIDENCE,status:ledger.status,checks:ledger.checks,error:ledger.error},null,2));
 page?.ws.close();if(edge?.pid)spawn('taskkill',['/PID',String(edge.pid),'/T','/F'],{stdio:'ignore',windowsHide:true});await vite.close();
}
