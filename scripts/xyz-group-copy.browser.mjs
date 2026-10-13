import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import net from "node:net";
import path from "node:path";
import os from "node:os";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";


const ROOT=path.resolve(import.meta.dirname,".."), EVIDENCE=path.join(os.tmpdir(),"opencode",`xyz-group-copy-browser-${Date.now()}-${process.pid}`);
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
  await page.choose('生成模式','XYZ');await page.wait("Boolean(document.querySelector('.xyz-prompt-deck'))&&!document.querySelector('.xyz-sidebar-models [role=status]')");await sleep(400);
 };
 const openManager=async()=>{await page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");await page.wait("Boolean(document.querySelector('.xyz-owned-lora-manager'))");};
 const category=async(label)=>{await page.click(byText('.xyz-owned-lora-manager .lora-categories button',label));await sleep(200);};
 const toGroups=async()=>{await category('组合');await page.wait("Boolean(document.querySelector('.xyz-owned-lora-manager [aria-label^=\"复制 \"]'))");};
 const copyButton=(name)=>`document.querySelector('[aria-label="复制 ${name} 到 XYZ"]')`;
 const hasDialog="Boolean(document.querySelector('.xyz-group-copy-dialog'))";
 const draft=()=>page.eval("window.__fixtureAppState().workspace.xyzByEngine?.by_engine?.Anima?.baseline||null");
 const liveDraft=async(predicate)=>{for(let i=0;i<60;i++){const value=await draft();if(value&&await page.eval(`(${predicate})(window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima.baseline)`))return value;await sleep(150);}throw new Error('Timeout draft '+predicate);};
 const focusInside=(selector)=>page.eval(`Boolean(document.activeElement&&document.querySelector(${JSON.stringify(selector)})?.contains(document.activeElement))`);
 const geometry=`(()=>{const box=s=>{const e=document.querySelector(s);if(!e)return null;const b=e.getBoundingClientRect();return {top:b.top,bottom:b.bottom,left:b.left,right:b.right,height:b.height,width:b.width,scrollW:e.scrollWidth,clientW:e.clientWidth};};
  const list=document.querySelector('.xyz-owned-lora-manager .lora-mounted-list,.xyz-owned-lora-manager .lora-group-list');
  return {vh:innerHeight,vw:innerWidth,h2:box('.xyz-owned-lora-manager .lora-modal-head h2'),modal:box('.xyz-owned-lora-manager'),content:box('.lora-browser-content'),notice:box('.lora-manager-notice'),foot:box('.lora-modal-foot'),
   alerts:document.querySelectorAll('.xyz-owned-lora-manager [role=alert]').length,list:list&&{client:list.clientHeight,scroll:list.scrollHeight}};})()`;

 await openFresh();
 // ---- S1 cancel, late result, resolution, success
 await openManager();await toGroups();
 const groupsBefore=await page.eval("window.__fixtureAppState().workspace.loraGroupsByEngine");
 await page.eval("window.__resolve.gate=new Promise(r=>{window.__releaseResolve=r;});0");
 await page.click(copyButton('NKStyleV2'));await page.wait(hasDialog);
 assert.equal(await page.eval("document.querySelector('.xyz-group-copy-dialog').getAttribute('aria-busy')"),'true');
 assert.ok(await page.eval("document.querySelector('.xyz-group-copy-dialog [role=status]').textContent.includes('正在向资产库核验')"));
 assert.equal(await page.eval("[...document.querySelectorAll('.xyz-group-copy-dialog footer button')].map(b=>b.textContent).join('|')"),'取消');
 await page.key('Escape');await page.wait(`!${hasDialog}`);
 assert.ok(await focusInside('.xyz-owned-lora-manager'),'cancel returns focus into the manager');
 await page.eval("window.__releaseResolve();0");await sleep(900);
 assert.equal(await page.eval(hasDialog),false,'a late resolution result must not reopen or commit');
 assert.equal(await page.eval("(window.__fixtureAppState().workspace.xyzByEngine?.by_engine?.Anima?.baseline.group_snapshots||[]).length"),0);
 ledgerCheck('G08 cancel during resolution aborts, returns focus, and the late result is dropped without any write');
 await page.eval("window.__resolve.gate=null;window.__resolveCalls.length=0;0");
 await page.click(copyButton('NKStyleV2'));await page.wait("document.querySelectorAll('.xyz-copy-bindings li').length===8");
 assert.equal(await page.eval("window.__resolveCalls.length"),8);
 assert.ok(await page.eval("window.__resolveCalls.every(c=>c.references.length===1&&!('enabled' in c.references[0])&&c.engine==='Anima')"),'binding-only per-member references');
 const dialogText=await page.eval("document.querySelector('.xyz-group-copy-dialog').textContent");
 for(const part of ['新绑定当前实例','没有内容哈希','挂载总数：现有 1 个 → 启用后 9 / 16'])assert.ok(dialogText.includes(part),part);
 assert.equal(await page.eval("[...document.querySelectorAll('.xyz-group-copy-dialog footer button')].map(b=>b.textContent+(b.disabled?'(off)':'')).join('|')"),'取消|仅复制（不启用）|复制并启用快照');
 await page.shot('s1-ready-1440');
 await page.click(byText('.xyz-group-copy-dialog .primary','复制并启用快照'));await page.wait(`!${hasDialog}`);
 assert.ok(await focusInside('.xyz-owned-lora-manager'),'success returns focus into the manager');
 assert.ok(await page.eval("Boolean(document.querySelector('.lora-manager-notice.success[role=status]'))&&!document.querySelector('.xyz-owned-lora-manager [role=alert]')"),'success is a status, never a red alert');
 const copied=await liveDraft("b=>b.loras.length===9&&b.group_snapshots.length===1");
 const mine=copied.loras.slice(1);
 assert.deepEqual(mine.map(i=>i.weight),[0.6,0.6,0.6,0.6,0.6,0,-1.2345,0.6]);assert.deepEqual(mine.map(i=>i.enabled),[true,true,true,false,true,true,true,true]);assert.equal(mine[6].precision,2);
 assert.ok(mine.every(i=>i.identitySchema===1&&/^[0-9a-f-]{36}$/.test(i.instanceId)&&/^[0-9a-f]{64}$/.test(i.contentHash)&&i.contentRevision===2&&i.groupId==='g-nkstyle'));
 assert.equal(copied.group_snapshots[0].members.length,8);assert.equal(copied.group_snapshots[0].enabled,true);assert.equal((copied.parameters.prompt.match(/nk style words/g)||[]).length,1);
 assert.deepEqual(await page.eval("window.__fixtureAppState().workspace.loraGroupsByEngine"),groupsBefore,'real group library untouched');
 ledgerCheck('G01/G04/G07 legacy shared eight-member group: per-member binding-only resolve, preview, confirm, exact weights/precision/enabled/order/text, library untouched, success status');

 // ---- S2 failures stay in place and are repairable by retry
 await page.click(copyButton('Broken refs'));await page.wait("document.querySelectorAll('.xyz-copy-issues li').length===2");
 const issues=await page.eval("document.querySelector('.xyz-group-copy-dialog').textContent");
 for(const part of ['ambiguous_two','never_registered','路径被复用','没有登记','均未改动'])assert.ok(issues.includes(part),part);
 assert.equal(await page.eval("[...document.querySelectorAll('.xyz-group-copy-dialog footer button')].map(b=>b.textContent).join('|')"),'取消|重新核验');
 await page.shot('s2-blocked-1440');
 await page.eval("window.__resolve.ambiguous.clear();window.__resolve.fallback.clear();0");
 await page.click(byText('.xyz-group-copy-dialog footer button','重新核验'));await page.wait("document.querySelectorAll('.xyz-copy-bindings li').length===4");
 await page.key('Escape');await page.wait(`!${hasDialog}`);
 assert.equal((await draft()).group_snapshots.length,1,'a cancelled or retried verification writes nothing');
 ledgerCheck('G03/G09 ambiguous and unregistered members are named with guidance, nothing is registered or written, retry works in place');

 // ---- S3 equal-content group-owned conflict: copy only; explicit later enable is refused with an error notice
 await page.click(copyButton('NekoX V2'));await page.wait("document.querySelectorAll('.xyz-copy-bindings li').length===3");
 const conflict=await page.eval("document.querySelector('.xyz-group-copy-dialog').textContent");
 assert.ok(conflict.includes('NKStyleV2')&&conflict.includes('内容相同')&&conflict.includes('共享库')&&conflict.includes('本地'));
 assert.equal(await page.eval("[...document.querySelectorAll('.xyz-group-copy-dialog footer button')].map(b=>b.textContent+(b.disabled?'(off)':'')).join('|')"),'取消|仅复制（不启用）|复制并启用快照(off)');
 await page.shot('s3-conflict-1440');
 await page.click(byText('.xyz-group-copy-dialog footer button','仅复制'));await page.wait(`!${hasDialog}`);
 const kept=await liveDraft("b=>b.group_snapshots.length===2");
 assert.equal(kept.loras.length,9);assert.equal(kept.group_snapshots[1].enabled,false);
 assert.ok(await page.eval("document.querySelector('.lora-manager-notice.success').textContent.includes('未启用')"));
 ledgerCheck('G06 equal-content copy owned by another enabled group blocks enabling; copy-only keeps both groups and mounts nothing');

 // ---- S4 notice geometry in groups and mounted views, three viewports
 for(const [width,height] of [[1440,900],[390,844],[390,430]]){
  await page.viewport(width,height);await category('已挂载');
  await page.eval("document.querySelector('.lora-manager-notice > button')?.click();0");await sleep(200);
  const quiet=await page.eval(geometry);
  await category('组合');
  await page.click(`(()=>{const card=[...document.querySelectorAll('.xyz-owned-lora-manager .lora-group-card')].find(e=>e.textContent.includes('NekoX V2'));return card.querySelector('[role=switch]');})()`);
  await page.wait("Boolean(document.querySelector('.lora-manager-notice.error[role=alert]'))");
  for(const where of ['组合','已挂载']){
   await category(where);
   const g=await page.eval(geometry);
   assert.equal(g.alerts,1,`${width}x${height} ${where}: exactly one announced alert`);
   assert.ok(g.notice.top>=g.content.bottom-1,`${width}x${height} ${where}: notice sits below the content`);
   assert.ok(g.notice.bottom<=g.foot.top+1,`${width}x${height} ${where}: notice stays above the footer`);
   assert.ok(g.foot.bottom<=g.vh+1&&g.foot.top>=0,`${width}x${height} ${where}: footer reachable`);
   assert.ok(Math.abs(g.content.height+g.notice.height-quiet.content.height)<=1.5,`${width}x${height} ${where}: the notice takes only its own height from the list (${g.content.height}+${g.notice.height} vs ${quiet.content.height})`);
   assert.ok(g.content.height>=20,`${width}x${height} ${where}: content keeps a reachable strip`);
   assert.ok(await page.eval("(()=>{const b=document.querySelector('.lora-manager-notice > button').getBoundingClientRect(),h=document.elementFromPoint(b.x+b.width/2,b.y+b.height/2);return h&&document.querySelector('.lora-manager-notice > button').contains(h);})()"),`${width}x${height} ${where}: dismiss button is hit-test reachable`);
   assert.ok(g.notice.scrollW<=g.notice.clientW+1,`${width}x${height} ${where}: notice does not overflow sideways`);
   assert.equal(Math.round(g.h2.height),Math.round(quiet.h2.height),`${width}x${height} ${where}: title keeps its height`);assert.equal(Math.round(g.h2.width),Math.round(quiet.h2.width),`${width}x${height} ${where}: title keeps its width`);
   assert.ok(await page.eval("document.querySelector('.xyz-owned-lora-manager .lora-modal-head h2').textContent==='全局 LoRA'"),'title is never replaced by the error');
   assert.equal(await page.eval("getComputedStyle(document.querySelector('.lora-manager-notice')).fontSize"),height<=600?'10px':'11px');
   await page.shot(`s4-notice-${where==='组合'?'groups':'mounted'}-${width}x${height}`);
  }
 }
 await page.viewport(1440,900);
 await page.click("document.querySelector('.lora-manager-notice > button')");assert.equal(await page.eval("Boolean(document.querySelector('.lora-manager-notice'))"),false);
 ledgerCheck('G10/G11/G12 error notice: own slot below content, one alert, title/footer intact, themed, at 1440x900, 390x844 and 390x430 in both groups and mounted views; dismiss works');
 await page.key('Escape');await page.wait("!document.querySelector('.xyz-owned-lora-manager')");

 // ---- S5 standalone conflicts need an explicit replace choice; weights never add
 await openFresh();await openManager();await toGroups();
 await page.click(copyButton('Portrait copies'));await page.wait("document.querySelectorAll('.xyz-copy-bindings li').length===2");
 const replaceText=await page.eval("document.querySelector('.xyz-group-copy-dialog').textContent");
 assert.ok(replaceText.includes('内容相同的另一个副本已独立挂载')&&replaceText.includes('现有权重 0.8137')&&replaceText.includes('权重不会相加'));
 assert.equal(await page.eval("[...document.querySelectorAll('.xyz-group-copy-dialog footer button')].map(b=>b.textContent+(b.disabled?'(off)':'')).join('|')"),'取消|仅复制（不启用）|替换 1 项并启用');
 await page.shot('s5-replace-1440');
 await page.click(byText('.xyz-group-copy-dialog .primary','替换 1 项并启用'));await page.wait(`!${hasDialog}`);
 const replaced=await liveDraft("b=>b.group_snapshots.length===1");
 assert.deepEqual(replaced.loras.map(i=>[i.weight,i.groupId]),[[0.3,'g-copies'],[0.4,'g-copies']]);
 ledgerCheck('G06 standalone equal-content copy is replaced only by the explicit button; the group weights win and are never summed');

 // ---- S6 drift during confirmation keeps the dialog open; double click commits once
 await openFresh();await openManager();await toGroups();
 await page.eval("window.__resolveCalls.length=0;window.__resolve.drift={after:8,name:'ashima_v4'};0");
 await page.click(copyButton('NKStyleV2'));await page.wait("document.querySelectorAll('.xyz-copy-bindings li').length===8");
 const previewed=await page.eval("document.querySelector('.xyz-copy-bindings li').textContent");
 await page.click(byText('.xyz-group-copy-dialog .primary','复制并启用快照'));
 await page.wait("Boolean(document.querySelector('.xyz-copy-error[role=alert]'))");
 assert.ok(await page.eval("document.querySelector('.xyz-copy-error').textContent.includes('资产库在确认期间发生了变化')"));
 assert.equal(await page.eval(hasDialog),true,'a failed confirmation stays open');
 assert.ok(await focusInside('.xyz-group-copy-dialog'));
 assert.equal(((await draft())?.group_snapshots||[]).length,0,'nothing was written');
 // The library now consistently reports the drifted binding the dialog is showing, so confirming it is accepted.
 await page.eval("window.__resolve.drift={after:0,name:'ashima_v4'};window.__resolve.gate=new Promise(r=>{window.__releaseResolve=r;});window.__resolveCalls.length=0;0");
 await page.eval("(()=>{const b=[...document.querySelectorAll('.xyz-group-copy-dialog .primary')].find(e=>e.textContent.includes('复制并启用快照'));b.click();b.click();})();0");
 await sleep(300);
 assert.equal(await page.eval("document.querySelector('.xyz-group-copy-dialog').getAttribute('aria-busy')"),'true');
 await page.eval("window.__releaseResolve();0");await page.wait(`!${hasDialog}`);
 assert.equal(await page.eval("window.__resolveCalls.length"),8,'double click re-verifies once');
 const once=await liveDraft("b=>b.group_snapshots.length===1");assert.equal(once.loras.length,9);
 ledgerCheck('G08/G09 changed evidence at confirmation keeps the dialog open in place; a double click verifies and commits exactly once');

 assert.deepEqual(page.errors,[]);assert.deepEqual(await page.eval('window.__xyzHostErrors'),[]);
 ledger.status='PASS';
} catch(error){ledger.status='FAIL';ledger.error=error.stack;if(page){await page.shot('failure').catch(()=>{});ledger.debug=await page.eval("({dialog:document.querySelector('.xyz-group-copy-dialog')?.textContent,notice:document.querySelector('.lora-manager-notice')?.textContent})").catch(()=>null);}}
finally {
 await writeFile(path.join(EVIDENCE,'RESULT.json'),JSON.stringify(ledger,null,2));
 console.log(JSON.stringify({evidence:EVIDENCE,status:ledger.status,checks:ledger.checks,error:ledger.error},null,2));
 page?.ws.close();if(edge?.pid)spawn('taskkill',['/PID',String(edge.pid),'/T','/F'],{stdio:'ignore',windowsHide:true});await vite.close();
}
