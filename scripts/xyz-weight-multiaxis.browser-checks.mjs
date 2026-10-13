import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';

/** Real App callbacks + closed private catalog/transport. Captures are then compiled in strict Python. */
export async function weightMultiaxisChecks({page,ledger,EVIDENCE,byText,privateState,original}) {
 const state=async()=>{await new Promise(r=>setTimeout(r,650));return page.eval('window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima');};
 const confirm=async()=>{if(await page.eval("Boolean(document.querySelector('.xyz-axis-confirm'))"))await page.click(byText('.xyz-axis-confirm footer button','确认更改'));};
 const field=async(key,text)=>{await page.choose(`${key} 轴参数`,text);await confirm();};
 const manager=()=>page.click("document.querySelector('.xyz-sidebar-mounts .lora-title button')");
 const closeManager=()=>page.click("document.querySelector('[aria-label=\"关闭 LoRA 管理器\"]')");
 const chooseTarget=async(key,name,cancel=false)=>{
   const before=await state();await page.click(`document.querySelector('[aria-label="${key} LoRA 目标实例"]')`);
   await page.click(byText('.lora-categories button','共享'));await page.click("document.querySelector('.lora-modal-shared-folder button')");
   await page.click(byText('.lora-library-card',name));assert.equal(await page.eval("document.querySelectorAll('.lora-library-card.mounted').length"),1);
   if(cancel)await page.click(byText('.lora-modal-foot button','取消'));else await page.click(byText('.lora-modal-foot button','确认轴选择'));
   const after=await state();assert.deepEqual(after.baseline,before.baseline);
   for(const other of ['x','y','z'])if(other!==key.toLowerCase())assert.deepEqual(after.axes[other],before.axes[other]);
   if(cancel)assert.deepEqual(after,before);
 };
  const ladder=async(key)=>{await page.click(byText(`.xyz-axis-card:nth-child(${['X','Y','Z'].indexOf(key)+1}) .xyz-axis-modes button`,'阶梯数值'));await confirm();for(const[k,v]of[['起点','0.2'],['终点','1'],['步长','0.3']])await page.fill(`[aria-label="${key} 阶梯${k}"]`,v);assert.equal(await page.eval(`document.querySelector('[aria-label="${key} 阶梯预览"]').textContent`),'0.2，0.5，0.8');};
 const mode=async(key,label)=>{await page.click(byText(`.xyz-axis-card:nth-child(${['X','Y','Z'].indexOf(key)+1}) .xyz-axis-modes button`,label));await confirm();};
 const list=async(key,values)=>{await mode(key,'自定义数值');await page.fill(`[aria-label="${key} 预期数量"]`,String(values.length));await page.key('Enter');await confirm();for(const[i,value]of values.entries())await page.fill(`[aria-label="${key} 第 ${i+1} 项"]`,String(value));};
 const range=async(key,start,end,step,preview)=>{await mode(key,'阶梯数值');for(const[k,value]of[['起点',start],['终点',end],['步长',step]])await page.fill(`[aria-label="${key} 阶梯${k}"]`,String(value));if(preview)assert.equal(await page.eval(`document.querySelector('[aria-label="${key} 阶梯预览"]').textContent`),preview);};
 const cases=[],bodies={};
 const registry=async()=>({schemaVersion:1,instances:Object.fromEntries((await page.eval('window.__weightCatalog.Anima')).map(m=>[m.instanceId,{...m,engine:'Anima',status:'present'}]))});
 const submit=async(name,wire)=>{
   const count=await page.eval('window.__xyzSubmitBodies?.length||0');await page.click(byText('.xyz-run-actions button','开始 XYZ'));await page.wait(`window.__xyzSubmitBodies?.length===${count+1}`);await page.wait("Boolean(document.querySelector('.xyz-results'))");
   const dto=await page.eval('window.__xyzSubmitBodies.at(-1)');assert.deepEqual(dto.draft,wire);assert.deepEqual((await state()).baseline,cases.find(c=>c.name===name).uiBaseline);cases.find(c=>c.name===name).submitDTO=dto;
   await writeFile(path.join(EVIDENCE,`${name}-submit-DTO.json`),JSON.stringify(dto,null,2));await page.click(byText('.xyz-tabs button','配置'));await page.wait("Boolean(document.querySelector('.xyz-axes'))");
 };
 const capture=async(name,cells,expected={},alsoSubmit=false)=>{
   const ui=await state(), count=await page.eval('window.__xyzPlanBodies?.length||0');
   const summary=await page.eval("import('/src/xyz-draft.js').then(m=>m.xyzLocalSummary(window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima))");assert.equal(summary.valid,true);assert.equal(summary.total,cells);
   assert.equal(await page.eval("document.querySelector('.xyz-run-actions button').disabled"),false);
   await page.click(byText('.xyz-run-actions button','服务端预检'));await page.wait(`window.__xyzPlanBodies?.length===${count+1}`);
   const wire=await page.eval('window.__xyzPlanBodies.at(-1)');bodies[name]=wire;
   assert.deepEqual(wire.baseline.loras.slice(0,ui.baseline.loras.length),ui.baseline.loras);
   assert.equal(wire.baseline.loras.length,ui.baseline.loras.length+new Set(Object.values(ui.axes).filter(a=>a?.field==='lora_weight').filter(a=>!ui.baseline.loras.some(m=>m.instanceId===a.target)).map(a=>a.target)).size);
   for(const key of ['x','y','z'])if(ui.axes[key]){const {editor_source,...axis}=ui.axes[key];assert.deepEqual(wire.axes[key],axis);assert.ok(['lora_weight','steps','cfg'].includes(axis.field));}
   assert.deepEqual((await state()).baseline,ui.baseline);
   cases.push({name,wire,uiDraft:ui,uiBaseline:ui.baseline,registry:await registry(),expected:{cells,...expected}});ledger.metrics[name]={summary,uiBaseline:ui.baseline,wireLoras:wire.baseline.loras};if(alsoSubmit)await submit(name,wire);return wire;
 };
 const blockedCases=[];
 const token=value=>value===''?'empty':String(value);
 const blockedNumeric=async(name,error,partialCells)=>{
   const ui=await state(), before=await page.eval("window.__xyzCalls.filter(c=>c.method==='POST').length");
   const summary=await page.eval("import('/src/xyz-draft.js').then(m=>m.xyzLocalSummary(window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima))");assert.equal(summary.valid,false);
   await page.click(byText('.xyz-run-actions button','服务端预检'));await page.click(byText('.xyz-run-actions button','开始 XYZ'));
   assert.equal(await page.eval("window.__xyzCalls.filter(c=>c.method==='POST').length"),before);assert.equal(await page.eval("document.querySelector('.xyz-run-actions button').disabled && document.querySelector('.xyz-run-actions .primary').disabled"),true);
   assert.deepEqual((await state()).baseline,ui.baseline);const wire=await page.eval("import('/src/xyz-prompt-source.js').then(m=>m.xyzWireDraft(window.__fixtureAppState().workspace.xyzByEngine.by_engine.Anima))");
   blockedCases.push({name,summary,postDelta:0,uiDraft:ui,wire});cases.push({name,wire,uiDraft:ui,uiBaseline:ui.baseline,registry:await registry(),expected:{error,...(partialCells===undefined?{}:{partialCells})}});
 };
 await page.click(byText('.model-seg','Anima'));await page.wait("document.querySelector('.model-seg.active').textContent.includes('Anima')");
 await manager();await page.click(byText('.lora-categories button','已挂载'));await page.click("document.querySelector('.lora-mounted-item .remove')");await closeManager();assert.equal((await state()).baseline.loras.length,0);
 await field('X','LoRA 实例权重');await chooseTarget('X','kiriaina_anima_style-000030',true);assert.equal((await state()).axes.x.target??null,null);
 await chooseTarget('X','kiriaina_anima_style-000030');await ladder('X');
 await field('Y','LoRA 实例权重');await chooseTarget('Y','yusano_v5_step1500');await ladder('Y');
 const user=await capture('user-nine',9,{},true);assert.equal(user.baseline.loras.length,2);assert.equal(user.baseline.parameters.engine,'Anima');await page.shot('user-nine-global-zero');
 await field('Z','LoRA 实例权重');await chooseTarget('Z','third-independent');await ladder('Z');await capture('triple-27',27);
 // Added user coverage: ordered custom weight lists, steps lists and range step are different inputs.
 await list('X',[0,.3,.9]);await list('Y',[.2,.7]);await list('Z',[-.1,1]);await capture('custom-three-weight-lists',12,{axisValues:{x:[0,.3,.9],y:[.2,.7],z:[-.1,1]}},true);
 await field('X','步数');await list('X',[15,23,28]);await list('Y',[0,.3,.9]);await capture('custom-steps-two-weight-lists',18,{axisValues:{x:[15,23,28],y:[0,.3,.9],z:[-.1,1]}},true);
 await range('X',12,20,2,'12，14，16，18，20');await ladder('Z');await capture('steps-range-weight-list-and-range',45,{axisValues:{x:[12,14,16,18,20],y:[0,.3,.9],z:[.2,.5,.8]}},true);
 await field('X','CFG');await range('X',4,6.5,.5,'4.0，4.5，5.0，5.5，6.0，6.5');await capture('cfg-float-range-two-weight-axes',54,{axisValues:{x:[4,4.5,5,5.5,6,6.5],y:[0,.3,.9],z:[.2,.5,.8]}},true);
 await field('X','步数');await list('X',[15,23,28]);await field('Y','步数');await list('Y',[10,20]);await list('Z',[-.1,1]);await capture('two-steps-writers-rejected',12,{error:'axis-conflict'});
 await field('X','LoRA 实例权重');await chooseTarget('X','kiriaina_anima_style-000030');await list('X',[0,.3,.9]);await field('Y','LoRA 实例权重');await chooseTarget('Y','kiriaina_anima_style-000030');await list('Y',[.2,.7]);await capture('two-same-instance-custom-writers-rejected',12,{error:'axis-conflict'});
 await field('X','步数');await list('X',[15]);await chooseTarget('Y','yusano_v5_step1500');await list('Y',[.2,.7]);
 for(const[value,error]of [[0,'bounds'],[-1,'bounds'],[101,'bounds'],[15.5,'bounds'],['','empty-value'],['0.00000000000000001','number'],['NaN','number']]){await list('X',[value]);await blockedNumeric(`steps-list-${token(value)}`,error);}
 await list('X',[1,100]);await capture('steps-list-min-max-boundary',8,{axisValues:{x:[1,100],y:[.2,.7],z:[-.1,1]}},true);
 await list('X',[15]);
 for(const[value,error]of [['-5.0001','bounds'],['5.0001','bounds'],['0.12345','precision'],['','empty-value'],['NaN','number']]){await list('Y',[value]);await blockedNumeric(`weight-list-${token(value)}`,error);}
 await list('Y',[-5,5]);await capture('weight-list-min-max-boundary',4,{axisValues:{x:[15],y:[-5,5],z:[-.1,1]}},true);await list('Y',[.2,.7]);
 for(const[start,end,step,error,partial]of [[0,0,1,'bounds'],[101,101,1,'bounds'],[12.5,12.5,1,'bounds'],[12,20,0,'range'],[12,20,-1,'range'],['',20,2,'number'],[12,'',2,'number'],[12,20,'','number'],[12,20,'0.00000000000000001','number'],[12,13,.5,'bounds',8]]){await range('X',start,end,step);await blockedNumeric(`steps-range-${token(start)}-${token(end)}-${token(step)}`,error,partial);}
 // Restore the original valid ranges before the original global/cap checks; no original assertion is removed.
 await field('X','LoRA 实例权重');await chooseTarget('X','kiriaina_anima_style-000030');await ladder('X');await ladder('Y');await field('Z','不使用');
 await manager();await page.click(byText('.lora-categories button','组合'));await page.click("document.querySelector('[aria-label=\"复制 Library Anima 到 XYZ\"]')");await page.wait("Boolean(document.querySelector('.xyz-group-copy-dialog .primary'))");await page.click("document.querySelector('.xyz-group-copy-dialog .primary')");await closeManager();
 const global=(await state()).baseline;assert.equal(global.loras.length,5);assert.ok(global.loras.some(m=>!m.enabled));assert.ok(global.loras.some(m=>m.weight===0));assert.equal(global.parameters.prompt.split('group preset').length-1,1);
  await capture('five-global-nine',9);await field('Z','LoRA 实例权重');await chooseTarget('Z','third-independent');await ladder('Z');await capture('five-global-triple',27);await page.shot('five-global-plus-three-targets');
 await list('X',[0,.3,.9]);await list('Y',[.2,.7]);await list('Z',[-.1,1]);await capture('five-global-custom-three-weight-lists',12,{axisValues:{x:[0,.3,.9],y:[.2,.7],z:[-.1,1]}},true);
 // Exactly 16 global mounts + external targets: real preflight/start attempts emit no plan or submit POST.
 await manager();await page.click(byText('.lora-categories button','上限夹具'));for(let i=0;i<11;i++)await page.click(`document.querySelectorAll('.lora-library-card')[${i}]`);await closeManager();assert.equal((await state()).baseline.loras.length,16);
 const posts=await page.eval("window.__xyzCalls.filter(c=>c.method==='POST').length");
 await page.click(byText('.xyz-run-actions button','服务端预检'));await page.click(byText('.xyz-run-actions button','开始 XYZ'));
 const blocked=await page.eval("({preflight:document.querySelector('.xyz-run-actions button').disabled,start:document.querySelector('.xyz-run-actions .primary').disabled,status:document.querySelector('.xyz-run-status').textContent,posts:window.__xyzCalls.filter(c=>c.method==='POST').length})");
 assert.equal(blocked.preflight,true);assert.equal(blocked.start,true);assert.match(blocked.status,/16/);assert.equal(blocked.posts,posts);ledger.metrics.mountCap={...blocked,postDelta:0};await page.shot('injected-over-cap-zero-post');
 assert.deepEqual(await page.eval(privateState),original);await writeFile(path.join(EVIDENCE,'basic-after.json'),JSON.stringify(await page.eval(privateState),null,2));
 await writeFile(path.join(EVIDENCE,'weight-multiaxis-wire.json'),JSON.stringify({cases},null,2));await writeFile(path.join(EVIDENCE,'captured-wire.json'),JSON.stringify(bodies,null,2));
  await writeFile(path.join(EVIDENCE,'closed-api-calls.json'),JSON.stringify(await page.eval('({app:window.__fixtureAppCalls,xyz:window.__xyzCalls})'),null,2));
 await writeFile(path.join(EVIDENCE,'numeric-blocked-cases.json'),JSON.stringify(blockedCases,null,2));
 ledger.checks.push('custom X/Y/Z weights .0/.3/.9 × .2/.7 × -.1/1 -> 12 cells; steps custom15/23/28 + 2 independent weight lists -> 18; steps12..20 by2 + list/range ->45; CFG4..6.5 by.5 ->6 values/54cells; all positive plan/submit drafts byte-equal','same steps/same LoRA writer conflicts stay authoritative; min/max/empty/integer/decimal precision/zero-or-wrong-direction range step negatives block both actions with zero POST');
 ledger.checks.push('actual Anima App: global0 + kiriaina/yusano independent ranges .2..1 step.3 -> 9-cell plan+submit DTO, .2/.5/.8 unchanged','third independent target -> 27 cells; 5 global mounts with order/negative/disabled/zero/precision/group preset -> nine/triple DTOs without writeback','axis cancel/confirm never writes global, other axes or Basic/i2i/groups; 16 global + injection blocks both buttons with zero POST');ledger.status='PASS';
}
