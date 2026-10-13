// Live CPU tags first; race/large-cache/folder/install scenarios explicitly mocked.
// No service startup, no HF transfer, no server TXT output. Evidence/profile/exports are temp-only.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, writeFile, readFile, readdir } from 'node:fs/promises';
import { createServer } from 'node:net';
import path from 'node:path';
const origin=process.env.WD14_UI_ORIGIN || 'http://127.0.0.1:7709', out=process.env.WD14_UI_EVIDENCE;
assert.ok(out && process.env.LOCALAPPDATA && path.resolve(out).toLowerCase().startsWith((path.join(process.env.LOCALAPPDATA, 'Temp', 'opencode') + path.sep).toLowerCase()));
assert.ok(['127.0.0.1','localhost'].includes(new URL(origin).hostname));
await mkdir(out,{recursive:true});await mkdir(path.join(out,'exports'),{recursive:true});
const server=createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;await new Promise(r=>server.close(r));
const browser=spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',['--headless=new','--disable-gpu','--disable-extensions','--window-size=1440,900','--no-first-run','--no-default-browser-check',`--remote-debugging-port=${port}`,`--user-data-dir=${path.join(out,'profile')}`,'about:blank'],{stdio:'ignore',windowsHide:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function wait(fn,timeout=45000){const end=Date.now()+timeout;while(Date.now()<end){if(await fn())return;await sleep(80);}throw new Error('Timeout: '+fn.toString());}
const report={checks:[],screenshots:[],catalogRequests:[],realTags:[],mockTags:[],lateResponses:[],errors:[]};
let ws,debug,debugShot,tagMode=process.env.WD14_UI_MOCK_TAGS==='1'?'mock':'real',rules=[],noModel=false,recovery=false,install=null,batch=null,batchBody=null,publishPolls=0;
const result=(text,count=1)=>({rating:{general:.8,sensitive:.1,questionable:.07,explicit:.03},tags:Array.from({length:count},(_,i)=>({name:count===1?text:`${text}_${i}_long_confidence_tag`,confidence:.9}))});
const tagRequestCount=()=>report.realTags.length+report.mockTags.length;
try{
  let tab;await wait(async()=>{try{tab=(await(await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(x=>x.type==='page');return tab;}catch{return false;}});
  ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.addEventListener('open',r,{once:true});ws.addEventListener('error',j,{once:true});});
  let serial=0;const pending=new Map();
  const send=(method,params={})=>{const id=++serial;return new Promise((r,j)=>{pending.set(id,{r,j});ws.send(JSON.stringify({id,method,params}));});};
  const fulfill=(requestId,body,status=200)=>send('Fetch.fulfillRequest',{requestId,responseCode:status,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify(body)).toString('base64')});
  ws.addEventListener('message',event=>{
    const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p?.j(new Error(m.error.message));else p?.r(m.result);}
    if(m.method==='Runtime.exceptionThrown')report.errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if(m.method==='Fetch.requestPaused'){
      const {requestId,request}=m.params,url=new URL(request.url);
      void(async()=>{
        if(url.pathname.endsWith('/tag')){
          if(tagMode==='real'){report.realTags.push({url:request.url});await send('Fetch.continueRequest',{requestId});return;}
          const rule=rules.shift() || {text:'large',count:500};report.mockTags.push({url:request.url,rule,started:Date.now()});
          await sleep(rule.delay || 0);
          report.lateResponses.push({text:rule.text,attempted:Date.now()});
          try{await fulfill(requestId,rule.error?{detail:rule.error}:result(rule.text,rule.count),rule.error?500:200);}catch(e){report.lateResponses.at(-1).aborted=true;}
        }else if(url.pathname.endsWith('/models')){
          report.catalogRequests.push(request.url);
          if(!noModel){await send('Fetch.continueRequest',{requestId});return;}
          const value=await(await fetch(origin+'/api/inference/wd14/models')).json();value.models=value.models.map(x=>({...x,installed:false}));await fulfill(requestId,value);
        }else if(url.pathname.endsWith('/active'))await fulfill(requestId,{jobs:recovery?[{id:'mock-recovery',kind:'download',state:'error',total:2,done:0,warning:'历史 marker 需要人工恢复；保留现场',error:'安装中断'}]:[]});
        else if(url.pathname.includes('/download/')){install={id:'mock-install',kind:'download',state:'running',done:1024,total:2048};await fulfill(requestId,install);}
        else if(url.pathname.endsWith('/mock-install/cancel')){install.state='publishing';await fulfill(requestId,{detail:'publishing'},409);}
        else if(url.pathname.endsWith('/mock-install')){const body={...install};if(body.state==='publishing' && ++publishPolls>=2){install.state='done';install.warning='历史备份请人工检查，不自动删除';}await fulfill(requestId,body);}
        else if(url.pathname.endsWith('/batch')){batchBody=JSON.parse(request.postData);batch={id:'mock-batch',kind:'batch',state:'running',done:1,skipped:1,failed:0,total:3,current:'synthetic.png',logs:['UI mock：没有写入用户 TXT']};await fulfill(requestId,batch,202);}
        else if(url.pathname.endsWith('/mock-batch/cancel')){batch.state='cancelled';await fulfill(requestId,{ok:true,state:'cancelled'});}
        else if(url.pathname.endsWith('/mock-batch'))await fulfill(requestId,batch);
        else await send('Fetch.continueRequest',{requestId});
      })().catch(e=>report.errors.push(e.message));
    }
  });
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);return r.result.value;};
  debug=evaluate;
  const check=(name,data=true)=>{report.checks.push({name,data});console.log('PASS',name,JSON.stringify(data));};
  const shot=async name=>{const s=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});const file=path.join(out,name+'.png');await writeFile(file,Buffer.from(s.data,'base64'));report.screenshots.push(file);};
  debugShot=shot;
  const click=async(selector,text)=>evaluate(`[...document.querySelectorAll(${JSON.stringify(selector)})].find(x=>x.textContent.includes(${JSON.stringify(text)})).click()`);
  const fill=async(selector,value)=>evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});Object.getOwnPropertyDescriptor(e.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  const select=async(selector,value)=>evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const view=async(w,h)=>{await send('Emulation.setDeviceMetricsOverride',{width:w,height:h,deviceScaleFactor:1,mobile:false});await sleep(150);};
  const nav=async text=>{await click('.main-nav button',text);await sleep(250);};
  const count=()=>evaluate("document.querySelectorAll('.wd14-history-strip button').length");
  const prompt=()=>evaluate("document.querySelector('.wd14-editor textarea').value");
  const add=async(name,color='#994840',kind='choose')=>evaluate(`(async()=>{const c=document.createElement('canvas');c.width=64;c.height=96;const x=c.getContext('2d');x.fillStyle=${JSON.stringify(color)};x.fillRect(0,0,64,96);x.fillStyle='#eeeecc';x.fillRect(15,15,24,30);const blob=await new Promise(r=>c.toBlob(r,'image/png'));const d=new DataTransfer();d.items.add(new File([blob],${JSON.stringify(name)},{type:'image/png'}));const drop=document.querySelector('.wd14-drop');if(${JSON.stringify(kind)}==='drop')drop.dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:d}));else if(${JSON.stringify(kind)}==='paste'){drop.focus();drop.dispatchEvent(new ClipboardEvent('paste',{bubbles:true,cancelable:true,clipboardData:d}));}else{const input=document.querySelector('.wd14-page input[type=file]');input.files=d.files;input.dispatchEvent(new Event('change',{bubbles:true}));}})()`);
  const done=()=>wait(()=>evaluate("document.querySelector('.wd14-work-foot').textContent.includes('反推完成')"),180000);
  const bounds=()=>evaluate("(()=>{const rect=s=>{const e=document.querySelector(s),r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,client:e.clientHeight,scroll:e.scrollHeight}};return {viewport:[innerWidth,innerHeight],documentHeight:document.documentElement.scrollHeight,documentWidth:document.documentElement.scrollWidth,work:rect('.wd14-work'),deck:rect('.wd14-analysis-deck'),history:rect('.wd14-history'),side:rect('.wd14-sidebar'),copy:rect('.wd14-results .wd14-actions'),count:document.querySelectorAll('.wd14-history-strip button').length}})()");
  const fit=async label=>{const b=await bounds();assert.ok(b.work.bottom<=b.viewport[1]);assert.ok(b.history.bottom<=b.viewport[1]);assert.ok(b.copy.bottom<=b.viewport[1]);assert.ok(b.work.scroll<=b.work.client+1);assert.ok(b.documentHeight<=b.viewport[1]);assert.equal(b.documentWidth,b.viewport[0]);assert.ok(b.work.width>b.side.width);
    const panes=await evaluate("(()=>{const s=document.querySelector('.wd14-source').getBoundingClientRect(),d=document.querySelector('.wd14-drop').getBoundingClientRect(),p=document.querySelector('.wd14-results').getBoundingClientRect();return {sourceRight:s.right,dropRight:d.right,promptLeft:p.left,sourceScroll:document.querySelector('.wd14-source').scrollWidth,sourceClient:document.querySelector('.wd14-source').clientWidth}})()");assert.ok(panes.dropRight<=panes.sourceRight+1);assert.ok(panes.dropRight<panes.promptLeft);assert.ok(panes.sourceScroll<=panes.sourceClient+1);
    if(b.count){const thumb=await evaluate("(()=>{const s=document.querySelector('.wd14-history-strip').getBoundingClientRect(),t=document.querySelector('.wd14-history-strip button.active').getBoundingClientRect();return {left:t.left,right:t.right,shelfLeft:s.left,shelfRight:s.right}})()");assert.ok(thumb.left>=thumb.shelfLeft-1 && thumb.right<=thumb.shelfRight+1);b.activeThumb=thumb;}b.panes=panes;check(label,b);};
  await send('Page.enable');await send('Runtime.enable');await send('DOM.enable');await send('Network.enable');await send('Network.setCacheDisabled',{cacheDisabled:true});
  await send('Emulation.setFocusEmulationEnabled',{enabled:true});
  await send('Fetch.enable',{patterns:[{urlPattern:'*/api/inference/wd14/*'}]});
  await send('Browser.grantPermissions',{origin,permissions:['clipboardReadWrite','clipboardSanitizedWrite']});
  await send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:path.join(out,'exports')});
  await view(1440,900);await send('Page.navigate',{url:origin});await wait(()=>evaluate("!!document.querySelector('.main-nav') && !document.querySelector('.app-loader')"));
  await wait(()=>evaluate("!!document.querySelector('.left-panel .model-segment .model-seg.active')"));
  const readT2iStyle=()=>evaluate(`(()=>{const describe=b=>{const s=getComputedStyle(b),n=b.querySelector('.seg-name'),d=b.querySelector('.seg-detail'),ns=getComputedStyle(n),ds=getComputedStyle(d),nr=n.getBoundingClientRect(),dr=d.getBoundingClientRect();return {button:{display:s.display,borderRadius:s.borderRadius,borderTopWidth:s.borderTopWidth,borderLeftWidth:s.borderLeftWidth,borderRightWidth:s.borderRightWidth,borderBottomWidth:s.borderBottomWidth,borderColor:s.borderColor,padding:s.padding,color:s.color,background:s.backgroundColor,fontFamily:s.fontFamily,lineHeight:s.lineHeight},name:{fontFamily:ns.fontFamily,fontSize:ns.fontSize,fontWeight:ns.fontWeight,top:nr.top,bottom:nr.bottom,left:nr.left,right:nr.right},detail:{fontFamily:ds.fontFamily,fontSize:ds.fontSize,fontWeight:ds.fontWeight,top:dr.top,bottom:dr.bottom,left:dr.left,right:dr.right},stacked:dr.top>=nr.bottom-1&&Math.abs(nr.left-dr.left)<1&&Math.abs(nr.right-dr.right)<1};};const grid=document.querySelector('.left-panel .model-segment');return {active:describe(grid.querySelector('.model-seg.active')),inactive:describe(grid.querySelector('.model-seg:not(.active):not(:disabled)')),frame:{background:getComputedStyle(grid).backgroundColor,borderRadius:getComputedStyle(grid).borderRadius,borderColor:getComputedStyle(grid).borderColor}}})()`);
  const t2iReference=await readT2iStyle();assert.ok(t2iReference.inactive,'T2I has an available inactive segment for comparison');
  const t2iHoverPoint=await evaluate("(()=>{const r=document.querySelector('.left-panel .model-segment .model-seg:not(.active):not(:disabled)').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()");await send('Input.dispatchMouseEvent',{type:'mouseMoved',...t2iHoverPoint});await sleep(250);
  t2iReference.hover=await evaluate("(()=>{const s=getComputedStyle(document.querySelector('.left-panel .model-segment .model-seg:not(.active):not(:disabled)'));return {color:s.color,background:s.backgroundColor}})()");
  await shot('t2i-reference-1440x900');
  await evaluate("window.__urls={made:[],revoked:[]};const c=URL.createObjectURL.bind(URL),r=URL.revokeObjectURL.bind(URL);URL.createObjectURL=b=>{const u=c(b);__urls.made.push(u);return u};URL.revokeObjectURL=u=>{__urls.revoked.push(u);r(u)}");
  await nav('图片反推');await wait(()=>evaluate("!!document.querySelector('.wd14-status .ready')"));
  const catalogCallsAfterMount=report.catalogRequests.length;
  assert.equal(await evaluate("document.querySelectorAll('.wd14-controls select')[0].options.length"),16);
  assert.equal(await evaluate("document.querySelectorAll('.wd14-source,.wd14-results').length"),2);assert.equal(await evaluate("document.querySelector('.wd14-results button').disabled"),true);
  const engineButtons='.wd14-engine-grid button';
  assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll(${JSON.stringify(engineButtons)}),b=>[b.dataset.engine,b.querySelector('.seg-name').textContent,b.querySelector('.seg-detail').textContent])`),[['wd','WD','SmilingWolf'],['cl','CL','cella110n'],['camie','Camie','Camais03'],['pixai','PixAI','PixAI Labs']]);
  assert.equal(await evaluate("getComputedStyle(document.querySelector('.wd14-engine-grid')).gridTemplateColumns.split(' ').length"),2);
  const wdActiveStyle=await evaluate(`(()=>{const describe=b=>{const s=getComputedStyle(b),n=b.querySelector('.seg-name'),d=b.querySelector('.seg-detail'),ns=getComputedStyle(n),ds=getComputedStyle(d),nr=n.getBoundingClientRect(),dr=d.getBoundingClientRect();return {button:{display:s.display,borderRadius:s.borderRadius,borderTopWidth:s.borderTopWidth,borderLeftWidth:s.borderLeftWidth,borderRightWidth:s.borderRightWidth,borderBottomWidth:s.borderBottomWidth,borderColor:s.borderColor,padding:s.padding,color:s.color,background:s.backgroundColor,fontFamily:s.fontFamily,lineHeight:s.lineHeight},name:{fontFamily:ns.fontFamily,fontSize:ns.fontSize,fontWeight:ns.fontWeight,top:nr.top,bottom:nr.bottom,left:nr.left,right:nr.right},detail:{fontFamily:ds.fontFamily,fontSize:ds.fontSize,fontWeight:ds.fontWeight,top:dr.top,bottom:dr.bottom,left:dr.left,right:dr.right},stacked:dr.top>=nr.bottom-1&&Math.abs(nr.left-dr.left)<1&&Math.abs(nr.right-dr.right)<1};};return describe(document.querySelector('.wd14-engine-grid .model-seg.active'))})()`);
  const wdHoverPoint=await evaluate("(()=>{const r=document.querySelector('.wd14-engine-grid button[data-engine=cl]').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()");await send('Input.dispatchMouseEvent',{type:'mouseMoved',...wdHoverPoint});await sleep(250);
  const wdHoverStyle=await evaluate("(()=>{const s=getComputedStyle(document.querySelector('.wd14-engine-grid button[data-engine=cl]'));return {color:s.color,background:s.backgroundColor}})()");assert.deepEqual(wdHoverStyle,t2iReference.hover,'WD actual hovered segment styles should follow T2I hover styles');
  const wdFrameStyle=await evaluate("(()=>{const e=document.querySelector('.wd14-engine-grid'),s=getComputedStyle(e);return {background:s.backgroundColor,borderRadius:s.borderRadius,borderColor:s.borderColor}})()");
  const styleKeys=['display','borderRadius','borderTopWidth','borderLeftWidth','borderRightWidth','borderBottomWidth','borderColor','padding','color','background','fontFamily','lineHeight'];
  const compareSegments=(reference,actual)=>{for(const key of styleKeys)assert.equal(actual.button[key],reference.button[key],`computed ${key} differs (${actual.button[key]} vs ${reference.button[key]})`);for(const part of ['name','detail'])for(const key of ['fontFamily','fontSize','fontWeight'])assert.equal(actual[part][key],reference[part][key],`computed ${part}.${key} differs`);assert.equal(actual.stacked,true,'engine label and author must stack as in T2I');};
  compareSegments(t2iReference.active,wdActiveStyle);assert.deepEqual(wdFrameStyle,t2iReference.frame);assert.equal(wdActiveStyle.stacked,true);
  await evaluate("document.querySelector('.wd14-engine-grid button[data-engine=cl]').click()");await wait(()=>evaluate("document.querySelector('.wd14-engine-grid [aria-pressed=true]')?.dataset.engine==='cl'"));await sleep(250);
  const wdInactiveStyle=await evaluate(`(()=>{const b=document.querySelector('.wd14-engine-grid button[data-engine=wd]'),s=getComputedStyle(b),n=b.querySelector('.seg-name'),d=b.querySelector('.seg-detail'),ns=getComputedStyle(n),ds=getComputedStyle(d),nr=n.getBoundingClientRect(),dr=d.getBoundingClientRect();return {className:b.className,ariaPressed:b.getAttribute('aria-pressed'),button:{display:s.display,borderRadius:s.borderRadius,borderTopWidth:s.borderTopWidth,borderLeftWidth:s.borderLeftWidth,borderRightWidth:s.borderRightWidth,borderBottomWidth:s.borderBottomWidth,borderColor:s.borderColor,padding:s.padding,color:s.color,background:s.backgroundColor,fontFamily:s.fontFamily,lineHeight:s.lineHeight},name:{fontFamily:ns.fontFamily,fontSize:ns.fontSize,fontWeight:ns.fontWeight,top:nr.top,bottom:nr.bottom,left:nr.left,right:nr.right},detail:{fontFamily:ds.fontFamily,fontSize:ds.fontSize,fontWeight:ds.fontWeight,top:dr.top,bottom:dr.bottom,left:dr.left,right:dr.right},stacked:dr.top>=nr.bottom-1&&Math.abs(nr.left-dr.left)<1&&Math.abs(nr.right-dr.right)<1}})()`);
  compareSegments(t2iReference.inactive,wdInactiveStyle);await evaluate("document.querySelector('.wd14-engine-grid button[data-engine=wd]').click()");await wait(()=>evaluate("document.querySelector('.wd14-engine-grid [aria-pressed=true]')?.dataset.engine==='wd'"));
  report.styleComparison={reference:'live T2I .left-panel .model-segment via getComputedStyle',t2i:t2iReference,wd:{active:wdActiveStyle,inactive:wdInactiveStyle,hover:wdHoverStyle,frame:wdFrameStyle},matchedKeys:styleKeys,stackedTitles:true};
  check('live T2I vs WD actual active/inactive/hover computed styles + vertical title/subtitle geometry match',{matchedKeys:styleKeys,active:wdActiveStyle,inactive:wdInactiveStyle,hover:wdHoverStyle,frame:wdFrameStyle});
  const requestsBeforeEngineChanges=tagRequestCount();
  await evaluate("document.querySelector('.wd14-engine-grid button[data-engine=wd]').focus()");await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Tab',code:'Tab'});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Tab',code:'Tab'});
  assert.equal(await evaluate("document.activeElement.dataset.engine"),'cl');assert.notEqual(await evaluate("getComputedStyle(document.activeElement).outlineStyle"),'none');
  await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13,text:'\r',unmodifiedText:'\r'});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13});
  await wait(()=>evaluate("document.querySelector('.wd14-engine-grid [aria-pressed=true]')?.dataset.engine==='cl'"));assert.equal(await evaluate("document.querySelectorAll('.wd14-controls select')[0].options.length"),1);
  for(const [id,count] of [['wd',16],['cl',1],['camie',1],['pixai',1]]){
    const button=`${engineButtons}[data-engine=${id}]`;
    await evaluate(`document.querySelector(${JSON.stringify(button)}).click()`);
    await wait(()=>evaluate(`document.querySelector('.wd14-engine-grid [aria-pressed=true]')?.dataset.engine===${JSON.stringify(id)}`));
    assert.equal(await evaluate("document.querySelectorAll('.wd14-controls select')[0].options.length"),count);
    const expected=await evaluate("document.querySelector('.wd14-controls select').value");assert.ok(expected);
    await evaluate("document.querySelector('.wd14-engine-heading .wd14-info-button').focus()");await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13,text:'\r',unmodifiedText:'\r'});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13});
    await wait(()=>evaluate('!!document.querySelector(".wd14-info-popover")'));
    const intro=await evaluate("document.querySelector('.wd14-info-popover h3').textContent");assert.ok(intro);
    await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape'});await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape'});
    await wait(()=>evaluate('!document.querySelector(".wd14-info-popover")'));
    assert.equal(await evaluate("document.activeElement.classList.contains('wd14-info-button')"),true);
    check(`engine ${id} filters ${count} models; keyboard selection and ${intro} intro Escape/focus restore`,{model:expected,intro});
  }
  assert.equal(tagRequestCount(),requestsBeforeEngineChanges,'engine selection must not auto-tag');
  await evaluate("document.querySelector('.wd14-engine-grid button[data-engine=wd]').click()");await wait(()=>evaluate("document.querySelector('.wd14-engine-grid [aria-pressed=true]')?.dataset.engine==='wd'"));
  await fit('engine selector desktop 1440x900');await shot('engine-selector-1440x900');
  await view(390,844);assert.equal(await evaluate('document.documentElement.scrollWidth'),390);await evaluate("document.querySelector('.wd14-engine-grid').scrollIntoView({block:'center'})");await shot('engine-selector-390x844');
  await view(1440,900);
  if(process.env.WD14_UI_ENGINE_ONLY==='1'){
    assert.equal(report.errors.length,0,report.errors.join('\n'));assert.equal(report.catalogRequests.length,catalogCallsAfterMount);assert.equal(tagRequestCount(),0);check('engine-only browser pass: engine controls do not refetch catalog or emit tags',{catalogRequests:report.catalogRequests,tagRequests:tagRequestCount()});
    await writeFile(path.join(out,'probe.json'),JSON.stringify(report,null,2));ws.close();browser.kill();await Promise.race([new Promise(r=>browser.exitCode!==null?r():browser.once('exit',r)),sleep(5000)]);console.log('Evidence',path.join(out,'probe.json'));process.exit(0);
  }
  await fit('empty always has two panes, complete 1440 desktop');await shot('1440-empty-two-panes');
  await add('real-first.png');await done();assert.equal(await count(),1);const first=await prompt();await fill('.wd14-editor textarea',first+', edited_first');
  await add('real-second.png','#326d70');await done();assert.equal(await count(),2);assert.equal(await evaluate("document.querySelector('.wd14-source-caption span').textContent"),'real-second.png');assert.ok(!(await prompt()).includes('edited_first'));
  await evaluate("document.querySelector('.wd14-history-strip button').click()");assert.equal(await prompt(),first+', edited_first');
  const sourceBeforeControlChanges=await evaluate("document.querySelector('.wd14-results .wd14-pane-head small').textContent"),requestsBeforeControlChanges=tagRequestCount();
  for(const id of ['cl','camie','pixai','wd'])await evaluate(`document.querySelector(${JSON.stringify(`${engineButtons}[data-engine=${id}]`)}).click()`);
  assert.equal(await prompt(),first+', edited_first');assert.equal(await evaluate("document.querySelector('.wd14-results .wd14-pane-head small').textContent"),sourceBeforeControlChanges);assert.equal(tagRequestCount(),requestsBeforeControlChanges);
  check('switching engine controls leaves historical Prompt/source unchanged and does not tag',{source:sourceBeforeControlChanges});
  const firstId=await evaluate("document.querySelector('.wd14-history-strip button.active').dataset.entryId");
  await fill('.wd14-slider input','0.60');const n=tagRequestCount();await sleep(300);assert.equal(tagRequestCount(),n);
  await click('.wd14-work-head button','重新反推');await done();assert.equal(await count(),2);assert.equal(await evaluate("document.querySelector('.wd14-history-strip button.active').dataset.entryId"),firstId);
  const initialTagRequests=report.realTags.length?report.realTags:report.mockTags;
  assert.ok(initialTagRequests.at(-1).url.includes('general=0.6'));assert.equal(tagRequestCount(),3);
  check(`${tagMode==='real'?'REAL':'MOCK'} auto tags on two files + one current-only rerun, edits restore, no duplicate`,{history:await count(),requests:initialTagRequests});
  const edited=(await prompt())+', exported_edit';await fill('.wd14-editor textarea',edited);await sleep(100);await send('Page.bringToFront');
  const copyPoint=await evaluate("(()=>{const r=document.querySelector('.wd14-results button').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()");
  await send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...copyPoint});await send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...copyPoint});
  await wait(()=>evaluate("document.querySelector('.wd14-work-foot').textContent.includes('已复制')"));await wait(async()=>(await evaluate('navigator.clipboard.readText()'))===edited,5000);
  await click('.wd14-results button','导出 TXT');await wait(async()=>(await readdir(path.join(out,'exports'))).includes('real-first.txt'));assert.equal(await readFile(path.join(out,'exports','real-first.txt'),'utf8'),edited);
  await fit('real result first-screen bounds');await shot('1440-real-auto-result');
  tagMode='mock';
  rules=[{text:'late_A',delay:1300},{text:'fast_B',delay:30}];const base=report.mockTags.length;
  await add('race-A.png');await wait(()=>report.mockTags.length===base+1);await add('race-B.png');await done();await sleep(1500);assert.equal(await prompt(),'fast_B');assert.equal(await count(),3);assert.equal(await evaluate("document.querySelector('.wd14-source-caption span').textContent"),'race-B.png');
  check('MOCK reverse-order replacement A/B, aborted A cannot contaminate B',{history:await count(),late:report.lateResponses.slice(-2)});
  rules=[{text:'background_C',delay:700}];await add('race-C.png');await evaluate("document.querySelector('.wd14-history-strip button').click()");await sleep(1000);assert.equal(await prompt(),edited);assert.equal(await evaluate("document.querySelector('.wd14-source-caption span').textContent"),'real-first.png');assert.equal(await count(),4);
  check('MOCK history click during pending completion preserves selection + edited Prompt');
  rules=[{text:'failed_rerun',error:'模拟重推失败，原编辑保留'}];await click('.wd14-work-head button','重新反推');await wait(()=>evaluate("document.querySelector('.wd14-work-foot').textContent.includes('模拟重推失败')"));assert.equal(await prompt(),edited);assert.equal(await count(),4);
  check('MOCK rerun failure preserves edited text and history');
  rules=[{text:'clear_late',delay:700}];await add('clear-pending.png');await click('.wd14-history-head button','清空结果');await sleep(1000);assert.equal(await count(),0);assert.equal(await prompt(),'');assert.equal(await evaluate("!!document.querySelector('.wd14-drop img')"),false);
  rules=[{text:'unmount_late',delay:700}];await add('unmount-pending.png');await nav('文生图');await sleep(1000);await nav('图片反推');await wait(()=>evaluate("!!document.querySelector('.wd14-results')"));assert.equal(await count(),0);assert.equal(await prompt(),'');check('MOCK clear + unmount abort/ignore late responses, remount empty');
  rules=[{text:'stop_late',delay:700}];await add('stop-pending.png');await click('.wd14-work-head button','停止等待');await sleep(1000);assert.equal(await count(),1);assert.equal(await prompt(),'');assert.match(await evaluate("document.querySelector('.wd14-work-foot').textContent"),/服务端当前 ONNX 计算可能仍会继续/);check('MOCK stop means stop waiting, not backend cancellation; retryable thumbnail retained');
  await click('.wd14-history-head button','清空结果');
  // Explicit multi-file rejection is lossless, never silently chooses the first.
  await evaluate("(()=>{const d=new DataTransfer();for(const n of ['multi-a.png','multi-b.png'])d.items.add(new File(['x'],n,{type:'image/png'}));document.querySelector('.wd14-drop').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:d}));})()");assert.match(await evaluate("document.querySelector('.wd14-work-foot').textContent"),/每次仅添加一张/);assert.equal(await count(),0);
  await evaluate("(()=>{const d=new DataTransfer();d.items.add(new File([new Uint8Array(20*1024*1024+1)],'large.png',{type:'image/png'}));document.querySelector('.wd14-drop').dispatchEvent(new ClipboardEvent('paste',{bubbles:true,cancelable:true,clipboardData:d}));})()");assert.match(await evaluate("document.querySelector('.wd14-work-foot').textContent"),/20 MiB/);check('multi selection explicit rejection + 20MiB admission limit');
  rules=[{text:'valid_drop'},{text:'valid_paste'}];await add('valid-drop.png','#994840','drop');await done();assert.equal(await prompt(),'valid_drop');await add('valid-paste.png','#326d70','paste');await done();assert.equal(await prompt(),'valid_paste');assert.equal(await count(),2);check('MOCK valid browser drop + focused image paste replace and auto-tag, first result retained');await click('.wd14-history-head button','清空结果');
  for(let i=0;i<30;i++){await add(`history-${i}-${'long-file-name-'.repeat(5)}.png`);await done();assert.equal(await count(),i+1);}
  await evaluate("document.querySelector('.wd14-tag-details').open=true");
  for(const [w,h] of [[1440,900],[1920,1080],[1366,768]]){await view(w,h);await fit(`${w}x${h} 30 thumbnails + 500 tags + long filename, no document/work scroll`);await shot(`${w}-30-history-500-tags`);}
  const beforeName=await evaluate("document.querySelector('.wd14-source-caption span').textContent");const requests=report.mockTags.length;await add('31st.png');assert.equal(await count(),30);assert.equal(report.mockTags.length,requests);assert.equal(await evaluate("document.querySelector('.wd14-source-caption span').textContent"),beforeName);assert.match(await evaluate("document.querySelector('.wd14-work-foot').textContent"),/30 项缓存上限/);
  check('full cache rejection preserves existing image/results and makes strategy visible');
  await view(390,844);await evaluate("document.querySelector('.wd14-page').scrollTop=0");assert.equal(await evaluate('document.documentElement.scrollWidth'),390);await shot('390-image-first');
  await evaluate("document.querySelector('.wd14-results button').focus()");let focus=await evaluate("(()=>{const r=document.activeElement.getBoundingClientRect();return {top:r.top,bottom:r.bottom,viewport:innerHeight}})()");assert.ok(focus.top>=54 && focus.bottom<=844);check('390 Prompt copy focus reachable, no horizontal overflow',focus);await shot('390-prompt-actions');
  await evaluate("document.querySelector('.wd14-history-strip button:last-child').focus()");await shot('390-history');
  await evaluate("document.querySelector('.wd14-batch button[type=submit]').focus()");focus=await evaluate("(()=>{const r=document.activeElement.getBoundingClientRect();return {top:r.top,bottom:r.bottom,viewport:innerHeight}})()");assert.ok(focus.top>=54 && focus.bottom<=844);check('390 final batch button reachable',focus);
  await click('.wd14-history-head button','清空结果');await sleep(1100);assert.equal(await evaluate('__urls.made.filter(u=>!__urls.revoked.includes(u)).length'),0);check('all preview/thumbnail/export object URLs revoked after clear/unmount',{created:await evaluate('__urls.made.length'),revoked:await evaluate('__urls.revoked.length')});
  await view(1440,900);noModel=true;await click('.wd14-controls button',''); // refresh icon is the first button
  await wait(()=>evaluate("document.querySelector('.wd14-status').textContent.includes('未安装')"));const noRequests=report.mockTags.length;await add('uninstalled.png');await sleep(200);assert.equal(report.mockTags.length,noRequests);assert.match(await evaluate("document.querySelector('.wd14-work-foot').textContent"),/模型未就绪/);assert.ok(await evaluate("!!document.querySelector('.wd14-drop img')"));check('uninstalled model keeps new preview, no invalid auto tag');
  await click('.wd14-controls button','安装所选模型');await wait(()=>evaluate("document.querySelector('.wd14-progress')?.textContent.includes('50%')"));await click('.wd14-controls button','取消下载');await wait(()=>evaluate("document.querySelector('.wd14-work-foot').textContent.includes('模型正在发布')"));assert.equal(await evaluate("[...document.querySelectorAll('.wd14-controls button')].some(x=>x.textContent.includes('取消下载'))"),false);await shot('mock-publishing-409');await wait(()=>evaluate("document.querySelector('.wd14-controls').textContent.includes('历史备份请人工检查')"));
  noModel=false;await click('.wd14-controls button','');await wait(()=>evaluate("!!document.querySelector('.wd14-status .ready')"));rules=[{text:'installed_explicit'}];await click('.wd14-work-head button','开始反推');await done();check('MOCK install publishing409/warning, explicit run after model ready');
  assert.equal(await evaluate("document.querySelector('.wd14-batch select').value"),'ignore');await fill('.wd14-batch form > label:first-child input',out);await fill('.wd14-batch form > label:nth-child(2) input',out);await select('.wd14-batch select','prepend');await evaluate("document.querySelector('.wd14-batch input[type=checkbox]').click()");await click('.wd14-batch button','开始批量');await wait(()=>evaluate("!!document.querySelector('.wd14-batch .wd14-progress')"));assert.equal(batchBody.conflict,'prepend');assert.equal(batchBody.recursive,true);await click('.wd14-batch button','取消任务');await wait(()=>evaluate("document.querySelector('.wd14-batch').textContent.includes('cancelled')"));check('MOCK batch default ignore/explicit prepend/output/recursive/progress/cancel',batchBody);
  for(const [text,selector] of [['文生图','.workspace'],['图生图','.image-workspace'],['工具箱','.toolbox-page']]){await nav(text);await wait(()=>evaluate(`!!document.querySelector('${selector}')`));assert.ok(await evaluate(`document.querySelector('${selector}').getBoundingClientRect().height>0`));check('sibling navigation '+text);}
  recovery=true;await nav('图片反推');await wait(()=>evaluate("document.querySelector('.wd14-controls').textContent.includes('人工恢复')"));check('MOCK marker manual recovery warning visible');await shot('mock-marker-warning');
  assert.equal(report.errors.length,0,report.errors.join('\n'));check('no uncaught browser exceptions');
}catch(error){report.failure=error.stack;report.failureDOM=await debug?.("({focus:document.hasFocus(),foot:document.querySelector('.wd14-work-foot')?.textContent,prompt:document.querySelector('.wd14-editor textarea')?.value,history:document.querySelectorAll('.wd14-history-strip button').length})").catch(()=>null);await debugShot?.('failure').catch(()=>{});console.error(error,report.failureDOM);process.exitCode=1;}
finally{await writeFile(path.join(out,'probe.json'),JSON.stringify(report,null,2));ws?.close();browser.kill();await Promise.race([new Promise(r=>browser.exitCode!==null?r():browser.once('exit',r)),sleep(5000)]);console.log('Evidence',path.join(out,'probe.json'));}
