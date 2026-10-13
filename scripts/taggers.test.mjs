import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TAGGER_ENGINES, TAGGER_CATEGORY_LABELS, taggerDefaults, taggerParameters, resultSource, popoverPosition } from '../src/tagger-settings.js';
import { Wd14Desk } from '../src/wd14-desk.js';
import { wd14Admission } from './wd14-access.mjs';
test('four engine defaults and complete categories preserve legacy WD and native PixAI semantics', () => {
  assert.deepEqual(TAGGER_ENGINES.map(row => row.id), ['wd','cl','camie','pixai']);
  assert.deepEqual(taggerDefaults('wd'), {general:.35,character:.85});
  const pix=taggerParameters('pixai','pixai-tagger-v1.0','gpu',taggerDefaults('pixai'),50,false);
  assert.equal(pix.style,.15);assert.equal(pix.copyright,.24);assert.equal(pix.engine,'pixai');assert.equal(pix.device,'gpu');
  assert.equal(taggerDefaults('cl').tag_model,.9);assert.equal(TAGGER_CATEGORY_LABELS.artist,'作者');
});
test('history across engines never reruns on select; explicit rerun keeps one id and actual source', async () => {
  const desk=new Wd14Desk();const id=desk.add([{name:'image.png',type:'image/png',size:1}]);let calls=0;
  const transport=async(f,p)=>{calls++;return {tags:[{name:p.engine,confidence:.9,category:'style'}],params:p,runtime:{name:p.engine==='pixai'?'PyTorch':'ONNX Runtime',provider:p.device==='gpu'?'CUDA':'CPUExecutionProvider'}};};
  await desk.run(id,taggerParameters('wd','wd-swinv2-v3','cpu',taggerDefaults('wd'),50,false),transport);
  desk.edit(id,'edited WD');desk.select(id);assert.equal(calls,1);assert.equal(desk.state.entries[0].text,'edited WD');
  await desk.run(id,taggerParameters('pixai','pixai-tagger-v1.0','gpu',taggerDefaults('pixai'),50,false),transport);
  assert.equal(desk.state.entries.length,1);assert.equal(desk.state.entries[0].id,id);assert.match(resultSource(desk.state.entries[0]),/PixAI|pixai/);assert.match(resultSource(desk.state.entries[0]),/PyTorch \/ CUDA/);
});
test('popover placement flips above and clamps both screen edges including 390px', () => {
  assert.deepEqual(popoverPosition({left:360,top:600,bottom:644},360,300,390,844),{left:18,top:292});
  assert.deepEqual(popoverPosition({left:-20,top:12,bottom:56},360,300,390,844),{left:12,top:64});
  const component=readFileSync(new URL('../src/TaggerInfoPopover.jsx',import.meta.url),'utf8');
  for(const pattern of [/createPortal/, /aria-haspopup="dialog"/, /aria-expanded=\{open\}/, /event.key === "Escape"/, /pointerdown/, /focusin/, /ResizeObserver/, /noopener noreferrer/, /closeButton.current\?\.focus/])assert.match(component,pattern);
});
test('runtime install requires same-origin and loopback even with forged proof', () => {
  const request={url:'/api/inference/wd14/runtime/install',method:'POST',headers:{origin:'http://127.0.0.1:7709',host:'127.0.0.1:7709'},socket:{remoteAddress:'127.0.0.1'}};
  assert.equal(wd14Admission(request).allowed,true);
  assert.equal(wd14Admission({...request,socket:{remoteAddress:'192.168.1.1'}}).allowed,false);
  assert.equal(wd14Admission({...request,headers:{...request.headers,origin:'https://evil.example'}}).allowed,false);
});
