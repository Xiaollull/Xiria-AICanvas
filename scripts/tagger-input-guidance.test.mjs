import test from 'node:test';
import assert from 'node:assert/strict';
import { initialInputGuidance, updateInputGuidance, inputAvailability } from '../src/tagger-input-guidance.js';
import { Wd14Desk } from '../src/wd14-desk.js';
const selection={engine:'cl',model:'cl-tagger-v1.02',device:'cpu'};
const catalog=row=>({models:[{id:selection.model,ready:true,...row}]});
const pending=catalog({installed:false,verification_pending:true});
const ready=catalog({installed:true,verification_pending:false});
const entry=()=>({id:'one',status:'idle',error:'',result:null,...initialInputGuidance(selection,pending,false)});
test('retained pending image is waiting, then manual-start ready guidance without install command',()=>{
  const input=entry();assert.match(input.notice,/等待完整性校验/);assert.doesNotMatch(input.notice,/请先安装/);
  const patch=updateInputGuidance(input,selection,ready,false);assert.match(patch.notice,/校验完成，可点击开始反推/);assert.doesNotMatch(patch.notice,/请先安装/);
  assert.equal(updateInputGuidance({...input,...patch},selection,ready,false),null);
});
test('engine/device/history context never reassigns pending guidance to a different input',()=>{
  const input=entry();assert.equal(updateInputGuidance(input,{...selection,engine:'camie',model:'camie-tagger-v2'},ready,false),null);
  assert.equal(updateInputGuidance(input,{...selection,device:'gpu'},ready,false),null);
  assert.equal(updateInputGuidance({id:'another',status:'idle'},selection,ready,false),null);
  assert.equal(updateInputGuidance({...input,status:'done',result:{tags:[]},notice:'copied'},selection,ready,false),null);
});
test('tag errors, copied and other notices are not owned by the recovery effect',()=>{
  for(const patch of [{error:'real tag failure'},{status:'error'},{notice:'已复制当前 Prompt'},{notice:'other action'},{status:'pending'},{status:'stopped'}])assert.equal(updateInputGuidance({...entry(),...patch},selection,ready,false),null);
});
test('missing invalid readiness and unavailable GPU are truthful, never false ready',()=>{
  assert.equal(inputAvailability(selection,catalog({installed:false}),false).kind,'missing');
  assert.match(inputAvailability(selection,catalog({installed:false}),false).text,/请先安装/);
  const invalid=inputAvailability(selection,catalog({installed:false,installation_reason:'模型完整 manifest 摘要不匹配'}),false);assert.equal(invalid.kind,'invalid');assert.doesNotMatch(invalid.text,/请先安装|校验完成/);
  assert.equal(inputAvailability(selection,catalog({installed:true,ready:false,readiness_reason:'未适配'}),false).kind,'blocked');
  assert.equal(inputAvailability({...selection,device:'gpu'},catalog({installed:true,devices:{gpu:{available:false,reason:'显存不足'}}}),false).kind,'blocked');
});
test('same entry stays manual after readiness, existing replacement/rerun epochs remain desk-owned',async()=>{
  const desk=new Wd14Desk();const file={name:'user.png',type:'image/png',size:1};const id=desk.add([file]);desk.update(id,initialInputGuidance(selection,pending,false));
  const old=desk.state.entries[0];desk.update(id,updateInputGuidance(old,selection,ready,false));assert.equal(desk.state.entries[0].status,'idle');assert.equal(desk.state.entries[0].result,null);
  let calls=0;await desk.run(id,{...selection,general:.9},async(f,p)=>{calls++;assert.equal(f,file);assert.equal(p.model,selection.model);return {tags:[{name:'fox ears',confidence:.9}]};});
  assert.equal(calls,1);assert.equal(desk.state.entries.length,1);assert.equal(desk.state.entries[0].id,id);
});
