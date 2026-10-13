import test from 'node:test';
import assert from 'node:assert/strict';
import {createXYZDraft,xyzLocalSummary} from '../src/xyz-draft.js';
import {xyzAxisValueIssue} from '../src/xyz-local-validation.js';
import {isAllowedToolbarBackgroundCall} from './xyz-seven-rework.browser-checks.mjs';
const member={value:'f.safetensors',weight:'0.8124',precision:4,enabled:true,identitySchema:1,instanceId:'00000000-0000-4000-8000-000000000001',contentHash:'a'.repeat(64),contentRevision:1};
const draft=(engine='SD')=>createXYZDraft({parameters:{engine,prompt:'positive',negative_prompt:'bad',seed:'7',steps:28,cfg:6.5,pag:{scale:.3,applied_layers:'mid'},width:1024,height:1536,postprocess_chain:[]},seed_mode:'fixed',loras:[member],group_snapshots:[],group_prompt:'',prompt_without_groups:'positive'});
for(const[field,bad,good]of [
 ['steps',['0','101','1.5','NaN'],['1','100','28']],
 ['cfg',['-0.1','31','30.0000000000000001','Infinity'],['0','30','4.5']],
 ['pag.scale',['-0.1','5.1'],['0','5','0.3']],
 ['clip_skip',['0','13','2.5'],['1','12']],
 ['lora_weight',['6','-5.0001','0.12345'],['-5','0','5','0.12340']],
 ['lora_multiplier',['6','-5.1','1e2'],['-5','0','.5','5']],
 ['seed',['18446744073709551616','-1','1.5',7],['0','18446744073709551615']],
])test(`local ${field} bounds: keep count separate, preserve raw, reject each value and accept neighbours`,()=>{
 for(const value of bad){const d=draft();d.axes.x={field,values:[{value}]};if(field==='lora_weight')d.axes.x.target=member.instanceId;if(field==='lora_multiplier')d.axes.x.snapshot={mode:'full',members:[{...member,weight:'1'}]};const before=structuredClone(d),s=xyzLocalSummary(d);assert.equal(s.total,1);assert.equal(s.valid,false,JSON.stringify({field,value,s}));assert.ok(s.error);assert.deepEqual(d,before);}
 for(const value of good)assert.equal(xyzAxisValueIssue(field,value,'SD'),'');
});
test('integer engine-specific CLIP guards and every staircase value are checked, not just range shape',()=>{
 assert.ok(xyzAxisValueIssue('clip_skip','1','iL'));assert.equal(xyzAxisValueIssue('clip_skip','2','iL'),'');assert.ok(xyzAxisValueIssue('clip_skip','2','Anima'));
 for(const[field,start,end,step]of [['steps','99','101','1'],['cfg','29','31','1'],['pag.scale','4.5','5.1','.3'],['lora_weight','4.9999','5.0001','.0001'],['steps','12.5','20.5','2']]){const d=draft();d.axes.x={field,values:[],range:{start,end,step}};const s=xyzLocalSummary(d);assert.ok(s.total>0);assert.equal(s.valid,false);assert.ok(s.error);}
});
test('legacy/new member-only multiplier check chosen weight and exact result bounds/precision without clamping',()=>{
 for(const memberOnly of [false,true])for(const[weight,factor,valid]of [['3','2',false],['0.8137','.5',false],['0.8124','.5',true],['-5','1',true],['6','0',false],[null,'1',false]]){const d=draft();d.axes.x={field:'lora_multiplier',values:[{value:factor}],snapshot:{mode:memberOnly?'keep_background':'full',members:[{...member,weight}]},...(memberOnly?{editor_source:{schema_version:1,member_only:true}}:{})};const before=structuredClone(d);assert.equal(xyzLocalSummary(d).valid,valid,JSON.stringify({memberOnly,weight,factor}));assert.deepEqual(d,before);}
});
test('source projected external weight, null pending, invalid explicit samples and raw snapshot stay blocked',()=>{
 const d=draft();d.axes.x={field:'lora_weight',target:'external',values:[{value:'6'}],editor_source:{schema_version:1,target_reference:{...member,instanceId:'external'}}};assert.equal(xyzLocalSummary(d).valid,false);
 d.axes.x={field:'steps',values:[{value:null},{value:'15'},{value:'28'}]};assert.equal(xyzLocalSummary(d).total,3);assert.equal(xyzLocalSummary(d).valid,false);
 d.axes.x={field:'lora_multiplier',values:[{value:'1'}],snapshot:'{invalid'};assert.equal(xyzLocalSummary(d).valid,false);assert.equal(d.axes.x.snapshot,'{invalid');
 d.axes.x=null;d.sample_seeds=['18446744073709551616'];assert.equal(xyzLocalSummary(d).valid,false);
});
test('explicit null snapshot/profile arrays are invalid raw data, never default empty or erased',()=>{
  for(const mode of ['full','keep_background','no_test','no_loras'])for(const key of ['members','test_targets']){const d=draft();d.axes.x={field:'lora_profile',values:[{value:{mode,members:[],test_targets:[],[key]:null}}]};const before=structuredClone(d);assert.equal(xyzLocalSummary(d).valid,false,`${mode}/${key}`);assert.deepEqual(d,before);}
  for(const memberOnly of [false,true]){const d=draft();d.axes.x={field:'lora_multiplier',values:[{value:'1'}],snapshot:{mode:'full',members:null},...(memberOnly?{editor_source:{schema_version:1,member_only:true}}:{})};assert.equal(xyzLocalSummary(d).valid,false);assert.equal(d.axes.x.snapshot.members,null);}
});
test('toolbar API filter permits only exact background-poll GETs and exposes raw writes/unknown routes',()=>{
  for(const url of ['/api/inference/health','/api/inference/jobs','/api/lora-relink/state','/api/console?after=0','/api/console?after=173456789'])assert.equal(isAllowedToolbarBackgroundCall({url,method:'GET'}),true,url);
  assert.equal(isAllowedToolbarBackgroundCall({url:'/api/lora-relink/state'}),true,'omitted fetch method defaults to GET');
  const rawCalls=[
    {url:'/api/lora-relink/state',method:'GET',body:null},
    {url:'/api/ui-state',method:'PUT',body:{workspace:{same:true}}},
    {url:'/api/ui-state',method:'GET',body:null},
    {url:'/api/inference/jobs/active',method:'GET',body:null},
    {url:'/api/inference/health?restart=1',method:'GET',body:null},
    {url:'/api/console?after=0&extra=1',method:'GET',body:null},
    {url:'/api/inference/health',method:'POST',body:{write:true}},
    {url:'/api/lora-relink/state',method:'PATCH',body:{write:true}},
    {url:'/api/inference/jobs',method:'DELETE',body:null},
    {url:'/api/unknown-background',method:'GET',body:null},
    {url:'/api/inference/health',method:null,body:null},
  ];
  const allowed=rawCalls.filter(isAllowedToolbarBackgroundCall),unexpected=rawCalls.filter(call=>!isAllowedToolbarBackgroundCall(call));
  assert.deepEqual(allowed,[rawCalls[0]]);
  assert.deepEqual(unexpected,rawCalls.slice(1),'a ui-state PUT and every unrecognized call remain visible to the capture assertion');
  const injected={url:'/api/ui-state',method:'PUT',body:{workspace:{same:true}}};
  assert.equal(isAllowedToolbarBackgroundCall(injected),false);
  assert.deepEqual([injected].filter(call=>!isAllowedToolbarBackgroundCall(call)),[injected],'the review injection fails the browser raw-call assertion instead of disappearing in its filter');
});
