import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { createXYZDraft, decimalRangeValues, inspectXYZDrafts, syncXYZBaseline, xyzLocalSummary } from '../src/xyz-draft.js';
import { xyzWireDraft } from '../src/xyz-prompt-source.js';
import { xyzApi } from '../src/xyz-client.js';
import { pythonDependencyOptions } from './python-dependency.test.helper.mjs';

const ref = i => ({ value:`private/${i}.safetensors`, name:`LoRA ${i}`, identitySchema:1,
 instanceId:`00000000-0000-4000-8000-${String(i).padStart(12,'0')}`, contentHash:i.toString(16).padStart(64,'0'),
 contentRevision:1, weight:'0.8124', precision:4, enabled:true });
const range = {start:'0.2',end:'1',step:'0.3'};
const axis = (m, values = null) => ({field:'lora_weight',target:m.instanceId,values:values?.map((value,i)=>({value,label:`label ${i}`})) || [],
 ...(values ? {} : {range:structuredClone(range)}),editor_source:{schema_version:1,target_reference:structuredClone(m)}});
const draft = (loras=[]) => createXYZDraft({parameters:{engine:'Anima',diffusion_model:'Anima-diffusion.safetensors',text_encoder:'Anima-encoder.safetensors',vae:'Anima-vae.safetensors',
 prompt:'global preset\n\nraw + selected preset',negative_prompt:'bad',width:768,height:1024,steps:28,cfg:6.5,seed:'7',sampler:'euler',scheduler:'simple',guidance:'none',pag:{scale:.3,applied_layers:'mid'},postprocess_chain:[]},
 seed_mode:'fixed',loras,group_snapshots:[{id:'g',presetPrompt:'global preset',enabled:true}],group_prompt:'global preset',prompt_without_groups:'raw + selected preset',library_revision:0});
const a=ref(21), b=ref(22), c=ref(23);
const two = (loras=[]) => {const d=draft(loras);d.axes={z:null,y:axis(b),x:axis(a)};return d;};

test('custom three-weight lists and steps/CFG combinations count independently; bounds/empty/zero/integer/precision stay fail-closed',()=>{
 const d=two();d.axes.x=axis(a,['0','0.3','0.9']);d.axes.y=axis(b,['0.2','0.7']);d.axes.z=axis(c,['-0.1','1']);
 assert.equal(xyzLocalSummary(d).total,12);assert.equal(xyzLocalSummary(d).valid,true);assert.deepEqual(xyzWireDraft(d).axes.x.values,d.axes.x.values);
 d.axes.x={field:'steps',values:['15','23','28'].map(value=>({value}))};d.axes.y=axis(b,['0','0.3','0.9']);assert.equal(xyzLocalSummary(d).total,18);assert.equal(xyzLocalSummary(d).valid,true);
 d.axes.x={field:'steps',values:[],range:{start:'12',end:'20',step:'2'}};d.axes.z=axis(c);assert.equal(xyzLocalSummary(d).total,45);assert.equal(xyzLocalSummary(d).valid,true);
 d.axes.x={field:'cfg',values:[],range:{start:'4',end:'6.5',step:'0.5'}};assert.equal(xyzLocalSummary(d).total,54);assert.equal(xyzLocalSummary(d).valid,true);
 for(const[field,values]of [['steps',[null,'','0','-1','101','15.5','NaN','0.00000000000000001']],['lora_weight',[null,'','-5.0001','5.0001','0.12345','NaN']]])for(const value of values){
  const bad=structuredClone(d);bad.axes.x=field==='steps'?{field,values:[{value}]}:axis(a,[value]);const before=structuredClone(bad);assert.equal(xyzLocalSummary(bad).valid,false,`${field}:${value}`);assert.deepEqual(bad,before);assert.equal(xyzWireDraft(bad).axes.x.values[0].value,value);
 }
 for(const[start,end,step]of [['0','0','1'],['101','101','1'],['12.5','12.5','1'],['12','20','0'],['12','20','-1'],['','20','2'],['12','','2'],['12','20',''],['12','20','0.00000000000000001'],['12','13','.5']]){
  const bad=structuredClone(d);bad.axes.x={field:'steps',values:[],range:{start,end,step}};const before=structuredClone(bad);assert.equal(xyzLocalSummary(bad).valid,false,JSON.stringify(bad.axes.x.range));assert.deepEqual(bad,before);
 }
});

test('unmounted multi-axis projection is pure, deterministic, idempotent, exact and shares inspect/Sync/request-key boundaries',()=>{
 const d=two(), before=structuredClone(d), wire=xyzWireDraft(d);
 assert.deepEqual(wire.baseline.loras,[a,b].map(m=>({...m,weight:'0',enabled:false})));
 for(const key of ['x','y']) {const {editor_source,...expected}=d.axes[key];assert.deepEqual(wire.axes[key],expected);assert.deepEqual(decimalRangeValues(wire.axes[key].range),['0.2','0.5','0.8']);}
 assert.deepEqual(xyzWireDraft(wire),wire);assert.deepEqual(xyzWireDraft(d),wire);assert.deepEqual(d,before);
 assert.deepEqual(xyzLocalSummary(d),{dimensions:{x:3,y:3,z:1},total:9,valid:true,issues:[],error:'',authoritative:false,seedsResolved:false});
 assert.equal(inspectXYZDrafts({schema_version:1,by_engine:{Anima:d}},'xyz_plot').fatal,false);
 const synced=syncXYZBaseline(d,draft([a]).baseline,['loras']);assert.deepEqual(synced.axes,d.axes);assert.deepEqual(xyzWireDraft(synced).baseline.loras,[a,{...b,weight:'0',enabled:false}]);
 const ui=structuredClone(d);ui.baseline.postprocess_source={schema_version:1,chain:[]};assert.deepEqual(xyzWireDraft(ui),wire);
 d.axes.z=axis(c,['0','-0.2','0.8']);const triple=xyzWireDraft(d);assert.deepEqual(triple.baseline.loras.map(m=>m.instanceId),[a,b,c].map(m=>m.instanceId));assert.deepEqual(triple.axes.z.values,d.axes.z.values);assert.equal(xyzLocalSummary(d).total,27);assert.equal(xyzLocalSummary(d).valid,true);
});
test('mounted binding is not overwritten; same target deduplicates but remains two weight writers; contradictory strong sources fail closed',()=>{
 const mounted={...a,weight:'-0.5',enabled:false,precision:2,name:'global metadata',category:'global',groupId:'g'}, d=two([mounted]);
 assert.deepEqual(xyzWireDraft(d).baseline.loras[0],mounted);
 d.axes.y=axis(a);assert.equal(xyzWireDraft(d).baseline.loras.length,1);assert.equal(xyzWireDraft(d).axes.y.field,'lora_weight');
 d.baseline.loras=[];assert.equal(xyzWireDraft(d).baseline.loras.length,1);
 for(const change of [{weight:'1'},{precision:2},{name:'conflicting'},{contentRevision:2},{value:'another/path'},{contentHash:'f'.repeat(64)},
  {originalReference:{value:'source-one'}},{originalReference:{value:'source-two'}}]) {
   const bad=structuredClone(d);bad.axes.x.editor_source.target_reference.originalReference={value:'source'};Object.assign(bad.axes.y.editor_source.target_reference,change);assert.throws(()=>xyzWireDraft(bad),/来源不一致/);
 }
 for(const change of [{instanceId:b.instanceId},{identitySchema:2},{contentHash:'bad'},{contentRevision:0},{contentRevision:1.5},{instanceId:'unresolved'},{value:''},{missing:true}]) {
   const bad=two();Object.assign(bad.axes.x.editor_source.target_reference,change);const before=structuredClone(bad);assert.throws(()=>xyzWireDraft(bad));assert.equal(xyzLocalSummary(bad).valid,false);assert.deepEqual(bad,before);
 }
 for(const change of [{contentRevision:2},{value:'other/path'},{contentHash:'f'.repeat(64)}]) {const bad=two([{...a,...change}]);assert.throws(()=>xyzWireDraft(bad),/冻结背景/);}
 assert.throws(()=>xyzWireDraft(two([{...b,value:a.value}])),/同路径/);
});
test('submission-only mount cap blocks both transports without a request; exactly 16 and existing targets stay usable',async()=>{
 const globals=Array.from({length:16},(_,i)=>ref(i+1)), d=two(globals), before=structuredClone(d);
 assert.equal(xyzLocalSummary(d).valid,false);assert.match(xyzLocalSummary(d).error,/16/);assert.deepEqual(d,before);
 const calls=[],api=xyzApi(async(...args)=>{calls.push(args);assert.fail('over-limit must not fetch');});
 assert.throws(()=>api.plan(d),/16/);assert.throws(()=>api.submit(d,{submissionId:'private'}),/16/);assert.deepEqual(calls,[]);
 d.baseline.loras=globals.slice(0,14);assert.equal(xyzWireDraft(d).baseline.loras.length,16);assert.equal(xyzLocalSummary(d).valid,true);
 d.baseline.loras=globals;d.axes.x=axis(globals[0]);d.axes.y=axis(globals[1]);assert.equal(xyzWireDraft(d).baseline.loras.length,16);
});
test('strict current Python compiler: 9/27 cells, global order/preset/disabled/zero, legacy mounted, dedup/conflict and P10 negatives',pythonDependencyOptions(process.env.P10_PYTHON || 'python',path.resolve(import.meta.dirname,'..'),['PIL','pydantic']),async()=>{
 const root=process.env.XYZ_EVIDENCE_ROOT || await mkdtemp(path.join(os.tmpdir(),'opencode','xyz-weight-multiaxis-20261012-'));
 const folder=path.join(root,`unit-compiler-${Date.now()}-${process.pid}`);await mkdir(folder,{recursive:true});
 const globals=[{...ref(4),weight:'-0.25'},{...ref(2),weight:'0',enabled:true},{...ref(5),weight:'1.2345',enabled:false},{...ref(1),weight:'0.2'},{...ref(3),weight:'-0.8124',groupId:'g'}];
 const registry={schemaVersion:1,instances:Object.fromEntries([...globals,a,b,c].map(m=>[m.instanceId,{...m,engine:'Anima',status:'present'}]))};
 const cases=[];
 const add=(name,d,expected={},r=registry)=>{const before=structuredClone(d),wire=xyzWireDraft(d);assert.deepEqual(d,before);cases.push({name,wire,uiBaseline:structuredClone(d.baseline),registry:structuredClone(r),expected});};
 add('user-nine',two(),{cells:9});const triple=two();triple.axes.z=axis(c);add('triple-27',triple,{cells:27});
 const global=two(globals);add('five-global-nine',global,{cells:9});global.axes.z=axis(c);add('five-global-triple',global,{cells:27});
 const mounted=two([a,b,...globals]);delete mounted.axes.x.editor_source;delete mounted.axes.y.editor_source;add('legacy-mounted-nine',mounted,{cells:9});
 const zero=two(globals);zero.axes.x=axis({...a,enabled:false,weight:'0'},['0','0.5']);zero.axes.y=axis(b,['0','0.8']);add('zero-disables',zero,{cells:4,weights:[0.5,0.8]});
 const custom=two();custom.axes.x=axis(a,['0','0.3','0.9']);custom.axes.y=axis(b,['0.2','0.7']);custom.axes.z=axis(c,['-0.1','1']);add('custom-three-weight-lists',custom,{cells:12,axisValues:{x:[0,.3,.9],y:[.2,.7],z:[-.1,1]}});
 custom.baseline.loras=globals;add('five-global-custom-three-weight-lists',custom,{cells:12,axisValues:{x:[0,.3,.9],y:[.2,.7],z:[-.1,1]}});
 custom.axes.x={field:'steps',values:['15','23','28'].map(value=>({value}))};custom.axes.y=axis(b,['0','.3','.9']);add('custom-steps-two-weight-lists',custom,{cells:18,axisValues:{x:[15,23,28],y:[0,.3,.9],z:[-.1,1]}});
 custom.axes.x={field:'steps',values:[],range:{start:'12',end:'20',step:'2'}};custom.axes.z=axis(c);add('steps-range-weight-list-and-range',custom,{cells:45,axisValues:{x:[12,14,16,18,20],y:[0,.3,.9],z:[.2,.5,.8]}});
 custom.axes.x={field:'cfg',values:[],range:{start:'4',end:'6.5',step:'.5'}};add('cfg-float-range-two-weight-axes',custom,{cells:54,axisValues:{x:[4,4.5,5,5.5,6,6.5],y:[0,.3,.9],z:[.2,.5,.8]}});
 custom.axes.x={field:'steps',values:[{value:'15'}]};custom.axes.y={field:'steps',values:[{value:'23'}]};add('two-steps-writers',custom,{error:'axis-conflict'});
 custom.axes.y=axis(b,['.2','.7']);custom.axes.z=axis(c,['-.1','1']);
 for(const[field,value,error]of [['steps','0','bounds'],['steps','101','bounds'],['steps','15.5','bounds'],['steps','','empty-value'],['steps','0.00000000000000001','number'],['lora_weight','-5.0001','bounds'],['lora_weight','5.0001','bounds'],['lora_weight','.12345','precision'],['lora_weight','','empty-value']]){
  const bad=structuredClone(custom);bad.axes.x=field==='steps'?{field,values:[{value}]}:axis(a,[value]);add(`numeric-${field}-${value||'empty'}`,bad,{error});
 }
 const same=two();same.axes.y=axis(a);add('same-instance',same,{error:'axis-conflict'});
 const mixed=two();mixed.axes.y={field:'lora_profile',values:[{value:{mode:'keep_background',members:[]}}]};add('profile-plus-weight',mixed,{error:'axis-conflict'});
 const copy=two();copy.axes.y=axis({...b,contentHash:a.contentHash});const copyRegistry=structuredClone(registry);copyRegistry.instances[b.instanceId].contentHash=a.contentHash;add('equal-content-copies',copy,{error:'duplicate-content'},copyRegistry);
 for(const [name,mutate]of [
  ['unresolved-registry',r=>delete r.instances[a.instanceId]],['wrong-engine',r=>r.instances[a.instanceId].engine='SD'],
  ['changed-content',r=>r.instances[a.instanceId].contentHash='f'.repeat(64)],['changed-revision',r=>r.instances[a.instanceId].contentRevision++],
  ['changed-path',r=>r.instances[a.instanceId].value='moved.safetensors'],['missing-instance',r=>r.instances[a.instanceId].status='missing']
 ]){const r=structuredClone(registry);mutate(r);add(name,two(),{error:'lora-identity'},r);}
 await writeFile(path.join(folder,'weight-multiaxis-wire.json'),JSON.stringify({cases},null,2));
 const result=spawnSync(process.env.P10_PYTHON||'python',['-B','scripts/xyz-seven.compile.py',folder,'--weight-multiaxis'],{cwd:path.resolve(import.meta.dirname,'..'),encoding:'utf8',timeout:60000});
 await writeFile(path.join(folder,'compiler-command.json'),JSON.stringify({args:['-B','scripts/xyz-seven.compile.py',folder,'--weight-multiaxis'],exit:result.status,stdout:result.stdout,stderr:result.stderr},null,2));
 assert.equal(result.status,0,result.stderr||result.stdout);console.log(`weight multiaxis strict evidence: ${folder}`);
});
