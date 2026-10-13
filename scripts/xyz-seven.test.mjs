import test from 'node:test';
import assert from 'node:assert/strict';
import {createXYZDraft,decimalRangeValues,inspectXYZDrafts,syncXYZBaseline,xyzLocalSummary} from '../src/xyz-draft.js';
import {xyzExpectedValues,xyzSampleSeeds} from '../src/xyz-editor-source.js';
import {xyzWireDraft} from '../src/xyz-prompt-source.js';
const member=i=>({value:`${i}.safetensors`,weight:'0.8124',precision:4,enabled:true,identitySchema:1,instanceId:`00000000-0000-4000-8000-${String(i).padStart(12,'0')}`,contentHash:String(i).repeat(64),contentRevision:1});
const fixture=()=>createXYZDraft({parameters:{engine:'SD',prompt:'global\n\nraw',negative_prompt:'bad',seed:'18446744073709551615',width:1024,height:1536,steps:28,cfg:6.5,pag:{scale:.3},postprocess_chain:[]},seed_mode:'fixed',loras:[1,2,3,4,5].map(member),group_snapshots:[],group_prompt:'global',prompt_without_groups:'raw',library_revision:1});
test('quantity creates exact pending nulls, preserves first values and refuses every invalid count without mutation',()=>{
 const before=[{value:'15',label:'a'},{value:'23',label:'b'},{value:'28',label:'c'}];assert.deepEqual(xyzExpectedValues(before,'5'),[...before,{value:null,label:''},{value:null,label:''}]);assert.deepEqual(xyzExpectedValues(before,'2'),before.slice(0,2));
 for(const s of ['0','-1','1.5','NaN','257','Infinity',''])assert.throws(()=>xyzExpectedValues(before,s));assert.equal(before.length,3);
 for(const field of ['steps','cfg','seed','prompt','sampler','components','lora_model','lora_profile','lora_weight','lora_multiplier']){const d=fixture();d.axes.x={field,values:xyzExpectedValues([],'5')};assert.match(xyzLocalSummary(d).error,/待定义/);}
});
test('exact range supports floating positive/negative steps, leading dot; invalid raw is never rewritten',()=>{
 assert.deepEqual(decimalRangeValues({start:'12',end:'20',step:'2'}),['12','14','16','18','20']);assert.deepEqual(decimalRangeValues({start:'4.0',end:'6.5',step:'.5'}),['4.0','4.5','5.0','5.5','6.0','6.5']);assert.deepEqual(decimalRangeValues({start:'.3',end:'.1',step:'-.1'}),['0.3','0.2','0.1']);
 for(const range of [{start:'0',end:'1',step:'0'},{start:'1',end:'0',step:'.1'},{start:'0',end:'257',step:'1'},{start:'NaN',end:'1',step:'1'}]){const before=structuredClone(range);assert.throws(()=>decimalRangeValues(range));assert.deepEqual(range,before);}
 const d=fixture();d.axes.x={field:'steps',values:[],range:{start:'12.5',end:'20.5',step:'2'}};assert.match(xyzLocalSummary(d).error,/整数/);
});
test('four-mode source projects exact sample uint64 lists with modulo wrap, shared cross-combinations and seed-axis priority',()=>{
 const d=fixture();d.repeat_count=3;for(const[mode,expected]of [['increment',['18446744073709551615','0','1']],['decrement',['18446744073709551615','18446744073709551614','18446744073709551613']],['fixed',Array(3).fill('18446744073709551615')],['random',null]]){
  d.baseline.seed_source={schema_version:1,mode};d.baseline.seed_mode=mode==='random'?'random':'fixed';assert.deepEqual(xyzSampleSeeds(d),expected);const wire=xyzWireDraft(d);assert.deepEqual(wire.sample_seeds,expected);assert.ok(!('seed_source' in wire.baseline));
 }
 d.baseline.seed_source.mode='decrement';d.baseline.parameters.seed='0';assert.deepEqual(xyzSampleSeeds(d),['0','18446744073709551615','18446744073709551614']);d.axes.x={field:'seed',values:[{value:'7'}]};assert.equal(xyzWireDraft(d).sample_seeds,null);
 delete d.baseline.seed_source;d.sample_seeds=['a'];assert.deepEqual(xyzWireDraft(d).sample_seeds,['a'],'legacy explicit raw preserved, not silently replaced');
});
test('legacy additions/multiplier keep profiles; external weight keeps instance axis and submission-only baseline slot',()=>{
 const d=fixture(),before=structuredClone(d.baseline),extra=member(6);
 d.axes.x={field:'lora_model',values:[{value:extra,label:'new'}],editor_source:{schema_version:1,operation:'add'}};let wire=xyzWireDraft(d);assert.equal(wire.axes.x.field,'lora_profile');assert.deepEqual(wire.axes.x.values[0].value.members,[extra]);assert.deepEqual(d.baseline,before);
  d.axes.x={field:'lora_weight',target:extra.instanceId,values:[],range:{start:'0.5',end:'1.0',step:'0.25'},editor_source:{schema_version:1,target_reference:extra}};wire=xyzWireDraft(d);assert.equal(wire.axes.x.field,'lora_weight');assert.deepEqual(wire.axes.x.range,d.axes.x.range);assert.deepEqual(decimalRangeValues(wire.axes.x.range),['0.50','0.75','1.00']);assert.deepEqual(wire.baseline.loras,[...before.loras,{...extra,weight:'0',enabled:false}]);assert.equal(xyzLocalSummary(d).valid,true);assert.deepEqual(d.baseline,before);
 d.axes.x={field:'lora_multiplier',values:[{value:'0.5'}],snapshot:{mode:'keep_background',members:[extra],test_targets:[],includePresetText:true,preset_prompt:'candidate'},editor_source:{schema_version:1,member_only:true}};wire=xyzWireDraft(d);assert.equal(wire.axes.x.values[0].value.members[0].weight,'0.4062');assert.equal(wire.axes.x.values[0].value.preset_prompt,'global\n\ncandidate');assert.deepEqual(d.baseline,before);
 d.axes.x.editor_source=undefined;assert.equal(xyzWireDraft(d).axes.x.field,'lora_multiplier','legacy multiplier and semantics retained');
 d.axes.x={field:'lora_profile',values:[{value:{mode:'keep_background',members:[extra],includePresetText:true,preset_prompt:'global'}}],editor_source:{schema_version:1,operation:'add'}};assert.equal(xyzWireDraft(d).axes.x.values[0].value.preset_prompt,'global','identical whole frozen group prefix only once');
});
test('new UI sources restore only known versions, Sync sampling replaces source, unrelated Sync preserves it',()=>{
 const d=fixture();d.baseline.seed_source={schema_version:1,mode:'increment'};d.axes.x={field:'lora_model',values:[{value:member(6)}],editor_source:{schema_version:1,operation:'add'}};
 assert.equal(inspectXYZDrafts({schema_version:1,by_engine:{SD:d}},'xyz_plot').fatal,false);assert.deepEqual(syncXYZBaseline(d,fixture().baseline,['canvas']).baseline.seed_source,d.baseline.seed_source);assert.equal(syncXYZBaseline(d,fixture().baseline,['sampling']).baseline.seed_source,undefined);
 for(const which of ['seed','axis']){const corrupt=structuredClone(d);(which==='seed'?corrupt.baseline.seed_source:corrupt.axes.x.editor_source).schema_version=2;const r=inspectXYZDrafts({schema_version:1,by_engine:{SD:corrupt}},'xyz_plot');assert.equal(r.fatal,true);assert.deepEqual(r.raw.by_engine.SD,corrupt);}
});
