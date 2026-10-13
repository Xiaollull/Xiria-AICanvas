import test from "node:test";
import assert from "node:assert/strict";
import { createXYZDraft, inspectXYZDrafts, syncXYZBaseline } from "../src/xyz-draft.js";
import { captureXYZPromptSource, patchXYZPrompt, toggleXYZPreset, refreshXYZPromptLibrary, xyzWireDraft, xyzPromptSource } from "../src/xyz-prompt-source.js";
import { copyXYZManagerGroup, updateXYZManagerGroups, updateXYZManagerMounts, mountXYZCatalogItem, replaceXYZCatalogSelection, xyzAxisLoraMarks } from "../src/xyz-manager-context.js";
import { composeGenerationPrompt } from "../src/prompt-presets.js";
import { xyzApi } from "../src/xyz-client.js";
import { readFile } from "node:fs/promises";

const member = (i) => ({ value: `folder/${i}.safetensors`, name: `LoRA ${i}`, category: "人物", weight: 0.8137, enabled: true, precision: 4,
  identitySchema: 1, instanceId: `00000000-0000-4000-8000-${String(i).padStart(12,"0")}`, contentHash: String(i % 10).repeat(64), contentRevision: 3 });
const group = { id:"g-one",name:"Group",enabled:false,collecting:false,presetPrompt:"group words",members:[member(2),member(3)] };
const records=[{id:"preset-a",type:"positive",groupId:"p-group",position:"start",order:0,content:"preset words",name:"A"},{id:"preset-n",type:"negative",groupId:"n-group",position:"end",order:0,content:"negative words",name:"N"}];
const baseline = (engine="SD") => ({ parameters:{engine,checkpoint:"model.safetensors",diffusion_model:"diff.safetensors",text_encoder:"te.safetensors",vae:"vae.safetensors",
  prompt:"raw words",negative_prompt:engine==="Flux"?"":"raw negative",steps:20,cfg:7,width:512,height:512,seed:"18446744073709551615",sampler:"euler",scheduler:"normal",guidance:"none",pag:{scale:0.3,applied_layers:"mid"},postprocess_chain:[]},
  loras:[member(1)],group_snapshots:[],prompt_without_groups:"raw words",group_prompt:"",seed_mode:"fixed",library_revision:0 });
const fresh = (engine="SD") => createXYZDraft(captureXYZPromptSource(baseline(engine),{positive:"raw words",negative:"raw negative",records,active:{positive:["preset-a"],negative:["preset-n"]}}));

test("axis toolbar exposes native unavailable controls, not global escape or engine no-op",async()=>{
  const source=async name=>readFile(new URL(`../src/${name}`,import.meta.url),"utf8");
  const [manager,modal,bar,picker]=await Promise.all([source("XYZLoraManager.jsx"),source("LoraManagerModal.jsx"),source("LoraFolderBrowser.jsx"),source("XYZAxisLoraPicker.jsx")]);
  assert.match(manager,/onExternal: transaction \? undefined : \(\) => window\.open\("\/lora"/);
  assert.match(manager,/externalBlockedReason: transaction \?/);
  assert.match(manager,/onEngine: transaction \? undefined : onEngine/);
  assert.match(manager,/transaction && target\.engine !== engine/);
  assert.match(modal,/disabled=\{Boolean\(c\.externalBlockedReason\) \|\| typeof c\.onExternal !== "function"\}/);
  assert.match(modal,/if \(!c\.externalBlockedReason && typeof c\.onExternal === "function"\) c\.onExternal\(\)/);
  assert.match(bar,/disabled=\{Boolean\(blockedReason\) \|\| typeof onSelectEngine !== "function"\}/);
  assert.match(bar,/aria-disabled=\{Boolean\(blockedReason\)/);
  assert.match(bar,/if \(!blockedReason && typeof onSelectEngine === "function"\) onSelectEngine\(name\)/);
  assert.doesNotMatch(picker,/onEngine=\{\(\)=>\{\}\}/);
});

test("capture and repeated raw edits compose each selected word once, identically to mother",()=>{
 let draft=fresh();
 for(let i=0;i<8;i++){draft=patchXYZPrompt(draft,"positive","raw words");assert.equal(draft.baseline.parameters.prompt,composeGenerationPrompt({prompt:"raw words",records,activeIds:["preset-a"]}));}
 assert.equal(draft.baseline.parameters.prompt.split("preset words").length,2);
 assert.equal(xyzPromptSource(draft).positive,"raw words");
});
test("legacy actual migration never guesses prefixes, deletes bytes, or auto-mounts presets",()=>{
 const draft=createXYZDraft(baseline());draft.baseline.parameters.prompt="  group words\n\npreset words\n\nlegacy raw  ";
 const before=structuredClone(draft);
 assert.equal(xyzPromptSource(draft).positive,before.baseline.parameters.prompt);assert.deepEqual(draft,before);
 const next=patchXYZPrompt(draft,"negative","new negative");assert.equal(next.baseline.parameters.prompt,before.baseline.parameters.prompt);
 assert.deepEqual(next.baseline.prompt_source.active,{positive:[],negative:[]});
});
test("library edits/deletion shared as data, no Basic/i2i selections or raw text mutation",()=>{
 const basic={positive:["preset-n"],negative:[]},image={positive:[],negative:["preset-n"]},before=structuredClone({basic,image,records});
 const draft=fresh(),changed=records.map(r=>r.id==="preset-a"?{...r,content:"edited shared words"}:r);
 let next=refreshXYZPromptLibrary(draft,changed);assert.equal(next.baseline.parameters.prompt,"edited shared words\n\nraw words");
 next=refreshXYZPromptLibrary(next,changed.filter(r=>r.id!=="preset-a"));assert.deepEqual(next.baseline.prompt_source.active.positive,[]);assert.equal(next.baseline.parameters.prompt,"raw words");
 assert.deepEqual({basic,image,records},before);assert.equal(draft.baseline.parameters.prompt,"preset words\n\nraw words");
});
test("native unsupported negative retained in provenance, ordinary wire negative unchanged",()=>{
 const draft=fresh("Flux");assert.equal(xyzPromptSource(draft).negative,"raw negative");assert.equal(draft.baseline.parameters.negative_prompt,"");
 assert.equal(xyzWireDraft(draft).baseline.parameters.negative_prompt,"");assert.ok(!Object.hasOwn(xyzWireDraft(draft).baseline,"prompt_source"));
});
test("frozen source survives persistence/explicit Sync and future source is refused",()=>{
 const draft=fresh(),raw={schema_version:1,by_engine:{SD:draft}};assert.equal(inspectXYZDrafts(raw).fatal,false);
 const bad=structuredClone(raw);bad.by_engine.SD.baseline.prompt_source.schema_version=99;assert.equal(inspectXYZDrafts(bad).fatal,true);
 const next=syncXYZBaseline(draft,captureXYZPromptSource(baseline(),{positive:"synced raw",negative:"synced negative",records,active:{positive:[],negative:[]}}),["prompts"]);
 assert.equal(next.baseline.prompt_source.positive,"synced raw");assert.equal(next.baseline.parameters.prompt,"synced raw");assert.deepEqual(next.axes,draft.axes);
});
test("copy/add/group off/on/weight/remove use only independent snapshots and ordered full IDs",()=>{
 const library=structuredClone(group),original=structuredClone(library);let draft=copyXYZManagerGroup(fresh(),library);
 assert.deepEqual(library,original);assert.equal(draft.baseline.loras.length,3);assert.equal(draft.baseline.parameters.prompt,"group words\n\npreset words\n\nraw words");
 draft=updateXYZManagerMounts(draft,m=>m.map((item,i)=>i===1?{...item,weight:-1.2345}:i===2?{...item,weight:0,enabled:false}:item));
 const members=structuredClone(draft.baseline.group_snapshots[0].members);
 draft=updateXYZManagerGroups(draft,g=>g.map(item=>({...item,enabled:false})));assert.equal(draft.baseline.loras.length,1);assert.equal(draft.baseline.parameters.prompt,"preset words\n\nraw words");
 draft=updateXYZManagerGroups(draft,g=>g.map(item=>({...item,enabled:true})));assert.deepEqual(draft.baseline.group_snapshots[0].members,members);assert.equal(draft.baseline.loras[2].weight,0);
 draft=updateXYZManagerMounts(draft,m=>[m[2],m[0],m[1]]);assert.equal(draft.baseline.loras[0].instanceId,member(3).instanceId);
 draft=updateXYZManagerMounts(draft,m=>m.filter(item=>item.instanceId!==member(2).instanceId));assert.equal(draft.baseline.group_snapshots[0].members.length,1);assert.deepEqual(library,original);
});
test("group-panel two callbacks preserve disabled members and then re-enable snapshot exactly",()=>{
 let draft=copyXYZManagerGroup(fresh(),group);const before=structuredClone(draft.baseline.group_snapshots[0].members);
 draft=updateXYZManagerMounts(draft,m=>m.filter(item=>item.groupId!==group.id),{syncGroups:false});
 draft=updateXYZManagerGroups(draft,g=>g.map(item=>({...item,enabled:false})));assert.deepEqual(draft.baseline.group_snapshots[0].members,before);
 draft=updateXYZManagerGroups(draft,g=>g.map(item=>({...item,enabled:true})));assert.deepEqual(draft.baseline.group_snapshots[0].members,before);
});
test("manager refuses cap/identity edits/path collisions, collectors remain draft-only",()=>{
 const draft=fresh();assert.throws(()=>updateXYZManagerMounts(draft,m=>m.map(item=>({...item,instanceId:member(9).instanceId}))));
 const tooMany={...group,members:Array.from({length:16},(_,i)=>member(i+2))};assert.throws(()=>copyXYZManagerGroup(draft,tooMany));
 const collision={...group,members:[{...member(2),value:member(1).value}]};assert.throws(()=>copyXYZManagerGroup(draft,collision));
 let next=updateXYZManagerGroups(draft,g=>[...g,{...group,members:[],enabled:true,collecting:true}]);next=mountXYZCatalogItem(next,member(4));
 assert.equal(next.baseline.loras.at(-1).groupId,group.id);assert.equal(next.baseline.group_snapshots[0].members[0].instanceId,member(4).instanceId);
});
test("single-choice picker replaces the mounted target instead of accumulating",()=>{
 const draft=fresh();assert.deepEqual(draft.baseline.loras.map(item=>item.instanceId),[member(1).instanceId]);
 const replaced=replaceXYZCatalogSelection(draft,member(2));
 assert.equal(replaced.baseline.loras.length,1);assert.equal(replaced.baseline.loras[0].instanceId,member(2).instanceId);
 assert.equal(replaceXYZCatalogSelection(replaced,member(2)),replaced);
 const switched=replaceXYZCatalogSelection(replaced,member(3));assert.equal(switched.baseline.loras.length,1);assert.equal(switched.baseline.loras[0].instanceId,member(3).instanceId);
 assert.deepEqual(draft.baseline.loras.map(item=>item.instanceId),[member(1).instanceId]);
});
test("cross-axis LoRA marks expose other axes' weight targets without touching them",()=>{
 const draft=fresh();
 draft.axes.x={field:"lora_weight",target:member(2).instanceId,values:[{value:null,label:""}],editor_source:{schema_version:1,target_reference:member(2)}};
 draft.axes.z={field:"lora_weight",target:null,values:[{value:null,label:""}]};
 assert.deepEqual(xyzAxisLoraMarks(draft,"y"),{[member(2).instanceId]:["X"],[member(2).value]:["X"]});
 assert.deepEqual(xyzAxisLoraMarks(draft,"x"),{});
 assert.deepEqual(xyzAxisLoraMarks(fresh(),"y"),{});
});
test("plan/submit strip only UI provenance and keep actual primitive axes and snapshot frozen",async()=>{
 const requests=[],draft=fresh();draft.axes.x={field:"prompt",values:[{value:"axis actual"}]};
 const api=xyzApi(async(url,options)=>{requests.push(JSON.parse(options.body));return {ok:true,json:async()=>({})};});
 await api.plan(draft);await api.submit(draft,{submissionId:"test-id"});assert.deepEqual(requests[0],xyzWireDraft(draft));assert.deepEqual(requests[1].draft,requests[0]);
 assert.equal(typeof requests[0].axes.x.values[0].value,"string");assert.ok(draft.baseline.prompt_source);assert.equal(requests[1].draft.baseline.parameters.seed,"18446744073709551615");
});
test("unknown saved references remain exact until explicit removal; duplicate/revision/bad group refuses atomically",()=>{
 const draft=fresh();draft.baseline.loras.push({name:"Saved missing",value:"missing.safetensors",weight:"0",enabled:false,precision:4});
 const next=updateXYZManagerMounts(draft,m=>[m[1],m[0]]);assert.deepEqual(next.baseline.loras[0],draft.baseline.loras[1]);
 assert.throws(()=>copyXYZManagerGroup(draft,{...group,members:[{...member(1),contentRevision:4}]}));
 assert.throws(()=>copyXYZManagerGroup(draft,{...group,members:[{...member(2),weight:8}]}));
 assert.throws(()=>copyXYZManagerGroup(draft,{...group,members:[member(2),member(2)]}));
 assert.equal(draft.baseline.loras.length,2);
});
