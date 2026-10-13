import { emptyLoraCardStore } from "../src/lora-cards.js";
import { createXYZDraft } from "../src/xyz-draft.js";
await import("./xyz-sol-app.fixture.jsx");
const base=window.__appFetch, engines=["SD","iL","Anima","Flux","Flux2","Krea2"];
const reference=(engine,index)=>({name:`${engine} Portrait ${index}`,value:`${engine}-portrait-${index}.safetensors`,weight:0.8137,precision:4,enabled:true,
 identitySchema:1,instanceId:`00000000-0000-4000-8000-${String(engines.indexOf(engine)*10+index+1).padStart(12,"0")}`,contentHash:String(index+1).repeat(64),contentRevision:1,size:1024*1024});
const metadata={status:"found",detailSchema:1,triggerReviewSchema:4,modelName:"Fixture source",versionName:"v1",baseModel:"SD",triggerGroups:[],previewFile:"fixture.png",sourceUrl:"https://example.invalid/fixture",sourceDomain:"Fixture"};
let cards=emptyLoraCardStore();
for(const engine of engines)cards.byEngine[engine].byInstance[reference(engine,0).instanceId]={...reference(engine,0),title:`Custom ${engine} portrait`,prompt:"card fixture words",cover:"a".repeat(64),tags:["Fixture"]};
const response=(data,status=200)=>new Response(JSON.stringify(data),{status,headers:{"Content-Type":"application/json","X-XirAI-LoRA-Revision":"0"}});
window.__appFetch=async(url,method,body)=>{
 if(url.startsWith("/api/lora-cards")){if(method==="PUT")cards=structuredClone(body.store);return response({store:cards});}
 if(url.startsWith("/api/lora-folder")){
   if(body.path.includes("outside"))return response({error:"Fixture: outside registered roots"},400);
   return response({folder:{engine:body.path.includes("Flux")?"Flux":body.engine,origin:"local",categoryId:"character",folder:"",display:"fixture/read-only/character"}});
 }
 if(url.startsWith("/api/lora-lookup"))return response({metadata});
 const answer=await base(url,method,body);
 if(url.startsWith("/api/loras?")){
   const data=await answer.json();const engine=data.engine;
   data.categories=[{id:"character",label:"人物",directory:"fixture/read-only/character",models:Array.from({length:5},(_,i)=>({...reference(engine,i),metadata}))},
    {id:"other",label:"其他",directory:"fixture/read-only/other",models:[{...reference(engine,5),name:`nested/${engine}-portrait-5.safetensors`,value:`nested/${engine}-portrait-5.safetensors`,metadata}]},
    {id:"shared",label:"共享",shared:true,models:[{...reference(engine,6),metadata}],roots:[{id:"fixture-root",label:"Shared fixture",path:"fixture/shared-read-only",files:1,bytes:1024*1024,folders:[{name:"portraits",label:"Portraits",models:[{...reference(engine,6),metadata}]}]}]}];return response(data);
 }
 if(url.startsWith("/api/ui-state")&&method!=="PUT"){
   const data=await answer.json();for(const engine of engines)data.state.workspace.loraGroupsByEngine.byEngine[engine]=[{id:`g-${engine.toLowerCase()}`,name:`Library ${engine}`,enabled:false,collecting:false,presetPrompt:"group preset",members:Array.from({length:5},(_,i)=>reference(engine,i))}];
   if(new URLSearchParams(location.search).has('legacy')) {
     data.state.workspace.model='Anima';data.state.workspace.generationMode='xyz_plot';
     data.state.workspace.xyzByEngine={schema_version:1,by_engine:{Anima:createXYZDraft({parameters:{engine:'Anima',diffusion_model:'Anima-diffusion.safetensors',text_encoder:'Anima-encoder.safetensors',vae:'Anima-vae.safetensors',prompt:'  group preset\n\nlegacy already composed  ',negative_prompt:'legacy negative',width:768,height:1024,steps:28,cfg:6.5,seed:'123456789',sampler:'euler',scheduler:'simple',guidance:'none',pag:{scale:0.3,applied_layers:'mid'},postprocess_chain:[]},loras:[reference('Anima',0)],group_snapshots:[],prompt_without_groups:'legacy already composed',group_prompt:'group preset',seed_mode:'fixed',library_revision:0})}};
   }
   return response(data);
 }
 return answer;
};
