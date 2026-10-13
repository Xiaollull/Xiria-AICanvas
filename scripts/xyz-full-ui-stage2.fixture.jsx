import { createResampleUnit, createAdetailerStage, createRtxStage } from "../src/resample-chain.js";
await import("./xyz-full-ui-stage1.fixture.jsx");
const base = window.__appFetch;
const response = (data) => new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json", "X-XirAI-LoRA-Revision": "0" } });
window.__appFetch = async (url, method, body) => {
  if(new URLSearchParams(location.search).has('seven') && url.startsWith('/api/ui-state') && method==='PUT') sessionStorage.setItem('xyz-seven-private-ui',JSON.stringify(body));
  const result = await base(url, method, body);
  if (!result) return null;
  if(new URLSearchParams(location.search).has('weight') && url.startsWith('/api/loras?')) {
    const data=await result.json(), sample=data.categories[0].models[0];
    const make=(i,name)=>({...sample,name,value:`private/${name}.safetensors`,instanceId:`00000000-0000-4000-8000-${String(2000+i).padStart(12,'0')}`,contentHash:(2000+i).toString(16).padStart(64,'0')});
    const shared=data.categories.find(c=>c.shared);shared.models=['kiriaina_anima_style-000030','yusano_v5_step1500','third-independent'].map((name,i)=>make(i,name));shared.roots[0].files=3;shared.roots[0].folders[0].models=shared.models;
    data.categories.push({id:'cap',label:'上限夹具',directory:'private/cap',models:Array.from({length:16},(_,i)=>make(100+i,`cap-${i}`))});
    (window.__weightCatalog ||= {})[data.engine]=data.categories.flatMap(c=>c.models);return response(data);
  }
  if(new URLSearchParams(location.search).has('rework') && url.startsWith('/api/loras?')) {
    const data=await result.json(), original=data.categories[0].models[0];
    const make=(i)=>({...original,name:`${data.engine} Tall metadata ${i}`,value:`tall-${i}.safetensors`,instanceId:`00000000-0000-4000-8000-${String(1000+i).padStart(12,'0')}`,contentHash:i.toString(16).padStart(64,'0'),metadata:{...original.metadata,modelName:'Long catalog model description '.repeat(12),versionName:'Revision with long descriptive metadata '.repeat(6),queriedAt:123}});
    data.categories.push({id:'tall',label:'高封面',directory:'private/tall',models:[{...make(71),name:'rossi_v2'}]});
    const shared=data.categories.find(c=>c.shared);shared.models=Array.from({length:68},(_,i)=>make(100+i));shared.roots[0].files=68;shared.roots[0].folders[0].models=shared.models;
    return response(data);
  }
  if (url.startsWith("/api/ui-state") && method !== "PUT") {
    const data = await result.json();
    if(new URLSearchParams(location.search).has('seven') && sessionStorage.getItem('xyz-seven-private-ui')) data.state=JSON.parse(sessionStorage.getItem('xyz-seven-private-ui'));
    if(new URLSearchParams(location.search).has('weight')) {
      const group=data.state.workspace.loraGroupsByEngine.byEngine.Anima[0], members=group.members;
      group.members=[4,1,3,0,2].map((i,position)=>({...members[i],weight:[-0.25,0,1.2345,0.2,-0.8124][position],enabled:position!==2}));
    }
    for (const [engine, settings] of Object.entries(data.state.workspace.engineSettingsByEngine.byEngine)) settings.postprocessChain = [createResampleUnit({ id: `r-${engine}`, enabled: false, expanded: false, scale: 1.5, seedMode: "random", denoise: 0 }, { engine }), createAdetailerStage({ id: `ade-${engine}`, enabled: false }), createRtxStage({ id: `rtx-${engine}`, enabled: false })];
    data.state.workspace.postprocessChain = data.state.workspace.engineSettingsByEngine.byEngine.SD.postprocessChain;
    return response(data);
  }
  if (url.startsWith("/api/inference/health")) {
    const data = await result.json();
    data.rtx_vsr = { available: true, supported: true, device: "CPU fixture display, not GPU", runtime_version: "fixture", compute_capability: "--" };
    for (const engine of Object.keys(data.engines)) data.engines[engine].features = { lora: true, hires: true, adetailer: true, rtx: true, pag: ["SD", "iL", "Anima"].includes(engine) };
    data.upscalers.runtime_available = true; return response(data);
  }
  if (url.startsWith("/api/inference/adetailer/models")) return response({ models: [{ value: "fixture-face.pt", name: "Fixture face" }], builtins: [], available: true, runtime_available: true, directory: "fixture/read-only", ...(window.__adeRefreshDefault ? { default: "fixture-new-default.pt" } : {}) });
  return result;
};
const fetchBase = window.fetch;
window.fetch = async (input, options) => String(input).endsWith('/api/inference/xyz-plots/capabilities') ? response({ executor_ready: true, max_concurrency: 6, protocol: 39 }) : fetchBase(input, options);
