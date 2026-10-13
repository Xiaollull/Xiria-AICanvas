import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { LoraRelinkService } from './lora-relink-service.mjs';
import { galleryOwnerParticipant } from './lora-gallery-owner.mjs';
import { emptyLoraRegistry, scanLoraInventory, registerInitialInventory } from './lora-library.mjs';
import { xyzPlotMiddleware } from './xyz-plot-http.mjs';
import { xyzWireDraft } from '../src/xyz-prompt-source.js';
import { pythonDependencyOptions } from './python-dependency.test.helper.mjs';

const cwd = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.P10_PYTHON || 'python';
const pythonOptions = pythonDependencyOptions(python, cwd, ['PIL', 'pydantic']);
const pythonTest = (name, body) => test(name, pythonOptions, body);
const axis = (field, values, extra = {}) => ({ field, values: values.map(value => ({ value })), ...extra });
async function fixture(names = ['A','B','C']) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'xyz-sol3-admission-'));
  const stateDirectory = path.join(root, 'state'), models = path.join(root, 'models');
  await mkdir(path.join(stateDirectory, 'lora-library'), { recursive: true });
  await mkdir(models);
  for (const name of names) await writeFile(path.join(models, `${name}.safetensors`), `owned adapter ${name}`);
  const gallery = galleryOwnerParticipant(stateDirectory, python);
  const roots = [{ engine: 'SD', id: 'local-SD', directory: models }];
  const scan = await scanLoraInventory(roots, { fileIdentity: file => gallery.identity(file) });
  const registry = registerInitialInventory(emptyLoraRegistry(), scan);
  const refs = Object.fromEntries(Object.values(registry.instances).map(item => [item.value[0], { value: item.value, instanceId: item.instanceId,
    contentHash: item.contentHash, contentRevision: item.contentRevision, identitySchema: 1, precision: 4, weight: '1', enabled: true }]));
  const put = async () => {
    await writeFile(path.join(stateDirectory, 'lora-library/registry.json'), JSON.stringify(registry));
    await writeFile(path.join(stateDirectory, 'lora-library/publication.json'), JSON.stringify({ schemaVersion: 1, phase: 'published', revision: registry.revision }));
  };
  await put();
  const result = spawnSync(python, ['-B', '-c', 'import json; from backend.tests.test_xyz_planner import baseline; print(json.dumps(baseline()))'], { cwd, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const baseline = JSON.parse(result.stdout);
  baseline.axes = { x: null, y: null, z: null };
  const service = new LoraRelinkService({ stateDirectory, gallery, getRoots: async () => roots, admitIdle: async () => {},
    validatePath: async (_engine, value) => { assert.ok(names.map(name=>`${name}.safetensors`).includes(value)); } });
  const values = async draft => (await service.resolveXYZ(draft)).effectiveReferences.map(item => item.value);
  return { root, stateDirectory, service, gallery, registry, refs, baseline, put, values };
}

pythonTest('real private planner + Node: all-full ignores stale unused baseline; mixed overlay requires it; no-test/no-loras/full-clear differ', async () => {
  const f = await fixture(), { A, B } = f.refs;
  f.registry.instances[A.instanceId].status = 'missing'; await f.put();
  const d = structuredClone(f.baseline); d.baseline.loras = [A];
  d.axes.x = axis('lora_profile', [{ mode: 'full', members: [B] }]);
  assert.deepEqual(await f.values(d), ['B.safetensors']);
  const forwarded = [];
  const route = xyzPlotMiddleware({ service: f.service, readJson: async request => request.body,
    admit: request => assert.equal(request.headers.origin, 'http://local'),
    forward: async (...args) => { forwarded.push(args); args[4].statusCode = 202; args[4].end('{}'); } });
  async function invoke() {
    const response = { statusCode: 200, setHeader() {}, end(text) { this.text = text; } };
    await route({ url: '/api/inference/xyz-plots/plan', method: 'POST', body: d,
      headers: { origin: 'http://local', 'x-xirai-lora-revision': String(f.registry.revision) } }, response, () => assert.fail());
    return response;
  }
  assert.equal((await invoke()).statusCode, 202, 'effective full profile reaches Node admission without unused stale A');
  assert.equal(forwarded.length, 1);
  d.axes.x = axis('lora_profile', [{ mode: 'full', members: [B] }, { mode: 'keep_background', members: [B] }]);
  await assert.rejects(f.values(d), /identity\/content\/revision/);
  assert.notEqual((await invoke()).statusCode, 202, 'the actually used stale baseline remains refused');
  assert.equal(forwarded.length, 1);
  d.axes.x = axis('lora_profile', [{ mode: 'no_loras' }, { mode: 'full', members: [] }]);
  assert.deepEqual(await f.values(d), []);
  d.axes.x = axis('lora_profile', [{ mode: 'no_test', test_targets: [A.instanceId] }]);
  assert.deepEqual(await f.values(d), []);
  d.axes.x = axis('lora_profile', [{ mode: 'no_test', test_targets: [] }]);
  await assert.rejects(f.values(d), /identity\/content\/revision/);
  d.axes.x = null; d.baseline.loras = [{ ...A, enabled: false }, { ...B, weight: '0' }];
  assert.deepEqual(await f.values(d), []);
});

pythonTest('new UI projection enters real Node→Python leased compiler/admission: five background + unmounted candidate, weight, member multiplier, explicit replace and CAS refusal',async()=>{
 const f=await fixture(['A','B','C','D','E','F']);const d=structuredClone(f.baseline),extra=f.refs.F;d.baseline.loras=['A','B','C','D','E'].map(k=>f.refs[k]);
 const before=structuredClone(d.baseline);
 for(const axis of [
  {field:'lora_model',values:[{value:extra}],editor_source:{schema_version:1,operation:'add'}},
  {field:'lora_weight',target:extra.instanceId,values:[],range:{start:'0.5',end:'1.0',step:'0.25'},editor_source:{schema_version:1,target_reference:extra}},
  {field:'lora_multiplier',values:[{value:'0.5'}],snapshot:{mode:'keep_background',members:[{...extra,weight:'0.8124'}]},editor_source:{schema_version:1,member_only:true}},
 ]){d.axes.x=axis;const wire=xyzWireDraft(d);assert.deepEqual(new Set(await f.values(wire)),new Set(['A','B','C','D','E','F'].map(k=>`${k}.safetensors`)));assert.deepEqual(d.baseline,before);}
 d.axes.x={field:'lora_model',target:f.refs.A.instanceId,values:[{value:extra}],editor_source:{schema_version:1,operation:'replace'}};assert.deepEqual(new Set(await f.values(xyzWireDraft(d))),new Set(['B','C','D','E','F'].map(k=>`${k}.safetensors`)));
 d.axes.x.editor_source.operation='add';delete d.axes.x.target;f.registry.instances[extra.instanceId].contentRevision+=1;await f.put();await assert.rejects(f.values(xyzWireDraft(d)),/identity\/content\/revision/);
});

pythonTest('real projection preserves decimal multipliers, targeted zero slots/model candidates, preset/group and replacement inline refusal', async () => {
  const f = await fixture(), { A, B, C } = f.refs;
  const d = structuredClone(f.baseline); d.baseline.loras = [A];
  d.axes.x = axis('lora_multiplier', ['0', '0.5'], { snapshot: { mode: 'full', members: [{ ...B, weight: '0.8124' }], includePresetText: true, preset_prompt: 'group keywords' } });
  const result = await f.service.resolveXYZ(d);
  assert.deepEqual(result.effectiveReferences.map(item => [item.value, item.weight]), [['B.safetensors', 0.4062]]);
  d.axes.x.snapshot.members[0].weight = '0.8137'; await assert.rejects(f.values(d), /no rounding applied/);
  d.axes.x.values = [{ value: '0' }]; assert.deepEqual(await f.values(d), []);
  d.axes.x = axis('lora_weight', ['0', '0.5'], { target: A.instanceId });
  d.baseline.loras = [{ ...A, enabled: false, weight: '0' }];
  assert.deepEqual(await f.values(d), ['A.safetensors']);
  d.axes.x = axis('lora_model', [B, C], { target: A.instanceId });
  assert.deepEqual(await f.values(d), ['B.safetensors', 'C.safetensors']);
  d.axes.x = axis('prompt', ['cat <lora:B:1>']);
  await assert.rejects(f.values(d), /Inline LoRA requires/); // Current planner NEVER executes inline XYZ.
  d.axes.x = axis('prompt_replace', [{ find: 'cat', replace: '<lora:C:1>', expected_matches: 1 }]);
  await assert.rejects(f.values(d), /Inline LoRA requires/);
  d.axes.x = axis('lora_profile', [{ mode: 'full', members: [], includePresetText: true, preset_prompt: '<lora:B:1>' }]);
  await assert.rejects(f.values(d), /Inline LoRA requires/);
});

pythonTest('unmounted independent weight axes enter unchanged leased Node→Python compiler; same-instance/profile/strong binding guards stay closed',async()=>{
 const f=await fixture(['A','B','C','D','E','F','G','H']), d=structuredClone(f.baseline);
 const weight=m=>({field:'lora_weight',target:m.instanceId,values:[],range:{start:'0.2',end:'1',step:'0.3'},editor_source:{schema_version:1,target_reference:m}});
 d.axes.x=weight(f.refs.F);d.axes.y=weight(f.refs.G);
 assert.deepEqual(new Set(await f.values(xyzWireDraft(d))),new Set(['F.safetensors','G.safetensors']));
 d.axes.z=weight(f.refs.H);assert.deepEqual(new Set(await f.values(xyzWireDraft(d))),new Set(['F.safetensors','G.safetensors','H.safetensors']));
 d.baseline.loras=['A','B','C','D','E'].map(k=>f.refs[k]);const before=structuredClone(d.baseline);
 assert.deepEqual(new Set(await f.values(xyzWireDraft(d))),new Set(['A','B','C','D','E','F','G','H'].map(k=>`${k}.safetensors`)));assert.deepEqual(d.baseline,before);
 const forwarded=[],route=xyzPlotMiddleware({service:f.service,readJson:async req=>req.body,admit:()=>{},forward:async(...args)=>{forwarded.push(args);args[4].statusCode=202;args[4].end('{}');}});
 const response={statusCode:200,setHeader(){},end(text){this.text=text;}};
 await route({url:'/api/inference/xyz-plots/plan',method:'POST',body:xyzWireDraft(d),headers:{'x-xirai-lora-revision':String(f.registry.revision)}},response,()=>assert.fail());assert.equal(response.statusCode,202);assert.equal(forwarded.length,1);
 d.axes.z=null;d.axes.y=weight(f.refs.F);await assert.rejects(f.values(xyzWireDraft(d)),/Multiple axes/);
 d.axes.y=axis('lora_profile',[{mode:'keep_background',members:[]}]);await assert.rejects(f.values(xyzWireDraft(d)),/Multiple axes/);
 d.axes.y=weight(f.refs.G);f.registry.instances[f.refs.G.instanceId].contentRevision++;await f.put();await assert.rejects(f.values(xyzWireDraft(d)),/identity\/content\/revision/);
});

pythonTest('effective identity/content/root/engine/trace guard and publication barrier remain closed; middleware still refuses stale revision and forged origin', async () => {
  const f = await fixture(), { A, B } = f.refs;
  const d = structuredClone(f.baseline); d.baseline.loras = [A];
  for (const bad of [{ ...B, instanceId: undefined }, { ...B, contentHash: 'f'.repeat(64) }, { ...B, contentRevision: 2 }, { ...B, value: '../B.safetensors' }]) {
    d.axes.x = axis('lora_profile', [{ mode: 'full', members: [bad] }]);
    await assert.rejects(f.values(d));
  }
  d.axes.x = axis('lora_profile', [{ mode: 'full', members: [B] }]);
  f.registry.instances[B.instanceId].engine = 'Anima'; await f.put(); await assert.rejects(f.values(d));
  f.registry.instances[B.instanceId].engine = 'SD'; f.registry.instances[B.instanceId].rootId = 'disabled-root'; await f.put();
  await assert.rejects(f.values(d), /root disabled/);
  f.registry.instances[B.instanceId].rootId = 'local-SD'; await f.put();
  await writeFile(path.join(f.root, 'models/B.safetensors'), 'changed bytes'); await assert.rejects(f.values(d), /bytes changed/);
  await writeFile(path.join(f.root, 'models/B.safetensors'), 'owned adapter B'); // only our fixture
  const forwarded = [];
  const route = xyzPlotMiddleware({ service: f.service, readJson: async request => request.body,
    admit: request => { if (request.headers.origin !== 'http://local') throw Object.assign(new Error('origin'), { statusCode: 403 }); },
    forward: async (...args) => { forwarded.push(args); args[4].end('{}'); } });
  async function invoke(headers) {
    const response = { statusCode: 200, setHeader() {}, end(text) { this.text = text; } };
    await route({ url: '/api/inference/xyz-plots/plan', method: 'POST', body: d, headers }, response, () => assert.fail()); return response;
  }
  assert.equal((await invoke({ origin: 'http://local', 'x-xirai-lora-revision': '0' })).statusCode, 409);
  assert.equal((await invoke({ origin: 'http://evil', 'x-xirai-wd14-proxy-proof': 'forged' })).statusCode, 403);
  assert.equal(forwarded.length, 0);
});

pythonTest('private stdin action rejects absent/forged lease (no new public endpoint or protocol-fingerprint authorization)', async () => {
  const f = await fixture();
  for (const token of [null, 'forged']) {
    const result = spawnSync(python, ['-B','-m','backend.lora_relink_owner'], { cwd, encoding: 'utf8',
      input: JSON.stringify({ action: 'xyz-references', stateDirectory: f.stateDirectory, token, draft: f.baseline }) });
    assert.notEqual(result.status, 0);
  }
  const before = await readFile(path.join(f.stateDirectory, 'lora-library/registry.json'));
  await assert.rejects(f.gallery.xyzReferences(f.baseline, 'forged'));
  assert.deepEqual(await readFile(path.join(f.stateDirectory, 'lora-library/registry.json')), before);
});
