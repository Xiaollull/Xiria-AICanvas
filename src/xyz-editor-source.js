import { decimalRangeValues } from './xyz-draft.js';
import { CONTENT_HASH, INSTANCE_ID, inspectLoraIdentity } from './lora-identity.js';

export const XYZ_SEED_MODES = [{ id: 'random', label: '全随机' }, { id: 'fixed', label: '固定' }, { id: 'increment', label: '递增' }, { id: 'decrement', label: '递减' }];
const MAX = 18446744073709551615n;
export const validXYZSeedSource = s => s && s.schema_version === 1 && XYZ_SEED_MODES.some(m => m.id === s.mode) && Object.keys(s).every(k => ['schema_version', 'mode'].includes(k));
export const validXYZAxisSource = s => s && s.schema_version === 1 && Object.keys(s).every(k => ['schema_version', 'operation', 'target_reference', 'member_only'].includes(k)) && (s.operation === undefined || ['add', 'replace'].includes(s.operation)) && (s.member_only === undefined || typeof s.member_only === 'boolean') && (s.target_reference === undefined || s.target_reference && typeof s.target_reference === 'object' && !Array.isArray(s.target_reference));
export function xyzSampleSeeds(draft) {
 const source = draft.baseline.seed_source;
 if (!source) return draft.sample_seeds;
 if (!validXYZSeedSource(source)) throw new Error('种子来源版本无法安全读取');
 if (Object.values(draft.axes).some(a => a?.field === 'seed') || source.mode === 'random') return null;
 const text = draft.baseline.parameters.seed;
 if (!/^\d{1,20}$/.test(text) || BigInt(text) > MAX) throw new Error('种子必须为 uint64 十进制文本');
 if (!Number.isInteger(draft.repeat_count) || draft.repeat_count < 1 || draft.repeat_count > 32) throw new Error('重复样本必须为 1–32 整数');
 const start = BigInt(text), delta = source.mode === 'increment' ? 1n : source.mode === 'decrement' ? -1n : 0n;
 return Array.from({length: draft.repeat_count}, (_, i) => String((start + BigInt(i) * delta + MAX + 1n) % (MAX + 1n)));
}
export function xyzExpectedValues(values, raw) {
 if (!/^\d{1,3}$/.test(raw) || Number(raw) < 1 || Number(raw) > 256) throw new Error('预期数量须为 1–256 整数');
 return Array.from({length: Number(raw)}, (_, i) => i < values.length ? structuredClone(values[i]) : {value: null, label: ''});
}
function multiply(a, b) {
 const parse = raw => { const s = String(raw); if (!/^-?(?:\d+(?:\.\d{1,16})?|\.\d{1,16})$/.test(s)) throw new Error('倍率与成员权重需要有限十进制文本'); const [w, f=''] = s.replace('-', '').split('.'); return [BigInt((w||'0')+f) * (s.startsWith('-') ? -1n : 1n), f.length]; };
 const [x,p] = parse(a), [y,q] = parse(b), n=x*y, places=p+q, scale=10n**BigInt(places), abs=n<0n?-n:n;
 return `${n<0n?'-':''}${abs/scale}${places?'.'+String(abs%scale).padStart(places,'0').replace(/0+$/,''):''}`.replace(/\.$/,'');
}
// New additive UI only: equal whole frozen prefixes are the same prefix, not a substring edit.
const additivePreset = (background, candidate) => background === candidate ? background : [background,candidate].filter(Boolean).join('\n\n');
const bindingFields = ['identitySchema', 'instanceId', 'contentHash', 'contentRevision', 'value'];
const stableSource = value => Array.isArray(value) ? value.map(stableSource) : value && typeof value === 'object'
 ? Object.fromEntries(Object.keys(value).sort().map(k => [k, stableSource(value[k])])) : value;
function weightTarget(baseline, axis, reference, sources) {
 if (reference.identitySchema !== 1 || !INSTANCE_ID.test(reference.instanceId) || !CONTENT_HASH.test(reference.contentHash)
   || !Number.isSafeInteger(reference.contentRevision) || reference.contentRevision < 1 || inspectLoraIdentity(reference).fatal
   || typeof reference.value !== 'string' || !reference.value || reference.value.length > 500 || reference.value.includes('\0') || reference.missing
   || axis.target !== reference.instanceId) throw new Error('LoRA 权重目标缺少完整强身份或与轴目标不一致；请明确重新选择。');
 // Equal ids never authorize contradictory frozen sources, even if one axis would overwrite the other.
 const signature = JSON.stringify(stableSource(reference));
 if (sources.has(axis.target) && sources.get(axis.target) !== signature) throw new Error('同一 LoRA 权重目标的轴来源不一致；不会静默合并。');
 sources.set(axis.target, signature);
 const existing = baseline.loras.filter(m => m.instanceId === axis.target);
 if (existing.length > 1 || existing.some(m => bindingFields.some(k => m[k] !== reference[k]))) throw new Error('LoRA 权重目标与冻结背景的实例 / 路径 / 内容修订不一致；请明确重新选择。');
 if (existing.length) return; // The global slot owns all metadata, weight, enabled and order.
 if (baseline.loras.some(m => m.value === reference.value)) throw new Error('同路径 LoRA 的实例不同；不会按路径重绑。');
 if (baseline.loras.length >= 16) throw new Error('轴目标加入提交基线后超过 16 个 LoRA 挂载上限；原始全局列表未改动。');
 // A submission-only inactive slot. The unchanged planner assigns each axis's value and enables it;
 // zero is omitted there. Never compose prompts/groups or write this slot back into the UI draft.
 baseline.loras.push({ ...reference, weight: '0', enabled: false });
}
export function projectXYZEditor(draft) {
 const next = structuredClone(draft);
 const weightSources = new Map();
 if (next.baseline?.seed_source) { next.sample_seeds = xyzSampleSeeds(next); delete next.baseline.seed_source; }
 for (const key of ['x','y','z']) {
  const axis=next.axes?.[key], s=axis?.editor_source;
  if (!s) continue;
  if (!validXYZAxisSource(s)) throw new Error('轴编辑来源版本无法安全读取');
  delete axis.editor_source;
  const profile = members => ({mode:'keep_background', members, test_targets:[], includePresetText:false, preset_prompt:''});
  if (axis.field === 'lora_model' && s.operation === 'add') {axis.field='lora_profile';axis.target=null;axis.values=axis.values.map(e=>({...e,value:profile([e.value])}));}
  else if (axis.field === 'lora_profile' && s.operation === 'add') {axis.values=axis.values.map(e=>({...e,value:e.value?.includePresetText && ['keep_background','no_test'].includes(e.value.mode) ? {...e.value,preset_prompt:additivePreset(next.baseline.group_prompt,e.value.preset_prompt)} : e.value}));}
  else if (axis.field === 'lora_weight' && s.target_reference) weightTarget(next.baseline, axis, s.target_reference, weightSources);
  else if (axis.field === 'lora_multiplier' && s.member_only && axis.snapshot?.mode === 'keep_background') {
   const values=axis.range?decimalRangeValues(axis.range).map(value=>({value,label:''})):axis.values;
   if(!axis.snapshot.members?.length)throw new Error('倍率测试集合尚未定义；请在独立管理器选择成员');
   axis.field='lora_profile';axis.range=null;axis.values=values.map(e=>({...e,value:{...axis.snapshot,members:axis.snapshot.members.map(m=>({...m,weight:multiply(m.weight,e.value)})),
    ...(axis.snapshot.includePresetText ? {preset_prompt:additivePreset(next.baseline.group_prompt,axis.snapshot.preset_prompt)} : {})}}));axis.snapshot=null;
  }
 }
 return next;
}
