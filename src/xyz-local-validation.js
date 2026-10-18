import capabilities from './xyz-capabilities.json' with { type: 'json' };

// Deterministic input guards, not resource admission or a second planner. Keep raw drafts intact.
const NUMERIC = new Set(['steps', 'cfg', 'clip_skip', 'pag.scale', 'lora_weight', 'lora_multiplier']);
const STRING_FIELDS = new Set(['prompt','negative_prompt','sampler','scheduler','guidance','checkpoint','diffusion_model','text_encoder','text_encoder_2','vae']);
const weightOf = member => Object.hasOwn(member, 'weight') ? member.weight : 1;
function decimal(value) {
  if (!['string', 'number'].includes(typeof value)) throw new Error('需要有限十进制数');
  const text = String(value);
  if (text.length > 64 || !/^-?(?:\d+(?:\.\d{0,16})?|\.\d{1,16})$/.test(text)) throw new Error('需要有限十进制数（最多 16 位小数）');
  const [whole, fraction = ''] = text.replace('-', '').split('.');
  const scale = 10n ** BigInt(fraction.length);
  const n = (BigInt(whole || '0') * scale + BigInt(fraction || '0')) * (text.startsWith('-') ? -1n : 1n);
  if (n > 10n ** 20n * scale || n < -(10n ** 20n * scale)) throw new Error('十进制数不能超过 ±1e20');
  return { n, scale };
}
function bounded(value, min, max, { integer = false, weight = false } = {}) {
  const d = decimal(value);
  if (d.n < BigInt(min) * d.scale || d.n > BigInt(max) * d.scale || integer && d.n % d.scale) throw new Error(`需要 ${min}–${max}${integer ? ' 整数' : ' 数值'}`);
  if (weight && d.n * 10000n % d.scale) throw new Error('权重最多 4 位有效小数；不会自动舍入');
  return d;
}
export function xyzAxisValueIssue(field, value, engine) {
  if (value === null || value === undefined || value === '') return '待定义';
  try {
    if (field === 'seed') {
      if (typeof value !== 'string' || !/^\d{1,20}$/.test(value) || BigInt(value) > 18446744073709551615n) throw new Error('种子需要 0–18446744073709551615 的 uint64 字符串');
    } else if (field === 'steps') bounded(value, 1, 100, {integer:true});
    else if (field === 'cfg') bounded(value, 0, 30);
    else if (field === 'clip_skip') {
      const limits = capabilities.engines[engine]?.clip_skip;
      if (!limits) throw new Error('当前引擎不支持 CLIP skip');
      bounded(value, ...limits, {integer:true});
    } else if (field === 'pag.scale') bounded(value, 0, 5);
    else if (field === 'lora_weight') bounded(value, -5, 5, {weight:true});
    else if (field === 'lora_multiplier') bounded(value, -5, 5);
    else if (STRING_FIELDS.has(field) && typeof value !== 'string') throw new Error('需要文本值；原始值保留');
    else if (field === 'guidance' && !capabilities.engines[engine]?.guidance.includes(value)) throw new Error('当前引擎不支持此引导类型');
    // An engine may narrow the shared vocabulary (MOSS offers only its exact k-diffusion samplers).
    else if (field === 'sampler' && !(capabilities.engines[engine]?.samplers || capabilities.samplers).includes(value) || field === 'scheduler' && !(capabilities.engines[engine]?.schedulers || capabilities.schedulers).includes(value)) throw new Error('未知采样器或调度器');
    else if (['components','prompt_replace','negative_replace','lora_model','lora_profile'].includes(field) && (typeof value !== 'object' || Array.isArray(value))) throw new Error('未解析 JSON；原文保留，请修复');
    else if (typeof value === 'string' && !value.trim()) throw new Error('待定义');
    return '';
  } catch (error) { return error.message; }
}
function memberIssue(member) {
  if (!member || typeof member !== 'object' || Array.isArray(member)) throw new Error('LoRA 成员尚未定义或含未解析 JSON');
  bounded(weightOf(member), -5, 5, {weight:true});
  if (![1,2,4].includes(member.precision === undefined ? 4 : member.precision)) throw new Error('LoRA 精度须为 1、2 或 4');
}
function upsert(entries, member, target) {
  const index = entries.findIndex(m => m.instanceId === (target || member.instanceId));
  if (target && index < 0) throw new Error('LoRA 目标不在冻结背景中；请明确选择');
  return index < 0 ? [...entries, member] : entries.map((m,i) => i === index ? member : m);
}
function profile(entries, value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('LoRA 快照含未解析 JSON');
  const members = value.members === undefined ? [] : value.members, targets = value.test_targets === undefined ? [] : value.test_targets;
  if (!Array.isArray(members) || !Array.isArray(targets)) throw new Error('LoRA 快照需要有序成员和测试目标');
  members.forEach(memberIssue);
  if (value.mode === 'no_loras') { if (members.length || targets.length) throw new Error('完全无 LoRA 必须为空集合'); return []; }
  if (value.mode === 'full') return members;
  if (!['keep_background','no_test'].includes(value.mode) || value.mode === 'no_test' && members.length) throw new Error('LoRA 快照模式或成员无效');
  let result = entries.filter(m => !targets.includes(m.instanceId) || members.some(n => n.instanceId === m.instanceId));
  for (const member of members) result = upsert(result, member);
  return result;
}
export function xyzLocalFieldIssues(draft, expanded, wire) {
  const issues = [];
  for (const key of ['x','y','z']) {
    const axis = draft.axes[key];
    if (!axis) continue;
    if (axis.range && (!NUMERIC.has(axis.field) || axis.values.length)) issues.push(`${key.toUpperCase()}：只有数值字段可使用阶梯，且阶梯/列表不能同时定义`);
    expanded[key].forEach((value,i) => {
      const error = xyzAxisValueIssue(axis.field, value, draft.engine);
      if (error) issues.push(`${key.toUpperCase()} 第 ${i+1} 项 · ${axis.field}：${error}`);
    });
    if (axis.field === 'lora_multiplier') try {
      if (!axis.snapshot || typeof axis.snapshot !== 'object' || Array.isArray(axis.snapshot)) throw new Error('倍率快照含未解析 JSON');
      const members = axis.snapshot.members === undefined ? [] : axis.snapshot.members;
      if (!Array.isArray(members)) throw new Error('倍率快照需要有序成员列表；原始值保留');
      members.forEach(memberIssue);
    } catch(error) { issues.push(`${key.toUpperCase()} 倍率快照：${error.message}`); }
  }
  // Baseline scalars are always parsed by the strict backend, even with random/seed axes.
  const p = draft.baseline.parameters;
  for (const [field,value] of [['steps',p.steps],['cfg',p.cfg],['seed',p.seed],['pag.scale',p.pag?.scale],...(p.clip_skip == null ? [] : [['clip_skip',p.clip_skip]])]) {
    const error = xyzAxisValueIssue(field,value,draft.engine);
    if (error) issues.push(`基线 ${field}：${error}`);
  }
  if (issues.length || !wire) return issues;
  if (wire.sample_seeds !== null && wire.sample_seeds !== undefined) {
    if (!Array.isArray(wire.sample_seeds) || wire.sample_seeds.length !== draft.repeat_count) issues.push('样本种子列表必须与重复数一致');
    else wire.sample_seeds.forEach((s,i)=>{const error=xyzAxisValueIssue('seed',s,draft.engine);if(error)issues.push(`样本 ${i+1}：${error}`);});
  }
  // Check effective weights, not unused declared background. Full/no_loras may explicitly clear it.
  let sets = [wire.baseline.loras];
  for (const key of ['x','y','z']) {
    const axis = wire.axes[key];
    if (!axis?.field.startsWith('lora_')) continue;
    const values = axis.range ? expanded[key] : axis.values.map(e=>e.value);
    const next = [];
    for (const entries of sets) for (const value of values) {
      try {
        let result = entries;
        if (axis.field === 'lora_profile') result = profile(entries,value);
        else if (axis.field === 'lora_multiplier') {
          result = profile(entries,axis.snapshot);
          const factor = bounded(value,-5,5);
          result.forEach(m=>{
            const weight=bounded(weightOf(m),-5,5,{weight:true});
            const n=weight.n*factor.n, scale=weight.scale*factor.scale;
            if (n < -5n*scale || n > 5n*scale || n*10000n%scale) throw new Error('倍率结果权重须为 −5–5 且最多 4 位有效小数；不会舍入');
          });
        } else if (axis.field === 'lora_model') result = upsert(entries,value,axis.target);
        else if (axis.field === 'lora_weight') {
          const target=entries.find(m=>m.instanceId===axis.target);
          if (!target) throw new Error('LoRA 权重目标尚未定义或不在冻结背景中');
          result = upsert(entries,{...target,weight:value,enabled:true},axis.target);
        }
        result.forEach(memberIssue);
        next.push(result);
      } catch(error) { issues.push(`${key.toUpperCase()} LoRA：${error.message}`); }
    }
    sets = next;
  }
  if (!issues.length) for (const entries of sets) try { entries.forEach(memberIssue); } catch(error) {issues.push(`LoRA：${error.message}`);}
  return [...new Set(issues)];
}
