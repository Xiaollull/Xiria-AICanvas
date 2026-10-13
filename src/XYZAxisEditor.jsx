import { useEffect, useId, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Plus, Trash2, X } from "lucide-react";
import { createPortal } from "react-dom";
import { useDialogLifecycle } from "./gallery-core.js";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import { groupXYZProfile, xyzAxisOptions, decimalRangeValues } from "./xyz-draft.js";
import { xyzExpectedValues } from './xyz-editor-source.js';
import XYZAxisLoraPicker from './XYZAxisLoraPicker.jsx';
import { xyzAxisValueIssue } from './xyz-local-validation.js';
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";
import { initialXYZAxisValue, initialXYZRange, parseXYZStructuredText, validXYZFiniteDecimal, validXYZSeed, xyzComponentFields, xyzStructuredText, xyzGuidanceOption, xyzGuidanceRuntimeStatus, xyzGuidanceStatusReason } from "./xyz-axis-editor.js";

const NUMERIC_FIELDS = new Set(["steps", "cfg", "clip_skip", "pag.scale", "lora_weight", "lora_multiplier"]);
// Axis options retired from the picker on 2026-10-12 (user request). Saved drafts that still carry one of
// these fields keep their raw definition, local preflight and wire meaning; only new axes cannot select them.
const RETIRED_AXIS_FIELDS = new Set(["lora_model", "lora_profile", "lora_multiplier"]);
const GUIDANCE_LABELS = { none: "无（None）", pag: "PAG", cfg_zero_star: "CFG-Zero*" };
const PROFILE_MODES = [
  { value: "keep_background", label: "保留背景，替换测试集合" },
  { value: "full", label: "方案作为完整列表" },
  { value: "no_test", label: "无测试 LoRA（保留独立背景）" },
  { value: "no_loras", label: "完全无 LoRA" },
];

function identityChoices(draft) {
  const byId = new Map();
  for (const group of draft.baseline.group_snapshots || []) for (const item of group.members || []) {
    if (item.instanceId && !byId.has(item.instanceId)) byId.set(item.instanceId, item);
  }
  for (const item of draft.baseline.loras || []) if (item.instanceId && !byId.has(item.instanceId)) byId.set(item.instanceId, item);
  return [...byId.values()];
}

function identityLabel(item) { return item?.name || item?.value || "未命名 LoRA"; }

function MemberList({ value, onChange, draft, showTestTargets = true, loraManager }) {
  const choices = useMemo(() => identityChoices(draft), [draft.baseline.group_snapshots, draft.baseline.loras]);
  const members = Array.isArray(value?.members) ? value.members : [];
  const updateMember = (index, patch) => onChange({ ...value, members: members.map((member, i) => i === index ? { ...member, ...patch } : member) });
  const addMember = (instanceId) => {
    const found = choices.find((item) => item.instanceId === instanceId);
    if (found && !members.some((item) => item.instanceId === instanceId)) onChange({ ...value, members: [...members, { ...structuredClone(found), weight: String(found.weight ?? "1"), enabled: true }] });
  };
  const missingMembers = members.filter((member) => !choices.some((item) => item.instanceId === member.instanceId));
  const selectedIds = new Set(members.map((member) => member.instanceId).filter(Boolean));
  const testTargets = Array.isArray(value?.test_targets) ? value.test_targets : [];
  const targetIds = [...new Set([...choices.map((item) => item.instanceId), ...testTargets.filter((id) => !choices.some((item) => item.instanceId === id))])];

  return <div className="xyz-profile-members">
    {(value?.mode === "full" || value?.mode === "keep_background") && <>
      <div className="xyz-profile-member-list">
        {members.map((member, index) => {
          const options = [
            ...(missingMembers.includes(member) && member.instanceId ? [{ value: member.instanceId, label: `${identityLabel(member)} · 身份未在冻结组中解析`, disabled: true }] : []),
            ...choices.filter((item) => item.instanceId === member.instanceId || !selectedIds.has(item.instanceId)).map((item) => ({ value: item.instanceId, label: identityLabel(item) })),
          ];
          const move = (delta) => {
            const next = [...members]; [next[index], next[index + delta]] = [next[index + delta], next[index]];
            onChange({ ...value, members: next });
          };
          return <div className="xyz-profile-member" key={`${member.instanceId || "unresolved"}-${index}`}>
            <XYZAxisLoraPicker label={`组合成员 ${index + 1}`} draft={draft} manager={loraManager} members={[member]} onSelect={([source])=>updateMember(index,source)} />
            <label>权重<input type="text" inputMode="decimal" value={Object.hasOwn(member,'weight') ? member.weight ?? '' : '1'} aria-invalid={Boolean(xyzAxisValueIssue('lora_weight',Object.hasOwn(member,'weight') ? member.weight : '1',draft.engine))} onChange={(event) => updateMember(index, { weight: event.target.value })} />{xyzAxisValueIssue('lora_weight',Object.hasOwn(member,'weight') ? member.weight : '1',draft.engine) && <small role="alert" className="xyz-value-message">权重无效</small>}</label>
            <div className="xyz-actions" aria-label={`组合成员 ${index + 1} 顺序`}>
              <button type="button" disabled={index === 0} aria-label={`成员 ${index + 1} 上移`} onClick={() => move(-1)}><ChevronUp size={13} /></button>
              <button type="button" disabled={index === members.length - 1} aria-label={`成员 ${index + 1} 下移`} onClick={() => move(1)}><ChevronDown size={13} /></button>
              <button type="button" aria-label={`移除成员 ${index + 1}`} onClick={() => onChange({ ...value, members: members.filter((_, i) => i !== index) })}><Trash2 size={13} /></button>
            </div>
          </div>;
        })}
      </div>
      <label>有序测试成员<XYZAxisLoraPicker label="管理轴组合成员" draft={draft} manager={loraManager} members={members} multiple onSelect={(members,preset_prompt)=>onChange({...value,members,...(preset_prompt?{preset_prompt}: {})})} /></label>
    </>}
    {showTestTargets && ["keep_background", "no_test"].includes(value?.mode) && <fieldset className="xyz-test-targets"><legend>测试集合（按冻结身份）</legend>
      <XYZAxisLoraPicker label="管理轴测试目标" draft={draft} manager={loraManager} members={testTargets.map(id=>[...members,...choices,...(loraManager?.catalog?.loras||[])].find(m=>m.instanceId===id)).filter(Boolean)} multiple onSelect={selected=>onChange({...value,test_targets:[...testTargets.filter(id=>![...members,...choices,...(loraManager?.catalog?.loras||[])].some(m=>m.instanceId===id)),...selected.map(m=>m.instanceId)]})} />
      {targetIds.length ? targetIds.map((id) => {
        const item = choices.find((candidate) => candidate.instanceId === id);
        return <label className="xyz-check" key={id}><input type="checkbox" checked={testTargets.includes(id)} aria-label={`测试目标 ${item ? identityLabel(item) : "未解析实例"}`} onChange={(event) => onChange({ ...value, test_targets: event.target.checked ? [...testTargets, id] : testTargets.filter((target) => target !== id) })} />{item ? identityLabel(item) : "冻结的未解析实例"}</label>;
      }) : <p>没有可选择的冻结实例；当前基线与组快照不会被改写。</p>}
    </fieldset>}
  </div>;
}

function JsonAdvanced({ label, value, onChange }) {
  const serialized = xyzStructuredText(value);
  const [text, setText] = useState(serialized);
  const [invalid, setInvalid] = useState(() => !parseXYZStructuredText(serialized).valid);
  const previousSerialized = useRef(serialized);
  useEffect(() => {
    if (previousSerialized.current === serialized) return;
    previousSerialized.current = serialized;
    if (text !== serialized) { setText(serialized); setInvalid(!parseXYZStructuredText(serialized).valid); }
  }, [serialized]);
  const commit = () => {
    const parsed = parseXYZStructuredText(text);
    setInvalid(!parsed.valid);
    onChange(parsed.value);
  };
  return <details className="xyz-raw-json"><summary>高级 JSON · {label}</summary>
    <textarea aria-label={`${label}高级 JSON`} rows={5} spellCheck="false" value={text} aria-invalid={invalid} onChange={(event) => setText(event.target.value)} onBlur={commit} />
    <small>{invalid ? "JSON 无效；原始文本已保留，服务端预检会拒绝它。" : "编辑后离开此栏时才解析；不会逐键格式化。"}</small>
  </details>;
}

function ProfileEditor({ value, onChange, draft, loraManager }) {
  const mode = value?.mode || "no_test";
  const previousValueState = useRef(new Map()), [pending, setPending] = useState(null);
  const selectMode = (nextMode) => {
    if (nextMode === mode) return;
    const next = previousValueState.current.get(nextMode) || { ...value, mode: nextMode,
      ...(nextMode === "no_test" ? { members: [] } : {}), ...(nextMode === "no_loras" ? { members: [], test_targets: [], includePresetText: false, preset_prompt: "" } : {}) };
    setPending({ next, message: `更改方案语义为「${nextMode}」；无测试与完全无 LoRA 不同，成员 / 预设词可能被移出本方案。` });
  };
  return <div className="xyz-profile-editor">
    <label>LoRA 方案<WorkspaceSelect ariaLabel="LoRA 方案" value={mode} options={PROFILE_MODES} onChange={selectMode} /></label>
    <MemberList value={{ ...value, mode }} onChange={onChange} draft={draft} loraManager={loraManager} />
    <label className="xyz-check"><input type="checkbox" checked={value?.includePresetText === true} onChange={(event) => onChange({ ...value, includePresetText: event.target.checked })} />同时使用组合预设词（仅本方案前缀一次）</label>
    {value?.includePresetText && <label>组合预设词<textarea rows={2} value={value?.preset_prompt || ""} onChange={(event) => onChange({ ...value, preset_prompt: event.target.value })} /></label>}
    {pending && <AxisReplaceDialog axisKey="LoRA 方案" pending={pending} onClose={() => setPending(null)} onConfirm={() => { previousValueState.current.set(mode, structuredClone(value)); onChange(structuredClone(pending.next)); setPending(null); }} />}
  </div>;
}

function AxisReplaceDialog({ axisKey, pending, onConfirm, onClose }) {
  const ref = useDialogLifecycle(true, onClose);
  const titleId=useId();
  return createPortal(<div className="xyz-sync-backdrop xyz-axis-confirm-backdrop" onMouseDown={event=>event.target===event.currentTarget&&onClose()}><section ref={ref} tabIndex={-1} className="xyz-sync-dialog gallery-small-dialog xyz-axis-confirm" role="alertdialog" aria-modal="true" aria-labelledby={titleId}>
    <header><div><h2 id={titleId}>更改 {axisKey.toUpperCase()} 配置？</h2></div><button type="button" aria-label="关闭轴确认" onClick={onClose}><X size={18} /></button></header>
    <div className="gallery-dialog-body"><p>{pending.message}</p><small>取消会完整保留当前定义。字段与模式切换的旧定义保留在本次会话中；不会写入服务端请求。</small></div>
    <footer><button type="button" onClick={onClose}>保留当前配置</button><button type="button" className="primary" onClick={onConfirm}>确认更改</button></footer>
  </section></div>, document.body);
}

function AxisValue({ axisKey, field, entry, index, last, draft, componentChoices, runtimeGuidance, onValue, onLabel, onMove, onRemove, loraManager }) {
  const [expanded, setExpanded] = useState(false);
  const value = entry.value;
  const isObjectValue = value && typeof value === "object" && !Array.isArray(value);

  const modelField = ["checkpoint", "diffusion_model", "text_encoder", "text_encoder_2", "vae"].includes(field);
  const choices = modelField ? componentChoices[field] || [] : field === "sampler" ? SAMPLER_NAMES.map((item) => ({ value: item, label: item }))
      : field === "scheduler" ? SCHEDULER_NAMES.map((item) => ({ value: item, label: item }))
        : field === "guidance" ? (draft.engine && (draft.engine === "SD" || draft.engine === "iL" ? ["none", "pag"] : draft.engine === "Anima" ? ["none", "pag", "cfg_zero_star"] : draft.engine === "Krea2" ? ["none", "cfg_zero_star"] : ["none"]))
          .map((item) => xyzGuidanceOption(runtimeGuidance, draft.engine, item, GUIDANCE_LABELS[item] || item)) : [];
  const validChoice = choices.find((item) => item.value === value);
  const renderSelect = choices.length > 0 || modelField;
  const guidanceStatus = field === "guidance" && value !== "none" ? xyzGuidanceRuntimeStatus(runtimeGuidance, draft.engine, value) : null;
  const guidanceIssue = guidanceStatus && guidanceStatus.status !== "available" ? xyzGuidanceStatusReason(guidanceStatus) : "";
   const valueIssue = xyzAxisValueIssue(field,value,draft.engine);
   const invalidSeed = field === "seed" && Boolean(valueIssue);
   const invalidDecimal = NUMERIC_FIELDS.has(field) && Boolean(valueIssue);
  const componentFields = field === "components" ? xyzComponentFields(draft.engine) : [];

  const updateProfile = (patch) => onValue({ ...value, ...patch });
  const simple = NUMERIC_FIELDS.has(field) && field !== "lora_multiplier" || ["seed", "sampler", "scheduler", "guidance"].includes(field) || modelField;
  return <div className={`xyz-value${simple ? " xyz-value-simple" : " xyz-value-complex"}`} data-expanded={expanded} data-axis-value={`${axisKey}:${index}`}>
    <strong className="xyz-value-index" title={entry.label || undefined}>{String(index + 1).padStart(2, "0")}</strong>
    <div className="xyz-value-body">
    {NUMERIC_FIELDS.has(field) && <label>{field === "steps" ? "步数" : field === "cfg" ? "CFG / 蒸馏引导" : field === "clip_skip" ? "CLIP skip" : field === "pag.scale" ? "PAG 强度" : field === "lora_weight" ? "目标权重" : "组合倍率"}
      <input type="text" inputMode="decimal" aria-label={`${axisKey.toUpperCase()} 第 ${index + 1} 项`} value={value ?? ""} aria-invalid={invalidDecimal} placeholder="待定义" onChange={(event) => onValue(event.target.value)} />
    </label>}
    {field === "seed" && <label>种子 · uint64 字符串<input type="text" inputMode="numeric" aria-label={`${axisKey.toUpperCase()} 第 ${index + 1} 项`} value={String(value ?? "")} aria-invalid={invalidSeed} onChange={(event) => onValue(event.target.value)} />
    </label>}
    {renderSelect && <label>{modelField ? "目录资产" : field === "sampler" ? "采样器" : field === "scheduler" ? "调度器" : "引导类型"}
      <WorkspaceSelect ariaLabel={`${axisKey.toUpperCase()} ${field}`} value={typeof value === "string" ? value : ""} options={[{value:'',label:'待定义 · 请选择',disabled:true},...(!validChoice && value ? [{ value, label: `${String(value)} · 当前值不在有效选项中`, disabled: true, disabledReason: field === "guidance" ? "未知引导模式；已保存值保留，预检不放行" : undefined }] : []), ...choices]} onChange={onValue} />
      {guidanceIssue && <small role="alert">保存的「{GUIDANCE_LABELS[value] || value}」当前不可用：{guidanceIssue}；原值保留，预检已阻止。</small>}
    </label>}
    {["prompt", "negative_prompt"].includes(field) && <label>{field === "prompt" ? "正向文本 · 按原文提交，不按逗号拆分" : "负向文本"}<textarea aria-label={`${axisKey.toUpperCase()} 第 ${index + 1} 项`} rows={4} placeholder="待定义" value={value === null ? '' : typeof value === "string" ? value : xyzStructuredText(value)} onChange={(event) => onValue(event.target.value)} /></label>}
    {["prompt_replace", "negative_replace"].includes(field) && (isObjectValue || value === null) && <div className="xyz-replace-fields">
      <label>查找原文<textarea rows={2} value={value?.find ?? ""} onChange={(event) => updateProfile({ find: event.target.value })} /></label>
      <label>替换为<textarea rows={2} value={value?.replace ?? ""} onChange={(event) => updateProfile({ replace: event.target.value })} /></label>
      <label>预期匹配次数<input type="number" min="1" step="1" value={value?.expected_matches ?? ""} aria-invalid={!Number.isInteger(value?.expected_matches) || value.expected_matches < 1} onChange={(event) => updateProfile({ expected_matches: event.target.value === "" ? "" : Number(event.target.value) })} /></label>
    </div>}
    {field === "components" && (isObjectValue || value === null) && <div className="xyz-component-fields"><p>只编辑当前引擎的模型组件字段；资源来自现有模型目录，后端仍会校验兼容性。</p>
      {componentFields.map((component) => {
        const rawOptions = componentChoices?.[component] || [];
        const options = rawOptions.map((item) => ({ value: item.value, label: item.label || item.name || item.value }));
        const current = value?.[component] ?? "";
        return <label key={component}>{component}
          {options.length ? <WorkspaceSelect ariaLabel={`${axisKey.toUpperCase()} ${component}`} value={current} options={[{ value: "", label: "未设置" }, ...(!options.some((item) => item.value === current) && current ? [{ value: current, label: `${current} · 当前选择不在目录中` }] : []), ...options]} onChange={(next) => onValue({ ...value, engine: draft.engine, [component]: next })} />
            : <input aria-label={`${axisKey.toUpperCase()} ${component}`} value={current} placeholder="输入冻结资产值" onChange={(event) => onValue({ ...value, engine: draft.engine, [component]: event.target.value })} />}
        </label>;
      })}
    </div>}
    {field === "lora_model" && (isObjectValue || value === null) && <label>候选 LoRA · 独立于全局挂载<XYZAxisLoraPicker label={`${axisKey.toUpperCase()} 第 ${index+1} 项候选 LoRA`} draft={draft} manager={loraManager} members={isObjectValue?[value]:[]} onSelect={([item])=>onValue(item)} /></label>}
    {field === "prompt_replace" || field === "negative_replace" || field === "components" || field === "lora_model" ? !isObjectValue && value !== null && <p role="alert" className="xyz-unsupported-note">此项是未解析原始文本；不会按空白字段替换。请在下方高级 JSON 修复，或明确替换 / 删除此项。</p> : null}
    {field === "lora_profile" && (isObjectValue || value === null) && <ProfileEditor value={value || {mode:'keep_background',members:[],test_targets:[],includePresetText:false,preset_prompt:''}} onChange={onValue} draft={draft} loraManager={loraManager} />}
    {field === "lora_profile" && typeof value === "string" && <p role="alert" className="xyz-unsupported-note">此项保存的是未解析原始文本；内容原样保留。请在下方高级 JSON 修复，或删除该项后重新添加。</p>}
    {field === "lora_multiplier" && <div className="xyz-profile-editor"><p>倍率作用于此处冻结的有序成员；不改真实 LoRA 组。</p>{entry.snapshot && typeof entry.snapshot === "object" && !Array.isArray(entry.snapshot) ? <ProfileEditor value={entry.snapshot} onChange={(snapshot) => {
      // Existing planner contract: multiplier members live in snapshot, not in the axis value.
      entry.onSnapshot?.(snapshot);
    }} draft={draft} loraManager={loraManager} /> : <p role="alert" className="xyz-unsupported-note">倍率快照含未解析原始文本；请在下方高级 JSON 修复。</p>}<JsonAdvanced label="倍率快照（原始 planner 字段）" value={entry.snapshot} onChange={entry.onSnapshot} /></div>}
    {["components", "prompt_replace", "negative_replace", "lora_model", "lora_profile"].includes(field) && <JsonAdvanced label={`${field} · 第 ${index + 1} 项`} value={value} onChange={onValue} />}
    {modelField && <JsonAdvanced label={`${field} · 第 ${index + 1} 项`} value={value} onChange={onValue} />}
    {!modelField && !NUMERIC_FIELDS.has(field) && !["seed", "sampler", "scheduler", "guidance", "prompt", "negative_prompt", "prompt_replace", "negative_replace", "components", "lora_model", "lora_profile", "lora_multiplier"].includes(field) && <JsonAdvanced label={`${field} · 第 ${index + 1} 项`} value={value} onChange={onValue} />}
    {valueIssue && <small className="xyz-value-message" role={valueIssue==='待定义'?'status':'alert'}>{valueIssue}</small>}
    </div>
    <div className="xyz-value-actions xyz-actions">
      <button type="button" aria-label={`${axisKey.toUpperCase()} 第 ${index + 1} 项操作`} aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>更多</button>
      <button type="button" disabled={index === 0} aria-label={`${axisKey.toUpperCase()} 第 ${index + 1} 项上移`} onClick={() => onMove(-1)}><ChevronUp size={14} /></button>
      <button type="button" disabled={last || index >= 255} aria-label={`${axisKey.toUpperCase()} 第 ${index + 1} 项下移`} onClick={() => onMove(1)}><ChevronDown size={14} /></button>
      <button type="button" aria-label={`${axisKey.toUpperCase()} 删除第 ${index + 1} 项`} onClick={onRemove}><Trash2 size={14} /></button>
    </div>
    {expanded && <label className="xyz-value-label">显示标签<input aria-label={`${axisKey.toUpperCase()} 第 ${index + 1} 项标签`} value={entry.label || ""} onChange={(event) => onLabel(event.target.value)} /></label>}
  </div>;
}

export default function XYZAxisEditor({ axisKey, draft, onChange, componentChoices = {}, runtimeGuidance, previousStates, loraManager }) {
  const previousValueState = useRef(previousStates || new Map());
  const [pending, setPending] = useState(null);
  const [collapsed,setCollapsed]=useState(false),[expected,setExpected]=useState(''),[quantityError,setQuantityError]=useState('');
  const axis = draft.axes?.[axisKey] || null;
  const options = xyzAxisOptions(draft.engine);
  const staticSupported = options.some((option) => option.value === axis?.field);
  const pagStatus = xyzGuidanceRuntimeStatus(runtimeGuidance, draft.engine, "pag");
  const pagIssue = axis?.field === "pag.scale" && pagStatus.status !== "available" ? xyzGuidanceStatusReason(pagStatus) : "";
  const supported = staticSupported && !pagIssue;
  const fieldOptions = options.map((option) => option.value !== "pag.scale" || pagStatus.status === "available" ? option : {
    ...option,
    label: `${option.label} · ${pagStatus.status === "unknown" ? "能力未知" : "当前不可用"}`,
    disabled: true,
    disabledReason: `${pagIssue}；已保存轴保留，预检会阻止`,
  });
  const update = (patch) => onChange({ ...axis, ...patch });
  const valueAxisChange = (nextField) => {
    if (nextField === (axis?.field || "")) return;
    const next = nextField ? previousValueState.current.get(nextField) || { field: nextField, values: [{ value: null, label: "" }], range: null,
      ...(nextField === 'lora_model' ? {target:null,editor_source:{schema_version:1,operation:'add'}} : {}),
      ...(nextField === 'lora_profile' ? {editor_source:{schema_version:1,operation:'add'}} : {}),
      ...(nextField === "lora_multiplier" ? { snapshot: { mode: "keep_background", members: [], test_targets: [], includePresetText: false, preset_prompt: "" },editor_source:{schema_version:1,member_only:true} } : {}) } : null;
    if (axis) setPending({ next, message: `将「${axis.field}」替换为「${nextField || "不使用"}」，现有项 / 范围 / 标签不再参与本轴。` });
    else onChange(structuredClone(next));
  };
  const replacementDialog = pending && <AxisReplaceDialog axisKey={axisKey} pending={pending} onClose={() => setPending(null)} onConfirm={() => {
    previousValueState.current.set(axis.field, structuredClone(axis));
    onChange(structuredClone(pending.next)); setPending(null);
  }} />;
  const selectField = <WorkspaceSelect ariaLabel={`${axisKey.toUpperCase()} 轴参数`} value={axis?.field || ""} options={[
    { value: "", label: "不使用 · 1 项" }, ...fieldOptions,
    ...(axis && !staticSupported ? [{ value: axis.field, label: `${axis.field} · ${RETIRED_AXIS_FIELDS.has(axis.field) ? "已停用" : "当前引擎不支持"}（原样保留）`, disabled: true }] : []),
  ]} onChange={valueAxisChange} />;

  if (!axis) return <section className="xyz-axis-card xyz-axis-unused xyz-axis-direct"><header><h3><b>{axisKey.toUpperCase()}</b><small>{axisKey === "x" ? "列" : axisKey === "y" ? "行" : "子网格"}</small></h3>{selectField}<span className="xyz-axis-count">1 项 · 未使用</span></header></section>;

  const numeric = NUMERIC_FIELDS.has(axis.field);
  const values = Array.isArray(axis.values) ? axis.values : [];
  const replaceValue = (index, value) => update({ values: values.map((entry, i) => i === index ? { ...entry, value } : entry) });
  const move = (index, delta) => {
    if (index + delta < 0 || index + delta >= values.length) return;
    const next = [...values]; [next[index], next[index + delta]] = [next[index + delta], next[index]]; update({ values: next });
  };
  const initialRange = initialXYZRange(axis.field, draft);
  let rangeValues=[],rangeError='';try{if(axis.range){rangeValues=decimalRangeValues(axis.range);const issue=rangeValues.map(v=>xyzAxisValueIssue(axis.field,v,draft.engine)).find(Boolean);if(issue)throw new Error(issue);}}catch(e){rangeError=e.message;}
  const setQuantity=()=>{try{const next=xyzExpectedValues(values,expected);setQuantityError('');if(next.length<values.length)setPending({next:{...axis,values:next},message:`减少至 ${next.length} 项将移除末尾 ${values.length-next.length} 项。确认后才生效；取消不改任何数据。`});else update({values:next});}catch(e){setQuantityError(e.message);}};
  return <section className={`xyz-axis-card${supported ? "" : " xyz-axis-unsupported"}`}>
    <header><h3><b>{axisKey.toUpperCase()}</b><small>{axisKey === "x" ? "列" : axisKey === "y" ? "行" : "子网格"}</small></h3>{selectField}<button type="button" className="xyz-axis-count" aria-label={`${axisKey.toUpperCase()} ${collapsed?'展开':'收起'}轴项`} aria-expanded={!collapsed} onClick={()=>setCollapsed(!collapsed)}>{axis.range ? `${rangeValues.length || '?'} 项 · 阶梯` : `${values.length} 个有序项`} {collapsed?<ChevronDown size={14}/>:<ChevronUp size={14}/>}</button></header>
     {!supported && <p role={pagIssue ? "alert" : "status"} className="xyz-unsupported-note">{pagIssue ? `已保存的 PAG 强度轴在当前运行时不可用：${pagIssue}；原始定义与数值仍保留。预检已阻止。` : RETIRED_AXIS_FIELDS.has(axis.field) ? `已保存的「${axis.field}」轴参数已停用；原始定义与数值仍保留。选择其他轴参数或明确选“不使用”才会替换。` : `当前引擎不支持已保存的「${axis.field}」轴；原始定义仍保留。选择其他字段或明确选“不使用”才会替换。`}</p>}
    {supported && !collapsed && <>
      {numeric && <div className="xyz-axis-modes" role="group" aria-label={`${axisKey.toUpperCase()} 数值模式`}>{[[false,'自定义数值'],[true,'阶梯数值']].map(([range,label])=><button type="button" key={label} aria-pressed={Boolean(axis.range)===range} onClick={()=>{
        if(Boolean(axis.range)===range)return;
        const key = `${axis.field}:${range ? "range" : "list"}`;
        const restored = previousValueState.current.get(key);
        setPending({ next: restored || { ...axis, range: range ? initialRange : null, values: range ? [] : [{ value: null, label: "" }] }, message: "切换自定义 / 阶梯模式会替换当前输入；确认后可选回原模式恢复。" });
        previousValueState.current.set(`${axis.field}:${axis.range ? "range" : "list"}`, structuredClone(axis));
      }}>{label}</button>)}</div>}
      {axis.range && <div className="xyz-range">{[["start", "起点"], ["end", "终点"], ["step", "步长"]].map(([key, label]) => {
        const issue = !validXYZFiniteDecimal(axis.range[key]) ? '有限十进制数' : key !== 'step' ? xyzAxisValueIssue(axis.field,axis.range[key],draft.engine) : '';
        return <label key={key}>{label}<input aria-label={`${axisKey.toUpperCase()} 阶梯${label}`} type="text" inputMode="decimal" value={axis.range[key] ?? ""} aria-invalid={Boolean(issue)} onChange={(event) => update({ range: { ...axis.range, [key]: event.target.value } })} />{issue && <small role="alert">{issue}</small>}</label>;
      })}</div>}
      {axis.range && <div className="xyz-range-preview" role="status">{rangeError ? <span role="alert">{rangeError} · 原始输入保留，预检已阻止</span> : <><strong>自动生成 {rangeValues.length} 项</strong><output aria-label={`${axisKey.toUpperCase()} 阶梯预览`}>{rangeValues.join('，')}</output></>}</div>}
      {!axis.range && <div className="xyz-axis-quantity"><label>预期数量<input type="text" inputMode="numeric" aria-label={`${axisKey.toUpperCase()} 预期数量`} value={expected} placeholder={String(values.length)} onChange={e=>setExpected(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'){e.preventDefault();setQuantity();}}} aria-invalid={Boolean(quantityError)} /></label><button type="button" onClick={setQuantity}>确认数量</button>{quantityError && <small role="alert">{quantityError}</small>}</div>}
      {!axis.range && <div className="xyz-values">
        {values.map((entry, index) => <AxisValue key={`${axisKey}:${axis.field}:${index}`} axisKey={axisKey} field={axis.field} entry={{ ...entry,
          snapshot: axis.field === "lora_multiplier" ? entry.snapshot ?? axis.snapshot : entry.snapshot,
          onSnapshot: (snapshot) => update({ snapshot }),
        }} index={index} last={index === values.length - 1} draft={draft} componentChoices={componentChoices} runtimeGuidance={runtimeGuidance} loraManager={loraManager}
          onValue={(value) => replaceValue(index, value)} onLabel={(label) => update({ values: values.map((item, i) => i === index ? { ...item, label } : item) })}
          onMove={(delta) => move(index, delta)} onRemove={() => update({ values: values.filter((_, i) => i !== index) })} />)}
        <button type="button" disabled={values.length >= 256} onClick={() => update({ values: [...values, { value: null, label: "" }] })}><Plus size={14} />添加独立项</button>
      </div>}
      {axis.field === 'lora_multiplier' && axis.range && (axis.snapshot && typeof axis.snapshot==='object' ? <ProfileEditor value={axis.snapshot} onChange={snapshot=>update({snapshot})} draft={draft} loraManager={loraManager} /> : <JsonAdvanced label="倍率快照" value={axis.snapshot} onChange={snapshot=>update({snapshot})} />)}
      {axis.field === 'lora_weight' && <label>权重目标 · 未挂载目标仅加选到每格<XYZAxisLoraPicker label={`${axisKey.toUpperCase()} LoRA 目标实例`} axisKey={axisKey} draft={draft} manager={loraManager} members={[axis.editor_source?.target_reference || draft.baseline.loras.find(m=>m.instanceId===axis.target)].filter(Boolean)} onSelect={([item])=>update({target:item.instanceId,editor_source:{schema_version:1,target_reference:item}})} /></label>}
      {axis.field === 'lora_model' && <><label>候选操作<WorkspaceSelect ariaLabel={`${axisKey.toUpperCase()} LoRA 候选操作`} value={axis.editor_source?.operation || 'replace'} options={[{value:'add',label:'保留全局背景 · 加选候选'},{value:'replace',label:'显式替换指定实例'}]} onChange={operation=>setPending({next:{...axis,editor_source:{schema_version:1,operation}},message:'更改候选操作会改变每格 LoRA 集合；全局挂载本身不会改变。'})} /></label>{axis.editor_source?.operation!=='add' && <label>替换目标 · 必须是背景已有实例<XYZAxisLoraPicker label={`${axisKey.toUpperCase()} LoRA 替换目标`} draft={draft} manager={loraManager} members={draft.baseline.loras.filter(m=>m.instanceId===axis.target)} onSelect={([item])=>update({target:item.instanceId})} /></label>}</>}
      {axis.field === "lora_profile" && !axis.range && <div className="xyz-profile-presets">
        <p>从冻结组创建方案；默认不跟随预设词。空测试集合与完整清空 LoRA 是两种操作。</p>
        {(draft.baseline.group_snapshots || []).map((group) => <button type="button" key={group.id} onClick={() => update({ values: [...values, { value: groupXYZProfile(group), label: group.name }] })}>添加冻结组：{group.name}</button>)}
        <button type="button" onClick={() => update({ values: [...values, { value: { mode: "no_test", members: [], test_targets: [], includePresetText: false, preset_prompt: "" }, label: "无测试 LoRA · 保留独立背景" }] })}>添加无测试 LoRA</button>
        <button type="button" onClick={() => update({ values: [...values, { value: { mode: "no_loras", members: [], test_targets: [], includePresetText: false, preset_prompt: "" }, label: "完全无 LoRA" }] })}>添加完全无 LoRA</button>
      </div>}
      {axis.field.startsWith("lora_") && <p className="xyz-axis-hint">身份来自冻结 P10 实例；未解析身份不会按同名文件替换。0 权重表示不启用目标。</p>}
    </>}
    {replacementDialog}
  </section>;
}
