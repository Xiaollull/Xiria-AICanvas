import { Layers3 } from "lucide-react";
import ModelAssetPicker from "./ModelAssetPicker.jsx";
import MountedLoraSummary from "./MountedLoraSummary.jsx";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import { SamplerPair, SeedControls, XYZNumberInput, GuidanceControls } from "./SamplingControls.jsx";
import { XYZ_SEED_MODES } from './xyz-editor-source.js';
import XYZPostprocessPanel from "./XYZPostprocessPanel.jsx";
import { XYZ_CAPABILITIES } from "./xyz-draft.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";
import { patchXYZParameters, xyzSidebarIssues } from "./xyz-sidebar-context.js";
import { validXYZSeed } from "./xyz-axis-editor.js";

const modelLabels = { checkpoint: "底模选择", diffusion_model: "扩散模型", text_encoder: "文本编码器", text_encoder_2: "文本编码器 · T5-XXL", vae: "VAE" };
function xyzModelLabel(engine, field) {
  if (field === "text_encoder" && engine === "Flux") return "文本编码器 · CLIP-L";
  if (field === "text_encoder" && ["Flux2", "Krea2", "MOSS"].includes(engine)) return "文本编码器 · 大语言模型";
  return modelLabels[field] || field;
}
function SavedSelect({ label, value, options, onChange, disabled = false }) {
  return <label>{label}<WorkspaceSelect ariaLabel={`XYZ ${label}`} value={value ?? ""} disabled={disabled} onChange={onChange} options={[
    { value: "", label: "请选择" }, ...(!options.some((item) => item.value === value) && value ? [{ value, label: `${value} · 保存值（当前不可用）`, disabled: true }] : []), ...options,
  ]} /></label>;
}

// `owner` names whose independent draft this rail edits (XYZ by default; the LoRA mix search reuses the same rail).
export default function XYZBaselineSidebar({ draft, onChange, onOpenManager, presentationFor, catalog, runtimeGuidance = ["none"], Slider, randomSeed, postprocessResources, error, owner = { title: "XYZ 独立基线", detail: "这里的修改只用于下次 XYZ，不改变基础生图 / 图生图。" } }) {
  if (error || !draft) return <p className="xyz-sidebar-error" role={error ? "alert" : "status"}>{error || "等待 XYZ 独立基线…"}</p>;
  const p = draft.baseline.parameters, cap = XYZ_CAPABILITIES.engines[draft.engine];
  const update = (patch) => onChange(patchXYZParameters(draft, patch));
  const modelFields = cap.model_fields.filter((field) => field !== "checkpoint").map((field) => {
    const label = xyzModelLabel(draft.engine, field), value = p[field] ?? "", items = catalog?.choices?.[field] || [];
    const verified = !catalog?.loading && !catalog?.error;
    return { kind: field, label, ariaLabel: `XYZ ${label}`, value, items, directory: catalog?.directories?.[field] || "", preserveUnavailable: true,
      missing: verified && Boolean(value) && !items.some((item) => item.value === value),
      missingLabel: `${value} · 保存值（当前不可用）`, missingMessage: `${label}文件当前不可用；已保留保存值`,
      onChange: (nextValue) => update({ [field]: nextValue }) };
  });
  const singleItems = catalog?.choices?.checkpoint || [], singleValue = p.checkpoint ?? "";
  const singleVerified = !catalog?.loading && !catalog?.error;
  const issues = xyzSidebarIssues(draft);
  return <div className="xyz-sidebar-fields">
    <p className="xyz-sidebar-owner"><b>{owner.title}</b><span>{owner.detail}</span></p>
    {catalog?.error && <p role="alert" className="xyz-sidebar-error">{catalog.error} · 已保存选择不变。</p>}
    <section className="xyz-sidebar-models" aria-label="XYZ 模型组件">
      {modelFields.length ? <ModelAssetPicker loading={catalog?.loading} refreshing={catalog?.refreshing} error={!catalog?.error && modelFields.some((field) => field.missing)} fields={modelFields}
        refreshTitle={`刷新 ${draft.engine} 模型组件`} refreshAriaLabel={`刷新 XYZ ${draft.engine} 模型组件`} onRefresh={catalog?.refresh} /> : <ModelAssetPicker loading={catalog?.loading} refreshing={catalog?.refreshing}
        error={!catalog?.error && Boolean(singleValue) && singleVerified && !singleItems.some((item) => item.value === singleValue)} single={{ ariaLabel: "XYZ 底模选择", value: singleValue, items: singleItems, directory: catalog?.directories?.checkpoint || "", preserveUnavailable: true,
          missing: singleVerified && Boolean(singleValue) && !singleItems.some((item) => item.value === singleValue), missingLabel: `${singleValue} · 保存值（当前不可用）`, missingMessage: "当前底模文件不可用；已保留保存值", onChange: (value) => update({ checkpoint: value }) }}
        refreshTitle={`刷新 ${draft.engine} 底模列表`} refreshAriaLabel={`刷新 XYZ ${draft.engine} 底模列表`} onRefresh={catalog?.refresh} />}
      {catalog?.loading && <small className="xyz-sidebar-catalog-status" role="status">正在读取 {draft.engine} 目录；不会自动选择第一项或清除已保存值。</small>}
    </section>
    <section className="xyz-sidebar-mounts" aria-label="XYZ 全局 LoRA">
      <div className="section-heading lora-title"><span>02</span><h2>全局 LoRA</h2><small className="lora-count">已挂载 {draft.baseline.loras.length}</small><button type="button" onClick={onOpenManager}><Layers3 size={14} />管理</button></div>
      <MountedLoraSummary loras={draft.baseline.loras} presentationFor={presentationFor || ((item) => ({ title: item.name || item.value, coverUrl: item.coverUrl || "", rowTitle: item.name || item.value }))} onOpenManager={onOpenManager} />
    </section>
    <div className="xyz-mother-controls"><GuidanceControls prefix="XYZ " value={p.guidance} detail={cap.negative_prompt ? ({none:'标准 CFG 采样',pag:'强度可调 · 默认仅作用于 Mid 层',cfg_zero_star:'优化缩放 · 前 4% 步零初始化'}[p.guidance]) : `${draft.engine} 蒸馏引导，无无条件分支`} options={cap.guidance.map(value=>({value,label:({none:'无（None）',pag:'PAG（扰动注意力引导）',cfg_zero_star:'CFG-Zero*（零星 CFG）'}[value]),disabled:!runtimeGuidance.includes(value)}))} disabled={!cap.negative_prompt} invalid={!runtimeGuidance.includes(p.guidance)} onChange={guidance=>update({guidance})} pag={{scale:p.pag?.scale ?? '',appliedLayers:p.pag?.applied_layers || 'mid'}} onPag={pag=>update({pag:{...(p.pag||{}),scale:pag.scale,applied_layers:pag.appliedLayers}})} pagDisabled={!runtimeGuidance.includes('pag')} Slider={Slider} NumberInput={XYZNumberInput} /></div>
    {!runtimeGuidance.includes(p.guidance) && <p className="guidance-unavailable" role="alert">保存的 {p.guidance} 当前不可用；原值保留，预检会拒绝。</p>}
    <section className="xyz-baseline-sampling xyz-mother-controls"><div className="section-heading"><span>03</span><h2>采样参数</h2></div>
      <div className="sampling-parameters">
      <SamplerPair prefix="XYZ " preserveUnavailable sampler={p.sampler} scheduler={p.scheduler} samplers={["SD", "iL"].includes(draft.engine) ? SAMPLER_NAMES : cap.samplers || XYZ_CAPABILITIES.samplers} schedulers={["SD", "iL"].includes(draft.engine) ? SCHEDULER_NAMES : cap.schedulers || XYZ_CAPABILITIES.schedulers} onSampler={(sampler) => update({ sampler })} onScheduler={(scheduler) => update({ scheduler })} />
      <div className="sliders">
        <Slider label="采样步数" value={p.steps} min={1} max={60} integer NumberInput={XYZNumberInput} onChange={(steps) => update({ steps })} />
        <Slider label={cap.negative_prompt ? "CFG 引导" : "蒸馏引导（Guidance）"} value={p.cfg} min={1} max={15} step={0.5} inputStep={0.1} NumberInput={XYZNumberInput} onChange={(cfg) => update({ cfg })} />
        {cap.clip_skip && <Slider label="CLIP skip" value={p.clip_skip ?? ""} min={cap.clip_skip[0]} max={cap.clip_skip[1]} integer NumberInput={XYZNumberInput} onChange={(clip_skip) => update({ clip_skip })} />}
      </div>
      <SeedControls prefix="XYZ 基线" seed={String(p.seed)} mode={draft.baseline.seed_source?.mode || draft.baseline.seed_mode} modes={XYZ_SEED_MODES} onSeed={(seed) => update({ seed })} onRandom={() => update({ seed: randomSeed() })} invalid={draft.baseline.seed_mode === "fixed" && !validXYZSeed(p.seed)} onMode={(mode) => onChange({ ...draft, baseline: { ...draft.baseline, seed_mode:mode === 'random'?'random':'fixed', seed_source:{schema_version:1,mode} } })} />
      <small className="xyz-seed-policy">每个重复样本的种子跨 XYZ 组合相同；种子轴优先。递增 / 递减按样本 ±1（uint64 环绕）；全随机仅服务端提交时抽取并冻结。</small>
      {!draft.baseline.seed_source && draft.baseline.seed_mode==='fixed' && <small className="xyz-seed-policy">旧草稿规则已保留：首样本固定，其余重复样本提交时随机。明确点击“固定”可改为所有重复样本同一种子。</small>}
      </div>
      {issues.length > 0 && <ul className="xyz-sidebar-error" role="alert">{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
    </section>
    <XYZPostprocessPanel draft={draft} onChange={onChange} Slider={Slider} randomSeed={randomSeed} resources={postprocessResources} />
  </div>;
}
