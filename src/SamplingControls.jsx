import { RefreshCw } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";

export function GuidanceControls({ value, options, detail, onChange, disabled = false, invalid = false, describedBy, prefix = '', pag, onPag, pagDisabled, Slider, NumberInput }) {
 return <>
  <label className="guidance-select"><span><b>引导增强（Guidance）</b><small>{detail}</small></span><WorkspaceSelect ariaLabel={`${prefix}引导增强`} value={value} disabled={disabled} ariaInvalid={invalid} ariaDescribedBy={describedBy} onChange={onChange} options={options} /></label>
  {value === 'pag' && <div className="guidance-settings">
   <Slider label="PAG 强度" value={pag.scale} min={0} max={5} step={0.1} inputStep={0.01} fixed={2} disabled={pagDisabled} NumberInput={NumberInput} onChange={(scale) => onPag({...pag,scale})} />
   <label><span>PAG 作用层</span><WorkspaceSelect ariaLabel={`${prefix}PAG 作用层`} value={pag.appliedLayers} disabled={pagDisabled} onChange={(appliedLayers)=>onPag({...pag,appliedLayers})} options={[{value:'mid',label:'Mid（推荐）'},{value:'all',label:'全部自注意力层（高风险）'}]} /></label>
   <small>建议从 0.3 开始；全部层会显著放大对比、描边和色彩，动漫模型尤其容易失真。</small>
  </div>}
 </>;
}

// Presentation only: the ordinary and XYZ owners keep their own callbacks and seed semantics.
export function SamplerPair({ sampler, scheduler, samplers, schedulers, onSampler, onScheduler, prefix = "", preserveUnavailable = false }) {
  const options = (value, names) => [...(preserveUnavailable && value && !names.includes(value) ? [{ value, label: `${value} · 保存值（当前不可用）`, disabled: true }] : []), ...names.map((name) => ({ value: name, label: name }))];
  return <div className="sampler-row">
    <label>采样器<WorkspaceSelect ariaLabel={`${prefix}采样器`} value={sampler} onChange={onSampler} options={options(sampler, samplers)} /></label>
    <label>调度器<WorkspaceSelect ariaLabel={`${prefix}调度器`} value={scheduler} onChange={onScheduler} options={options(scheduler, schedulers)} /></label>
  </div>;
}

export function SeedControls({ seed, mode, modes, onSeed, onBlur, onRandom, onMode, prefix = "", invalid = false, disabled = false }) {
  return <>
    <label className="seed-field"><span>随机种子</span><div><input aria-label={`${prefix}种子`} inputMode="numeric" maxLength="20" value={seed} aria-invalid={invalid || undefined} disabled={disabled} onChange={(event) => onSeed(event.target.value)} onBlur={onBlur} /><button type="button" title="生成随机种子" aria-label={`${prefix}重抽种子`} disabled={disabled} onClick={onRandom}><RefreshCw size={15} /></button></div></label>
    <div className="seed-modes" aria-label={`${prefix}种子生成模式`}>{modes.map((item) => <button type="button" key={item.id} className={mode === item.id ? "active" : ""} aria-pressed={mode === item.id} onClick={() => onMode(item.id)}>{item.label}</button>)}</div>
    <p className="seed-range">范围 0 ～ 18446744073709551615</p>
  </>;
}

// Same mother stepper, but invalid typed XYZ drafts are data, not values to clamp on blur.
export function XYZCountField({ label, detail, value, max, onChange, invalid }) {
  return <label className="count-field xyz-count-field" title={detail}>
    <span><strong>{label}</strong><small>{detail}</small></span>
    <span className="count-stepper"><button type="button" aria-label={`减少${label}`} disabled={!Number.isInteger(value) || value <= 1} onClick={() => onChange(value - 1)}>−</button>
      <input aria-label={`${label}数值`} type="text" inputMode="numeric" value={value ?? ""} aria-invalid={invalid || undefined} onChange={(event) => { const raw = event.target.value; onChange(/^\d+$/.test(raw) ? Number(raw) : raw); }} />
      <button type="button" aria-label={`增加${label}`} disabled={!Number.isInteger(value) || value >= max} onClick={() => onChange(value + 1)}>+</button></span>
  </label>;
}

export function XYZNumberInput({ value, onCommit, onChange, ariaLabel, min, max, integer, step, disabled, className }) {
  const invalid = value === "" || !Number.isFinite(Number(value)) || (integer && !Number.isInteger(Number(value))) || Number(value) < min || Number(value) > max;
  return <input type="text" className={className} inputMode={integer ? "numeric" : "decimal"} aria-label={ariaLabel} aria-invalid={invalid || undefined} disabled={disabled} value={value ?? ""} onChange={(event) => {
    const raw = event.target.value, valid = /^-?\d+(\.\d+)?$/.test(raw);
    (onCommit || onChange)(valid ? Number(raw) : raw);
  }} />;
}
