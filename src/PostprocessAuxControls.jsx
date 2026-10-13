import WorkspaceSelect from "./WorkspaceSelect.jsx";

// The actual ADetailer unit controls, shared by Basic and the independent XYZ chain owner.
export function ADetailerUnitControls({ unit, unitLocked, models, engineAllowsNegativePrompt, update, Slider, NumberInput, unitSteps }) {
  return <div className="adetailer-unit-body">
    <label className="adetailer-model">检测模型<WorkspaceSelect ariaLabel="ADetailer 检测模型" value={unit.detector} disabled={unitLocked} onChange={(detector) => update({ detector })} options={[
      ...(unit.detector && !models.some((item) => item.value === unit.detector) ? [{ value: unit.detector, label: `${unit.detector} · 保存值（当前不可用）`, disabled: true }] : []),
      ...(models.length ? models.map((item) => ({ value: item.value, label: `${item.label || item.name}${item.source === "community" ? " · 社区" : ""}` })) : [{ value: "", label: "暂无模型", disabled: true }]),
    ]} /></label>
    <div className="adetailer-sliders">
      <Slider label="检测置信度" value={unit.confidence} min={0.05} max={1} step={0.05} inputStep={0.01} disabled={unitLocked} onChange={(confidence) => update({ confidence })} />
      <Slider label="最多处理区域" value={unit.maxDetections} min={1} max={8} integer disabled={unitLocked} onChange={(maxDetections) => update({ maxDetections })} />
      <Slider label="最小区域比例" value={unit.maskMinRatio} min={0} max={0.5} step={0.01} disabled={unitLocked} onChange={(maskMinRatio) => update({ maskMinRatio, maskMaxRatio: typeof maskMinRatio === "number" ? Math.max(maskMinRatio, unit.maskMaxRatio) : unit.maskMaxRatio })} />
      <Slider label="最大区域比例" value={unit.maskMaxRatio} min={0.05} max={1} step={0.05} disabled={unitLocked} onChange={(maskMaxRatio) => update({ maskMaxRatio, maskMinRatio: typeof maskMaxRatio === "number" ? Math.min(maskMaxRatio, unit.maskMinRatio) : unit.maskMinRatio })} />
    </div>
    <div className="adetailer-mask-grid">{[["dilateErode", "膨胀 / 腐蚀", -128, 128], ["maskBlur", "蒙版模糊", 0, 64], ["padding", "局部边距", 0, 256]].map(([field, label, min, max]) => <label key={field}>{label}<NumberInput ariaLabel={label} value={unit[field]} min={min} max={max} integer disabled={unitLocked} onCommit={(value) => update({ [field]: value })} /></label>)}</div>
    <div className="adetailer-denoise"><Slider label="重绘强度" value={unit.denoise} min={0.05} max={1} step={0.05} inputStep={0.01} disabled={unitLocked} onChange={(denoise) => update({ denoise })} /></div>
    <div className="adetailer-overrides">
      <label><input type="checkbox" checked={unit.useSteps} disabled={unitLocked} onChange={(event) => update({ useSteps: event.target.checked })} /><span>独立步数</span><NumberInput ariaLabel="ADetailer 独立步数" value={unit.steps} min={1} max={100} integer disabled={unitLocked || !unit.useSteps} onCommit={(steps) => update({ steps })} /></label>
      <label><input type="checkbox" checked={unit.useCfg} disabled={unitLocked} onChange={(event) => update({ useCfg: event.target.checked })} /><span>独立 CFG</span><NumberInput ariaLabel="ADetailer 独立 CFG" value={unit.cfg} min={0} max={30} step={0.1} disabled={unitLocked || !unit.useCfg} onCommit={(cfg) => update({ cfg })} /></label>
    </div>
    <div className="adetailer-prompts"><label>正向提示词<textarea value={unit.prompt} disabled={unitLocked} onChange={(event) => update({ prompt: event.target.value })} placeholder="留空继承主提示词；[PROMPT] 插入主提示词" /></label><label>负向提示词<textarea value={unit.negativePrompt} disabled={unitLocked || !engineAllowsNegativePrompt} onChange={(event) => update({ negativePrompt: event.target.value })} placeholder={engineAllowsNegativePrompt ? "留空继承主负向提示词" : "FLUX.1 没有无条件分支，负向提示词不会参与生成"} /></label></div>
    {unit.enabled && unitSteps < 1 && <p className="adetailer-unavailable">该单元有效步数为 0，请提高主采样或独立步数，或提高重绘强度。</p>}
  </div>;
}

export function RTXControls({ stage, health, statusReason, trace, baseSize, finalSize, disabled, update, Slider }) {
  return <>
    {!health.available && statusReason && <p className={`rtx-unavailable ${health.probing ? "probing" : ""}`} role="status">{statusReason}</p>}
    {health.available && health.warning && statusReason && <p className="rtx-unavailable">{health.warning}</p>}
    <div className="rtx-runtime-grid" aria-label="RTX VSR 运行环境"><div><span>状态</span><strong>{health.probing ? "PROBING" : health.available ? "READY" : health.supported ? "RUNTIME REQUIRED" : "UNSUPPORTED"}</strong></div><div><span>运行时</span><strong>{health.runtime_version || "--"}</strong></div><div><span>设备</span><strong title={health.device}>{health.device || "--"}</strong></div><div><span>CUDA 能力</span><strong>{health.compute_capability || "--"}</strong></div><div><span>驱动</span><strong>{health.driver_version || "--"}</strong></div></div>
    <div className="rtx-scale"><Slider label="放大倍率" value={stage.scale} min={1} max={4} step={0.01} inputStep={0.01} fixed={2} disabled={disabled} onChange={(scale) => update({ scale: typeof scale === "number" ? Math.round(scale * 100) / 100 : scale })} suffix="x" /><p><span>INPUT {trace?.input.width || baseSize.width} × {trace?.input.height || baseSize.height}</span><b>OUTPUT {finalSize.valid === false ? "待修正" : `${trace?.output.width || baseSize.width} × ${trace?.output.height || baseSize.height}`}</b></p><p><span>CHAIN FINAL</span><b>{finalSize.valid === false ? "待修正" : `${finalSize.width} × ${finalSize.height}`}</b></p></div>
    <label className="rtx-quality">处理质量<WorkspaceSelect ariaLabel="RTX VSR 处理质量" value={stage.quality} disabled={disabled} onChange={(quality) => update({ quality })} options={[{ value: "low", label: "LOW · 低" }, { value: "medium", label: "MEDIUM · 中" }, { value: "high", label: "HIGH · 高" }, { value: "ultra", label: "ULTRA · 极致" }, ...(!["low", "medium", "high", "ultra"].includes(stage.quality) ? [{ value: stage.quality, label: `${stage.quality} · 保存值`, disabled: true }] : [])]} /></label>
    <p className="rtx-proprietary">RTX VSR 依赖 NVIDIA 专有 Video Effects 运行时，仅在受支持的 x64 Windows / Linux NVIDIA RTX 设备上可用；生成内容仍在本机处理。</p>
  </>;
}
