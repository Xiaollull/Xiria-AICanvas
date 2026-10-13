import { useMemo, useState } from "react";
import { Check, ChevronDown, ChevronLeft, ChevronRight, Download, RefreshCw, Settings2 } from "lucide-react";
import ResampleChainPanel from "./ResampleChainPanel.jsx";
import { ADetailerUnitControls, RTXControls } from "./PostprocessAuxControls.jsx";
import { XYZNumberInput } from "./SamplingControls.jsx";
import { editXYZPostprocess, xyzPostprocessSource } from "./xyz-postprocess-source.js";
import { resampleChainPlan, resampleChainIssue, createAdetailerStage, createRtxStage } from "./resample-chain.js";
import { adetailerStageIssue, adetailerUnitSteps } from "./adetailer-units.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";
import { XYZ_CAPABILITIES } from "./xyz-draft.js";

export function XYZStageCard({ title, detail, enabled, expanded, onToggleEnabled, onToggleExpanded, actions, children, switchDisabled, Header }) {
  if (Header) return <section className={`postprocess-stage-card ${enabled ? "enabled" : ""} ${expanded ? "expanded" : ""}`}><Header {...{ title, detail, enabled, expanded, onToggleEnabled, onToggleExpanded, actions, switchDisabled }} />{children}</section>;
  return <section className={`postprocess-stage-card ${enabled ? "enabled" : ""} ${expanded ? "expanded" : ""}`}><div className="postprocess-stage-head"><button type="button" className="postprocess-stage-summary" aria-expanded={expanded} onClick={onToggleExpanded}><strong>{title}</strong><small>{detail}</small></button><div className="postprocess-stage-head-actions">{actions}<button type="button" role="switch" aria-label={`启用 ${title}`} aria-checked={enabled} className={enabled ? "active" : ""} disabled={switchDisabled} onClick={onToggleEnabled}><i /></button><button type="button" className="postprocess-stage-expand" aria-label={`${expanded ? "收起" : "展开"} ${title} 参数`} onClick={onToggleExpanded}><ChevronDown size={15} /></button></div></div>{children}</section>;
}
export default function XYZPostprocessPanel({ draft, onChange, Slider, randomSeed, resources = {} }) {
  const source = xyzPostprocessSource(draft), chain = source.chain, p = draft.baseline.parameters;
  const [pages, setPages] = useState({});
  const sliders = useMemo(() => (props) => <Slider {...props} NumberInput={XYZNumberInput} />, [Slider]);
  const updateChain = (next) => onChange(editXYZPostprocess(draft, next));
  const updateStage = (stage, patch) => updateChain(chain.map((item) => item.id === stage.id ? { ...item, ...patch } : item));
  const baseSize = { width: p.width, height: p.height };
  let plan, issue;
  try {
    plan = resampleChainPlan(baseSize, chain.filter((s) => !s.legacy), draft.engine);
    issue = draft.inherit_postprocess ? resampleChainIssue(chain.filter((s) => !s.legacy), { engine: draft.engine, baseSteps: p.steps, size: baseSize, upscalers: resources.upscalers }) : "";
  } catch {
    plan = { width: null, height: null, trace: [], valid: false };
    issue = "后处理链数值无效；原值保留，请修正后预检。";
  }
  const cap = XYZ_CAPABILITIES.engines[draft.engine];
  return <section className="xyz-postprocess xyz-mother-controls postprocess-stage-stack">
    <div className="section-heading"><span>04</span><h2>后处理增强</h2></div>
    <label className="xyz-post-inherit"><input type="checkbox" checked={draft.inherit_postprocess === true} onChange={(event) => onChange({ ...draft, inherit_postprocess: event.target.checked })} /><span>执行 XYZ 冻结后处理链<small>{draft.inherit_postprocess ? "按当前顺序执行；最终尺寸可改变" : "默认关闭 · 编辑保留，不执行"}</small></span></label>
    <ResampleChainPanel chain={chain} onChange={updateChain} engine={draft.engine} samplers={["SD", "iL"].includes(draft.engine) ? SAMPLER_NAMES : XYZ_CAPABILITIES.samplers} schedulers={["SD", "iL"].includes(draft.engine) ? SCHEDULER_NAMES : XYZ_CAPABILITIES.schedulers} baseSize={baseSize} baseSteps={p.steps} baseCfg={p.cfg} preserveRawInputs
      {...Object.fromEntries(["upscalers", "upscalerRuntimeReady", "upscalersRefreshing", "onRefreshUpscalers", "upscalerDownloadJob", "upscalerDownloadBusy", "upscalerDownloadError", "onDownloadUpscaler", "onRetryUpscaler"].map((key) => [key, resources[key]]))}
      Slider={sliders} NumberInput={XYZNumberInput} randomSeed={randomSeed} chainIssue={issue} sizeLineLabel={draft.inherit_postprocess ? "尺寸链" : "冻结链预览 · 未执行"} renderCard={(props) => <XYZStageCard {...props} Header={resources.StageHeader} />}
      renderStage={(stage, index, actions) => {
        if (stage.legacy) return <p className="xyz-warning" role="alert">旧链阶段 {index + 1} 无法安全编辑；原始参数保留，预检由服务端判定。</p>;
        const update = (patch) => updateStage(stage, patch), title = stage.type === "adetailer" ? "ADetailer" : "RTX VSR";
        const unit = stage.units?.find((u) => u.id === pages[stage.id]) || stage.units?.[0], position = stage.units?.indexOf(unit) || 0;
        const adeIssue = stage.type === "adetailer" && stage.enabled ? adetailerStageIssue(stage, p.steps, (detector) => resources.adetailerModels?.some((item) => item.value === detector), draft.engine) : "";
        return <XYZStageCard Header={resources.StageHeader} title={title} detail={stage.enabled ? stage.type === "rtx" ? `${stage.scale}x` : `${stage.units.filter((u) => u.enabled).length} 个检测单元` : "关闭"} enabled={stage.enabled} expanded={stage.expanded} actions={actions} onToggleExpanded={() => update({ expanded: !stage.expanded })} onToggleEnabled={() => update({ enabled: !stage.enabled })}>
          {stage.expanded && (stage.type === "adetailer" ? <div className={`adetailer-parameters ${stage.enabled ? "enabled" : ""}`}>
            {adeIssue && <p className="adetailer-unavailable" role="alert">{adeIssue} · 已保留参数</p>}
            <div className="adetailer-model-library"><div className="adetailer-library-heading"><strong>基础推荐模型</strong><span><small>{resources.adetailerInfo?.builtins?.filter((m) => m.installed).length || 0} / {resources.adetailerInfo?.builtins?.length || 0} 已就绪</small><button type="button" title="刷新 YOLO 模型列表" aria-label="刷新 XYZ YOLO 模型列表" disabled={resources.adetailerInfo?.loading} onClick={resources.onRefreshADetailer}><RefreshCw size={11} /></button></span></div><div className="adetailer-recommended-model"><span><b>ADetailer 基础模型包</b><small>检测资源共享，参数只属于 XYZ</small></span>{resources.adetailerInfo?.builtins?.length > 0 && resources.adetailerInfo.builtins.every((m) => m.installed) ? <em><Check size={12} />已安装</em> : <button type="button" disabled={!resources.onDownloadADetailer || resources.adetailerDownload?.status === "downloading"} onClick={resources.onDownloadADetailer}><Download size={12} />下载模型包</button>}</div></div>
            {resources.adetailerDownload && <div className={`adetailer-download-status ${resources.adetailerDownload.status}`} aria-live="polite"><div><b>{resources.adetailerDownload.label}{resources.adetailerDownload.modelIndex ? ` · ${resources.adetailerDownload.modelIndex} / ${resources.adetailerDownload.totalModels}` : ""}</b><span>{resources.adetailerDownload.message}</span></div>{resources.adetailerDownload.totalBytes > 0 && <i><i style={{ width: `${Math.min(100, resources.adetailerDownload.currentBytes / resources.adetailerDownload.totalBytes * 100)}%` }} /></i>}</div>}
            {unit && <div className="adetailer-units"><nav className="adetailer-pager" aria-label="ADetailer 检测单元"><button type="button" aria-label="上一个检测单元" disabled={!position} onClick={() => setPages({ ...pages, [stage.id]: stage.units[position - 1].id })}><ChevronLeft size={14} /></button><ol>{stage.units.map((item, i) => <li key={item.id}><button type="button" className={`${unit.id === item.id ? "current" : ""} ${item.enabled ? "on" : ""}`} onClick={() => setPages({ ...pages, [stage.id]: item.id })}>{i + 1}</button></li>)}</ol><button type="button" aria-label="下一个检测单元" disabled={position === stage.units.length - 1} onClick={() => setPages({ ...pages, [stage.id]: stage.units[position + 1].id })}><ChevronRight size={14} /></button></nav><section key={unit.id} className={`adetailer-unit ${unit.enabled ? "on" : ""}`}><header><span>{String(position + 1).padStart(2, "0")}</span><strong>ADetailer {position + 1}</strong><small>{unit.detector || "未选择模型"}</small><button type="button" role="switch" aria-label={`启用第 ${position + 1} 个 ADetailer 单元`} aria-checked={unit.enabled} className={unit.enabled ? "active" : ""} onClick={() => update({ units: stage.units.map((item) => item.id === unit.id ? { ...item, enabled: !item.enabled } : item) })}><i /></button></header><ADetailerUnitControls unit={unit} models={resources.adetailerModels || []} engineAllowsNegativePrompt={cap.negative_prompt} update={(patch) => update({ units: stage.units.map((item) => item.id === unit.id ? { ...item, ...patch } : item) })} Slider={sliders} NumberInput={XYZNumberInput} unitSteps={adetailerUnitSteps(unit, p.steps, draft.engine)} /></section></div>}
          </div> : <div className={`rtx-parameters ${stage.enabled ? "enabled" : ""}`}><RTXControls stage={stage} health={resources.rtxHealth || {}} statusReason={resources.rtxStatusReason || "RTX VSR 能力未知；已保存值保留"} trace={plan.trace.find((t) => t.id === stage.id)} baseSize={baseSize} finalSize={plan} update={update} Slider={sliders} /></div>)}
        </XYZStageCard>;
      }} />
    {[['adetailer', createAdetailerStage], ['rtx', createRtxStage]].filter(([type]) => !chain.some((s) => s.type === type)).map(([type, create]) => <button type="button" key={type} onClick={() => updateChain([...chain, create({ id: `xyz-${type}` })])}>添加 {type === "rtx" ? "RTX VSR" : "ADetailer"} 阶段</button>)}
    <button type="button" className="xyz-performance-entry" onClick={resources.onOpenSettings}><Settings2 size={14} />全局性能设置</button>
  </section>;
}
