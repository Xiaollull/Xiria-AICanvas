import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  ChevronLeft,
  ChevronRight,
  Download,
  FolderOpen,
  ImageIcon,
  ImagePlus,
  Layers3,
  Maximize2,
  Pause,
  Play,
  RefreshCw,
  Square,
  Upload,
  X,
  Zap,
} from "lucide-react";
import {
  IMAGE_TO_IMAGE_MODES,
  RESIZE_MODES,
  SIZE_MODES,
  SOURCE_IMAGE_ACCEPT,
  imageToImageBlockers,
  normalizeImageToImageSettings,
  outputSize,
  postprocessSourceIssue,
  resizeModeMatters,
  sourceImageSummary,
  validateSourceFile,
} from "./image-to-image";
import SizeGrid from "./SizeGrid";
import { normalizePostprocessOrder, postprocessTargetSize } from "./postprocessing";
import WorkspaceSelect from "./WorkspaceSelect";
import { hiresEffectiveSteps, secureRandomUint64Seed } from "./hires-settings";
import { formatWeight } from "./lora-weight";
import {
  ADETAILER_UNIT_LIMIT,
  DISTILLED_GUIDANCE_ENGINES,
  activeADetailerUnits,
  adetailerPageIndex,
  adetailerStageIssue,
  adetailerStepUnitId,
  adetailerSummary,
  adetailerUnitLabel,
  adetailerUnitSteps,
} from "./adetailer-units";

const SEED_MODES = [
  { id: "random", label: "随机" },
  { id: "fixed", label: "固定" },
  { id: "increment", label: "递增" },
  { id: "decrement", label: "递减" },
];

function RangeField({ label, hint, value, min, max, step, onChange, disabled, format }) {
  const progress = ((value - min) / (max - min)) * 100;
  return (
    <label className="i2i-range">
      <span><b>{label}</b>{hint && <small>{hint}</small>}<i>{format ? format(value) : value}</i></span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} style={{ "--i2i-range-progress": `${Math.max(0, Math.min(100, progress))}%` }} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function NumberInput({ value, min, max, step = 1, integer = false, disabled, onChange, ariaLabel }) {
  const inputRef = useRef(null);
  const cancelBlurRef = useRef(false);
  const displayValue = (next) => String(next);
  const [draft, setDraft] = useState(displayValue(value));

  useEffect(() => {
    if (document.activeElement !== inputRef.current) setDraft(displayValue(value));
  }, [value]);

  const reset = () => setDraft(displayValue(value));
  const commit = () => {
    const parsed = Number(draft);
    if (!draft.trim() || !Number.isFinite(parsed)) { reset(); return; }
    const stepped = Math.round(parsed / step) * step;
    const next = Math.max(min, Math.min(max, integer ? Math.round(stepped) : Math.round(stepped * 100) / 100));
    setDraft(displayValue(next));
    onChange(next);
  };

  return <input ref={inputRef} className="i2i-number-input" type="text" inputMode={integer ? "numeric" : "decimal"} value={draft} disabled={disabled} aria-label={ariaLabel} onChange={(event) => setDraft(event.target.value)} onBlur={() => {
    if (cancelBlurRef.current) { cancelBlurRef.current = false; reset(); return; }
    commit();
  }} onKeyDown={(event) => {
    if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
    if (event.key === "Escape") { event.preventDefault(); cancelBlurRef.current = true; reset(); event.currentTarget.blur(); }
  }} />;
}

function NumberField({ label, ...props }) {
  return <label className="i2i-number-field"><span>{label}</span><NumberInput ariaLabel={`${label}数值`} {...props} /></label>;
}

function readSourceFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("图片读取失败"));
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const probe = new window.Image();
      probe.onerror = () => reject(new Error("图片解码失败，请换一张图片"));
      probe.onload = () => resolve({
        dataUrl,
        name: file.name || "source.png",
        type: file.type || "",
        size: file.size || 0,
        width: probe.naturalWidth,
        height: probe.naturalHeight,
      });
      probe.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

function StageHeader({ title, detail, enabled, disabled, onToggle, children }) {
  return (
    <div className="i2i-stage-head">
      <div><strong>{title}</strong><small>{detail}</small></div>
      <div className="i2i-stage-head-actions">{children}<button type="button" role="switch" aria-checked={enabled} disabled={disabled} onClick={onToggle}><i /></button></div>
    </div>
  );
}

function PostprocessControls({ config, engine, postprocess, running, dimensions, updateStage, moveStage, order, postprocessOnly }) {
  // Which unit the ADetailer pager is showing. View state, not configuration: it
  // is never persisted, because where the user last paged to says nothing about
  // the render.
  const [adetailerPage, setADetailerPage] = useState("");
  const upscalers = Array.isArray(postprocess.upscalers) ? postprocess.upscalers : [];
  const adetailerModels = Array.isArray(postprocess.adetailerModels) ? postprocess.adetailerModels : [];
  const upscalerRuntime = postprocess.upscalerRuntime !== false;
  const adetailerRuntime = postprocess.adetailerInfo?.runtimeAvailable === true;
  const rtxAvailable = postprocess.rtxHealth?.available === true;
  // A stage the engine has not declared, or that the memory mode forbids outright, cannot be turned
  // on at all: leaving the switch live would let the page accept a setting the run then refuses.
  // An already-enabled stage stays switchable so it can be turned back off.
  const stageLocked = (stage) => running || postprocess.ultraLow === true || engine.features?.[stage] === false;
  // A switch that cannot be moved has to say why. Shown in place of "关闭", which is the one line
  // already on screen for a stage that is off.
  const stageReason = (stage) => {
    if (postprocess.ultraLow) return "极限省存模式下不可用";
    if (engine.features?.[stage] === false) return `当前 ${engine.name} 运行时未声明该能力`;
    if (stage === "hires" && !upscalerRuntime) return "超分运行环境尚未配置";
    if (stage === "adetailer" && !adetailerRuntime) return postprocess.adetailerInfo?.error || "YOLO 检测运行环境尚未就绪";
    if (stage === "rtx" && !rtxAvailable) return postprocess.rtxHealth?.reason || "RTX VSR 运行时不可用";
    return "";
  };
  const selectedUpscaler = upscalers.find((item) => item.id === config.hires.model);
  const hiresSteps = hiresEffectiveSteps(config.hires, engine.name);
  // A detail pass encodes its own negative prompt through the mounted runtime, so an engine with
  // no unconditional branch cannot use one. Same rule as the main negative prompt: the box is
  // disabled and the request drops the text, but the text itself is kept.
  const engineAllowsNegativePrompt = !DISTILLED_GUIDANCE_ENGINES.includes(engine.name);
  const distilledEngineLabel = engine.name === "Flux2" ? "FLUX.2" : "FLUX.1";
  const adetailerUnits = config.adetailer.units;
  const updateUnit = (id, updates) => updateStage("adetailer", {
    units: adetailerUnits.map((unit) => (unit.id === id ? { ...unit, ...updates } : unit)),
  });
  const adetailerIndex = adetailerPageIndex(adetailerUnits, adetailerPage);
  const adetailerUnit = adetailerUnits[adetailerIndex];

  return (
    <div className="i2i-postprocess-stack">
      <div className="section-heading"><span>06</span><h2>后处理增强</h2><small className="i2i-section-kicker">{postprocessOnly ? "直接作用于来源图" : "生成后按顺序执行"}</small></div>
      {postprocessOnly && <p className="i2i-note">后处理模式下这里就是本次任务的全部内容：至少启用一个阶段，未启用的阶段会被跳过。</p>}
      <section className={`i2i-stage-card ${config.hires.enabled ? "enabled" : ""}`}>
        <StageHeader title="Hires.fix" detail={config.hires.enabled ? `${config.hires.scale.toFixed(1)}× · ${dimensions.width} × ${dimensions.height}` : stageReason("hires") || "关闭"} enabled={config.hires.enabled} disabled={stageLocked("hires") || (!upscalerRuntime && !config.hires.enabled)} onToggle={() => updateStage("hires", { enabled: !config.hires.enabled })} />
        {config.hires.enabled && <div className="i2i-stage-body i2i-hires-body">
          <label>超分模型<WorkspaceSelect ariaLabel="超分模型" value={config.hires.model} disabled={running} onChange={(value) => updateStage("hires", { model: value })} options={upscalers.filter((item) => item.compatible !== false).map((item) => ({ value: item.id, label: item.label || item.id })).concat(upscalers.some((item) => item.compatible !== false) ? [] : [{ value: "", label: "暂无兼容模型", disabled: true }])} /></label>
          <RangeField label="放大倍率" value={config.hires.scale} min={1} max={4} step={0.1} disabled={running} onChange={(value) => updateStage("hires", { scale: Math.round(value * 10) / 10 })} format={(value) => `${value.toFixed(1)}×`} />
          <div className="i2i-compact-grid"><RangeField label="二次重绘" value={config.hires.denoise} min={0.05} max={1} step={0.05} disabled={running} onChange={(value) => updateStage("hires", { denoise: value })} format={(value) => value.toFixed(2)} /><RangeField label="Hires 步数" value={config.hires.steps} min={1} max={100} step={1} disabled={running} onChange={(value) => updateStage("hires", { steps: value })} /></div>
          <RangeField label="Hires CFG" value={config.hires.cfg} min={0} max={30} step={0.5} disabled={running} onChange={(value) => updateStage("hires", { cfg: value })} format={(value) => value.toFixed(1)} />
          <label>Hires Seed 模式<WorkspaceSelect ariaLabel="Hires Seed 模式" value={config.hires.seedMode} disabled={running} onChange={(value) => updateStage("hires", { seedMode: value, seed: value === "fixed" ? config.hires.seed : "" })} options={[{ value: "inherit", label: "继承首轮 Seed" }, { value: "fixed", label: "固定 Hires Seed" }, { value: "random", label: "每张安全随机" }]} /></label>
          {config.hires.seedMode === "fixed" && <label className="i2i-fixed-seed">固定 Hires Seed<span><input className="i2i-inline-input" inputMode="numeric" maxLength="20" value={config.hires.seed} disabled={running} onChange={(event) => updateStage("hires", { seed: event.target.value.replace(/\D/g, "") })} placeholder="0–18446744073709551615" /><button type="button" title="生成固定 Hires Seed" disabled={running} onClick={() => updateStage("hires", { seed: secureRandomUint64Seed() })}><RefreshCw size={13} /></button></span></label>}
          {engine.name === "Anima" && <label>重绘方式<WorkspaceSelect ariaLabel="Hires 重绘方式" value={config.hires.executionMode} disabled={running} onChange={(value) => updateStage("hires", { executionMode: value })} options={[{ value: "usdu_tiled", label: "USDU 分块" }, { value: "full_frame", label: "整图" }]} /></label>}
          <div className="i2i-compact-grid i2i-select-grid">
            <label>Hires 采样器<WorkspaceSelect ariaLabel="Hires 采样器" value={config.hires.sampler || ""} disabled={running} onChange={(value) => updateStage("hires", { sampler: value || null })} options={[{ value: "", label: "跟随首轮" }, ...engine.samplers.map((name) => ({ value: name, label: name }))]} /></label>
            <label>Hires 调度器<WorkspaceSelect ariaLabel="Hires 调度器" value={config.hires.scheduler || ""} disabled={running} onChange={(value) => updateStage("hires", { scheduler: value || null })} options={[{ value: "", label: "跟随首轮" }, ...engine.schedulers.map((name) => ({ value: name, label: name }))]} /></label>
          </div>
          {engine.name === "Anima" && config.hires.executionMode === "usdu_tiled" && <><div className="i2i-readonly-grid"><label>扩散重绘分块宽度<output>Auto（只读）</output></label><label>扩散重绘分块高度<output>Auto（只读）</output></label></div><p className="i2i-note">Auto 使用首轮源图宽高；padding 32 · mask blur 8 · uniform tiles · tiled decode。</p></>}
          <div className="i2i-number-grid two"><NumberField label="像素放大分块" value={config.hires.tileSize} min={32} max={2048} integer disabled={running} onChange={(tileSize) => updateStage("hires", { tileSize, tileOverlap: Math.min(config.hires.tileOverlap, Math.floor(tileSize / 2)) })} /><NumberField label="分块重叠" value={config.hires.tileOverlap} min={0} max={Math.min(512, Math.floor(config.hires.tileSize / 2))} integer disabled={running} onChange={(tileOverlap) => updateStage("hires", { tileOverlap })} /></div>
          {!selectedUpscaler && <p className="i2i-stage-warning">请选择兼容的超分模型</p>}
          {config.hires.seedMode === "fixed" && !config.hires.seed && <p className="i2i-stage-warning">固定 Hires Seed 需要填写 0 ～ 18446744073709551615</p>}
          {hiresSteps < 1 && <p className="i2i-stage-warning">Hires 有效步数至少需要 1 步</p>}
        </div>}
      </section>

      <section className={`i2i-stage-card ${config.adetailer.enabled ? "enabled" : ""}`}>
        <StageHeader title="ADetailer" detail={config.adetailer.enabled ? adetailerSummary(config.adetailer) : stageReason("adetailer") || "关闭"} enabled={config.adetailer.enabled} disabled={stageLocked("adetailer") || (!adetailerRuntime && !config.adetailer.enabled)} onToggle={() => updateStage("adetailer", { enabled: !config.adetailer.enabled })} />
        {config.adetailer.enabled && <div className="i2i-stage-body i2i-adetailer-body">
          {adetailerUnits.length > 1 && <nav className="i2i-adetailer-pager" aria-label="ADetailer 检测单元">
            <button type="button" aria-label="上一个检测单元" disabled={adetailerIndex < 1} onClick={() => setADetailerPage(adetailerStepUnitId(adetailerUnits, adetailerPage, -1))}><ChevronLeft size={14} /></button>
            <ol>
              {adetailerUnits.map((entry, position) => <li key={entry.id}>
                <button
                  type="button"
                  className={`${position === adetailerIndex ? "current" : ""} ${entry.enabled ? "on" : ""}`}
                  aria-current={position === adetailerIndex ? "true" : undefined}
                  aria-label={`第 ${position + 1} 个检测单元${entry.enabled ? "" : "（已关闭）"}`}
                  onClick={() => setADetailerPage(entry.id)}
                >{position + 1}</button>
              </li>)}
            </ol>
            <button type="button" aria-label="下一个检测单元" disabled={adetailerIndex >= adetailerUnits.length - 1} onClick={() => setADetailerPage(adetailerStepUnitId(adetailerUnits, adetailerPage, 1))}><ChevronRight size={14} /></button>
          </nav>}
          {adetailerUnit && (() => {
            const unit = adetailerUnit;
            const index = adetailerIndex;
            const unitSteps = adetailerUnitSteps(unit, config.steps, engine.name);
            // Keyed by the unit so paging *replaces* the controls rather than reusing
            // them: `NumberInput` holds an uncommitted draft of its own, and reused
            // controls would carry one unit's half-typed value onto the next unit's field.
            return <section className={`i2i-adetailer-unit ${unit.enabled ? "on" : ""}`} key={unit.id}>
              <header>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{adetailerUnitLabel(index, adetailerUnits.length)}</strong>
                <small>{unit.detector ? unit.detector.split("/").pop() : "未选择模型"}</small>
                <button type="button" role="switch" aria-label={`启用第 ${index + 1} 个 ADetailer 单元`} aria-checked={unit.enabled} className={unit.enabled ? "active" : ""} disabled={running} onClick={() => updateUnit(unit.id, { enabled: !unit.enabled })}><i /></button>
              </header>
              <div className="i2i-adetailer-unit-body">
                <label>检测模型<WorkspaceSelect ariaLabel={`第 ${index + 1} 个 ADetailer 检测模型`} value={unit.detector} disabled={running} onChange={(value) => updateUnit(unit.id, { detector: value })} options={adetailerModels.map((item) => ({ value: item.value, label: item.label || item.name })).concat(adetailerModels.length ? [] : [{ value: "", label: "暂无模型", disabled: true }])} /></label>
                <div className="i2i-compact-grid"><RangeField label="检测置信度" value={unit.confidence} min={0.05} max={1} step={0.05} disabled={running} onChange={(value) => updateUnit(unit.id, { confidence: value })} format={(value) => value.toFixed(2)} /><RangeField label="最多处理区域" value={unit.maxDetections} min={1} max={8} step={1} disabled={running} onChange={(value) => updateUnit(unit.id, { maxDetections: value })} /></div>
                <div className="i2i-compact-grid"><RangeField label="最小区域比例" value={unit.maskMinRatio} min={0} max={0.5} step={0.01} disabled={running} onChange={(value) => updateUnit(unit.id, { maskMinRatio: value, maskMaxRatio: Math.max(value, unit.maskMaxRatio) })} format={(value) => value.toFixed(2)} /><RangeField label="最大区域比例" value={unit.maskMaxRatio} min={0.05} max={1} step={0.05} disabled={running} onChange={(value) => updateUnit(unit.id, { maskMaxRatio: value, maskMinRatio: Math.min(value, unit.maskMinRatio) })} format={(value) => value.toFixed(2)} /></div>
                <div className="i2i-number-grid three"><NumberField label="膨胀 / 腐蚀" value={unit.dilateErode} min={-128} max={128} integer disabled={running} onChange={(dilateErode) => updateUnit(unit.id, { dilateErode })} /><NumberField label="蒙版模糊" value={unit.maskBlur} min={0} max={64} integer disabled={running} onChange={(maskBlur) => updateUnit(unit.id, { maskBlur })} /><NumberField label="局部边距" value={unit.padding} min={0} max={256} integer disabled={running} onChange={(padding) => updateUnit(unit.id, { padding })} /></div>
                <RangeField label="局部重绘强度" value={unit.denoise} min={0.05} max={1} step={0.05} disabled={running} onChange={(value) => updateUnit(unit.id, { denoise: value })} format={(value) => value.toFixed(2)} />
                <div className="i2i-override-grid"><label><input type="checkbox" checked={unit.useSteps} disabled={running} onChange={(event) => updateUnit(unit.id, { useSteps: event.target.checked })} /><span>独立步数</span><NumberInput ariaLabel={`第 ${index + 1} 个 ADetailer 独立步数`} value={unit.steps} min={1} max={100} integer disabled={running || !unit.useSteps} onChange={(value) => updateUnit(unit.id, { steps: value })} /></label><label><input type="checkbox" checked={unit.useCfg} disabled={running} onChange={(event) => updateUnit(unit.id, { useCfg: event.target.checked })} /><span>独立 CFG</span><NumberInput ariaLabel={`第 ${index + 1} 个 ADetailer 独立 CFG`} value={unit.cfg} min={0} max={30} step={0.1} disabled={running || !unit.useCfg} onChange={(cfg) => updateUnit(unit.id, { cfg })} /></label></div>
                <div className="i2i-detail-prompts"><label>正向提示词<textarea value={unit.prompt} disabled={running} onChange={(event) => updateUnit(unit.id, { prompt: event.target.value })} placeholder="留空继承主提示词；[PROMPT] 插入主提示词" /></label><label>负向提示词<textarea value={unit.negativePrompt} disabled={running || !engineAllowsNegativePrompt} onChange={(event) => updateUnit(unit.id, { negativePrompt: event.target.value })} placeholder={engineAllowsNegativePrompt ? "留空继承主负向提示词" : `${distilledEngineLabel} 没有无条件分支，负向提示词不会参与生成`} /></label></div>
                <p className="i2i-note">{unitSteps} 个有效检测重绘步数</p>
                {unit.enabled && unitSteps < 1 && <p className="i2i-stage-warning">该单元有效步数至少需要 1 步</p>}
              </div>
            </section>;
          })()}
          <div className="i2i-adetailer-add">
            <small>{ADETAILER_UNIT_LIMIT} 个检测单元 · 已启用 {activeADetailerUnits(config.adetailer).length} 个 · 按编号顺序依次执行，后一个在前一个的结果上继续修复</small>
          </div>
          <p className="i2i-note">重绘继承当前采样器、调度器、底模与 LoRA。</p>
        </div>}
      </section>

      <section className={`i2i-stage-card ${config.rtx.enabled ? "enabled" : ""}`}>
        <StageHeader title="RTX VSR" detail={config.rtx.enabled ? `${config.rtx.scale.toFixed(2)}× · ${dimensions.width} × ${dimensions.height}` : stageReason("rtx") || "关闭"} enabled={config.rtx.enabled} disabled={stageLocked("rtx") || (!rtxAvailable && !config.rtx.enabled)} onToggle={() => updateStage("rtx", { enabled: !config.rtx.enabled })} />
        {config.rtx.enabled && <div className="i2i-stage-body"><div className="i2i-compact-grid"><RangeField label="放大倍率" value={config.rtx.scale} min={1} max={4} step={0.01} disabled={running} onChange={(value) => updateStage("rtx", { scale: Math.round(value * 100) / 100 })} format={(value) => `${value.toFixed(2)}×`} /><label>处理质量<WorkspaceSelect ariaLabel="RTX VSR 处理质量" value={config.rtx.quality} disabled={running} onChange={(value) => updateStage("rtx", { quality: value })} options={[{ value: "low", label: "LOW" }, { value: "medium", label: "MEDIUM" }, { value: "high", label: "HIGH" }, { value: "ultra", label: "ULTRA" }]} /></label></div><p className="i2i-note">最终预估 {dimensions.width} × {dimensions.height} · 上限 8192 边 / 32MP</p></div>}
      </section>

      <div className="i2i-order-card"><div className="i2i-order-head"><span>07</span><strong>处理顺序</strong><b>FINAL {dimensions.width} × {dimensions.height}</b></div>{order.map((stage, index) => <div className={`i2i-order-row ${config[stage].enabled ? "enabled" : ""}`} key={stage}><span>{String(index + 1).padStart(2, "0")}</span><strong>{stage === "hires" ? "Hires.fix" : stage === "adetailer" ? "ADetailer" : "RTX VSR"}</strong><small>{config[stage].enabled ? "启用" : "关闭"}</small><div><button type="button" title="上移" disabled={running || index === 0} onClick={() => moveStage(stage, -1)}><ArrowUp size={13} /></button><button type="button" title="下移" disabled={running || index === order.length - 1} onClick={() => moveStage(stage, 1)}><ArrowDown size={13} /></button></div></div>)}</div>
    </div>
  );
}

export default function ImageToImagePage({
  engine,
  modelPicker,
  job,
  settings,
  source,
  onSettingsChange,
  onSourceChange,
  onGenerate,
  onControl,
  onSelectOutput,
  onOpenViewer,
  onOpenLoraManager,
  onAddToGallery,
  onNotice,
  postprocess = {},
}) {
  const config = normalizeImageToImageSettings(settings, { samplers: engine.samplers, schedulers: engine.schedulers });
  const postprocessOnly = config.mode === "postprocess";
  const canvas = outputSize(source, config);
  const order = normalizePostprocessOrder(config.postprocessOrder);
  const dimensions = postprocessTargetSize(canvas, order, config);
  const selectedUpscaler = postprocess.upscalers?.find((item) => item.id === config.hires.model);
  const hiresSteps = hiresEffectiveSteps(config.hires, engine.name);
  // Mirrors the generate page: a guidance-distilled engine encodes no negative prompt, so the box
  // is disabled and the request drops the text while the text itself survives an engine switch.
  const pageAllowsNegativePrompt = !DISTILLED_GUIDANCE_ENGINES.includes(engine.name);
  const distilledPageLabel = engine.name === "Flux2" ? "FLUX.2" : "FLUX.1";
  const hiresReady = !config.hires.enabled || (engine.features?.hires !== false && postprocess.upscalerRuntime !== false && Boolean(selectedUpscaler) && hiresSteps >= 1 && config.hires.tileOverlap <= Math.floor(config.hires.tileSize / 2));
  const adetailerIssue = config.adetailer.enabled
    ? adetailerStageIssue(config.adetailer, config.steps, (detector) => postprocess.adetailerModels?.some((item) => item.value === detector), engine.name)
    : "";
  const adetailerReady = !config.adetailer.enabled || (engine.features?.adetailer !== false && postprocess.adetailerInfo?.available === true && !adetailerIssue);
  const rtxReady = !config.rtx.enabled || (engine.features?.rtx !== false && postprocess.rtxHealth?.available === true && dimensions.valid);
  const [dragging, setDragging] = useState(false);
  const [loadingSource, setLoadingSource] = useState(false);
  const fileInputRef = useRef(null);
  const sourceTokenRef = useRef(0);
  const running = job.status === "running";
  const update = (patch) => onSettingsChange({ ...config, ...patch });
  const updateStage = (stage, patch) => update({ [stage]: { ...config[stage], ...patch } });
  const moveStage = (stage, direction) => {
    if (running) return;
    const index = order.indexOf(stage);
    const target = index + direction;
    if (index < 0 || target < 0 || target >= order.length) return;
    const next = [...order];
    [next[index], next[target]] = [next[target], next[index]];
    update({ postprocessOrder: next });
  };
  const acceptFile = async (file) => {
    if (running) return;
    const problem = validateSourceFile(file);
    if (problem) { onNotice(problem, true); return; }
    const token = ++sourceTokenRef.current;
    setLoadingSource(true);
    try {
      const nextSource = await readSourceFile(file);
      if (token === sourceTokenRef.current && !running) onSourceChange(nextSource);
    } catch (error) {
      if (token === sourceTokenRef.current) onNotice(error.message, true);
    } finally {
      if (token === sourceTokenRef.current) setLoadingSource(false);
    }
  };
  const useLastOutput = async () => {
    if (running) return;
    const output = job.outputs[job.selectedIndex] || job.outputs[0];
    if (!output?.url) return;
    const token = ++sourceTokenRef.current;
    setLoadingSource(true);
    try {
      const response = await fetch(output.url);
      if (!response.ok) throw new Error("无法读取上一张生成结果");
      const blob = await response.blob();
      if (token === sourceTokenRef.current) await acceptFile(new File([blob], `output-${output.index ?? 0}.png`, { type: blob.type || "image/png" }));
    } catch (error) {
      if (token === sourceTokenRef.current) onNotice(error.message, true);
    } finally {
      if (token === sourceTokenRef.current) setLoadingSource(false);
    }
  };
  useEffect(() => {
    const onPaste = (event) => {
      if (running) return;
      const file = [...(event.clipboardData?.files || [])][0];
      if (file) { event.preventDefault(); void acceptFile(file); }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });
  const blocker = imageToImageBlockers({
    source,
    settings: config,
    engine: engine.name,
    engineReady: engine.ready,
    serviceReady: engine.serviceReady,
    running,
    postprocess: {
      hiresReady,
      hiresReason: postprocess.upscalerRuntime === false ? "Hires.fix 运行环境尚未配置" : "Hires 参数或超分模型未就绪",
      adetailerReady,
      adetailerReason: "ADetailer 检测模型尚未就绪",
      rtxReady,
      rtxReason: postprocess.rtxHealth?.reason || "RTX VSR 运行时不可用",
      dimensions,
    },
  });
  const totalImages = config.imagesPerBatch * config.batchCount;
  const selectedOutput = job.outputs[job.selectedIndex];
  const enabledStages = order.filter((stage) => config[stage].enabled);
  const stageLabel = (stage) => (stage === "hires" ? "Hires.fix" : stage === "adetailer" ? "ADetailer" : "RTX VSR");
  const sourceIssue = postprocessOnly ? postprocessSourceIssue(source) : "";

  return (
    <section className="image-workspace">
      <header className="i2i-page-head">
        <h1 className="eyebrow">{postprocessOnly ? "IMAGE POST-PROCESSING WORKBENCH" : "IMAGE TRANSFORMATION WORKBENCH"}</h1>
        <div className="i2i-head-status"><span className={source ? "ready" : ""}><i />{source ? "来源图已就绪" : "等待来源图"}</span><b>{canvas.width} × {canvas.height}</b></div>
      </header>

      {/* The two pictures are the page: everything else is a control for turning the left one into
          the right one, so they share one stage at equal size and every parameter moves to a rail. */}
      <section className="i2i-stage-panel panel">
        {/* Pinned above the pictures. What the run is about to do, and the button that does it,
            are the one thing on this page that must never move when something below it grows. */}
        <div className="i2i-run-bar">
          {/* A mode, not a parameter: it changes what the button below it does, so it sits with the
              button rather than in the numbered parameter stack on the rail. */}
          <div className="i2i-mode-switch" role="group" aria-label="处理模式">
            {IMAGE_TO_IMAGE_MODES.map((mode) => <button
              type="button"
              key={mode.id}
              className={config.mode === mode.id ? "active" : ""}
              aria-pressed={config.mode === mode.id}
              disabled={running}
              onClick={() => update({ mode: mode.id })}
            ><strong>{mode.label}</strong><small>{mode.detail}</small></button>)}
          </div>
          <div className="i2i-run-actions">
            <div className="generation-info">
              <div title={engine.detail}><span>当前模型</span><strong>{engine.label}</strong></div>
              {postprocessOnly
                ? <div title={enabledStages.map(stageLabel).join(" → ") || "尚未启用任何阶段"}><span>处理链</span><strong>{enabledStages.length ? enabledStages.map(stageLabel).join(" → ") : "未启用"}</strong></div>
                : <div><span>重绘强度</span><strong>{config.denoise.toFixed(2)}</strong></div>}
              <div><span>最终预估</span><strong>{dimensions.width} × {dimensions.height}</strong></div>
            </div>
            <button className="generate-button" title={blocker || (postprocessOnly ? "开始后处理" : "开始生成")} onClick={onGenerate} disabled={Boolean(blocker)}>
              <span className="generate-icon">{running ? <RefreshCw className="spin" size={18} /> : <Zap size={18} />}</span>
              <span><strong>{running ? (postprocessOnly ? "正在后处理" : "正在生成") : blocker ? "暂时无法执行" : postprocessOnly ? "开始后处理" : "开始图生图"}</strong><small>{running ? `批次 ${job.batchIndex || 1}/${job.batchCount} · ${job.completedImages}/${job.totalImages} 张` : blocker || `${config.imagesPerBatch} 张 × ${config.batchCount} 批 · 共 ${totalImages} 张`}</small></span>
              <kbd>Ctrl/⌘ ↵</kbd>
            </button>
          </div>
          {running && <div className="generation-progress-dock">
            <div className="generation-progress-line"><i style={{ width: `${job.progress}%` }} /></div>
            <div className="generation-progress-detail"><span>{job.phase || "正在生成"}</span><span>STEP {job.step} / {job.totalSteps || config.steps} · BATCH {job.batchIndex || 1}/{job.batchCount} · {job.completedImages}/{job.totalImages} IMAGES</span></div>
          </div>}
          {running && <div className="generation-controls">
            <button className="pause-control" disabled={job.controlBusy === "cancel" || job.taskStatus === "cancelling" || job.taskStatus === "pausing"} onClick={() => onControl(job.taskStatus === "paused" ? "resume" : "pause")}>{job.taskStatus === "paused" ? <Play size={15} /> : <Pause size={15} />}<span>{job.taskStatus === "paused" ? "继续生成" : "暂停生成"}</span></button>
            <button className="cancel-control" disabled={job.controlBusy === "cancel" || job.taskStatus === "cancelling"} onClick={() => onControl("cancel")}><Square size={14} /><span>{job.taskStatus === "cancelling" ? "正在终止" : "终止生成"}</span></button>
          </div>}
          {job.warning && <p className="generation-warning">{job.warning}</p>}
          <p className={`backend-note ${engine.serviceReady ? "online" : ""}`}><span />项目内推理服务 · <b>{engine.serviceLabel}</b></p>
        </div>

        <div className="i2i-prompt-deck">
          <div className="section-heading"><span>01</span><h2>提示词</h2><small className="i2i-section-kicker">{postprocessOnly ? "供各阶段重绘继承" : "描述改变方向"}</small></div>
          <label className="i2i-prompt"><span>正向提示词</span><textarea value={config.positive} disabled={running} placeholder="描述你希望这张图变成什么样" onChange={(event) => update({ positive: event.target.value })} /></label>
          <label className="i2i-prompt"><span>反向提示词</span><textarea className="negative" value={config.negative} disabled={running || !pageAllowsNegativePrompt} placeholder={pageAllowsNegativePrompt ? "不希望出现的内容" : `${distilledPageLabel} 没有无条件分支，负向提示词不会参与生成`} onChange={(event) => update({ negative: event.target.value })} /></label>
        </div>
        <div className="i2i-compare">
          <article className="i2i-compare-pane">
            <header className="i2i-compare-head">
              <div><span className="eyebrow">SOURCE</span><h2>来源图片</h2></div>
              <div className="i2i-compare-actions">
                <button type="button" title="把上一次生成结果作为来源" aria-label="把上一次生成结果作为来源" disabled={running || loadingSource || !job.outputs.length} onClick={useLastOutput}><ArrowLeft size={15} /></button>
                <button type="button" title="选择来源图片" aria-label="选择来源图片" disabled={running || loadingSource} onClick={() => fileInputRef.current?.click()}><Upload size={15} /></button>
                <button type="button" title="移除来源图片" aria-label="移除来源图片" disabled={running || !source} onClick={() => onSourceChange(null)}><X size={15} /></button>
              </div>
            </header>
            <div
              className={`i2i-compare-figure i2i-dropzone ${dragging ? "dragging" : ""} ${source ? "filled" : ""}`}
              onDragOver={(event) => { if (!running) { event.preventDefault(); setDragging(true); } }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { if (running) return; event.preventDefault(); setDragging(false); const file = [...(event.dataTransfer?.files || [])][0]; if (file) void acceptFile(file); }}
            >
              {source ? <img src={source.dataUrl} alt={`来源图片 ${source.name}`} /> : <div className="i2i-dropzone-empty"><Upload size={26} /><strong>{loadingSource ? "正在读取图片" : "拖入、粘贴或点击选择图片"}</strong><small>PNG · JPEG · WebP</small></div>}
              <button type="button" className="i2i-dropzone-hit" disabled={running || loadingSource} onClick={() => fileInputRef.current?.click()} aria-label="选择来源图片" />
              <input ref={fileInputRef} type="file" accept={SOURCE_IMAGE_ACCEPT} hidden onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void acceptFile(file); }} />
              <div className="corner corner-tl" /><div className="corner corner-tr" /><div className="corner corner-bl" /><div className="corner corner-br" />
            </div>
            <footer className="i2i-compare-foot">
              {source
                ? <><b title={source.name}>{source.name}</b><span>{sourceImageSummary(source)} → {canvas.width} × {canvas.height}</span></>
                : <span className="i2i-compare-hint">图片仅保存在本次会话，不写入工作区配置</span>}
            </footer>
          </article>

          <article className="i2i-compare-pane">
            <header className="i2i-compare-head">
              <div><span className="eyebrow">RESULT</span><h2>生成结果</h2></div>
              <div className="i2i-compare-actions">
                <button type="button" title="把结果加入画廊" aria-label="把结果加入画廊" disabled={!job.outputs.some((output) => output.asset_id)} onClick={onAddToGallery}><ImagePlus size={15} /></button>
                <button type="button" title="打开图片预览与拼图工作区" aria-label="打开图片预览与拼图工作区" onClick={onOpenViewer}><Maximize2 size={15} /></button>
                <button type="button" title="下载当前结果" aria-label="下载当前结果" disabled={!selectedOutput?.url} onClick={() => {
                  if (!selectedOutput?.url) return;
                  const link = document.createElement("a");
                  link.href = selectedOutput.url;
                  link.download = `XirAI-i2i-${selectedOutput.seed || selectedOutput.index || 0}.png`;
                  link.click();
                }}><Download size={15} /></button>
              </div>
            </header>
            <div className={`i2i-compare-figure preview-stage ${job.status}`}>
              <div className="preview-grid" />
              {job.status === "idle" && <div className="empty-preview"><div className="empty-orbit"><ImageIcon size={29} /></div><strong>{source ? "已就绪" : "等待来源图片"}</strong><p>{source ? (postprocessOnly ? "选择增强阶段与顺序后开始" : "调整重绘强度后开始生成") : "先放入一张来源图片"}</p><button type="button" onClick={onOpenViewer}><FolderOpen size={14} />打开预览工作区</button></div>}
              {job.status === "running" && <div className="generating-preview"><div className="step-preview-empty">{job.livePreview ? null : <><ImageIcon size={28} /><span>正在生成</span></>}</div>{job.livePreview && <img className="step-preview-image" src={job.livePreview} alt="实时预览" />}</div>}
              {job.status === "complete" && selectedOutput && <button className="completed-preview" onClick={onOpenViewer} title="点击放大预览"><img className="generated-image" src={selectedOutput.url} alt={`第 ${job.selectedIndex + 1} 张图生图结果`} /></button>}
              {job.status === "cancelled" && <div className="generation-cancelled"><Square size={30} /><strong>生成已终止</strong><p>已安全停止任务</p><button onClick={() => onControl("reset")}>返回调整参数</button></div>}
              {job.status === "error" && <div className="generation-error"><X size={32} /><strong>生成失败</strong><p>{job.error}</p><button onClick={() => onControl("reset")}>返回调整参数</button></div>}
              <div className="corner corner-tl" /><div className="corner corner-tr" /><div className="corner corner-bl" /><div className="corner corner-br" />
            </div>
            <footer className="i2i-compare-foot">
              {job.status === "complete" && selectedOutput
                ? <><b>{selectedOutput.width || canvas.width} × {selectedOutput.height || canvas.height}</b><span>批 {selectedOutput.batch_index || 1} / 图 {selectedOutput.image_index || 1} · Seed {selectedOutput.seed ?? config.seed} · {job.elapsed.toFixed(1)} 秒</span></>
                : <span className="i2i-compare-hint">{running ? `${job.phase || "正在生成"} · ${job.completedImages}/${job.totalImages}` : `预计输出 ${dimensions.width} × ${dimensions.height}`}</span>}
            </footer>
          </article>
        </div>

        {job.status === "complete" && job.outputs.length > 1 && <div className="output-selector" aria-label="生成结果选择器">
          <div className="output-selector-head"><span>结果浏览</span><b>第 {job.selectedIndex + 1} / {job.outputs.length} 张</b></div>
          <div className="output-image-tabs">{job.outputs.map((output) => <button type="button" key={output.index} className={job.selectedIndex === output.index ? "active" : ""} title={`批次 ${output.batch_index} · 图片 ${output.image_index} · Seed ${output.seed}`} onClick={() => onSelectOutput(output.index)}><img src={output.url} alt="" /><span>{output.image_index}</span></button>)}</div>
        </div>}

        {/* The primary block. A prompt box is the widest thing a user types into and the thing this
            page is actually operated through, so it leads the stage at full width; the pictures are
            what it produces and read as the supporting pair below it. */}
      </section>

      <aside className="i2i-controls-panel panel">
        <div className="panel-scroll">
          <div className="section-heading"><span>02</span><h2>模型引擎</h2><small className="i2i-section-kicker">与文生图共享</small></div>
          <div className="i2i-model-picker">
            <div className="model-segment">
              {modelPicker.engines.map((item) => <button
                type="button"
                key={item.name}
                disabled={!item.ready || running || modelPicker.switching}
                title={`${item.name} · ${item.detail}`}
                className={`model-seg ${engine.name === item.name ? "active" : ""}`}
                onClick={() => modelPicker.onSelectEngine(item.name)}
              ><span className="seg-name">{item.name}</span><span className="seg-detail">{item.ready ? item.detail : "即将支持"}</span></button>)}
            </div>
            {["Anima", "Flux", "Flux2", "Krea2"].includes(engine.name) ? <div className={`checkpoint-picker split-model-picker ${modelPicker.error || modelPicker.assets.some((item) => item.missing) ? "error" : ""}`}>
              <span className="checkpoint-label"><b>模型组件</b><span className="asset-list-actions"><small>{modelPicker.loading ? "扫描中" : `${modelPicker.assets.reduce((total, item) => total + item.options.length, 0)} 个资源`}</small><button type="button" className="asset-refresh" title={`刷新 ${engine.name} 模型组件`} aria-label={`刷新 ${engine.name} 模型组件`} disabled={modelPicker.loading || modelPicker.refreshing || running || modelPicker.switching} onClick={() => modelPicker.onRefresh()}><RefreshCw className={modelPicker.refreshing ? "spin" : ""} size={13} /></button></span></span>
              {modelPicker.assets.map((asset) => <label className="split-model-field" key={asset.kind}>
                <span>{asset.label}<small>{modelPicker.loading ? "扫描中" : `${asset.options.length} 个`}</small></span>
                <span className="checkpoint-select"><WorkspaceSelect ariaLabel={asset.label} value={asset.value} disabled={running || modelPicker.switching || modelPicker.loading || asset.options.length === 0} onChange={(value) => modelPicker.onSelectAsset(asset.kind, value)} options={[
                  ...(modelPicker.loading ? [{ value: "", label: "正在加载模型目录...", disabled: true }] : []),
                  ...(!modelPicker.loading && asset.options.length === 0 ? [{ value: "", label: "未检测到可用资源", disabled: true }] : []),
                  ...(asset.missing && asset.value ? [{ value: asset.value, label: `${asset.value}（文件已删除）` }] : []),
                  ...asset.options.map((item) => ({ value: item.value, label: item.name })),
                ]} /></span>
                <span className="checkpoint-path" title={modelPicker.error || (asset.missing ? `${asset.label}文件已删除，请选择其他资源` : asset.directory)}>{modelPicker.error || (asset.missing ? `${asset.label}文件已删除，请选择其他资源` : asset.directory || "正在解析模型路径...")}</span>
              </label>)}
            </div> : <div className={`checkpoint-picker ${modelPicker.error || modelPicker.checkpointMissing ? "error" : ""}`}>
              <span className="checkpoint-label"><b>底模选择</b><span className="asset-list-actions"><small>{modelPicker.loading ? "扫描中" : `${modelPicker.checkpoints.length} 个模型`}</small><button type="button" className="asset-refresh" title="刷新底模列表" aria-label="刷新底模列表" disabled={modelPicker.loading || modelPicker.refreshing || running || modelPicker.switching} onClick={() => modelPicker.onRefresh()}><RefreshCw className={modelPicker.refreshing ? "spin" : ""} size={13} /></button></span></span>
              <span className="checkpoint-select"><WorkspaceSelect ariaLabel="图生图底模选择" value={modelPicker.checkpoint} disabled={running || modelPicker.switching || modelPicker.loading || modelPicker.checkpoints.length === 0} onChange={modelPicker.onSelectCheckpoint} options={[
                ...(modelPicker.loading ? [{ value: "", label: "正在加载模型目录...", disabled: true }] : []),
                ...(!modelPicker.loading && modelPicker.checkpoints.length === 0 ? [{ value: "", label: "未检测到可用底模", disabled: true }] : []),
                ...(modelPicker.checkpointMissing && modelPicker.checkpoint ? [{ value: modelPicker.checkpoint, label: `${modelPicker.checkpoint}（文件已删除）` }] : []),
                ...modelPicker.checkpoints.map((item) => ({ value: item.value, label: item.name })),
              ]} /></span>
              <span className="checkpoint-path" title={modelPicker.error || (modelPicker.checkpointMissing ? "当前所选底模文件已删除，请选择其他模型" : modelPicker.checkpointDirectory)}>{modelPicker.error || (modelPicker.checkpointMissing ? "当前所选底模文件已删除，请选择其他模型" : modelPicker.checkpointDirectory || "正在解析模型路径...")}</span>
            </div>}
          </div>

          <div className="section-heading"><span>03</span><h2>{postprocessOnly ? "继承参数" : "重绘参数"}</h2><small className="i2i-section-kicker">{postprocessOnly ? "各阶段的默认值" : "基础采样"}</small></div>
          {/* Post-processing runs no base pass, so the strength that would drive it has nothing to
              act on and is left out rather than shown as a control with no effect. Steps, CFG,
              sampler, scheduler and seed stay: every stage that redraws inherits them. */}
          {!postprocessOnly && <div className="i2i-denoise">
            <RangeField label="重绘强度" hint="越低越接近原图" value={config.denoise} min={0.05} max={1} step={0.05} disabled={running} onChange={(value) => update({ denoise: Math.round(value * 100) / 100 })} format={(value) => value.toFixed(2)} />
            <div className="i2i-denoise-scale"><span>保留构图</span><span>改动结构</span><span>几乎重画</span></div>
          </div>}
          <div className="i2i-slider-grid">
            <RangeField label="采样步数" value={config.steps} min={1} max={60} step={1} disabled={running} onChange={(value) => update({ steps: value })} />
            <RangeField label="CFG 引导" value={config.cfg} min={1} max={15} step={0.5} disabled={running} onChange={(value) => update({ cfg: value })} format={(value) => value.toFixed(1)} />
          </div>
          <p className="i2i-note">{postprocessOnly
            ? <>后处理不执行基础采样：这里的步数与 CFG 只在 ADetailer 未单独设置时被继承，Hires 使用自己的步数与 CFG。</>
            : <>实际执行步数约为采样步数 × 重绘强度，当前约 <b>{Math.max(1, Math.floor(config.steps * config.denoise))}</b> 步。</>}</p>
          <div className="i2i-select-row">
            <label>采样器<WorkspaceSelect ariaLabel="采样器" value={config.sampler} disabled={running} onChange={(value) => update({ sampler: value })} options={engine.samplers.map((name) => ({ value: name, label: name }))} /></label>
            <label>调度器<WorkspaceSelect ariaLabel="调度器" value={config.scheduler} disabled={running} onChange={(value) => update({ scheduler: value })} options={engine.schedulers.map((name) => ({ value: name, label: name }))} /></label>
          </div>
          <label className="i2i-seed"><span>随机种子</span><div><input inputMode="numeric" maxLength="20" value={config.seed} disabled={running} onChange={(event) => update({ seed: event.target.value.replace(/\D/g, "") })} /><button type="button" title="生成随机种子" disabled={running} onClick={() => update({ seed: String(Math.floor(Math.random() * 4294967296)) })}><RefreshCw size={15} /></button></div></label>
          <div className="i2i-mode-row" role="group" aria-label="种子生成模式">{SEED_MODES.map((mode) => <button key={mode.id} type="button" className={config.seedMode === mode.id ? "active" : ""} disabled={running} onClick={() => update({ seedMode: mode.id })}>{mode.label}</button>)}</div>
          <div className="i2i-slider-grid">
            <RangeField label="单批图片数" value={config.imagesPerBatch} min={1} max={10} step={1} disabled={running} onChange={(value) => update({ imagesPerBatch: value })} />
            <RangeField label="生成批次数" value={config.batchCount} min={1} max={20} step={1} disabled={running} onChange={(value) => update({ batchCount: value })} />
          </div>
          <p className="i2i-note">{postprocessOnly
            ? <>本任务共输出 <b>{totalImages}</b> 张：同一张来源图会按不同 Seed 各走一遍处理链，纯 RTX 链没有随机性，结果会完全一致。</>
            : <>本任务共生成 <b>{totalImages}</b> 张。</>}</p>

          <div className="section-heading"><span>04</span><h2>输出画布</h2>{postprocessOnly && <small className="i2i-section-kicker">原图尺寸，不缩放</small>}</div>
          {/* Post-processing hands the picture to the first stage exactly as it arrived, so there is
              no canvas to choose and no resize mode to apply. Showing either would offer control
              over something the run does not do. */}
          {postprocessOnly ? <>
            <p className="i2i-canvas-readout"><span>{source ? `${source.width} × ${source.height}` : "未选择图片"}</span><i>→</i><b>{dimensions.width} × {dimensions.height}</b></p>
            <p className="i2i-note">后处理不重采样来源图：原始像素直接进入第一个阶段，最终尺寸由启用的阶段依次决定。</p>
            {sourceIssue && <p className="i2i-stage-warning">{sourceIssue}</p>}
          </> : <>
            <div className="i2i-mode-row" role="group" aria-label="输出尺寸模式">{SIZE_MODES.map((mode) => <button key={mode.id} type="button" className={config.sizeMode === mode.id ? "active" : ""} title={mode.detail} disabled={running} onClick={() => update({ sizeMode: mode.id, ...(mode.id === "custom" ? { width: canvas.width, height: canvas.height } : {}) })}>{mode.label}</button>)}</div>
            {config.sizeMode === "scale" && <RangeField label="缩放倍数" value={config.scale} min={0.25} max={4} step={0.05} disabled={running} onChange={(value) => update({ scale: value })} format={(value) => `${value.toFixed(2)}×`} />}
            {config.sizeMode === "custom" && <div className="i2i-canvas-picker"><div className="i2i-canvas-picker-head"><span>拖拽右下角调整画布</span><b>{canvas.width} × {canvas.height}</b></div><SizeGrid width={config.width} height={config.height} min={64} max={2048} step={64} disabled={running} onChange={(width, height) => update({ width, height })} /></div>}
            <p className="i2i-canvas-readout"><span>{source ? `${source.width} × ${source.height}` : "未选择图片"}</span><i>→</i><b>{canvas.width} × {canvas.height}</b></p>
            <p className="i2i-note">输出尺寸对齐到 64 的倍数，最长边不超过 2048。</p>
            <div className="i2i-mode-row" role="group" aria-label="缩放方式">{RESIZE_MODES.map((mode) => <button key={mode.id} type="button" className={config.resizeMode === mode.id ? "active" : ""} title={mode.detail} disabled={running} onClick={() => update({ resizeMode: mode.id })}>{mode.label}</button>)}</div>
            <p className="i2i-note">{resizeModeMatters(source, canvas) ? RESIZE_MODES.find((mode) => mode.id === config.resizeMode)?.detail : "画布与原图比例一致，缩放方式暂时没有区别。"}</p>
          </>}

          {/* Same markup as the text-to-image panel, so a mount list reads identically on both
              pages — the two share one mounted library, and two spellings of it would invite the
              two to drift. */}
          <div className="section-heading lora-title"><span>05</span><h2>LoRA 挂载</h2><small className="lora-count">已挂载 {engine.loras.length}</small><button onClick={onOpenLoraManager}><Layers3 size={14} />管理</button></div>
          {engine.loras.length === 0 ? <button className="empty-lora" onClick={onOpenLoraManager}><Layers3 size={16} /><span>尚未挂载 LoRA<small>点击管理，从分类库中选择</small></span></button> : <div className="lora-summary">
            <div className="lora-summary-table">
              {engine.loras.map((lora) => (
                <div key={lora.value} className={`lora-summary-row ${lora.enabled === false ? "disabled" : ""}`} title={`${lora.name} · 分类 ${lora.category}`}>
                  <span className="lora-summary-name">{lora.enabled === false && <Square size={8} />}{lora.name}</span>
                  <b>{formatWeight(lora.weight)}</b>
                </div>
              ))}
            </div>
          </div>}
          <p className="i2i-note">切换引擎或底模会同步到文生图；LoRA 挂载按引擎隔离并随之切换。</p>

          <PostprocessControls config={config} engine={engine} postprocess={postprocess} running={running} dimensions={dimensions} updateStage={updateStage} moveStage={moveStage} order={order} postprocessOnly={postprocessOnly} />
        </div>
      </aside>
    </section>
  );
}
