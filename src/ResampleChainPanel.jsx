// The post-processing chain, as both generation pages draw it.
//
// One component because one state shape: the workspace and the image page hold the same chain, send
// the same `postprocess_chain`, and would otherwise have to grow two editors for it that drift
// apart. What they do not share is their card chrome — the workspace's stage cards and the image
// page's are styled and laid out differently, and both were there first — so each page passes a
// `renderCard` that wraps a stage in its own header, and passes its own slider and number input for
// the same reason. Everything about what a unit *is* lives here.

import { useId, useState } from "react";
import { ArrowDown, ArrowUp, Check, ChevronDown, Download, Plus, RefreshCw, Trash2 } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import {
  LATENT_ALGORITHMS,
  RESAMPLE_UNIT_LIMIT,
  RESIZE_ALGORITHMS,
  addResampleUnit,
  chainSizeLine,
  moveChainStage,
  removeChainStage,
  resampleChainPlan,
  resampleAlgorithmLabel,
  resampleAlgorithmSelectionUpdate,
  resampleModelSelectionUpdate,
  resampleUnitAlgorithmIssue,
  resampleUnitEffectiveSteps,
  resampleUnitNumber,
  resampleUnitSummary,
  resampleUnitTitle,
  resampleUnits,
  isRelativeUpscalerId,
  updateChainStage,
} from "./resample-chain.js";
import { supportsUsduTiled } from "./hires-settings.js";
import { formatFileSize } from "./format-size.js";

const SEED_MODE_OPTIONS = [
  { value: "inherit", label: "继承每张首轮 Seed" },
  { value: "fixed", label: "固定 Seed" },
  { value: "random", label: "每张安全随机" },
];

const SEED_MODE_NOTE = {
  inherit: "每张结果继承该张首轮 Seed",
  fixed: "所有结果使用同一个无损 uint64 Seed",
  random: "每张结果在后端独立解析一次安全 uint64 Seed",
};

function sizeText(size) {
  return size ? `${size.width} × ${size.height}` : "—";
}

function upscalerCatalogParts(value) {
  if (Array.isArray(value)) return { models: value, builtins: [], directory: "" };
  const source = value && typeof value === "object" ? value : {};
  return {
    models: Array.isArray(source.models) ? source.models : [],
    builtins: Array.isArray(source.builtins) ? source.builtins : [],
    directory: typeof source.directory === "string" ? source.directory : "",
  };
}

function downloadItemFor(job, builtin) {
  if (!job || !builtin) return null;
  return (job.items || []).find((item) => (
    (item.artifactId && item.artifactId === builtin.id)
    || (item.sha256 && builtin.sha256 && item.sha256.toLowerCase() === builtin.sha256.toLowerCase())
  )) || null;
}

function downloadStatusLabel(item) {
  if (item?.status === "waiting" || item?.status === "queued") return "排队中";
  if (item?.status === "resolving" || item?.status === "metadata") return "解析中";
  if (item?.status === "downloading") return "下载中";
  if (["verifying", "installing"].includes(item?.status)) return "校验中";
  if (item?.status === "complete") return "已完成";
  if (item?.status === "error") return "下载失败";
  return "";
}

function downloadProgress(item) {
  if (!item || !(item.totalBytes > 0)) return 0;
  return Math.max(0, Math.min(100, item.currentBytes / item.totalBytes * 100));
}

function downloadDirectorySummary(builtins, job, error = "") {
  if (error) return "下载失败，可重试";
  const progressItems = builtins.map((item) => downloadItemFor(job, item)).filter(Boolean);
  const active = progressItems.filter((item) => !["complete", "error"].includes(item.status));
  if (active.length > 0 || job?.active === true) {
    const label = downloadStatusLabel(active[0]) || "下载中";
    return `${label} · ${active.length || 1} 项`;
  }
  const failed = progressItems.filter((item) => item.status === "error").length;
  if (failed) return `${failed} 项失败，可重试`;
  return `${builtins.filter((item) => item.installed).length} / ${builtins.length} 已安装`;
}

/** One shared catalogue per ResampleChainPanel, rather than one ten-row list per unit. */
function UpscalerDownloadDirectory({
  builtins,
  directory,
  job,
  locked,
  busy,
  error,
  onDownload,
  onRetry,
}) {
  const [expanded, setExpanded] = useState(false);
  const rawId = useId();
  const contentId = `resample-upscaler-downloads-${rawId.replaceAll(":", "")}`;
  const summary = downloadDirectorySummary(builtins, job, error);

  const renderRow = (item) => {
    const progressItem = downloadItemFor(job, item);
    const progressActive = progressItem && !["complete", "error"].includes(progressItem.status);
    const failed = progressItem?.status === "error";
    const completed = progressItem?.status === "complete";
    const globallyBusy = busy || job?.active === true;
    const canDownload = Boolean(item.download?.family_id && item.download?.artifact_id);
    const disabled = locked || !canDownload || globallyBusy || completed || (failed && job?.retryAvailable === false) || item.installed;
    const actionLabel = item.installed && item.compatible !== false
      ? "已安装"
      : item.installed && item.compatible === false
        ? "不可选"
        : progressActive
          ? downloadStatusLabel(progressItem)
          : completed
            ? "已完成"
            : failed
              ? "重试"
              : "下载";
    const statusLabel = item.installed
      ? item.compatible === false ? "已安装 · 不兼容" : "已安装 · 兼容"
      : "未下载";
    const reason = item.reason || (item.compatible === false ? "当前运行库不支持该模型" : "");
    const progressPercent = downloadProgress(progressItem);
    return <div className={`resample-builtin-row ${item.compatible === false ? "incompatible" : ""}`} key={item.id}>
      <div className="resample-builtin-copy">
        <b title={item.label || item.id}>{item.label || item.id}</b>
        <small>{formatFileSize(item.size)} · {statusLabel}{reason ? ` · ${reason}` : ""}</small>
      </div>
      <button
        type="button"
        className="resample-builtin-download"
        aria-label={`${actionLabel} ${item.label || item.id}`}
        title={reason && item.compatible === false ? `${actionLabel}：${reason}` : actionLabel}
        disabled={disabled || progressActive}
        onClick={() => {
          if (failed) onRetry?.(item);
          else onDownload?.(item);
        }}
      >
        {(item.installed && item.compatible !== false) || completed ? <Check size={11} /> : failed ? <RefreshCw size={11} /> : <Download size={11} />}
        {actionLabel}
      </button>
      {progressItem && <div className={`resample-builtin-progress ${failed ? "error" : ""}`} aria-live="polite">
        <span>{downloadStatusLabel(progressItem)}{progressItem.error ? ` · ${progressItem.error}` : ""}</span>
        {progressItem.totalBytes > 0 && <i><i style={{ width: `${progressPercent}%` }} /></i>}
        <small>{progressItem.totalBytes > 0 ? `${formatFileSize(progressItem.currentBytes || 0)} / ${formatFileSize(progressItem.totalBytes)}` : (progressItem.message || "等待下载")}{progressItem.speedBps > 0 ? ` · ${formatFileSize(progressItem.speedBps)}/s` : ""}</small>
      </div>}
    </div>;
  };

  return <section className="resample-download-disclosure">
    <button
      type="button"
      className="resample-download-summary"
      aria-expanded={expanded}
      aria-controls={contentId}
      onClick={() => setExpanded((current) => !current)}
    >
      <span><Download size={13} /><strong>放大模型下载</strong><small>{summary}</small></span>
      <ChevronDown size={15} className={expanded ? "expanded" : ""} />
    </button>
    <div id={contentId} className="resample-download-content" hidden={!expanded}>
      <div className="resample-builtin-library" aria-label="内置超分模型目录">
        <header><span><b>内置超分模型目录</b><small>{summary}</small></span><small title={directory}>{directory || "models/upscalers"}</small></header>
        {error && <p className="resample-unavailable" role="alert">{error}</p>}
        {builtins.map(renderRow)}
      </div>
    </div>
  </section>;
}

/** One resample unit's editor. The three parts of a unit, in the order they execute. */
function ResampleUnitBody({
  unit,
  chain,
  entry,
  engine,
  samplers,
  schedulers,
  baseSteps,
  baseCfg,
  upscalers,
  upscalerRuntimeReady,
  upscalersRefreshing,
  onRefreshUpscalers,
  locked,
  unavailableReason,
  Slider,
  NumberInput,
  update,
  randomSeed,
  preserveRawInputs = false,
}) {
  const number = resampleUnitNumber(chain, unit);
  const catalog = upscalerCatalogParts(upscalers);
  const localModels = catalog.models;
  const builtins = catalog.builtins;
  const installedBuiltinIds = new Set(builtins.filter((item) => item.installed && isRelativeUpscalerId(item.local_id)).map((item) => item.local_id));
  const selectableLocalModels = localModels.filter((item) => isRelativeUpscalerId(item.id) && !installedBuiltinIds.has(item.id));
  const selectedBuiltin = builtins.find((item) => item.installed && isRelativeUpscalerId(item.local_id) && item.local_id === unit.upscaler) || null;
  const selectedLocal = localModels.find((item) => isRelativeUpscalerId(item.id) && item.id === unit.upscaler) || null;
  const selectedUnavailableReason = unit.upscaler && !selectedBuiltin && !selectedLocal
    ? "当前超分模型未在目录中找到，可能已移除或尚未刷新。"
    : selectedBuiltin?.compatible === false || selectedLocal?.compatible === false
      ? (selectedBuiltin?.reason || selectedLocal?.error || "当前超分模型与运行库不兼容。")
      : "";
  const latentEngineAvailable = engine === "SD" || engine === "iL";
  const latentDisabledReason = !latentEngineAvailable
    ? "潜空间算法仅支持 SD / iL 引擎"
    : unit.executionMode !== "full_frame"
      ? "潜空间算法仅支持 full_frame 整图重绘"
      : unit.upscaler
        ? "潜空间算法不能与超分模型同时使用"
        : "";
  const algorithmIssue = unit.enabled ? resampleUnitAlgorithmIssue(unit, engine) : "";
  const tiles = supportsUsduTiled(engine) && unit.denoise > 0;
  const effectiveSteps = resampleUnitEffectiveSteps(unit, baseSteps);
  const inputSize = entry?.input || null;

  const modelOptions = [
    { value: "", label: "不使用模型 · 仅算法缩放" },
    // The picker only exposes files that are actually local.  Undownloaded recommendations stay
    // in the disclosure below; an installed but incompatible ScuNET entry remains visible here as
    // disabled so the reason cannot be mistaken for a missing model.
    ...builtins.filter((item) => item.installed && isRelativeUpscalerId(item.local_id)).map((item) => {
      const selectable = item.installed && item.compatible !== false && isRelativeUpscalerId(item.local_id);
      const reason = !item.installed
        ? "尚未下载"
        : item.compatible === false
          ? (item.reason || "与当前超分运行库不兼容")
          : !item.local_id ? "安装文件未在本地目录中找到" : "";
      return {
        value: item.local_id,
        label: `${item.label || item.id}${selectable ? "" : ` · ${reason}`}`,
        group: item.installed ? "内置 · 已安装" : "内置 · 未下载",
        disabled: !selectable,
        disabledReason: reason,
      };
    }),
    ...selectableLocalModels.map((item) => {
      const compatible = item.compatible !== false;
      const reason = compatible ? "" : (item.error || "与当前超分运行库不兼容");
      return {
        value: item.id,
        label: `${item.label || item.name || item.id}${compatible ? "" : ` · ${reason}`}`,
        group: "本地模型",
        disabled: !compatible,
        disabledReason: reason,
      };
    }),
    ...(unit.upscaler && !selectedBuiltin && !selectedLocal && isRelativeUpscalerId(unit.upscaler) ? [{
      value: unit.upscaler,
      label: `${unit.upscaler} · 文件缺失`,
      group: "当前记录",
      disabled: true,
      disabledReason: selectedUnavailableReason,
    }] : []),
  ];

  return (
    <div className={`resample-parameters ${unit.enabled ? "enabled" : ""}`}>
      <p className="postprocess-stage-description">
        {unit.upscaler ? "先用超分模型放大" : `先按 ${resampleAlgorithmLabel(unit)} 缩放`}
        {unit.denoise > 0 ? "，再用当前底模、LoRA 和提示词重绘。" : "，不进行扩散重绘。"}
      </p>
      {unavailableReason && <p className="resample-unavailable">{unavailableReason}</p>}
      {algorithmIssue && <p className="resample-unavailable">{algorithmIssue}</p>}

      <label className="resample-name">
        单元名称
        <input
          type="text"
          maxLength={80}
          value={unit.name}
          disabled={locked}
          placeholder={`重采样 ${number}`}
          aria-label={`重采样 ${number} 名称`}
          onChange={(event) => update({ name: event.target.value })}
        />
      </label>

      <label className="resample-model">
        超分模型
        <span>
          <WorkspaceSelect
            ariaLabel={`重采样 ${number} 超分模型`}
            value={unit.upscaler}
            disabled={locked}
            onChange={(value) => update(resampleModelSelectionUpdate(value))}
            options={modelOptions}
          />
          {onRefreshUpscalers && <button type="button" aria-label="重新扫描超分模型" title="重新扫描超分模型" disabled={upscalersRefreshing || locked} onClick={onRefreshUpscalers}>
            <RefreshCw className={upscalersRefreshing ? "spin" : ""} size={12} />
          </button>}
        </span>
      </label>
      {unit.upscaler && selectedUnavailableReason && <p className="resample-unavailable">{selectedUnavailableReason}</p>}
      {unit.upscaler && !upscalerRuntimeReady && <p className="resample-unavailable">超分运行库尚未安装，请重新运行环境配置器。</p>}
      <label className="resample-model">
        缩放算法
        <WorkspaceSelect
          ariaLabel={`重采样 ${number} 缩放算法`}
          value={unit.latentAlgorithm || unit.resizeAlgorithm}
          disabled={locked}
          onChange={(value) => update(resampleAlgorithmSelectionUpdate(value))}
          options={[
            ...LATENT_ALGORITHMS.map((item) => ({
              value: item.id,
              label: `${item.label} · ${item.detail}`,
              group: "潜空间算法 · Latent",
              disabled: Boolean(latentDisabledReason),
              disabledReason: latentDisabledReason,
            })),
            ...RESIZE_ALGORITHMS.map((item) => ({ value: item.id, label: `${item.label} · ${item.detail}`, group: "像素算法 · Pixel" })),
          ]}
        />
      </label>

      <div className="resample-scale">
        <Slider label="放大倍率" value={unit.scale} min={1} max={4} step={0.1} inputStep={0.1} fixed={1} suffix="×" disabled={locked} onChange={(value) => update({ scale: preserveRawInputs && typeof value !== "number" ? value : Math.round(value * 10) / 10 })} />
        <p>
          <span>INPUT {sizeText(inputSize)}</span>
          <b>OUTPUT {sizeText(entry?.output)}</b>
        </p>
        {entry?.rounded && <small className="resample-rounded">
          目标 {sizeText(entry.target)} · {engine} 重绘画布对齐到 {sizeText(entry.redraw)}
          {unit.exactOutput ? " · 已按精确尺寸回落" : ""}
        </small>}
      </div>

      <div className="resample-sliders">
        <Slider label="重绘强度" value={unit.denoise} min={0} max={1} step={0.05} inputStep={0.01} fixed={2} disabled={locked} onChange={(value) => update({ denoise: preserveRawInputs && typeof value !== "number" ? value : Math.round(value * 100) / 100 })} />
      </div>
      {unit.denoise === 0 && <p className="resample-hint">重绘强度为 0：本单元只放大，不调用扩散模型。</p>}

      {unit.denoise > 0 && <>
        <div className="resample-overrides">
          <label>
            <input type="checkbox" checked={unit.steps !== null} disabled={locked} onChange={(event) => update({ steps: event.target.checked ? baseSteps : null })} />
            <span>独立步数</span>
            <NumberInput ariaLabel={`重采样 ${number} 步数`} value={unit.steps ?? baseSteps} min={1} max={100} integer className="bounded-number" disabled={locked || unit.steps === null} onChange={(value) => update({ steps: value })} />
          </label>
          <label>
            <input type="checkbox" checked={unit.cfg !== null} disabled={locked} onChange={(event) => update({ cfg: event.target.checked ? baseCfg : null })} />
            <span>独立 CFG</span>
            <NumberInput ariaLabel={`重采样 ${number} CFG`} value={unit.cfg ?? baseCfg} min={0} max={30} step={0.5} fixed={1} className="bounded-number" disabled={locked || unit.cfg === null} onChange={(value) => update({ cfg: value })} />
          </label>
        </div>
        <p className="resample-hint">未勾选的项目跟随首轮设置：步数 {baseSteps} · CFG {Number(baseCfg).toFixed(1)}。</p>

        <div className="resample-seed-settings">
          <label>
            Seed 模式
            <WorkspaceSelect ariaLabel={`重采样 ${number} Seed 模式`} value={unit.seedMode} disabled={locked} onChange={(value) => update({ seedMode: value, seed: value === "fixed" ? unit.seed : "" })} options={SEED_MODE_OPTIONS} />
          </label>
          {unit.seedMode === "fixed" && <label className="resample-seed-field">
            Seed
            <span>
              <input inputMode="numeric" maxLength={20} value={unit.seed} disabled={locked} aria-label={`重采样 ${number} Seed`} onChange={(event) => update({ seed: preserveRawInputs ? event.target.value : event.target.value.replace(/\D/g, "") })} placeholder="0–18446744073709551615" />
              {randomSeed && <button type="button" title="生成固定 Seed" disabled={locked} onClick={() => update({ seed: randomSeed() })}><RefreshCw size={13} /></button>}
            </span>
          </label>}
          <small>{SEED_MODE_NOTE[unit.seedMode]}</small>
        </div>

        {tiles && <label className="resample-model">
          重绘方式
          <WorkspaceSelect
            ariaLabel={`重采样 ${number} 重绘方式`}
            value={unit.executionMode}
            disabled={locked}
            onChange={(value) => update({ executionMode: value })}
            options={[{ value: "usdu_tiled", label: "USDU 分块重绘（推荐）" }, { value: "full_frame", label: "整图重绘（兼容）" }]}
          />
        </label>}

        <div className="resample-tile-grid">
          <label>采样器<WorkspaceSelect ariaLabel={`重采样 ${number} 采样器`} value={unit.sampler || ""} disabled={locked} onChange={(value) => update({ sampler: value || null })} options={[{ value: "", label: "跟随首轮" }, ...samplers.map((name) => ({ value: name, label: name }))]} /></label>
          <label>调度器<WorkspaceSelect ariaLabel={`重采样 ${number} 调度器`} value={unit.scheduler || ""} disabled={locked} onChange={(value) => update({ scheduler: value || null })} options={[{ value: "", label: "跟随首轮" }, ...schedulers.map((name) => ({ value: name, label: name }))]} /></label>
        </div>

        {tiles && unit.executionMode === "usdu_tiled" && <p className="resample-incompatible">
          Auto = 进入本单元的画布宽高；padding 32；mask blur 8；uniform tiles；per-tile VAE tiled decode；seam None；每 tile 执行本单元步数。
        </p>}

        {unit.enabled && <p className="resample-hint">有效步数 {effectiveSteps}：重绘强度只决定调度起点，不会缩减执行步数。</p>}
      </>}

      <label className="resample-exact">
        <input type="checkbox" checked={unit.exactOutput} disabled={locked || unit.denoise === 0} onChange={(event) => update({ exactOutput: event.target.checked })} />
        <span><strong>精确输出尺寸</strong><small>重绘后再缩放回 {sizeText(entry?.target)}，代价是一次额外重采样</small></span>
      </label>

      {unit.upscaler && <div className="resample-tile-grid">
        <label>像素放大分块<NumberInput ariaLabel={`重采样 ${number} 像素放大分块`} value={unit.tileSize} min={32} max={2048} integer className="bounded-number" disabled={locked} onChange={(tileSize) => update({ tileSize, tileOverlap: Math.min(unit.tileOverlap, Math.floor(tileSize / 2)) })} /></label>
        <label>像素放大分块重叠<NumberInput ariaLabel={`重采样 ${number} 像素放大分块重叠`} value={unit.tileOverlap} min={0} max={Math.min(512, Math.floor(unit.tileSize / 2))} integer className="bounded-number" disabled={locked} onChange={(tileOverlap) => update({ tileOverlap })} /></label>
      </div>}
    </div>
  );
}

export default function ResampleChainPanel({
  chain,
  onChange,
  engine,
  samplers = [],
  schedulers = [],
  baseSize,
  baseSteps,
  baseCfg,
  upscalers = [],
  upscalerRuntimeReady = true,
  upscalersRefreshing = false,
  onRefreshUpscalers = null,
  upscalerDownloadJob = null,
  upscalerDownloadBusy = false,
  upscalerDownloadError = "",
  onDownloadUpscaler = null,
  onRetryUpscaler = null,
  locked = false,
  actionsLocked = locked,
  addDisabled = false,
  unitUnavailableReason = "",
  Slider,
  NumberInput,
  randomSeed = null,
  chainIssue = "",
  renderCard,
  renderStage,
  preserveRawInputs = false,
  sizeLineLabel = "尺寸链",
}) {
  let plan, sizeLine;
  try { plan = resampleChainPlan(baseSize, chain, engine); sizeLine = chainSizeLine(baseSize, chain, engine); }
  catch (error) {
    if (!preserveRawInputs) throw error;
    plan = { trace: [], valid: false, reason: "后处理链数值无效；原值保留，请修正后预检。" };
    sizeLine = `${baseSize.width} × ${baseSize.height} → 待修正`;
  }
  const entries = new Map(plan.trace.map((entry) => [entry.id, entry]));
  const units = resampleUnits(chain);
  const catalog = upscalerCatalogParts(upscalers);
  const canAdd = !actionsLocked && !addDisabled && units.length < RESAMPLE_UNIT_LIMIT;
  const addUnit = () => {
    if (!canAdd) return;
    onChange(addResampleUnit(chain, {}, { engine }));
  };

  const stageControls = (stage, index) => {
    const name = stage.type === "resample"
      ? `重采样 ${resampleUnitNumber(chain, stage)}`
      : stage.type === "adetailer" ? "ADetailer" : stage.type === "rtx" ? "RTX VSR" : "该阶段";
    const canMoveUp = !actionsLocked && index > 0;
    const canMoveDown = !actionsLocked && index < chain.length - 1;
    return (
      <div className="resample-stage-actions">
        <button type="button" aria-label={`上移${name}`} title="上移" disabled={!canMoveUp} onClick={() => {
          if (!canMoveUp) return;
          onChange(moveChainStage(chain, stage.id, -1));
        }}><ArrowUp size={13} /></button>
        <button type="button" aria-label={`下移${name}`} title="下移" disabled={!canMoveDown} onClick={() => {
          if (!canMoveDown) return;
          onChange(moveChainStage(chain, stage.id, 1));
        }}><ArrowDown size={13} /></button>
        {stage.type === "resample" && <button type="button" className="resample-stage-remove" aria-label={`删除重采样 ${resampleUnitNumber(chain, stage)}`} title="删除该单元" disabled={actionsLocked} onClick={() => {
          if (actionsLocked) return;
          onChange(removeChainStage(chain, stage.id));
        }}><Trash2 size={13} /></button>}
      </div>
    );
  };

  return (
    <div className="resample-chain">
      <div className="resample-chain-line" aria-live="polite">
        <span>{sizeLineLabel}</span>
        <b>{sizeLine}</b>
      </div>
      {chainIssue && <p className="resample-chain-error" role="alert">{chainIssue}</p>}
      {!chainIssue && !plan.valid && <p className="resample-chain-error">{plan.reason}</p>}

      <div className="resample-chain-list">
        {chain.map((stage, index) => {
          const actions = stageControls(stage, index);
          if (stage.type !== "resample") return <div key={stage.id} className="resample-chain-item">{renderStage(stage, index, actions)}</div>;
          const entry = entries.get(stage.id) || null;
          const update = (updates) => onChange(updateChainStage(chain, stage.id, updates));
          return <div key={stage.id} className="resample-chain-item">
            {renderCard({
              stage,
              title: preserveRawInputs && typeof stage.name !== "string" ? `重采样 ${resampleUnitNumber(chain, stage)} · 保存名称待修正` : resampleUnitTitle(chain, stage),
              detail: stage.enabled ? preserveRawInputs && (typeof stage.scale !== "number" || typeof stage.denoise !== "number") ? "保存数值待修正 · 原值保留" : `${resampleUnitSummary(stage)} · ${sizeText(entry?.output)}` : "关闭",
              enabled: stage.enabled,
              expanded: stage.expanded,
              onToggleEnabled: () => update({ enabled: !stage.enabled }),
              onToggleExpanded: () => update({ expanded: !stage.expanded }),
              actions,
              children: stage.expanded ? <ResampleUnitBody
                unit={stage}
                chain={chain}
                entry={entry}
                engine={engine}
                samplers={samplers}
                schedulers={schedulers}
                baseSteps={baseSteps}
                baseCfg={baseCfg}
                upscalers={upscalers}
                upscalerRuntimeReady={upscalerRuntimeReady}
                upscalersRefreshing={upscalersRefreshing}
                onRefreshUpscalers={onRefreshUpscalers}
                locked={locked}
                unavailableReason={unitUnavailableReason}
                Slider={Slider}
                NumberInput={NumberInput}
                update={update}
                randomSeed={randomSeed}
                preserveRawInputs={preserveRawInputs}
              /> : null,
            })}
          </div>;
        })}
      </div>

      {catalog.builtins.length > 0 && <UpscalerDownloadDirectory
        builtins={catalog.builtins}
        directory={catalog.directory}
        job={upscalerDownloadJob}
        locked={locked}
        busy={upscalerDownloadBusy}
        error={upscalerDownloadError}
        onDownload={onDownloadUpscaler}
        onRetry={onRetryUpscaler}
      />}

      <div className="resample-chain-add">
        <button type="button" aria-label="添加重采样单元" title={canAdd ? "添加重采样单元" : "已达到重采样单元上限或当前已锁定"} disabled={!canAdd} onClick={addUnit}>
          <Plus size={14} />添加重采样单元
        </button>
        <small>{units.length} / {RESAMPLE_UNIT_LIMIT} 个单元 · 按上下箭头调整执行顺序，可与 ADetailer、RTX VSR 交错</small>
      </div>
    </div>
  );
}
