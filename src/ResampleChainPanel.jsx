// The post-processing chain, as both generation pages draw it.
//
// One component because one state shape: the workspace and the image page hold the same chain, send
// the same `postprocess_chain`, and would otherwise have to grow two editors for it that drift
// apart. What they do not share is their card chrome — the workspace's stage cards and the image
// page's are styled and laid out differently, and both were there first — so each page passes a
// `renderCard` that wraps a stage in its own header, and passes its own slider and number input for
// the same reason. Everything about what a unit *is* lives here.

import { ArrowDown, ArrowUp, Plus, RefreshCw, Trash2 } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import {
  RESAMPLE_UNIT_LIMIT,
  RESIZE_ALGORITHMS,
  addResampleUnit,
  chainSizeLine,
  moveChainStage,
  removeChainStage,
  resampleChainPlan,
  resampleUnitEffectiveSteps,
  resampleUnitNumber,
  resampleUnitSummary,
  resampleUnitTitle,
  resampleUnits,
  updateChainStage,
} from "./resample-chain.js";
import { supportsUsduTiled } from "./hires-settings.js";
import { NATIVE_STEP_ENGINES } from "./adetailer-units.js";

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
}) {
  const number = resampleUnitNumber(chain, unit);
  const compatible = upscalers.filter((item) => item.compatible !== false);
  const tiles = supportsUsduTiled(engine) && unit.denoise > 0;
  const effectiveSteps = resampleUnitEffectiveSteps(unit, baseSteps, engine, NATIVE_STEP_ENGINES);
  const inputSize = entry?.input || null;

  return (
    <div className={`resample-parameters ${unit.enabled ? "enabled" : ""}`}>
      <p className="postprocess-stage-description">
        {unit.upscaler ? "先用超分模型放大" : "先按所选算法缩放"}
        {unit.denoise > 0 ? "，再用当前底模、LoRA 和提示词重绘。" : "，不进行扩散重绘。"}
      </p>
      {unavailableReason && <p className="resample-unavailable">{unavailableReason}</p>}

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
            onChange={(value) => update({ upscaler: value })}
            options={[
              // The explicit entry is what makes a model-less unit a choice rather than an accident
              // of leaving the picker alone.
              { value: "", label: "不使用模型 · 仅算法缩放" },
              ...compatible.map((item) => ({
                value: item.id,
                label: [item.label || item.id, item.architecture, item.scale ? `${item.scale}x` : null].filter(Boolean).join(" · "),
              })),
            ]}
          />
          {onRefreshUpscalers && <button type="button" title="重新扫描超分模型" disabled={upscalersRefreshing || locked} onClick={onRefreshUpscalers}>
            <RefreshCw className={upscalersRefreshing ? "spin" : ""} size={12} />
          </button>}
        </span>
      </label>
      {unit.upscaler && !upscalerRuntimeReady && <p className="resample-unavailable">超分运行库尚未安装，请重新运行环境配置器。</p>}
      {unit.upscaler && upscalerRuntimeReady && compatible.length === 0 && <p className="resample-unavailable">尚无兼容超分模型，可改为仅算法缩放。</p>}

      <label className="resample-model">
        缩放算法
        <WorkspaceSelect
          ariaLabel={`重采样 ${number} 缩放算法`}
          value={unit.resizeAlgorithm}
          disabled={locked}
          onChange={(value) => update({ resizeAlgorithm: value })}
          options={RESIZE_ALGORITHMS.map((item) => ({ value: item.id, label: `${item.label} · ${item.detail}` }))}
        />
      </label>

      <div className="resample-scale">
        <Slider label="放大倍率" value={unit.scale} min={1} max={4} step={0.1} inputStep={0.1} fixed={1} suffix="×" disabled={locked} onChange={(value) => update({ scale: Math.round(value * 10) / 10 })} />
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
        <Slider label="重绘强度" value={unit.denoise} min={0} max={1} step={0.05} inputStep={0.01} fixed={2} disabled={locked} onChange={(value) => update({ denoise: Math.round(value * 100) / 100 })} />
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
              <input inputMode="numeric" maxLength={20} value={unit.seed} disabled={locked} aria-label={`重采样 ${number} Seed`} onChange={(event) => update({ seed: event.target.value.replace(/\D/g, "") })} placeholder="0–18446744073709551615" />
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

        {unit.enabled && effectiveSteps < 1 && <p className="resample-unavailable">当前步数与重绘强度不会产生有效扩散步骤，请提高其中一项。</p>}
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
  locked = false,
  addDisabled = false,
  unitUnavailableReason = "",
  Slider,
  NumberInput,
  randomSeed = null,
  renderCard,
  renderStage,
}) {
  const plan = resampleChainPlan(baseSize, chain, engine);
  const entries = new Map(plan.trace.map((entry) => [entry.id, entry]));
  const units = resampleUnits(chain);
  const canAdd = !locked && !addDisabled && units.length < RESAMPLE_UNIT_LIMIT;

  const stageControls = (stage, index) => (
    <div className="resample-stage-actions">
      <button type="button" aria-label={`上移${stage.type === "resample" ? `重采样 ${resampleUnitNumber(chain, stage)}` : "该阶段"}`} title="上移" disabled={locked || index === 0} onClick={() => onChange(moveChainStage(chain, stage.id, -1))}><ArrowUp size={13} /></button>
      <button type="button" aria-label={`下移${stage.type === "resample" ? `重采样 ${resampleUnitNumber(chain, stage)}` : "该阶段"}`} title="下移" disabled={locked || index === chain.length - 1} onClick={() => onChange(moveChainStage(chain, stage.id, 1))}><ArrowDown size={13} /></button>
      {stage.type === "resample" && <button type="button" className="resample-stage-remove" aria-label={`删除重采样 ${resampleUnitNumber(chain, stage)}`} title="删除该单元" disabled={locked} onClick={() => onChange(removeChainStage(chain, stage.id))}><Trash2 size={13} /></button>}
    </div>
  );

  return (
    <div className="resample-chain">
      <div className="resample-chain-line" aria-live="polite">
        <span>尺寸链</span>
        <b>{chainSizeLine(baseSize, chain, engine)}</b>
      </div>
      {!plan.valid && <p className="resample-chain-error">{plan.reason}</p>}

      <div className="resample-chain-list">
        {chain.map((stage, index) => {
          const actions = stageControls(stage, index);
          if (stage.type !== "resample") return <div key={stage.id} className="resample-chain-item">{renderStage(stage, index, actions)}</div>;
          const entry = entries.get(stage.id) || null;
          const update = (updates) => onChange(updateChainStage(chain, stage.id, updates));
          return <div key={stage.id} className="resample-chain-item">
            {renderCard({
              stage,
              title: resampleUnitTitle(chain, stage),
              detail: stage.enabled ? `${resampleUnitSummary(stage)} · ${sizeText(entry?.output)}` : "关闭",
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
              /> : null,
            })}
          </div>;
        })}
      </div>

      <div className="resample-chain-add">
        <button type="button" disabled={!canAdd} onClick={() => onChange(addResampleUnit(chain))}>
          <Plus size={14} />添加重采样单元
        </button>
        <small>{units.length} / {RESAMPLE_UNIT_LIMIT} 个单元 · 按上下箭头调整执行顺序，可与 ADetailer、RTX VSR 交错</small>
      </div>
    </div>
  );
}
