import { MOSS_DIT_DEPTH, describeMossSampling, normalizeMossSampling, parseSkipLayers } from "./moss-engine.js";

// MOSS-Image's own sampling controls: the flow shift (ComfyUI `ModelSamplingSD3`) and negative-branch
// block skipping (`SkipLayerGuidanceDiTSimple`). Presentation only; the owner keeps the block and the
// typed skip text, and parses the text into the block only when it is valid, so a half-typed list never
// reaches a request. Standard positive-branch SLG is a different technique and is not offered here.
export default function MossSamplingControls({ value, skipText, onSkipText, onChange, cfg, disabled = false, Slider }) {
  const settings = normalizeMossSampling(value);
  const parsed = parseSkipLayers(skipText);
  const update = (patch) => onChange(normalizeMossSampling({ ...settings, ...patch }));
  const editSkip = (text) => {
    onSkipText(text);
    const next = parseSkipLayers(text);
    if (!next.error) update({ uncondSkipLayers: next.layers });
  };
  const cfgOne = Number(cfg) === 1;
  return (
    // Laid out like the sampling sliders and the seed field around it, so the block reads as more
    // sampling parameters rather than as a guidance warning.
    <div className="moss-sampling" role="group" aria-label="MOSS 采样设置">
      <div className="sliders">
        <Slider label="Shift（流偏移）" value={settings.shift} min={0.5} max={20} step={0.1} inputStep={0.01} disabled={disabled} onChange={(shift) => update({ shift })} />
      </div>
      <label className="seed-field moss-skip-field">
        <span>负向分支跳过区块</span>
        <div>
          <input
            type="text"
            inputMode="numeric"
            aria-label="负向分支跳过区块"
            aria-invalid={parsed.error ? true : undefined}
            aria-describedby="moss-skip-help"
            disabled={disabled}
            value={skipText}
            placeholder="留空 = 不跳层"
            onChange={(event) => editSkip(event.target.value)}
          />
        </div>
      </label>
      {parsed.error
        ? <p className="guidance-unavailable" role="alert" id="moss-skip-help">{parsed.error}；已保留上次有效的区块列表。</p>
        : <p className="clip-skip-note" id="moss-skip-help">区块编号 0～{MOSS_DIT_DEPTH - 1}，逗号分隔；官方默认 22，留空即关闭。只作用于负向（无条件）分支，不增加前向次数。</p>}
      <div className="sliders">
        <Slider label="跳层起点" value={settings.uncondSkipStart} min={0} max={1} step={0.05} inputStep={0.01} disabled={disabled}
          onChange={(uncondSkipStart) => update({ uncondSkipStart: Math.min(uncondSkipStart, settings.uncondSkipEnd) })} />
        <Slider label="跳层终点" value={settings.uncondSkipEnd} min={0} max={1} step={0.05} inputStep={0.01} disabled={disabled}
          onChange={(uncondSkipEnd) => update({ uncondSkipEnd: Math.max(uncondSkipEnd, settings.uncondSkipStart) })} />
      </div>
      <p className="clip-skip-note moss-sampling-summary"><b>{describeMossSampling(settings, cfg)}</b>{cfgOne ? "" : " · 区间按 ComfyUI sigma 百分比计"}</p>
      <p className="clip-skip-note">标准 SLG（正向分支额外前向）本版本未提供。</p>
    </div>
  );
}
